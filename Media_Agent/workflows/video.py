# -*- coding: utf-8 -*-
"""口播视频工作流 —— 提词器 / 数字人出镜

课案出处：自媒体课案 → 口播视频 → workflows/video.py

双模式（与课案一致）
    📺 **提词器模式**：用户原文 → 提词器滚动显示，**不做任何 LLM 改写**
    🎭 **数字人模式**：用户原文 → TTS 配音 → 数字人出镜 → MP4

与课案的差异
    | 课案 | 本项目 |
    |---|---|
    | 数字人引擎：本地 HeyGem（Docker 三容器 / AutoDL） | 百炼爱诗 PixVerse 视频对口型 |
    | 声音克隆：AutoDL Fish-Speech | 百炼 CosyVoice 声音复刻（``tools/voice_clone.py``） |
    | 降级配音：Edge TTS | 勾了克隆时不变（克隆失败 → edge-tts → PixVerse 内置 TTS）；**不勾克隆**则跳过前两级，直接用平台内置音色 |
    | 模特素材目录：HeyGem 的 Docker 挂载目录 | ``MEDIA_AVATAR_INPUT_DIR``（项目内 ``.cache/avatars``） |
    | 模式名 ``mode="heygem"`` | 改为 ``mode="avatar"``（不再用 HeyGem，沿用旧名会误导） |

**修掉课案的一个真 bug**
    课案 ``node_generate_video`` 里写的是 ``state['optimized']``，
    但 ``VideoState`` 里根本没有 ``optimized`` 这个字段（只有 ``raw_script``）——
    数字人模式一跑就 ``KeyError``。本项目统一用 ``state["raw_script"]``。

**本项目特有的两处**（都不改 ``VideoState`` 的键，页面与 ``refresh_avatar_task`` 照旧）
    ① **配音分流**：页面的「优先使用克隆音色」勾选框不只是一个降级开关 ——
       **不勾它**意味着用户已经在上方下拉里挑好了「PixVerse 内置音色」，那就该一步出片。
       原写法是不勾也照样先跑一次 edge-tts，于是 ``speaker_id`` 几乎永远用不上：
       用户选了内置音色，实际听到的却是 edge-tts 的通用音色。现在的分支::

           use_cloned_voice=True  （默认）克隆音色 → 失败降级 edge-tts → 再失败才用 PixVerse 内置 TTS
           use_cloned_voice=False 跳过克隆与 edge-tts，直接 PixVerse 内置 TTS（speaker_id 生效）

       课案里根本没有「内置音色」这条用户可见的路（是被克隆在本机跑不通逼出来的备选），
       所以这条分流的语义由本项目定义。
    ② **模特视频校验**：课案原文是 ``if not avatar_video or not os.path.exists(avatar_video)``，
       本项目只判了非空 —— 路径失效时不会提前拦，要走到 ``tools/avatar_client.py``
       才报「人脸视频不存在」，而那时 TTS 已经白跑完了（本节点注释本来就在强调
       「不要白跑一次 TTS」）。现已把存在性校验补回。

课案的设计意图（保留）
    台词**原样使用，不做 LLM 改写** —— 用户是拿它当提词器读稿的，
    改写等于换了稿子。所以这条链路上一次 LLM 都不调。

踩过的坑
    · **本文件所有 ``tools.*`` 的 import 都写在函数体内部**（``node_generate_video`` 里的
      ``from tools.avatar_client import submit_lipsync``、两个 ``_try_*`` 里的
      ``voice_clone`` / ``media_tools``）。这不是随手写的：自检要先用
      ``sys.modules["tools.xxx"] = 假模块`` 把真模块顶掉，**函数体内 import**
      才会在每次调用时重新查 ``sys.modules``、拿到假模块。
      若有人「顺手整理」成文件顶部的 import，自检会静默地用回真模块 ——
      那一刻它就不再离线：会真发 edge-tts 的网络请求，
      并真实消耗 CosyVoice 的音色创建配额（配额满了整条克隆链都不再可用）。
    · ``from config import settings`` 在文件顶部，它能被 import 到靠的是 venv 的
      ``python_base_root.pth`` 把**仓库根**加进了 ``sys.path``（``config.py`` 在仓库根，
      不在 ``Media_Agent/`` 下）。所以 ``__main__`` 里那段路径引导补的是
      ``Media_Agent/`` 这一层 —— 那是给自检里 ``tools.*`` 用的。
    · 数字人任务提交后**绝不等待**：``tools/avatar_client.wait_task()`` 按 15 秒间隔轮询，
      官方说 1~5 分钟完成；在 Streamlit 的按钮回调里同步等就是整页白屏。
      本节点只提交、留下 ``task_code``，把「等」交给用户点「刷新进度」。

运行方式
    离线自检（不联网、不消耗任何额度）::

        Set-Location F:\\ProGram\\Python_Base\\Media_Agent
        & ..\\.venv\\Scripts\\python.exe workflows\\video.py
"""

import os
import sys

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class VideoState(TypedDict, total=False):
    """口播视频工作流的状态。

    说明：课案把这里写成 ``TypedDict`` 且节点里读 ``state["optimized"]``，
    本项目改成 ``total=False`` —— 所有键都显式声明在这里，谁写哪个字段一目了然，
    并且允许 ``invoke`` 时只传其中一部分键（节点各补自己那份）。
    注意 ``total=False`` **不会**让「读一个没被写过的键」变得安全：
    ``state["x"]`` 照样 KeyError，所以节点里统一用 ``state.get(..., "")``。
    """

    raw_script: str          # 用户输入的台词（原样使用）
    mode: str                # "teleprompter" | "avatar"
    avatar_path: str         # 数字人模特视频路径（10~30 秒正面说话）
    use_cloned_voice: bool   # True=优先克隆音色（失败降级 edge-tts）；False=直接用内置音色
    speaker_id: str          # 走 PixVerse 内置 TTS 时的音色 ID（克隆与 edge-tts 双双失败时生效）

    teleprompter: str        # 提词器内容 = 用户原文
    audio_path: str          # 生成的配音音频
    video_path: str          # 生成的数字人视频（提交后需轮询，这里通常为空）
    task_code: str           # 数字人任务 ID
    avatar_msg: str          # 给界面看的进度/错误说明


# ==========================================================================
# 节点 1：提词器（原样保留，不做任何改写）
# ==========================================================================
def node_teleprompter(state: VideoState) -> dict:
    """用户输入即提词器内容，不做任何 LLM 改写。

    这是「提词器模式」的全部实现：这条路上一次外部调用都没有，
    所以它永远不会失败、也不需要任何密钥。

    Args:
        state: 只读 ``raw_script``。

    Returns:
        差量 dict，只含 ``teleprompter``（= 用户原文，一字不改）。
    """
    return {"teleprompter": state.get("raw_script", "")}


# ==========================================================================
# 节点 2：生成视频（TTS 配音 → 数字人提交）
# ==========================================================================
def node_generate_video(state: VideoState) -> dict:
    """数字人模式：TTS 配音 → 提交对口型任务。

    提词器模式直接短路返回。

    配音走哪条路，由 ``use_cloned_voice`` 决定（详见文件头）：
    ``True`` 克隆音色 → 失败降级 edge-tts → 再失败才用 PixVerse 内置 TTS；
    ``False`` 直接用 PixVerse 内置 TTS（页面的内置音色下拉，一步出片）。

    **本节点不阻塞等待**：对口型任务是异步的（官方说 1~5 分钟），
    在 Streamlit 里阻塞会把界面卡死。这里只提交、留下 ``task_code``，
    由页面的「刷新进度」按钮去轮询。

    Args:
        state: 读 ``mode`` / ``raw_script`` / ``avatar_path`` /
            ``use_cloned_voice`` / ``speaker_id`` 五个键（见 ``VideoState``）。

    Returns:
        差量 dict，含 ``audio_path`` / ``video_path`` / ``task_code`` / ``avatar_msg``
        四个键 —— **每条返回语句都给全这四个键**（页面不判分支，直接按固定键塞进
        ``st.session_state``；少一个键就会留下上一轮的旧值）。
        其中 ``video_path`` 在这里**恒为空串**：成品要等 ``refresh_avatar_task()``
        轮询到完成后再下载落盘。

    Raises:
        不抛异常：参数缺失、模特视频不存在、两级配音都失败、提交失败……
        一律转成 ``avatar_msg`` 里的中文说明。
    """
    mode = state.get("mode", "teleprompter")

    if mode == "teleprompter":
        # 四个键**显式写成空串**而不是 `return {}`：`VideoState` 是 `total=False`，
        # LangGraph 的 invoke 只返回「被写过的键」；节点主动把该清的空值写出来，
        # 结果字典的形状在两条路径上才一致，页面不必分支判断（也不会残留上一轮结果）。
        return {"video_path": "", "audio_path": "", "task_code": "", "avatar_msg": ""}

    script = state.get("raw_script", "")      # ← 课案这里错写成 state["optimized"]
    avatar_video = state.get("avatar_path", "")
    if not script.strip():
        # 用 `strip()` 判空：只有空格/换行的台词交给 TTS 也只会得到「文本为空」，
        # 不如在这里就拦下 —— 省一次网络往返，也省掉一次 PixVerse 的提交尝试。
        return {"avatar_msg": "台词为空", "task_code": "", "audio_path": "", "video_path": ""}

    # 先校验模特视频：没给、或路径已失效，都直接返回提示，**不要白跑一次 TTS**。
    # （课案的顺序是先配音后检查，会白白消耗一次合成额度。）
    # 这里只判得出「文件在不在」，判不出「里面是不是有效的人脸视频」—— 那是 PixVerse 侧的
    # 事；`tools/avatar_client.submit_lipsync()` 里还会再 `Path(...).is_file()` 兜一次。
    if not avatar_video or not os.path.exists(avatar_video):
        # 分清「还没选模特」和「选过但文件没了」：后者在页面上看不出异常，
        # 把失效路径带上才能看出是文件被删/被移走了。
        stale = f"模特视频不存在：`{avatar_video}`\n\n" if avatar_video else ""
        return {
            "audio_path": "",
            "video_path": "",
            "task_code": "",
            "avatar_msg": (
                f"{stale}"
                "请先上传一段模特视频！\n\n"
                "数字人对口型需要一段 10~30 秒的正面说话视频（mp4/mov），"
                "用它的面部运动特征来驱动口型。\n"
                "**照片只能生成静态画面**，必须用视频。"
            ),
        }

    # ---------- Step 1: 产出配音音频 ----------
    # 不勾「优先使用克隆音色」= 用户已经在下拉里挑好了 PixVerse 内置音色 →
    # 这里就不要再用 edge-tts 顶替（否则 speaker_id 永远轮不到，用户听到的是通用音色）。
    # 默认值 `True` 与页面 checkbox 的 `value=True` 一致：不经页面直接调 `run_video()`
    # 时走的是课案那条「克隆优先」链路。
    audio_path = ""
    if state.get("use_cloned_voice", True):
        # 参考音频直接用**模特视频**（课案的 `heygem_voice_clone(source_video)` 也是这么传的）：
        # 数字人自带的声音最自然。`tools/voice_clone.py` 内部会用 FFmpeg 把它转成
        # 16kHz 单声道 wav，再上传成公网 URL 去创建音色。
        audio_path = _try_cloned_voice(script, avatar_video)
        if not audio_path:
            # 勾了克隆但没成功 → 走课案的降级链：edge-tts 通用音色兜底。
            # 这里**不判** edge-tts 的返回值：它失败同样给空串，
            # 于是 `audio_path` 仍是空串，自然落到 Step 2 的 else 分支用内置 TTS 兜底。
            audio_path = _try_edge_tts(script)

    # ---------- Step 2: 提交数字人任务 ----------
    # **延迟 import 是刻意的**，两个原因：
    #   ① 自检要先往 `sys.modules["tools.avatar_client"]` 塞假模块，
    #      函数体内 import 才会在调用时重新查 `sys.modules` 拿到假的
    #      （提到文件顶部就再也打不上桩）；
    #   ② dashscope SDK 是可选依赖，延迟到这里能让「只用提词器模式」的人
    #      不装 SDK 也能 import 本模块。
    from tools.avatar_client import submit_lipsync

    if audio_path:
        # 有配音 → 音频驱动（音色就是我们克隆/合成的那个）
        # 走 `media=[{video_url}, {audio_url}]` 两个元素：人脸视频与音频分别经
        # `tools/dashscope_upload.py` 传到百炼临时存储，再由 PixVerse 按音频驱动口型。
        # 这条就是课案「克隆声音 → TTS → 对口型」的链路。
        submitted = submit_lipsync(video_path=avatar_video, audio_path=audio_path)
        driver = "音频驱动（用生成的配音）"
    else:
        # 没抽音频（用户没勾克隆音色，或克隆与 edge-tts 都失败）→
        # 退回 PixVerse 内置 TTS，用选好的音色一步出片。
        # `media` 换成 `[{video_url}, {lip_sync_tts_speaker_id, id}, {lip_sync_tts_content, content}]`；
        # 官方**不允许** audio 与 tts 同时传，所以这两条分支天然互斥。
        # `speaker_id` 默认 `"auto"`（随机音色）；页面不勾克隆时会给出内置音色下拉。
        submitted = submit_lipsync(
            video_path=avatar_video,
            tts_text=script,
            speaker_id=state.get("speaker_id", "auto"),
        )
        driver = "TTS 文本驱动（PixVerse 内置音色）"

    # `submit_lipsync` 的契约：**绝不抛异常**，返回
    # `{"success": bool, "task_id": str, "message": str}`。
    # 成功时 task_id 是 PixVerse 的任务号（后面靠它轮询）；失败原因全在 message 里。
    if submitted["success"]:
        # `video_path` 保持空串：任务刚提交，成品还没有 —— 页面据此走
        # 「显示任务编号 + 刷新进度按钮」那一支。
        # `audio_path` 原样带出去是为了让用户**先听配音再等出片**（页面会渲染 `<audio>`）；
        # 失败分支同样带上它，免得一次提交失败就把已经花钱合成好的配音白扔。
        return {
            "audio_path": audio_path,
            "video_path": "",
            "task_code": submitted["task_id"],
            "avatar_msg": (
                f"任务已提交（{driver}）\n\n"
                f"任务编号：`{submitted['task_id']}`\n\n"
                "约 2~5 分钟后点下方「刷新进度」查看成品。"
            ),
        }
    # 失败：`task_code` 给空串 —— 页面正是靠它决定要不要显示「刷新进度」按钮，
    # 空串 ＝ 没有可轮询的任务（提交都没成功），这是正确行为，不是漏填。
    # 具体原因直接用 `submit_lipsync` 的 `message`（它把未配密钥 / 上传失败 /
    # 参数互斥等原因都写清楚了）。
    return {
        "audio_path": audio_path,
        "video_path": "",
        "task_code": "",
        "avatar_msg": submitted["message"],
    }


def _try_cloned_voice(script: str, avatar_video: str) -> str:
    """尝试「克隆音色 → 合成」。任何一步失败都静默返回空串，让上游降级。

    Args:
        script: 要合成的台词（原样使用）。
        avatar_video: 模特视频路径，既当人脸素材也当音色参考。

    Returns:
        合成出来的音频路径；**克隆或合成任一失败都返回空串**，
        由调用方接着试 edge-tts。不抛异常。
    """
    try:
        # 同上：函数体内 import 才打得动桩；voice_clone 还要拉 dashscope SDK。
        from tools.voice_clone import clone_voice, tts_with_cloned_voice

        # `clone_voice` 的契约是**绝不抛异常**，返回
        # `{"success", "voice_id", "from_cache", "message"}`。
        # 本机最常见的失败是「没有公网托管」：CosyVoice 要求参考音频是真正的
        # http(s) URL，百炼临时存储的 `oss://` 会被 400 拒（见 tools/voice_clone.py）；
        # 其次是音色创建配额满、或没配 DASHSCOPE_API_KEY。
        # 这里只打日志、返回空串 —— 数字人功能不该因为用不上克隆音色就整体不可用。
        cloned = clone_voice(avatar_video)
        if not cloned["success"]:
            print(f"[口播视频] 声音克隆不可用（{cloned['message']}），降级 edge-tts")
            return ""

        # 合成同样以空串表示失败（未配密钥 / 文本为空 / SDK 报错都在内部收干净了）。
        path = tts_with_cloned_voice(script, cloned["voice_id"])
        if path:
            print(f"[口播视频] 克隆音色配音完成: {path}")
            return path
        print("[口播视频] 克隆音色合成失败，降级 edge-tts")
        return ""
    except Exception as exc:  # noqa: BLE001 —— 降级链上任何异常都不该中断
        # 兜住 import 失败与 SDK 里那些「契约之外」的运行时异常
        # （dashscope 在缺全局密钥时会抛 InputRequired 之类）。
        # 一旦让它冒出去，节点就不再降级、而是整条链路失败。
        print(f"[口播视频] 声音克隆异常（{exc}），降级 edge-tts")
        return ""


def _try_edge_tts(script: str) -> str:
    """edge-tts 兜底配音（免费、无需密钥）。

    Args:
        script: 要合成的台词。

    Returns:
        生成好的 mp3 路径；**失败返回空串**（未装 edge-tts、被
        ``MEDIA_TTS_ENABLED=false`` 关掉、文本为空、网络不通都会走这里），
        让调用方落到最后一级 PixVerse 内置 TTS。
    """
    try:
        # 同上：延迟 import 才打得动桩，也避免把 edge-tts 的可用性绑到模块导入上。
        from tools.media_tools import generate_tts

        # `generate_tts` 内部已经把网络异常收干净（失败返回空串），并支持
        # `MEDIA_TTS_FALLBACK_VOICE` 指定音色（默认 zh-CN-XiaoxiaoNeural）。
        # 它是降级链的中间一级：音色通用、但免费且不需要任何密钥。
        return generate_tts(script)
    except Exception as exc:  # noqa: BLE001
        # 兜的是 import 本身失败与未预料的异常 —— 降级链的最后一环也不能把异常甩给节点。
        print(f"[口播视频] edge-tts 异常: {exc}")
        return ""


# ==========================================================================
# 构图
# ==========================================================================
# 两个节点、一条直线：`teleprompter` 先跑（不分模式都写一次提词器内容），
# `generate` 自己按 `mode` 决定短路还是真出片。
# 为什么不在图上做按模式的分支：两种模式**共用同一份 state 和同一个出口**，
# 分支图只会让「结果字典缺哪个键」多出一种情况；把短路放进节点里
# （`mode == "teleprompter"` 直接返回四个空串）反而更简单。
# 代价是数字人模式也白跑一次提词器节点 —— 那只是一次字典构造 + 一次字符串取值，0 次外部调用。
builder = StateGraph(VideoState)
builder.add_node("teleprompter", node_teleprompter)
builder.add_node("generate", node_generate_video)

builder.add_edge(START, "teleprompter")
builder.add_edge("teleprompter", "generate")
builder.add_edge("generate", END)

video_graph = builder.compile()


# ==========================================================================
# 对外接口
# ==========================================================================
def run_video(
    raw_script: str,
    mode: str = "teleprompter",
    avatar_path: str = "",
    use_cloned_voice: bool = True,
    speaker_id: str = "auto",
) -> dict:
    """运行口播视频工作流。

    Args:
        raw_script: 用户原始台词（原样使用，不做改写）。
        mode: ``"teleprompter"``（只看稿）或 ``"avatar"``（数字人出镜）。
        avatar_path: 模特视频路径（数字人模式必填，且必须真实存在）。
        use_cloned_voice: 是否优先尝试克隆音色。``True`` 时克隆失败依次降级
            edge-tts、PixVerse 内置 TTS；``False`` 时直接走 PixVerse 内置 TTS
            （即 ``speaker_id``）一步出片，不跑 edge-tts。
        speaker_id: 走 PixVerse 内置 TTS 时的音色 ID。

    Returns:
        ``{"teleprompter","audio_path","video_path","task_code","avatar_msg"}``
        —— 失败也在 ``avatar_msg`` 里给中文说明，不抛异常。

    Raises:
        无。两个节点都自己兜底；``video_graph.invoke`` 拿到的是完整字典，
        并且**每个键都存在**（两个节点都显式写全了自己负责的字段），
        调用方可以直接下标取值。
    """
    return video_graph.invoke({
        "raw_script": raw_script,
        "mode": mode,
        "avatar_path": avatar_path,
        "use_cloned_voice": use_cloned_voice,
        "speaker_id": speaker_id,
    })


def refresh_avatar_task(task_code: str) -> dict:
    """轮询数字人任务并（完成后）把成品下载到本地。

    给页面「刷新进度」按钮用。**这是唯一会把成品落盘的地方** ——
    ``node_generate_video`` 只提交任务就返回了（在 Streamlit 里同步等待会白屏）。
    每次调用只查一次，整条链路不阻塞；1~5 分钟的等待交给用户按多次按钮。

    Args:
        task_code: ``node_generate_video`` 交出来的任务号；空串直接判失败。

    Returns:
        ``{"finished","success","video_path","message"}``：

        * ``finished=False`` → 还在跑（PENDING/RUNNING），``message`` 是当前状态；
        * ``finished=True, success=False`` → 任务失败（FAILED/CANCELED/UNKNOWN，
          其中 UNKNOWN 通常是超过了 task_id 的 24 小时有效期），
          **或**生成成功但下载失败（``message`` 里带上 URL 让用户手动打开）；
        * ``finished=True, success=True`` → ``video_path`` 是本地成品路径。

        不抛异常：``query_task`` / ``download_result`` 都以空值表示失败。
    """
    if not task_code:
        # 页面在 task_code 为空时不会渲染刷新按钮，但直接调用还得扛得住
        # （session_state 里可能残留空串）。
        return {"finished": False, "success": False, "video_path": "",
                "message": "任务编号为空"}

    # 同上：延迟 import 才能被自检的假模块顶掉，也避免 SDK 变成模块级依赖。
    from tools.avatar_client import download_result, query_task

    # `query_task` 返回 `{"status","finished","success","video_url","message"}`，
    # 绝不抛异常；PENDING / RUNNING 都只是「还没好」，不算错误。
    q = query_task(task_code)
    if not q["finished"]:
        return {"finished": False, "success": False, "video_path": "",
                "message": q["message"]}

    if not q["success"]:
        # 真失败：把 tools 层给的原文（含状态码与 `code`/`message`）透给页面。
        return {"finished": True, "success": False, "video_path": "",
                "message": q["message"]}

    # 官方说成品 URL 不保证长期有效，所以一查到就立刻下载落盘。
    local = download_result(q["video_url"])
    if local:
        return {"finished": True, "success": True, "video_path": local,
                "message": "生成完成，已保存到本地"}
    # 下载失败**不算任务失败**：把 URL 交出去让用户手动打开，
    # 总比只回一句「失败」而把已经花掉额度生成好的成品丢掉强。
    #
    # 注意：这里返回 `finished=True` 但页面**不会**清掉 `task_code`，所以用户
    # 还能再点一次「刷新进度」重试下载 —— 这是有意的（网络抖动重试一次就好了）。
    # 重复下载不会产生垃圾文件：`download_result()` 的落盘名是 video_url 的 md5，
    # 同一份成品只会覆盖写同一个路径。代价仅是重试时多一次 `query_task`（RPS 20，
    # 且 `query_task` 只是取当次快照、不改变任务状态）。
    return {"finished": True, "success": False, "video_path": "",
            "message": f"生成完成但下载失败，可手动访问: {q['video_url']}"}


if __name__ == "__main__":
    # 直接跑本文件时 sys.path[0] 是 workflows/，需要把 Media_Agent 加进来。
    # 补的是 `Media_Agent/` 这一层（`__file__` 的爷爷目录）：`tools/` 包在它下面。
    # 文件顶部的 `from config import settings` 不靠这段 —— 它靠 venv 的
    # `python_base_root.pth` 把仓库根加进了 sys.path（见文件头「踩过的坑」）。
    from pathlib import Path

    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    print("=== 口播视频工作流自检（离线）===")

    # 1) 图结构
    g = video_graph.get_graph()
    nodes = sorted(g.nodes)
    assert "teleprompter" in nodes and "generate" in nodes, nodes
    print(f"  图节点: {nodes}  OK")

    # 2) 提词器模式：原样返回，不调任何外部服务
    r = run_video("第一行台词\n第二行台词", mode="teleprompter")
    assert r["teleprompter"] == "第一行台词\n第二行台词", r
    assert r["audio_path"] == "" and r["task_code"] == "", r
    print("  提词器模式原样返回         OK")

    # 3) 数字人模式：没给模特视频 → 必须给出中文提示而不是抛异常
    r2 = run_video("你好", mode="avatar", avatar_path="")
    assert r2["task_code"] == "", r2
    assert "模特视频" in r2["avatar_msg"], r2["avatar_msg"]
    print("  数字人缺模特 → 中文提示    OK")

    # 3b) 模特路径已失效（文件被删/被移走）→ 同样在跑 TTS 之前就拦下
    _missing = str(Path(__file__).resolve().parent / "no_such_avatar.mp4")
    r2b = run_video("你好", mode="avatar", avatar_path=_missing)
    assert r2b["task_code"] == "" and r2b["audio_path"] == "", r2b
    assert "不存在" in r2b["avatar_msg"] and _missing in r2b["avatar_msg"], r2b["avatar_msg"]
    print("  模特路径失效 → 提前拦下    OK")

    # 4) 空台词
    r3 = run_video("", mode="avatar", avatar_path="x.mp4")
    assert "台词为空" in r3["avatar_msg"], r3
    print("  空台词处理                 OK")

    # 5) 课案的 KeyError 已修复：不再读 state['optimized']
    #    用字节码常量判断而不是搜源码 —— 源码里的**注释**也提到了这个词，
    #    搜字符串会误报（本条自检第一版就是这么挂的）。
    #    `co_consts` 是**本函数体**的字符串常量表（含它自己的 docstring）；
    #    注释不进字节码，所以这个判据既不会漏、也不会被注释误导。
    consts = node_generate_video.__code__.co_consts
    assert "optimized" not in consts, "仍然引用了不存在的 state['optimized']"
    print("  课案 KeyError 已修复       OK")

    # 6) 刷新接口的失败路径
    #    只测空参这一条：其余分支要调 dashscope 的查询接口（联网），
    #    离线自检不碰它 —— 那几条由 `tools/avatar_client.py` 自己的自检覆盖。
    assert refresh_avatar_task("")["finished"] is False
    print("  refresh_avatar_task 空参   OK")

    # 7) 配音分流（本轮修的那个 bug）：不勾克隆音色 = 用户已经挑了 PixVerse 内置音色，
    #    就该跳过克隆与 edge-tts，把 tts_text / speaker_id 交给 submit_lipsync 一步出片。
    #    离线打桩：用假的 tools.* 模块顶掉真实导入 —— 不联网、也不依赖 dashscope/edge-tts。
    #    **为什么必须打桩、打掉了什么**：
    #      · `tools.avatar_client.submit_lipsync` —— 真调用要把素材传到百炼临时存储、
    #        再提交一个**付费**任务；
    #      · `tools.voice_clone.clone_voice` —— 真调用会**创建真实音色**，而音色有配额
    #        （官方原话「达到配额上限后将无法创建」），拿自检去烧配额是不可接受的；
    #      · `tools.media_tools.generate_tts` —— 真实 edge-tts 是网络调用。
    #    手法是往 `sys.modules` 里塞 `types.ModuleType` 假模块，而不是替换
    #    `workflows.video` 里的全局名：`node_generate_video` 与两个 `_try_*` 的 import
    #    都写在**函数体内**，每次调用才去查 `sys.modules`，所以假模块能生效。
    #    模块一旦被人挪到文件顶部 import，这里就再也打不上桩（见文件头「踩过的坑」）。
    import types

    # 先记下真模块，`finally` 里原样放回：自检不能把 `sys.modules` 留在假模块状态，
    # 否则同一进程里之后 `import tools.avatar_client` 的代码会拿到假货。
    _stubbed = ("tools.avatar_client", "tools.media_tools", "tools.voice_clone")
    _saved_modules = {n: sys.modules.get(n) for n in _stubbed}
    seen: dict = {}                    # submit_lipsync 实收到的关键字参数
    counts = {"clone": 0, "edge": 0}   # 两级配音各被尝试了几次 —— 分流 bug 的判据

    try:
        fake_client = types.ModuleType("tools.avatar_client")

        # 用 `**kwargs` 收参（而不是照抄真签名）：自检关心的是**到底传了哪几个参数**，
        # `"audio_path" not in seen` / `seen["tts_text"]` 这些断言全靠这个字典。
        def _fake_submit(**kwargs):          # 只记录实收参数，不发请求
            seen.clear()
            seen.update(kwargs)
            return {"success": True, "task_id": "stub-task", "message": ""}

        fake_client.submit_lipsync = _fake_submit

        fake_media = types.ModuleType("tools.media_tools")

        # 计数是为了验证「不勾克隆时一次都不该跑」；返回一个**假装存在**的路径就够 ——
        # 后面的 `submit_lipsync` 也被打桩了，不会有谁去读这个文件。
        def _fake_tts(text, output_path=None):   # 顶掉真实 edge-tts（那是网络调用）
            counts["edge"] += 1
            return "C:/fake/edge.mp3"

        fake_media.generate_tts = _fake_tts

        fake_voice = types.ModuleType("tools.voice_clone")

        # **故意让克隆失败**：这样 7b 走的就是课案的降级链（克隆 → edge-tts），
        # 7a 则验证「不勾克隆时连试都不该试」。
        def _fake_clone(path):
            counts["clone"] += 1
            return {"success": False, "message": "stub 故意失败"}

        fake_voice.clone_voice = _fake_clone
        # 这个桩在自检里其实**永远不会被调用**（克隆永远失败，走不到合成那一步）；
        # 留它是为了让假模块的接口与真模块一致，免得将来改自检时才发现缺函数。
        fake_voice.tts_with_cloned_voice = lambda *a, **k: "C:/fake/cloned.mp3"

        sys.modules["tools.avatar_client"] = fake_client
        sys.modules["tools.media_tools"] = fake_media
        sys.modules["tools.voice_clone"] = fake_voice

        # 节点只做 `os.path.exists` 检查，不解析视频内容，所以随便一个真实存在的文件都能过；
        # 用本文件省得依赖外部测试素材。
        _avatar = str(Path(__file__).resolve())      # 只要是真实存在的文件就行，这里用本文件

        # 7a) 不勾克隆音色 → 一次 TTS 都不跑，文本与音色直接交给 PixVerse
        #     `counts == {"clone": 0, "edge": 0}` 才是这条用例的核心：
        #     修 bug 前这里会是 `{"clone": 1, "edge": 1}`（白跑两级 TTS，
        #     于是用户挑好的 `speaker_id` 永远轮不到）。
        r7a = run_video("你好", mode="avatar", avatar_path=_avatar,
                        use_cloned_voice=False, speaker_id="piccolo")
        assert counts == {"clone": 0, "edge": 0}, f"不勾克隆音色时不该再跑 TTS: {counts}"
        assert seen.get("tts_text") == "你好", seen
        assert seen.get("speaker_id") == "piccolo", seen
        # 走的必须是「TTS 文本驱动」，即 `submit_lipsync` 只该收到 tts_text 而不是 audio_path
        assert "audio_path" not in seen, seen
        assert r7a["task_code"] == "stub-task" and "内置音色" in r7a["avatar_msg"], r7a
        print("  不勾克隆 → 内置音色出片   OK")

        # 7b) 勾了克隆（默认）→ 克隆失败降级 edge-tts，仍是音频驱动（课案降级链不变）
        #     不传 `use_cloned_voice` 就是默认的 True —— 顺手把默认值也验了。
        run_video("你好", mode="avatar", avatar_path=_avatar)
        assert counts == {"clone": 1, "edge": 1}, f"克隆失败应降级一次 edge-tts: {counts}"
        assert seen.get("audio_path") == "C:/fake/edge.mp3", seen
        assert "tts_text" not in seen, seen
        print("  勾了克隆 → 失败降级 edge  OK")

        # 7c) 克隆与 edge-tts 都失败 → 退回 PixVerse 内置 TTS（最后一级保底还在）
        #     直接改**假模块的属性**就能模拟「edge-tts 也失败」：
        #     `_try_edge_tts` 每次调用都重新 import，拿到的始终是这个假模块的最新属性。
        fake_media.generate_tts = lambda text, output_path=None: ""
        run_video("你好", mode="avatar", avatar_path=_avatar, speaker_id="auto")
        assert seen.get("tts_text") == "你好", seen
        print("  两级都失败 → 内置 TTS 兜底 OK")
    finally:
        # 还原 `sys.modules`：原本没被 import 过的（None）就删掉，import 过的放回原对象。
        for _name, _mod in _saved_modules.items():
            if _mod is None:
                sys.modules.pop(_name, None)
            else:
                sys.modules[_name] = _mod

    # 把实际用的模型名打出来：`MEDIA_AVATAR_MODEL` 可覆盖，
    # 一眼能确认跑的是不是 PixVerse 那条链路。
    print(f"\n  数字人模型: {settings.media.avatar_model}")
    print("全部自检通过")
