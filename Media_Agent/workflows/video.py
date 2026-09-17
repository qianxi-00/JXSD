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
    本项目改成 ``total=False`` 并要求所有键都显式声明 ——
    少一个键就是 KeyError，``total=False`` 至少让「没赋值」和「没声明」区分开。
    """

    raw_script: str          # 用户输入的台词（原样使用）
    mode: str                # "teleprompter" | "avatar"
    avatar_path: str         # 数字人模特视频路径（10~30 秒正面说话）
    use_cloned_voice: bool   # True=优先克隆音色（失败降级 edge-tts）；False=直接用内置音色
    speaker_id: str          # 走 PixVerse 内置 TTS 时的音色 ID（不勾克隆音色时生效）

    teleprompter: str        # 提词器内容 = 用户原文
    audio_path: str          # 生成的配音音频
    video_path: str          # 生成的数字人视频（提交后需轮询，这里通常为空）
    task_code: str           # 数字人任务 ID
    avatar_msg: str          # 给界面看的进度/错误说明


# ==========================================================================
# 节点 1：提词器（原样保留，不做任何改写）
# ==========================================================================
def node_teleprompter(state: VideoState) -> dict:
    """用户输入即提词器内容，不做任何 LLM 改写。"""
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
    """
    mode = state.get("mode", "teleprompter")

    if mode == "teleprompter":
        return {"video_path": "", "audio_path": "", "task_code": "", "avatar_msg": ""}

    script = state.get("raw_script", "")      # ← 课案这里错写成 state["optimized"]
    avatar_video = state.get("avatar_path", "")
    if not script.strip():
        return {"avatar_msg": "台词为空", "task_code": "", "audio_path": "", "video_path": ""}

    # 先校验模特视频：没给、或路径已失效，都直接返回提示，**不要白跑一次 TTS**。
    # （课案的顺序是先配音后检查，会白白消耗一次合成额度。）
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
    audio_path = ""
    if state.get("use_cloned_voice", True):
        audio_path = _try_cloned_voice(script, avatar_video)
        if not audio_path:
            # 勾了克隆但没成功 → 走课案的降级链：edge-tts 通用音色兜底
            audio_path = _try_edge_tts(script)

    # ---------- Step 2: 提交数字人任务 ----------
    from tools.avatar_client import submit_lipsync

    if audio_path:
        # 有配音 → 音频驱动（音色就是我们克隆/合成的那个）
        submitted = submit_lipsync(video_path=avatar_video, audio_path=audio_path)
        driver = "音频驱动（用生成的配音）"
    else:
        # 没抽音频（用户没勾克隆音色，或克隆与 edge-tts 都失败）→
        # 退回 PixVerse 内置 TTS，用选好的音色一步出片
        submitted = submit_lipsync(
            video_path=avatar_video,
            tts_text=script,
            speaker_id=state.get("speaker_id", "auto"),
        )
        driver = "TTS 文本驱动（PixVerse 内置音色）"

    if submitted["success"]:
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
    return {
        "audio_path": audio_path,
        "video_path": "",
        "task_code": "",
        "avatar_msg": submitted["message"],
    }


def _try_cloned_voice(script: str, avatar_video: str) -> str:
    """尝试「克隆音色 → 合成」。任何一步失败都静默返回空串，让上游降级。"""
    try:
        from tools.voice_clone import clone_voice, tts_with_cloned_voice

        cloned = clone_voice(avatar_video)
        if not cloned["success"]:
            print(f"[口播视频] 声音克隆不可用（{cloned['message']}），降级 edge-tts")
            return ""

        path = tts_with_cloned_voice(script, cloned["voice_id"])
        if path:
            print(f"[口播视频] 克隆音色配音完成: {path}")
            return path
        print("[口播视频] 克隆音色合成失败，降级 edge-tts")
        return ""
    except Exception as exc:  # noqa: BLE001 —— 降级链上任何异常都不该中断
        print(f"[口播视频] 声音克隆异常（{exc}），降级 edge-tts")
        return ""


def _try_edge_tts(script: str) -> str:
    """edge-tts 兜底配音（免费、无需密钥）。"""
    try:
        from tools.media_tools import generate_tts

        return generate_tts(script)
    except Exception as exc:  # noqa: BLE001
        print(f"[口播视频] edge-tts 异常: {exc}")
        return ""


# ==========================================================================
# 构图
# ==========================================================================
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

    给页面「刷新进度」按钮用。

    Returns:
        ``{"finished","success","video_path","message"}``
    """
    if not task_code:
        return {"finished": False, "success": False, "video_path": "",
                "message": "任务编号为空"}

    from tools.avatar_client import download_result, query_task

    q = query_task(task_code)
    if not q["finished"]:
        return {"finished": False, "success": False, "video_path": "",
                "message": q["message"]}

    if not q["success"]:
        return {"finished": True, "success": False, "video_path": "",
                "message": q["message"]}

    local = download_result(q["video_url"])
    if local:
        return {"finished": True, "success": True, "video_path": local,
                "message": "生成完成，已保存到本地"}
    return {"finished": True, "success": False, "video_path": "",
            "message": f"生成完成但下载失败，可手动访问: {q['video_url']}"}


if __name__ == "__main__":
    # 直接跑本文件时 sys.path[0] 是 workflows/，需要把 Media_Agent 加进来
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
    consts = node_generate_video.__code__.co_consts
    assert "optimized" not in consts, "仍然引用了不存在的 state['optimized']"
    print("  课案 KeyError 已修复       OK")

    # 6) 刷新接口的失败路径
    assert refresh_avatar_task("")["finished"] is False
    print("  refresh_avatar_task 空参   OK")

    # 7) 配音分流（本轮修的那个 bug）：不勾克隆音色 = 用户已经挑了 PixVerse 内置音色，
    #    就该跳过克隆与 edge-tts，把 tts_text / speaker_id 交给 submit_lipsync 一步出片。
    #    离线打桩：用假的 tools.* 模块顶掉真实导入 —— 不联网、也不依赖 dashscope/edge-tts。
    import types

    _stubbed = ("tools.avatar_client", "tools.media_tools", "tools.voice_clone")
    _saved_modules = {n: sys.modules.get(n) for n in _stubbed}
    seen: dict = {}
    counts = {"clone": 0, "edge": 0}

    try:
        fake_client = types.ModuleType("tools.avatar_client")

        def _fake_submit(**kwargs):          # 只记录实收参数，不发请求
            seen.clear()
            seen.update(kwargs)
            return {"success": True, "task_id": "stub-task", "message": ""}

        fake_client.submit_lipsync = _fake_submit

        fake_media = types.ModuleType("tools.media_tools")

        def _fake_tts(text, output_path=None):   # 顶掉真实 edge-tts（那是网络调用）
            counts["edge"] += 1
            return "C:/fake/edge.mp3"

        fake_media.generate_tts = _fake_tts

        fake_voice = types.ModuleType("tools.voice_clone")

        def _fake_clone(path):
            counts["clone"] += 1
            return {"success": False, "message": "stub 故意失败"}

        fake_voice.clone_voice = _fake_clone
        fake_voice.tts_with_cloned_voice = lambda *a, **k: "C:/fake/cloned.mp3"

        sys.modules["tools.avatar_client"] = fake_client
        sys.modules["tools.media_tools"] = fake_media
        sys.modules["tools.voice_clone"] = fake_voice

        _avatar = str(Path(__file__).resolve())      # 只要是真实存在的文件就行，这里用本文件

        # 7a) 不勾克隆音色 → 一次 TTS 都不跑，文本与音色直接交给 PixVerse
        r7a = run_video("你好", mode="avatar", avatar_path=_avatar,
                        use_cloned_voice=False, speaker_id="piccolo")
        assert counts == {"clone": 0, "edge": 0}, f"不勾克隆音色时不该再跑 TTS: {counts}"
        assert seen.get("tts_text") == "你好", seen
        assert seen.get("speaker_id") == "piccolo", seen
        assert "audio_path" not in seen, seen
        assert r7a["task_code"] == "stub-task" and "内置音色" in r7a["avatar_msg"], r7a
        print("  不勾克隆 → 内置音色出片   OK")

        # 7b) 勾了克隆（默认）→ 克隆失败降级 edge-tts，仍是音频驱动（课案降级链不变）
        run_video("你好", mode="avatar", avatar_path=_avatar)
        assert counts == {"clone": 1, "edge": 1}, f"克隆失败应降级一次 edge-tts: {counts}"
        assert seen.get("audio_path") == "C:/fake/edge.mp3", seen
        assert "tts_text" not in seen, seen
        print("  勾了克隆 → 失败降级 edge  OK")

        # 7c) 克隆与 edge-tts 都失败 → 退回 PixVerse 内置 TTS（最后一级保底还在）
        fake_media.generate_tts = lambda text, output_path=None: ""
        run_video("你好", mode="avatar", avatar_path=_avatar, speaker_id="auto")
        assert seen.get("tts_text") == "你好", seen
        print("  两级都失败 → 内置 TTS 兜底 OK")
    finally:
        for _name, _mod in _saved_modules.items():
            if _mod is None:
                sys.modules.pop(_name, None)
            else:
                sys.modules[_name] = _mod

    print(f"\n  数字人模型: {settings.media.avatar_model}")
    print("全部自检通过")
