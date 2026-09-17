# -*- coding: utf-8 -*-
"""数字人口播视频 —— 百炼爱诗 PixVerse 视频对口型（替代课案的本地 HeyGem）

课案出处：自媒体课案 → 口播视频 → HeyGem / tools/heygem_client.py

课案原本本地部署 **HeyGem（硅基智能开源数字人）**，Docker 三容器：
``gen-video:8383`` / ``tts:18180`` / ``asr:10095``，硬件门槛 NVIDIA GPU + 32GB RAM。
本项目换成百炼的 **爱诗 PixVerse 视频对口型**，本机不需要 GPU。

⚠️ 先纠正一个容易搞错的对标方向
    HeyGem 的 ``POST /easy/submit`` 入参是 ``{audio_url, video_url}``，
    本质是 **video-to-video 唇形同步**（一段人脸视频 + 一段音频 → 口型对齐的视频），
    不是「文字生成数字人」。所以对标的是 **lipsync API**，不是数字人 SaaS。
    课案代码里也写死了这条约束：「HeyGem 必须有模特视频（不是照片！）……
    照片只能生成静态画面」。

接口对应关系
    | 课案 HeyGem | 本项目 PixVerse | 说明 |
    |---|---|---|
    | ``POST {HEYGEM}/easy/submit`` | ``VideoSynthesis.async_call(model="pixverse/pixverse-lipsync", media=[...])`` | 提交任务，拿 task_id |
    | ``GET  {HEYGEM}/easy/query?code=`` | ``VideoSynthesis.fetch(task=...)`` | 轮询进度 |
    | 状态码 1/2/3 = 进行中/完成/失败 | ``task_status`` = PENDING/RUNNING/SUCCEEDED/FAILED | 见 ``_normalize_status`` |
    | 输出落在 ``temp/{task_code}/result.avi`` | ``output.video_url``（公网 MP4，需要自己下载回来） | 本项目会自动下载到本地 |
    | ``chaofen`` 4K 超分参数 | 无对应能力 | 本项目不做超分（见 README「与课案的差异」） |

两种驱动方式（PixVerse 特有，二选一）
    ① **音频驱动**：``media=[{video_url}, {audio_url}]``
       —— 配 ``tools/voice_clone.py`` 克隆出的音色使用，对应课案「克隆声音 → TTS → 对口型」那条链路。
    ② **TTS 文本驱动**：``media=[{video_url}, {lip_sync_tts_speaker_id,id:"auto"},
       {lip_sync_tts_content,content:"台词"}]``
       —— 由 PixVerse 内置 TTS 直接读文本，**一步出片**，
       相当于把课案的「声音克隆 + TTS + 提交 HeyGem」三跳合并成一跳。
       代价是只能用平台内置音色（见下方 ``PIXVERSE_SPEAKERS``），不能用自己的声音。

三个必须知道的约束
    1. **素材必须是「可访问的资源 URL」**：本地模特视频/音频要先经
       ``tools/dashscope_upload.py`` 换成 ``oss://`` 临时 URL，
       且上传时的 ``model`` 必须就是 ``pixverse/pixverse-lipsync``。

       ✅ **实测结论：PixVerse 直接吃 ``oss://``**（VERIFY_REPORT.md 5.11：
       6 秒素材真实出片 ``avatar_27280.mp4``）。所以数字人**不需要**自建公网托管 ——
       这一点和声音克隆正好相反（``create_voice`` 明确拒收 ``oss://``，
       见 ``tools/voice_clone.py``）。原因是 ``oss://`` 属于百炼自家临时存储，
       SDK 调用时会自动带上 ``X-DashScope-OssResourceResolve: enable`` 头让服务端解析它。

    2. **只在华北2（北京）地域提供**，且官方文档只给「业务空间专属域名」形式：
       ``https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1``。
       所以 ``DASHSCOPE_WORKSPACE_ID`` 建议配上；没配则用通用域名试。
       **还需要先在百炼控制台搜索 PixVerse 并点「立即开通」。**
    3. **时长限制**：视频 ≤250MB / ≤300 秒；音频 ≤100MB / ≤300 秒。
       计费按音频时长向上取整（TTS 模式按文本 UTF-8 字节数 ÷ 15 向上取整）。
       所以「10~30 秒的模特视频」既是最佳效果区间，也是最省钱的做法。

两个使用上的硬约束（代码里会直接拦掉）
    · ``audio_path`` 与 ``tts_text`` **只能二选一**（官方接口不允许同时传）；
    · **任务天生异步**：``async_call`` 只提交，成品要等几分钟才出来
      （页面提示「约 2~5 分钟」，等待上限见 ``MEDIA_AVATAR_TIMEOUT``，默认 900 秒）。
      所以 ``submit_lipsync()`` 拿到 ``task_id`` 就返回、**绝不在这里阻塞** ——
      页面把 task_id 显示给用户，隔一会儿点「刷新进度」再走 ``query_task()``。
      等不到结果的两条早退路径见 ``wait_task()``：等满超时（``timed_out``），
      或**查询本身连续失败 3 次**（``aborted``，不再干等 900 秒）。

内置音色
    ``PIXVERSE_SPEAKERS`` 共 14 个（含 ``auto`` = 随机）。TTS 文本驱动只能用它，
    用不上自己克隆的声音；要用自己的声音就走音频驱动那条路（配 ``voice_clone.py``）。

文件末尾的 ``__main__`` 是**离线打桩**自检：只跑参数校验与失败路径，不提交真实任务。
"""

import hashlib
import os
import sys
import time
from pathlib import Path

import requests

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# PixVerse 内置 TTS 音色（官方文档的取值表，id 传字符串）
# · 键是**字符串**形式的 id（调用处也按 `str(speaker_id)` 传）；
#   值是给页面显示的中文名，页面下拉就是遍历这个 dict 生成的
#   （`views/video.py:325` 拼成 `"<id> - <中文名>"`，回读时再 `split(" - ")[0]`）。
# · 编号不连续（2 / 4 / 6 / 10 … 21）是官方取值表本来就这样，
#   这里没有做筛选或重排 —— 对不上号时先怀疑上游改表，不要自行补号。
PIXVERSE_SPEAKERS = {
    "auto": "随机",
    "2": "詹有鱼",
    "4": "外国阿利",
    "6": "李解",
    "10": "姜姜好",
    "11": "老森",
    "12": "李杰克",
    "13": "钱多多",
    "14": "呆萌王小拍",
    "16": "屯里大嗓",
    "18": "豫语汉子",
    "19": "宝岛囡囡",
    "20": "陕西掌柜",
    "21": "港风阿sir",
}

# 官方状态枚举
# 终态取值口径：只有 SUCCEEDED 算成功；FAILED / CANCELED / UNKNOWN 都是「别再等了」。
# UNKNOWN 通常是 task_id 超过 24 小时有效期（见 query_task 的提示）。
_TERMINAL_OK = "SUCCEEDED"
_TERMINAL_FAIL = {"FAILED", "CANCELED", "UNKNOWN"}


def is_configured() -> bool:
    """百炼密钥是否已配置。

    Returns:
        bool：``settings.dashscope_api_key`` 非空为 True。**只判有没有配**，
        不校验密钥能否通过（那要真提交一次任务才知道）。
    """
    return bool(settings.dashscope_api_key)


def _out(rsp) -> dict:
    """把 dashscope 响应对象里的 output 统一取成 dict。

    dashscope 的 ``output`` 是 DictMixin，属性访问和下标访问都能用；
    但失败响应有时是普通 dict —— 两种都兼容。

    Args:
        rsp: ``VideoSynthesis.async_call()`` / ``fetch()`` 返回的响应对象
            （也可能是个 dict，或干脆没有 ``output``）。

    Returns:
        dict：output 的内容；取不到就给空 dict（调用方一律用 ``.get()`` 读字段，
        所以空 dict 会自然退化成「这几个字段都是空」，不会炸）。
    """
    output = getattr(rsp, "output", None)
    if output is None:
        return {}
    if isinstance(output, dict):
        return output
    try:
        return dict(output)
    except Exception:  # noqa: BLE001
        # 退化成逐个读常见字段
        # （DictMixin 在个别失败响应上转不成 dict，但属性访问还是好的）
        return {
            k: getattr(output, k, None)
            for k in ("task_id", "task_status", "video_url", "code", "message", "results")
        }


def _api_kwargs() -> dict:
    """拼 dashscope 调用的公共参数（密钥 + 可选业务空间）。

    Returns:
        dict：至少含 ``api_key``；配了 ``DASHSCOPE_WORKSPACE_ID`` 时多一个 ``workspace``。
        这个 dict 直接用 ``**`` 展开进 ``async_call()`` / ``fetch()``。

    为什么要抽成函数：PixVerse 只在华北2 提供，官方给的域名是
    ``https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1`` 这种**业务空间专属**
    形式，所以 workspace 必须跟着每一次调用走；散在两处手写容易漏掉其中一处。
    """
    kwargs = {"api_key": settings.dashscope_api_key}
    # 没配业务空间就不传这个键（传 None/空串都不如直接不传 —— 让 SDK 走默认域名）。
    if settings.dashscope_workspace_id:
        kwargs["workspace"] = settings.dashscope_workspace_id
    return kwargs


# --------------------------------------------------------------------------
# 提交任务
# --------------------------------------------------------------------------
def submit_lipsync(
    video_path: str,
    audio_path: str = "",
    tts_text: str = "",
    speaker_id: str = "auto",
    watermark: bool = False,
) -> dict:
    """提交对口型任务（**不等待**），返回 task_id。

    Args:
        video_path: 本地人脸视频路径（课案里的「模特视频」，10~30 秒正面说话最佳）。
        audio_path: 本地音频路径。与 ``tts_text`` 二选一。
        tts_text: 台词文本，走 PixVerse 内置 TTS（与 ``audio_path`` 二选一）。
        speaker_id: 内置音色 ID，见 ``PIXVERSE_SPEAKERS``，默认 ``"auto"``。
        watermark: 是否加「AI 生成」水印。

    Returns:
        ``{"success": bool, "task_id": str, "message": str}`` —— **绝不抛异常**。
        失败时 ``success=False``、``task_id=""``，中文原因在 ``message`` 里
        （没配密钥 / 视频不存在 / 参数互斥 / 上传失败 / 提交异常）。
        **成功只代表任务已提交**，不代表成片已生成 —— 出片要另走 ``query_task()``。
    """
    result = {"success": False, "task_id": "", "message": ""}

    if not is_configured():
        result["message"] = (
            "未配置 DASHSCOPE_API_KEY（见根目录 .env）。"
            "数字人对口型走百炼爱诗 PixVerse。"
        )
        print(f"[数字人] {result['message']}")
        return result

    if not video_path or not Path(video_path).is_file():
        # 这里拦住「照片」也拦住「远程 URL」：接口要的是**本地可上传的人脸视频文件**，
        # 照片只能生成静态画面（课案原话，见模块 docstring 的对标方向）。
        result["message"] = (
            f"人脸视频不存在: {video_path}\n"
            "需要一段 10~30 秒的正面说话视频（mp4/mov/webm）——"
            "和课案一样，照片只能生成静态画面。"
        )
        print(f"[数字人] {result['message']}")
        return result

    # 下面两条是官方接口的硬约束，先在本机拦掉：真发过去也只会拿到一句难懂的
    # 参数错误，不如在这里直接给中文说明（页面直接把 message 显示出来）。
    if not audio_path and not tts_text:
        result["message"] = "必须提供 audio_path（音频驱动）或 tts_text（TTS 文本驱动）之一"
        print(f"[数字人] {result['message']}")
        return result
    if audio_path and tts_text:
        result["message"] = "audio_path 与 tts_text 只能二选一（官方接口不允许同时传）"
        print(f"[数字人] {result['message']}")
        return result

    model = settings.media.avatar_model
    from tools.dashscope_upload import upload_file

    # ---- 上传人脸视频 ----
    # 上传时必须带上**用途模型名**：百炼临时存储把文件和模型绑定，
    # 传错模型不会在上传时报错，只会在真正调用模型时说「文件不可用」。
    print(f"[数字人] 上传人脸视频（用途模型 {model}）...")
    video_url = upload_file(str(video_path), model)
    if not video_url:
        result["message"] = "人脸视频上传到百炼临时存储失败（见上方日志）"
        print(f"[数字人] {result['message']}")
        return result

    # ---- 组装 media 列表 ----
    # 这是 PixVerse 最容易写错的地方：`media` 是一个列表（官方叫 at-least-one，
    # 至少要有一项），但**每种元素的字段名不一样** —— 前两项用 `url`，
    # 后两项用 `id` / `content`，全靠 `type` 区分。
    # 拿一个字段名一路拼到底，只会换来一句难懂的参数错误。
    media = [{"type": "video_url", "url": video_url}]
    if audio_path:
        if not Path(audio_path).is_file():
            result["message"] = f"音频文件不存在: {audio_path}"
            print(f"[数字人] {result['message']}")
            return result
        print("[数字人] 上传配音音频...")
        # 音频同样走百炼临时存储换 oss:// —— 实测 PixVerse 认这个前缀，不用自建托管。
        audio_url = upload_file(str(audio_path), model)
        if not audio_url:
            result["message"] = "配音音频上传到百炼临时存储失败（见上方日志）"
            print(f"[数字人] {result['message']}")
            return result
        media.append({"type": "audio_url", "url": audio_url})
        mode_desc = "音频驱动"
    else:
        # TTS 文本驱动：一步出片（PixVerse 自己合成语音并对口型），
        # 代价是只能用平台内置音色，用不了克隆音色。
        media.append({"type": "lip_sync_tts_speaker_id", "id": str(speaker_id)})
        media.append({"type": "lip_sync_tts_content", "content": tts_text})
        mode_desc = f"TTS 文本驱动（音色 {speaker_id}）"

    # ---- 提交 ----
    try:
        from dashscope import VideoSynthesis

        print(f"[数字人] 提交任务（{mode_desc}），模型 {model}")
        # async_call：**只提交不等待**。成品要等几分钟，阻塞在这里会把 Streamlit
        # 页面卡死（这也是本模块只暴露 submit / query 两个动作的原因）。
        rsp = VideoSynthesis.async_call(
            model=model, media=media, watermark=watermark, **_api_kwargs()
        )
    except Exception as exc:  # noqa: BLE001
        # 模型没开通、欠费、地域不对、media 形状写错…… 都在这里收成中文提示。
        result["message"] = f"提交任务异常: {exc}"
        print(f"[数字人] {result['message']}")
        return result

    output = _out(rsp)
    task_id = str(output.get("task_id") or "")
    status = str(output.get("task_status") or "")
    if not task_id:
        # 提交「没抛异常但没有 task_id」= 服务端拒绝了这次调用，
        # code / message 才是真因，一起拼进 message 给页面看。
        result["message"] = (
            f"提交失败（未返回 task_id）: {output.get('code', '')} "
            f"{output.get('message', '')}".strip()
        )
        print(f"[数字人] {result['message']}")
        return result

    result.update(success=True, task_id=task_id, message=f"任务已提交（{status}，{mode_desc}）")
    print(f"[数字人] task_id={task_id} 状态={status}")
    return result


# --------------------------------------------------------------------------
# 查询进度
# --------------------------------------------------------------------------
# 连续查询失败多少次就放弃等待（`wait_task()` 的早退阈值）。
# 取 3 的依据：轮询间隔默认 15 秒（官方建议的下限），3 × 15 = 45 秒 ——
# 公网抖动**连续三次**都失败的概率极低；真连续三次，说明服务端或网络已经不可用，
# 而按老的写法（只等 900 秒超时）用户就是一分钟一分钟地干等，还看不到失败原因。
_MAX_CONSECUTIVE_QUERY_ERRORS = 3


def query_task(task_id: str) -> dict:
    """查询任务进度。

    对应课案的 ``heygem_query_task()``。

    Args:
        task_id: ``submit_lipsync()`` 返回的任务编号（**有效期 24 小时**，过期后查回来
            的 ``task_status`` 是 ``UNKNOWN``）。

    Returns:
        ``{"status": str, "finished": bool, "success": bool,
           "video_url": str, "message": str, "error": bool}`` —— 绝不抛异常。

        三态语义（页面按它决定是「继续等」还是「收摊」）：
        · ``finished=False, error=False`` —— PENDING / RUNNING，**不是错误**，接着等；
        · ``finished=True, success=True`` —— 出片了，``video_url`` 可直接下载；
        · ``finished=True, success=False`` —— FAILED / CANCELED / UNKNOWN，别等了。

        ``error`` 是**新增键**（不动其它键的语义）：它标记的是「**这次查询动作本身**
        失败」（网络抖动 / task_id 查不动等走 except 的那一支），与任务状态无关。
        那一支刻意**不置 finished**（一次查询失败不代表任务真的挂了），代价是调用方
        光看 ``finished`` 分不清「还在跑」和「查不动」—— ``wait_task()`` 就是靠累计
        它来提前收摊的（见 ``_MAX_CONSECUTIVE_QUERY_ERRORS``）。
        查询成功时（哪怕 ``status`` 是空串）``error`` 一定是 False。
    """
    result = {
        "status": "", "finished": False, "success": False,
        "video_url": "", "message": "", "error": False,
    }

    if not task_id:
        result["message"] = "task_id 为空"
        return result
    if not is_configured():
        result["message"] = "未配置 DASHSCOPE_API_KEY"
        return result

    try:
        from dashscope import VideoSynthesis

        # 查询接口没有 RPS 压力（官方给的是 20），但页面是「用户点一下刷一次」，
        # 不存在高频轮询的问题；`fetch` 按 task_id 取当次快照，不改变任务状态。
        rsp = VideoSynthesis.fetch(task=task_id, **_api_kwargs())
    except Exception as exc:  # noqa: BLE001
        # 网络抖动 / task_id 不存在都会抛。这里**不置 finished** ——
        # 让 `wait_task()` 的超时机制去收尾，比在这里判定「永久失败」更稳
        # （一次查询失败不代表任务真的挂了）。
        # 但要置 `error=True`：不暴露「这次查询没成功」的话，`wait_task()` 只能一路
        # 轮询到 900 秒超时 —— 服务端持续报错时用户就是干等 900 秒且看不到原因。
        result["error"] = True
        result["message"] = f"查询异常: {exc}"
        print(f"[数字人] {result['message']}")
        return result

    output = _out(rsp)
    status = str(output.get("task_status") or "")
    result["status"] = status

    if status == _TERMINAL_OK:
        # 文档里 video_url 是 output 的直接字段；部分响应放在 results 里，两处都找
        # （官方给的两个字段位置不一致，实测按「先直取、再下探 results」两处兜住）。
        video_url = output.get("video_url") or ""
        if not video_url:
            results = output.get("results")
            if isinstance(results, dict):
                video_url = results.get("video_url") or ""
        result.update(success=True, finished=True, video_url=str(video_url or ""))
        result["message"] = "生成完成"
        print(f"[数字人] 任务完成: {video_url}")
    elif status in _TERMINAL_FAIL:
        result["message"] = (
            f"任务失败/不可查（{status}）: {output.get('code', '')} "
            f"{output.get('message', '')}".strip()
            # UNKNOWN 最常见的原因就是 task_id 过期，直接把这句话写给用户，
            # 省一轮「为什么我的任务查不到」的排查。
            + ("\n提示：task_id 有效期 24 小时，超时会返回 UNKNOWN。" if status == "UNKNOWN" else "")
        )
        print(f"[数字人] {result['message']}")
    else:
        # PENDING / RUNNING 都算「还没好」，不是错误
        # （finished 保持 False，调用方据此继续等，message 只用来在页面上显示进度）。
        result["message"] = f"处理中（{status or '未知状态'}）"
    return result


def wait_task(task_id: str, timeout: int = None, interval: int = 15) -> dict:
    """轮询直到任务结束（或超时 / 查询持续失败）。

    官方建议轮询间隔 ≥15 秒（查询接口默认 RPS 20）。

    Args:
        task_id: ``submit_lipsync()`` 返回的任务编号。
        timeout: 最长等待秒数；默认 ``settings.media.avatar_timeout``（900 秒）。
            官方说成品要等几分钟，900 秒是给到 3 倍余量又不至于把页面挂太久。
        interval: 两次查询之间的间隔秒数，默认 15（官方建议的下限）。

    Returns:
        与 ``query_task()`` 同构，多两个键：

        · ``"timed_out": bool`` —— 等满 ``timeout`` 秒仍未到终态；
        · ``"aborted": bool`` —— 提前收摊，两种情况：
          ① **连续 ``_MAX_CONSECUTIVE_QUERY_ERRORS`` 次** ``query_task()``
          返回 ``error=True``；
          ② ``task_id`` 为空 —— 这种情况一次查询都不会发（原因见函数内注释）。

        三条早退路径**互斥且可区分**：超时是 ``timed_out=True, aborted=False``；
        查询持续失败与空 task_id 都是 ``aborted=True, timed_out=False``
        （``message`` 里写明是哪一种）；正常拿到终态时两个都是 False。
        三条路径都**不动 ``finished`` / ``success``** —— 超时不算任务失败，查询失败
        也不算：任务在服务端可能还在跑，用户之后仍可点「刷新进度」再查一次。

        ⚠️ 多出的 ``"error"`` 键是从 ``query_task()`` 原样带过来的，
        ``wait_task`` 自己不改它。
    """
    timeout = timeout or settings.media.avatar_timeout
    deadline = time.time() + timeout
    # 先摆一个「还没开始查」的默认值：循环一次都没跑成（timeout<=0）时也有返回值，
    # 不至于把 last 留成未定义。
    last = {"status": "", "finished": False, "success": False,
            "video_url": "", "message": "未开始", "error": False,
            "timed_out": False, "aborted": False}

    if not task_id:
        # 空 task_id 必须在**进循环之前**就拦掉：`query_task("")` 走的是它自己的
        # 「任务编号为空」前置校验支路，那条路返回 `finished=False, error=False`，
        # 既不是终态也不算查询失败 —— 于是外层会一路轮询到 900 秒才靠超时收摊。
        # 这里直接收摊，标记成 aborted（与「查询持续失败」同一类：都是没等到结果、
        # 但**任务本身未必失败**，所以不动 finished / success）。
        last["aborted"] = True
        last["message"] = "任务编号为空，未开始轮询"
        print(f"[数字人] {last['message']}")
        return last

    # 只数**连续**失败：中间只要有一次查询成功（无论 status 是 PENDING/RUNNING
    # 还是终态）就清零，偶发一次网络抖动不该累积到阈值上。
    consecutive_errors = 0
    while time.time() < deadline:
        last = query_task(task_id)
        if last.get("error"):
            consecutive_errors += 1
            if consecutive_errors >= _MAX_CONSECUTIVE_QUERY_ERRORS:
                last["timed_out"] = False
                last["aborted"] = True
                # message 里带失败次数 + 最后一次的原始异常文本
                # （`query_task()` 的 message 就是「查询异常: <原文>」），
                # 否则用户只知道「提前结束了」，不知道到底报了什么。
                last["message"] = (
                    f"连续 {consecutive_errors} 次查询失败，已提前结束等待"
                    f"（每 {interval} 秒查一次）：{last.get('message', '')}"
                )
                print(f"[数字人] {last['message']}")
                return last
        else:
            consecutive_errors = 0
            if last["finished"]:
                # 明确的终态（成功或失败）就不再循环；同时把两个早退标记显式置 False，
                # 让调用方只读这两个键就能区分「等到了结果」「等超时了」「查询持续失败」。
                last["timed_out"] = False
                last["aborted"] = False
                return last
        # 睡在每次查询之后：第一查立即执行（刚提交的任务也该马上看一次状态）。
        time.sleep(interval)

    last["timed_out"] = True
    last["aborted"] = False
    # 超时**不算任务失败**：任务在服务端可能还在跑，用户之后仍可点「刷新进度」再查一次
    # （页面就是这么用的）。所以这里只改 message，不动 success / finished。
    last["message"] = f"等待超时（{timeout} 秒），最后状态: {last.get('status') or '未知'}"
    print(f"[数字人] {last['message']}")
    return last


# --------------------------------------------------------------------------
# 下载成品
# --------------------------------------------------------------------------
def download_result(video_url: str, output_path: str = None) -> str:
    """把生成好的视频下载到本地（官方说链接不保证长期有效，及时落盘）。

    Args:
        video_url: ``query_task()`` 成功时返回的 ``video_url``（公网 http(s)）。
        output_path: 本地保存路径；默认 ``<视频输出目录>/avatar_<md5 前 8 位>.mp4``。

    Returns:
        str：本地文件路径；失败返回空字符串（**不抛异常**）。

    注意：
        页面（``workflows/video.py``）只把 ``success`` 当作「任务完成」，
        下载失败会让它走「生成成功但下载失败」那条提示 —— 因为 ``video_url``
        还有效，用户可以再点一次下载，不该因为一次下载失败就把任务判死。
    """
    if not video_url:
        print("[数字人] 视频 URL 为空")
        return ""

    if output_path is None:
        # 用 md5 而不是内置 hash()：字符串 hash 带**进程级随机盐**，
        # 同一段素材每次新进程会得到不同文件名 → 只会堆积、不会复用。
        _digest = hashlib.md5(video_url.encode("utf-8")).hexdigest()[:8]
        output_path = os.path.join(
            settings.media.get_video_output_dir(),
            f"avatar_{_digest}.mp4",
        )

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 流式下载 + 64KB 分块：成片是几十 MB 的 MP4，一次性读进内存没必要；
        # timeout=300 是连接/读取超时（出片链接通常在 OSS 上，正常很快）。
        with requests.get(video_url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    # iter_content 会吐出 keep-alive 的空块，跳过它们。
                    if chunk:
                        fh.write(chunk)
    except Exception as exc:  # noqa: BLE001
        print(f"[数字人] 下载成品失败: {exc}")
        return ""

    # 1 KB 下限：失败时可能已经建了一个空文件，不能只看「路径存在」就当成功。
    if Path(output_path).is_file() and Path(output_path).stat().st_size > 1024:
        size_mb = Path(output_path).stat().st_size / 1024 / 1024
        print(f"[数字人] 成品已保存: {output_path}（{size_mb:.1f} MB）")
        return output_path
    print("[数字人] 下载文件过小，判定失败")
    return ""


def generate_avatar_video(
    video_path: str,
    audio_path: str = "",
    tts_text: str = "",
    speaker_id: str = "auto",
    watermark: bool = False,
    timeout: int = None,
) -> dict:
    """一步到位：提交 → 轮询 → 下载。

    Args:
        video_path: 本地人脸视频路径（10~30 秒正面说话最佳）。
        audio_path: 本地音频路径。与 ``tts_text`` 二选一。
        tts_text: 台词文本，走 PixVerse 内置 TTS。与 ``audio_path`` 二选一。
        speaker_id: 内置音色 ID，见 ``PIXVERSE_SPEAKERS``，默认 ``"auto"``。
        watermark: 是否加「AI 生成」水印。
        timeout: 轮询总超时秒数；``None`` 时用 ``settings.media.avatar_timeout``。

    Returns:
        ``{"success": bool, "video_path": str, "video_url": str, "task_id": str, "message": str}``
        —— 绝不抛异常。**三种失败要分清楚**（页面按 ``message`` 显示）：
        · 提交失败 —— ``task_id`` 是空串；
        · 任务失败/超时 —— ``task_id`` 有值、``video_path`` 为空；
        · 出片但下载失败 —— ``video_url`` 有值、``video_path`` 为空（可重试下载）。
    """
    # 三步各自都是「失败给空值 + 中文原因」的结构，这里只做串联：
    # 前一步不成功就不再往下走，把它的 message 原样传出去。
    submitted = submit_lipsync(
        video_path=video_path,
        audio_path=audio_path,
        tts_text=tts_text,
        speaker_id=speaker_id,
        watermark=watermark,
    )
    if not submitted["success"]:
        return {"success": False, "video_path": "", "video_url": "",
                "task_id": "", "message": submitted["message"]}

    task_id = submitted["task_id"]
    queried = wait_task(task_id, timeout=timeout)
    if not queried["success"]:
        # 超时也走这一支（timed_out=True 时 success 仍为 False）——
        # 页面会把 task_id 一起显示出来，用户之后还能手动刷新进度。
        return {"success": False, "video_path": "", "video_url": "",
                "task_id": task_id, "message": queried["message"]}

    local = download_result(queried["video_url"])
    # success 看的是**本地有没有文件**：生成成功但没落盘，对下游等于没产出。
    # 同时把 video_url 带出去，让用户能在链接过期前自己再下一遍。
    return {
        "success": bool(local),
        "video_path": local,
        "video_url": queried["video_url"],
        "task_id": task_id,
        "message": "数字人视频生成成功" if local else "生成成功但下载失败",
    }


if __name__ == "__main__":
    # 离线自检：一个真实任务都不提交，只覆盖参数校验与失败路径。
    print("=== 数字人对口型模块自检（离线，不需要密钥）===")

    assert submit_lipsync("")["success"] is False
    assert submit_lipsync("不存在的模特.mp4", tts_text="你好")["success"] is False
    print("  submit_lipsync 失败路径    OK")

    # 二选一约束
    # 用 __file__ 当一个「存在但不是视频」的路径：这两条断言要的是**参数校验**
    # 先于上传发生，所以只要文件存在就能走到互斥判断，不会真去上传。
    r = submit_lipsync(__file__, audio_path="a.wav", tts_text="你好")
    assert r["success"] is False and "二选一" in r["message"], r
    r2 = submit_lipsync(__file__)
    assert r2["success"] is False and "必须提供" in r2["message"], r2
    print("  参数互斥校验               OK")

    # 空 task_id / 空 URL 都在各函数最前面那两道校验里直接返回，
    # 所以下面几条断言既不联网、也不需要密钥。
    assert query_task("")["finished"] is False
    # 新增的 `error` 键在前置校验分支上必须是 False：它不是「没查到」的标记，
    # 只标记「本次查询抛异常」那一支（见 query_task 的三态说明）。
    assert query_task("")["error"] is False
    print("  query_task 失败路径        OK")

    # 查询持续失败必须**早退**，而不是一路轮询到 900 秒超时。
    # 打桩顶掉模块全局名 `query_task`（wait_task 内部查的就是它），interval=0 秒回。
    _real_query = query_task

    def _always_error(_task_id: str) -> dict:
        return {"status": "", "finished": False, "success": False, "video_url": "",
                "message": "查询异常: 打桩的假异常", "error": True}

    try:
        query_task = _always_error  # noqa: F811 —— 本节自检就是要有意顶掉它
        aborted = wait_task("stub-task", timeout=9999, interval=0)
    finally:
        query_task = _real_query
    assert aborted["aborted"] is True and aborted["timed_out"] is False, aborted
    assert str(_MAX_CONSECUTIVE_QUERY_ERRORS) in aborted["message"], aborted
    # 早退同样不改「任务失败」语义：任务在服务端可能还在跑，用户还能手动刷新
    assert aborted["finished"] is False and aborted["success"] is False, aborted
    print(f"  wait_task 连续失败早退      OK  阈值 {_MAX_CONSECUTIVE_QUERY_ERRORS} 次")

    # 空 task_id：`query_task("")` 自己的前置校验支路返回的是
    # `finished=False, error=False`，既不是终态也不算查询失败 —— 不在这里拦掉的话，
    # 外层会一路轮询到 timeout（默认 900 秒）才靠超时收摊。
    # 断言「一次查询都没发」才是关键：只断言 aborted=True 的话，
    # 把它当成超时路径实现（等满 deadline 再返回）也照样能通过。
    _calls = []

    def _counting_query(task_id: str) -> dict:
        _calls.append(task_id)
        return {"status": "", "finished": False, "success": False, "video_url": "",
                "message": "任务编号为空", "error": False}

    try:
        query_task = _counting_query  # noqa: F811 —— 同上，有意顶掉
        # `timeout=1`（而不是 9999）：万一守卫被删掉，这里也只空转 1 秒就返回，
        # 断言照样能红，不会把自检挂住 9999 秒。
        empty = wait_task("", timeout=1, interval=0)
    finally:
        query_task = _real_query
    assert _calls == [], f"空 task_id 不该发出任何查询，实际发了 {len(_calls)} 次"
    assert empty["aborted"] is True and empty["timed_out"] is False, empty
    assert empty["finished"] is False and empty["success"] is False, empty
    print("  wait_task 空 task_id 即退   OK  0 次查询")

    assert download_result("") == ""
    print("  download_result 失败路径   OK")

    # 只报告本机配置状态（配没配密钥、用的哪个模型、内置音色表有几项），不做任何请求
    print(f"\n  密钥已配置: {is_configured()} | 模型: {settings.media.avatar_model}")
    print(f"  可用内置音色: {len(PIXVERSE_SPEAKERS)} 个")
    print("全部自检通过")
