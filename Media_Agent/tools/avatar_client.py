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
    1. **素材必须是公网可访问 URL**：本地模特视频/音频要先经
       ``tools/dashscope_upload.py`` 换成 ``oss://`` 临时 URL，
       且上传时的 ``model`` 必须就是 ``pixverse/pixverse-lipsync``。
    2. **只在华北2（北京）地域提供**，且官方文档只给「业务空间专属域名」形式：
       ``https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1``。
       所以 ``DASHSCOPE_WORKSPACE_ID`` 建议配上；没配则用通用域名试。
       **还需要先在百炼控制台搜索 PixVerse 并点「立即开通」。**
    3. **时长限制**：视频 ≤250MB / ≤300 秒；音频 ≤100MB / ≤300 秒。
       计费按音频时长向上取整（TTS 模式按文本 UTF-8 字节数 ÷ 15 向上取整）。
"""

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
_TERMINAL_OK = "SUCCEEDED"
_TERMINAL_FAIL = {"FAILED", "CANCELED", "UNKNOWN"}


def is_configured() -> bool:
    """百炼密钥是否已配置。"""
    return bool(settings.dashscope_api_key)


def _out(rsp) -> dict:
    """把 dashscope 响应对象里的 output 统一取成 dict。

    dashscope 的 ``output`` 是 DictMixin，属性访问和下标访问都能用；
    但失败响应有时是普通 dict —— 两种都兼容。
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
        return {
            k: getattr(output, k, None)
            for k in ("task_id", "task_status", "video_url", "code", "message", "results")
        }


def _api_kwargs() -> dict:
    """拼 dashscope 调用的公共参数（密钥 + 可选业务空间）。"""
    kwargs = {"api_key": settings.dashscope_api_key}
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
        result["message"] = (
            f"人脸视频不存在: {video_path}\n"
            "需要一段 10~30 秒的正面说话视频（mp4/mov/webm）——"
            "和课案一样，照片只能生成静态画面。"
        )
        print(f"[数字人] {result['message']}")
        return result

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
    print(f"[数字人] 上传人脸视频（用途模型 {model}）...")
    video_url = upload_file(str(video_path), model)
    if not video_url:
        result["message"] = "人脸视频上传到百炼临时存储失败（见上方日志）"
        print(f"[数字人] {result['message']}")
        return result

    # ---- 组装 media 列表 ----
    media = [{"type": "video_url", "url": video_url}]
    if audio_path:
        if not Path(audio_path).is_file():
            result["message"] = f"音频文件不存在: {audio_path}"
            print(f"[数字人] {result['message']}")
            return result
        print("[数字人] 上传配音音频...")
        audio_url = upload_file(str(audio_path), model)
        if not audio_url:
            result["message"] = "配音音频上传到百炼临时存储失败（见上方日志）"
            print(f"[数字人] {result['message']}")
            return result
        media.append({"type": "audio_url", "url": audio_url})
        mode_desc = "音频驱动"
    else:
        media.append({"type": "lip_sync_tts_speaker_id", "id": str(speaker_id)})
        media.append({"type": "lip_sync_tts_content", "content": tts_text})
        mode_desc = f"TTS 文本驱动（音色 {speaker_id}）"

    # ---- 提交 ----
    try:
        from dashscope import VideoSynthesis

        print(f"[数字人] 提交任务（{mode_desc}），模型 {model}")
        rsp = VideoSynthesis.async_call(
            model=model, media=media, watermark=watermark, **_api_kwargs()
        )
    except Exception as exc:  # noqa: BLE001
        result["message"] = f"提交任务异常: {exc}"
        print(f"[数字人] {result['message']}")
        return result

    output = _out(rsp)
    task_id = str(output.get("task_id") or "")
    status = str(output.get("task_status") or "")
    if not task_id:
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
def query_task(task_id: str) -> dict:
    """查询任务进度。

    对应课案的 ``heygem_query_task()``。

    Returns:
        ``{"status": str, "finished": bool, "success": bool,
           "video_url": str, "message": str}`` —— 绝不抛异常。
    """
    result = {
        "status": "", "finished": False, "success": False,
        "video_url": "", "message": "",
    }

    if not task_id:
        result["message"] = "task_id 为空"
        return result
    if not is_configured():
        result["message"] = "未配置 DASHSCOPE_API_KEY"
        return result

    try:
        from dashscope import VideoSynthesis

        rsp = VideoSynthesis.fetch(task=task_id, **_api_kwargs())
    except Exception as exc:  # noqa: BLE001
        result["message"] = f"查询异常: {exc}"
        print(f"[数字人] {result['message']}")
        return result

    output = _out(rsp)
    status = str(output.get("task_status") or "")
    result["status"] = status

    if status == _TERMINAL_OK:
        # 文档里 video_url 是 output 的直接字段；部分响应放在 results 里，两处都找
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
            + ("\n提示：task_id 有效期 24 小时，超时会返回 UNKNOWN。" if status == "UNKNOWN" else "")
        )
        print(f"[数字人] {result['message']}")
    else:
        # PENDING / RUNNING 都算「还没好」，不是错误
        result["message"] = f"处理中（{status or '未知状态'}）"
    return result


def wait_task(task_id: str, timeout: int = None, interval: int = 15) -> dict:
    """轮询直到任务结束（或超时）。

    官方建议轮询间隔 ≥15 秒（查询接口默认 RPS 20）。

    Returns:
        与 ``query_task()`` 同构，多一个 ``"timed_out": bool``。
    """
    timeout = timeout or settings.media.avatar_timeout
    deadline = time.time() + timeout
    last = {"status": "", "finished": False, "success": False,
            "video_url": "", "message": "未开始", "timed_out": False}

    while time.time() < deadline:
        last = query_task(task_id)
        if last["finished"]:
            last["timed_out"] = False
            return last
        time.sleep(interval)

    last["timed_out"] = True
    last["message"] = f"等待超时（{timeout} 秒），最后状态: {last.get('status') or '未知'}"
    print(f"[数字人] {last['message']}")
    return last


# --------------------------------------------------------------------------
# 下载成品
# --------------------------------------------------------------------------
def download_result(video_url: str, output_path: str = None) -> str:
    """把生成好的视频下载到本地（官方说链接不保证长期有效，及时落盘）。

    Returns:
        本地文件路径；失败返回空字符串。
    """
    if not video_url:
        print("[数字人] 视频 URL 为空")
        return ""

    if output_path is None:
        output_path = os.path.join(
            settings.media.get_video_output_dir(),
            f"avatar_{abs(hash(video_url)) % 100000:05d}.mp4",
        )

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with requests.get(video_url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
    except Exception as exc:  # noqa: BLE001
        print(f"[数字人] 下载成品失败: {exc}")
        return ""

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

    Returns:
        ``{"success": bool, "video_path": str, "video_url": str, "task_id": str, "message": str}``
    """
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
        return {"success": False, "video_path": "", "video_url": "",
                "task_id": task_id, "message": queried["message"]}

    local = download_result(queried["video_url"])
    return {
        "success": bool(local),
        "video_path": local,
        "video_url": queried["video_url"],
        "task_id": task_id,
        "message": "数字人视频生成成功" if local else "生成成功但下载失败",
    }


if __name__ == "__main__":
    print("=== 数字人对口型模块自检（离线，不需要密钥）===")

    assert submit_lipsync("")["success"] is False
    assert submit_lipsync("不存在的模特.mp4", tts_text="你好")["success"] is False
    print("  submit_lipsync 失败路径    OK")

    # 二选一约束
    r = submit_lipsync(__file__, audio_path="a.wav", tts_text="你好")
    assert r["success"] is False and "二选一" in r["message"], r
    r2 = submit_lipsync(__file__)
    assert r2["success"] is False and "必须提供" in r2["message"], r2
    print("  参数互斥校验               OK")

    assert query_task("")["finished"] is False
    print("  query_task 失败路径        OK")

    assert download_result("") == ""
    print("  download_result 失败路径   OK")

    print(f"\n  密钥已配置: {is_configured()} | 模型: {settings.media.avatar_model}")
    print(f"  可用内置音色: {len(PIXVERSE_SPEAKERS)} 个")
    print("全部自检通过")
