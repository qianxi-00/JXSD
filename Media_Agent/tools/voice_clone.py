# -*- coding: utf-8 -*-
"""声音克隆 + 克隆音色语音合成 —— 百炼 CosyVoice 声音复刻（替代课案的本地 Fish-Speech）

课案出处：自媒体课案 → 口播视频 → AutoDL Fish-Speech TTS 启动 / tools/heygem_client.py

课案原本在 AutoDL 上跑 **Fish-Speech 1.5**（RTX 3080 Ti 12G），
通过 ``/v1/preprocess_and_tran``（声音预处理）+ ``/v1/invoke``（克隆音色合成）两个接口调用。
本项目换成百炼 **CosyVoice 声音复刻**，本机不需要 GPU。

接口对应关系
    | 课案 Fish-Speech | 本项目百炼 | 说明 |
    |---|---|---|
    | ``POST {TTS}/v1/preprocess_and_tran`` | ``VoiceEnrollmentService.create_voice(target_model, prefix, url)`` | 输入参考音频，换取一个音色 ID |
    | 返回 ``asr_format_audio_url`` + ``reference_audio_text`` | 返回 ``voice_id`` | 百炼把「预处理 + ASR 对齐」都包在内部了，不用自己传参考文本 |
    | ``POST {TTS}/v1/invoke``（returns 音频二进制） | ``SpeechSynthesizer(model, voice=voice_id).call(text)``（returns bytes） | 合成 |

三个必须知道的约束（都会真实咬人）
    1. ⚠️ **参考音频必须是真正的公网 http(s) URL —— 百炼临时存储的 oss:// 不行！**

       实测（2026-09）：把参考音频走 ``tools/dashscope_upload.upload_file()`` 换成
       ``oss://dashscope-instant/...`` 后调用 ``create_voice``，直接 400：

           Code: InvalidParameter
           Error Message: audio url should start with http or https

       ``oss://`` 这套临时存储是给「多模态/图像/视频」类模型用的（走模型调用时带
       ``X-DashScope-OssResourceResolve: enable`` 解析），**CosyVoice 不收**。
       所以声音克隆这一环必须有自己的公网托管，可选：
         · 阿里云 OSS（同账号最顺，需开 OSS 并配 AK/SK）
         · 你自己的公网服务器（放一个静态目录 + HTTP 服务即可）
         · 内网穿透（Cloudflare Tunnel / ngrok 之类）
       没配托管时的行为：``clone_voice()`` 返回失败，``workflows/video.py`` 会自动
       降级到 edge-tts 通用音色 / PixVerse 内置 TTS —— **数字人功能不受影响**，
       只是用不上「克隆你自己的声音」。

    2. **音色有配额**：官方原话「避免频繁调用。每次调用都会创建新音色，
       达到配额上限后将无法创建。」—— 所以本项目**强制走缓存**：
       同一个源音频（按 路径+大小+修改时间 做指纹）只创建一次音色，
       voice_id 落在 ``MEDIA_AGENT_DIR/.cache/voices.json``。

    3. **``target_model`` 必须与合成时用的模型一致**，否则合成会失败。
       两边都取 ``settings.media.tts_model``，不要在调用处硬编码。

    4. ⚠️ **dashscope SDK 不读我们传的密钥**（实测踩到）：
       ``SpeechSynthesizer`` 的构造签名里**没有 api_key 参数**，它只认全局
       ``dashscope.api_key`` 或环境变量；而本项目密钥在根 ``.env``，由
       pydantic-settings 加载、**没进进程环境**。不显式绑定就会
       ``InputRequired: apikey is required!``。见 ``_bind_dashscope_key()``。
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def is_configured() -> bool:
    """百炼密钥是否已配置（声音复刻走同一个密钥）。"""
    return bool(settings.dashscope_api_key)


def _bind_dashscope_key():
    """把密钥绑到 dashscope SDK 的**全局** ``api_key`` 上。

    ⚠️ 实测踩过的坑（2026-09，dashscope 1.27.4）：

        ``SpeechSynthesizer(...)`` 的构造签名里**没有 api_key 参数**，
        它从全局 ``dashscope.api_key`` 或环境变量 ``DASHSCOPE_API_KEY`` 取凭据。
        而本项目的密钥来自根 ``.env``，由 pydantic-settings 加载进 ``Settings``，
        **并没有写进进程环境变量** —— 所以不显式设置全局值就会直接
        ``InputRequired: apikey is required!`` / ``AuthenticationError``。

        ``VoiceEnrollmentService`` 倒是接受 ``api_key=`` 参数，但它和
        ``SpeechSynthesizer`` 共用底层客户端，统一走这里更稳。

    这是库的设计限制，不是可以绕开的写法；只在真正要调用前设置，避免在 import
    阶段就产生副作用。
    """
    import dashscope

    if settings.dashscope_api_key:
        dashscope.api_key = settings.dashscope_api_key
        if settings.dashscope_workspace_id:
            dashscope.workspace = settings.dashscope_workspace_id
    return dashscope.api_key


# --------------------------------------------------------------------------
# 音色缓存：音色有配额，同一个源音频绝不能重复创建
# --------------------------------------------------------------------------
def _cache_path() -> Path:
    p = Path(settings.media.voice_cache_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache() -> dict:
    try:
        p = _cache_path()
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 —— 缓存坏了就当没有，不影响主流程
        print(f"[声音克隆] 音色缓存读取失败（忽略）: {exc}")
    return {}


def _save_cache(cache: dict) -> None:
    try:
        _cache_path().write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[声音克隆] 音色缓存写入失败（忽略）: {exc}")


def _fingerprint(source_audio: str) -> str:
    """源音频指纹：绝对路径 + 大小 + 修改时间。文件变了就重新克隆。"""
    p = Path(source_audio).resolve()
    st = p.stat()
    return f"{p}|{st.st_size}|{int(st.st_mtime)}"


# --------------------------------------------------------------------------
# 参考音频准备
# --------------------------------------------------------------------------
def _to_wav_16k(source: str) -> str:
    """把任意音频/视频转成 16kHz 单声道 wav —— 声音复刻的参考音频要求。

    已经是 16k 单声道 wav 且文件较小时直接复用，省一次转码。
    """
    src = Path(source)
    if src.suffix.lower() == ".wav":
        try:
            # 用 ffprobe 探一下，确认已经是 16k 单声道
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", "-select_streams", "a:0", str(src)],
                capture_output=True, timeout=30, text=True,
            )
            if probe.returncode == 0:
                info = json.loads(probe.stdout or "{}")
                streams = info.get("streams") or []
                if streams:
                    s = streams[0]
                    if int(s.get("sample_rate", 0)) == 16000 and int(s.get("channels", 0)) == 1:
                        print(f"[声音克隆] 参考音频已是 16k 单声道 wav，直接复用")
                        return str(src)
        except Exception:  # noqa: BLE001 —— 探测失败就老实转码
            pass

    out = Path(settings.media.get_video_output_dir()) / f"voice_ref_{src.stem}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(out)],
            check=True, capture_output=True, timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[声音克隆] 参考音频转码失败: {exc}")
        return ""

    if out.is_file() and out.stat().st_size > 1024:
        print(f"[声音克隆] 参考音频已转码: {out}")
        return str(out)
    print("[声音克隆] 转码输出为空或过小")
    return ""


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------
def clone_voice(source_media: str, prefix: str = "mediaclone", use_cache: bool = True,
                lang: str = "zh") -> dict:
    """从一段音频/视频里克隆音色，返回 ``voice_id``。

    对应课案的 ``heygem_voice_clone()``。

    Args:
        source_media: 模特视频或音频的本地路径（课案传的是模特视频）。
        prefix: 音色名前缀，只允许数字和英文字母，≤10 字符。
        use_cache: 是否复用缓存（默认 True，强烈建议保持 True —— 音色有配额）。
        lang: 参考音频的语言提示，课案支持 ``zh`` / ``en`` / ``ja`` / ``ko``
            （``heygem_voice_clone(source_video, lang="zh")``）。不传时与旧行为
            完全一致 —— 以前这里把 ``["zh"]`` 写死了，英文/日文素材没法声明。

    Returns:
        ``{"success": bool, "voice_id": str, "from_cache": bool, "message": str}``
        —— **绝不抛异常**。
    """
    result = {"success": False, "voice_id": "", "from_cache": False, "message": ""}

    if not source_media or not Path(source_media).is_file():
        result["message"] = f"参考文件不存在: {source_media}"
        print(f"[声音克隆] {result['message']}")
        return result

    if not is_configured():
        result["message"] = (
            "未配置 DASHSCOPE_API_KEY（见根目录 .env），声音克隆不可用。"
            "口播视频会自动降级为 edge-tts 通用音色。"
        )
        print(f"[声音克隆] {result['message']}")
        return result

    # ---- 查缓存 ----
    key = _fingerprint(source_media)
    cache = _load_cache() if use_cache else {}
    cached = cache.get(key)
    if cached and cached.get("voice_id"):
        result.update(
            success=True, voice_id=cached["voice_id"], from_cache=True,
            message=f"命中音色缓存（创建于 {cached.get('created_at', '?')}）",
        )
        print(f"[声音克隆] {result['message']}: {cached['voice_id']}")
        return result

    # ---- 准备参考音频 ----
    ref_wav = _to_wav_16k(source_media)
    if not ref_wav:
        result["message"] = "参考音频准备失败（ffmpeg 转码失败）"
        return result

    # ---- 上传到公网可访问位置换 http(s) URL ----
    #
    # ⚠️ 实测：**不能**用百炼临时存储。走 tools/dashscope_upload 拿到的是
    #    ``oss://...``，create_voice 会直接 400：
    #        Code: InvalidParameter
    #        Error Message: audio url should start with http or https
    #    ``oss://`` 那套是给多模态/图像/视频模型用的（调用时靠
    #    X-DashScope-OssResourceResolve 头解析），CosyVoice 不收。
    ref_url, host_err = _publish_reference(ref_wav)
    if not ref_url:
        result["message"] = host_err
        print(f"[声音克隆] {result['message']}")
        return result

    # ---- 创建音色 ----
    try:
        from dashscope.audio.tts_v2 import VoiceEnrollmentService

        _bind_dashscope_key()
        service = VoiceEnrollmentService(
            api_key=settings.dashscope_api_key,
            workspace=settings.dashscope_workspace_id or None,
        )
        voice_id = service.create_voice(
            target_model=settings.media.tts_model,
            prefix=prefix,
            url=ref_url,
            # language_hints 是 create_voice 的正式形参，不是靠 **kwargs 蒙进去的
            # （dashscope 1.27.4，inspect.signature 核实）
            language_hints=[lang],
        )
    except Exception as exc:  # noqa: BLE001 —— 配额/网络/参数错都收成中文提示
        result["message"] = (
            f"创建音色失败: {exc}\n"
            "常见原因：音色配额已满（每次调用都会新建音色，注意复用）、"
            "参考音频不合格（需人声清晰、无背景音乐）、或地域不在华北2（北京）。"
        )
        print(f"[声音克隆] {result['message']}")
        return result

    if not voice_id:
        result["message"] = "创建音色返回空 ID"
        print(f"[声音克隆] {result['message']}")
        return result

    result.update(success=True, voice_id=str(voice_id), message="音色创建成功")
    print(f"[声音克隆] OK voice_id={voice_id}")

    if use_cache:
        from datetime import datetime

        cache[key] = {
            "voice_id": str(voice_id),
            "source": str(Path(source_media).resolve()),
            "target_model": settings.media.tts_model,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_cache(cache)
    return result


def _publish_reference(ref_wav: str) -> tuple:
    """把参考音频放到「真正的公网 http(s)」上，返回 ``(url, error)``。

    ⚠️ 为什么不复用 ``tools/dashscope_upload``：

        百炼临时存储返回的是 ``oss://`` 前缀，而 ``create_voice`` 明确拒绝它
        （实测 400：``audio url should start with http or https``）。
        那套临时存储是给多模态 / 图像 / 视频模型用的，TTS 不收。

    本项目的做法（按用户指定）：**用自己的公网服务器做静态托管**，
    上传逻辑封装在 ``tools/asset_host.py`` 里（scp + nginx 只读分发）。

    优先级：
        ① ``MEDIA_VOICE_REF_URL`` 已预置 → 直接用（适合"参考音频固定不变"的场景）；
        ② ``MEDIA_ASSET_*`` 三项配齐 → 现传现用（``asset_host.publish``）；
        ③ 都没有 → 明确失败，让上层降级到 edge-tts / PixVerse 内置 TTS。
    """
    ref_wav_path = Path(ref_wav)

    # ① 预置的固定参考音频 URL
    preset = (getattr(settings.media, "voice_ref_url", "") or "").strip()
    if preset:
        if not preset.startswith(("http://", "https://")):
            return "", f"MEDIA_VOICE_REF_URL 必须以 http/https 开头，当前是 {preset!r}"
        print(f"[声音克隆] 使用 .env 里预置的参考音频 URL: {preset[:80]}")
        return preset, ""

    # ② 现传现用：走自己的公网素材托管
    try:
        from tools.asset_host import is_configured as host_ready
        from tools.asset_host import publish

        if host_ready():
            url = publish(str(ref_wav_path))
            if url:
                return url, ""
            return "", "参考音频上传到公网素材托管失败（见上方 [托管] 日志）"
    except Exception as exc:  # noqa: BLE001 —— 托管模块出问题也要给出可读原因
        print(f"[声音克隆] 调用素材托管失败: {exc}")

    # ③ 都没配
    return "", (
        "声音克隆需要一个**真正的公网 http(s) 参考音频 URL**，"
        "百炼临时存储的 oss:// 在 create_voice 上会被拒（实测 400："
        "audio url should start with http or https）。\n"
        "二选一：\n"
        "  ① 配好公网素材托管（推荐，项目已内置 tools/asset_host.py）：\n"
        "       MEDIA_ASSET_SSH=ubuntu@<你的服务器>\n"
        "       MEDIA_ASSET_REMOTE_DIR=/var/www/media-assets\n"
        "       MEDIA_ASSET_BASE_URL=http://<你的服务器>/media-assets\n"
        "  ② 或者手工把一个参考音频传上去，把 URL 填进 MEDIA_VOICE_REF_URL。\n"
        "（不配也不影响数字人：会自动降级到 edge-tts 通用音色 / PixVerse 内置 TTS）"
    )


def tts_with_cloned_voice(text: str, voice_id: str, output_path: str = None,
                          audio_format: str = "wav") -> str:
    """用克隆出的音色合成语音。

    对应课案的 ``heygem_tts_with_cloned_voice()``。

    Args:
        text: 要合成的文本。
        voice_id: ``clone_voice()`` 返回的音色 ID。
        output_path: 输出路径；默认 ``<视频输出目录>/voice_cloned_<hash>.wav``。
        audio_format: ``"wav"`` 或 ``"mp3"``。

    Returns:
        音频文件路径；失败返回空字符串。
    """
    if not text or not text.strip():
        print("[克隆TTS] 文本为空")
        return ""
    if not voice_id:
        print("[克隆TTS] 音色 ID 为空")
        return ""
    if not is_configured():
        print("[克隆TTS] 未配置 DASHSCOPE_API_KEY")
        return ""

    suffix = ".mp3" if audio_format == "mp3" else ".wav"
    if output_path is None:
        # 用 md5 而非内置 hash()：hash() 每进程随机加盐，同一文本换进程会得到不同文件名
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        output_path = os.path.join(
            settings.media.get_video_output_dir(),
            f"voice_cloned_{digest}{suffix}",
        )

    try:
        from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

        _bind_dashscope_key()   # SpeechSynthesizer 不接 api_key，必须靠全局值
        fmt = (
            AudioFormat.MP3_24000HZ_MONO_256KBPS
            if audio_format == "mp3"
            else AudioFormat.WAV_24000HZ_MONO_16BIT
        )
        synthesizer = SpeechSynthesizer(
            model=settings.media.tts_model,
            voice=voice_id,
            format=fmt,
            workspace=settings.dashscope_workspace_id or None,
        )
        # call() 不设 callback 时，阻塞到收完音频并返回完整 bytes
        audio: bytes = synthesizer.call(text)
    except Exception as exc:  # noqa: BLE001
        print(f"[克隆TTS] 合成失败: {exc}")
        return ""

    if not audio:
        print("[克隆TTS] 返回音频为空")
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Path(output_path).write_bytes(audio)
    if Path(output_path).stat().st_size > 1024:
        print(f"[克隆TTS] OK {output_path}（{len(audio) / 1024:.1f} KB）")
        return output_path
    print("[克隆TTS] 输出文件过小，判定失败")
    return ""


def list_cloned_voices() -> list:
    """列出本地缓存的音色（给页面做「选用已有音色」用）。"""
    cache = _load_cache()
    items = []
    for k, v in cache.items():
        items.append({
            "voice_id": v.get("voice_id", ""),
            "source": v.get("source", ""),
            "target_model": v.get("target_model", ""),
            "created_at": v.get("created_at", ""),
        })
    return items


def forget_voice(source_media: str) -> bool:
    """删掉某个源素材在本地的音色缓存记录。

    什么时候要用：**音色在云端被删掉之后**（配额释放、控制台手工清理、
    或跑完测试做清理）。不然缓存里留着一条指向失效 voice_id 的记录，
    下次同样的素材会命中它，然后在合成阶段失败 —— 而且报错点离真正的原因很远。

    Returns:
        是否删掉了一条记录。
    """
    if not source_media or not Path(source_media).is_file():
        return False
    cache = _load_cache()
    key = _fingerprint(source_media)
    if key not in cache:
        return False
    removed = cache.pop(key)
    _save_cache(cache)
    print(f"[声音克隆] 已清除本地缓存: {removed.get('voice_id')}")
    return True


if __name__ == "__main__":
    print("=== 声音克隆模块自检（离线，不需要密钥）===")

    # 1) 缓存读写往返
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "a.wav"
        fake.write_bytes(b"x" * 2048)
        fp1 = _fingerprint(str(fake))
        fp2 = _fingerprint(str(fake))
        assert fp1 == fp2, "同一文件指纹应稳定"
        fake.write_bytes(b"y" * 4096)          # 改内容（大小变了）
        assert _fingerprint(str(fake)) != fp1, "文件变化后指纹应改变"
        print("  _fingerprint              OK")

    # 2) 失败路径必须返回结构化 dict，不抛异常
    r = clone_voice("")
    assert r["success"] is False and r["message"], r
    r2 = clone_voice("不存在的模特视频.mp4")
    assert r2["success"] is False and "不存在" in r2["message"], r2
    print("  clone_voice 失败路径       OK")

    assert tts_with_cloned_voice("", "v1") == ""
    assert tts_with_cloned_voice("你好", "") == ""
    print("  tts_with_cloned_voice      OK")

    # 3) 缓存文件结构可用
    assert isinstance(list_cloned_voices(), list)
    print(f"  list_cloned_voices        OK  当前缓存 {len(list_cloned_voices())} 条")

    print(f"\n  密钥已配置: {is_configured()} | TTS 模型: {settings.media.tts_model}")
    print("全部自检通过")
