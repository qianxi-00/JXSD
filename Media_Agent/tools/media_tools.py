# -*- coding: utf-8 -*-
"""媒体处理工具 —— 视频下载 / 音频提取 / 语音识别 / 语音合成 / 图片生成 / 文章抓取

课案出处：自媒体课案 → 内容复刻（工具函数）

与课案的落地差异（重要）
    | 课案 | 本项目 | 为什么 |
    |---|---|---|
    | 视频下载三级降级 videodl → yt-dlp → stub | 只保留 yt-dlp | 课案自己在 FAQ 里记了 videodl 的 quickjs 编译坑；yt-dlp 已能覆盖抖音/B站/小红书，少一个重依赖 |
    | 语音识别走本地 FunASR | 走 ``tools/audio_transcriber.py``（百炼 Qwen-Audio-3.0-ASR-Flash） | 本项目不部署本地模型 |
    | 抓取失败返回「演示文案」 | 返回空串 + 中文提示 | 拿假文案冒充真实文章，会让下游 LLM 一本正经地分析虚构内容 —— 降级要降得诚实 |
    | ``extract_audio_text`` 里两段完全相同的 URL 判断 | 去重 | 课案原样复制粘贴的冗余 |

依旧保留课案的设计
    · **所有函数绝不抛异常**：失败返回空串 / 空列表。
    · **自动降级**：依赖不可用时打印中文提示并返回可用占位。
"""

import asyncio
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ==========================================================================
# URL 提取
# ==========================================================================
def extract_url(text: str) -> str:
    """从混合文本里提取第一个 URL（处理小红书 / 抖音的分享口令文本）。

    小红书分享格式示例::

        "57 【标题】 😆 ID 😆 https://www.xiaohongshu.com/..."

    抖音分享格式示例::

        "7.65 复制打开抖音，看看【某某的作品】https://v.douyin.com/xxxx/ ..."

    Args:
        text: 用户粘贴的原始文本。

    Returns:
        提取到的第一个 http(s) URL；没找到则原样返回输入。
    """
    if not text:
        return text

    text = text.strip()
    if text.startswith(("http://", "https://")):
        return text

    # 中文标点也算 URL 的终止符，否则会把后面的说明文字一起吞进来
    matches = re.findall(r"https?://[^\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+", text)
    if matches:
        return matches[0].rstrip(".,;:!?）)】』\"'")
    return text


# ==========================================================================
# 视频下载
# ==========================================================================
def _yt_dlp_available() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def download_video(url: str, output_dir: str = None) -> str:
    """下载视频到本地（yt-dlp）。

    Args:
        url: 视频地址，允许是混合文本（内部先走 ``extract_url``）。
        output_dir: 输出目录，默认 ``settings.media.get_video_output_dir()/downloads``。

    Returns:
        下载后的文件路径；失败返回空字符串（**不抛异常**）。
    """
    url = extract_url(url)
    if not url or not url.startswith(("http://", "https://")):
        print(f"[下载] 不是有效链接: {str(url)[:80]}")
        return ""

    if output_dir is None:
        output_dir = os.path.join(settings.media.get_video_output_dir(), "downloads")
    os.makedirs(output_dir, exist_ok=True)

    if not _yt_dlp_available():
        print("[下载] 未安装 yt-dlp，跳过。安装：uv add yt-dlp")
        return ""

    try:
        import yt_dlp

        # 用 Python API 而不是 subprocess：不依赖 PATH 里的 yt-dlp 可执行文件，
        # 也避免 Windows 上 ffmpeg 合并音视频时的管道/编码坑。
        opts = {
            "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            # 只下已合并/单文件流，避免额外触发 ffmpeg 合并
            "format": "bv*+ba/b",
        }
        print(f"[下载] yt-dlp 解析: {url[:80]}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                print("[下载] 解析结果为空")
                return ""

            # 播放列表/多条目时取第一个
            if "entries" in info:
                entries = [e for e in (info.get("entries") or []) if e]
                if not entries:
                    print("[下载] 列表为空")
                    return ""
                info = entries[0]

            path = ydl.prepare_filename(info)
            if not os.path.isfile(path):
                # 容器被转封装过时扩展名会变，按同目录同名文件兜底找
                stem = os.path.splitext(path)[0]
                for cand in Path(output_dir).glob(os.path.basename(stem) + ".*"):
                    if cand.is_file() and cand.stat().st_size > 1024:
                        path = str(cand)
                        break

        if os.path.isfile(path) and os.path.getsize(path) > 1024:
            print(f"[下载] OK {path}")
            return path

        print(f"[下载] 未找到有效输出文件: {path}")
        return ""
    except Exception as exc:  # noqa: BLE001 —— 下载失败是常态（反爬/失效链接）
        print(f"[下载] 失败: {exc}")
        return ""


# ==========================================================================
# 音频提取 / 语音识别
# ==========================================================================
def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def video_to_audio(
    input_video: str,
    output_audio: str = None,
    bitrate: str = "192k",
    sample_rate: int = 44100,
) -> str:
    """用 FFmpeg 从视频里抽音频。

    Args:
        input_video: 输入视频路径。
        output_audio: 输出 mp3 路径；默认与视频同目录同名 ``.mp3``。
        bitrate: 音频码率。
        sample_rate: 采样率（课案里这个参数名叫 ``fps``，其实传的是 ``-ar``，已正名）。

    Returns:
        音频文件路径；失败返回空串。
    """
    if not _ffmpeg_available():
        print("[音频提取] 未找到 ffmpeg。Windows: winget install ffmpeg")
        return ""
    if not input_video or not os.path.isfile(input_video):
        print(f"[音频提取] 视频不存在: {input_video}")
        return ""

    if output_audio is None:
        output_audio = os.path.splitext(input_video)[0] + "_audio.mp3"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_video,
                "-vn",                    # 丢掉视频流
                "-b:a", bitrate,
                "-ar", str(sample_rate),
                "-ac", "1",               # 单声道：语音识别不需要立体声
                output_audio,
            ],
            check=True, capture_output=True, timeout=300,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[音频提取] 失败: {exc}")
        return ""

    if os.path.exists(output_audio) and os.path.getsize(output_audio) > 0:
        print(f"[音频提取] OK {output_audio}")
        return output_audio
    print("[音频提取] 输出文件为空")
    return ""


def _transcribe_text(audio_path: str) -> str:
    """调语音识别取文本；识别失败时把 ``error`` 里的中文原因打印出来。

    ``transcribe()`` 失败时 ``text`` 是空串、真因在 ``error`` 里（未配置
    ``DASHSCOPE_API_KEY`` / 文件不存在 / 接口报错…）。以前只取 ``["text"]``，
    于是链路断掉时页面只会说「视频与文章两条路都没拿到文案」，真因只进控制台。
    **返回 str 的契约不变**，只多一行日志。
    """
    from tools.audio_transcriber import transcribe

    res = transcribe(audio_path, want_timestamps=False)
    if res["error"]:
        print(f"[提文案] 语音识别失败: {res['error']}")
    return res["text"]


def extract_audio_text(video_path: str) -> str:
    """视频 → 文本（完整流水线）：FFmpeg 抽音频 → 百炼识别。

    课案这里是「FFmpeg + 本地 FunASR」，本项目换成百炼 Qwen-Audio-3.0-ASR-Flash
    （Fun-ASR 家族的云托管版，模型名由 ``MEDIA_ASR_MODEL`` 决定）。

    Args:
        video_path: 本地视频/音频路径，或已是公网 URL。

    Returns:
        转写文本；失败返回空串（失败原因见 ``_transcribe_text()`` 打印的日志）。
    """
    if not video_path:
        print("[提文案] 路径为空（上游下载可能失败了）")
        return ""

    # 已经是 URL：直接交给识别服务，省掉本地下载
    if video_path.startswith(("http://", "https://", "oss://")):
        return _transcribe_text(video_path)

    if not os.path.exists(video_path):
        print(f"[提文案] 文件不存在: {video_path}")
        return ""

    # 已经是音频：直接识别
    if Path(video_path).suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
        return _transcribe_text(video_path)

    audio_path = video_to_audio(video_path)
    if audio_path:
        text = _transcribe_text(audio_path)
        if text:
            return text

    # 兜底：跳过抽音频，让识别服务直接吃视频文件
    print("[提文案] 抽音频失败或识别为空，改让识别服务直接处理视频文件")
    return _transcribe_text(video_path)


# ==========================================================================
# 语音合成（edge-tts，免费无需密钥）
# ==========================================================================
def _edge_tts_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


async def _edge_tts_async(text: str, output_path: str, voice: str) -> bool:
    import edge_tts

    if len(text) > 3000:      # 单次请求太长会被服务端拒
        text = text[:3000]
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)
    return os.path.exists(output_path)


def generate_tts(text: str, output_path: str = None, voice: str = None) -> str:
    """用 edge-tts 合成配音（免费、无需密钥，但无法克隆音色）。

    课案把它作为「克隆声音失败」时的降级分支，本项目沿用这一角色。

    Args:
        text: 要合成的文本。
        output_path: 输出 mp3 路径。
        voice: edge-tts 音色名，默认 ``MEDIA_TTS_FALLBACK_VOICE``。

    Returns:
        音频文件路径；失败返回空串。
    """
    if not settings.media.tts_enabled:
        print("[TTS] 已通过 MEDIA_TTS_ENABLED=false 禁用")
        return ""
    if not text or not text.strip():
        print("[TTS] 文本为空")
        return ""
    if not _edge_tts_available():
        print("[TTS] 未安装 edge-tts。安装：uv add edge-tts")
        return ""

    voice = voice or settings.media.tts_fallback_voice
    if output_path is None:
        # 用 md5 而不是内置 hash()：**字符串 hash 带进程级随机盐**，
        # 同一段文案每次新进程都会算出不同的文件名 → 缓存只堆积不复用。
        output_path = os.path.join(
            settings.media.get_video_output_dir(),
            f"tts_{hashlib.md5(text.encode('utf-8')).hexdigest()[:8]}.mp3",
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # 每次新建事件循环：Streamlit 在主线程里可能已经有 loop 了，用 asyncio.run 会炸
        loop = asyncio.new_event_loop()
        try:
            ok = loop.run_until_complete(_edge_tts_async(text, output_path, voice))
        finally:
            loop.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[TTS] 失败: {exc}")
        return ""

    if ok and os.path.getsize(output_path) > 0:
        print(f"[TTS] OK {output_path}（音色 {voice}）")
        return output_path
    print("[TTS] 生成失败或文件为空")
    return ""


# ==========================================================================
# 图片生成（OpenAI 兼容接口，可选）
# ==========================================================================
def _image_configured() -> bool:
    return bool(settings.media.image_api_key and settings.media.image_model)


def generate_image(prompt: str, output_path: str = None, size: str = "1024x1024") -> str:
    """调 OpenAI 兼容的图片生成接口并保存到本地。

    未配置 ``MEDIA_IMAGE_*`` 时降级为本地占位图（**不返回假图片 URL**）。

    Returns:
        保存后的图片路径；失败返回空串。
    """
    if output_path is None:
        # 同上：md5 保证跨进程稳定，同名提示词复用同一张图
        output_path = os.path.join(
            settings.media.get_image_output_dir(),
            f"img_{hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]}.png",
        )

    if not _image_configured():
        print("[图片生成] 未配置 MEDIA_IMAGE_API_KEY / MEDIA_IMAGE_MODEL，降级为占位图")
        return _stub_generate_image(prompt, output_path)

    try:
        import httpx
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.media.image_api_key,
            base_url=settings.media.image_base_url or None,
        )
        print(f"[图片生成] {settings.media.image_model}: {prompt[:60]}...")
        response = client.images.generate(
            model=settings.media.image_model, prompt=prompt, n=1, size=size
        )
        datum = response.data[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = getattr(datum, "url", None)
        b64 = getattr(datum, "b64_json", None)

        if url:
            data = httpx.get(url, timeout=60, follow_redirects=True).content
        elif b64:
            import base64

            data = base64.b64decode(b64)
        else:
            print("[图片生成] 响应里既没有 url 也没有 b64_json")
            return _stub_generate_image(prompt, output_path)

        Path(output_path).write_bytes(data)
        print(f"[图片生成] OK {output_path}")
        return output_path
    except Exception as exc:  # noqa: BLE001
        print(f"[图片生成] 失败: {exc}，降级为占位图")
        return _stub_generate_image(prompt, output_path)


def _stub_generate_image(prompt: str, output_path: str) -> str:
    """生成一张纯色占位图，把提示词写在中间 —— 让链路能跑完，但不冒充真图。"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.new("RGB", (1080, 1920), color=(20, 20, 30))
        draw = ImageDraw.Draw(img)
        for y in range(0, 1920, 4):     # 竖向渐变，观感比纯色好一点
            t = y / 1920
            draw.line([(0, y), (1080, y)], fill=(int(20 + t * 30), int(20 + t * 25), int(30 + t * 50)))

        text = prompt[:40] + "..." if len(prompt) > 40 else prompt
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 36)
        except Exception:  # noqa: BLE001 —— 字体缺失时用默认字体，不中断
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(
            ((1080 - (bbox[2] - bbox[0])) / 2, (1920 - (bbox[3] - bbox[1])) / 2),
            text, fill=(200, 200, 220), font=font,
        )
        img.save(output_path)
        print(f"[图片生成-占位图] {output_path}")
        return output_path
    except Exception as exc:  # noqa: BLE001
        print(f"[图片生成-占位图] 失败: {exc}")
        return ""


# ==========================================================================
# 文章抓取
# ==========================================================================
def fetch_article(url: str) -> str:
    """抓取网页正文（trafilatura）。

    课案在失败时返回一段「演示文案」，本项目改为返回空串 ——
    否则下游 LLM 会把虚构的示例文案当成真实文章来分析。

    Returns:
        提取到的正文；失败返回空串。
    """
    if not url or not url.startswith(("http://", "https://")):
        print(f"[文章抓取] 不是有效链接: {str(url)[:80]}")
        return ""

    try:
        import trafilatura
    except ImportError:
        print("[文章抓取] 未安装 trafilatura。安装：uv add trafilatura")
        return ""

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            print("[文章抓取] 页面下载失败（可能被反爬，或链接需要登录）")
            return ""
        text = trafilatura.extract(downloaded)
        if text and len(text) > 50:
            print(f"[文章抓取] OK 抓到 {len(text)} 字")
            return text
        print("[文章抓取] 正文过短，判定为抓取失败")
        return ""
    except Exception as exc:  # noqa: BLE001
        print(f"[文章抓取] 失败: {exc}")
        return ""


if __name__ == "__main__":
    print("=== media_tools 自检（纯逻辑，不联网）===")

    assert extract_url("https://a.com/x") == "https://a.com/x"
    assert extract_url("57 【标题】 😆 https://www.xiaohongshu.com/abc?x=1 复制打开") == \
        "https://www.xiaohongshu.com/abc?x=1"
    assert extract_url("没有链接的一段话") == "没有链接的一段话"
    assert extract_url("") == ""
    print("  extract_url               OK")

    assert download_video("") == ""
    assert download_video("不是链接") == ""
    print("  download_video 失败路径    OK")

    assert video_to_audio("") == ""
    assert video_to_audio("不存在.mp4") == ""
    print("  video_to_audio 失败路径    OK")

    assert fetch_article("") == ""
    print("  fetch_article 失败路径     OK")

    print(f"  ffmpeg 可用: {_ffmpeg_available()} | yt-dlp: {_yt_dlp_available()} | "
          f"edge-tts: {_edge_tts_available()} | 图片生成已配置: {_image_configured()}")
    print("\n全部自检通过")
