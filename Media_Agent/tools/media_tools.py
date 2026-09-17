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

三条实测踩出来的坑（细节写在各自函数处，这里给第一次读的人一个索引）
    1. **产物文件名一律用 ``hashlib.md5(...).hexdigest()[:8]``，不能退回内置 ``hash()``。**

       Python 的字符串 ``hash()`` 带**进程级随机盐**（``PYTHONHASHSEED`` 默认随机），
       同一段文案换个进程算出来的数字就不同 —— 「同内容 → 同文件名」的复用语义
       直接失效，输出目录只会静静堆垃圾。见 ``generate_tts()`` / ``generate_image()``。

    2. ``asyncio.run()`` **在 Streamlit 里会炸**：页面主线程可能已经有事件循环，
       报 ``RuntimeError: asyncio.run() cannot be called from a running event loop``。
       所以 ``generate_tts()`` 自己 ``new_event_loop()`` 再 ``run_until_complete()``。

    3. **ASR 失败时 ``text`` 是空串、真因在 ``error`` 字段里**：只取 ``text`` 会让链路
       断掉时页面只说「两条路都没拿到文案」，真因只进控制台 —— 见 ``_transcribe_text()``。

文件末尾的 ``__main__`` 是**离线打桩**自检：只跑纯逻辑与失败路径，不联网、不调付费接口。
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

    # 中文标点也算 URL 的终止符，否则会把后面的说明文字一起吞进来。
    # 排除的三个码点区间：汉字（\u4e00-\u9fff）、CJK 标点（\u3000-\u303f）、
    # 全角形式（\uff00-\uffef）—— 分享口令里的「，」「【」「】」都落在这些区间里。
    matches = re.findall(r"https?://[^\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+", text)
    if matches:
        # 半角标点可能紧贴着 URL（".../abc." 这种），剥掉尾部的收尾符再返回，
        # 否则它会变成 URL 的一部分被原样请求出去。只取第一个匹配：一条输入只处理一条链接。
        return matches[0].rstrip(".,;:!?）)】』\"'")
    return text


# ==========================================================================
# 视频下载
# ==========================================================================
def _yt_dlp_available() -> bool:
    """探测 yt-dlp 装了没有，决定 ``download_video()`` 走真实下载还是直接降级。

    Returns:
        bool：能 import 到 ``yt_dlp`` 为 True。**只探测、不把它引进模块命名空间**
        （``# noqa: F401`` 就是为这个补的）；探测失败是设计内的预期路径。
    """
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        # yt-dlp 是可选依赖（课案还叠了一层 videodl，按上面的差异表本项目已去掉）。
        # 不让 ImportError 冒出去：调用方拿到 False 后会「打印中文提示 + 返回空串」，
        # 页面照常能开。
        return False


def download_video(url: str, output_dir: str = None) -> str:
    """下载视频到本地（yt-dlp）。

    Args:
        url: 视频地址，允许是混合文本（内部先走 ``extract_url``）。
        output_dir: 输出目录，默认 ``settings.media.get_video_output_dir()/downloads``。

    Returns:
        下载后的文件路径；失败返回空字符串（**不抛异常**）。

    注意：
        返回值是空串就是「这条路没成」的**唯一**信号 —— ``workflows/replicate.py``
        靠它决定改走 ``fetch_article()`` 抓正文，所以任何失败都必须是空串而不是异常。
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
            # 文件名按「视频 ID + 真实扩展名」落盘；ID 由站点决定且稳定，
            # 重复下载同一链接会直接覆盖同一份产物，不会越下越多。
            "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
            # 分享链接后面可能挂着合集/播放列表，只下当前这一条（下面还有 entries 兜底）。
            "noplaylist": True,
            # quiet + no_warnings：关掉 yt-dlp 自己的进度条与警告，
            # 免得刷屏盖住本模块的 `[下载] ...` 中文日志（排障全靠这几行）。
            "quiet": True,
            "no_warnings": True,
            # 合并后的容器固定 mp4：下游抽音频 / 上传百炼 / 喂 PixVerse 都按常见容器处理。
            "merge_output_format": "mp4",
            # 选择器 `bv*+ba/b`：优先「最佳视频流 + 最佳音频流」两条分立流，取不到再退回
            # 「单条最佳合并流」。抖音 / B 站这类站点给的就是**分立流**，所以 `bv*+ba`
            # 这一支是常态 —— 它**需要** ffmpeg 参与合并（本机 ffmpeg 在 PATH 里，
            # `_ffmpeg_available()` 为 True），末尾那个 `/b` 才是给老站点留的兜底。
            # ⚠️ 原注释写的是「只下已合并/单文件流，避免额外触发 ffmpeg 合并」，
            #    与选择器语义相反（`bv*+ba` 恰恰是**要**合并的那一支），已按代码更正。
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
                # prepare_filename() 只是**按 outtmpl 推算出来的预期路径**；容器被转封装过
                # （webm → mp4 之类）时扩展名就对不上，预期路径根本不存在。
                # 兜底：按「同目录 + 同主文件名」找一遍真实落盘的文件。
                stem = os.path.splitext(path)[0]
                for cand in Path(output_dir).glob(os.path.basename(stem) + ".*"):
                    # 1 KB 是「这不可能是有效媒体」的经验下限：用来挡住 0 字节、
                    # 或被中断留下的半截空壳，别把空文件当成功结果交给下游。
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
    """探测 PATH 里有没有 ``ffmpeg``（抽音频、yt-dlp 合并音视频都要它）。

    Returns:
        bool：``shutil.which("ffmpeg")`` 命中为 True。**只查 PATH 不试执行**，
        所以它返回 True 也不代表这个 ffmpeg 一定能解码手里的文件。
    """
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

    注意：
        ffmpeg 缺席、源文件不存在、转码报错、产物为空 —— 四种情况统一返回空串，
        由 ``extract_audio_text()`` 决定要不要改让识别服务直接吃视频。
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
        # timeout=300：抽音频是纯本地 CPU 操作（5 分钟的视频通常几秒完成），
        # 给到 5 分钟只为挡住「ffmpeg 卡在损坏文件上不返回」这种挂死。
        # 超时抛的 TimeoutExpired 会被下面的 except 收成中文提示 + 空串，不往上抛。
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

    Args:
        audio_path: 本地音/视频路径，或 ``http(s)`` / ``oss://`` 公网 URL。

    Returns:
        str：转写文本；识别失败返回空串（**不抛异常** —— 这是 ASR 收口的契约，
        上游 ``extract_audio_text()`` 与页面都按「空串 = 没拿到」处理降级）。
    """
    # 函数内 import：只有真要识别时才拉起 ASR 那一套（`tools/audio_transcriber`
    # 自带 `requests` 与百炼配置读取），模块顶层保持只有标准库 + ``config``。
    from tools.audio_transcriber import transcribe

    res = transcribe(audio_path, want_timestamps=False)
    # 关键：`error` 才是失败原因，`text` 这时候一定是空串。
    # 这里只打印、不改返回值 —— 上层靠「空串」判断降级，改抛异常会打断整条复刻链路。
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
        # 上游 `download_video()` 失败时给的就是空串 —— 这里显式拦一下，
        # 免得把 "" 当成路径送进 os.path.exists() 和识别接口。
        print("[提文案] 路径为空（上游下载可能失败了）")
        return ""

    # 已经是 URL：直接交给识别服务，省掉本地下载
    # `oss://` 也算 —— 那是百炼自家的临时存储，服务端认（HTTP 侧靠
    # X-DashScope-OssResourceResolve 头解析，SDK 会自己加）。
    if video_path.startswith(("http://", "https://", "oss://")):
        return _transcribe_text(video_path)

    if not os.path.exists(video_path):
        print(f"[提文案] 文件不存在: {video_path}")
        return ""

    # 已经是音频：直接识别
    # 白名单里是识别接口自己能解的音频容器；mp4 / mov 这类**视频容器不在其中**，
    # 所以会往下走 ffmpeg 抽音频 —— 用扩展名判断比探测真实容器便宜得多。
    if Path(video_path).suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
        return _transcribe_text(video_path)

    audio_path = video_to_audio(video_path)
    if audio_path:
        text = _transcribe_text(audio_path)
        if text:
            return text

    # 兜底：跳过抽音频，让识别服务直接吃视频文件
    # 能走到这里只有两种可能：本机没 ffmpeg，或抽出来的音频识别结果为空。
    # 识别接口本身能解 mp4 视频容器，所以这条路常常还能救回来，救不回来才落到空串。
    print("[提文案] 抽音频失败或识别为空，改让识别服务直接处理视频文件")
    return _transcribe_text(video_path)


# ==========================================================================
# 语音合成（edge-tts，免费无需密钥）
# ==========================================================================
def _edge_tts_available() -> bool:
    """探测 edge-tts 装了没有（它是配音的兜底路径，也是可选依赖）。

    Returns:
        bool：能 import 到 ``edge_tts`` 为 True。
    """
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


async def _edge_tts_async(text: str, output_path: str, voice: str) -> bool:
    """edge-tts 的协程实现：合成并落盘（**自己不兜异常**，由调用方 ``generate_tts()`` 兜）。

    Args:
        text: 要合成的文本，超过 3000 字会被截断。
        output_path: 输出 mp3 路径。
        voice: edge-tts 音色名，形如 ``zh-CN-XiaoxiaoNeural``。

    Returns:
        bool：文件确实落盘了为 True（只判存在，不判大小）。
    """
    import edge_tts

    # 3000 字是上限保护：edge-tts 走 WebSocket，文本过长服务端会直接拒/断流，
    # 截断总比整段合成失败好（课案原样保留的做法）。
    if len(text) > 3000:      # 单次请求太长会被服务端拒
        text = text[:3000]
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)
    # 这里只回答「文件在不在」；「是不是有效音频」由调用方的 >0 字节检查负责。
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

    注意：
        **不抛异常** —— ``workflows/video.py`` 的降级链（克隆音色 → edge-tts →
        PixVerse 内置 TTS）是按「空串 = 这一步没成，换下一档」写的。
    """
    # 总开关：MEDIA_TTS_ENABLED=false 时连 edge-tts 都不走（离线验收 / 省钱用）。
    if not settings.media.tts_enabled:
        print("[TTS] 已通过 MEDIA_TTS_ENABLED=false 禁用")
        return ""
    if not text or not text.strip():
        print("[TTS] 文本为空")
        return ""
    if not _edge_tts_available():
        print("[TTS] 未安装 edge-tts。安装：uv add edge-tts")
        return ""

    # 兜底音色：克隆音色不可用时，用户实际听到的就是这个通用音色
    # （默认 zh-CN-XiaoxiaoNeural，见 MEDIA_TTS_FALLBACK_VOICE）。
    voice = voice or settings.media.tts_fallback_voice
    if output_path is None:
        # 用 md5 而不是内置 hash()：**字符串 hash 带进程级随机盐**，
        # 同一段文案每次新进程都会算出不同的文件名 → 缓存只堆积不复用。
        output_path = os.path.join(
            settings.media.get_video_output_dir(),
            f"tts_{hashlib.md5(text.encode('utf-8')).hexdigest()[:8]}.mp3",
        )
    # 调用方可能传一个还不存在的目录（页面里手填输出路径），先建出来再写。
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # 每次新建事件循环：Streamlit 在主线程里可能已经有 loop 了，用 asyncio.run 会炸
        loop = asyncio.new_event_loop()
        try:
            ok = loop.run_until_complete(_edge_tts_async(text, output_path, voice))
        finally:
            loop.close()
    except Exception as exc:  # noqa: BLE001
        # 断网 / 被墙 / 服务端变更都会炸在这一步。配音是降级链上的一环，
        # 异常绝不能抛给上层 —— 大不了这段视频没配音，也要让口播链路出得来片。
        print(f"[TTS] 失败: {exc}")
        return ""

    # 两道判定都要过：协程自报成功 **且** 文件真有内容
    # （0 字节 mp3 是失败留下的空壳，光看文件名会把它当成功）。
    if ok and os.path.getsize(output_path) > 0:
        print(f"[TTS] OK {output_path}（音色 {voice}）")
        return output_path
    print("[TTS] 生成失败或文件为空")
    return ""


# ==========================================================================
# 图片生成（OpenAI 兼容接口，可选）
# ==========================================================================
def _image_configured() -> bool:
    """图片生成是否配齐（`MEDIA_IMAGE_API_KEY` 与 `MEDIA_IMAGE_MODEL` 两项都要）。

    Returns:
        bool：两项都非空为 True。**只认这两项** —— ``MEDIA_IMAGE_BASE_URL``
        留空是合法的（那就用 OpenAI 官方端点），所以不参与判定。
    """
    return bool(settings.media.image_api_key and settings.media.image_model)


def generate_image(prompt: str, output_path: str = None, size: str = "1024x1024") -> str:
    """调 OpenAI 兼容的图片生成接口并保存到本地。

    未配置 ``MEDIA_IMAGE_*`` 时降级为本地占位图（**不返回假图片 URL**）。

    Args:
        prompt: 提示词；也参与默认文件名的 md5，所以同一段提示词会复用同一张图。
        output_path: 输出 png 路径；默认 ``<图片输出目录>/img_<md5 前 8 位>.png``。
        size: 传给接口的尺寸字符串，默认 ``1024x1024``。

    Returns:
        str：保存后的图片路径；接口不可用或调用失败时返回**占位图路径**，
        连占位图都生成不出来才返回空串。**不抛异常**。
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
        # 函数内 import：图片生成是可选能力，没配齐配置时连这两个包都不该被加载。
        import httpx
        from openai import OpenAI

        # base_url 留空时传 None：让 SDK 用它自己的默认端点（OpenAI 官方）；
        # 换第三方兼容服务（如通义 / 硅基流动）时由 MEDIA_IMAGE_BASE_URL 覆盖。
        client = OpenAI(
            api_key=settings.media.image_api_key,
            base_url=settings.media.image_base_url or None,
        )
        print(f"[图片生成] {settings.media.image_model}: {prompt[:60]}...")
        # n=1：页面一次只要一张素材，多要只是多花钱。
        response = client.images.generate(
            model=settings.media.image_model, prompt=prompt, n=1, size=size
        )
        datum = response.data[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 兼容服务有两种回包形态：给 url（要自己再下一次）或给 b64_json（直接是图）。
        # 不能假定一定有哪一个，所以两个都用 getattr 探测。
        url = getattr(datum, "url", None)
        b64 = getattr(datum, "b64_json", None)

        if url:
            # follow_redirects=True 是必须的：图片常放在 CDN 上，直链基本都会 302。
            # timeout=60 够一张几 MB 的图下完，又不至于在慢链路上挂死。
            data = httpx.get(url, timeout=60, follow_redirects=True).content
        elif b64:
            import base64

            data = base64.b64decode(b64)
        else:
            print("[图片生成] 响应里既没有 url 也没有 b64_json")
            return _stub_generate_image(prompt, output_path)

        # 二进制直写，不过 PIL：少一次解码/编码，也避免格式被改坏。
        Path(output_path).write_bytes(data)
        print(f"[图片生成] OK {output_path}")
        return output_path
    except Exception as exc:  # noqa: BLE001
        # 图片是可选素材：接口挂 / 欠费 / 模型名写错都不该让整条链路失败，
        # 统一退到占位图 —— 链路能跑完，但产物一眼看得出不是真图。
        print(f"[图片生成] 失败: {exc}，降级为占位图")
        return _stub_generate_image(prompt, output_path)


def _stub_generate_image(prompt: str, output_path: str) -> str:
    """生成一张纯色占位图，把提示词写在中间 —— 让链路能跑完，但不冒充真图。

    Args:
        prompt: 提示词，只取前 40 字画在图上（用来标记「这张是给哪句提示占的位」）。
        output_path: 输出 png 路径。

    Returns:
        str：占位图路径；连 PIL 都不可用 / 写不出去时返回空串。

    为什么要有这一层：未配置图片接口时，与其让上游拿到空串去猜，不如给一张
    **一眼看出是占位**的图，页面与剪辑链路都能继续跑。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 1080×1920 = 竖屏 9:16，与短视频成片的常用规格一致，
        # 占位图能直接塞进剪辑的素材位，不用再缩放。
        img = Image.new("RGB", (1080, 1920), color=(20, 20, 30))
        draw = ImageDraw.Draw(img)
        for y in range(0, 1920, 4):     # 竖向渐变，观感比纯色好一点
            t = y / 1920
            draw.line([(0, y), (1080, y)], fill=(int(20 + t * 30), int(20 + t * 25), int(30 + t * 50)))

        # 只画前 40 字：再多就超出画面宽度，糊成一团反而看不清。
        text = prompt[:40] + "..." if len(prompt) > 40 else prompt
        try:
            # 硬编码 Windows 字体路径（本机是 Windows + 微软雅黑）。
            # 换平台或字体缺失时走下面的 load_default()，画面会难看但不会中断。
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
        # 到这一步已经无可再降级了：返回空串，让调用方按「没图」处理。
        print(f"[图片生成-占位图] 失败: {exc}")
        return ""


# ==========================================================================
# 文章抓取
# ==========================================================================
def fetch_article(url: str) -> str:
    """抓取网页正文（trafilatura）。

    课案在失败时返回一段「演示文案」，本项目改为返回空串 ——
    否则下游 LLM 会把虚构的示例文案当成真实文章来分析。

    Args:
        url: 文章链接，必须是 http(s)（分享口令请先用 ``extract_url()`` 洗一遍）。

    Returns:
        str：提取到的正文；失败返回空串（**不抛异常**）。
        ``workflows/replicate.py`` 就是靠「空串」判断要不要走这条降级路的。
    """
    if not url or not url.startswith(("http://", "https://")):
        print(f"[文章抓取] 不是有效链接: {str(url)[:80]}")
        return ""

    # 函数内 import：trafilatura 是可选依赖，缺了只断这一条路，
    # 不该让整个模块 import 失败（页面还要用别的工具函数）。
    try:
        import trafilatura
    except ImportError:
        print("[文章抓取] 未安装 trafilatura。安装：uv add trafilatura")
        return ""

    try:
        # 下载与正文提取分两步，是为了让失败原因可区分：
        # fetch 失败 = 网络/反爬/要登录；extract 失败 = 页面拿到了但正文认不出来。
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            print("[文章抓取] 页面下载失败（可能被反爬，或链接需要登录）")
            return ""
        text = trafilatura.extract(downloaded)
        # 50 字是「这算不算正文」的经验下限：抓到导航栏 / 版权声明这类十几字的残渣时，
        # 宁可判定失败让上游降级，也不要拿垃圾去喂 LLM（课案在这里返回的是演示文案，本项目不干）。
        if text and len(text) > 50:
            print(f"[文章抓取] OK 抓到 {len(text)} 字")
            return text
        print("[文章抓取] 正文过短，判定为抓取失败")
        return ""
    except Exception as exc:  # noqa: BLE001
        print(f"[文章抓取] 失败: {exc}")
        return ""


if __name__ == "__main__":
    # 只跑**纯逻辑 + 失败路径**：真实下载 / 识别 / 合成都要联网或要密钥，自检不碰。
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

    # 这里只报告本机三个可选依赖在不在（不安装、不联网），方便对「为什么走了降级分支」有个第一手结论
    print(f"  ffmpeg 可用: {_ffmpeg_available()} | yt-dlp: {_yt_dlp_available()} | "
          f"edge-tts: {_edge_tts_available()} | 图片生成已配置: {_image_configured()}")
    print("\n全部自检通过")
