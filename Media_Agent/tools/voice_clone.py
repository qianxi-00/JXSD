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

四个必须知道的约束（都会真实咬人）
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

与课案的其它落地差异
    | 项 | 课案（Fish-Speech） | 本项目（CosyVoice） |
    |---|---|---|
    | 跑在哪 | AutoDL 上的 RTX 3080 Ti 12G | 百炼托管 API，本机不需要 GPU |
    | 参考文本 | 要拿 ``reference_audio_text`` 自己对齐 | 百炼内部做 ASR 对齐，不用传 |
    | 音色额度 | 自建服务，随便建 | **有配额，必须走缓存**（见上面第 2 条） |
    | 失败行为 | 本地服务调不通就报错 | ``clone_voice()`` 返回 ``success=False`` + 中文提示，**绝不抛异常** |

失败时的降级链（谁在用这个模块）
    ``workflows/video.py`` 的口播链路是「勾了克隆 → ``clone_voice()`` →
    ``tts_with_cloned_voice()``」，任何一步拿不到结果就退到 ``generate_tts()``
    （edge-tts 通用音色），再不行才退到 PixVerse 内置 TTS。所以本模块的每个
    对外函数都按「失败给空值/结构化失败，不抛异常」写。

文件末尾的 ``__main__`` 是**离线打桩**自检：不需要密钥、不建音色、不合成。
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
    """百炼密钥是否已配置（声音复刻走同一个密钥）。

    Returns:
        bool：``settings.dashscope_api_key`` 非空为 True。**只判有没有配**，
        不校验密钥是否真的有效（那要真调一次才知道，自检不干这事）。
    """
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

        证据（本机安装的包，不用猜）：``SpeechSynthesizer.__init__`` 里就是
        ``self.apikey = dashscope.api_key``（``dashscope/audio/tts_v2/speech_synthesizer.py:495``，
        dashscope 1.27.4）—— 签名里根本没有 api_key 形参可传。

    Returns:
        str：绑定后的 ``dashscope.api_key``（空串表示没配密钥，调用方据此可自行判断）。

    这是库的设计限制，不是可以绕开的写法；只在真正要调用前设置，避免在 import
    阶段就产生副作用。
    """
    # 函数内 import：dashscope 是可选依赖，只在真要调用的路径上加载
    # （模块顶层保持只有标准库 + ``config``，自检就不会因为缺包而挂）。
    import dashscope

    if settings.dashscope_api_key:
        # 写 SDK 的**全局变量** —— 这是 SpeechSynthesizer 唯一认的凭据入口
        # （不接受构造参数、也不看 pydantic 的 settings 对象）。
        dashscope.api_key = settings.dashscope_api_key
        if settings.dashscope_workspace_id:
            # workspace（业务空间 ID）同理只有全局入口。PixVerse 这类只提供
            # 「专属域名」形式的模型必须配上它，CosyVoice 配了也会走专属域名。
            dashscope.workspace = settings.dashscope_workspace_id
    # 回传绑定后的值：调用方可以据此判断「到底绑上了没有」，而不用再读一次 settings。
    return dashscope.api_key


# --------------------------------------------------------------------------
# 音色缓存：音色有配额，同一个源音频绝不能重复创建
# --------------------------------------------------------------------------
def _cache_path() -> Path:
    """音色缓存文件的路径（``MEDIA_VOICE_CACHE_FILE``，默认 ``.cache/voices.json``）。

    Returns:
        Path：文件路径。**会顺手建出父目录** —— 缓存文件是第一条记录写入时才产生的，
        但 ``.cache/`` 本身要提前存在，否则首次写盘会失败（然后被忽略，见 ``_save_cache``）。
    """
    p = Path(settings.media.voice_cache_file)
    # parents=True + exist_ok=True：路径里可能有多层目录，且并发调用时重复建也不算错。
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache() -> dict:
    """读音色缓存。

    Returns:
        dict：``{指纹: {voice_id, source, target_model, created_at}}``；
        文件不存在、内容不是合法 JSON、读不动 —— 一律返回空 dict。
    """
    try:
        p = _cache_path()
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 —— 缓存坏了就当没有，不影响主流程
        # 缓存只是「省配额」的优化，不是数据源：坏掉最多多重一次音色，
        # 绝不能让「缓存文件损坏」把声音克隆整条路挡死。
        print(f"[声音克隆] 音色缓存读取失败（忽略）: {exc}")
    return {}


def _save_cache(cache: dict) -> None:
    """写音色缓存（**整体覆盖**，不是追加）。

    Args:
        cache: 完整的缓存字典。调用方负责先 ``_load_cache()`` 再改这个 dict，
            所以这里直接覆盖写 —— 提交新记录时不会把别人的记录弄丢。

    写失败只打印、不抛异常 —— 又是「落盘失败不该阻断主流程」的那条约定。
    """
    try:
        _cache_path().write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[声音克隆] 音色缓存写入失败（忽略）: {exc}")


def _fingerprint(source_audio: str) -> str:
    """源音频指纹：绝对路径 + 大小 + 修改时间。文件变了就重新克隆。

    Args:
        source_audio: 源音频/视频路径。**可以不存在**，也可以是非法路径
            （例如含 ``\\0``）—— 都返回空串，不抛异常。

    Returns:
        str：形如 ``F:\\...\\模特.mp4|10485760|1758000000`` 的指纹串，用作缓存键；
        **返回空串 = 没法指纹**（路径不是文件 / 路径非法 / stat 失败），
        调用方按「不能复用缓存」处理。

    为什么不用「内容 md5」：模特视频动辄几十上百 MB，算内容哈希要整读一遍；
    而这套指纹只需要 ``stat()`` 一次 —— 满足「素材换了就重克隆」已经足够。

    为什么存在性判断放在**本函数里**：早先这里不判、靠两个调用方各自先 ``is_file()``
    兜着，第三个调用方（或调用方判过之后文件被删/改名）就会踩 ``FileNotFoundError``。
    返回空串是最省事的收口：不动调用契约，也不新增异常类型去牵动调用方。

    为什么连 ``ValueError`` 一起接：``Path.resolve()`` 对含 ``\\0`` 的路径直接抛
    ``ValueError: stat: embedded null character in path``（``Path.is_file()`` 自己会
    吞掉 ``ValueError``，``resolve()`` 不吞）—— 实测 `_fingerprint("a\\0b.wav")` 就
    倒在这一步。契约是「拿不到指纹就给空串」，就不能只接 ``OSError``。
    """
    try:
        p = Path(source_audio).resolve()
        if not p.is_file():
            return ""
        st = p.stat()
    except (OSError, ValueError):
        return ""
    # mtime 取整数秒：指纹只用来判「素材变没变」，没必要精确到亚秒 ——
    # 同一份素材被复制/重新落盘时的亚秒误差不该触发重复建音色（配额很贵）。
    return f"{p}|{st.st_size}|{int(st.st_mtime)}"


# --------------------------------------------------------------------------
# 参考音频准备
# --------------------------------------------------------------------------
def _to_wav_16k(source: str) -> str:
    """把任意音频/视频转成 16kHz 单声道 wav —— 声音复刻的参考音频要求。

    已经是 16kHz 单声道的 wav 就直接复用，省一次转码（用 ffprobe 探真实参数，
    不只看扩展名：同名的 ``.wav`` 可能是 44.1kHz 立体声）。

    Args:
        source: 本地音频或视频路径（课案传的是模特视频，所以这里有 ``-vn``）。

    Returns:
        str：可直接投递的 wav 路径；转码失败或产物过小返回空串（**不抛异常**）。
    """
    src = Path(source)
    if src.suffix.lower() == ".wav":
        try:
            # 用 ffprobe 探一下，确认已经是 16k 单声道
            # `-select_streams a:0` 只取第一条音频流；timeout=30 足够 ——
            # ffprobe 只读容器头，正常毫秒级返回。
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
            # 没有 ffprobe / 输出不是合法 JSON / 探测超时 —— 任何一种都不该在这里中断：
            # 下面的 ffmpeg 转码本来就能把参数纠正过来（对已合规的文件只是白跑一次）。
            pass

    out = Path(settings.media.get_video_output_dir()) / f"voice_ref_{src.stem}.wav"
    try:
        # 参考音频通常几十秒，timeout=180 只为挡住「卡在损坏文件上不返回」。
        # 输出路径按源文件主名固定：同一素材重复走这条路会覆盖同一份产物，不会越堆越多。
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(out)],
            check=True, capture_output=True, timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[声音克隆] 参考音频转码失败: {exc}")
        return ""

    # 1 KB 下限：0 字节 / 几百字节的 wav 只可能是 ffmpeg 失败留下的空壳，
    # 送去建音色只会换来一个更难懂的服务端报错。
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
        prefix: 音色名前缀。SDK 的参数说明是「only digits and lowercase」、
            长度 ≤10 字符（``dashscope/audio/tts_v2/enrollment.py:96``，dashscope 1.27.4），
            默认值 ``mediaclone`` 即全小写。
        use_cache: 是否复用缓存（默认 True，强烈建议保持 True —— 音色有配额）。
        lang: 参考音频的语言提示，课案支持 ``zh`` / ``en`` / ``ja`` / ``ko``
            （``heygem_voice_clone(source_video, lang="zh")``）。不传时与旧行为
            完全一致 —— 以前这里把 ``["zh"]`` 写死了，英文/日文素材没法声明。

    Returns:
        ``{"success": bool, "voice_id": str, "from_cache": bool, "message": str}``
        —— **绝不抛异常**。失败时 ``success=False``、``voice_id=""``，
        具体原因（未配密钥 / 参考文件不存在 / 上传失败 / 建音色报错…）在 ``message`` 里。
    """
    # 先把「失败」的完整结构摆好：下面每条提前 return 的分支只需要回填 message。
    # 调用方（workflows/video.py）只读 success / voice_id / message 三个键，
    # 少一个键就会在页面里变成 KeyError。
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
    # 空指纹 = 这一刻拿不到这个文件（上面刚判过存在，这里只可能是极小概率的竞态）：
    # 不能拿空串当缓存键查（也不该写进去），按「没缓存」处理，照常往下建音色。
    cached = cache.get(key) if key else None
    if cached and cached.get("voice_id"):
        # 缓存命中是**正常路径**，不是优化：CosyVoice 建音色有配额，
        # 每次重跑页面都新建一次会把配额烧光（见模块 docstring 第 2 条）。
        # 命中时连 ffmpeg 转码和公网上传都省掉了 —— 它们在下面才发生。
        result.update(
            success=True, voice_id=cached["voice_id"], from_cache=True,
            message=f"命中音色缓存（创建于 {cached.get('created_at', '?')}）",
        )
        print(f"[声音克隆] {result['message']}: {cached['voice_id']}")
        return result

    # ---- 准备参考音频 ----
    ref_wav = _to_wav_16k(source_media)
    if not ref_wav:
        # 这里不用重复打印：`_to_wav_16k()` 内部已经把具体原因打出来了
        # （ffmpeg 缺失 / 转码报错 / 产物过小）。
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
        # VoiceEnrollmentService 与 SpeechSynthesizer 不同：它**接受** api_key= 形参。
        # 这里还是先绑一次全局 —— 两者共用底层客户端，统一绑全局是更稳的写法。
        service = VoiceEnrollmentService(
            api_key=settings.dashscope_api_key,
            # 没配业务空间时传 None：SDK 对 None 与空串都不加 workspace 头，
            # 显式传 None 意图更清楚（走通用域名）。
            workspace=settings.dashscope_workspace_id or None,
        )
        voice_id = service.create_voice(
            # target_model 必须与合成时用的模型**完全一致**，否则合成会失败 ——
            # 两边都取 settings.media.tts_model，不要在调用处硬编码模型名。
            target_model=settings.media.tts_model,
            prefix=prefix,
            url=ref_url,
            # language_hints 是 create_voice 的正式形参，不是靠 **kwargs 蒙进去的
            # （dashscope 1.27.4，inspect.signature 核实）
            language_hints=[lang],
        )
    except Exception as exc:  # noqa: BLE001 —— 配额/网络/参数错都收成中文提示
        # 这一层绝不能让异常冒出去（页面会直接 500）。把最常见的三个原因一起写进
        # message：配额满 / 参考音频不合格 / 地域不对 —— 报错原文在头部，便于对号入座。
        result["message"] = (
            f"创建音色失败: {exc}\n"
            "常见原因：音色配额已满（每次调用都会新建音色，注意复用）、"
            "参考音频不合格（需人声清晰、无背景音乐）、或地域不在华北2（北京）。"
        )
        print(f"[声音克隆] {result['message']}")
        return result

    if not voice_id:
        # 没抛异常但也没给 ID（接口返回 200 + 空 output 的情况），同样按失败处理，
        # 否则会把空字符串当 voice_id 写进缓存，下次命中它再在合成阶段莫名其妙地失败。
        result["message"] = "创建音色返回空 ID"
        print(f"[声音克隆] {result['message']}")
        return result

    result.update(success=True, voice_id=str(voice_id), message="音色创建成功")
    print(f"[声音克隆] OK voice_id={voice_id}")

    # 空指纹不写缓存：写进去就是一条谁都对不上的记录（键是 ""），下次也命中不了，
    # 只会把缓存文件搞脏。
    if use_cache and key:
        from datetime import datetime

        # 记下 source / target_model / created_at：排查「这个 voice_id 是哪来的、
        # 用哪个模型建的」时全靠这三个字段（页面的「选用已有音色」也直接展示它们）。
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

    Args:
        ref_wav: 已经转好格式的参考音频（16kHz 单声道 wav）本地路径。

    Returns:
        tuple：``(url, error)``。成功是 ``(公网 http(s) URL, "")``，
        失败是 ``("", 中文原因)``。失败原因会被 ``clone_voice()`` 原样塞进
        ``message`` 显示到页面上，所以 ③ 那一支写得比较长（直接教怎么配）。
    """
    ref_wav_path = Path(ref_wav)

    # ① 预置的固定参考音频 URL
    # getattr 兜底：`voice_ref_url` 是后加的配置项，万一 config.py 被回退成旧版
    # （没有这个字段），这里也不该 AttributeError —— 取不到就当没配，往下走 ②。
    preset = (getattr(settings.media, "voice_ref_url", "") or "").strip()
    if preset:
        # 只校验协议头，不校验这个 URL 是否真能访问：真访问不到会在 create_voice
        # 那一步报错，报错原文同样会回到页面（见 clone_voice 的 message）。
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
            # publish() 失败只回空串（原因它自己已经打印成 [托管] 日志），
            # 这里换成一句用户能看懂的说明，别把空串直接抛给页面。
            return "", "参考音频上传到公网素材托管失败（见上方 [托管] 日志）"
    except Exception as exc:  # noqa: BLE001 —— 托管模块出问题也要给出可读原因
        # 函数内 import + 整体 try：托管依赖本机 ssh/scp 与 .env 三项配置，
        # 它自己（或它的依赖）出问题不能把 clone_voice 弄崩 —— 记一行日志，落到 ③。
        print(f"[声音克隆] 调用素材托管失败: {exc}")

    # ③ 都没配
    # 给一份可照抄的配置清单。这段文本会**直接显示在页面的失败提示里**，
    # 所以写得比一般错误信息长 —— 目标是「照着做就能修好」，而不是「知道失败了」。
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

    注意：
        **不抛异常** —— 失败（文本空 / 音色 ID 空 / 没密钥 / 合成报错 / 产物过小）
        统一给空串，``workflows/video.py`` 靠它决定降级到 edge-tts。
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

    # 后缀与下面的 AudioFormat 枚举由同一个 `audio_format` 推出，必须成对 ——
    # 写错后缀会让下游按错误的容器去解这个文件（听不出来，但解码会失败）。
    suffix = ".mp3" if audio_format == "mp3" else ".wav"
    if output_path is None:
        # 用 md5 而非内置 hash()：hash() 每进程随机加盐，同一文本换进程会得到不同文件名
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        output_path = os.path.join(
            settings.media.get_video_output_dir(),
            f"voice_cloned_{digest}{suffix}",
        )

    try:
        # 函数内 import：dashscope 是可选依赖，模块顶层不加载它。
        from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

        _bind_dashscope_key()   # SpeechSynthesizer 不接 api_key，必须靠全局值
        # AudioFormat 枚举里**每一项都是 24kHz 单声道**（dashscope 1.27.4，
        # dashscope/audio/tts_v2/speech_synthesizer.py:94-124），所以这里挑的
        # 其实只是容器与码率：mp3 → 256kbps，wav → 16bit。
        fmt = (
            AudioFormat.MP3_24000HZ_MONO_256KBPS
            if audio_format == "mp3"
            else AudioFormat.WAV_24000HZ_MONO_16BIT
        )
        # model 必须与 create_voice 时的 target_model 一致（都是 settings.media.tts_model），
        # voice 就是那次返回的声音 ID —— 两者对不上时合成会直接失败。
        synthesizer = SpeechSynthesizer(
            model=settings.media.tts_model,
            voice=voice_id,
            format=fmt,
            workspace=settings.dashscope_workspace_id or None,
        )
        # call() 不设 callback 时，阻塞到收完音频并返回完整 bytes
        audio: bytes = synthesizer.call(text)
    except Exception as exc:  # noqa: BLE001
        # 音色被云端删掉、配额/欠费、网络超时都会走到这里。返回空串让上层降级，
        # 不要把异常抛进页面（用户要的是「这段视频还能不能出片」）。
        print(f"[克隆TTS] 合成失败: {exc}")
        return ""

    if not audio:
        print("[克隆TTS] 返回音频为空")
        return ""

    # 调用方可能给一个还不存在的目录（页面里手填输出路径），先建出来。
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Path(output_path).write_bytes(audio)
    # 1 KB 下限：几十字节的「音频」只可能是服务端返回的错误体被原样写成文件了。
    if Path(output_path).stat().st_size > 1024:
        print(f"[克隆TTS] OK {output_path}（{len(audio) / 1024:.1f} KB）")
        return output_path
    print("[克隆TTS] 输出文件过小，判定失败")
    return ""


def list_cloned_voices() -> list:
    """列出本地缓存的音色（给页面做「选用已有音色」用）。

    Returns:
        list[dict]：每项含 ``voice_id`` / ``source`` / ``target_model`` / ``created_at``
        四个键（缺失的字段补空串，页面可以直接渲染）。缓存读不出来时返回空列表。

    注意：
        这里**只读本地缓存**，不向百炼查这个 voice_id 还在不在云端 ——
        云端被删过的记录要显式清掉（``forget_voice()``），
        否则缓存会变成毒药：命中一条失效 ID，然后在合成阶段才失败。
        ⚠️ 但 `forget_voice()` **页面上没有任何入口、全仓 0 个调用方** ——
        只能在 Python 里自己调它（见那边的说明：源文件必须还在，否则清不掉）。
    """
    cache = _load_cache()
    items = []
    for k, v in cache.items():
        # 刻意丢掉缓存键（那串「路径|大小|mtime」指纹）——页面上没用，
        # 只暴露四个可读字段，顺便把缓存文件的内部结构挡在模块内部。
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

    ⚠️ 两个已知边界（**刻意保留，不是待修的 bug**，写在这里免得下次又被当成缺口）：
        · **源文件已被删除时清不掉**：缓存键 = 绝对路径 + 大小 + mtime，文件没了就
          算不出键。这里保留 `is_file()` 前置检查，不为了「删了也能清」改成按路径
          前缀扫缓存 —— 文件既然没了，`clone_voice()` 也不可能再命中那条记录，
          留着不影响使用。
        · **页面上没有入口、全仓 0 个调用方**：要清只能在 Python 里自己调它，
          「云端音色被删过、合成阶段才失败」目前在 UI 上没有解法。

    Args:
        source_media: 建音色时用的那个源文件路径。必须与交给 ``clone_voice()`` 的是
            **同一个路径** —— 缓存键按「绝对路径 + 大小 + mtime」算，改过内容
            （大小或 mtime 变了）的素材本来就是另一条记录；且**必须仍然存在**（见上）。

    Returns:
        是否删掉了一条记录。路径为空、文件不存在、或缓存里没有对应键都返回 False
        （不抛异常）。
    """
    if not source_media or not Path(source_media).is_file():
        return False
    cache = _load_cache()
    key = _fingerprint(source_media)
    # 刻意**不写 `if not key`**：上一行刚判过存在，`_fingerprint()` 现在又是全称的
    # （任何 stat / 路径异常都回空串），两步之间只剩一个 TOCTOU 窗口；而缓存里
    # **不可能**存在 ``""`` 键 —— 写入侧 `clone_voice()` 用 `if use_cache and key`
    # 挡掉了空指纹（见那边注释）。所以 `if not key` 那层只是看着像防御的空壳
    # （删掉它自检照样全绿），真撞上竞态时 `key not in cache` 本来就成立、照样返回 False。
    if key not in cache:
        return False
    # 先 pop 再整体回写：缓存文件是整份覆盖的（见 _save_cache），
    # 所以「改一份 dict 再写回」就是删除语义。
    removed = cache.pop(key)
    _save_cache(cache)
    print(f"[声音克隆] 已清除本地缓存: {removed.get('voice_id')}")
    return True


if __name__ == "__main__":
    # 离线自检：不建音色、不合成、不读 .env 里的密钥是否有效（只打印配没配）。
    print("=== 声音克隆模块自检（离线，不需要密钥）===")

    # 1) 缓存读写往返
    # 用临时目录造一个假的「模特音频」：指纹只依赖路径/大小/mtime，不需要真音频内容。
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "a.wav"
        fake.write_bytes(b"x" * 2048)
        fp1 = _fingerprint(str(fake))
        fp2 = _fingerprint(str(fake))
        assert fp1 == fp2, "同一文件指纹应稳定"
        fake.write_bytes(b"y" * 4096)          # 改内容（大小变了）
        assert _fingerprint(str(fake)) != fp1, "文件变化后指纹应改变"
        # 路径不是文件 → 返回空串（**不是**抛 FileNotFoundError）：这是为「第三个
        # 调用方」加的兜底，钉住它免得以后又被改回「让异常自然抛给调用方」。
        assert _fingerprint(str(Path(td) / "不存在.wav")) == "", "不存在的路径应返回空串"
        # 含 `\0` 的非法路径：`Path.resolve()` 会抛 `ValueError`（实测
        # `ValueError: stat: embedded null character in path`），本函数必须收成空串 ——
        # 契约是「拿不到指纹就给空串」，抛出去会让调用方接不住。
        assert _fingerprint("a\0b.wav") == "", r"含 \0 的非法路径应返回空串而不是抛 ValueError"
        print("  _fingerprint              OK  含 \\0 的非法路径也不抛")

    # 1b) `forget_voice()` 真的能删掉一条记录。它全仓 0 个调用方（页面没有入口），
    #     所以这个行为只能靠自检钉住；`_cache_path` 打桩到临时目录，
    #     **绝不碰本机真实缓存**（否则自检会删掉用户真金白银建出来的音色记录）。
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "ref.wav"
        fake.write_bytes(b"x" * 2048)
        _real_cache_path = _cache_path
        try:
            _cache_path = lambda: Path(td) / "voices.json"  # noqa: E731 —— 自检里顶掉
            _k = _fingerprint(str(fake))
            _save_cache({_k: {"voice_id": "v-stub", "source": str(fake)}})
            assert _load_cache().get(_k), "打桩缓存应能读回来"
            assert forget_voice(str(fake)) is True, "应删掉一条记录"
            assert _k not in _load_cache(), "记录应已从缓存里消失"
            assert forget_voice(str(fake)) is False, "已删过 → 第二次返回 False"
            fake.unlink()
            # 源文件删掉后**清不掉**（缓存键含大小/mtime，算不出键）—— 这是 docstring 里
            # 写明的**刻意边界**，不是期望行为：哪天真要让它支持「删了也能清」，
            # 记得把 `<forget_voice>` 的说明一起改掉。
            assert forget_voice(str(fake)) is False, "源文件已删 → 按已知边界返回 False"
        finally:
            _cache_path = _real_cache_path
        print("  forget_voice 清缓存        OK  删记录 / 幂等 / 源文件已删 三条边界")

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
    # 只断言返回类型、不断言条数：缓存里有没有内容取决于本机跑过几次克隆，
    # 断言条数会让这个自检变得不可重复。
    assert isinstance(list_cloned_voices(), list)
    print(f"  list_cloned_voices        OK  当前缓存 {len(list_cloned_voices())} 条")

    print(f"\n  密钥已配置: {is_configured()} | TTS 模型: {settings.media.tts_model}")
    print("全部自检通过")
