# -*- coding: utf-8 -*-
"""语音转文字 —— 百炼 Qwen-Audio-3.0-ASR-Flash（替代课案的本地 FunASR）

课案出处：自媒体课案 → 内容复刻（提文案）/ 视频剪辑（生成 SRT 字幕）

课案原本本地部署了**两个** FunASR 模型：
    · 纯文本通道：``FunAudioLLM/Fun-ASR-Nano-2512`` + ``fsmn-vad``
    · 时间戳通道：``iic/speech_seaco_paraformer_large_asr_nat-...`` + ``fsmn-vad`` + ``ct-punc``
      （懒加载，请求帧带 ``"return_timestamps": true`` 时才走）
两个模型在 GPU 服务器上跑一个自写的 WebSocket 服务（``deploy/funasr_ws_server.py``，端口 6006）。

本项目换成百炼的 **Qwen-Audio-3.0-ASR-Flash 非实时识别**
（Fun-ASR 家族的云托管版；模型名由 ``MEDIA_ASR_MODEL`` 决定，见 ``config.py``）：
一个 HTTP 接口同时提供「纯文本」与「句级 + 词级时间戳」，
**一个接口就替掉了课案的两个模型**，本机不需要 GPU、不下载权重。

接口对应关系
    | 课案 FunASR（本地 WS 服务） | 本项目百炼 | 说明 |
    |---|---|---|
    | ``Fun-ASR-Nano-2512``（纯文本通道） | 同一个 HTTP 接口 | 不带 ``words`` 时只取 ``output.text`` |
    | ``speech_seaco_paraformer_large``（时间戳通道，懒加载） | 同一个 HTTP 接口 | 非流式一次就返回覆盖全音频的 ``words[]``，见下方实测结论二 |
    | 请求帧 ``{"return_timestamps": true}`` | 无需开关，响应里自带 ``output.sentence.words`` | 所以本模块没有「要不要走时间戳模型」的判断 |
    | 自带 ``fsmn-vad`` + ``ct-punc`` 做切句 | 无 VAD、无标点模型 —— **切句自己用 ``words[]`` 的标点做** | 见 ``words_to_sentences()``，逻辑移植自课案 ``_split_sentences()`` |

谁在用它（都只看 ``transcribe()`` 返回的 ``error`` 字段决定要不要降级）
    · ``tools/media_tools.py`` 的 ``extract_audio_text()``（视频 → 文案，内容复刻链路），
      经 ``_transcribe_text()`` 收口：把 ``error`` 里的中文原因打出来再返回文本；
    · ``workflows/mashup.py`` 让 deepagent 写脚本调 ``transcribe_to_srt()`` 生成字幕；
    · ``verify_all.py`` 用它跑真实识别验收。

⚠️ **``error`` 必须上浮，别只看 ``text``**

    本模块所有函数都遵守「返回空值 + 中文原因」的约定：失败时 ``text=""``，
    **真因在 ``error`` 里**（未配置密钥 / 文件不存在 / 接口报错 / 上传失败）。
    早先的调用方只读 ``["text"]``，于是密钥没配、文件路径写错这类问题在页面上
    一律表现为「没拿到文案」，真因只进控制台 —— 所以 ``media_tools._transcribe_text()``
    专门加了一行 ``print(f"[提文案] 语音识别失败: {res['error']}")`` 把这个字段捞出来。
    改这里的返回值时，**先看谁在消费 ``error``**（本模块内部一个都不消费，全靠上层）。

================================================================
接口行为（**本机实测结论，与官方文档的印象不一致，务必先读这段**）
================================================================
端点: ``POST {settings.dashscope_api_endpoint}/services/aigc/multimodal-generation/generation``
请求::

    {"model": "...",
     "input": {"messages": [{"role": "user", "content": [
         {"type": "input_audio", "input_audio": {"data": "<公网URL 或 data:...;base64,xxx>"}}]}]},
     "parameters": {"format": "mp3",   # 按输入自身的容器推导，见 _guess_audio_format()
                    "sample_rate": "16000", "language_hints": ["zh"]}}

响应（实测，208 秒音频）::

    output.text           = 1922 字（完整转写文本）
    output.sentence.words = 418 个词，每个带 begin_time/end_time(milliseconds)/punctuation
    output.sentence.begin_time/end_time = 19010 ~ 207970ms  ← **覆盖全音频**
    usage.duration        = 208

⚠️ **关键实测结论一：``output.sentence`` 不是「一句」，而是「到目前为止的全量快照」**

    走 SSE 时每一帧都会返回一个 ``sentence``，但它不是逐句递进，而是**整段累积**：

        帧 1: sentence_id=1, begin_time=19010, end_time=34770,  words=36,  text 长 140
        帧 2: sentence_id=2, begin_time=19010, end_time=56010,  words=71,  text 长 318
        帧 3: sentence_id=3, begin_time=19010, end_time=76950,  words=106, text 长 496
        ...                                        ↑ begin_time 恒定，text/words 单调增长

    也就是说 **SSE 给不出「逐句切分」**，按 ``sentence_id`` 收集只会得到一堆逐级变长的重复文本
    （第一版实现就是这么错的，SRT 里每句都是前面内容的累加）。

✅ **关键实测结论二：不需要 SSE。非流式一次调用就返回覆盖全音频的完整 ``words[]``**

    同样 208 秒音频：
        非流式: words 401 个（覆盖 19010~207970ms），耗时 34s
        SSE   : 需要 14 帧，最后一帧 words 406 个，耗时 81s

    非流式**又全又快**，所以本模块只走非流式。

✅ **关键实测结论三：句级切分自己做 —— 用 ``words[]`` 的标点聚合**

    验证过 ``"".join(w["text"] + w["punctuation"] for w in words)`` 与 ``output.text`` **完全一致**，
    所以 ``words[]`` 是完整的、可直接用来切句（见 ``words_to_sentences()``）。

⚠️ **坑：``words[]`` 里有独立的空格 token**

    英文转写会产出 ``{"text": " ", "punctuation": ""}`` 这样的纯空白词（实测 418 个词里有 22 个）。
    切句时**必须把它们拼进去**，丢掉就会得到 ``so doi feel`` 这种粘连文本 ——
    第一版就是在这里丢的（当时写了 ``if not token.strip(): continue``）。

================================================================
其它两个必须知道的差异（相对课案）
================================================================
1. **音频要以 URL 或 Base64 Data URI 传入**，不能直接传本地路径。
   本模块自动处理（分流点在 ``_to_audio_input()``，条件是
   ``settings.media.asr_inline_max_bytes``，**默认 10MB**，环境键
   ``MEDIA_ASR_INLINE_MAX_BYTES``）：

       · ≤ 阈值 → ``data:<mime>;base64,<...>`` 内嵌（连对象存储都不用，最省事的一条路）
       · > 阈值 → 先经 ``tools/dashscope_upload`` 换 ``oss://`` 临时 URL 再传
       · 传进来的本来就是 http(s) / ``oss://`` URL → 原样透传（不下载、不转码）

   为什么阈值取 10MB：Base64 会把体积抬到约 4/3，10MB 音频对应的请求体已 ≈ 13.3MB，
   再往上加，瓶颈会先出现在请求体而不是模型上（``config.py`` 的
   ``asr_inline_max_bytes`` 那边只写了「超过就走临时存储」，这条推导补在这里）。
   ⚠️ 用 ``oss://`` 时 HTTP 请求必须带 ``X-DashScope-OssResourceResolve: enable``
   （见 ``_post()``；``avatar_client`` 走同一套约定）。
2. 时间戳单位是**毫秒**，与课案 FunASR 的输出一致，课案那套聚合逻辑可直接复用。
"""

import base64
import json
import mimetypes
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 句级切分用的结束标点（命中即结束当前句）
# 与课案 funasr_ws_server 的 _SENT_END_CHARS 保持一致 —— 注意**包含中文逗号**，
# 这是刻意的：短视频字幕讲究短句一行，长句会被画面裁掉。
_SENT_END_CHARS = set("，。！？；,.!?;…")

# 单条字幕的时长上限（毫秒）。**这是课案没有、本项目补的**：
# 实测 ASR 在某些片段（尤其是唱歌、连读、背景音）会连续不吐标点，
# 只按标点切会切出跨 16 秒、140 字符的巨型字幕行，画面上根本没法看。
# 超过这个时长就在当前词处收束一句 —— 宁可断在半个句子上，也不要糊满整屏。
_MAX_SEGMENT_MS = 8000

# 音频后缀 → MIME（Base64 Data URI 要带正确的 mediatype）
# 为什么手写这张表、而不是直接用 ``mimetypes.guess_type()``（本机实测差异）：
#     mimetypes 在 Windows 上会先读注册表，几个常见后缀的结果与本表不一致 ——
#         .aac  → 'audio/vnd.dlna.adts'（本表要 audio/aac）
#         .opus → 'audio/ogg'          （本表要 audio/opus）
#         .flac → 'audio/x-flac'       （本表要 audio/flac）
#     这三类错 MIME 会一路传到 ``_guess_audio_format()`` 的反查里，可能推出别的容器名。
#     表内优先、mimetypes 只当兜底（见 ``_to_audio_input()``）。
# ⚠️ 顺序有意义：``_guess_audio_format()`` 是按本表遍历做 MIME → 后缀的反查，
#    而 ``.m4a`` 与 ``.mp4`` 的 MIME 不同（audio/mp4 / video/mp4）恰好不冲突；
#    若哪天插入同 MIME 的两项，**命中的是排在前面那个**（实测 audio/mp4 → 'm4a'）。
_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}

# ``parameters.format`` 的兜底值：输入里既看不出容器、也认不出 MIME 时才用它
# （例如 http URL 后缀是 .php、或没有后缀 —— 见 ``_guess_audio_format()``）
_DEFAULT_AUDIO_FORMAT = "wav"


# ==========================================================================
# 对外主函数
# ==========================================================================
def transcribe(audio_path: str, want_timestamps: bool = False) -> dict:
    """把音频（或视频）转成文本，可选返回句级时间戳。

    这是本模块对外的**主入口**：上传/内嵌的分流、调接口、切句、错误收口都在这里串好，
    上层（``media_tools.extract_audio_text`` / ``workflows/mashup``）只需要调它。

    Args:
        audio_path: 本地文件路径、或已经是公网可访问的 http(s)/oss:// URL。
        want_timestamps: 是否要句级时间戳（生成 SRT 字幕时必须为 True）。

    Returns:
        ``{"text": str, "sentences": [{"text","start","end"}], "error": str}``
        —— **绝不抛异常**。失败时 ``text`` 为空串、``error`` 为中文说明。
        时间戳单位毫秒，与课案 FunASR 输出一致。
        不需要时间戳时 ``sentences`` 恒为 ``[]``（不是「没切出来」，是没切）。

    失败时的返回（调用方请判 ``error`` 而不是只看 ``text``）：
        · ``audio_path`` 为空          → error="音频路径为空"
        · 没配 ``DASHSCOPE_API_KEY``   → error 里带「见根目录 .env」的指引
        · 文件不存在 / 大文件上传失败  → error 来自 ``_to_audio_input()``
        · 网络异常 / 接口非 200 / 解析失败 → error="调用百炼语音识别失败: ..."
        · 响应里既没有 text 也没有句子 → error 里带响应体前 300 字符
    """
    result = {"text": "", "sentences": [], "error": ""}

    # 三关前置校验都直接 return 带 error 的结果：这一层不抛异常（见文件头约定），
    # 因为调用点都在 LangGraph 节点里，抛出去整条图就断了。
    if not audio_path:
        result["error"] = "音频路径为空"
        return result

    # 密钥缺失是最常见的「识别不出东西」原因，提示里直接把该去哪儿配写清楚，
    # 免得使用者去翻源码找键名。
    if not settings.dashscope_api_key:
        result["error"] = (
            "未配置 DASHSCOPE_API_KEY（见根目录 .env）。"
            "语音识别走百炼 Qwen-Audio-3.0-ASR-Flash，需要该密钥。"
        )
        print(f"[ASR] {result['error']}")
        return result

    # 这里决定走 Base64 内嵌还是 oss:// 临时 URL（分流条件见 _to_audio_input）。
    data_uri, err = _to_audio_input(audio_path)
    if err:
        result["error"] = err
        return result

    try:
        text, sentences, err = _call_once(data_uri)
    except Exception as exc:  # noqa: BLE001 —— 网络/解析异常一律收成中文提示
        # _post() 在非 200 时是 raise 的（本模块唯一会抛的地方），requests 本身
        # 也会抛连接/超时异常，两者都收在这一层，对外只留中文 error。
        err, text, sentences = f"调用百炼语音识别失败: {exc}", "", []

    if err:
        result["error"] = err
        print(f"[ASR] {err}")
        return result

    result["text"] = text
    # 只有明确要时间戳时才切句：切句本身不花钱，但没必要给调用方塞多余结构
    result["sentences"] = sentences if want_timestamps else []
    print(
        f"[ASR] 识别完成：{len(text)} 字"
        + (f"，切出 {len(result['sentences'])} 句" if want_timestamps else "")
    )
    return result


def sentences_to_srt(sentences: list, output_path: str) -> str:
    """把句级时间戳写成 SRT 字幕文件。

    Args:
        sentences: ``transcribe()`` 返回的 ``sentences`` 列表（毫秒）。
        output_path: 输出 .srt 路径。

    Returns:
        写入的文件路径；无有效句子时返回空串。

    失败时的行为：
        句子为空 / 全被过滤掉 → 打印中文提示、返回空串，**不创建文件**；
        目录不存在会自动 ``mkdir(parents=True)``。写文件本身（``write_text``）没有
        try 包住，真写不进去会抛 —— 但这属于「磁盘/权限坏了」，不该静默；
        实际调用点只有 ``transcribe_to_srt()`` 与本文件自检。
    """
    # 过滤掉「没文本」和「没有结束时间」的句子。后者是必需的：
    # SRT 必须写 start --> end，end 缺失时只能填 0，会产出零长字幕
    # （播放器上表现为闪一下就没了），不如不写这一条。
    valid = [
        s for s in (sentences or [])
        if str(s.get("text", "")).strip() and s.get("end") is not None
    ]
    if not valid:
        print("[ASR] 没有可写入字幕的句子")
        return ""

    lines = []
    # SRT 序号从 1 开始、且按顺序排（序号是播放器的排序依据，跳号/从 0 起都可能被拒；
    # 本文件自检里那条 ``content.startswith("1\n")`` 就是钉这个的）。
    for idx, seg in enumerate(valid, 1):
        # start 缺失时按 0 处理（字幕从头出现），end 上面已经保证有值。
        start = _ms_to_srt_time(int(seg.get("start") or 0))
        end = _ms_to_srt_time(int(seg.get("end") or 0))
        # 每段自带尾部 \n，段落之间就靠它隔开一个空行 —— SRT 的空行是分段符，
        # 少了它播放器会把两段并成一条。
        lines.append(f"{idx}\n{start} --> {end}\n{str(seg['text']).strip()}\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # 显式 utf-8：字幕里有中文，Windows 默认编码会写坏（课案的 FFmpeg 烧字幕同理）。
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[ASR] 已写出字幕（{len(valid)} 句）: {output_path}")
    return output_path


def transcribe_to_srt(audio_path: str, output_path: str) -> str:
    """一步到位：音频（或视频）→ SRT 字幕。给视频剪辑链路用。

    替代课案里让 deepagent 自己写脚本调本地 funasr 生成字幕那一步。

    Args:
        audio_path: 本地文件路径或公网 URL（透传给 ``transcribe()``）。
        output_path: 输出 .srt 路径。

    Returns:
        SRT 文件路径；识别失败或没有可用句子时返回空串。

    为什么失败时返回空串而不是落下「空字幕文件」：
        剪辑链路（``workflows/mashup``）会拿返回的路径去给 moviepy 烧字幕，
        空 SRT 会让它以为「字幕准备好了」而烧出一版没有字幕的视频 ——
        返回空串能让它明确看到「没字幕可用」。
    """
    res = transcribe(audio_path, want_timestamps=True)
    if res["error"]:
        return ""
    return sentences_to_srt(res["sentences"], output_path)


def transcribe_words(audio_path: str) -> tuple:
    """取「完整文本 + 原始词级时间戳」，给需要自己切片字幕的调用方用。

    与 ``transcribe()`` 的区别：这一条**不做句级切分**，把百炼返回的
    ``output.sentence.words`` 原样交出去（调用方想按自己的规则切就自己切）；
    也不过滤 ``error``，失败时第 1、2 项为空。

    Args:
        audio_path: 本地文件路径或公网 URL（内部同样走 ``_to_audio_input()`` 分流）。

    Returns:
        ``(text, words, error)``；words 是百炼原样的词列表。
        失败时 ``("", [], 中文原因)`` —— 同样不抛异常。
    """
    if not settings.dashscope_api_key:
        return "", [], "未配置 DASHSCOPE_API_KEY"
    data_uri, err = _to_audio_input(audio_path)
    if err:
        return "", [], err
    try:
        resp = _post(data_uri)
    except Exception as exc:  # noqa: BLE001
        # 与 transcribe() 相同的收口：_post 非 200 会 raise，这里转成 error 字符串。
        return "", [], f"调用百炼语音识别失败: {exc}"

    # 只解 text / words 两个字段，不切句也不做兜底 —— 这条路径的契约就是「原样」。
    body = resp.json()
    output = body.get("output") or {}
    sentence = output.get("sentence") or {}
    return (
        str(output.get("text") or "").strip(),
        sentence.get("words") or [],
        "",
    )


# ==========================================================================
# 内部实现
# ==========================================================================
def _ms_to_srt_time(ms: int) -> str:
    """毫秒 → SRT 时间格式 ``HH:MM:SS,mmm``。

    Args:
        ms: 毫秒（百炼返回的时间戳就是这个单位）。允许负数与超大值。

    Returns:
        形如 ``00:00:01,500`` 的字符串（小时段不封顶，超过 24 小时也照写）。
    """
    # 负值夹到 0：ASR 偶尔会给极小的负偏移，写进 SRT 会得到 "-1:-1:-1"，
    # 播放器直接解析不了整条字幕。
    if ms < 0:
        ms = 0
    # 注意分隔符：SRT 的毫秒位是**逗号**不是点（点号是 WebVTT 的写法）。
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def _to_audio_input(audio_path: str) -> tuple:
    """把输入整理成百炼能接受的 ``data`` 字段。

    公网 URL / oss:// 原样返回；本地文件 ≤限制则转 Base64，更大则先换临时 URL。

    **两条传输路径的分流条件**（本模块最关键的一处判断）：

        · 已经是 ``http(s)://`` / ``oss://`` → 直接透传。这两个前缀就是「外部能取到」
          的标志，服务端自己去拉，本机不必下载也不再传一遍（省流量也省时间）。
        · 本地文件 ``size <= settings.media.asr_inline_max_bytes``（**默认 10MB**）
          → 读全文件、Base64 编码、拼成 ``data:<mime>;base64,<...>`` 内嵌在请求体里。
          这条路径**不依赖任何对象存储**，是最不容易出故障的一条。
        · 本地文件更大 → 先 ``upload_file(path, settings.media.asr_model)`` 换 ``oss://``
          临时 URL，再把 URL 当 ``data`` 传（后面 ``_post()`` 会自动带上那枚
          ``X-DashScope-OssResourceResolve`` 头）。

    Args:
        audio_path: 本地路径，或已经可被外部访问的 URL。

    Returns:
        ``(data, error)``：成功时 error 为空串；失败时 data 为空串、error 是中文原因。
        **不抛异常** —— 文件读不动、上传失败都收成 error 字符串。
    """
    # URL 短路放在最前面：对 URL 做本地文件检查没有意义（也判断不了远端资源在不在），
    # 而「外部能取到」这件事只有 http(s)/oss:// 前缀能代表，所以直接透传。
    # oss:// 也在这里 —— 它是百炼内部协议，与 http 一样属于「服务端自己能取到」的资源。
    if audio_path.startswith(("http://", "https://", "oss://")):
        return audio_path, ""

    path = Path(audio_path)
    if not path.is_file():
        return "", f"音频文件不存在: {audio_path}"

    size = path.stat().st_size
    # 阈值来自 config（默认 10MB）：调大它就会把更多文件塞进 Base64 内嵌，
    # 代价是请求体按 4/3 膨胀 —— 改之前先确认服务端的请求体上限。
    limit = settings.media.asr_inline_max_bytes
    ext = path.suffix.lower()

    if size <= limit:
        # MIME 推导链：本模块的表（Windows 上比 mimetypes 准，见 _MIME_BY_EXT 注释）
        #   → mimetypes 兜底（覆盖表里没有的后缀，实测 .wma → 'audio/x-ms-wma'；
        #     注意它也可能返回 None，如 .amr / .mpga）
        #   → 最后 audio/wav 保底（宁可 MIME 不准，也不能让 Data URI 缺 mediatype，
        #     缺了整条 data:...;base64, 都构不成合法 URI）。
        mime = _MIME_BY_EXT.get(ext) or mimetypes.guess_type(path.name)[0] or "audio/wav"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        # 打印实际体积与最终 MIME：出问题时（服务端说容器不对）这行是唯一线索。
        print(f"[ASR] 音频 {size / 1024 / 1024:.2f} MB，走 Base64 内嵌（{mime}）")
        return f"data:{mime};base64,{b64}", ""

    print(f"[ASR] 音频 {size / 1024 / 1024:.2f} MB 超过 {limit / 1024 / 1024:.0f} MB，"
          f"改走百炼临时存储")
    # 延迟导入：只有「大文件」这一条路径才需要 dashscope SDK，
    # 小文件与 URL 两条路走到底都不会 import 它。
    from tools.dashscope_upload import upload_file

    # ⚠️ model 必须与 _post() 里 payload 的 model 用同一个值（都取 settings.media.asr_model）：
    # 百炼的临时存储是「文件与模型绑定」的，两处不一致不会在上传时报错，
    # 只会在识别时报「文件不可用」。
    oss_url = upload_file(str(path), settings.media.asr_model)
    if not oss_url:
        # upload_file 失败时已经把真因打出来了，这里只补一句「是这条路走不通」。
        return "", "音频过大且上传临时存储失败（见上方日志）"
    return oss_url, ""


def _guess_audio_format(data: str) -> str:
    """从输入自身推出 ``parameters.format`` 用的容器名。

    ``_to_audio_input()`` 交到 ``_post()`` 的只有两种东西：
    ``data:<mime>;base64,xxx`` 或 http(s)/oss:// URL —— 到这一层**本地后缀已经没了**，
    所以 data URI 读它自带的 mediatype、URL 读路径后缀，认不出才退回
    ``_DEFAULT_AUDIO_FORMAT``。

    ⚠️ 以前这里硬编码 ``"wav"``，而函数实际吃 mp3/mp4/m4a 各种容器
    （MIME 表见 ``_MIME_BY_EXT``），docstring 的示例又写 ``"mp3"`` —— 三方不一致。
    实测 mp3 输入照样能识别（服务端似乎不拿这个值校验容器），所以不是功能故障；
    但**声明与输入不符**，换模型 / 换地域就可能踩，这里按输入推导掉。

    **为什么必须按容器推导、不能硬编码**：``parameters.format`` 是「我这次送上来的是
    什么容器」的声明。硬编码 wav 时，送 mp4 却声明 wav —— 当前服务端宽松所以能跑，
    但这个不一致是「依赖服务端宽不宽松」，不是「我们对了」。真正的容器就在输入里
    （Data URI 的 mediatype / URL 的后缀），零成本就能取到，所以按它推导。

    Args:
        data: ``_to_audio_input()`` 的产物 —— ``data:<mime>;base64,...`` 或 http(s)/oss:// URL。

    Returns:
        小写扩展名的容器名（``'mp3'`` / ``'m4a'`` / ``'mp4'`` / ``'wav'``...）；
        两边都认不出时返回 ``_DEFAULT_AUDIO_FORMAT``（``'wav'``）。

    本机实测的映射（``_MIME_BY_EXT`` 反查的结果，2026-09 用探针逐条跑过）::

        data:audio/mpeg;base64,...   -> mp3
        data:audio/wav  / audio/x-wav -> wav      （x- 前缀会被去掉）
        data:audio/mp4;base64,...    -> m4a       （表里 .m4a 在前，先命中它）
        data:video/mp4;base64,...    -> mp4
        data:audio/weird-thing;...   -> weird-thing  ← 不认识的 MIME 原样带过去
        data:audio/x-ms-wma;base64,..-> ms-wma    ← 只削 x- 前缀，子类型原样保留
        https://h/a.M4A?sign=x       -> m4a       （后缀大小写不敏感）
        https://h/noext  /  x.php    -> wav       （认不出 → 兜底值）
    """
    if data.startswith("data:"):
        # 取 mediatype：`data:audio/mpeg;base64,AAA` → `audio/mpeg`
        # （split(";", 1) 是为了兼容未来可能出现的 `;charset=` 之类参数）
        mime = data[5:].split(";", 1)[0].strip().lower()
        # 反查：表里第一个 MIME 相等的项胜出 —— 所以 **表的顺序是语义的一部分**
        # （.m4a 与 .mp4 的 MIME 不同，恰好不冲突；插入同 MIME 的两项时命中的是前者）。
        for ext, known in _MIME_BY_EXT.items():
            if known == mime:
                return ext.lstrip(".")
        # mimetypes 猜出来的别名（如 audio/x-m4a）→ 取子类型、去掉 x- 前缀
        return mime.split("/")[-1].removeprefix("x-") or _DEFAULT_AUDIO_FORMAT

    suffix = Path(urlparse(data).path).suffix.lower()
    # 只认表里认识的容器，别把 URL 上奇怪的后缀（.php/.aspx）当格式发上去
    if suffix in _MIME_BY_EXT:
        return suffix.lstrip(".")
    return _DEFAULT_AUDIO_FORMAT


def _post(data: str) -> requests.Response:
    """发一次**非流式**请求。

    ``X-DashScope-OssResourceResolve`` 只在传 ``oss://`` 时必须，
    但它对普通 URL / Base64 也无害，所以统一带上、少一个分支。

    ``format`` 由输入自身的容器推导（见 ``_guess_audio_format()``），不再硬编码。

    Args:
        data: 音频数据 —— ``data:<mime>;base64,...`` 或 http(s)/oss:// URL。

    Returns:
        已确认 HTTP 200 的 ``requests.Response``。

    Raises:
        RuntimeError: 状态码非 200（消息里带状态码与响应体前 300 字符）。
            这是**本模块唯一会抛异常的地方**，两个调用方
            （``transcribe()`` / ``transcribe_words()``）都在 try 里收成中文 error。
        requests.RequestException: 网络层异常（连接失败 / 超时 / DNS），同上收口。
    """
    params = {
        # 声明本次送上来的是什么容器；服务端当前不严格校验（实测 mp3 声明 wav 也能过），
        # 但声明与输入一致才不依赖服务端的宽松程度。
        "format": _guess_audio_format(data),
        # ``sample_rate`` 同样是对「本次输入的采样率」的声明：这里固定 16k，
        # 与课案那条链路一致（课案抽音频写死 ``-ar 16000 -ac 1``）。
        # ⚠️ 已知口径不一致（**本轮只记录、未实测过故障、也未被实测证实无害**）：
        #    本项目走 ``tools/media_tools.video_to_audio()`` 抽音频时用的是它的默认值
        #    44100，所以对「抽出来的 wav」这条声明并不成立。
        "sample_rate": "16000",
    }
    # language_hints 只在配置了语种时才带：不带 = 让模型自己判；
    # 而带上空串会变成 language_hints=[""]，等于「有一个语种叫空字符串」，
    # 不是「没指定」—— 所以这里显式判空，不能直接写 [settings.media.asr_language]。
    lang = (settings.media.asr_language or "").strip()
    if lang:
        params["language_hints"] = [lang]

    # 请求体形状就是百炼「多模态生成」的通用形状（messages + content 数组），
    # ASR 只是把 content 的类型换成 input_audio。字段名一个字都别改。
    payload = {
        "model": settings.media.asr_model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": data}}
                    ],
                }
            ]
        },
        "parameters": params,
    }

    url = (
        f"{settings.dashscope_api_endpoint}"
        "/services/aigc/multimodal-generation/generation"
    )
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
            # 明确关掉 SSE：默认会按流式返回，而流式的 sentence 是全量快照、
            # 且慢一倍（见文件头实测结论一/二）。这一行是「只走非流式」的开关。
            "X-DashScope-SSE": "disable",
            # 传 oss:// 时服务端必须靠这个头才知道要去解析 DashScope 的临时存储。
            # 对 Base64 / 普通 URL 无害，所以无条件带上，少一个 if 分支。
            "X-DashScope-OssResourceResolve": "enable",
        },
        json=payload,
        # 300s：实测 208 秒音频非流式耗时 34s；长音频 + 排队 + Base64 上传体
        # 都要算在这一次请求里，给约 9 倍余量，宁可慢也不要在转写中途断开。
        timeout=300,
    )
    # 非 200 一律 raise：交给 transcribe()/transcribe_words() 收成中文 error，
    # 这里抛是为了让「HTTP 层出问题」和「响应里没结果」在日志里能区分开。
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp


def _call_once(data: str) -> tuple:
    """非流式调用：一次拿到完整文本 + 完整词级时间戳，再本地切句。

    之所以能这么简单，全靠文件头那条实测结论二：非流式一次返回的 ``words[]``
    就覆盖了全音频（SSE 要走 14 帧、慢一倍，且给不出逐句切分）。

    Args:
        data: 已整理好的音频输入（``data:...`` 或 URL）。

    Returns:
        ``(text, sentences, error)``：
        文本取自 ``output.text``；``sentences`` 由 ``words_to_sentences()`` 本地切出；
        连 text 和句子都没有时返回 error（消息里带响应体前 300 字符），便于比对形状变化。
    """
    resp = _post(data)
    body = resp.json()
    # 逐层 or {}：宁可在这几行上多写几个空字典，也不要让 API 少给一层时报 KeyError
    # ——那样异常会带上调用栈，反而看不出是「响应形状变了」。
    output = body.get("output") or {}
    sentence = output.get("sentence") or {}

    text = str(output.get("text") or "").strip()
    words = sentence.get("words") or []

    sentences = words_to_sentences(words) if words else []

    # 兜底：万一这一版 API 没给 words，但给了 sentence 的起止时间，就当成一整句
    # （整段字幕总比一条都没有强；课案的 FunASR 通道也允许没有词级信息）
    if not sentences:
        fallback = _sentence_from(sentence)
        if fallback:
            sentences = [fallback]

    if not text and not sentences:
        # 把响应体裁到 300 字符一起报出去：这种失败基本都是「返回形状变了」，
        # 不贴原文就只能靠猜。截断是为了不把 Base64 回显刷满日志。
        return "", [], f"响应里没有识别结果: {json.dumps(body, ensure_ascii=False)[:300]}"
    if not text:
        # 有句子没 text 时用句子拼回完整文本 —— 上层只认 text（比如提文案链路），
        # 返回空串会让「识别成功但没文本」看起来像失败。
        text = "".join(s["text"] for s in sentences)
    return text, sentences, ""


def _sentence_from(node) -> dict | None:
    """把百炼返回的 sentence 对象转成单句格式；字段不全时返回 None。

    Args:
        node: ``output.sentence``（可能不是 dict，也可能字段缺失）。

    Returns:
        ``{"text", "start", "end"}``（毫秒）；不可用时返回 None，由调用方决定怎么兜。
    """
    if not isinstance(node, dict):
        return None
    text = str(node.get("text") or "").strip()
    begin = node.get("begin_time")
    end = node.get("end_time")
    # 三个字段缺一不可：没有 text 的句子是空字幕，没有 begin/end 的写不进 SRT。
    if not text or begin is None or end is None:
        return None
    # 时间倒挂/零长的直接拒掉：SRT 里会出现 "00:00:01,000 --> 00:00:01,000"，
    # 有的播放器会整条字幕都不显示，比缺一句更糟。
    if int(end) <= int(begin):
        return None
    return {"text": text, "start": int(begin), "end": int(end)}


def words_to_sentences(words: list) -> list:
    """把**词级**时间戳按标点聚合成句级段落。

    移植自课案 ``deploy/funasr_ws_server.py`` 的 ``_split_sentences()``，
    但针对百炼的实际返回做了两处必要适配：

    1. 课案那版按「逐字 + 每字一个 timestamp」推进；
       百炼给的是「多字词 + 每词自己的 begin/end + 该词之后的标点」，
       所以改成按词推进。语义一致：**遇到结束标点就收束为一句**。
    2. **保留纯空白词**。百炼的英文转写会产出 ``{"text": " "}`` 这种空格 token
       （实测 418 个词里有 22 个），丢了就会拼出 ``so doi feel``。
       实测 ``"".join(text + punctuation)`` 能精确还原 ``output.text``，
       所以这里对每个词都原样拼接，只在记录时间时跳过空白词。

    Args:
        words: ``[{"text","begin_time","end_time","punctuation"}, ...]``，毫秒。
            直接来自 ``output.sentence.words``；非 dict 的项、彻头彻尾的空 token 会被跳过。

    Returns:
        ``[{"text": str, "start": int, "end": int}, ...]``
        没有词、或全是空白词时返回 ``[]``（不抛异常、不返回 None）。
        单个句子的时长可能略超 ``_MAX_SEGMENT_MS`` —— 兜底是「预判」式的，
        只保证本词不再被并入超长句（见下面时长兜底那段注释与自检 4c）。
    """
    sentences: list = []
    cur_text: list = []      # 当前句的 token 片段（含标点），最后一次性 join
    cur_start = None         # 当前句起点：由「第一个有实际内容的词」决定
    cur_end = None           # 当前句终点：由「最后一个有实际内容的词」决定

    def flush() -> None:
        """收束当前句并入结果；空句（全是空白 token）直接丢弃。"""
        nonlocal cur_text, cur_start, cur_end
        # strip() 后才判空：只有空白词的段落不该产出句子（否则 SRT 里会出现空行）。
        seg = "".join(cur_text).strip()
        if seg and cur_start is not None and cur_end is not None:
            sentences.append({"text": seg, "start": int(cur_start), "end": int(cur_end)})
        # 无论是否入结果都清空状态，避免上一句的残片带到下一句。
        cur_text, cur_start, cur_end = [], None, None

    for word in words or []:
        # 容错：API 换代/字段异常时，一个坏元素不该让整段转写全丢。
        if not isinstance(word, dict):
            continue
        token = str(word.get("text") or "")
        punct = str(word.get("punctuation") or "")
        if not token and not punct:
            continue

        begin = word.get("begin_time")
        end = word.get("end_time")

        # 时长兜底：模型长时间不吐标点时，靠这一条把字幕行截短（见 _MAX_SEGMENT_MS）。
        # ⚠️ 关键在判断时机 —— 必须把**当前这个词**算进去做预判，
        #    而且要赶在它被并进 cur_text 之前：
        #      · 用「上一个词」的结束时间判断 → 下一个词跳得远时仍会撑长当前句
        #        （实测粤语新闻里切出过 9235ms > 8000ms）；
        #      · 先并入再判断 → 本词的时间已经记进 cur_end，同样撑长。
        #    预判命中就先收束上一句，让本词成为新句的开头。
        if (
            cur_start is not None
            and cur_text
            and end is not None
            and int(end) - int(cur_start) > _MAX_SEGMENT_MS
        ):
            flush()

        # 时间只认有实际内容的词；纯空白词不参与，避免把句子起点带到空格上
        if token.strip():
            if cur_start is None and begin is not None:
                cur_start = begin
            if end is not None:
                cur_end = end

        # ⚠️ token 一律原样并入（**包括纯空白 token**）：
        #    实测 "".join(text + punctuation) 能精确还原 output.text，
        #    丢掉空格就会得到 "so doi feel" 这种粘连文本（第一版就栽在这）。
        cur_text.append(token)
        if punct:
            cur_text.append(punct)
            # 用「集合交集」而不是 `punct in _SENT_END_CHARS`：当前实测 punctuation
            # 都是单字符（两种写法等价），但它是字符串字段，一旦变成 "。 " 这类组合，
            # 整串成员判断就会漏；strip() 则是不看空白、只看真正起作用的标点。
            if set(punct.strip()) & _SENT_END_CHARS:
                flush()
    # 收尾：最后一段没有结束标点的内容（口语长尾很常见）也要落成一句。
    flush()
    return sentences


if __name__ == "__main__":
    # 自检：纯逻辑，不依赖网络与密钥
    # （本段必须保持离线、秒回：verify_all.py 第 1 层与每次改动后都会跑它。
    #   真实验证要花钱调 ASR，见文件末尾那句提示。）
    print("=== 语音转文字模块自检 ===")

    # 1) SRT 时间格式化
    #    3661_500 = 1 小时 1 分 1.5 秒，一眼就能看出时/分/秒/毫秒有没有错位；
    #    -5 是盯「负值夹到 0」那条（ASR 偶尔给极小负偏移）。
    assert _ms_to_srt_time(0) == "00:00:00,000"
    assert _ms_to_srt_time(3661_500) == "01:01:01,500"
    assert _ms_to_srt_time(-5) == "00:00:00,000"
    print("  _ms_to_srt_time           OK")

    # 2) sentence 解析（含字段不全/时间倒挂应被拒）
    #    四条分别对应：正常 / 零长（end == begin）/ 无文本 / 整个对象不是 dict。
    assert _sentence_from({"text": "你好", "begin_time": 0, "end_time": 500})
    assert _sentence_from({"text": "你好", "begin_time": 500, "end_time": 500}) is None
    assert _sentence_from({"text": "", "begin_time": 0, "end_time": 500}) is None
    assert _sentence_from(None) is None
    print("  _sentence_from            OK")

    # 3) 词级 → 句级聚合
    #    注意：结束标点集合**包含中文逗号**，所以「你好，」本身就算一句 ——
    #    这是刻意的，短视频字幕讲究短句一行，长句会被画面裁掉。
    words = [
        {"text": "你好", "begin_time": 0, "end_time": 200, "punctuation": "，"},
        {"text": "世界", "begin_time": 200, "end_time": 500, "punctuation": "。"},
        {"text": "第二句", "begin_time": 600, "end_time": 900, "punctuation": "！"},
    ]
    sents = words_to_sentences(words)
    assert len(sents) == 3, f"逗号也算句末，应切成 3 句，实际 {len(sents)}"
    assert sents[0]["text"] == "你好，" and (sents[0]["start"], sents[0]["end"]) == (0, 200)
    assert sents[1]["text"] == "世界。" and (sents[1]["start"], sents[1]["end"]) == (200, 500)
    assert sents[2]["text"] == "第二句！" and (sents[2]["start"], sents[2]["end"]) == (600, 900)
    print("  words_to_sentences        OK")

    # 3b) **回归：纯空白词不能被丢掉**（第一版就是在这里出 bug 的）
    spaced = [
        {"text": "so", "begin_time": 0, "end_time": 100, "punctuation": ""},
        {"text": " do", "begin_time": 100, "end_time": 200, "punctuation": ""},
        {"text": " ", "begin_time": 200, "end_time": 300, "punctuation": ""},
        {"text": "i", "begin_time": 300, "end_time": 400, "punctuation": ""},
        {"text": " feel", "begin_time": 400, "end_time": 500, "punctuation": "."},
    ]
    joined = words_to_sentences(spaced)
    assert len(joined) == 1, joined
    assert joined[0]["text"] == "so do i feel.", repr(joined[0]["text"])
    print("  words_to_sentences(含空格) OK  →", repr(joined[0]["text"]))

    # 3c) 拼接结果应能还原原文（这是判断切句没丢东西的最硬标准）
    raw = [
        {"text": "We", "begin_time": 0, "end_time": 10, "punctuation": ""},
        {"text": "'re", "begin_time": 10, "end_time": 20, "punctuation": ""},
        {"text": " here", "begin_time": 20, "end_time": 30, "punctuation": "."},
    ]
    assert "".join(w["text"] + w["punctuation"] for w in raw) == "We're here.", "拼接基准"
    print("  拼接还原原文基准           OK")

    # 4) 无标点时整段兜底为一句
    sents2 = words_to_sentences([{"text": "没有标点", "begin_time": 0, "end_time": 100}])
    assert len(sents2) == 1 and sents2[0]["text"] == "没有标点"
    print("  words_to_sentences(无边标) OK")

    # 4b) 时长兜底：长时间不吐标点必须被截断，否则会切出跨十几秒的巨型字幕行
    long_run = [
        {"text": f"w{i}", "begin_time": i * 1000, "end_time": (i + 1) * 1000,
         "punctuation": ""}
        for i in range(20)
    ]
    cut = words_to_sentences(long_run)
    assert len(cut) > 1, f"20 秒无标点应被 _MAX_SEGMENT_MS 截断，实际只有 {len(cut)} 句"
    # 判据：收束出来的句子不能超过上限 + 一个词的长度（兜底是按「已收进来的词」判断的）
    assert all(s["end"] - s["start"] <= _MAX_SEGMENT_MS + 1000 for s in cut), cut
    print(f"  words_to_sentences(时长兜底) OK  20s → {len(cut)} 句，"
          f"最长 {max(s['end'] - s['start'] for s in cut)}ms")

    # 4c) 回归：兜底必须把「当前词」算进去预判
    #     （曾出现 9235ms > 8000ms：判断用的是上一个词的结束时间，下一个词跳得远照样撑长）
    sparse = [
        {"text": "a", "begin_time": 0, "end_time": 100, "punctuation": ""},
        {"text": "b", "begin_time": 7000, "end_time": 7100, "punctuation": ""},
        # 这一跳很大：若把它的 end 并进来再判断，首句就变成 0~9100ms
        {"text": "c", "begin_time": 9000, "end_time": 9100, "punctuation": ""},
    ]
    sp = words_to_sentences(sparse)
    assert len(sp) == 2, f"应被截成 2 句，实际 {sp}"
    assert sp[0]["end"] <= _MAX_SEGMENT_MS, f"首句被下一个词撑长了: {sp}"
    assert (sp[0]["text"], sp[1]["text"]) == ("ab", "c"), sp
    print(f"  words_to_sentences(兜底预判) OK  首句 {sp[0]['start']}-{sp[0]['end']}ms / "
          f"次句 {sp[1]['start']}-{sp[1]['end']}ms")

    # 5) 全空白词不应产出任何句子
    assert words_to_sentences([{"text": " ", "begin_time": 0, "end_time": 10}]) == []
    print("  words_to_sentences(全空白) OK")

    # 6) 失败路径必须返回结构化结果而不是抛异常
    #    ⚠️ 这两条断言**不区分失败原因**：第二条（文件不存在）在没配 API 密钥的机器上
    #    会先被密钥检查拦下，同样满足「error 非空 + text 为空」。
    #    所以它证明的是「契约成立」，不是「文件不存在这条分支被测到了」。
    r = transcribe("", want_timestamps=False)
    assert r["error"] and r["text"] == "", r
    r2 = transcribe("这个文件不存在.wav")
    assert r2["error"] and r2["text"] == "", r2
    print("  transcribe 失败路径        OK")

    # 7) SRT 写出（写到临时文件后删除）
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "t.srt")
        # 空句子必须返回空串且**不建文件**（上面 sents 是第 3 段切出来的 3 句，
        # 时间戳 0-200 / 200-500 / 600-900ms，正好用来核对 SRT 的行格式与序号）
        assert sentences_to_srt([], out) == "", "空句子应返回空串"
        written = sentences_to_srt(sents, out)
        assert written and Path(out).is_file()
        content = Path(out).read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:00,200" in content, content
        assert "00:00:00,600 --> 00:00:00,900" in content, content
        # 首行必须是序号 1（SRT 序号从 1 起，播放器按它排序）
        assert content.startswith("1\n")
    print("  sentences_to_srt          OK")

    print("\n全部自检通过")
    # ⚠️ 这里不要提 `--live`：**本文件没有这个入口**（全文件 ``sys.argv`` 出现 0 次），
    # 照着敲只会静默无反应。真实识别在 `verify_all.py --live` 那一层（ASR 接线检查），
    # 真要转写自己的一段音频则直接调 ``transcribe()``（要密钥 + 音频，会计费）。
    print("（本文件只做离线自检；真实识别请跑 verify_all.py --live 或直接调用 transcribe()）")
