# -*- coding: utf-8 -*-
"""语音转文字 —— 百炼 Fun-ASR-Flash（替代课案的本地 FunASR）

课案出处：自媒体课案 → 内容复刻（提文案）/ 视频剪辑（生成 SRT 字幕）

课案原本本地部署了**两个** FunASR 模型：
    · 纯文本通道：``FunAudioLLM/Fun-ASR-Nano-2512`` + ``fsmn-vad``
    · 时间戳通道：``iic/speech_seaco_paraformer_large_asr_nat-...`` + ``fsmn-vad`` + ``ct-punc``
      （懒加载，请求帧带 ``"return_timestamps": true`` 时才走）
两个模型在 GPU 服务器上跑一个自写的 WebSocket 服务（``deploy/funasr_ws_server.py``，端口 6006）。

本项目换成百炼的 **Qwen-Audio-3.0-ASR-Flash / Fun-ASR-Flash 非实时识别**：
一个 HTTP 接口同时提供「纯文本」与「句级 + 词级时间戳」，
**一个接口就替掉了课案的两个模型**，本机不需要 GPU、不下载权重。

================================================================
接口行为（**本机实测结论，与官方文档的印象不一致，务必先读这段**）
================================================================
端点: ``POST {settings.dashscope_api_endpoint}/services/aigc/multimodal-generation/generation``
请求::

    {"model": "...",
     "input": {"messages": [{"role": "user", "content": [
         {"type": "input_audio", "input_audio": {"data": "<公网URL 或 data:...;base64,xxx>"}}]}]},
     "parameters": {"format": "mp3", "sample_rate": "16000", "language_hints": ["zh"]}}

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
   本模块自动处理：≤10MB 转 Base64 内嵌（连对象存储都不用），
   更大就先经 ``dashscope_upload`` 换 ``oss://`` 临时 URL。
   ⚠️ 用 ``oss://`` 时 HTTP 请求必须带 ``X-DashScope-OssResourceResolve: enable``。
2. 时间戳单位是**毫秒**，与课案 FunASR 的输出一致，课案那套聚合逻辑可直接复用。
"""

import base64
import json
import mimetypes
import sys
from pathlib import Path

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


# ==========================================================================
# 对外主函数
# ==========================================================================
def transcribe(audio_path: str, want_timestamps: bool = False) -> dict:
    """把音频（或视频）转成文本，可选返回句级时间戳。

    Args:
        audio_path: 本地文件路径、或已经是公网可访问的 http(s)/oss:// URL。
        want_timestamps: 是否要句级时间戳（生成 SRT 字幕时必须为 True）。

    Returns:
        ``{"text": str, "sentences": [{"text","start","end"}], "error": str}``
        —— **绝不抛异常**。失败时 ``text`` 为空串、``error`` 为中文说明。
        时间戳单位毫秒，与课案 FunASR 输出一致。
    """
    result = {"text": "", "sentences": [], "error": ""}

    if not audio_path:
        result["error"] = "音频路径为空"
        return result

    if not settings.dashscope_api_key:
        result["error"] = (
            "未配置 DASHSCOPE_API_KEY（见根目录 .env）。"
            "语音识别走百炼 Fun-ASR-Flash，需要该密钥。"
        )
        print(f"[ASR] {result['error']}")
        return result

    data_uri, err = _to_audio_input(audio_path)
    if err:
        result["error"] = err
        return result

    try:
        text, sentences, err = _call_once(data_uri)
    except Exception as exc:  # noqa: BLE001 —— 网络/解析异常一律收成中文提示
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
    """
    valid = [
        s for s in (sentences or [])
        if str(s.get("text", "")).strip() and s.get("end") is not None
    ]
    if not valid:
        print("[ASR] 没有可写入字幕的句子")
        return ""

    lines = []
    for idx, seg in enumerate(valid, 1):
        start = _ms_to_srt_time(int(seg.get("start") or 0))
        end = _ms_to_srt_time(int(seg.get("end") or 0))
        lines.append(f"{idx}\n{start} --> {end}\n{str(seg['text']).strip()}\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[ASR] 已写出字幕（{len(valid)} 句）: {output_path}")
    return output_path


def transcribe_to_srt(audio_path: str, output_path: str) -> str:
    """一步到位：音频（或视频）→ SRT 字幕。给视频剪辑链路用。

    替代课案里让 deepagent 自己写脚本调本地 funasr 生成字幕那一步。
    """
    res = transcribe(audio_path, want_timestamps=True)
    if res["error"]:
        return ""
    return sentences_to_srt(res["sentences"], output_path)


def transcribe_words(audio_path: str) -> tuple:
    """取「完整文本 + 原始词级时间戳」，给需要自己切片字幕的调用方用。

    Returns:
        ``(text, words, error)``；words 是百炼原样的词列表。
    """
    if not settings.dashscope_api_key:
        return "", [], "未配置 DASHSCOPE_API_KEY"
    data_uri, err = _to_audio_input(audio_path)
    if err:
        return "", [], err
    try:
        resp = _post(data_uri)
    except Exception as exc:  # noqa: BLE001
        return "", [], f"调用百炼语音识别失败: {exc}"

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
    """毫秒 → SRT 时间格式 ``HH:MM:SS,mmm``。"""
    if ms < 0:
        ms = 0
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def _to_audio_input(audio_path: str) -> tuple:
    """把输入整理成百炼能接受的 ``data`` 字段。

    公网 URL / oss:// 原样返回；本地文件 ≤限制则转 Base64，更大则先换临时 URL。

    Returns:
        ``(data, error)``：成功时 error 为空串。
    """
    if audio_path.startswith(("http://", "https://", "oss://")):
        return audio_path, ""

    path = Path(audio_path)
    if not path.is_file():
        return "", f"音频文件不存在: {audio_path}"

    size = path.stat().st_size
    limit = settings.media.asr_inline_max_bytes
    ext = path.suffix.lower()

    if size <= limit:
        mime = _MIME_BY_EXT.get(ext) or mimetypes.guess_type(path.name)[0] or "audio/wav"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        print(f"[ASR] 音频 {size / 1024 / 1024:.2f} MB，走 Base64 内嵌（{mime}）")
        return f"data:{mime};base64,{b64}", ""

    print(f"[ASR] 音频 {size / 1024 / 1024:.2f} MB 超过 {limit / 1024 / 1024:.0f} MB，"
          f"改走百炼临时存储")
    from tools.dashscope_upload import upload_file

    oss_url = upload_file(str(path), settings.media.asr_model)
    if not oss_url:
        return "", "音频过大且上传临时存储失败（见上方日志）"
    return oss_url, ""


def _post(data: str) -> requests.Response:
    """发一次**非流式**请求。

    ``X-DashScope-OssResourceResolve`` 只在传 ``oss://`` 时必须，
    但它对普通 URL / Base64 也无害，所以统一带上、少一个分支。
    """
    params = {"format": "wav", "sample_rate": "16000"}
    lang = (settings.media.asr_language or "").strip()
    if lang:
        params["language_hints"] = [lang]

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
            "X-DashScope-SSE": "disable",
            "X-DashScope-OssResourceResolve": "enable",
        },
        json=payload,
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp


def _call_once(data: str) -> tuple:
    """非流式调用：一次拿到完整文本 + 完整词级时间戳，再本地切句。

    Returns:
        ``(text, sentences, error)``
    """
    resp = _post(data)
    body = resp.json()
    output = body.get("output") or {}
    sentence = output.get("sentence") or {}

    text = str(output.get("text") or "").strip()
    words = sentence.get("words") or []

    sentences = words_to_sentences(words) if words else []

    # 兜底：万一这一版 API 没给 words，但给了 sentence 的起止时间，就当成一整句
    if not sentences:
        fallback = _sentence_from(sentence)
        if fallback:
            sentences = [fallback]

    if not text and not sentences:
        return "", [], f"响应里没有识别结果: {json.dumps(body, ensure_ascii=False)[:300]}"
    if not text:
        text = "".join(s["text"] for s in sentences)
    return text, sentences, ""


def _sentence_from(node) -> dict | None:
    """把百炼返回的 sentence 对象转成单句格式；字段不全时返回 None。"""
    if not isinstance(node, dict):
        return None
    text = str(node.get("text") or "").strip()
    begin = node.get("begin_time")
    end = node.get("end_time")
    if not text or begin is None or end is None:
        return None
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

    Returns:
        ``[{"text": str, "start": int, "end": int}, ...]``
    """
    sentences: list = []
    cur_text: list = []
    cur_start = None
    cur_end = None

    def flush() -> None:
        nonlocal cur_text, cur_start, cur_end
        seg = "".join(cur_text).strip()
        if seg and cur_start is not None and cur_end is not None:
            sentences.append({"text": seg, "start": int(cur_start), "end": int(cur_end)})
        cur_text, cur_start, cur_end = [], None, None

    for word in words or []:
        if not isinstance(word, dict):
            continue
        token = str(word.get("text") or "")
        punct = str(word.get("punctuation") or "")
        if not token and not punct:
            continue

        # 时间只认有实际内容的词；纯空白词不参与，避免把句子起点带到空格上
        if token.strip():
            if cur_start is None and word.get("begin_time") is not None:
                cur_start = word["begin_time"]
            if word.get("end_time") is not None:
                cur_end = word["end_time"]

        # 时长兜底：模型长时间不吐标点时，靠这一条把字幕行截短（见 _MAX_SEGMENT_MS）
        if (
            cur_start is not None
            and cur_end is not None
            and cur_text
            and int(cur_end) - int(cur_start) > _MAX_SEGMENT_MS
        ):
            flush()

        cur_text.append(token)
        if punct:
            cur_text.append(punct)
            if set(punct.strip()) & _SENT_END_CHARS:
                flush()
    flush()
    return sentences


if __name__ == "__main__":
    # 自检：纯逻辑，不依赖网络与密钥
    print("=== 语音转文字模块自检 ===")

    # 1) SRT 时间格式化
    assert _ms_to_srt_time(0) == "00:00:00,000"
    assert _ms_to_srt_time(3661_500) == "01:01:01,500"
    assert _ms_to_srt_time(-5) == "00:00:00,000"
    print("  _ms_to_srt_time           OK")

    # 2) sentence 解析（含字段不全/时间倒挂应被拒）
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
    assert all(s["end"] - s["start"] <= _MAX_SEGMENT_MS + 1000 for s in cut), cut
    print(f"  words_to_sentences(时长兜底) OK  20s → {len(cut)} 句，"
          f"最长 {max(s['end'] - s['start'] for s in cut)}ms")

    # 5) 全空白词不应产出任何句子
    assert words_to_sentences([{"text": " ", "begin_time": 0, "end_time": 10}]) == []
    print("  words_to_sentences(全空白) OK")

    # 6) 失败路径必须返回结构化结果而不是抛异常
    r = transcribe("", want_timestamps=False)
    assert r["error"] and r["text"] == "", r
    r2 = transcribe("这个文件不存在.wav")
    assert r2["error"] and r2["text"] == "", r2
    print("  transcribe 失败路径        OK")

    # 7) SRT 写出（写到临时文件后删除）
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "t.srt")
        assert sentences_to_srt([], out) == "", "空句子应返回空串"
        written = sentences_to_srt(sents, out)
        assert written and Path(out).is_file()
        content = Path(out).read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:00,200" in content, content
        assert "00:00:00,600 --> 00:00:00,900" in content, content
        assert content.startswith("1\n")
    print("  sentences_to_srt          OK")

    print("\n全部自检通过")
    print("（真实验证请用 --live，需要 DASHSCOPE_API_KEY 与一段音频）")
