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

接口（实测自官方文档，2026-09）
    端点: POST {api_base}/services/aigc/multimodal-generation/generation
    基址: settings.dashscope_api_endpoint  （默认 https://dashscope.aliyuncs.com/api/v1）
    请求: {"model":..., "input":{"messages":[{"role":"user","content":[
              {"type":"input_audio","input_audio":{"data":"<URL 或 data:...;base64,xxx>"}}]}]},
           "parameters":{"format":"wav","sample_rate":"16000","language_hints":["zh"]}}
    响应: output.text（全文）/ output.sentence.{begin_time,end_time,text,words[]}
    时间戳单位是**毫秒**，与课案 FunASR 的输出一致，可直接喂课案那套聚合逻辑。

两个必须知道的差异（相对课案）
    1. **音频要以 URL 或 Base64 Data URI 传入**，不能直接传本地路径。
       本模块自动处理：≤10MB 转 Base64 内嵌（连对象存储都不用），
       更大就先经 dashscope_upload 换 ``oss://`` 临时 URL。
       ⚠️ 用 ``oss://`` 时 HTTP 请求必须带 ``X-DashScope-OssResourceResolve: enable``。
    2. **要完整句级时间戳必须走 SSE 流式模式**。
       非流式响应里 ``output.sentence`` 只有一个句子对象（文档示例即如此），
       不足以拼出整段 SRT。所以 ``want_timestamps=True`` 时本模块带
       ``X-DashScope-SSE: enable`` 并**累积所有 event:result 帧**，按 sentence_id 去重。
       （官方说明：SSE 仅在音频 ≥1 分钟时才会分多次返回；短音频即使带了该头
         也可能只回一帧或直接回普通 JSON —— 两种情况本模块都兼容。）
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

# 句级切分用的结束标点（命中即结束当前句）—— 与课案 funasr_ws_server 保持一致
_SENT_END_CHARS = set("，。！？；,.!?;…")

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


# --------------------------------------------------------------------------
# 对外主函数
# --------------------------------------------------------------------------
def transcribe(audio_path: str, want_timestamps: bool = False) -> dict:
    """把音频（或视频）转成文本，可选返回句级时间戳。

    Args:
        audio_path: 本地文件路径、或已经是公网可访问的 http(s) URL。
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

    model = settings.media.asr_model
    try:
        if want_timestamps:
            text, sentences, err = _call_sse(data_uri, model)
        else:
            text, sentences, err = _call_once(data_uri, model)
    except Exception as exc:  # noqa: BLE001 —— 网络/解析异常一律收成中文提示
        err = f"调用百炼语音识别失败: {exc}"
        text, sentences = "", []

    if err:
        result["error"] = err
        print(f"[ASR] {err}")
        return result

    result["text"] = text
    result["sentences"] = sentences
    print(f"[ASR] 识别完成：{len(text)} 字，{len(sentences)} 句")
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
    """一步到位：音频 → SRT 字幕。给视频剪辑链路用（替代课案本地 funasr 生成字幕）。"""
    res = transcribe(audio_path, want_timestamps=True)
    if res["error"]:
        return ""
    return sentences_to_srt(res["sentences"], output_path)


# --------------------------------------------------------------------------
# 内部实现
# --------------------------------------------------------------------------
def _ms_to_srt_time(ms: int) -> str:
    """毫秒 → SRT 时间格式 ``HH:MM:SS,mmm``。"""
    if ms < 0:
        ms = 0
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def _to_audio_input(audio_path: str) -> tuple[str, str]:
    """把输入整理成百炼能接受的 ``data`` 字段：公网 URL 原样返回，本地文件转 Base64 或临时 URL。

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

    # 超过内嵌上限：先传百炼临时存储换 oss:// URL
    print(f"[ASR] 音频 {size / 1024 / 1024:.2f} MB 超过 {limit / 1024 / 1024:.0f} MB，"
          f"改走百炼临时存储")
    from tools.dashscope_upload import upload_file

    oss_url = upload_file(str(path), settings.media.asr_model)
    if not oss_url:
        return "", "音频过大且上传临时存储失败（见上方日志）"
    return oss_url, ""


def _headers(sse: bool, data: str) -> dict:
    """构造请求头。

    ``X-DashScope-OssResourceResolve`` 只在传 ``oss://`` 时必须，
    但它对普通 URL / Base64 也无害，所以统一带上、少一个分支。
    """
    return {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "enable" if sse else "disable",
        "X-DashScope-OssResourceResolve": "enable",
    }


def _payload(data: str) -> dict:
    """构造请求体。语言提示取配置里的一项（Fun-ASR 系列只认第一个值）。"""
    params = {
        "format": "wav",
        "sample_rate": "16000",
    }
    lang = (settings.media.asr_language or "").strip()
    if lang:
        params["language_hints"] = [lang]

    return {
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


def _call_once(data: str, model: str) -> tuple[str, list, str]:
    """非流式调用：只要纯文本时走这条（最快、最省）。"""
    url = f"{settings.dashscope_api_endpoint}/services/aigc/multimodal-generation/generation"
    resp = requests.post(
        url, headers=_headers(sse=False, data=data), json=_payload(data), timeout=180
    )
    if resp.status_code != 200:
        return "", [], f"HTTP {resp.status_code}: {resp.text[:300]}"

    body = resp.json()
    output = body.get("output") or {}
    text = str(output.get("text") or "").strip()
    sentences = []
    seg = _sentence_from(output.get("sentence"))
    if seg:
        sentences.append(seg)
    if not text and not sentences:
        return "", [], f"响应里没有识别结果: {json.dumps(body, ensure_ascii=False)[:300]}"
    return text, sentences, ""


def _call_sse(data: str, model: str) -> tuple[str, list, str]:
    """SSE 流式调用：累积所有帧，拼出完整句级时间戳列表。

    官方只在音频 ≥1 分钟时才分多帧返回；短音频可能只回一帧、
    甚至无视 SSE 头直接回普通 JSON —— 两种都兼容。
    """
    url = f"{settings.dashscope_api_endpoint}/services/aigc/multimodal-generation/generation"
    resp = requests.post(
        url,
        headers=_headers(sse=True, data=data),
        json=_payload(data),
        timeout=600,
        stream=True,
    )
    if resp.status_code != 200:
        return "", [], f"HTTP {resp.status_code}: {resp.text[:300]}"

    content_type = (resp.headers.get("content-type") or "").lower()

    # 情况一：服务端没按 SSE 回，直接给了普通 JSON
    if "event-stream" not in content_type:
        body = resp.json()
        output = body.get("output") or {}
        text = str(output.get("text") or "").strip()
        seg = _sentence_from(output.get("sentence"))
        return text, ([seg] if seg else []), ""

    # 情况二：真·SSE，累积 event:result 帧里的 data:
    by_id: dict = {}
    order: list = []
    full_text = ""

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk:
            continue
        try:
            frame = json.loads(chunk)
        except json.JSONDecodeError:
            continue

        output = frame.get("output") or {}
        if output.get("text"):
            full_text = str(output["text"])

        node = output.get("sentence") or {}
        seg = _sentence_from(node)
        if not seg:
            continue

        # 同一句在识别过程中会被多次返回（sentence_end=False 表示还会变），
        # 用 sentence_id 去重、后到的覆盖先到的；只有 sentence_end=True 才是最终结果。
        sid = node.get("sentence_id")
        key = sid if sid is not None else f"_anon_{len(order)}"
        if key not in by_id:
            order.append(key)
        if node.get("sentence_end", True):
            by_id[key] = seg
        else:
            by_id.setdefault(key, seg)

    sentences = [by_id[k] for k in order if by_id.get(k)]
    sentences.sort(key=lambda s: s["start"])

    if not full_text:
        full_text = "".join(s["text"] for s in sentences)

    if not full_text and not sentences:
        return "", [], "SSE 流里没有解析到任何识别结果"
    return full_text.strip(), sentences, ""


def _sentence_from(node) -> dict | None:
    """把百炼返回的 sentence 对象转成统一格式；字段不全时返回 None。"""
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

    移植自课案 ``deploy/funasr_ws_server.py`` 的 ``_split_sentences()``：
    那版是针对「逐字 + 词级 timestamp 一一对应」写的；百炼这里给的是
    「多字词 + 每词自己的 begin/end + 该词之后的标点」，所以改成按词推进，
    语义完全一致：**遇到结束标点就收束为一句**。

    只在 ``output.sentence`` 缺失、但 ``words`` 可用时作为兜底使用。
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
        if not token.strip():
            continue
        if cur_start is None and word.get("begin_time") is not None:
            cur_start = word["begin_time"]
        if word.get("end_time") is not None:
            cur_end = word["end_time"]

        cur_text.append(token)
        punct = str(word.get("punctuation") or "")
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

    # 3) 词级 → 句级聚合（移植自课案的标点切句逻辑）
    #    注意：课案的结束标点集合**包含中文逗号**，所以「你好，」本身就算一句 ——
    #    这是刻意的，短视频字幕讲究短句一行，长句会被画面裁掉。
    words = [
        {"text": "你好", "begin_time": 0, "end_time": 200, "punctuation": "，"},
        {"text": "世界", "begin_time": 200, "end_time": 500, "punctuation": "。"},
        {"text": "第二句", "begin_time": 600, "end_time": 900, "punctuation": "！"},
    ]
    sents = words_to_sentences(words)
    assert len(sents) == 3, f"逗号也算句末，应切成 3 句，实际 {len(sents)}"
    assert sents[0]["text"] == "你好，", sents[0]["text"]
    assert (sents[0]["start"], sents[0]["end"]) == (0, 200)
    assert sents[1]["text"] == "世界。" and (sents[1]["start"], sents[1]["end"]) == (200, 500)
    assert sents[2]["text"] == "第二句！" and (sents[2]["start"], sents[2]["end"]) == (600, 900)
    print("  words_to_sentences        OK")

    # 4) 无标点时整段兜底为一句
    sents2 = words_to_sentences([{"text": "没有标点", "begin_time": 0, "end_time": 100}])
    assert len(sents2) == 1 and sents2[0]["text"] == "没有标点"
    print("  words_to_sentences(无边标) OK")

    # 5) 失败路径必须返回结构化结果而不是抛异常
    r = transcribe("", want_timestamps=False)
    assert r["error"] and r["text"] == "", r
    r2 = transcribe("这个文件不存在.wav")
    assert r2["error"] and r2["text"] == "", r2
    print("  transcribe 失败路径        OK")

    # 6) SRT 写出（写到临时文件后删除）
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
