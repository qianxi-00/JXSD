# -*- coding: utf-8 -*-
"""本地文件 → 公网可访问 URL（百炼免费临时存储）

课案出处：本项目新增（课案没有这一步）

为什么需要它
    课案用本地 HeyGem / Fish-Speech 时，素材是「本地路径」直接交给本机服务。
    改成百炼托管 API 后，多个接口都**只收公网可访问的 URL**：
        · 爱诗 PixVerse 视频对口型 —— video_url / audio_url 必须是 URL
        · CosyVoice 声音复刻   —— create_voice(url=...) 必须是公网 URL
    阿里云百炼为此提供了**免费临时存储**：上传本地文件，换回一个 ``oss://`` 前缀的
    临时 URL（有效期 48 小时），调用模型时把这个 URL 传进去即可。
    这样就不需要自己开 OSS、也不用内网穿透。

实测确认（本机 dashscope 1.27.4）
    ``dashscope.utils.oss_utils.OssUtils.upload(model=..., file_path=..., api_key=...)``
    返回 ``(oss_url, 上传凭证信息)``，就是 CLI ``dashscope oss upload`` 内部用的同一个方法。
    注意 ``dashscope.oss`` 这个模块**不存在**（只有 ``dashscope.cli.oss`` 这个命令行入口），
    别按文档写成 ``dashscope.oss.upload``。

三个必须知道的限制（踩了才知道）
    1. **文件与模型绑定**：上传时必须指定 ``model``，且后续调用必须用**同一个模型**。
       所以这里的 ``upload_file(path, model)`` 强制要求传模型名，不给默认值 —— 
       传错模型名不会报错，只会在真正调用模型时报「文件不可用」。
    2. **48 小时过期**：过期后文件被清理，URL 失效。不要把它当长期存储。
    3. **不可查询/下载**：上传后只能通过 URL 在模型调用里用，拿不回来。

谁在用它 / 谁故意不用（用的两处都要求「上传时用的 model == 调用时用的 model」）
    · ``tools/audio_transcriber.py`` —— 只在本机音频超过 ``MEDIA_ASR_INLINE_MAX_BYTES``
      （默认 10MB，转 Base64 内嵌塞不下）时才进这条分支，model 传 ``settings.media.asr_model``。
    · ``tools/avatar_client.py`` —— 模特视频与音频两段素材都先传这里换 URL，model 传 PixVerse 那个。
    · ⚠️ ``tools/voice_clone.py`` **故意绕开它**：CosyVoice 声音复刻明确不收 ``oss://``
      （实测 400 ``audio url should start with http or https``），那条链路改走
      ``tools/asset_host.py`` 的真 http(s) 托管。别为了「统一」把这里换回去。
"""

import sys
from pathlib import Path

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def is_configured() -> bool:
    """百炼密钥是否已配置（临时存储也走同一个密钥）。

    调用方（``audio_transcriber`` / ``avatar_client``）可以先问一句再决定走不走上传，
    免得走进来才发现没有密钥、白跑一趟文件检查。

    Returns:
        是否配置了 ``DASHSCOPE_API_KEY``。
    """
    return bool(settings.dashscope_api_key)


def upload_file(file_path: str, model: str) -> str:
    """把本地文件上传到百炼临时存储，返回 ``oss://`` 形式的临时 URL。

    Args:
        file_path: 本地文件路径。
        model: **这个文件将要用于哪个模型**。必须与后续调用的模型完全一致，例如
            ``"pixverse/pixverse-lipsync"`` 或 ``"qwen-audio-3.0-asr-flash"``。

    Returns:
        ``oss://dashscope-instant/...`` 形式的 URL；失败返回空字符串。

    注意：
        HTTP 直连调用该 URL 时**必须**带请求头 ``X-DashScope-OssResourceResolve: enable``，
        否则服务端不认识 ``oss://`` 前缀。用 dashscope SDK 调用则由 SDK 自动加。
        本项目的 ``avatar_client`` / ``audio_transcriber`` 走 HTTP，已经带上了这个头。

    失败时的行为（所有分支一致）：
        打印一行中文原因 + **返回空串**，绝不抛异常。调用它的两条链路分别在 LangGraph
        节点与 Streamlit 回调里，异常会把整条图/整个页面带崩；所以调用方只需要判空串，
        然后走自己的降级分支（ASR 报「音频过大且上传失败」、数字人退回内置 TTS）。
        失败包括：文件不存在 / 没配密钥 / 没传 model / 上传抛异常 /
        **返回形态异常**（不是本机这版 SDK 的 2 元组）/ 返回的不是 ``oss://``。
    """
    # 三类前置校验都收成「打印 + 返回空串」。刻意保持「文件 → 密钥 → model」这个顺序：
    # 先确认文件真的存在，免得把「路径写错」误报成「密钥没配」而查错方向。
    if not file_path or not Path(file_path).is_file():
        print(f"[临时存储] 文件不存在: {file_path}")
        return ""

    if not is_configured():
        print("[临时存储] 未配置 DASHSCOPE_API_KEY，无法上传（见根目录 .env）")
        return ""

    # model 没有默认值，也不允许空 —— 文件与模型绑定，传错不会在这里报错，
    # 只会在真正调模型时报「文件不可用」（见文件头「三个必须知道的限制」第 1 条）。
    if not model:
        print("[临时存储] 必须指定 model —— 上传的文件与模型绑定，二者不一致时调用会失败")
        return ""

    try:
        # 函数内延迟导入：dashscope SDK 只有「传大文件」这一条路径用得到，
        # 没必要让模块 import 期就依赖它；老版本 dashscope 没有 utils.oss_utils 时，
        # 受影响的也只是这条分支，不会把 ASR / 数字人的其它功能一起带崩。
        from dashscope.utils.oss_utils import OssUtils

        result = OssUtils.upload(
            model=model,
            file_path=str(file_path),
            api_key=settings.dashscope_api_key,
        )
    except Exception as exc:  # noqa: BLE001 —— 上传失败不应该让整条链路崩掉
        # OssUtils.upload 失败时是 raise（1.27.4 源码里抛 UploadFileException，
        # 响应非 200 时带上服务端报错文本），这里统一收成中文提示 + 空串。
        print(f"[临时存储] 上传失败: {exc}")
        return ""

    # 形状校验（**这是防御，不是兼容分支**）：本机 dashscope 1.27.4 实测返回 2 元组
    # `(oss_url, 上传凭证信息)`，源码里就是
    # `return "oss://" + form_data["key"], upload_info` —— 所以早先那个
    # `elif isinstance(result, str)` 在这版 SDK 上**永远走不到**（读起来像死代码）。
    # 现在只校验「长度恰为 2 的序列」：形状不对（SDK 升级改成返回 dict / 单串 / None）
    # 就判失败 —— 在这里报出来，好过把它当 URL 一路带到模型调用那一步才报「文件不可用」。
    if not (isinstance(result, (tuple, list)) and len(result) == 2):
        # ⚠️ 这里**只打形状（类型名 + 元素个数），绝不打值本身**：本机 SDK 那个元组的
        # 第 2 项就是**上传凭证**（`dashscope/utils/oss_utils.py` 里含
        # `oss_access_key_id` / `signature` / `policy`），原先 `{result!r}` 写进日志
        # 等于把临时凭证摊到控制台与 CI 日志里。定位「是不是 SDK 改了返回形态」只需
        # 知道「期望 2 元组、实际是什么形状」，不需要值。
        _shape = (
            f"{type(result).__name__}，长度 {len(result)}"
            if isinstance(result, (tuple, list))
            else type(result).__name__
        )
        print("[临时存储] upload 返回形态异常（期望 2 元组 (oss_url, 凭证)，实际 "
              f"{_shape}）：疑似 SDK 变更")
        return ""
    oss_url = str(result[0] or "")

    # 形态兜底：拿到个非 oss 的字符串就当失败。放过去的话，下游会把它当普通公网 URL
    # 直接调模型，报错要等到那一步才出现，日志里看不到「上传其实没成功」这个根因。
    if not oss_url.startswith("oss://"):
        print(f"[临时存储] 返回的不是 oss:// URL，视为失败: {oss_url!r}")
        return ""

    # 日志里带上大小与用途模型：临时存储「48 小时过期 + 与 model 绑定」，
    # 48 小时后出问题，这两项是唯一能回溯原因的线索。
    size_mb = Path(file_path).stat().st_size / 1024 / 1024
    print(
        f"[临时存储] 上传成功（{size_mb:.1f} MB，用途模型={model}，有效期 48 小时）"
    )
    return oss_url


if __name__ == "__main__":
    # 自检：只验证「不崩」与参数校验分支，不依赖网络与密钥
    # （真实上传会把文件送上去、消耗账号额度，所以这里一次都不发请求。
    #   `verify_all.py` 第 1 层跑的就是本段，必须是纯离线、秒回。）
    print("=== 临时存储模块自检 ===")
    print(f"密钥已配置: {is_configured()}")
    # 下面三条各自打在一条校验分支上，且刻意用「能过前一条、卡在后一条」的入参，
    # 这样一旦校验顺序被改动，断言会立刻发现：
    #   · 空路径                    → 卡在「文件存在」检查
    #   · __file__（真实存在）+ 空 model → 能过文件检查，卡在「model 必填」
    #   · 不存在的路径 + 合法 model  → 卡在「文件存在」检查
    assert upload_file("", "x") == "", "空路径应返回空串"
    assert upload_file(__file__, "") == "", "空 model 应返回空串"
    assert upload_file("不存在的文件.txt", "x") == "", "不存在的文件应返回空串"
    print("参数校验分支全部正确返回空串，未抛异常 —— OK")

    # 形态异常分支**只许打形状，不许打值**：本机 SDK 那个元组的第 2 项就是上传凭证
    # （`oss_access_key_id` / `signature` / `policy`），一旦 `repr` 进日志，凭证就跟着
    # 控制台输出、CI 日志一起外泄。这里打桩一个「含假凭证的 3 元组」，把 stdout 抓下来
    # 断言那串假凭证**不出现** —— 去掉脱敏（改回 `{result!r}`）这条必红。
    import contextlib
    import io
    import types as _types

    _MARKER = "MARKER-ACCESS-KEY"
    _fake_oss_utils = _types.ModuleType("dashscope.utils.oss_utils")

    class _FakeOssUtils:
        @staticmethod
        def upload(model=None, file_path=None, api_key=None):
            # 形状照本机 SDK 的 3 元组故意给错，第 2 项塞假凭证
            return ("oss://dashscope-instant/fake", {_MARKER: _MARKER},
                    {"signature": _MARKER})

    _fake_oss_utils.OssUtils = _FakeOssUtils
    # `upload_file()` 是在函数体里 `from dashscope.utils.oss_utils import OssUtils`，
    # 所以往 `sys.modules` 塞个假模块就能顶掉它（导入器先查 sys.modules，不再找父包），
    # 全程离线、不发请求、不计费。
    _real_oss_utils = sys.modules.get("dashscope.utils.oss_utils")
    _real_is_configured = is_configured
    _buf = io.StringIO()
    try:
        sys.modules["dashscope.utils.oss_utils"] = _fake_oss_utils
        is_configured = lambda: True  # noqa: E731 —— 自检里顶掉模块全局名
        with contextlib.redirect_stdout(_buf):
            _got = upload_file(__file__, "pixverse/pixverse-lipsync")
    finally:
        is_configured = _real_is_configured
        if _real_oss_utils is None:
            sys.modules.pop("dashscope.utils.oss_utils", None)
        else:
            sys.modules["dashscope.utils.oss_utils"] = _real_oss_utils

    _log = _buf.getvalue()
    assert _got == "", f"形态异常必须判失败并返回空串，实际 {_got!r}"
    assert "形态异常" in _log, _log
    # 形状信息必须留下（否则这个分支等于没报，排查 SDK 变更时看不到线索）
    assert "tuple" in _log and "长度 3" in _log, _log
    # 关键断言：凭证字符串一个字符都不许出现在日志里
    assert _MARKER not in _log, f"日志泄漏了上传凭证: {_log}"
    print("  形态异常只打形状           OK  类型/长度留下，凭证未进日志")
