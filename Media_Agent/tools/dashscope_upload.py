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
"""

import sys
from pathlib import Path

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def is_configured() -> bool:
    """百炼密钥是否已配置（临时存储也走同一个密钥）。"""
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
    """
    if not file_path or not Path(file_path).is_file():
        print(f"[临时存储] 文件不存在: {file_path}")
        return ""

    if not is_configured():
        print("[临时存储] 未配置 DASHSCOPE_API_KEY，无法上传（见根目录 .env）")
        return ""

    if not model:
        print("[临时存储] 必须指定 model —— 上传的文件与模型绑定，二者不一致时调用会失败")
        return ""

    try:
        from dashscope.utils.oss_utils import OssUtils

        result = OssUtils.upload(
            model=model,
            file_path=str(file_path),
            api_key=settings.dashscope_api_key,
        )
    except Exception as exc:  # noqa: BLE001 —— 上传失败不应该让整条链路崩掉
        print(f"[临时存储] 上传失败: {exc}")
        return ""

    # 实测返回 (oss_url, certificate)；这里对两种形态都做兼容，
    # 免得 dashscope 后续版本改成只返回字符串时静默失效。
    oss_url = ""
    if isinstance(result, (tuple, list)) and result:
        oss_url = str(result[0] or "")
    elif isinstance(result, str):
        oss_url = result

    if not oss_url.startswith("oss://"):
        print(f"[临时存储] 返回的不是 oss:// URL，视为失败: {oss_url!r}")
        return ""

    size_mb = Path(file_path).stat().st_size / 1024 / 1024
    print(
        f"[临时存储] 上传成功（{size_mb:.1f} MB，用途模型={model}，有效期 48 小时）"
    )
    return oss_url


if __name__ == "__main__":
    # 自检：只验证「不崩」与参数校验分支，不依赖网络与密钥
    print("=== 临时存储模块自检 ===")
    print(f"密钥已配置: {is_configured()}")
    assert upload_file("", "x") == "", "空路径应返回空串"
    assert upload_file(__file__, "") == "", "空 model 应返回空串"
    assert upload_file("不存在的文件.txt", "x") == "", "不存在的文件应返回空串"
    print("参数校验分支全部正确返回空串，未抛异常 —— OK")
