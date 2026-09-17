# -*- coding: utf-8 -*-
"""本地素材 → 公网 http(s) URL（托管在自己的服务器上）

课案出处：本项目新增（课案没有这一步）

为什么必须要有它
    ``tools/dashscope_upload.py`` 那套「百炼免费临时存储」拿到的是 ``oss://`` URL，
    而它**不是所有模型都吃**：

        · ASR（Qwen-Audio-ASR）—— 不走它也行（我们走 Base64 Data URI，已实测可用）
        · **CosyVoice 声音复刻 —— 明确不收**，实测 400：
              InvalidParameter: audio url should start with http or https
        · PixVerse 视频对口型 —— 官方要求公网 URL，是否吃 ``oss://`` 尚未实测

    所以「真正的 http(s) 托管」是声音克隆（以及可能的数字人）的硬前提。

本项目采用的方式（按用户指定）
    用一台自己的公网服务器 + nginx 静态目录：
        本地 --scp--> <user>@<host>:<remote_dir>   （nginx 以只读方式分发）
        对外 URL = <base_url>/<文件名>

    nginx 侧只需要一个只读 ``location``：::

        location /media-assets/ {
            alias /var/www/media-assets/;
            autoindex off;
        }

    上传走 scp（不是 HTTP PUT），所以 nginx 那边**不需要任何写权限**。

安全说明
    · 文件上传后是**公网可读**的 —— 别往这里放私密素材，用完可以 ``unpublish()``。
    · 依赖 ``ssh`` / ``scp`` 在 PATH 里，且已配好免密（本项目用的是
      ``~/.ssh/config`` 里的既有条目或默认私钥）。
    · 全程不带口令参数，不把任何凭据写进代码或日志。
"""

import hashlib
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# scp/ssh 的超时（秒）。素材通常几 MB，给足余量但不至于挂死。
_SCP_TIMEOUT = 300

# 允许的远端文件名：首字符必须字母/数字，其余字母数字与 . _ -
# 这条正则同时挡住 `..`、隐藏文件、以及任何可能被 shell 解释的字符。
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_configured() -> bool:
    """三项配置是否齐全（ssh 目标 / 远端目录 / 对外基址）。"""
    m = settings.media
    return bool(m.asset_ssh and m.asset_remote_dir and m.asset_base_url)


def _remote_name(local_path: str) -> str:
    """生成远端文件名：内容哈希 + 原扩展名。

    用内容哈希而不是原名，有两个好处：
      · 同名不同内容的文件不会互相覆盖；
      · 重复上传同一个文件时名字稳定，便于命中下游服务的缓存。
    """
    path = Path(local_path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"{sha}{path.suffix.lower()}"


def publish(local_path: str, keep_name: bool = False) -> str:
    """把本地文件传到公网服务器，返回可直接给外部 API 用的 http(s) URL。

    Args:
        local_path: 本地文件路径。
        keep_name: True 时保留原文件名（调试时好认），默认用内容哈希命名。

    Returns:
        公网 URL；**失败返回空字符串**（不抛异常）。

    用法::

        url = publish("model.mp4")
        # -> "http://<host>/media-assets/3f2a...c1.mp4"
    """
    if not local_path or not Path(local_path).is_file():
        print(f"[托管] 文件不存在: {local_path}")
        return ""

    if not is_configured():
        print(
            "[托管] 未配置。请在根 .env 里填：\n"
            "        MEDIA_ASSET_SSH=ubuntu@<你的服务器>\n"
            "        MEDIA_ASSET_REMOTE_DIR=/var/www/media-assets\n"
            "        MEDIA_ASSET_BASE_URL=http://<你的服务器>/media-assets"
        )
        return ""

    m = settings.media
    path = Path(local_path)
    name = path.name if keep_name else _remote_name(local_path)
    remote = f"{m.asset_ssh}:{m.asset_remote_dir.rstrip('/')}/{name}"

    try:
        proc = subprocess.run(
            [
                "scp",
                "-o", "BatchMode=yes",          # 绝不交互提问（没有 TTY）
                "-o", "ConnectTimeout=15",
                str(path),
                remote,
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_SCP_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 —— 网络/超时/无 scp 都收成中文提示
        print(f"[托管] 上传失败: {exc}")
        return ""

    if proc.returncode != 0:
        print(f"[托管] 上传失败（scp 返回 {proc.returncode}）: {(proc.stderr or '').strip()[:300]}")
        return ""

    url = f"{m.asset_base_url.rstrip('/')}/{name}"
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[托管] 已上传（{size_mb:.2f} MB）: {url}")
    return url


def unpublish(url: str) -> bool:
    """从服务器上删掉一个已发布的文件（用完清理，别一直占着公网）。

    只接受**本项目基址下、文件名严格合法**的 URL ——
    一方面防路径穿越，另一方面防把任意 URL 传进来导致误删别人的东西。

    Returns:
        是否删除成功。删不掉也不报错（只是个清理动作）。
    """
    base = (settings.media.asset_base_url or "").rstrip("/")
    if not url or not base or not url.startswith(base + "/"):
        print(f"[托管] 拒绝删除：URL 不属于本托管基址（{base or '未配置'}）")
        return False

    name = url[len(base) + 1:]
    # 严格白名单：不含 `/`、不以 `.` 开头、只允许 [A-Za-z0-9._-]
    if not _SAFE_NAME.match(name):
        print(f"[托管] 拒绝删除可疑文件名: {name!r}")
        return False

    m = settings.media
    target = f"{m.asset_remote_dir.rstrip('/')}/{name}"
    try:
        proc = subprocess.run(
            # ⚠️ 必须用 shlex.quote 做 shell 引用 ——
            #    Python 的 repr() **不是 shell 安全的**：含单引号的字符串被 repr 成
            #    'a\'b' 后，在 sh 里单引号会提前闭合，内容被当命令执行。
            #    第一版就是写的 f"rm -f {target!r}"，被自检抓出来了。
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             m.asset_ssh, f"rm -f -- {shlex.quote(target)}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[托管] 删除失败: {exc}")
        return False

    ok = proc.returncode == 0
    print(f"[托管] {'已删除' if ok else '删除失败'}: {target}")
    return ok


def verify(url: str, timeout: int = 20) -> tuple:
    """确认 URL 真的能从公网取到（本地视角）。

    ⚠️ 注意：**本地能取到 ≠ 百炼能取到**。本机常挂代理，测国内/海外地址的结果
    不代表云端可达性。真正的判据是拿这个 URL 去调一次真实 API。

    Returns:
        ``(ok, detail)``
    """
    if not url:
        return False, "URL 为空"
    try:
        import requests

        started = time.time()
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        cost = time.time() - started
        if r.status_code == 200:
            size = r.headers.get("content-length", "?")
            return True, f"HTTP 200，{size} 字节，{cost:.2f}s"
        return False, f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    print("=== 素材托管模块自检（离线）===")
    m = settings.media
    print(f"  配置齐备      : {is_configured()}")
    print(f"  ssh 目标      : {m.asset_ssh or '（未配置）'}")
    print(f"  远端目录      : {m.asset_remote_dir}")
    print(f"  对外基址      : {m.asset_base_url or '（未配置）'}")

    # 1) 失败路径必须返回空串，不抛异常
    assert publish("") == ""
    assert publish("不存在.mp4") == ""
    print("  publish 失败路径       OK")

    # 2) 远端命名：内容哈希 + 原扩展名，且同名同内容稳定
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        f1 = Path(td) / "素材.MP4"
        f1.write_bytes(b"x" * 2048)
        n1 = _remote_name(str(f1))
        n2 = _remote_name(str(f1))
        assert n1 == n2, "同一文件命名应稳定"
        assert n1.endswith(".mp4"), f"扩展名应保留并小写，实际 {n1}"
        assert len(n1) == 16 + 4, f"应为 16 位哈希 + 扩展名，实际 {n1!r}"

        f2 = Path(td) / "另一个.mp4"
        f2.write_bytes(b"y" * 2048)
        assert _remote_name(str(f2)) != n1, "不同内容应得到不同名字"
        print(f"  _remote_name           OK  {n1}")

    # 3) unpublish 的防护（这几个分支都必须是 False，且**不能真的去删**）
    assert unpublish("") is False, "空 URL 应拒绝"
    assert unpublish("http://h/media-assets/../etc/passwd") is False, \
        "不属于本托管基址的 URL 应拒绝（第一版这里会去 rm，已修）"
    if m.asset_base_url:
        base = m.asset_base_url.rstrip("/")
        for bad in (f"{base}/../etc/passwd", f"{base}/..", f"{base}/.hidden",
                    f"{base}/a/b.mp4", f"{base}/x;rm -rf ~;.mp4", f"{base}/'"):
            assert unpublish(bad) is False, f"应拒绝可疑名: {bad}"
    print("  unpublish 防护         OK  （路径穿越 / 隐藏文件 / 子目录 / shell 元字符 全拒）")

    # 4) 文件名白名单本身
    for good in ("a.mp4", "1d1801f753ccd9fa.mp4", "素材-1.wav" if False else "abc-1.wav"):
        assert _SAFE_NAME.match(good), good
    for bad in ("..", ".", ".hidden", "a/b", "a b", "a;b", "a'b", "a$b", "", "-x"):
        assert not _SAFE_NAME.match(bad), f"不该通过: {bad!r}"
    print("  _SAFE_NAME 白名单      OK")

    if is_configured():
        print("\n  ⚠️ 已配置托管 —— 真实上传请跑：")
        print("     python tools/asset_host.py --upload <本地文件>")
    else:
        print("\n  未配置托管，跳过联网部分")

    if "--upload" in sys.argv:
        idx = sys.argv.index("--upload")
        if idx + 1 < len(sys.argv):
            target = sys.argv[idx + 1]
            print(f"\n=== 真实上传: {target} ===")
            u = publish(target, keep_name=True)
            print("URL:", u or "(失败)")
            if u:
                ok, detail = verify(u)
                print(f"本地回验: {'✓' if ok else '✗'} {detail}")

    print("\n自检完成")
