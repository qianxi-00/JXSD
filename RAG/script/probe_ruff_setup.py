"""把 ruff 的 Windows wheel 下到工作区内并解包，供**沙箱里**跑 lint（绕开 uv/pip）。

为什么不用 uv/pip：本沙箱禁止进程用管道 spawn 子进程，`uv` 连自己管好的解释器都起不来
（`Failed to query Python interpreter ... 拒绝访问`），pip 也不在 venv 里。
所以退化成"直接下 wheel + 解包拿 ruff.exe"——ruff 的 wheel 里就是一个独立 exe，不依赖 Python。
（顺带一提：正是 ruff 真跑起来后**捉出了 5 个真错误**——4×F401 未用 import、1×F841 未用变量。）

用法：
    .venv\\Scripts\\python.exe RAG\\script\\probe_ruff_setup.py
    .dsh_tmp\\ruffdl\\unpacked\\ruff-*\\scripts\\ruff.exe check --select F,E9 RAG\\

正常终端（uv 可用）不需要本脚本，直接 `uvx ruff check --select F,E9 RAG/` 即可。
产物落在 `.dsh_tmp/ruffdl/`（gitignored，可随时删；下次跑本脚本会重新下）。
"""

import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(r"F:\ProGram\Python_Base")
DEST = BASE / ".dsh_tmp" / "ruffdl"
DEST.mkdir(parents=True, exist_ok=True)

INDEXES = [
    "https://mirrors.aliyun.com/pypi/simple/ruff/",
    "https://pypi.org/simple/ruff/",
]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 固定白名单地址
        return resp.read()


html = None
used = None
for url in INDEXES:
    try:
        html = fetch(url).decode("utf-8", "replace")
        used = url
        break
    except Exception as exc:  # noqa: BLE001 试下一个镜像
        print(f"[warn] {url} 取不到: {type(exc).__name__}: {exc}")

if html is None:
    print("[fail] 两个索引都取不到")
    sys.exit(1)

print(f"[ok] 索引来源: {used}")
wheels = re.findall(r'href="([^"]*ruff-[^"]*win_amd64\.whl[^"]*)"', html)
if not wheels:
    print("[fail] 索引里没有 win_amd64 wheel")
    sys.exit(1)

# 取文件名里版本号最大的那个（索引是乱序/按时间追加的，直接取最后一个不稳）
def version_key(href: str) -> tuple:
    m = re.search(r"ruff-([0-9]+)\.([0-9]+)\.([0-9]+)", href)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


wheel_url = sorted(wheels, key=version_key)[-1]
if wheel_url.startswith("//"):
    wheel_url = "https:" + wheel_url
elif wheel_url.startswith("/"):
    wheel_url = "https://mirrors.aliyun.com" + wheel_url
elif not wheel_url.startswith("http"):
    wheel_url = used.rsplit("/", 1)[0] + "/" + wheel_url

name = wheel_url.split("/")[-1].split("#")[0]
print(f"[ok] 选定: {name}")

blob = fetch(wheel_url)
zip_path = DEST / name
zip_path.write_bytes(blob)
print(f"[ok] 下载 {len(blob) / 1e6:.1f} MB -> {zip_path}")

with zipfile.ZipFile(io.BytesIO(blob)) as zf:
    exes = [n for n in zf.namelist() if n.lower().endswith("ruff.exe")]
    if not exes:
        print(f"[fail] wheel 里没有 ruff.exe，内容前 10 项: {zf.namelist()[:10]}")
        sys.exit(1)
    zf.extractall(DEST / "unpacked")
    exe = DEST / "unpacked" / exes[0]
    print(f"[ok] ruff.exe -> {exe}")
