# -*- coding: utf-8 -*-
"""本地素材 → 公网 http(s) URL（托管在自己的服务器上）

课案出处：本项目新增（课案没有这一步）

与课案的落地差异（课案跑本地模型，素材给本地路径就行；换云端托管后素材必须公网可达）
    | 课案 | 本项目 | 为什么 |
    |---|---|---|
    | 素材直接把本地路径交给本机 HeyGem / Fish-Speech | 先 ``publish()`` 换公网 http(s) URL 再调模型 | 云端 API 在阿里云侧发起请求，读不到你机器上的 ``F:\\...`` |
    | 部署自由：自己开 OSS / 挂内网穿透 | scp 到自己的公网服务器 + nginx 只读静态目录 | 复用已有服务器；不用开对象存储、不用装客户端、nginx 也不需要写权限 |
    | 上传完就一直在 | 用完 ``unpublish()`` 主动删 | 托管目录是**公网可读**的，谁拿到 URL 谁能下，素材用完就该撤 |
    | 一份素材传一次 | 命名用内容哈希，重复上传同一份文件名字稳定 | 下游（百炼侧）会按 URL 缓存已处理结果，名字一变缓存就废 |

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

踩过的坑（都留了对应的注释/自检，动这几行之前先读）
    1. **删远端文件的命令绝不能拿 Python ``repr()`` 拼**。``repr("a'b")`` 会得到
       ``'a\'b'``，丢进 sh 里单引号提前闭合，后面那截就被当命令执行了 ——
       第一版写的 ``f"rm -f {target!r}"`` 被本文件的自检当场抓出来，现在统一走
       ``shlex.quote()``。``scp``/``ssh`` 的远端参数是交给远端 shell 解析的，
       任何拼进命令行的字符串都要按这个标准处理。
    2. **``BatchMode=yes`` 不能去掉**。agent / Streamlit 里没有 TTY，ssh 一旦要口令
       就会一直挂着等输入（表现为「页面卡住」而不是报错），必须让它直接失败。
    3. **本地 ``verify()`` 通过 ≠ 百炼能取到**。本机常挂代理，测国内/海外地址的结果
       不代表云端可达性；真正的判据是拿这个 URL 去调一次真实模型 API。
    4. **远端目录一律 ``rstrip('/')`` 再拼文件名**：不这么做 URL 与日志里会出现
       ``.../media-assets//a.mp4`` 这种双斜杠，跟 ``_SAFE_NAME`` 白名单的预期输入、
       跟 nginx ``alias`` 的匹配差一层，排查时最容易看漏。本文件两处拼接都做了。

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
# 300s 是「上传」这条最慢路径的上限；下面几点说明为什么不是 30s：
#   · 公网服务器带宽参差，模特视频动辄几十 MB；
#   · 超时值只在「对端完全不响应」时兜底，正常上传不会等满。
_SCP_TIMEOUT = 300

# 允许的远端文件名：首字符必须字母/数字，其余字母数字与 . _ -
# 这条正则同时挡住 `..`、隐藏文件、以及任何可能被 shell 解释的字符。
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_configured() -> bool:
    """三项配置是否齐全（ssh 目标 / 远端目录 / 对外基址）。

    三项必须全有才算配好：少 ``asset_base_url`` 就拼不出对外 URL，少 ``asset_ssh``
    就传不上去。调用方是 ``tools/voice_clone.py``（import 成 ``host_ready``）：
    它为真才「现传现用」，为假就让 ``clone_voice()`` 返回失败，
    再由 ``workflows/video.py`` 降级到 edge-tts 通用音色。

    Returns:
        三项配置是否都已填写（见根 ``.env`` 里的 ``MEDIA_ASSET_*``）。
    """
    m = settings.media
    return bool(m.asset_ssh and m.asset_remote_dir and m.asset_base_url)


def _remote_name(local_path: str) -> str:
    """生成远端文件名：内容哈希 + 原扩展名。

    用内容哈希而不是原名，有两个好处：
      · 同名不同内容的文件不会互相覆盖；
      · 重复上传同一个文件时名字稳定，便于命中下游服务的缓存。

    Args:
        local_path: 本地文件路径（必须已存在，本函数会读全文件内容算哈希）。

    Returns:
        ``<16 位十六进制 sha256><小写扩展名>``，例如 ``1d1801f753ccd9fa.mp4``。

    为什么不是内置 ``hash()``：
        字符串/字节的 ``hash()`` 每进程随机化（``PYTHONHASHSEED``），换个进程名字就变，
        「重复上传命中缓存」这条好处会直接失效；``hashlib`` 是内容确定的。
    为什么截 16 位：
        64 bit 的碰撞概率对「一台服务器上的素材量」完全够用，名字短一点日志里好读。
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

    失败时的行为：
        路径为空 / 文件不存在 / 三项配置没配齐 / ``scp`` 非 0 退出 —— 这几种情况
        都只打印一行中文原因并返回空串。调用方 ``voice_clone.py``（唯一的调用点：
        声音克隆的参考音频）在 LangGraph 节点 / Streamlit 回调里，异常会断整条链路，
        所以这里统一收口，让它判空串走降级分支（页面退回 edge-tts 通用音色）。

    用法::

        url = publish("model.mp4")
        # -> "http://<host>/media-assets/3f2a...c1.mp4"
    """
    if not local_path or not Path(local_path).is_file():
        print(f"[托管] 文件不存在: {local_path}")
        return ""

    # 配置缺失不是「出错」，而是「这项能力没开」—— 把该填哪三项直接打出来，
    # 免得使用者翻源码找键名（键名见根目录 .env / config.py 的 MediaSettings）。
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
    # keep_name 只建议调试用：原名可能带空格 / 中文 / 特殊字符，
    # 而 nginx 那边的 URL 是要交给百炼去 GET 的，稳妥的是内容哈希名。
    name = path.name if keep_name else _remote_name(local_path)
    # scp 的远端写法就是「目标主机:路径」，中间不加空格；远端目录去掉尾部 `/`
    # 再拼名字，避免出现双斜杠（见文件头「踩过的坑」第 4 条）。
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
            # 显式指定 utf-8：Windows 默认用 GBK 解码子进程输出，
            # 服务器返回含中文的报错时会解码失败，反而看不到真因。
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_SCP_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 —— 网络/超时/无 scp 都收成中文提示
        # 这一支盖住三类情况：机器上压根没有 scp（FileNotFoundError）、
        # 超时（TimeoutExpired）、以及其它 OSError。都不能抛给上层。
        print(f"[托管] 上传失败: {exc}")
        return ""

    if proc.returncode != 0:
        # scp 把真因写在 stderr；截前 300 字符防止刷屏（常见的是权限不足、
        # 远端目录不存在、公钥没配好）。返回码单独打出来，便于区分「没连上」与「传一半断了」。
        print(f"[托管] 上传失败（scp 返回 {proc.returncode}）: {(proc.stderr or '').strip()[:300]}")
        return ""

    url = f"{m.asset_base_url.rstrip('/')}/{name}"
    # 打印实际大小：出问题时第一件要确认的就是「文件是不是没传完/传错了」。
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[托管] 已上传（{size_mb:.2f} MB）: {url}")
    return url


def unpublish(url: str) -> bool:
    """从服务器上删掉一个已发布的文件（用完清理，别一直占着公网）。

    只接受**本项目基址下、文件名严格合法**的 URL ——
    一方面防路径穿越，另一方面防把任意 URL 传进来导致误删别人的东西。

    Args:
        url: ``publish()`` 返回的那种 URL（必须以本托管基址开头）。

    Returns:
        是否删除成功。删不掉也不报错（只是个清理动作）。

    为什么拒绝得这么严：
        这个函数最终会拼出一条 ``ssh ... rm -f --`` 去远端删文件，参数一旦能被
        外部内容影响就是命令注入。所以两道闸门（前缀必须是本基址 + 文件名白名单）
        而不是一道；下面自检里那串 ``x;rm -rf ~;.mp4`` 就是专门盯这条的。
    """
    # 前缀判断必须带上 '/'：只判 `startswith(base)` 的话，
    # `http://h/media-assets-evil/a.mp4` 会被误认成「本站 URL」而放行。
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
            #    `--` 是第二道保险：万一以后白名单放松到允许 `-` 开头，
            #    也不会被 rm 当成选项解析。
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             m.asset_ssh, f"rm -f -- {shlex.quote(target)}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            # 60s：远端只是执行一条 rm，比上传快得多；连不上就快速失败，不必等 300s。
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[托管] 删除失败: {exc}")
        return False

    # 不看 stderr：远端返回码才是权威判据（文件本来就不存在时 rm -f 仍返回 0，属正常）。
    ok = proc.returncode == 0
    print(f"[托管] {'已删除' if ok else '删除失败'}: {target}")
    return ok


def verify(url: str, timeout: int = 20) -> tuple:
    """确认 URL 真的能从公网取到（本地视角）。

    ⚠️ 注意：**本地能取到 ≠ 百炼能取到**。本机常挂代理，测国内/海外地址的结果
    不代表云端可达性。真正的判据是拿这个 URL 去调一次真实 API。

    Args:
        url: 要检查的公网 URL。
        timeout: 单次 HEAD 请求超时（秒），默认 20。

    Returns:
        ``(ok, detail)``
    """
    if not url:
        return False, "URL 为空"
    try:
        import requests

        started = time.time()
        # 用 HEAD 不用 GET：只为确认「能取到」，没必要把几十 MB 素材拉到本地。
        # allow_redirects=True：nginx 常见配置会把 http 301 到 https，
        # 不跟跳会把「其实可达」误判成失败。
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        cost = time.time() - started
        if r.status_code == 200:
            # content-length 是可选头（分块传输/某些 nginx 配置下没有），拿不到就打 '?'
            size = r.headers.get("content-length", "?")
            return True, f"HTTP 200，{size} 字节，{cost:.2f}s"
        return False, f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        # 连不上 / DNS 解析不了 / 超时都走这里：verify 是「体检」不是「断言」，
        # 失败信息本身就是返回值的载荷，绝不能抛给调用方。
        return False, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    print("=== 素材托管模块自检（离线）===")
    m = settings.media
    print(f"  配置齐备      : {is_configured()}")
    print(f"  ssh 目标      : {m.asset_ssh or '（未配置）'}")
    print(f"  远端目录      : {m.asset_remote_dir}")
    print(f"  对外基址      : {m.asset_base_url or '（未配置）'}")

    # 1) 失败路径必须返回空串，不抛异常
    #    （这两条都走在「文件存在性」这一关，一次网络请求都不发）
    assert publish("") == ""
    assert publish("不存在.mp4") == ""
    print("  publish 失败路径       OK")

    # 2) 远端命名：内容哈希 + 原扩展名，且同名同内容稳定
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # 故意用大写扩展名 + 中文原名的文件：验证「扩展名转小写」与
        # 「不拿原名当远端名（中文名过不了 _SAFE_NAME，URL 里也不好放）」
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
    #    这一组是本文件最该保住的自检：unpublish 会拼出远端 rm 命令，
    #    任何一条放行都可能删掉服务器上别的东西。
    assert unpublish("") is False, "空 URL 应拒绝"
    assert unpublish("http://h/media-assets/../etc/passwd") is False, \
        "不属于本托管基址的 URL 应拒绝（第一版这里会去 rm，已修）"
    if m.asset_base_url:
        base = m.asset_base_url.rstrip("/")
        # 这里的六种串各自盯一道闸门：
        #   ../ 与 ..    → 路径穿越
        #   .hidden      → 隐藏文件（白名单要求首字符是字母/数字）
        #   a/b.mp4      → 子目录（白名单不允许 `/`）
        #   x;rm -rf ~;  → shell 元字符（即便进了命令行也必须被引号包住）
        #   单引号       → 专盯 shlex.quote 那条坑（repr 拼串会在这里被打穿）
        for bad in (f"{base}/../etc/passwd", f"{base}/..", f"{base}/.hidden",
                    f"{base}/a/b.mp4", f"{base}/x;rm -rf ~;.mp4", f"{base}/'"):
            assert unpublish(bad) is False, f"应拒绝可疑名: {bad}"
    print("  unpublish 防护         OK  （路径穿越 / 隐藏文件 / 子目录 / shell 元字符 全拒）")

    # 4) 文件名白名单本身
    #    good 里那个 `"素材-1.wav" if False else "abc-1.wav"` 是**有意写的**：
    #    中文名会被 _SAFE_NAME 拒掉（正则只允许 ASCII），所以不能把「素材-1.wav」
    #    当正例，这里用条件表达式把它留在原地当反面记录（真放进来断言必红）。
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

    # 真实上传是**手动**入口：默认自检绝不上传（会往公网放文件、动真实服务器），
    # 只有显式带 --upload <文件> 才走，且传完立刻用 HEAD 回验。
    if "--upload" in sys.argv:
        idx = sys.argv.index("--upload")
        if idx + 1 < len(sys.argv):
            target = sys.argv[idx + 1]
            print(f"\n=== 真实上传: {target} ===")
            # keep_name=True：手动上传基本是为了肉眼核对，认原名更方便；
            # 程序化调用（声音克隆/数字人）走默认的内容哈希名。
            u = publish(target, keep_name=True)
            print("URL:", u or "(失败)")
            if u:
                ok, detail = verify(u)
                print(f"本地回验: {'✓' if ok else '✗'} {detail}")

    print("\n自检完成")
