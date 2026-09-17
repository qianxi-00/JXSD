# -*- coding: utf-8 -*-
"""数据复盘页面 —— 抖音主页链接 → 真实作品数据 → AI 多维诊断

课案出处：自媒体课案 → 数据复盘 → ``views/review.py``

页面结构（与课案一致）
    🍪 Cookie 配置区 → 🔗 主页链接 → 「获取数据并分析」按钮
    → 📊 数据概览卡片 → 📋 作品明细 → 三个 tab（漏斗诊断 / 内容评估 / 优化策略）
    → 📥 下载完整 Markdown 报告

与课案的落地差异
    | 项 | 课案 | 本项目 |
    |---|---|---|
    | Cookie 存取 | 读写本地 douyin 项目的 ``C:/Users/13261/.../config/settings.json``（硬编码） | 读 ``settings.media.douyin_cookie``；页面里粘的 Cookie 存**本次运行时覆写**（``os.environ[MEDIA_DOUYIN_COOKIE]``），因为契约不允许改 ``config.py`` / ``.env`` |
    | Cookie 自动获取 | ``_read_edge_cookies()`` 走 CDP 读 Edge | 同思路（v20 之后本地 Cookies 库解不开，走浏览器自己的 API 才是正解），但**必须连页级目标** —— ``Network`` 是页级域，课案连的 browser 级目标上没这个方法 |
    | 采集触发 | 本地爬虫 | 自托管 REST 服务；**服务不可达时页面自动提示降级入口** |
    | 降级入口 | 无 | 新增「📋 手动粘贴作品数据」文本框 → ``run_review_from_json()``，跑同一套三节点诊断 |
    | 作品明细 | 无播放量时 5 列卡片 | 同（沿用课案的 ``has_plays`` 判断） |
    | 操作历史 | 每次渲染都 append（点一次下载就多一条） | 加 token 去重，同一份结果只记一条 |

踩过的坑
    · **结果必须先存 ``st.session_state`` 再渲染** —— ``st.download_button`` 被点击会
      触发整页 rerun，不缓存结果的话页面会当场变空白（课案反复强调的坑）。
    · ``_add_history`` 如果直接在渲染路径里调用，下载按钮的每次 rerun 都会重复记录，
      所以用 ``review_history_token`` 去重。
    · **CDP 取 Cookie 必须连页级目标**：``/json/version`` 给的是 **browser 级**地址，
      在它上面调 ``Network.getCookies`` 会回 ``-32601 "'Network.getCookies' wasn't found"``
      （本机 Edge 153 实测），Cookie 恒为空 —— 已登录的用户也会被告知「请先登录」。
      正确做法是从 ``/json/list`` 挑 ``type == "page"`` 的目标。
    · **握手必须去掉 Origin 头**：``websocket-client`` 默认带一个从 URL 推出的 ``Origin``，
      而 Edge 对带 Origin 的 CDP WebSocket 一律回 ``403 Forbidden``（同机实测）。
      连页级目标解决了方法不存在的问题，但光换目标仍会卡在这一步 —— 两处都得对
      （见 ``_cdp_connect()``）。
    · 直接 ``python views/review.py`` 时 ``sys.path[0]`` 是 ``views/`` 目录，
      所以路径引导必须在文件顶部、项目内 import 之前。

运行方式
    页面模式（正常用法，由 main.py 侧边栏路由调用）::

        Set-Location F:\\ProGram\\Python_Base\\Media_Agent
        & ..\\.venv\\Scripts\\python.exe -m streamlit run main.py

    离线自检模式（不启服务、不联网，走文件末尾的 ``__main__`` 块）::

        Set-Location F:\\ProGram\\Python_Base\\Media_Agent
        & ..\\.venv\\Scripts\\python.exe views\\review.py
"""

import sys
from pathlib import Path

# ---- 路径引导：必须在 import 项目内模块（tools/*、workflows/*）之前执行 ----
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from datetime import datetime  # noqa: E402

import streamlit as st  # noqa: E402

from tools.douyin_client import (  # noqa: E402
    COOKIE_ENV_KEY,
    SAMPLE_MANUAL_JSON,
    api_base,
    cookie_hint,
    is_available,
    resolve_cookie,
)
# 这六个名字都来自 tools/douyin_client：页面只做展示与转发，
# 采集、Cookie 解析、手动数据解析的逻辑全在那边（单测也落在那边）

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 课案原样的 Edge cookies 路径（只用于定位，不直接读库 —— v20 解不开）
EDGE_COOKIES_DIR = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data\Default\Network")
EDGE_COOKIES_FILE = os.path.join(EDGE_COOKIES_DIR, "Cookies")


# ============================================================
# 一、Cookie：从 Edge 读取（CDP）+ 运行时覆写
# ============================================================

def _port_alive(port: int, timeout: float = 2.0) -> bool:
    """Edge 的 CDP 调试端口是否已经开着。

    Args:
        port: 要探测的本地端口。
        timeout: 单次探测的超时（秒）。默认 2.0 的依据：本机回环连接要么立刻成功、
            要么被立刻拒绝，2 秒只是为了兜住「端口被防火墙丢包」这种最坏情况 ——
            再长会让 ``_find_debug_port()`` 一次扫 4 个端口时明显变慢。

    Returns:
        True 表示 ``/json/version`` 有响应（说明是带调试端口的 Chromium 系浏览器）。
    """
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=timeout)
        return True
    except Exception:
        return False


def _find_debug_port() -> int | None:
    """在常见调试端口里找一个已经在跑的 Edge。

    端口列表（9222~9225）来自课案原文：9222 是 Chromium 系浏览器的默认调试端口，
    其余几个是历史上被占用或多开时的备选。四个都没在监听就返回 None，
    由调用方决定要不要自己起一个。
    """
    for port in (9222, 9223, 9224, 9225):
        if _port_alive(port):
            return port
    return None


def _launch_edge(port: int):
    """起一个带调试端口的无头 Edge；返回 Popen 句柄（失败返回 None）。

    先 taskkill 掉后台 msedge，否则用户数据目录被锁、新实例会直接复用旧进程而
    不监听调试端口 —— 课案踩过的坑，保留同样处理。
    """
    for proc_name in ("msedge.exe", "msedgewebview2.exe"):
        try:
            # 必须先杀干净：Edge 用同一个 --user-data-dir 时会**复用已有进程**，
            # 新实例不会监听调试端口（课案踩过的坑）。msedgewebview2.exe 也要杀 ——
            # 它同样占着那个用户数据目录
            subprocess.run(["taskkill", "/f", "/im", proc_name],
                           capture_output=True, timeout=10)
        except Exception:
            pass
    # 等进程真正退出、文件锁释放：taskkill 是异步的，
    # 不睡这一下就可能紧接着启动失败（拿不到用户数据目录）
    time.sleep(1.5)

    user_data = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data")
    # 两个安装路径都试：Edge 一般装在 x86 那个目录，但 64 位安装包会在另一个
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for exe in candidates:
        if not os.path.exists(exe):
            continue
        try:
            proc = subprocess.Popen(
                # --headless=new 是新版无头模式：不弹窗口打扰用户，
                # 且能正常监听 CDP 端口（启动参数与课案一致）
                [exe, f"--remote-debugging-port={port}",
                 f"--user-data-dir={user_data}", "--headless=new"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[复盘] 启动 Edge 失败({exe}): {exc}")
            continue
        # 最多等 7.5 秒（15 × 0.5）：Edge 冷启动一般 2~3 秒，
        # 上限既够用又不会让页面按钮卡太久
        for _ in range(15):
            time.sleep(0.5)
            if _port_alive(port):
                return proc
        return proc   # 起了但没监听成功，交给调用方判断并清理
    return None


# 抖音的三个域名：主站、一级域、以及分享/移动域。
# 一次全传给 ``Network.getCookies`` 的 ``urls``，不再「一个域名问一次」——
# ``domain`` 不是 CDP 的合法参数，传了会被静默忽略，那三次调用拿到的其实是同一批。
_DOUYIN_COOKIE_URLS = (
    "https://www.douyin.com/",
    "https://douyin.com/",
    "https://www.iesdouyin.com/",
)


def _cdp_json(debug_port: int, path: str, timeout: float = 5):
    """读一个 CDP 的 HTTP 端点（``/json/version``、``/json/list``）并解析成 JSON。"""
    resp = urllib.request.urlopen(f"http://127.0.0.1:{debug_port}{path}", timeout=timeout)
    return json.loads(resp.read().decode())


def _cdp_connect(ws_url: str, timeout: float = 10):
    """连 CDP 的 WebSocket —— **必须**带 ``suppress_origin=True``，否则现代 Edge 回 403。

    websocket-client 默认会按 URL 造一个 ``Origin`` 头，而 Chromium/Edge 对**带 Origin** 的
    CDP WebSocket 连接一律拒绝（本机 Edge 153 + websocket-client 1.9.0 实测）::

        Handshake status 403 Forbidden
        Rejected an incoming WebSocket connection from the http://127.0.0.1:9223 origin.

    为什么在客户端去掉 Origin、而不是给 Edge 加 ``--remote-allow-origins``：
    ``_find_debug_port()`` 的设计意图就是**复用用户自己已经开着的**那个带调试端口的 Edge
    （只有它带着登录态），而那种进程没法事后补命令行开关。本地工具不需要 Origin，
    Chromium 对**没有** Origin 的连接是放行的 —— 客户端这一改两种情况都成立。

    Args:
        ws_url: 目标（页级或 browser 级）的 ``webSocketDebuggerUrl``。
        timeout: 握手超时（秒）。

    Note:
        ``suppress_origin`` 不在 ``create_connection()`` 的具名签名里（1.9.0 的签名是
        ``(url, timeout=None, class_=WebSocket, **options)``），它经 ``**options`` 传给
        ``WebSocket``；实测生效，别照签名把它当无效参数删掉。
    """
    from websocket import create_connection

    return create_connection(ws_url, timeout=timeout, suppress_origin=True)


def _pick_page_target(targets: list) -> str:
    """从 ``/json/list`` 的目标列表里挑一个**页级**目标的 WebSocket 调试地址。

    为什么不能用 ``/json/version`` 返回的那个地址：那是 **browser 级**目标，
    而 ``Network`` 是页级域 —— 连上去调 ``Network.getCookies`` 会直接回
    ``-32601 "'Network.getCookies' wasn't found"``（本机 Edge 153 实测），
    一个 Cookie 都取不到，最终表现成「已经登录的用户被告知请先登录」。

    优先挑 url 里含 ``douyin`` 的页面（通常是用户刚登录过的那个标签页），
    没有再退化成第一个页级目标 —— Cookie 罐是**浏览器级**的，跟当前页面停在哪个
    站点无关：实测页面停在 ``edge://sync-confirmation-dialog/`` 时照样能取到
    抖音的 ttwid / sessionid_probe。

    Args:
        targets: ``/json/list`` 返回的 JSON 数组。

    Returns:
        ``webSocketDebuggerUrl``；没有「带地址的 ``type == "page"`` 目标」时返回空串。
    """
    pages = [
        t for t in targets or []
        if t.get("type") == "page" and t.get("webSocketDebuggerUrl")
    ]
    if not pages:
        return ""
    for target in pages:
        if "douyin" in str(target.get("url", "")).lower():
            return target["webSocketDebuggerUrl"]
    return pages[0]["webSocketDebuggerUrl"]


def _create_blank_page_target(debug_port: int) -> str:
    """一个页级目标都没有时，用 browser 级连接造一个 ``about:blank`` 页并返回它的地址。

    ``Target`` 域在 browser 级目标是**可用**的（本机实测 ``Target.getTargets`` /
    ``Target.createTarget`` 都正常），所以「无头 Edge 起完一个 page 都没有」这种
    边界情况还能救回来；救不回来（拿不到 browser 地址 / 建不出目标）就返回空串，
    由调用方给一句明确的中文错误。

    新目标要重新列一次 ``/json/list`` 才会带 ``webSocketDebuggerUrl``，而且刚建好的
    一瞬间可能还没注册完，所以这里轮询几次而不是只查一遍。
    """
    try:
        browser_ws = _cdp_json(debug_port, "/json/version").get("webSocketDebuggerUrl", "")
        if not browser_ws:
            return ""
        ws = _cdp_connect(browser_ws)
        try:
            ws.send(json.dumps({
                "id": 1,
                "method": "Target.createTarget",
                "params": {"url": "about:blank"},
            }))
            ws.recv()
        finally:
            ws.close()
    except Exception:  # noqa: BLE001 —— 兜底路径失败就返回空串，由调用方统一给中文错误
        return ""

    # 最多等 2 秒（4 × 0.5）：目标是本机进程内建的，正常一次就能列到
    for _ in range(4):
        ws_url = _pick_page_target(_cdp_json(debug_port, "/json/list"))
        if ws_url:
            return ws_url
        time.sleep(0.5)
    return ""


def _read_edge_cookies() -> tuple[str, "str | None"]:
    """通过 CDP（Chrome DevTools Protocol）从 Edge 直接读取 douyin.com Cookie。

    课案原样保留这套做法：**走浏览器自己的 Network.getCookies 接口**，
    自动跨过 Edge 的 v20 加密与文件锁定，不需要解密本地 Cookies 数据库。

    与课案的差异：必须连**页级**目标（``/json/list`` 里 ``type == "page"`` 的那个），
    不能用 ``/json/version`` 给的 browser 级地址 —— ``Network`` 是页级域，
    在 browser 级目标上这个方法根本不存在，Cookie 恒为空。

    Returns:
        ``(cookie_string, error_message)`` —— 成功时 error 为 None。
    """
    debug_port = _find_debug_port()
    edge_proc = None
    launched = False

    if debug_port is None:
        # 已经有端口就直接用（可能是别人手动开着、带着已登录会话的 Edge）；
        # 都没有才自己起一个，固定挑 9223 —— `_find_debug_port()` 刚扫过 9222~9225
        # 全都没在监听，挑哪个都一样，这个值是课案原文
        debug_port = 9223
        edge_proc = _launch_edge(debug_port)
        launched = edge_proc is not None
        # 三种失败各给各的中文指引，页面直接把这段文本 st.error 出来
        if not launched:
            return "", "无法启动 Edge（未找到 msedge.exe），请确认 Edge 已安装"
        if not _port_alive(debug_port):
            return "", (
                f"Edge 已启动但调试端口 {debug_port} 未就绪。\n"
                "请手动关掉所有 Edge 窗口后重试。"
            )

    try:
        # 存在性检查：真正的连接走 `_cdp_connect()`（那里统一带 suppress_origin=True）；
        # 缺库时给的是「装什么」，而不是让它退化成一个含义模糊的握手失败
        try:
            import websocket  # noqa: F401
        except ImportError:
            return "", "缺少 websocket-client 库，请执行: uv add websocket-client"

        # Step 1: 拿**页级**目标的 WebSocket 端点。
        # ⚠️ 不能用 /json/version 里的那个：那是 browser 级目标，而 `Network` 是页级域，
        #    在它上面调 Network.getCookies 会回 -32601（本机 Edge 153 实测）——
        #    Cookie 恒为空，已登录的用户也会被下面那条文案劝去登录。
        ws_url = _pick_page_target(_cdp_json(debug_port, "/json/list"))
        if not ws_url:
            # 兜底：无头 Edge 可能一个 page 目标都没有。browser 级目标上 `Target.*`
            # 是可用的，所以还能现造一个 about:blank 页出来救场。
            ws_url = _create_blank_page_target(debug_port)
        if not ws_url:
            return "", (
                "Edge 里没有可用的页级调试目标（连新建 about:blank 页也失败了）。\n"
                "请手动关掉所有 Edge 窗口后重试。"
            )

        # Step 2: 连上去，按 URL 问 Cookie（`_cdp_connect` 负责去掉 Origin 头，见那里的说明）
        ws = _cdp_connect(ws_url)
        try:
            # `urls` 一次传齐三个抖音域名：Network.getCookies 只认 `urls`（`domain`
            # 会被静默忽略），而且这个参数是**真的在过滤** —— 反向对照传 example.com
            # 实测返回 0 个。Cookie 罐是浏览器级的，跟当前页面停在哪个站点无关。
            # CDP 是「发一条收一条」的同步协议，所以三条 URL 也只发一帧，固定 "id": 1
            ws.send(json.dumps({
                "id": 1,
                "method": "Network.getCookies",
                "params": {"urls": list(_DOUYIN_COOKIE_URLS)},
            }))
            payload = json.loads(ws.recv())
            all_cookies = payload.get("result", {}).get("cookies", []) or []
        finally:
            ws.close()

        if not all_cookies:
            # 库里没有抖音 Cookie = 用户没在 Edge 里登录过：
            # 错误文案直接给出下一步动作（先登录，再回来点这个按钮）。
            # 修掉「连错目标」那个 bug 之后，这条分支才是**真的只在没登录时**成立。
            return "", (
                "Edge 中未找到 douyin.com 的 Cookie。\n"
                "请先在 Edge 浏览器里打开 douyin.com 并登录，再回来点这个按钮。"
            )

        # 去重后拼成请求头要的字符串：同一个 Cookie 名在不同 domain/path 下
        # 可能有多份，而 HTTP 请求头里同名 Cookie 只能出现一次 —— 取先出现的那个
        seen, parts = set(), []
        for cookie in all_cookies:
            name = cookie.get("name")
            if name and name not in seen:
                seen.add(name)
                parts.append(f"{name}={cookie.get('value', '')}")
        return "; ".join(parts), None

    except Exception as exc:  # noqa: BLE001 —— 页面按钮不能因为读 Cookie 失败而崩
        return "", f"CDP 读取 Cookie 失败: {exc}"

    finally:
        # Step 3: 清理我们自己起的 Edge（别人起的窗口不动 ——
        # 杀掉用户手动开的窗口会丢他的登录会话）
        if launched and edge_proc is not None:
            try:
                edge_proc.terminate()
                edge_proc.wait(timeout=5)
            except Exception:
                try:
                    edge_proc.kill()
                except Exception:
                    pass


def _save_douyin_cookie(cookie: str) -> None:
    """把 Cookie 存成**本次运行时的覆写**。

    为什么不是写文件：课案写的是本地 douyin 项目的 ``settings.json``，本项目已经
    没有那个项目了；写回根 ``.env`` 又超出本模块的写入范围（别人正在同时改）。
    所以走 ``os.environ[MEDIA_DOUYIN_COOKIE]`` ——
    ``tools/douyin_client.resolve_cookie()`` 会在**每次调用时**读它，立即生效，
    且重启进程即失效（不会留下一个忘了改的持久凭据）。
    """
    cookie = (cookie or "").strip()
    # 两个位置都要写：环境变量给 tools/douyin_client.resolve_cookie() 每次调用时读
    # （所以立刻生效），session_state 只用来让页面知道「现在处于运行时覆写状态」
    os.environ[COOKIE_ENV_KEY] = cookie
    st.session_state["douyin_cookie_runtime"] = cookie


def _clear_douyin_cookie() -> None:
    """丢掉运行时覆写，回到根 .env 的配置。

    这是「本次会话粘贴/抓取的 Cookie」唯一的撤销入口，
    两个位置一起清才不会出现「环境变量清了、页面还显示已覆写」的半截状态。
    """
    os.environ.pop(COOKIE_ENV_KEY, None)
    st.session_state.pop("douyin_cookie_runtime", None)


@st.cache_data(ttl=15, show_spinner=False)
def _service_available() -> bool:
    """探活（缓存 15 秒）。

    为什么不每次渲染都探：下载按钮触发的每一次 rerun 都会重跑整页，
    服务不可达时那是实打实的超时等待。

    ttl 取 15 秒的依据：用户看到状态不对、手动再点一次按钮的间隔通常就是十几秒，
    这个值够让「服务刚起来」立刻反映到页面上，又不会每轮 rerun 都去打一次网络；
    ``show_spinner=False`` 是为了不让每次 rerun 都闪一下加载动画。
    """
    return is_available()


# ============================================================
# 二、页面小工具
# ============================================================

def _add_history(action: str, summary: str) -> None:
    """往 ``st.session_state.history`` 追加一条（首页展示最近 10 条）。

    这个函数本身不做去重 —— 去重由调用方 ``_render_result()`` 用
    ``review_history_token`` 负责（见那里的注释）。
    """
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        # `or ""` 是防御：摘要可能是 None（直接切片会抛 TypeError）；
        # 200 字符的上限对应首页那一行的展示宽度
        "summary": (summary or "")[:200],
    })


def _load_items(data_str: str) -> list:
    """把工作流返回的 ``video_data``（JSON 字符串）还原成列表。

    图 state 里只能放字符串，所以 ``video_data`` 是 ``json.dumps`` 过的字符串，
    页面要先还原成 ``list[dict]`` 才能求和、才能交给 ``st.dataframe``。

    Returns:
        ``list``；输入是空串 / None / 坏 JSON / 合法但非数组的 JSON（``{}`` / ``null`` /
        数字）时一律返回 ``[]`` —— 让调用方走「没有数据」分支，而不是在求和时炸掉。
    """
    try:
        # `data_str or "[]"` 同时兜住 None 与空串；TypeError 兜的正是 None 那种输入
        items = json.loads(data_str or "[]")
    except (ValueError, TypeError):
        return []
    return items if isinstance(items, list) else []


def _build_report(items: list, url: str, funnel: str, content: str, suggestions: str) -> str:
    """拼出可下载的 Markdown 复盘报告（纯函数，方便离线自检）。

    版式与课案一致：概览表 → 作品明细表 → 三段分析。
    """
    total_likes = sum(i.get("点赞", 0) for i in items)
    total_comments = sum(i.get("评论", 0) for i in items)
    total_shares = sum(i.get("分享", 0) for i in items)
    total_collects = sum(i.get("收藏", 0) for i in items)
    total_plays = sum(i.get("播放", 0) for i in items)
    # 播放量不是每个来源都有（非本人主页的公开数据通常拿不到），
    # 所以「有没有播放量」决定概览表里那一行在不在
    has_plays = total_plays > 0

    report = f"""# 📊 抖音数据复盘报告

> 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
> 分析链接：{url}
> 作品数量：{len(items)} 条

---

## 📈 总体数据概览

| 指标 | 数值 |
|------|------|
| 作品数 | {len(items)} |
"""
    if has_plays:
        # 有播放量才加这一行：写「| 👀 总播放 | 0 |」会让人以为播放量是 0，
        # 实际是「这个来源拿不到」
        report += f"| 👀 总播放 | {total_plays:,} |\n"
    report += (
        f"| ❤️ 总点赞 | {total_likes:,} |\n"
        f"| 💬 总评论 | {total_comments:,} |\n"
        f"| 🔁 总分享 | {total_shares:,} |\n"
        f"| ⭐ 总收藏 | {total_collects:,} |\n"
    )

    report += "\n---\n\n## 📋 作品数据明细\n\n"
    if items:
        # 用**第一条**作品的键当表头：不同来源的字段集合可能不一样，
        # 固定取第一条能保证列名稳定，缺字段的行留空（见下面的 item.get(k, "")）
        keys = list(items[0].keys())
        report += "| " + " | ".join(keys) + " |\n"
        report += "|" + "|".join(["------"] * len(keys)) + "|\n"
        for item in items:
            report += "| " + " | ".join(str(item.get(k, "")) for k in keys) + " |\n"

    # 三段分析原样拼入：失败时它们是工作流回填的中文提示串，
    # 也照原样进报告 —— 报告要如实反映「这次跑出了什么」
    report += f"\n---\n\n## 📉 漏斗诊断\n\n{funnel}\n"
    report += f"\n---\n\n## 📝 内容评估\n\n{content}\n"
    report += f"\n---\n\n## 💡 优化策略\n\n{suggestions}\n"
    return report


# ============================================================
# 三、页面
# ============================================================

def _render_cookie_panel() -> None:
    """🍪 Cookie 配置区（课案结构，落点从 settings.json 换成运行时覆写）。

    页面上的控件与动作
        · ``st.button``「🔍 从 Edge 浏览器获取」→ ``_read_edge_cookies()``
          （走 CDP 读 Edge 里已登录的抖音 Cookie）；成功则 ``_save_douyin_cookie()``
          并 rerun，失败把中文原因直接 ``st.error`` 出来；
        · ``st.text_area``「或手动粘贴 Cookie 字符串」+ ``st.button``「💾 使用这份 Cookie」
          —— 粘贴为空时只出黄条，不覆盖已有的配置；
        · ``st.button``「♻️ 恢复 .env 配置」→ ``_clear_douyin_cookie()``。

    落点：``os.environ["MEDIA_DOUYIN_COOKIE"]``（``resolve_cookie()`` 每次调用时读它）
    + ``st.session_state["douyin_cookie_runtime"]``。只在本次进程内生效，
    重启回到根 ``.env`` 的配置。
    """
    # 默认收起：Cookie 是偶尔才需要动的配置，不该每次都占满首屏
    with st.expander("🍪 抖音 Cookie 配置", expanded=False):
        st.caption("两种方式：① 点按钮自动从 Edge 读取　② 手动复制粘贴。"
                   "本次会话内生效，重启后回到根目录 .env 的配置。")
        # ⚠️ 必须在按钮**旁边**先说清楚这个副作用：`_launch_edge()` 在
        # 9222~9225 都没有监听时会 `taskkill /f /im msedge.exe` 再自己起一个 ——
        # 也就是**会关掉用户当前打开的所有 Edge 窗口**（课案原有做法，为了拿到
        # 那个带登录态的 user-data-dir；用户自己开着的 Edge 是补不了
        # --remote-debugging-port 的）。标签页通常能被 Edge 恢复，但不该让人
        # 点完才发现，所以这里明写。
        st.caption("⚠️ 点这个按钮时，若检测不到已开启的调试端口，程序会**重启 Edge**"
                   "（你当前打开的所有 Edge 窗口会被关掉，标签页一般可由 Edge 自行恢复）。"
                   "不想被打扰时请改用下面的手动粘贴。")

        # 1:3 的宽度比：左边按钮窄，右边那行提示文字长
        col_auto, col_tip = st.columns([1, 3])
        with col_auto:
            if st.button("🔍 从 Edge 浏览器获取", width="stretch",
                         help="自动读取 Edge 里已登录的抖音 Cookie（走 CDP，不需要解密）"):
                # 只有点下去才会去找/启动 Edge（``_launch_edge()`` 会 taskkill 掉
                # 现有 msedge 进程），不做任何后台自动探测
                cookie_str, err = _read_edge_cookies()
                if err:
                    st.error(err)
                else:
                    _save_douyin_cookie(cookie_str)
                    st.success(f"✅ 已从 Edge 获取 Cookie（{len(cookie_str)} 字符）")
                    st.rerun()
        with col_tip:
            st.caption(cookie_hint())

        cookie_input = st.text_area(
            "或手动粘贴 Cookie 字符串",
            # value="" 是**初值**，不从 .env 带出来：那份由下面的「♻️ 恢复 .env 配置」
            # 负责，不往输入框里回填整条凭据；带 key 是为了让控件在 rerun 之间稳定，
            # 用户粘到一半不会被清掉
            value="",
            placeholder="从浏览器 F12 → Application → Cookies → 复制完整 Cookie...",
            height=100,
            key="douyin_cookie_input",
        )
        col_save, col_clear = st.columns([1, 1])
        with col_save:
            if st.button("💾 使用这份 Cookie", width="stretch"):
                if not cookie_input.strip():
                    # 空输入只提醒、不写入：否则点一下就把已经配好的覆写清掉了
                    st.warning("请先粘贴 Cookie")
                else:
                    _save_douyin_cookie(cookie_input)
                    st.success("Cookie 已生效（仅本次会话）")
                    st.rerun()
        with col_clear:
            if st.button("♻️ 恢复 .env 配置", width="stretch",
                         help="丢掉本次会话里粘贴的 Cookie，回到根 .env 的 MEDIA_DOUYIN_COOKIE"):
                # 运行时覆写只活在进程里，这个按钮是唯一的撤销入口
                _clear_douyin_cookie()
                st.rerun()


def _render_manual_entry() -> None:
    """📋 降级入口：手动粘贴作品数据（采集服务不可达时走这条）。

    页面上的控件与动作
        · ``st.text_area``（key ``review_manual_input``，占位符是 ``SAMPLE_MANUAL_JSON``
          这份能跑通的示例）→ ``st.button``「📝 用粘贴的数据诊断」调
          ``workflows.review.run_review_from_json()`` —— 跳过采集节点，
          直接跑同一套「漏斗诊断 → 内容评估 → 优化策略」。

    成功后：结果写 ``st.session_state["review_result"]``、
    ``["review_url"] = "（手动粘贴数据）"``，并清掉 ``review_history_token``
    （让这次的新结果也能记一条历史），然后 rerun 交给主入口渲染。

    失败时页面显示什么
        粘贴为空 → 黄条「请先粘贴作品数据」，不调工作流；
        解析失败 / 里面没有作品记录 → 工作流写进 ``error_msg``，
        由 ``_render_result()`` 渲染成红条。
    """
    # 默认收起：这是备用路径，主路径是上面的「🔍 获取数据并分析」
    with st.expander("📋 手动粘贴作品数据（采集服务不可达时的降级入口）", expanded=False):
        # 粘贴来源五花八门（JSON 数组 / 抖音原始响应 / 从表格复制的 CSV/TSV），
        # 这里把支持的口径一次说清，省得用户试三次都失败
        st.caption(
            "粘贴 JSON 数组、抖音接口的原始响应，或从表格复制的 CSV/TSV（首行表头）。"
            "字段名支持中文（标题/发布时间/时长/点赞/评论/分享/收藏），"
            "也支持抖音原始英文名（desc/create_time/video.duration/digg_count…）。"
            "这条路会跳过采集节点，直接跑同一套 漏斗诊断 → 内容评估 → 优化策略。"
        )
        raw = st.text_area(
            "作品数据",
            height=180,
            key="review_manual_input",
            # 占位符直接用 tools/douyin_client 里那份能跑通的示例：
            # 用户不知道格式时照着改就能用
            placeholder=SAMPLE_MANUAL_JSON,
        )
        if st.button("📝 用粘贴的数据诊断", width="stretch"):
            if not raw.strip():
                st.warning("请先粘贴作品数据")
                return
            with st.spinner("正在诊断（跳过采集节点）..."):
                # 延迟 import：这个模块导入时会建两张 LangGraph 图，
                # 放在这里页面至少能先把 Cookie 面板与链接输入渲染出来
                from workflows.review import run_review_from_json

                result = run_review_from_json(raw)
            st.session_state["review_result"] = result
            # 固定写成「（手动粘贴数据）」：报告抬头与历史摘要都靠它区分数据来源
            st.session_state["review_url"] = "（手动粘贴数据）"
            # 清掉去重 token：换了一份数据就该记一条新历史，
            # 否则会被「同一份结果只记一次」的逻辑挡掉
            st.session_state.pop("review_history_token", None)
            st.rerun()


def _render_result(result: dict) -> None:
    """渲染复盘结果（结果由 ``st.session_state`` 传进来，rerun 不会丢）。

    只读入参与 ``st.session_state``，不调任何工作流。渲染顺序：数字概览卡片 →
    「📋 作品数据明细」折叠表 → 三个 Tab（漏斗诊断 / 内容评估 / 优化策略）→
    「📥 下载完整复盘报告」，最后按 token 去重追加一条操作历史。

    失败时页面显示什么
        · ``error_msg`` 非空（采集/解析阶段就失败）→ 红条显示工作流给的原因，
          再加一行蓝字指向上方的「📋 手动粘贴作品数据」降级入口；
        · 拿到了结果但没有作品数据 → 黄条给出三条排查方向
          （服务与 Cookie / 链接 / 降级入口）。
    """
    error = result.get("error_msg", "")
    if error:
        # error_msg 非空 = 采集/解析阶段就断了，此时 video_data 与三段分析都是空的：
        # 直接显示原因并提前 return，不给用户看三张空表
        st.error(error)
        st.info("可改用上方的「📋 手动粘贴作品数据」降级入口 —— 后面的诊断链路完全一样。")
        return

    url = st.session_state.get("review_url", "")
    items = _load_items(result.get("video_data", "[]"))
    if not items:
        # 工作流没报错但也没数据：把用户最需要方向的三条排查路径一次给全
        st.warning("未获取到作品数据。请检查：\n"
                   "1. 采集服务是否已启动、Cookie 是否有效\n"
                   "2. 链接是否正确\n"
                   "3. 或改用上方「手动粘贴作品数据」降级入口")
        return

    funnel_diag = result.get("funnel_diagnosis", "")
    content_eval = result.get("content_assessment", "")
    suggestions = result.get("suggestions", "")

    # ---- 数据概览卡片 ----
    total_likes = sum(i.get("点赞", 0) for i in items)
    total_comments = sum(i.get("评论", 0) for i in items)
    total_shares = sum(i.get("分享", 0) for i in items)
    total_collects = sum(i.get("收藏", 0) for i in items)
    total_plays = sum(i.get("播放", 0) for i in items)
    # has_plays 与 _build_report() 里同一套判据（沿用课案写法），
    # 保证页面上的卡片和下载报告里的概览表行数一致
    has_plays = total_plays > 0

    if has_plays:
        # 拿得到播放量才是 6 张卡；拿不到就少一张，
        # 硬塞一张「0 播放」会把「拿不到」说成「真的是 0」
        cols = st.columns(6)
        cols[0].metric("📊 作品数", len(items))
        cols[1].metric("👀 总播放", f"{total_plays:,}")
        cols[2].metric("❤️ 总点赞", f"{total_likes:,}")
        cols[3].metric("💬 总评论", f"{total_comments:,}")
        cols[4].metric("🔁 总分享", f"{total_shares:,}")
        cols[5].metric("⭐ 总收藏", f"{total_collects:,}")
    else:
        cols = st.columns(5)
        cols[0].metric("📊 作品数", len(items))
        cols[1].metric("❤️ 总点赞", f"{total_likes:,}")
        cols[2].metric("💬 总评论", f"{total_comments:,}")
        cols[3].metric("🔁 总分享", f"{total_shares:,}")
        cols[4].metric("⭐ 总收藏", f"{total_collects:,}")

    # ---- 作品明细 ----
    with st.expander("📋 作品数据明细", expanded=False):
        # 直接传 list[dict]：Streamlit 按第一条的键自动列表格，
        # 列名与下载报告里的明细表一致
        st.dataframe(items, width="stretch")

    # ---- 三段分析 ----
    tab1, tab2, tab3 = st.tabs(["📉 漏斗诊断", "📝 内容评估", "💡 优化策略"])
    with tab1:
        # `or "（无内容）"` 兜底：LLM 段失败时字段可能是空串，
        # 空白 Tab 分不清是「没跑完」还是「这段没内容」
        st.markdown(funnel_diag or "（无内容）")
    with tab2:
        st.markdown(content_eval or "（无内容）")
    with tab3:
        st.markdown(suggestions or "（无内容）")

    # ---- 完整报告下载 ----
    st.markdown("---")
    # 报告每轮 rerun 都重拼一次（纯字符串操作，成本可忽略），
    # 换来「下载到的内容」与屏幕上看到的一致
    report = _build_report(items, url, funnel_diag, content_eval, suggestions)
    st.download_button(
        "📥 下载完整复盘报告",
        # data= 显式写出：report 是本轮从 session_state 恢复的结果拼出来的，
        # 不是按钮那次的局部变量（按钮点击会整页 rerun）
        data=report,
        file_name=f"抖音复盘报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        width="stretch",
    )

    # ---- 操作历史：同一份结果只记一条 ----
    # 下载按钮每点一次都会 rerun 整页，不设 token 的话历史会被同一次复盘刷屏。
    # 摘要用 md5 而不是内置 hash()：全项目统一，避免误读成"跨进程稳定"的标识。
    digest = hashlib.md5(report.encode("utf-8")).hexdigest()[:8]
    # token 由「链接 + 数据长度 + 报告指纹」组成：三者任一变化就说明是新的一次复盘，
    # 值得记一条；下载触发的 rerun 三者都不变，于是被跳过
    token = f"{url}|{len(result.get('video_data', ''))}|{digest}"
    if st.session_state.get("review_history_token") != token:
        # 与 token 比较而不是「有没有记过」：同一页面连着跑两次不同链接，
        # 第二次照样要记
        st.session_state["review_history_token"] = token
        _add_history("数据复盘", url[:60])


def show_review() -> None:
    """数据复盘页面（无参，供 main.py 路由调用）。

    页面上的控件
        · 🍪 Cookie 面板 → ``_render_cookie_panel()``（从 Edge 读 / 手动粘贴 / 恢复 .env）
        · 一行探活文字 → ``_service_available()``（``is_available()``，结果缓存 15 秒）
        · ``st.text_input``「🔗 抖音个人主页链接」→ 默认 ``https://www.douyin.com/user/self``
        · ``st.button``「🔍 获取数据并分析」→ 调 ``workflows.review.run_review()``
        · 降级入口 → ``_render_manual_entry()``（调 ``run_review_from_json()``）
        · 结果区 → ``_render_result()``

    数据流
        两条入口都把结果写进 ``st.session_state["review_result"]`` 与
        ``["review_url"]``，再统一交给 ``_render_result()`` 渲染；
        操作历史由 ``_render_result()`` 用 token 去重后追加。

    失败时页面显示什么
        · 链接为空 → 黄条「请输入抖音主页链接」，并且**不提前 return** ——
          降级入口与结果区在同一次运行里还要继续渲染；
        · 没配 Cookie → 蓝条说明「非本人主页的公开数据通常拿不到播放量」并引导去
          Cookie 面板，但仍然继续采集（公开数据这条路本来就能走）；
        · 采集服务不可达 → 顶部灰字带上服务地址，并指向下面的手动粘贴入口。
    """
    st.title("📊 数据复盘智能体")
    st.markdown("输入抖音主页链接 → 采集真实作品数据 → AI 多维度诊断分析")

    # Cookie 面板排在链接输入之前：采集拿不到数据时，第一反应就是查 Cookie
    _render_cookie_panel()

    # ---- 采集服务状态 ----
    base = api_base()
    if _service_available():
        st.caption(f"✅ 采集服务可达：`{base}`")
    else:
        # 探活结果只影响这一行字，**不挡**采集按钮：服务慢或刚起来时探活也可能失败，
        # 真点下去未必失败 —— 所以这里给的是提示而不是禁用
        st.caption(f"⚠️ 采集服务不可达（`{base}`）—— 可先用下面的降级入口粘贴数据跑诊断")

    st.markdown("---")

    # ---- 主路径：采集 ----
    url = st.text_input(
        "🔗 抖音个人主页链接",
        # 默认给「本人主页」：这是最常见的用法，配合 Cookie 能拿到含播放量的完整数据
        value="https://www.douyin.com/user/self",
        placeholder="https://www.douyin.com/user/xxxxx...",
    )

    if st.button("🔍 获取数据并分析", type="primary", width="stretch"):
        if not url.strip():
            # 这里刻意用 else 而不是 return：下面的降级入口与结果渲染
            # 在同一次运行里还要继续执行，提前 return 会把它们一起吞掉
            st.warning("请输入抖音主页链接")
        else:
            # 探一次「当前生效的 Cookie」（运行时覆写优先，其次 .env）；
            # 没配只是提醒，不阻断 —— 公开主页不带 Cookie 也能拿到一部分数据
            if not resolve_cookie():
                st.info("未配置 Cookie —— 非本人主页的公开数据通常拿不到播放量，"
                        "建议先在上方 🍪 抖音 Cookie 配置 里配一个。")
            with st.spinner("正在通过自托管服务采集抖音作品数据..."):
                # 延迟 import：这个模块导入时会建两张 LangGraph 图（采集版与纯诊断版），
                # 放在这里页面至少能先把 Cookie 面板与输入框渲染出来
                from workflows.review import run_review

                result = run_review(url.strip())
            # 先存 session_state 再渲染（下载按钮触发的 rerun 会清空页面）
            st.session_state["review_result"] = result
            st.session_state["review_url"] = url.strip()
            # 清掉去重 token：新的一次采集应该记一条新历史
            st.session_state.pop("review_history_token", None)

    # ---- 降级入口 ----
    _render_manual_entry()

    # ---- 渲染结果（从 session_state 恢复）----
    result = st.session_state.get("review_result")
    if result:
        # 分隔线与结果一起渲染：没结果时页面上不该多出一条孤零零的横线
        st.markdown("---")
        _render_result(result)


# ============================================================
# 四、离线自检
# ============================================================

# 离线自检块：只有 `python views/review.py` 直接跑时才执行。
# 页面模式下本模块是被 main.py import 的（__name__ == "views.review"），所以不会跑。
if __name__ == "__main__":
    print("=== 数据复盘页面自检 ===")

    sample = [
        {"标题": "AI工具提效实测", "发布时间": "2026-09-01", "时长": "45秒",
         "点赞": 1200, "评论": 35, "分享": 12, "收藏": 180},
        {"标题": "三个Python技巧", "发布时间": "2026-08-25", "时长": "60秒",
         "点赞": 320, "评论": 8, "分享": 3, "收藏": 45},
    ]

    # 1) _load_items：坏 JSON / 非数组都要安全降级
    assert _load_items(json.dumps(sample, ensure_ascii=False)) == sample
    for bad in ("", "不是 JSON", "{}", "null", None):
        assert _load_items(bad) == [], bad
    print("  _load_items              OK  正常 JSON / 坏输入都安全")

    # 2) 报告拼装：概览数字、明细表、三段分析都在
    report = _build_report(sample, "https://www.douyin.com/user/xxx", "漏斗A", "评估B", "策略C")
    assert report.startswith("# 📊 抖音数据复盘报告")
    assert "| ❤️ 总点赞 | 1,520 |" in report, report[:900]      # 1200 + 320
    assert "| 💬 总评论 | 43 |" in report                        # 35 + 8
    assert "| 🔁 总分享 | 15 |" in report
    assert "| ⭐ 总收藏 | 225 |" in report
    assert "👀 总播放" not in report, "样例没有播放量，不该出现播放行"
    assert "AI工具提效实测" in report and "三个Python技巧" in report
    assert "## 📉 漏斗诊断" in report and "漏斗A" in report
    assert "## 📝 内容评估" in report and "评估B" in report
    assert "## 💡 优化策略" in report and "策略C" in report
    print(f"  _build_report            OK  {len(report)} 字符，概览/明细/三段分析齐全")

    # 3) 有播放量时报告要带播放行（且顺序在点赞之前）
    with_play = [dict(sample[0], 播放=8600)]
    report = _build_report(with_play, "x", "f", "c", "s")
    assert "| 👀 总播放 | 8,600 |" in report
    assert report.index("总播放") < report.index("总点赞")
    print("  _build_report(含播放)     OK  概览表多一行播放")

    # 4) CDP 端口探活失败时不抛异常（端口 1 必然连不上，且是立即拒绝）
    assert _port_alive(1, timeout=1) is False
    assert isinstance(_find_debug_port(), (int, type(None)))
    print("  _port_alive / _find_debug_port OK  不可达返回 False，无异常")

    # 5) 契约：两个对外函数都在
    assert callable(show_review) and callable(_read_edge_cookies)
    assert callable(_save_douyin_cookie) and callable(_clear_douyin_cookie)
    print("  show_review / _read_edge_cookies OK  契约函数就位")

    # 6) Cookie 运行时覆写走的是同一个键（env → client 读得到）
    assert COOKIE_ENV_KEY == "MEDIA_DOUYIN_COOKIE"
    os.environ[COOKIE_ENV_KEY] = "sessionid=test"
    assert resolve_cookie() == "sessionid=test"
    os.environ.pop(COOKIE_ENV_KEY, None)
    print("  Cookie 覆写键             OK  env 覆写能被 resolve_cookie() 读到")

    # 7) `_pick_page_target`：只用真 CDP 的字段形状，覆盖三种输入
    #    （有 douyin 页 / 只有非 douyin 页 / 一个 page 都没有）。
    #    这条是「连错目标」那个 bug 的回归闸门：改成连 browser 级目标的话，
    #    第 3 个用例会期望落空。
    assert _pick_page_target([
        {"type": "page", "url": "edge://newtab/", "webSocketDebuggerUrl": "ws://x/page/1"},
        {"type": "page", "url": "https://www.douyin.com/", "webSocketDebuggerUrl": "ws://x/page/2"},
    ]) == "ws://x/page/2"
    assert _pick_page_target([
        {"type": "page", "url": "edge://newtab/", "webSocketDebuggerUrl": "ws://x/page/1"},
    ]) == "ws://x/page/1"
    assert _pick_page_target([
        {"type": "service_worker", "url": "https://www.douyin.com/sw.js",
         "webSocketDebuggerUrl": "ws://x/sw/1"},
        {"type": "page", "url": "edge://newtab/"},
    ]) == ""
    print("  _pick_page_target        OK  优先 douyin 页级目标，挑不到就退化/返空")

    print("\n自检完成")
