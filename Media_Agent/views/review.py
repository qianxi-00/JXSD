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
    | Cookie 自动获取 | ``_read_edge_cookies()`` 走 CDP 读 Edge | **原样保留**（v20 之后本地 Cookies 库解不开，走浏览器自己的 API 才是正解） |
    | 采集触发 | 本地爬虫 | 自托管 REST 服务；**服务不可达时页面自动提示降级入口** |
    | 降级入口 | 无 | 新增「📋 手动粘贴作品数据」文本框 → ``run_review_from_json()``，跑同一套三节点诊断 |
    | 作品明细 | 无播放量时 5 列卡片 | 同（沿用课案的 ``has_plays`` 判断） |
    | 操作历史 | 每次渲染都 append（点一次下载就多一条） | 加 token 去重，同一份结果只记一条 |

踩过的坑
    · **结果必须先存 ``st.session_state`` 再渲染** —— ``st.download_button`` 被点击会
      触发整页 rerun，不缓存结果的话页面会当场变空白（课案反复强调的坑）。
    · ``_add_history`` 如果直接在渲染路径里调用，下载按钮的每次 rerun 都会重复记录，
      所以用 ``review_history_token`` 去重。
    · 直接 ``python views/review.py`` 时 ``sys.path[0]`` 是 ``views/`` 目录，
      所以路径引导必须在文件顶部、项目内 import 之前。
"""

import sys
from pathlib import Path

# ---- 路径引导：必须在 import 项目内模块（tools/*、workflows/*）之前执行 ----
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
    """Edge 的 CDP 调试端口是否已经开着。"""
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=timeout)
        return True
    except Exception:
        return False


def _find_debug_port() -> int | None:
    """在常见调试端口里找一个已经在跑的 Edge。"""
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
            subprocess.run(["taskkill", "/f", "/im", proc_name],
                           capture_output=True, timeout=10)
        except Exception:
            pass
    time.sleep(1.5)

    user_data = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data")
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for exe in candidates:
        if not os.path.exists(exe):
            continue
        try:
            proc = subprocess.Popen(
                [exe, f"--remote-debugging-port={port}",
                 f"--user-data-dir={user_data}", "--headless=new"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[复盘] 启动 Edge 失败({exe}): {exc}")
            continue
        for _ in range(15):
            time.sleep(0.5)
            if _port_alive(port):
                return proc
        return proc   # 起了但没监听成功，交给调用方判断并清理
    return None


def _read_edge_cookies() -> tuple[str, "str | None"]:
    """通过 CDP（Chrome DevTools Protocol）从 Edge 直接读取 douyin.com Cookie。

    课案原样保留这套做法：**走浏览器自己的 Network.getCookies 接口**，
    自动跨过 Edge 的 v20 加密与文件锁定，不需要解密本地 Cookies 数据库。

    Returns:
        ``(cookie_string, error_message)`` —— 成功时 error 为 None。
    """
    debug_port = _find_debug_port()
    edge_proc = None
    launched = False

    if debug_port is None:
        debug_port = 9223
        edge_proc = _launch_edge(debug_port)
        launched = edge_proc is not None
        if not launched:
            return "", "无法启动 Edge（未找到 msedge.exe），请确认 Edge 已安装"
        if not _port_alive(debug_port):
            return "", (
                f"Edge 已启动但调试端口 {debug_port} 未就绪。\n"
                "请手动关掉所有 Edge 窗口后重试。"
            )

    try:
        # Step 1: 拿 WebSocket 端点
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{debug_port}/json/version", timeout=5
        )
        ws_url = json.loads(resp.read().decode()).get("webSocketDebuggerUrl", "")
        if not ws_url:
            return "", "无法获取 Edge CDP WebSocket URL"

        # Step 2: 用 websocket-client 连上去，按域名问 Cookie
        try:
            from websocket import create_connection
        except ImportError:
            return "", "缺少 websocket-client 库，请执行: uv add websocket-client"

        ws = create_connection(ws_url, timeout=10)
        try:
            all_cookies = []
            for domain in ("www.douyin.com", ".douyin.com", ".iesdouyin.com"):
                ws.send(json.dumps({
                    "id": 1,
                    "method": "Network.getCookies",
                    "params": {"domain": domain},
                }))
                payload = json.loads(ws.recv())
                all_cookies.extend(payload.get("result", {}).get("cookies", []))
        finally:
            ws.close()

        if not all_cookies:
            return "", (
                "Edge 中未找到 douyin.com 的 Cookie。\n"
                "请先在 Edge 浏览器里打开 douyin.com 并登录，再回来点这个按钮。"
            )

        # 去重后拼成请求头要的字符串
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
        # Step 3: 清理我们自己起的 Edge（别人起的窗口不动）
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
    os.environ[COOKIE_ENV_KEY] = cookie
    st.session_state["douyin_cookie_runtime"] = cookie


def _clear_douyin_cookie() -> None:
    """丢掉运行时覆写，回到根 .env 的配置。"""
    os.environ.pop(COOKIE_ENV_KEY, None)
    st.session_state.pop("douyin_cookie_runtime", None)


@st.cache_data(ttl=15, show_spinner=False)
def _service_available() -> bool:
    """探活（缓存 15 秒）。

    为什么不每次渲染都探：下载按钮触发的每一次 rerun 都会重跑整页，
    服务不可达时那是实打实的超时等待。
    """
    return is_available()


# ============================================================
# 二、页面小工具
# ============================================================

def _add_history(action: str, summary: str) -> None:
    """往 ``st.session_state.history`` 追加一条（首页展示最近 10 条）。"""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        "summary": (summary or "")[:200],
    })


def _load_items(data_str: str) -> list:
    """把工作流返回的 ``video_data``（JSON 字符串）还原成列表。"""
    try:
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
        report += f"| 👀 总播放 | {total_plays:,} |\n"
    report += (
        f"| ❤️ 总点赞 | {total_likes:,} |\n"
        f"| 💬 总评论 | {total_comments:,} |\n"
        f"| 🔁 总分享 | {total_shares:,} |\n"
        f"| ⭐ 总收藏 | {total_collects:,} |\n"
    )

    report += "\n---\n\n## 📋 作品数据明细\n\n"
    if items:
        keys = list(items[0].keys())
        report += "| " + " | ".join(keys) + " |\n"
        report += "|" + "|".join(["------"] * len(keys)) + "|\n"
        for item in items:
            report += "| " + " | ".join(str(item.get(k, "")) for k in keys) + " |\n"

    report += f"\n---\n\n## 📉 漏斗诊断\n\n{funnel}\n"
    report += f"\n---\n\n## 📝 内容评估\n\n{content}\n"
    report += f"\n---\n\n## 💡 优化策略\n\n{suggestions}\n"
    return report


# ============================================================
# 三、页面
# ============================================================

def _render_cookie_panel() -> None:
    """🍪 Cookie 配置区（课案结构，落点从 settings.json 换成运行时覆写）。"""
    with st.expander("🍪 抖音 Cookie 配置", expanded=False):
        st.caption("两种方式：① 点按钮自动从 Edge 读取　② 手动复制粘贴。"
                   "本次会话内生效，重启后回到根目录 .env 的配置。")

        col_auto, col_tip = st.columns([1, 3])
        with col_auto:
            if st.button("🔍 从 Edge 浏览器获取", use_container_width=True,
                         help="自动读取 Edge 里已登录的抖音 Cookie（走 CDP，不需要解密）"):
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
            value="",
            placeholder="从浏览器 F12 → Application → Cookies → 复制完整 Cookie...",
            height=100,
            key="douyin_cookie_input",
        )
        col_save, col_clear = st.columns([1, 1])
        with col_save:
            if st.button("💾 使用这份 Cookie", use_container_width=True):
                if not cookie_input.strip():
                    st.warning("请先粘贴 Cookie")
                else:
                    _save_douyin_cookie(cookie_input)
                    st.success("Cookie 已生效（仅本次会话）")
                    st.rerun()
        with col_clear:
            if st.button("♻️ 恢复 .env 配置", use_container_width=True,
                         help="丢掉本次会话里粘贴的 Cookie，回到根 .env 的 MEDIA_DOUYIN_COOKIE"):
                _clear_douyin_cookie()
                st.rerun()


def _render_manual_entry() -> None:
    """📋 降级入口：手动粘贴作品数据（采集服务不可达时走这条）。"""
    with st.expander("📋 手动粘贴作品数据（采集服务不可达时的降级入口）", expanded=False):
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
            placeholder=SAMPLE_MANUAL_JSON,
        )
        if st.button("📝 用粘贴的数据诊断", use_container_width=True):
            if not raw.strip():
                st.warning("请先粘贴作品数据")
                return
            with st.spinner("正在诊断（跳过采集节点）..."):
                from workflows.review import run_review_from_json

                result = run_review_from_json(raw)
            st.session_state["review_result"] = result
            st.session_state["review_url"] = "（手动粘贴数据）"
            st.session_state.pop("review_history_token", None)
            st.rerun()


def _render_result(result: dict) -> None:
    """渲染复盘结果（结果由 ``st.session_state`` 传进来，rerun 不会丢）。"""
    error = result.get("error_msg", "")
    if error:
        st.error(error)
        return

    url = st.session_state.get("review_url", "")
    items = _load_items(result.get("video_data", "[]"))
    if not items:
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
    has_plays = total_plays > 0

    if has_plays:
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
        st.dataframe(items, use_container_width=True)

    # ---- 三段分析 ----
    tab1, tab2, tab3 = st.tabs(["📉 漏斗诊断", "📝 内容评估", "💡 优化策略"])
    with tab1:
        st.markdown(funnel_diag or "（无内容）")
    with tab2:
        st.markdown(content_eval or "（无内容）")
    with tab3:
        st.markdown(suggestions or "（无内容）")

    # ---- 完整报告下载 ----
    st.markdown("---")
    report = _build_report(items, url, funnel_diag, content_eval, suggestions)
    st.download_button(
        "📥 下载完整复盘报告",
        data=report,
        file_name=f"抖音复盘报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # ---- 操作历史：同一份结果只记一条 ----
    # 下载按钮每点一次都会 rerun 整页，不设 token 的话历史会被同一次复盘刷屏。
    token = f"{url}|{len(result.get('video_data', ''))}|{hash(report)}"
    if st.session_state.get("review_history_token") != token:
        st.session_state["review_history_token"] = token
        _add_history("数据复盘", url[:60])


def show_review() -> None:
    """数据复盘 —— 接真实抖音作品数据（自托管服务 / 手动粘贴两条路）。"""
    st.title("📊 数据复盘智能体")
    st.markdown("输入抖音主页链接 → 采集真实作品数据 → AI 多维度诊断分析")

    _render_cookie_panel()

    # ---- 采集服务状态 ----
    base = api_base()
    if _service_available():
        st.caption(f"✅ 采集服务可达：`{base}`")
    else:
        st.caption(f"⚠️ 采集服务不可达（`{base}`）—— 可先用下面的降级入口粘贴数据跑诊断")

    st.markdown("---")

    # ---- 主路径：采集 ----
    url = st.text_input(
        "🔗 抖音个人主页链接",
        value="https://www.douyin.com/user/self",
        placeholder="https://www.douyin.com/user/xxxxx...",
    )

    if st.button("🔍 获取数据并分析", type="primary", use_container_width=True):
        if not url.strip():
            st.warning("请输入抖音主页链接")
        else:
            if not resolve_cookie():
                st.info("未配置 Cookie —— 非本人主页的公开数据通常拿不到播放量，"
                        "建议先在上方 🍪 抖音 Cookie 配置 里配一个。")
            with st.spinner("正在通过自托管服务采集抖音作品数据..."):
                from workflows.review import run_review

                result = run_review(url.strip())
            # 先存 session_state 再渲染（下载按钮触发的 rerun 会清空页面）
            st.session_state["review_result"] = result
            st.session_state["review_url"] = url.strip()
            st.session_state.pop("review_history_token", None)

    # ---- 降级入口 ----
    _render_manual_entry()

    # ---- 渲染结果（从 session_state 恢复）----
    result = st.session_state.get("review_result")
    if result:
        st.markdown("---")
        _render_result(result)


# ============================================================
# 四、离线自检
# ============================================================

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

    print("\n自检完成")
