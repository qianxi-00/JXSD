# -*- coding: utf-8 -*-
"""热点监控页面 —— 选平台 + 填赛道 → 抓热榜 → 表格 + AI 筛选 + 选题建议

课案出处：《3.自媒体Agent》→ 热点监控 → views/hot_topic.py

本节要讲什么
    1. **原始数据一定要露出来**。AI 筛选是个黑盒，页面同时把「采集到的 N 条原始热点」
       用 ``st.dataframe`` 摊在折叠面板里 —— 出问题时能一眼分清
       「没抓到数据」和「LLM 筛得不好」，这是课案里很实用的一手。
    2. **报告拼装**：Markdown 表格（表头 + 逐行）+ 两段 AI 结论，
       拼成一个可以直接 ``st.download_button`` 下载的 ``.md`` 文件。
    3. 结果依然**先落 session_state 再渲染**，否则点下载按钮触发的 rerun 会把页面清空。

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 平台下拉框 | `["抖音", "小红书", "全部"]` 三个 | 六项：全部 / 抖音 / 微博 / 知乎 / 小红书 / B站 | 工作流的 `PLATFORM_SENDS` 本来就支持这 6 个，课案只放开了 3 个，等于把已实现的能力藏起来了 |
    | 失效平台 | 无（与其它平台无差别，选中后白等一次注定 0 条的抓取） | 下拉项标「（当前不可用）」（只改显示文案，值仍是纯名）；选中即黄条贴出原因且**不调工作流** | 小红书的数据源已失效，工作流侧只收口了「全部」别把死平台算进去那条路；用户显式选中时，页面这一拦才省得下那十几秒 |
    | 报告抬头 | 直接用渲染分支的局部变量 `{platform}` `{field}` | 取 session_state 里的 `ht_platform` / `ht_field` | 课案在 session_state 里存了这两个值却没用于报告：用户改一下下拉框（不点按钮）就会「报告抬头写新平台、表格里是旧数据」 |
    | 操作历史 | `_add_history(...)` 放在渲染分支末尾 | 移到「获取热点选题」按钮分支里 | 同定位页：渲染分支每次 rerun 都会重放，点一下下载就多一条历史 |
    | 按钮/表格宽度 | `use_container_width=True` | `width="stretch"` | Streamlit 1.61 已弃用 `use_container_width`，`st.dataframe` 上会直接弹弃用警告 |
    | 无用 import | `import json`（页面里没用到） | 去掉 | 只留真正用到的 `pandas` |
    | 热度列 | 表头直接写「热度」 | 「热度(估算)」（表格与下载报告两处同改） | NewsNow 不返回真实热度，`_estimate_heat()` 是按排名估的；不标注会被当成真实热度写进分析结论 |
    | 失败提示 | 无（只判 `raw_topics` 空不空） | 两段结论各自认 `_FAIL_PREFIXES` 前缀 → 红条 + 原文 | 否则「图级崩了」与「一条没抓到」在页面上是同一条黄条，而两者的排查方向完全不同 |
    | 绝对路径 | 无 | 无 | —— |

踩过的坑
    · **报告抬头必须跟着「生成结果的那次输入」走**，不能跟着当前控件值走 ——
      否则会出现「抬头是新平台、内容却是旧结果」的错位报告。
    · 平台列在原始数据里是 ``platform``（中文名，如「抖音」），
      但每个抓取节点写进 state 的字段其实是 ``source``；
      表格读的是 ``platform`` —— 这是 ``fetch_platform_hot`` 返回的字段，
      两条链路都能拿到，所以两个键都读不到时才退化成空。
    · 直接以脚本方式运行时 ``sys.path[0]`` 是 ``views/``，顶部同样要做 path 引导。

运行方式（由 main.py 侧边栏路由调用）::

    Set-Location F:\\ProGram\\Python_Base\\Media_Agent
    & ..\\.venv\\Scripts\\python.exe -m streamlit run main.py
"""

import sys
from pathlib import Path

# Windows 控制台默认 GBK，本模块会打印中文日志
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 路径引导：让 `views.xxx` / `workflows.xxx` 在任意工作目录下都能解析
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# 有平台名、但当前拿不到数据的平台（中文名 → 中文原因），与工作流共用同一张表。
# 故意不写成 try/except ImportError 兜底：这是同一个仓库内部的契约，真断了就该当场红，
# 包起来只会把「平台被静默当成能抓」的问题藏到运行时
from tools.trend_radar_client import UNAVAILABLE_PLATFORMS  # noqa: E402

# 与工作流 PLATFORM_SENDS 对齐；"全部" 走 5 路并行抓取。
# ⚠️ 这里必须是**纯平台名**：不可用平台只靠 `_platform_label()` 改显示文案，
#    选项值（也就是传给工作流的值）不能带任何标记，否则 `NAME_TO_IDS` 查不到
PLATFORM_OPTIONS = ["全部", "抖音", "微博", "知乎", "小红书", "B站"]


def _platform_label(name: str) -> str:
    """下拉里显示的文案：不可用平台带个短标记，**选项值本身仍是纯平台名**。

    标记只写「当前不可用」四个字：完整原因留给选中后的 ``st.warning``，
    一整句中文塞进下拉项既撑宽度也读不清。
    """
    return f"{name}（当前不可用）" if name in UNAVAILABLE_PLATFORMS else name


def _platform_block_reason(name: str) -> str:
    """选中的平台不可用时返回中文原因，可用时返回空串（页面据此「选中即拦」）。

    为什么页面也要拦一道：工作流那边收口的是「选『全部』时别把死平台算进并行分支」，
    而用户**显式选中**死平台时照样会走一遍抓取 —— 结果注定 0 条，白等十几秒。
    """
    return UNAVAILABLE_PLATFORMS.get(name, "")

# 工作流用**回填失败串**表达失败（不抛异常），页面必须认得出这些前缀，
# 否则一段「模型未配置」的文本会被当成 AI 的正常结论渲染成黑字。
# 前缀逐条对着 workflows/hot_topic.py 与 workflows/base.py 的 return 抄的：
_FAIL_PREFIXES = (
    "[LLM调用失败",        # workflows/base.py 的 llm_call 兜底
    "[LLM未配置",          # 同上：根 .env 里没配 API_KEY（`[LLM未配置]`，后面没有冒号）
    "[热点筛选失败",        # workflows/hot_topic.py 节点② 的 except
    "[选题建议生成失败",    # workflows/hot_topic.py 节点③ 的 except
    "[热点工作流失败",      # workflows/hot_topic.py 图级 except（回填进 filtered）
)


def _first_fail_prefix(text: str) -> str:
    """``text`` 是不是工作流回填的失败串；是就返回命中的前缀，否则返回空串。

    用 ``startswith`` 而不是子串匹配：正文里恰好引用到「[LLM调用失败…」这类字样时
    不该误报（同 ``views/positioning.py``）。
    """
    value = text or ""
    for prefix in _FAIL_PREFIXES:
        if value.startswith(prefix):
            return prefix
    return ""


def _topic_state(raw_topics: list, filtered: str, suggestions: str) -> str:
    """本轮结果是「失败 / 没抓到 / 正常」三态里的哪一态。

    必须分三态的理由：图级失败时 ``raw_topics`` 同样是**空列表**，只按空列表判断
    就会把「工作流崩了」说成「今天没热点、可能被限流」—— 两者对用户意味着完全
    不同的处置（一个要查日志/.env，一个只要稍后重试）。

    Returns:
        ``"failed"``：两段结论里任一命中 ``_FAIL_PREFIXES``；
        ``"empty"``：没报错，但一条热点都没抓到；
        ``"ok"``：有原始数据、结论也不是失败串。
    """
    if _first_fail_prefix(filtered) or _first_fail_prefix(suggestions):
        return "failed"
    if not raw_topics:
        return "empty"
    return "ok"


def _render_conclusion(title: str, text: str) -> None:
    """渲染一段 AI 结论；命中失败前缀时走红条 + 原文，不当正常结论展示。"""
    st.markdown(title)
    if _first_fail_prefix(text):
        # 红条里贴**原文**：失败串本身就是工作流写的中文原因，用户看一眼就知道断在哪
        st.error(f"这一段没有跑成功，工作流返回：\n\n{text}")
    else:
        st.markdown(text or "（无）")


def _add_history(action: str, summary: str) -> None:
    """往本次操作记录里追加一条（首页展示最近 10 条）。

    ``main.py`` 已经初始化了 ``st.session_state.history``；
    这里的 ``setdefault`` 只是让页面脱离 main.py 单独跑时也不炸。
    """
    from datetime import datetime

    history = st.session_state.setdefault("history", [])
    history.append({
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        # 只截 200 字符：首页那行只展示摘要，整段赛道描述没必要一直占着内存
        "summary": summary[:200],
    })


def _build_report(result: dict, platform: str, field: str, raw_topics: list) -> str:
    """把原始热点 + AI 结论拼成一份可下载的 Markdown 报告。

    纯字符串拼接，不读 ``st.session_state`` —— 三个值由调用方取好传进来。

    Args:
        result: ``run_hot_topic()`` 的返回值（只取 ``filtered`` 与 ``suggestions``）。
        platform: 平台名 —— 必须是**生成结果那次**的输入，不能传当前下拉框的值，
            否则会出现「抬头写新平台、表格里是旧数据」的错位报告。
        field: 赛道名，同上。
        raw_topics: 原始热点列表，每条含 ``rank`` / ``title`` / ``platform`` /
            ``source`` / ``heat``。

    Returns:
        完整 Markdown 文本；``raw_topics`` 为空时跳过表格段，只留两段 AI 结论。
    """
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    table = ""
    # 没抓到数据时连表头都不输出：空的 Markdown 表渲染出来是一片空白，
    # 不如让上面那句「原始数据（0 条）」把条数说清楚
    if raw_topics:
        rows = "\n".join(
            # 平台列两个键都读：抓取节点写进 state 的字段是 `source`，
            # 而 `fetch_platform_hot` 返回的是 `platform` —— 两条链路都覆盖到，
            # 都拿不到时才退化成空
            f"| {t.get('rank', '')} | {t.get('title', '')} | "
            f"{t.get('platform', '') or t.get('source', '')} | {t.get('heat', '')} |"
            for t in raw_topics
        )
        # 表头带「(估算)」是必须的：NewsNow 不返回真实热度，heat 由 `_estimate_heat()`
        # 按排名估出来；不标注就会被人当成真实热度写进后续分析结论
        table = f"| 排名 | 标题 | 平台 | 热度(估算) |\n|---|---|---|---|\n{rows}\n"

    return f"""# 🔥 热点监控报告

> 生成时间：{now}  平台：{platform}  赛道：{field}

## 📡 原始数据（{len(raw_topics)} 条）

{table}
## 📋 赛道相关热点筛选

{result.get('filtered', '')}

## 💡 选题创作建议

{result.get('suggestions', '')}
"""


def show_hot_topic() -> None:
    """热点监控页面（无参，供 main.py 路由调用）。

    页面上的控件
        · ``st.selectbox``「选择平台」→ 六项 ``PLATFORM_OPTIONS``（「全部」= 5 路并行抓取）；
          不可用平台由 ``format_func=_platform_label`` 标成「（当前不可用）」，选中后
          被 ``_platform_block_reason()`` 拦下（黄条 + 原因，不调工作流）
        · ``st.text_input``「你的赛道/领域」→ 唯一的必填项
        · ``st.button``「🔍 获取热点选题」→ 调 ``workflows.hot_topic.run_hot_topic()``
        · 折叠面板「📡 采集到的原始数据」里的 ``st.dataframe``（只读）
        · ``st.download_button``「📥 下载完整热点报告」

    数据流
        平台 + 赛道两个输入交给 ``run_hot_topic()``；返回的 ``raw_topics``
        （原始热点）渲染成表格、``filtered`` / ``suggestions`` 渲染成两段 Markdown，
        一起拼进可下载的报告。结果存 ``st.session_state["ht_result"]``，
        生成时的两个输入另存 ``["ht_platform"]`` / ``["ht_field"]``（报告抬头要用），
        并往 ``st.session_state["history"]`` 追加一条（只在按钮分支里）。

    失败时页面显示什么
        · 赛道为空 → 黄条「请输入你的赛道」，直接 return，不调工作流；
        · 选中的平台已失效（``UNAVAILABLE_PLATFORMS`` 里有）→ 黄条把原因原样贴出，
          并点明「不是网络问题、也不是今天没热搜」，同样直接 return，不调工作流；
        · 一条热点都没抓到 → 工作流回填空列表而不是抛异常，页面出黄条说明
          「可能被限流或被墙」，并告知 AI 筛选与选题建议已跳过；
        · 工作流 / LLM 失败 → `filtered` / `suggestions` 以 `_FAIL_PREFIXES` 开头时
          各自出红条并贴出原文；图级失败（`[热点工作流失败: …]`）时 `raw_topics`
          同样是空的，页面给红条说明链路断了，不会误报成「没抓到」；
        · 还没跑过 → 蓝条引导「选择平台、填写赛道后点击…」。
    """
    st.title("🔥 热点监控智能体")
    st.markdown("实时追踪平台热点，AI筛选与你的赛道相关的选题机会")

    col1, col2 = st.columns(2)
    with col1:
        # format_func 只改**显示文案**：选中不可用平台时返回值仍是纯名「小红书」
        platform = st.selectbox("选择平台", PLATFORM_OPTIONS, format_func=_platform_label)
    with col2:
        field = st.text_input(
            "你的赛道/领域", placeholder="科技测评 / 职场成长 / 美妆护肤..."
        )

    # ========== 抓取 + 分析 ==========
    if st.button("🔍 获取热点选题", type="primary", width="stretch"):
        # 平台有默认值、赛道没有：空赛道时 LLM 没有筛选判据，所以这是唯一的必填项
        if not field:
            st.warning("请输入你的赛道")
            return

        # 选中已知失效的平台：直接拦住，连那十几秒都别让用户等
        block_reason = _platform_block_reason(platform)
        if block_reason:
            st.warning(
                f"「{platform}」当前不可用，本轮不发起抓取。\n\n"
                f"原因：{block_reason}\n\n"
                "这不是你的网络问题，也不是今天没有热搜 —— 请换一个平台；"
                "选「全部」时该平台已被自动跳过，不影响其它平台。"
            )
            return

        # 两段耗时分开写（抓取 10~30 秒 / AI 30~90 秒）：用户觉得慢了，
        # 能自己判断是卡在抓取还是卡在 LLM
        with st.spinner(f"正在抓取{platform}热点并分析（抓取 10~30 秒，AI 分析 30~90 秒）..."):
            # 延迟 import：这个模块导入时会建 LangGraph 图（「全部」还带 5 条并行分支），
            # 放在这里页面至少能先把标题与输入控件渲染出来
            from workflows.hot_topic import run_hot_topic

            result = run_hot_topic(platform, field)

        # 存入 session_state 防止刷新 / 下载 rerun 丢失
        st.session_state["ht_result"] = result
        # 输入也一起存：报告抬头要用「生成结果那次」的平台与赛道，
        # 只读当前控件值的话，用户改一下下拉框（不点按钮）就会抬头与数据错位
        st.session_state["ht_platform"] = platform
        st.session_state["ht_field"] = field
        _add_history("热点监控", f"{platform} - {field}")

    # ========== 从 session_state 恢复结果 ==========
    result = st.session_state.get("ht_result")
    if not result:
        # 还没跑过：只渲染引导，不渲染下面的空表格与空结论
        st.info("选择平台、填写赛道后点击「获取热点选题」。")
        return

    # 报告抬头用「生成结果那次的输入」，不是当前控件值
    saved_platform = st.session_state.get("ht_platform", platform)
    saved_field = st.session_state.get("ht_field", field)
    raw_topics = result.get("raw_topics", []) or []
    filtered = result.get("filtered", "")
    suggestions = result.get("suggestions", "")
    # 先算三态再渲染：`raw_topics` 空既可能是「没抓到」也可能是「图级崩了」，
    # 只看空列表会把后者写成「可能被限流」（见 `_topic_state`）
    state = _topic_state(raw_topics, filtered, suggestions)

    if raw_topics:
        st.success(f"✅ 抓到 {len(raw_topics)} 条热点")
        # 默认展开：原始数据一定要露出来 —— AI 筛选是个黑盒，
        # 摊开才能一眼分清「没抓到数据」和「LLM 筛得不好」
        with st.expander(f"📡 采集到的原始数据（{len(raw_topics)} 条）", expanded=True):
            rows = [
                {
                    "排名": t.get("rank", ""),
                    "标题": t.get("title", ""),
                    "平台": t.get("platform", "") or t.get("source", ""),
                    "热度(估算)": t.get("heat", ""),
                    "链接": t.get("url", ""),
                }
                for t in raw_topics
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch")
    elif state == "failed":
        # 图级失败同样是空列表，但原因不是「限流」而是「链路断了」：
        # 具体原文在下面对应那段的红条里，这里只负责别把它误报成没抓到
        st.error("本轮热点分析没有跑成功：原始数据为空，失败原因见下方的红条。")
    else:
        # 一条都没抓到时工作流返回的是**空列表**而不是异常，所以必须在这里显式提示；
        # 否则页面只剩两段「（无）」，看着像功能没跑而不是数据没来
        st.warning(
            "这次一条热点都没抓到（可能是热榜接口限流或被墙）。"
            "AI 筛选与选题建议已自动跳过，可稍后重试。"
        )

    st.caption(f"平台：{saved_platform}　赛道：{saved_field}")

    # 两段结论各自判失败：筛选中了、选题没中时也各自显示（部分失败比整页红条更有信息量）
    _render_conclusion("### 📋 赛道相关热点筛选", filtered)
    _render_conclusion("### 💡 选题创作建议", suggestions)

    # ========== 下载完整报告 ==========
    from datetime import datetime

    # 报告每轮 rerun 都重拼一次（纯字符串拼接，成本可忽略）；
    # 换来的好处是「下载到的内容」永远与屏幕上看到的那份一致
    report = _build_report(result, saved_platform, saved_field, raw_topics)
    st.download_button(
        "📥 下载完整热点报告",
        report,
        # 文件名带分钟级时间戳：连下几次不会互相覆盖
        file_name=f"热点监控报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        width="stretch",
    )
