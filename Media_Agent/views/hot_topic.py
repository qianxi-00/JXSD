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
    | 报告抬头 | 直接用渲染分支的局部变量 `{platform}` `{field}` | 取 session_state 里的 `ht_platform` / `ht_field` | 课案在 session_state 里存了这两个值却没用于报告：用户改一下下拉框（不点按钮）就会「报告抬头写新平台、表格里是旧数据」 |
    | 操作历史 | `_add_history(...)` 放在渲染分支末尾 | 移到「获取热点选题」按钮分支里 | 同定位页：渲染分支每次 rerun 都会重放，点一下下载就多一条历史 |
    | 按钮/表格宽度 | `use_container_width=True` | `width="stretch"` | Streamlit 1.61 已弃用 `use_container_width`，`st.dataframe` 上会直接弹弃用警告 |
    | 无用 import | `import json`（页面里没用到） | 去掉 | 只留真正用到的 `pandas` |
    | 热度列 | 表头直接写「热度」 | 「热度(估算)」（表格与下载报告两处同改） | NewsNow 不返回真实热度，`_estimate_heat()` 是按排名估的；不标注会被当成真实热度写进分析结论 |
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

# 与工作流 PLATFORM_SENDS 对齐；"全部" 走 5 路并行抓取
PLATFORM_OPTIONS = ["全部", "抖音", "微博", "知乎", "小红书", "B站"]


def _add_history(action: str, summary: str) -> None:
    """往本次操作记录里追加一条（首页展示最近 10 条）。"""
    from datetime import datetime

    history = st.session_state.setdefault("history", [])
    history.append({
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        "summary": summary[:200],
    })


def _build_report(result: dict, platform: str, field: str, raw_topics: list) -> str:
    """把原始热点 + AI 结论拼成一份可下载的 Markdown 报告。"""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    table = ""
    if raw_topics:
        rows = "\n".join(
            f"| {t.get('rank', '')} | {t.get('title', '')} | "
            f"{t.get('platform', '') or t.get('source', '')} | {t.get('heat', '')} |"
            for t in raw_topics
        )
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
    """热点监控页面（无参，供 main.py 路由调用）。"""
    st.title("🔥 热点监控智能体")
    st.markdown("实时追踪平台热点，AI筛选与你的赛道相关的选题机会")

    col1, col2 = st.columns(2)
    with col1:
        platform = st.selectbox("选择平台", PLATFORM_OPTIONS)
    with col2:
        field = st.text_input(
            "你的赛道/领域", placeholder="科技测评 / 职场成长 / 美妆护肤..."
        )

    # ========== 抓取 + 分析 ==========
    if st.button("🔍 获取热点选题", type="primary", width="stretch"):
        if not field:
            st.warning("请输入你的赛道")
            return

        with st.spinner(f"正在抓取{platform}热点并分析（抓取 10~30 秒，AI 分析 30~90 秒）..."):
            from workflows.hot_topic import run_hot_topic

            result = run_hot_topic(platform, field)

        # 存入 session_state 防止刷新 / 下载 rerun 丢失
        st.session_state["ht_result"] = result
        st.session_state["ht_platform"] = platform
        st.session_state["ht_field"] = field
        _add_history("热点监控", f"{platform} - {field}")

    # ========== 从 session_state 恢复结果 ==========
    result = st.session_state.get("ht_result")
    if not result:
        st.info("选择平台、填写赛道后点击「获取热点选题」。")
        return

    # 报告抬头用「生成结果那次的输入」，不是当前控件值
    saved_platform = st.session_state.get("ht_platform", platform)
    saved_field = st.session_state.get("ht_field", field)
    raw_topics = result.get("raw_topics", []) or []

    if not raw_topics:
        st.warning(
            "这次一条热点都没抓到（可能是热榜接口限流或被墙）。"
            "AI 筛选与选题建议已自动跳过，可稍后重试。"
        )
    else:
        st.success(f"✅ 抓到 {len(raw_topics)} 条热点")
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

    st.caption(f"平台：{saved_platform}　赛道：{saved_field}")

    st.markdown("### 📋 赛道相关热点筛选")
    st.markdown(result.get("filtered", "") or "（无）")

    st.markdown("### 💡 选题创作建议")
    st.markdown(result.get("suggestions", "") or "（无）")

    # ========== 下载完整报告 ==========
    from datetime import datetime

    report = _build_report(result, saved_platform, saved_field, raw_topics)
    st.download_button(
        "📥 下载完整热点报告",
        report,
        file_name=f"热点监控报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        width="stretch",
    )
