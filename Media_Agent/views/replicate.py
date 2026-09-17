# -*- coding: utf-8 -*-
"""内容复刻页面 —— 粘贴链接 → 提取文案 → 爆款拆解 → 仿写 → 标题

课案出处：《3.自媒体Agent》→ 内容复刻 → views/replicate.py

本节要讲什么
    1. **四个 Tab 对应工作流的四段产出**：原文案 / 爆款拆解 / 仿写文案 / 标题方案。
       前三个都是「看一眼就过」的中间物，最后拼一份完整报告供下载。
    2. 结果仍然**先落 session_state 再渲染**：这里是全项目最能体现这个坑的页面 ——
       页面上一共两个下载按钮（仿写文案、完整报告），点哪个都会 rerun。
    3. **带 key 的输入框会"记住"上一次的值**。`st.text_area(..., key="orig_text")`
       在第一次生成之后，session_state 里就有了 ``orig_text``，
       第二次复刻别的视频时 Streamlit 会**优先用 session_state 的旧值**、
       忽略新传进来的 ``value`` —— 于是页面显示的还是上一条视频的文案。
       本实现让 key 跟着「第几次生成」走（``rep_run_id``），每次新结果都是新控件。

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 报告 f-string | 三引号 f-string 里直接内嵌 `st.session_state.get("rep_url","")`（内层复用了外层的 `"`） | 先 `rep_url = st.session_state.get("rep_url", "")`，再 `{rep_url}` 插值 | 课案那写法要 PEP 701（3.12+）才合法，3.11 直接 SyntaxError；而且同引号套嵌语义含糊，跑通也难读 |
    | 输入框 key | 固定 `key="orig_text"` / `"new_text"` | `key=f"rep_orig_{run_id}"` / `f"rep_new_{run_id}"` | 带 key 的控件跨 rerun 保留旧值，第二次复刻会显示上一条视频的文案 |
    | 操作历史 | `_add_history(...)` 放在渲染分支末尾 | 移到「分析 & 仿写」按钮分支里 | 渲染分支每次 rerun 都重放，点一下下载就多一条历史 |
    | 按钮宽度 | `use_container_width=True` | `width="stretch"` | Streamlit 1.61 已弃用 `use_container_width` |
    | 失败提示 | 无 | 提取不到文案时红条提示（工作流会返回 `⚠️` 开头的字段） | 否则页面显示「仿写结果」是一句提示文本，看着像写完了 |
    | 绝对路径 | 无 | 无 | —— |

踩过的坑
    · ``st.download_button`` 的 data 必须取自 **session_state 恢复出来的 result**，
      不能依赖按钮那次的局部变量（rerun 后局部变量是空的）。
    · ``st.text_area(value=..., key=...)`` 同时给值时，key 已存在就用 key 的值。
      这是「结果缓存」思路的反面：**输入类控件要让它每次都是新控件**，
      而结果类展示（markdown）随便重放都无所谓。

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

import streamlit as st  # noqa: E402

# 工作流链路中断时，字段统一以这个标记开头
_SKIP_MARK = "⚠️"


def _add_history(action: str, summary: str) -> None:
    """往本次操作记录里追加一条（首页展示最近 10 条）。"""
    from datetime import datetime

    history = st.session_state.setdefault("history", [])
    history.append({
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        "summary": summary[:200],
    })


def _build_report(result: dict, rep_url: str) -> str:
    """把四段产出拼成一份可下载的 Markdown 复刻报告。"""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    # 课案在这里直接把 st.session_state.get("rep_url", "") 写进 f-string 里，
    # 内层双引号和外层三引号打架（PEP 701 之前是 SyntaxError）——
    # 先取成变量再插值，语义清楚、也不挑 Python 版本。
    return f"""# 📝 内容复刻报告

> 生成时间：{now}
> 来源链接：{rep_url}

---

## 📄 原文案

{result.get('original_text', '')}

---

## 🔍 爆款拆解

{result.get('viral_analysis', '')}

---

## ✍️ 仿写文案

{result.get('rewritten', '')}

---

## 🏷️ 标题方案

{result.get('titles', '')}
"""


def show_replicate() -> None:
    """内容复刻页面（无参，供 main.py 路由调用）。"""
    st.title("📝 内容复刻智能体")
    st.markdown("粘贴爆款视频/文章链接 → 提取文案 → 拆解爆款公式 → 仿写新文案")

    url = st.text_input("📎 视频链接", placeholder="粘贴抖音/B站链接，或公众号/知乎文章链接...")

    # ========== 分析 & 仿写 ==========
    if st.button("🔄 分析 & 仿写", type="primary", width="stretch"):
        if not url:
            st.warning("请输入链接")
            return

        with st.spinner("提取文案并分析中（下载 + 语音识别 + 3 次 LLM，可能需要 1~3 分钟）..."):
            from workflows.replicate import run_replicate

            result = run_replicate(url)

        # 存入 session_state 防止下载按钮触发 rerun 后丢失
        st.session_state["rep_result"] = result
        st.session_state["rep_url"] = url
        # 每次生成换一组控件 key，避免输入框顽固地显示上一次的文案
        st.session_state["rep_run_id"] = st.session_state.get("rep_run_id", 0) + 1
        _add_history("内容复刻", url)

    # ========== 从 session_state 恢复结果 ==========
    result = st.session_state.get("rep_result")
    if not result:
        st.info("粘贴链接后点击「分析 & 仿写」。")
        return

    rep_url = st.session_state.get("rep_url", "")
    run_id = st.session_state.get("rep_run_id", 0)

    original = result.get("original_text", "")
    analysis = result.get("viral_analysis", "")
    rewritten = result.get("rewritten", "")
    titles = result.get("titles", "")

    st.caption(f"来源：{rep_url}")

    if original.startswith(_SKIP_MARK):
        st.error(original)
    else:
        st.success(f"✅ 提取到文案 {len(original)} 字")

    tabs = st.tabs(["📄 原文案", "🔍 爆款拆解", "✍️ 仿写文案", "🏷️ 标题"])
    with tabs[0]:
        st.text_area("原文", original, height=200, key=f"rep_orig_{run_id}")
    with tabs[1]:
        st.markdown(analysis or "（无）")
    with tabs[2]:
        st.text_area("仿写结果", rewritten, height=200, key=f"rep_new_{run_id}")
        st.download_button(
            "📥 下载仿写文案",
            rewritten,
            file_name="仿写文案.txt",
            width="stretch",
        )
    with tabs[3]:
        st.markdown(titles or "（无）")

    # ========== 完整报告下载 ==========
    st.markdown("---")
    from datetime import datetime

    report = _build_report(result, rep_url)
    st.download_button(
        "📥 下载完整复刻报告",
        report,
        file_name=f"内容复刻报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        width="stretch",
    )
