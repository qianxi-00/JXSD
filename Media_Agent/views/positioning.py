# -*- coding: utf-8 -*-
"""账号定位页面 —— 收集背景信息 → 调定位工作流 → Tab 展示三块结果

课案出处：《3.自媒体Agent》→ 账号定位 → views/positioning.py

本节要讲什么
    1. **前端只干两件事**：收集输入、渲染 ``workflows/`` 的返回值。
       这里不写任何业务逻辑、不直接调 LLM —— 所以定位工作流可以脱离界面
       单独用 ``python workflows/positioning.py --live`` 验证。
    2. **`st.session_state` 是页面的唯一结果缓存**。
       Streamlit 的下载按钮会触发整页 rerun（脚本从头再执行一遍），
       如果结果只存在局部变量里，点一次「下载方案」页面就变空白 ——
       课案反复强调的坑，本模块所有页面都遵守：
       **先写 session_state，再渲染**，渲染一律从 session_state 读。
    3. Tab 分栏：画像分析 / 对标账号 / 完整方案。
       前两块是「中间过程」，第三块是最终交付物（带下载按钮）。

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 操作历史 | `_add_history(...)` 放在**渲染分支末尾** | 移到「生成方案」按钮分支里 | 课案那行每次 rerun 都会执行一次 —— 点一下下载按钮就多一条历史记录 |
    | 按钮宽度 | `use_container_width=True` | `width="stretch"` | Streamlit 1.61 已弃用前者（`st.dataframe` 上会弹弃用警告），两者行为完全一致 |
    | 失败提示 | 无 | 三块结果里任一带 `_FAIL_PREFIXES` 里的前缀就红条提示：LLM 层 `[LLM调用失败/未配置]` + 工作流节点级 `[画像分析失败/对标账号分析失败/定位方案生成失败]` + 图级 `[工作流执行失败]` | 否则页面上是一坨「模型未配置」或兜底文案的文本，看着像正常输出 |
    | 绝对路径 | 无 | 无 | 课案其余页面里的 `C:/Users/13261/...` 一律不带过来 |
    | 无结果显示 | 什么都不显示 | 提示「填写后点击生成」 | 空页面让人以为功能坏了 |

踩过的坑
    · **只把结果写进 session_state 还不够，`_add_history` 也必须挪进按钮分支** ——
      这是课案「结果缓存」那个坑的孪生兄弟：渲染分支会在每次 rerun 时重放。
    · 下载按钮的 ``data`` 参数要传**渲染时刻**的字符串；
      所以 ``plan`` 必须从 session_state 恢复出来的 result 里取，不能依赖按钮那次的局部变量。
    · 直接以脚本方式运行本文件时 ``sys.path[0]`` 是 ``views/``，
      因此顶部同样要做 path 引导（在工作流里是必须的，这里是为了能单独 import 自测）。

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

# 失败结果的文本前缀（本页只做展示，不做重试）。
# 不能只认 LLM 层的两条：`workflows/positioning.py` 自己也会回填三种**节点级**失败文案，
# 图级 `except` 还会回填一条 —— 漏掉它们，失败文本就会被当正常输出渲染，红条不出现。
_FAIL_PREFIXES = (
    "[LLM调用失败",        # 见 workflows/base.py 的 llm_call 兜底
    "[LLM未配置",          # 同上：根目录 .env 里没配 API_KEY
    "[画像分析失败",        # workflows/positioning.py 节点① 的 except
    "[对标账号分析失败",    # workflows/positioning.py 节点② 的 except
    "[定位方案生成失败",    # workflows/positioning.py 节点③ 的 except
    "[工作流执行失败",      # workflows/positioning.py 图级 except
)


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
        "summary": summary[:200],
    })


def show_positioning() -> None:
    """账号定位页面（无参，供 main.py 路由调用）。"""
    st.title("🎯 账号定位智能体")
    st.markdown("输入你的背景信息，AI帮你制定专属的账号定位策划方案")

    col1, col2 = st.columns(2)
    with col1:
        job = st.text_input("你的职业", placeholder="程序员 / 设计师 / 运营 / 学生...")
        skills = st.text_area(
            "核心技能", placeholder="Python开发、视频剪辑、公开演讲...", height=100
        )
    with col2:
        interests = st.text_area(
            "兴趣领域", placeholder="科技测评、美食探店、知识分享...", height=100
        )
        platform = st.selectbox("目标平台", ["抖音", "小红书", "B站", "视频号", "全平台"])

    # ========== 生成 ==========
    if st.button("🚀 生成定位方案", type="primary", width="stretch"):
        if not job and not skills:
            st.warning("请至少填写职业或技能再生成")
            return

        user_input = f"职业：{job}\n技能：{skills}\n兴趣：{interests}\n目标平台：{platform}"

        with st.spinner("AI 正在分析你的画像（三步串行，约 30~90 秒）..."):
            from workflows.positioning import run_positioning

            result = run_positioning(user_input)

        # 必须先落 session_state：下载按钮会触发 rerun，不缓存页面就空白
        st.session_state["pos_result"] = result
        st.session_state["pos_input"] = user_input
        # 历史只在这里记一次（课案放在渲染分支里，每次 rerun 都会重复记一条）
        _add_history("账号定位", user_input)

    # ========== 从 session_state 恢复结果 ==========
    result = st.session_state.get("pos_result")
    if not result:
        st.info("填写上方信息后点击「生成定位方案」。")
        return

    profile = result.get("profile", "")
    competitors = result.get("competitors", "")
    plan = result.get("plan", "")

    failed = [
        name
        for name, value in (("画像分析", profile), ("对标账号", competitors), ("定位方案", plan))
        if value.startswith(_FAIL_PREFIXES)
    ]
    if failed:
        # ⚠️ 失败时**不能**再打「方案生成完成」的绿条 —— 红绿并存会让人以为只是部分降级。
        st.error(
            "以下环节没有跑成功：" + "、".join(failed) +
            "。可以直接重试；若一直失败，检查根目录 .env 里的 API_KEY / BASE_URL / MODEL_NAME。"
        )
    else:
        st.success("✅ 方案生成完成！")
    st.caption(f"输入：{st.session_state.get('pos_input', '')}")
    tab1, tab2, tab3 = st.tabs(["📊 画像分析", "🔍 对标账号", "📝 完整方案"])
    with tab1:
        st.markdown(profile or "分析中...")
    with tab2:
        st.markdown(competitors or "搜索中...")
    with tab3:
        st.markdown(plan or "生成中...")
        st.download_button(
            "📥 下载方案",
            plan,
            file_name="账号定位方案.md",
            mime="text/markdown",
            width="stretch",
        )
