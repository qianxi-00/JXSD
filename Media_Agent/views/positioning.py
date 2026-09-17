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
        # 只截 200 字符：首页那行只展示摘要，整段输入（可能上千字）没必要一直占着内存
        "summary": summary[:200],
    })


def show_positioning() -> None:
    """账号定位页面（无参，供 main.py 路由调用）。

    页面上的控件
        · ``st.text_input``「你的职业」→ ``job``（可为空）
        · ``st.text_area``「核心技能」→ ``skills``（可为空）
        · ``st.text_area``「兴趣领域」→ ``interests``
        · ``st.selectbox``「目标平台」→ ``platform``（抖音 / 小红书 / B站 / 视频号 / 全平台）
        · ``st.button``「🚀 生成定位方案」→ 调 ``workflows.positioning.run_positioning()``
        · 三个 Tab 里的 ``st.download_button``「📥 下载方案」

    数据流
        四个控件拼成一段带字段名的文本 ``user_input``（工作流把它原样塞进 prompt，
        不做二次解析），交给 ``run_positioning()``；返回的 ``profile`` /
        ``competitors`` / ``plan`` 分别渲染到「画像分析 / 对标账号 / 完整方案」三个 Tab。
        整个 result 存进 ``st.session_state["pos_result"]``、输入原文存进
        ``st.session_state["pos_input"]``，并往 ``st.session_state["history"]``
        追加一条 —— 这三件事都只发生在按钮分支里（渲染分支每次 rerun 都会重放）。

    失败时页面显示什么
        · 职业与技能都空 → 黄条「请至少填写职业或技能再生成」，直接 return，不调工作流；
        · 三块产出里任一块以 ``_FAIL_PREFIXES`` 的前缀开头 → 红条列出没跑成功的环节，
          并提示去查根目录 ``.env`` 的 ``API_KEY`` / ``BASE_URL`` / ``MODEL_NAME``；
        · 还没生成过（session_state 里没有 ``pos_result``）→ 蓝条引导先填信息再点按钮。
    """
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
    # width="stretch" 是 use_container_width=True 的替代写法：行为一致，
    # 但后者在 streamlit 1.61 上会打弃用告警
    if st.button("🚀 生成定位方案", type="primary", width="stretch"):
        # 只把「职业 / 技能」当必填：兴趣与平台在 prompt 里只是补充维度，
        # 这两项全空时 LLM 手里没有任何可分析的信息，只能编
        if not job and not skills:
            st.warning("请至少填写职业或技能再生成")
            return

        # 拼成一段带字段名的纯文本：工作流不解析，直接把整段塞进 prompt，
        # 所以这里的字段名与行序就是 LLM 实际看到的输入格式
        user_input = f"职业：{job}\n技能：{skills}\n兴趣：{interests}\n目标平台：{platform}"

        # 30~90 秒的依据：三个节点串行、每个节点一次 LLM 调用；
        # 把预期写进 spinner，免得用户以为页面卡死又点一次
        with st.spinner("AI 正在分析你的画像（三步串行，约 30~90 秒）..."):
            # 延迟到点击后才 import：这个模块导入时会建 LangGraph 图，
            # 放在这里页面至少能先把标题与输入框渲染出来
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
        # falsy 判断同时覆盖「没生成过」与「生成过但 result 是空 dict」两种情况；
        # 空白页会让人以为功能坏了，所以至少给一句操作引导
        st.info("填写上方信息后点击「生成定位方案」。")
        return

    profile = result.get("profile", "")
    competitors = result.get("competitors", "")
    plan = result.get("plan", "")

    # startswith 接受元组，一次比对六个前缀；只认前缀而不认子串，
    # 这样正文里恰好引用到「[LLM调用失败…」这类字样时不会误报
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
    # 输入原文从 session_state 取，不用按钮分支里的局部变量 ——
    # 渲染分支每次 rerun 都会重放，那时 user_input 早就不存在了
    st.caption(f"输入：{st.session_state.get('pos_input', '')}")
    tab1, tab2, tab3 = st.tabs(["📊 画像分析", "🔍 对标账号", "📝 完整方案"])
    with tab1:
        # `or` 兜底是课案原文的措辞；工作流保证失败也回填中文提示，
        # 所以真走到这里只可能是空串（给句占位文案好过一片空白）
        st.markdown(profile or "分析中...")
    with tab2:
        st.markdown(competitors or "搜索中...")
    with tab3:
        st.markdown(plan or "生成中...")
        # data 传「渲染时刻」的 plan（本轮从 session_state 恢复出来的那份）：
        # 下载按钮点击后会整页 rerun，按钮那次的局部变量那时已经不存在了
        st.download_button(
            "📥 下载方案",
            plan,
            file_name="账号定位方案.md",
            mime="text/markdown",
            width="stretch",
        )
