# -*- coding: utf-8 -*-
"""自媒体 Agent —— Streamlit 页面包

课案出处：自媒体课案 → 各模块的 Streamlit 页面

每个文件一个页面函数（``show_xxx()``），由 ``main.py`` 的侧边栏路由调用。
页面只做两件事：收集输入、渲染 ``workflows/`` 返回的结果 ——
**不写业务逻辑，也不直接调 LLM**，这样工作流可以脱离界面单独测试。

统一约定
    · 结果一律先存 ``st.session_state`` 再渲染。
      Streamlit 的下载按钮会触发整页 rerun，不缓存结果页面会变空白 ——
      这是课案反复强调的坑（见各页面注释）。
    · 每次操作往 ``st.session_state.history`` 追加一条，首页展示最近 10 条。
    · 页面顶部标题用 emoji + 中文名，与课案一致。
"""
