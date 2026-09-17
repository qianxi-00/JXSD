# -*- coding: utf-8 -*-
"""自媒体 AI 创作全流程平台 —— Streamlit 入口

课案出处：自媒体课案 → 整合 → 主界面入口

启动方式（两种都行，见 README）：:

    # 方式一：从仓库根目录启动（推荐，与其它课案目录一致）
    uv run streamlit run Media_Agent/main.py

    # 方式二：进入子项目目录启动（与课案原文一致）
    cd Media_Agent
    uv run streamlit run main.py

为什么两种都能跑
    下面的 ``_bootstrap_path()`` 把 Media_Agent 目录塞进 ``sys.path``，
    这样 ``from views.xxx import ...`` / ``from workflows.xxx import ...``
    在任意工作目录下都能解析。
    （根目录的 ``config`` 包则靠 .venv 里的 ``python_base_root.pth`` 解析，
      两种启动方式下都可用。）
"""

import sys
from pathlib import Path

# ---- 路径引导：必须在导入 views/workflows 之前执行 ----
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st  # noqa: E402

from config import settings  # noqa: E402

# ==================== 初始化 ====================
st.set_page_config(
    page_title="自媒体AI创作平台",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "history" not in st.session_state:
    st.session_state.history = []


def add_to_history(action: str, summary: str) -> None:
    """记录一条操作历史（首页会展示最近 10 条）。"""
    from datetime import datetime

    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        "summary": summary[:200],
    })


# ==================== 侧边栏导航 ====================
st.sidebar.title("🎬 自媒体AI创作平台")
st.sidebar.markdown("---")

PAGES = {
    "🏠 首页": "home",
    "🎯 账号定位": "positioning",
    "🔥 热点监控": "hot_topic",
    "📝 内容复刻": "replicate",
    "🎥 口播视频": "video",
    "🎬 视频剪辑": "mashup",
    "📊 数据复盘": "review",
}

page = st.sidebar.radio("导航菜单", list(PAGES.keys()))

st.sidebar.markdown("---")
with st.sidebar.expander("📋 项目信息"):
    st.caption("技术栈：Streamlit + LangGraph + DeepAgents")
    st.caption("课案：3.自媒体Agent")
    st.caption(f"文本模型：`{settings.media_llm_model()}`")
    st.caption(f"编排模型：`{settings.media_deepagent_model()}`")
    # 密钥状态一眼可见，省得排查「为什么一直降级」
    st.caption(
        "百炼密钥：" + ("✅ 已配置" if settings.dashscope_api_key else "⚠️ 未配置")
    )
    if not settings.media.image_api_key:
        st.caption("图片生成：⚠️ 未配置（将用占位图）")

# ==================== 路由 ====================
from views.home import show_home  # noqa: E402
from views.hot_topic import show_hot_topic  # noqa: E402
from views.mashup import show_mashup  # noqa: E402
from views.positioning import show_positioning  # noqa: E402
from views.replicate import show_replicate  # noqa: E402
from views.review import show_review  # noqa: E402
from views.video import show_video  # noqa: E402

ROUTES = {
    "home": show_home,
    "positioning": show_positioning,
    "hot_topic": show_hot_topic,
    "replicate": show_replicate,
    "video": show_video,
    "mashup": show_mashup,
    "review": show_review,
}

route_key = PAGES.get(page, "home")
ROUTES.get(route_key, show_home)()
