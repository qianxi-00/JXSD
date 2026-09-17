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

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 模块 docstring | 「路由到 ``ui/pages/`` 下各功能页面」 | 写明真实落点是 ``views/``，并交代两种启动方式 | 课案那句话与它自己下面 `from views.xxx import ...` 的写法自相矛盾（``ui/pages`` 应是早期版本残留），照抄会把读者带到不存在的目录 |
    | 路径引导 | 无（docstring 只写「启动: ``streamlit run main.py``」，隐含要先 cd 进子目录） | ``_HERE`` 插进 ``sys.path`` + stdout 转 UTF-8 | 补上引导后「仓库根 / 子目录」两种启动方式都能用；不补则从仓库根启动必 ``ModuleNotFoundError`` |
    | ``add_to_history()`` | 定义在 ``main.py`` 里（内联 ``from datetime import datetime``、``summary[:200]``） | **删除**，原地留几行说明 | 全仓 0 调用方；各页面自己都有同逻辑的 ``_add_history``，留两份容易只改一处 |
    | 侧边栏「项目信息」 | 1 行：``技术栈：LangGraph + Streamlit`` | 6 行：技术栈 / 课案名 / 文本模型 / 编排模型 / 百炼密钥 / 图片生成（末行按需） | 密钥没配时相关链路会静默降级，把状态摆出来能省掉一轮「点了没反应」的排查 |
    | 路由收尾 | ``route_fn = ROUTES.get(...)`` 再单独 ``route_fn()`` | 一行内联 ``ROUTES.get(route_key, show_home)()`` | 语义等价，少一个只出现一次的中间变量 |
    | ``set_page_config`` / ``PAGES`` / ``ROUTES`` | 原样 | **逐字一致**（6 行 / 9 行 / 9 行） | 这三处是「界面与课案对齐」的锚点，改它们等于改导航结构 |
    | 绝对路径 | 无 | 无 | —— |

踩过的坑
    · **``import streamlit as st`` 之前必须先做完 ``sys.path`` 引导**：
      直接从仓库根启动时，进程的工作目录不是 ``Media_Agent/``，
      ``from views.xxx import ...`` 会找不到包 —— 这正是 ``_HERE`` 要插进 ``sys.path`` 的原因。
      文件里那几处 ``# noqa: E402``（module level import not at top of file）是**故意**的：
      import 必须排在引导之后，工具链的默认告警在这里不适用。
    · **``st.set_page_config()`` 必须是第一个 ``st.*`` 调用**（Streamlit 的硬性要求），
      所以它排在最前面，前后那两条 ``# =====`` 分隔线也不能挪位置。
    · **``from views.xxx import ...`` 写在脚本中段**，而不是文件顶部：
      这些页面模块在 import 阶段就会用到 streamlit 对象，顺序必须晚于 ``set_page_config``。
    · 侧边栏 ``radio`` 每交互一次就触发整页 rerun，脚本从头重跑一遍 ——
      ``PAGES`` / ``ROUTES`` 每次 rerun 都会重新构造，别试图「缓存」它们。
    · 密钥状态那两行取值路径不一样：``settings.dashscope_api_key`` 是扁平字段
      （由 ``Settings.__getattr__`` 转发给 ``_CoreSettings``），
      而 ``settings.media.image_api_key`` 在分组的 ``settings.media`` 里，别写混。
"""

import sys
from pathlib import Path

# ---- 路径引导：必须在导入 views/workflows 之前执行 ----
# resolve() 是为了把相对路径变成绝对路径，避免 sys.path 里出现「同一个目录的两种写法」。
_HERE = Path(__file__).resolve().parent
# 幂等：已经有就跳过。插在队首是为了让项目内的模块优先于同名第三方包命中
#（views / workflows 都是很常见的名字）。
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Windows 控制台默认 GBK；不转 UTF-8 的话，各页面里带中文 / emoji 的输出会在控制台报编码错。
# hasattr 守卫：stdout 被换成没有 reconfigure 的对象时（IDE、被重定向的管道）不能硬调。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 下面两行必须晚于上面的路径引导，所以带 noqa: E402 —— 这是有意为之，不是遗漏。
import streamlit as st  # noqa: E402

# config 包在**仓库根**，靠 .venv 里的 python_base_root.pth 解析
#（实测该 .pth 内容就是仓库根那一行），所以两种启动方式下都可用。
from config import settings  # noqa: E402

# ==================== 初始化 ====================
# Streamlit 的硬性要求：set_page_config 必须是第一个 st.* 调用，否则启动直接报错。
# 四项取值与课案逐字一致（layout="wide" 是首页三列卡片布局的前提，别改成默认的 centered）。
st.set_page_config(
    page_title="自媒体AI创作平台",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "history" not in st.session_state:
    # 操作历史由各页面自己 append（见 views/*.py 的 _add_history）。
    # 课案在这里还带了一份 add_to_history()，但全仓 0 调用方，已删 ——
    # 两份同逻辑实现容易只改一处（各页面的 _add_history 已用 setdefault 加固）。
    # 只在主入口初始化一次；各页面用 setdefault 兜底，所以「经不经 main.py」都不会缺键。
    # 形状固定为 list[dict]，每条 3 个键（time / action / summary），首页按这个结构渲染。
    st.session_state.history = []


# ==================== 侧边栏导航 ====================
# 侧边栏与主区域是两套上下文：st.title() 画在主区域，st.sidebar.title() 画在侧栏，别写混。
st.sidebar.title("🎬 自媒体AI创作平台")
st.sidebar.markdown("---")

# 「菜单中文名 → 英文路由键」。dict 保序（Python 3.7+），所以这里的书写顺序就是菜单顺序。
# 九行与课案逐字一致 —— 改它等于改导航结构，不只是改文案。
PAGES = {
    "🏠 首页": "home",
    "🎯 账号定位": "positioning",
    "🔥 热点监控": "hot_topic",
    "📝 内容复刻": "replicate",
    "🎥 口播视频": "video",
    "🎬 视频剪辑": "mashup",
    "📊 数据复盘": "review",
}

# radio 每点一次都会触发整页 rerun：脚本从头重跑一遍，下面的路由分发因此天然生效，
# 不需要任何回调或状态机。
page = st.sidebar.radio("导航菜单", list(PAGES.keys()))

st.sidebar.markdown("---")
with st.sidebar.expander("📋 项目信息"):
    st.caption("技术栈：Streamlit + LangGraph + DeepAgents")
    st.caption("课案：3.自媒体Agent")
    # 模型名与密钥状态都从 config 现取（settings 是进程级单例，取用代价可忽略）。
    st.caption(f"文本模型：`{settings.media_llm_model()}`")
    st.caption(f"编排模型：`{settings.media_deepagent_model()}`")
    # 密钥状态一眼可见，省得排查「为什么一直降级」
    # 这里读的是扁平字段 dashscope_api_key（由 Settings.__getattr__ 转发给 _CoreSettings）。
    st.caption(
        "百炼密钥：" + ("✅ 已配置" if settings.dashscope_api_key else "⚠️ 未配置")
    )
    # 这一项在分组的 settings.media 里，取值路径与上面那条不同；只提示、不阻断。
    if not settings.media.image_api_key:
        st.caption("图片生成：⚠️ 未配置（将用占位图）")

# ==================== 路由 ====================
# 页面模块在这里才 import（而不是文件顶部）：它们在 import 阶段就会用到 streamlit 对象，
# 顺序必须晚于上面的 set_page_config；那几处 noqa: E402 同理，是有意为之。
from views.home import show_home  # noqa: E402
from views.hot_topic import show_hot_topic  # noqa: E402
from views.mashup import show_mashup  # noqa: E402
from views.positioning import show_positioning  # noqa: E402
from views.replicate import show_replicate  # noqa: E402
from views.review import show_review  # noqa: E402
from views.video import show_video  # noqa: E402

# 与 PAGES 的「值」一一对应：PAGES 决定菜单文案，ROUTES 决定点进去执行哪个函数。
# 两张表必须成对改（只改一张的后果见文件末尾的两级兜底）。
ROUTES = {
    "home": show_home,
    "positioning": show_positioning,
    "hot_topic": show_hot_topic,
    "replicate": show_replicate,
    "video": show_video,
    "mashup": show_mashup,
    "review": show_review,
}

# 两级兜底：PAGES.get 防「菜单里出现没有路由键的项」，ROUTES.get 防「PAGES 加了项却忘了加路由」——
# 两种情况都退回首页，而不是抛 KeyError 让界面白屏。
route_key = PAGES.get(page, "home")
# 直接在这里调用、不包成 if __name__ == "__main__"：Streamlit 每次 rerun 都重新执行整个脚本，
# 模块顶层顺手把页面渲染出来正是它要的模式。
ROUTES.get(route_key, show_home)()
