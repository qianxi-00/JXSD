# -*- coding: utf-8 -*-
"""首页 —— 功能总览 + 本次操作记录

课案出处：自媒体课案 → 整合 → 主界面入口 → views/home.py

与课案一致：三列卡片展示六大功能，每张卡写「一句话简介 + 核心链路」。
与课案的差异：卡片里的链路描述改成了**本项目实际用的技术**
（课案写的是本地 FunASR / HeyGem，本项目是百炼托管 API），
免得界面在骗人。

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 模块 / 函数 docstring | 模块 docstring 只有「show_home页面」五个字，函数 docstring 只有「首页」两个字 | 说明性的模块 docstring + 带 Returns 的函数 docstring | 首页是新读者第一个打开的文件，「卡片文案为什么跟课案不一样」写在这里，省得以后有人照着课案「改回去」 |
    | 卡片文案 | 6 张卡共 24 个字段 | 6 张卡的图标与标题逐字一致，卡 1「账号定位」4 个字段全一致；其余 5 张的「简介 / 核心链路」按本项目技术栈改写 | 本地 FunASR / HeyGem / videodl / HyperFrames 已换成百炼托管 API + yt-dlp + PixVerse；照抄课案会让界面描述与实际链路对不上 |
    | HTML 卡片模板 | 内联样式那 11 行 | **内容逐行一致**（课案提取件用 U+00A0 当缩进，行内文本相同） | 版式属另一件事，本轮不动 |
    | 操作历史读取 | ``if st.session_state.history:`` 直接取属性 | ``st.session_state.get("history", [])`` | 直接取属性在 key 不存在时抛 ``AttributeError``，``get`` 兜底让首页能脱离 ``main.py`` 单独渲染 |
    | 空历史 | 无 else 分支（整块不显示） | 多一行 ``st.info("还没有操作记录…")`` | 首次进来时那块是空的，用户不知道这个区域是干什么的 |
    | 运行环境自检区 | 无 | 三个 ``st.metric`` + 「华北2（北京）」地域提示 + 「♻️ 重置运行时缓存」按钮 | 「为什么某功能一直降级」是最高频的疑问，直接摆在首页，省掉一轮排查；改完根 ``.env`` 也有一个（不完整的）出口，见下方踩坑 |
    | 绝对路径 | 无 | 无 | —— |

踩过的坑
    · ``st.session_state.get("history", [])`` 这个兜底不是多余的：``main.py`` 里那段
      初始化只在**主入口**跑过，任何「不经过 ``main.py`` 就调用 ``show_home()``」的场合
      （无头渲染、单独测一个页面）拿到的 ``session_state`` 都是空的，
      直接写 ``st.session_state.history`` 会 ``AttributeError``。
    · ``history`` 里每条固定是 ``{"time", "action", "summary"}`` 三个键
      （各页面的 ``_add_history`` 统一这么写）。首页按这三个键取值 ——
      加新页面时照抄这个结构，少一个键首页就会在渲染时炸。
    · 卡片简介里的 ``\n`` 靠 CSS ``white-space:pre-line`` 才换行；
      去掉那条样式，两行简介会挤成一行。
    · ``unsafe_allow_html=True`` 是必需的：Streamlit 默认会把 HTML 当纯文本转义，
      卡片会原样把 ``<div style=...>`` 打出来。
    · **「♻️ 重置运行时缓存」不等于重启**：``config.py`` 的 ``get_settings()`` 是
      ``@lru_cache`` 的进程级单例，模块级 ``settings = get_settings()`` 又已被各模块
      ``from config import settings`` 各自绑定过引用 —— 清缓存只影响**之后**的
      ``get_settings()`` 调用，页面与工作流手里的那个旧对象不会变。所以按钮文案
      必须如实说明「点完仍是旧值就重启」，不能写成等价于重启。
"""

import streamlit as st


def show_home() -> None:
    """首页：功能总览 + 本次操作记录。

    由 ``main.py`` 的侧边栏路由调用；无参数、无返回值，
    所有内容都渲染在 Streamlit 的当前页面上下文里。

    Returns:
        None。全部效果通过 ``st.*`` 调用产生。
    """
    # 标题与课案逐字一致（26 字符），别为了排版顺手改 —— 它是「界面与课案对齐」的可见锚点。
    st.title("🎬 自媒体AI创作全流程平台")
    st.caption(
        "课案《3.自媒体Agent》实现 —— 本地模型已全部替换为阿里云百炼托管 API，"
        "本机不需要 GPU，也不下载任何模型权重。"
    )

    # 三列功能卡片：(图标, 标题, 简介, 核心链路)
    # 顺序与 main.py 侧边栏的 PAGES 一致（首页 → 账号定位 → … → 数据复盘），方便两边对照。
    # 六张卡按「三列两行」排：靠下面的 i % 3 回绕。
    cards = [
        ("🎯", "账号定位", "输入背景信息\nAI生成专属定位方案",
         "分析画像 → 对标账号 → 输出方案"),
        ("🔥", "热点监控", "实时追踪平台热点\n筛选赛道相关选题",
         "NewsNow 抓取 → LLM筛选 → 选题建议"),
        ("📝", "内容复刻", "下载爆款视频\n拆解结构 → 仿写新文案",
         "yt-dlp下载 → 百炼ASR转文字 → LLM仿写"),
        ("🎥", "口播视频", "文案 → 提词器 / 数字人出镜",
         "克隆音色 → TTS配音 → PixVerse对口型"),
        ("🎬", "视频剪辑", "口播视频后期剪辑\n自动加字幕/素材/BGM",
         "DeepAgent + video-use技能 → MP4"),
        ("📊", "数据复盘", "真实作品数据\n多维诊断优化策略",
         "抖音采集 → 漏斗诊断 → 内容评估 → 优化策略"),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc, flow) in enumerate(cards):
        # i % 3 让第 4 张卡回到第一列（第二行开头）；直接写 cols[i] 会在 i=3 时 IndexError。
        with cols[i % 3]:
            # 内联 HTML 卡片的样式与课案一致：min-height 让两行文字长短不一的卡片等高，
            # white-space:pre-line 才认简介里的换行符。
            st.markdown(
                f"""
                <div style="padding:20px; border:1px solid #e0e0e0;
                     border-radius:10px; margin:8px 0; text-align:center;
                     min-height:180px;">
                <h1>{icon}</h1>
                <h4>{title}</h4>
                <p style="color:#555;font-size:13px;white-space:pre-line">{desc}</p>
                <hr style="margin:8px 0;">
                <small style="color:#999;">{flow}</small>
                </div>
                """,
                # 必须显式打开：Streamlit 默认把 HTML 当纯文本转义，卡片会原样打出标签。
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ---- 环境自检：把「为什么某功能一直降级」直接摆在首页 ----
    # 放在函数体内延迟导入：模块顶层只依赖 streamlit，
    # 这样 import views.home（verify_all.py 第 2 层做的事）不会连带去读 config / 根 .env。
    from config import settings

    st.markdown("### 🔧 运行环境")
    # 用 metric 而不是 caption：这三项是状态值，要能一眼扫到，而不是当成说明文字去读。
    c1, c2, c3 = st.columns(3)
    c1.metric("文本模型", settings.media_llm_model())
    c2.metric(
        "百炼密钥",
        "已配置" if settings.dashscope_api_key else "未配置",
        # help 里点明依赖的是根 .env 的哪一项 —— 「该改哪个文件」是最高频的追问。
        help="语音识别 / 声音复刻 / 数字人对口型 都依赖它（根目录 .env 的 DASHSCOPE_API_KEY）",
    )
    c3.metric(
        "图片生成",
        # 判据必须与 `tools/media_tools.py` 里真正生效的那处 `_image_configured()`
        # **保持一致**：它要求 `image_api_key` 与 `image_model` 两项都非空。
        # 只看 KEY 的话，「配了 KEY、没配 MODEL」会被首页说成「已配置」，
        # 实际那次调用仍然降级成占位图 —— 界面在骗人，而用户会去查网络
        "已配置" if (settings.media.image_api_key and settings.media.image_model)
        else "未配置（用占位图）",
        help="根目录 .env 的 MEDIA_IMAGE_API_KEY 与 MEDIA_IMAGE_MODEL 两项都配置才算已配置",
    )
    # 这条是提醒而不是校验：代码不判断 Key 属于哪个地域，用错了要等调用被拒才发现。
    st.caption(
        "百炼的语音识别 / 声音复刻 / 视频对口型**只在华北2（北京）**提供，"
        "需使用该地域的 API Key。"
    )

    # ---- 运行时缓存重置：改完 .env 不必重启 Streamlit ----
    # 上一轮点按钮时置的一次性标记：`st.rerun()` 会把那一轮连同 st.success 一起丢掉，
    # 所以提示要留到下一轮再兑现
    if st.session_state.pop("_runtime_cache_reset", False):
        st.success(
            "已重置运行时缓存（`get_settings` 的 lru_cache + 模型实例缓存都清了），"
            "下次调用会重新读根目录 `.env`。"
        )

    col_tip, col_reset = st.columns([4, 1])
    with col_tip:
        st.caption(
            "改完根 `.env`（密钥 / 模型名 / 平台开关）不用重启 Streamlit：点一下 → 的两个缓存都会清掉。"
            "**但它不等于重启**：`config.py` 里的 `settings` 是导入时就 `get_settings()` 好的那个对象，"
            "各模块又各自 `from config import settings` 拿了引用 —— 页面上的模型名 / 密钥要是点完还是旧的，"
            "请重启 Streamlit。"
        )
    with col_reset:
        if st.button("♻️ 重置运行时缓存", width="stretch",
                     help="改完根 .env 后点一下，省掉重启 Streamlit（缓存清不掉的旧值仍需重启）"):
            # 两个 import 都放按钮分支里：页面顶部导入会连带拉起 langchain 初始化，
            # 而首页 99% 的时间只是在看卡片，没必要为此付启动成本
            from config import get_settings
            from workflows import clear_model_cache

            # 顺序有意义：先让 .env 能被重新读取（`settings` 是 @lru_cache 的进程级单例），
            # 再清按 temperature 缓存的模型实例 —— 反过来的话，模型缓存下次仍会用旧配置重建
            get_settings.cache_clear()
            clear_model_cache()
            # 置标记再 rerun，让成功提示在干净的一轮里显示（见上面的 pop）
            st.session_state["_runtime_cache_reset"] = True
            st.rerun()

    # ---- 本次操作记录（课案功能，保留）----
    # get(..., []) 兜底：main.py 里的初始化只在主入口跑过，
    # 不经过 main.py 直接渲染首页时 session_state 里根本没有 history 这个键。
    history = st.session_state.get("history", [])
    if history:
        st.markdown("---")
        st.markdown("### 📋 本次操作记录")
        # 先切片再反转：history[-10:] 只取最近 10 条，reversed 让最新一条排在最上面。
        # 10 条是首页的显示预算 —— 再多会把上面的功能卡片挤出首屏（完整历史仍在 session_state 里）。
        for h in reversed(history[-10:]):
            # 各页面写入时把 summary 截到 200 字符（见它们的 _add_history），
            # 这里再截到 80：caption 只占一行，展示预算和存储预算不是一回事。
            st.caption(f"🕐 {h['time']} | {h['action']} | {h['summary'][:80]}...")
    else:
        # 没有 history 键、或列表为空，都落到这里（首次进入就是这个状态）。
        st.info("还没有操作记录。从左侧导航选一个功能开始吧。")
