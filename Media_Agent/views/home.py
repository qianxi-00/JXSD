# -*- coding: utf-8 -*-
"""首页 —— 功能总览 + 本次操作记录

课案出处：自媒体课案 → 整合 → 主界面入口 → views/home.py

与课案一致：三列卡片展示六大功能，每张卡写「一句话简介 + 核心链路」。
与课案的差异：卡片里的链路描述改成了**本项目实际用的技术**
（课案写的是本地 FunASR / HeyGem，本项目是百炼托管 API），
免得界面在骗人。
"""

import streamlit as st


def show_home() -> None:
    """首页：功能总览 + 本次操作记录。"""
    st.title("🎬 自媒体AI创作全流程平台")
    st.caption(
        "课案《3.自媒体Agent》实现 —— 本地模型已全部替换为阿里云百炼托管 API，"
        "本机不需要 GPU，也不下载任何模型权重。"
    )

    # 三列功能卡片：(图标, 标题, 简介, 核心链路)
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
        with cols[i % 3]:
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
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ---- 环境自检：把「为什么某功能一直降级」直接摆在首页 ----
    from config import settings

    st.markdown("### 🔧 运行环境")
    c1, c2, c3 = st.columns(3)
    c1.metric("文本模型", settings.media_llm_model())
    c2.metric(
        "百炼密钥",
        "已配置" if settings.dashscope_api_key else "未配置",
        help="语音识别 / 声音复刻 / 数字人对口型 都依赖它（根目录 .env 的 DASHSCOPE_API_KEY）",
    )
    c3.metric(
        "图片生成",
        "已配置" if settings.media.image_api_key else "未配置（用占位图）",
    )
    st.caption(
        "百炼的语音识别 / 声音复刻 / 视频对口型**只在华北2（北京）**提供，"
        "需使用该地域的 API Key。"
    )

    # ---- 本次操作记录（课案功能，保留）----
    history = st.session_state.get("history", [])
    if history:
        st.markdown("---")
        st.markdown("### 📋 本次操作记录")
        for h in reversed(history[-10:]):
            st.caption(f"🕐 {h['time']} | {h['action']} | {h['summary'][:80]}...")
    else:
        st.info("还没有操作记录。从左侧导航选一个功能开始吧。")
