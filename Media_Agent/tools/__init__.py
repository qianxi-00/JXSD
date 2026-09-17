# -*- coding: utf-8 -*-
"""自媒体 Agent —— 能力层（tools/）

课案出处：自媒体课案 → 各模块的「工具函数」

三层架构里的最底层：每个文件封装一类外部能力，向上只暴露普通函数，
不掺 LangGraph / Streamlit 的东西，方便单独测试。

    media_tools.py         视频下载 / 音频提取 / 语音合成 / 图片生成 / 文章抓取
    audio_transcriber.py   语音转文字（百炼 Qwen-Audio-3.0-ASR-Flash，替代课案的本地 FunASR）
    dashscope_upload.py    本地文件 → 百炼临时存储 URL（替代「自己搭对象存储」）
    avatar_client.py       数字人对口型（百炼爱诗 PixVerse，替代课案的本地 HeyGem）
    trend_radar_client.py  多平台热点抓取（TrendRadar / NewsNow 公共 API）
    douyin_client.py       抖音作品数据采集（自托管 Douyin_TikTok_Download_API）

统一约定（所有 tools 模块都遵守，写新模块时照做）
    1. **绝不抛异常**：失败返回空串 / 空列表 / 带 ``success=False`` 的 dict。
       课案里每个调用点都在 LangGraph 节点里，节点抛异常整条图就断了。
    2. **依赖缺失即降级**：缺少密钥或服务不可达时打印中文提示并返回可用占位，
       应用整体仍能启动 —— 这是本仓库四个课案目录的共同风格。
    3. **打印中文日志**：形如 ``[模块名] 说明``，方便在 Streamlit 控制台追链路。
"""
