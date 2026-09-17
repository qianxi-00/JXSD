# -*- coding: utf-8 -*-
"""自媒体 Agent —— 能力层（tools/）

课案出处：自媒体课案 → 各模块的「工具函数」

三层架构里的最底层：每个文件封装一类外部能力，向上只暴露普通函数，
不掺 LangGraph / Streamlit 的东西，方便单独测试。

    media_tools.py         视频下载 / 音频提取 / 语音合成 / 图片生成 / 文章抓取
                           （``extract_audio_text`` 是「视频 → 文案」的入口，内部再调
                           ``audio_transcriber``）
    audio_transcriber.py   语音转文字（百炼 Qwen-Audio-3.0-ASR-Flash，替代课案的本地 FunASR）
    dashscope_upload.py    本地文件 → 百炼临时存储 URL（替代「自己搭对象存储」）
    asset_host.py          本地素材 → 公网 http(s) URL（自建服务器 scp + nginx 静态目录）
                           —— 声音克隆的硬前提：CosyVoice 不收 ``oss://``，只收 http(s)
    avatar_client.py       数字人对口型（百炼爱诗 PixVerse，替代课案的本地 HeyGem）
    voice_clone.py         声音复刻 + 克隆音色语音合成（百炼 CosyVoice，替代课案的本地 Fish-Speech）
    trend_radar_client.py  多平台热点抓取（TrendRadar / NewsNow 公共 API）
    douyin_client.py       抖音作品数据采集（自托管 Douyin_TikTok_Download_API）

谁在用这一层
    ``workflows/`` 与 ``views/`` 通过 ``from tools.x import y`` 取用：多数调用点写成
    **函数内的延迟导入**（真要走到那一步才 import），个别页面在模块级 import 并带
    ``# noqa: E402``（因为要先把自己的目录塞进 ``sys.path`` 才 import 得到 ``config``）。
    层内互相引用只有四处，且**全是函数内延迟导入**（模块 import 期互不依赖）：
    ``audio_transcriber`` / ``avatar_client`` → ``dashscope_upload``（各自的大文件分支）、
    ``media_tools`` → ``audio_transcriber``、``voice_clone`` → ``asset_host``。
    这一层不 import 上层，所以每个模块都能单独跑、单独测。

统一约定（所有 tools 模块都遵守，写新模块时照做）
    1. **绝不抛异常**：失败返回空串 / 空列表 / 带 ``success=False`` 的 dict。
       课案里每个调用点都在 LangGraph 节点里，节点抛异常整条图就断了。
    2. **依赖缺失即降级**：缺少密钥或服务不可达时打印中文提示并返回可用占位，
       应用整体仍能启动 —— 这是本仓库四个课案目录的共同风格。
    3. **打印中文日志**：形如 ``[模块名] 说明``，方便在 Streamlit 控制台追链路。
    4. **每个模块自带离线 ``__main__`` 自检**：``python tools/<模块>.py`` 直接跑，
       只验纯逻辑与失败分支（不联网、不动密钥、不产生费用），退出码 0 才算过。
       除本文件（纯文档、无可执行代码）外，其余 8 个模块都已登记在
       ``verify_all.py`` 第 1 层 ``MODULE_SELF_CHECKS`` 里，由它逐个执行。
"""
