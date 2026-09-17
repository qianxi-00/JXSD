# 《3.自媒体Agent》课案解读 + 本地模型能否改用外部 API

> 调研对象：`自媒体Agent.html`（2.42 MB，Typora 导出，标题「3.自媒体Agent」，1725 行 HTML / 正文约 15 万字）
> 解析产物：`_html_parse/courseware.txt`（去 CSS 噪音的正文）、`_html_parse/courseware_code_only.txt`（44 个代码块）
> 调研时间：本机 DSH 会话

---

## 一、课案到底讲了什么

### 1.1 一句话

从零手撸一个 **「自媒体 AI 创作全流程平台」**：Streamlit 做界面，LangGraph 做工作流编排，
把「账号定位 → 热点选题 → 内容复刻 → 口播视频 → 视频剪辑 → 数据复盘」六个环节串成运营闭环。

### 1.2 三层架构（贯穿全文的唯一范式）

```
views/*.py        前端页面     Streamlit 组件 + session_state 缓存
   ↓ 调用 run_xxx()
workflows/*.py    LangGraph     StateGraph 编排节点 → LLM 推理 → 结构化输出
   ↓ 调用工具函数
tools/*.py        能力层       下载/ASR/TTS/数字人/热点/爬虫
```

课案原文总结得很直白：**"新增功能时复制 views/ + workflows/ 配对即可快速扩展"**。
六个模块的代码结构完全一致，属于"同一套骨架换业务"。

### 1.3 六个模块的链路与目标

| 模块 | 目标 | 链路 | 编排特点 |
|---|---|---|---|
| 🎯 账号定位 | 输入背景 → 输出定位方案 | 画像分析 → 对标搜索 → 方案生成 | 纯串行，3 个 LLM 节点 |
| 🔥 热点监控 | 抓热点 → AI 筛选 → 选题建议 | 并行抓 5 平台 → 合并去重 → LLM 筛选 → 生成选题 | **LangGraph `Send` 并行**，`Annotated[list, operator.add]` 自动合并 |
| 📝 内容复刻 | 爆款链接 → 文案 → 爆款拆解 → 仿写 + 标题 | 下载视频 → ASR 提文案 → 拆爆款 5 维度 → 仿写 → 5 类标题 | 串行 4 节点，前置是媒体工具链 |
| 🎥 口播视频 | 台词 → 提词器 / 数字人出镜 | 提词器：纯前端滚动<br>数字人：声音克隆 → TTS → HeyGem 唇形驱动 | 双模式分支，**台词原样使用不改写** |
| 🎬 视频剪辑 | 口播视频后期（字幕/特效/转场/BGM） | deepagent 加载 `video-use` SKILL.md 全权负责：转录 → HyperFrames 生成 6~10 个动画素材 → 上下排布穿插 → BGM 混音 → 渲染 | **DeepAgents**（不是手写节点），`LocalShellBackend` 虚拟沙箱 |
| 📊 数据复盘 | 抖音真实数据 → 多维诊断 | 本地 douyin 爬虫采 20 条作品 → 漏斗诊断 → 内容评估 → 优化策略 | 串行 4 节点，首个节点接真实爬虫 |

### 1.4 技术栈（课案自述）

| 层 | 技术 | 备注 |
|---|---|---|
| 前端 | Streamlit | `session_state` 存结果防 rerun 丢数据（课案反复强调的点） |
| 工作流 | LangGraph | `StateGraph` / `Send` 并行 / `add_conditional_edges` |
| LLM | LangChain + OpenAI 兼容 API | **统一 `llm_call()` 入口，可换任意 OpenAI 兼容模型** |
| Agent 编排 | DeepAgents | 只用在一个地方：视频剪辑 |
| 视频下载 | videodl | 三级降级 videodl → yt-dlp → stub |
| 语音识别 | FunASR（阿里达摩院） | 本地运行 |
| TTS | Edge-TTS / Fish-Speech | 通用配音 + 克隆音色，自动降级 |
| 热点 | TrendRadar / NewsNow API | 公共 API，可自部署 `ourongxing/newsnow` |
| 数字人 | HeyGem（硅基智能开源） | Docker 三容器 |
| 动画渲染 | HyperFrames | npm CLI，HTML/CSS/GSAP → MP4 |
| 数据采集 | erma0/douyin 爬虫 | 本地项目 + CDP 从 Edge 读 Cookie |

### 1.5 课案里最值得抄的工程经验（不是代码，是踩坑）

这些是课案真正的价值所在，都是"跑出来才知道"的东西：

1. **DeepAgent 剪辑的硬约束**（写进了 system_prompt，很细）：
   - 字幕**只能** `SubtitlesClip` 加载 SRT，禁止 `for` 循环 + `TextClip`（会重复）
   - **禁止** `.with_mask()` / `.set_mask()` / `.to_mask()`（产生隐式蒙版）
   - `CompositeVideoClip` 的 `size` 必须 `(w, h + h//3)`，写 `(w, h)` 会把底部素材裁掉
   - `SubtitlesClip` 必须传 `make_textclip` + `with_position` 指定位置，否则默认贴到合成帧最底部（素材区）被裁
   - HyperFrames 生成的 HTML 每个 `<head>` 必须加中文字体 `<style>`，否则中文全是方框
2. **Windows 适配**：`sys.stdout.reconfigure(encoding='utf-8')`、`subprocess.Popen` monkey-patch（`CREATE_NO_WINDOW` + 1MB `bufsize` 防 ffmpeg stderr 撑爆管道）、`MSYS_NO_PATHCONV=1` 禁 Git Bash 路径转换。
3. **`normalize_path()`**：专门修 `C:\c\Users\...` / `/c/Users/...` 这类 Git Bash 与 Windows 路径混用产生的畸形路径，还补了 `LocalShellBackend._resolve_path`。
4. **`os.getenv` 读不到 `.env`**：必须走 `config.settings`（pydantic-settings 才加载 `.env`）。课案在代码注释里明确标注了。
5. **剪贴板路径**：热点并行抓取加 `random.uniform` jitter 间隔；HeyGem 用 `code=10000/10001` 区分"任务创建成功 / 服务忙碌"。
6. **CDP 读 Cookie**：不硬解 Edge 的加密 Cookie 库，直接 kill 掉 Edge → 用 `--remote-debugging-port` + `--headless=new` 拉起 → 走 `Network.getCookies` 域名查询 → 关闭进程。绕开了 v20 加密和文件锁。

### 1.6 实现路径（按课案章节顺序落地）

```
1. uv python pin 3.11
   uv add streamlit langgraph langchain langchain-openai pydantic pydantic-settings \
          python-dotenv videofetch requests funasr openai edge-tts Pillow httpx niquests
2. 写 config.py（Pydantic Settings 读 .env）+ workflows/base.py
   → ⚠️ 课案明确要求「先把 .env 和 config.py 写好」，与本机 AGENTS.md 的约定一致
3. 按模块推进，每个模块都是 views/ + workflows/ 一对：
   账号定位（最简单，先跑通骨架）
   → 热点监控（学 LangGraph Send 并行）
   → 内容复刻（接 media_tools + FunASR）
   → 口播视频（接 TTS + HeyGem）
   → 视频剪辑（上 DeepAgents + video-use skill，最难）
   → 数据复盘（接本地爬虫）
4. main.py 侧边栏路由把 7 个页面串起来
5. 前置环境（Windows）：winget install ffmpeg / npm install -g hyperframes
```

### 1.7 课案里几处明显的问题（照抄会踩）

读代码时发现的，不确认他是否已在真实仓库修过：

| 位置 | 问题 |
|---|---|
| `workflows/video.py` `node_generate_video` | 用了 `state['optimized']`，但 `VideoState` 里**没有** `optimized` 字段（只有 `raw_script`）→ 数字人模式必 `KeyError` |
| `workflows/hot_topic.py` 文件末尾 | 模块级执行 `graph = hot_topic_graph.get_graph()` + `graph.draw_png("positioning_workflow.png")` → import 就炸（需 pygraphviz），且图名/文件名张冠李戴 |
| `workflows/positioning.py` `run_positioning` | 调用未 import 的 `get_session()` / `PositioningRecord`（DB 层课案没给），被 `except Exception` 吞掉只打印一行 |
| `views/replicate.py` | f-string 内层复用了外层的 `"`（`f"""...{st.session_state.get("rep_url","")}..."""`）。**PEP 701 之前（3.11 及以下）这是 SyntaxError**，而课案 pin 的正是 3.11 —— 需实机确认 |
| `config.py`（课案展示版） | 只给了部分字段，但代码里还用了 `settings.image_gen_base_url` / `image_gen_model` / `get_image_api_key()` / `heygem_tts_host` / `get_heygem_tts_data_dir()` —— 展示的是精简版 |
| `workflows/mashup.py` vs `views/mashup.py` | 两边 `CACHE_DIR` 不一致（`.cache/videos` vs `.cache/mashup`），兜底查找扫不到 |
| `views/mashup.py` / `workflows/mashup.py` | `DEFAULT_BGM` 硬编码了 `C:\Users\13261\Pictures\风格\...wav`，是作者本机路径，直接跑必然找不到 |
| `workflows/review.py` | `DOUYIN_PROJECT` 硬编码 `C:/Users/13261/Documents/project/douyin` |

---

## 二、课案里"本地部署的模型"清单（重点）

课案里**真正**需要部署/下载权重的模型只有 **4 个（其中 2 个是 FunASR 的不同模型）**，全都集中在"语音"和"口型"两件事上：

| # | 模型 | 干什么 | 部署方式 | 服务接口 | 硬件门槛 |
|---|---|---|---|---|---|
| 1 | **FunASR `FunAudioLLM/Fun-ASR-Nano-2512` + `fsmn-vad`** | 语音转文字（内容复刻提文案） | ① AutoDL GPU 上跑自写 WS 服务 `deploy/funasr_ws_server.py`<br>② 本地 `funasr` pip 包兜底 | ① `ws://0.0.0.0:6006`，经 AutoDL 代理 `wss://u823409-….seetacloud.com:8443`<br>② `AutoModel(...).generate()` 直调 | 官方说 Nano CPU 可跑；课案 `device="cuda"` |
| 2 | **FunASR `seaco-paraformer-large` + `fsmn-vad` + `ct-punc`** | 带**句级时间戳**的转写（生成 SRT 字幕） | 同一个 WS 服务，懒加载 | 同一个 WS，请求帧带 `"return_timestamps": true` | GPU |
| 3 | **Fish-Speech 1.5**（`fish-speech-1.5` llama + `firefly-gan-vq-fsq-8x1024-21hz-generator.pth` 解码器） | 声音克隆 + 克隆音色 TTS | AutoDL 社区镜像 `fishaudio/fish-speech:v6.2`，RTX 3080 Ti 12G，`python -m tools.api_server --mode tts --listen 0.0.0.0:6006` | `https://u823409-852e-….seetacloud.com:8443`<br>`/v1/health`、`/v1/preprocess_and_tran`、`/v1/invoke` | **必须 GPU**（12G 起） |
| 4 | **HeyGem / Duix Avatar**（硅基智能开源数字人） | 唇形驱动：静音模特视频 + 音频 → 4K 口播视频 | ① 本地 Docker 三容器 `gen-video:8383` / `tts:18180` / `asr:10095`<br>② AutoDL 上 `HeyGem-Linux-Python-Hack/app.py`（Gradio@6006，SSH 隧道） | 视频生成 `POST /easy/submit`、`GET /easy/query` | **NVIDIA GPU + 32GB RAM（推荐 RTX 4070）** |

FunASR 模型在课案里被**三种方式**调用（替换时三条都要照顾到）：
1. `tools/media_tools.py:_get_funasr_model()` —— 本地 `AutoModel`，`device="cuda"`
2. `deploy/funasr_ws_server.py` —— GPU 服务器上的 WS 服务（内容复刻走这条，`ASR_MODE=remote`）
3. `workflows/mashup.py` 的 deepagent system_prompt 里 —— 让 agent 自己写脚本调 `iic/speech_seaco_paraformer_large_asr_nat-...`（剪辑字幕走这条）

### 2.1 容易被误当成"本地模型"的（其实不是）

| 东西 | 真相 |
|---|---|
| **Edge-TTS** | 本来就是调微软的**在线**服务，免费、无需 key。课案写 `tts_voice="zh-CN-XiaoxiaoNeural"` 就是 Edge 的音色名 |
| **HyperFrames** | 不是模型，是 npm CLI（`npx hyperframes`），内部用 headless Chromium 把 HTML/CSS/GSAP 渲成 MP4。课案还专门找 Edge/Chrome 的路径塞进 `CHROMIUM_PATH` |
| **videodl / yt-dlp** | 下载器，不是模型 |
| **erma0/douyin** | 爬虫代码，不是模型 |
| **LLM（DeepSeek）** | 本来就是外部 API（`https://api.deepseek.com/v1`） |
| **图片生成** | 本来就是外部 API，`generate_image()` 走 OpenAI 兼容的 `images.generate`，注释写了"支持 SiliconFlow、本地 Stable Diffusion 等" |

**关键洞察**：课案所谓的"本地部署"，本质是**"不走商业云 API"**，而不是"模型跑在这台 Windows 上"。
FunASR / Fish-Speech / HeyGem 全都跑在 **AutoDL GPU 实例**上，通过 HTTP/WS 访问 ——
也就是说，**课案从第一天起就是"客户端 → 远端服务"结构**，只不过远端是自己租的机器。
这一点决定了后面替换的成本极低（见第四节）。

---

## 三、本地机器现状核对（本会话实测）

| 项 | 结果 |
|---|---|
| `ffmpeg` | ✅ `F:\ProGramApp\ffmpeg-2026-01-19-git-43dbc011fa-essentials_build\bin\ffmpeg.exe` |
| `npx` | ✅ `F:\ProGramApp\nodejs\npx.ps1` |
| `uv` | ✅ `F:\ProGramApp\Anaconda\Scripts\uv.exe`（本会话 pwsh 调用被沙箱拒绝，见下） |
| `docker` | ✅ `C:\Program Files\Docker\Docker\resources\bin\docker.exe` |
| `nvidia-smi` | 存在，但本会话执行被沙箱拒绝（`拒绝访问`），**没能实测出显卡型号** |
| `hyperframes` | ❌ 未安装（`npm install -g hyperframes` 可补） |
| 相关项目 | `F:\ProGram\Python_Base\Media_Agent`（空目录）、`F:\Job Coding\ASR`（无关，是工作项目） |

> 远端资源：`dsh-ssh.json` 里有 3 台（含 A800 GPU 服务器），课案用的 AutoDL 实例不在清单里。

---

## 四、核心问题：这些本地模型能不能改用外部 API？

### 4.0 结论

**能，而且 4 个全都有等价物。** 但三件事的替代成本差别很大：

| 本地模型 | 替代难度 | 最推荐的外部 API | 结论 |
|---|---|---|---|
| FunASR（ASR） | ⭐ 极低，几乎是 **1:1** | 阿里云百炼 **Fun-ASR-Flash / Qwen-Audio-3.0-ASR-Flash** | **同一模型家族的云托管版**，连输出格式都对得上 |
| Fish-Speech（声音克隆 TTS） | ⭐ 低 | **Fish Audio 官方 Open API** | 就是同一家/同一谱系的模型 |
| HeyGem（唇形驱动） | ⭐⭐⭐ 中，但要算钱 | **fal.ai sync-lipsync v2** / **Fish Audio 口型同步** | 能力对得上，但 **$3/分钟**且素材要公网可访问 |
| （连带）FunASR 时间戳通道 | ⭐ 低 | 上面 ASR 接口自带句级+词级时间戳 | 直接能生成 SRT |

### 4.1 ASR —— 最容易，几乎是照抄

**首选：阿里云百炼 非实时语音识别（Fun-ASR-Flash / Qwen-Audio-3.0-ASR-Flash）HTTP API**

- 端点：`POST https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation`
- 模型：`fun-asr-flash-2026-06-15` / `qwen-audio-3.0-asr-flash` / `fun-asr-mtl-*`
- **同步返回**（不用排队等数分钟），SSE 可选
- 输入支持 **公网 URL 或 Base64 Data URI**（`data:audio/wav;base64,...`，≤10MB）→ **本地文件可以直接内嵌，不需要先传对象存储**，这是最关键的便利点
- 返回结构：
  ```
  output.sentence: { begin_time(ms), end_time(ms), text, sentence_end,
                     sentence_id, channel_id,
                     words: [{ text, begin_time, end_time, punctuation, fixed }] }
  ```
- → 这套 `begin_time/end_time` + `words[].punctuation` **正好能塞进课案已有的 `_normalize_sentences()` / `_split_sentences()`**，生成 SRT 的代码不用重写

**这条路线还有一个"历史巧合"**：课案原文写着
> 「语音识别：**从 DashScope 云 API 替换为** 阿里达摩院 FunASR（Fun-ASR-Nano + fsmn-vad），内置人声检测，支持 URL 直传，无需联网、完全本地可用」

也就是说，作者是**从云 API 换到本地的**。现在要换回去，等于走回头路 —— 但当年那个 DashScope 是老 batch 接口（`paraformer-v2` 录音文件识别），
**只收公网 URL、异步排队"数分钟内"**，体验确实不如本地 WS 同步返回。而现在 Fun-ASR-Flash 是同步 + 支持 base64，当年劝退的理由基本消失了。

**备选：**
- `paraformer-v2` 录音文件识别（老 batch）——有句级+词级时间戳，但**只接受公网 URL、异步排队**，不如 Flash 顺手
- OpenAI `whisper-1`（`verbose_json` 出 `segments` 带 start/end，能直接产 SRT）/ `gpt-4o-transcribe`；Groq 的 `whisper-large-v3-turbo` 更快更便宜 —— 中文标点和词级时间戳精度不如 Fun-ASR
- 火山引擎 / 腾讯云 / 讯飞 录音文件识别 —— 国内合规、有时间戳

**⚠️ 选型红线**：剪辑链路**必须要句级/词级时间戳**才能生成 SRT。
所以像 SenseVoice 这类**纯文本、无时间戳**的 ASR 不能用于剪辑字幕（可以用于内容复刻提文案）。

### 4.2 声音克隆 TTS —— 可替代，Fish Audio 官方云是"同一家人"

**首选：Fish Audio Open API（fishaudio.org）** —— 课案用的就是 fish-speech，迁移语义最自然

| 课案本地端点 | Fish Audio 云端对应 |
|---|---|
| `heygem_voice_clone()` → `POST {TTS}/v1/preprocess_and_tran`<br>`{format, reference_audio, lang}` → `asr_format_audio_url` + `reference_audio_text` | `POST /api/open/v1/voices`（multipart，`audioFiles` + `referenceText`）<br>→ `{voiceId}` |
| `heygem_tts_with_cloned_voice()` → `POST {TTS}/v1/invoke`<br>`{speaker, text, reference_audio, reference_text, topP, temperature...}` | 用 `voiceId` 调 Fish Audio 的 TTS 接口 |

**额外彩蛋**：Fish Audio 还有 `POST /api/open/v1/media/video-dubbing/jobs`
`{video_url, text, reference_id}` → **一次调用同时完成 TTS + 口型同步**。
这等于把课案的「声音克隆 + TTS + HeyGem 提交」三跳合并成一跳。

**国内备选：**
- **阿里云百炼 CosyVoice / Qwen-Audio-TTS 声音复刻**：`VoiceEnrollmentService.create_voice(target_model, prefix, url, language_hints, max_prompt_audio_length)` → `voice_id`，再合成。⚠️ `url` 要求**公网可访问**
- **MiniMax**：`/v1/files` 上传音频拿 `file_id` → `POST /v1/voice_clone` `{file_id, voice_id, ...}` → TTS
- **火山引擎 豆包语音 声音复刻 2.0**：上传音频 → 拿 `speaker_id` → 合成

**零成本路径**：课案本来就内置了 `edge-tts` 作为降级（免费、无需 key、**但不能克隆音色**）。
如果"用别人/自己的声音"不是硬需求，**Fish-Speech 整个模块可以不要**，只留 edge-tts。

### 4.3 数字人 / 唇形驱动 —— 能替代，但这是唯一要真算钱的地方

**先纠正一个容易搞错的对标方向**：
HeyGem 的 `/easy/submit` 入参是 `{audio_url, video_url}`，本质是 **video-to-video 唇形同步**
（一段静音模特视频 + 一段音频 → 口型对齐的视频），**不是**"文字生成数字人"。
所以对标的是 **lipsync API**，不是数字人 SaaS 平台。课案代码里也明确写了
"HeyGem 必须有模特视频（不是照片！）……照片只能生成静态画面"。

| 候选 | 入参/出参 | 计费 | 备注 |
|---|---|---|---|
| **fal.ai `fal-ai/sync-lipsync/v2`**（Sync Labs Lipsync 2.0） | `{video_url, audio_url, model: lipsync-2\|lipsync-2-pro, sync_mode: cut_off\|loop\|bounce\|silence\|remap}` → `{video:{url}}` | **$3/分钟视频**（pro $5/分钟） | 有 queue 模式（POST → request_id → 轮询），**和 HeyGem 的 submit/query 语义几乎一样** |
| **Fish Audio 口型同步** | `POST /api/open/v1/media/lip-sync/jobs` `{video_url, audio_url}` → `{id, status, credits_used, billing_duration_seconds}`；`GET .../jobs/{jobId}` → `result_url` | 按视频/音频**较短**时长向上取整（响应直接给 `credits_used`） | 和上面的 TTS 同一家，账号/额度可共用 |
| fal.ai `fal-ai/heygen/v3/lipsync/precision` | HeyGen 自家 lipsync，托管在 fal | — | — |
| Sync.so 官方 API / Replicate（`sync/lipsync-2`、`latentsync`、`sadtalker`） | 类似 | — | — |
| 国内：硅基智能 **Duix 开放平台**（就是 HeyGem 的厂商）/ 腾讯云数字人 / 百度曦灵 / 火山引擎虚拟人 | — | — | 多为"数字人 SaaS"，形态与"自备模特视频+音频"不完全对齐，需逐个确认 |

**本地文件怎么办？** fal 有 CDN 上传：`fal_client.upload_file("path/to/model.mp4")`
→ 返回 `https://v3b.fal.media/files/...`，直接当 `video_url` 用（大文件自动分片，10MB/块）。
Fish Audio 则要求 URL "能被服务端直接访问"，私有素材要给短期签名 URL，或自己搭个临时公网入口。

**⚠️ 这一环的真实取舍（要跟决策者摊开说的）：**

| 维度 | 本地 HeyGem | 外部 lipsync API |
|---|---|---|
| 边际成本 | 电费 ≈ 0 | **$3/分钟** × 一条 3 分钟口播 ≈ $9/条 |
| 批量生产 | 越跑越划算 | 跑得越多越贵 |
| 4K 超分 | 有（`chaofen` 参数） | 不一定有 |
| 数据合规 | 人脸视频不出本地 | **模特人脸视频要传到第三方** |
| 部署门槛 | ROOT：NVIDIA GPU + 32GB RAM，或租 AutoDL | 一张能跑 Streamlit 的机器即可 |
| 排队/并发 | 单任务（`code=10001` 忙碌） | 云端队列 |

→ **少量视频 → 外部 API 更省事；批量量产 → 本地 GPU 反而更便宜。**

### 4.4 三档替换方案（按"要不要显卡"分）

| 档 | ASR | TTS | 数字人 | 需要本地 GPU？ | 单条成本 |
|---|---|---|---|---|---|
| **A · 零本地** | 百炼 Fun-ASR-Flash | edge-tts（免费）或 CosyVoice/Fish Audio | fal sync-lipsync 或 Fish Audio lip-sync | **不需要** | 数字人那步 $3/分钟 |
| **B · 只留 ASR 本地**（推荐起步） | 本机 Fun-ASR-Nano（**CPU 可跑**，课案原话"Nano 模型轻量高效，CPU 可运行"） | edge-tts（免费） | 外部 API | **不需要** | $3/分钟 |
| **C · 全本地** | FunASR | Fish-Speech | HeyGem | A800 服务器 / AutoDL 实例 | 电费 |

**B 档是最务实的起点**：ASR 本地跑（免费、不上传音频），TTS 用免费的 edge-tts，
只有"数字人出镜"这一件事花钱 —— 而且它是**可选功能**，不做数字人整条链路就是零成本。

---

## 五、改造路线：为什么这次替换特别省事

**核心原因：课案的"本地"本来就是远端 HTTP/WS 服务，而且每个能力函数都已经写了降级链。**

课案里所有媒体函数都是这个形状：

```python
def download_video(url):       # videodl → yt-dlp → stub
def generate_tts(text):        # edge-tts → 空
def generate_image(prompt):    # OpenAI 兼容 API → _stub_generate_image
def audio_to_text(path):       # funasr
def heygem_voice_clone(video): # httpx.post(TTS_HOST/v1/preprocess_and_tran)
def heygem_query_task(code):   # httpx.get(HEYGEM_HOST/easy/query)
```

**所有远端地址都已经在 `.env` 里**（`FUNASR_WS_URL`、`HEYGEM_HOST`、`HEYGEM_TTS_HOST`），
LLM 和图片本来就是任意 OpenAI 兼容 `base_url`。

所以最小改动路径是：

### 方案 1（推荐）：**写协议适配器，业务代码一行不改**

在本地起一个薄壳服务，对外**仍然暴露课案原本的那几个端点**：

```
ws://localhost:xxxx            ← 扮 FunASR WS 服务   → 转发百炼 Fun-ASR-Flash HTTP
http://localhost:8383/easy/{submit,query}   ← 扮 HeyGem      → 转发 fal sync-lipsync 的 queue
http://localhost:18180/v1/{health,preprocess_and_tran,invoke} ← 扮 Fish-Speech → 转发 Fish Audio / CosyVoice
```

然后 `.env` 里把 URL 指回 `localhost` 即可。
**`workflows/` 和 `views/` 完全不用动**，风险最低，也最容易做 A/B（改回真服务只要换 URL）。

### 方案 2：在降级链里插一层 cloud 分支

给每个函数加 provider 开关，例如：
```python
# ASR_PROVIDER=cloud|local|remote
# TTS_PROVIDER=edge|cosyvoice|fishaudio
# AVATAR_PROVIDER=none|heygem|fal
```
改动比方案 1 大，但更"正统"，长期可维护性更好。

### 方案 3：只替换，不保留本地

直接删掉 FunASR/Fish-Speech/HeyGem 相关代码，`tools/` 瘦身。
适合确定不再自建 GPU 服务的情况。

---

## 六、待确认 / 建议下一步

1. **确认目标是哪一档**（A 零本地 / B 只留 ASR 本地 / C 全本地），这决定要不要买 API 额度。
2. **要不要先只打通 ASR 一条链**？它是收益最直接、风险最低的一块（内容复刻模块立刻从"要 GPU 服务器"变成"本机就能跑"），
   而且百炼支持 base64 内嵌，连对象存储都不用搭。
3. **数字人这条链要不要保留**？如果保留，先明确"每条视频愿意花多少钱"，$3/分钟这个量级是否能接受。
4. **本机显卡型号还没测出来**（本会话 `nvidia-smi` 被沙箱拒绝）。这决定 B 档能不能跑本地 ASR（不过课案说 Nano CPU 也能跑，大概率不影响）。
5. 课案里那几处硬编码路径（`C:\Users\13261\...`）和 `state['optimized']` 的 KeyError，**在真开始改之前要先修**，否则替换完也跑不通。

---

## 参考资料

- [阿里云百炼 · 非实时语音识别（Qwen-Audio-3.0-ASR-Flash/Fun-ASR-Flash）HTTP API](https://www.alibabacloud.com/help/zh/model-studio/fun-asr-flash-recorded-speech-recognition-http-api)
- [阿里云百炼 · Paraformer 非实时语音识别 Python SDK（paraformer-v2，句/词级时间戳）](https://www.alibabacloud.com/help/zh/model-studio/paraformer-recorded-speech-recognition-python-sdk)
- [阿里云百炼 · 声音复刻 Python SDK（VoiceEnrollmentService）](https://www.alibabacloud.com/help/zh/model-studio/voice-clone-python-sdk)
- [Fish Audio API · 声音克隆](https://docs.fishaudio.org/zh/docs/api-reference/voices/create)
- [Fish Audio API · 口型同步 / 视频配音](https://docs.fishaudio.org/zh/docs/api-reference/lip-sync/jobs)
- [fal.ai · Sync Lipsync 2.0（$3/分钟）](https://fal.ai/models/fal-ai/sync-lipsync/v2) ｜ [API / OpenAPI schema](https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/sync-lipsync/v2)
- [fal.ai · fal CDN 文件上传（fal_client.upload_file）](https://fal.ai/docs/documentation/model-apis/fal-cdn)
- [MiniMax API · Voice Clone](https://platform.minimax.io/docs/api-reference/voice-cloning-clone)
- [Duix（硅基智能开源数字人，HeyGem 上游）文档](http://duix.guiji.ai/duix-document/questions/)
- 课案内引用：[TrendRadar (sansan0/TrendRadar)](https://github.com/sansan0/TrendRadar)、[HeyGem.ai (GuijiAI)](https://github.com/GuijiAI/HeyGem.ai)、erma0/douyin、CharlesPikachu/videodl
