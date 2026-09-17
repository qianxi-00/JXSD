# VERIFY_REPORT —— Media_Agent 实测记录

> 对应课案：《3.自媒体Agent》
> 验证时间：2026-09-17
> 运行环境：Windows / PowerShell，Python 3.12.12（`F:\ProGram\Python_Base\.venv`）
> 复现命令见每一节，全部可原样再跑一遍。

---

## 结论摘要

| 验证项 | 结果 |
|---|---|
| 模块自检（13 个模块 × 子进程） | **13/13 通过** |
| Streamlit 视图导入（7 个页面） | **通过** |
| 环境契约检查（配置 / 依赖 API 形状 / 关键文件 / 降级路径） | **通过** |
| 真实联网检查（LLM / 热点 / 百炼接线） | **4/4 通过** |
| Streamlit 应用启动 + 七页渲染 | **通过（0 console error / 0 stException）** |
| 端到端链路（账号定位 / 热点监控 / 数据复盘 / 内容复刻） | **4/4 通过** |
| 百炼 ASR 真实转写（208 秒音频 → 文本 + SRT） | **通过**，并**修复了 2 个真 bug**（见 5.4） |
| 声音克隆 / 数字人出片 / DeepAgents 出片 / 抖音真实采集 | **未验证**（见第 8 节） |

**一键复现**：

```powershell
Set-Location F:\ProGram\Python_Base\Media_Agent

# 离线（零密钥也必须全绿）
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' verify_all.py

# 附加真实 API 调用
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' verify_all.py --live
```

---

## 1. 验证环境

| 项 | 值 |
|---|---|
| 操作系统 | Windows（PowerShell 7） |
| Python | 3.12.12，`F:\ProGram\Python_Base\.venv`（与仓库其它子项目**共用**） |
| 文本模型 | `grok-4.6` @ `https://qianxi7988.me/v1`（根 `.env` 的 `MODEL_NAME` / `BASE_URL`） |
| 百炼 | `DASHSCOPE_API_KEY` 已配置，端点 `https://dashscope.aliyuncs.com/api/v1` |
| ffmpeg / yt-dlp / edge-tts | 均已安装可用 |
| HyperFrames | **未安装**（剪辑的动画素材会走 moviepy 降级分支） |
| Docker 服务 | **未启动**（抖音采集走降级入口） |
| GPU | 不需要（本项目不含本地模型） |

---

## 2. 模块自检（第 1 层）

**命令**：`verify_all.py` 第 1 层，每个模块用独立子进程跑它自己的 `__main__` 自检块。

```
  ✓ tools/dashscope_upload.py          退出码=0    0.3s
  ✓ tools/audio_transcriber.py         退出码=0    0.4s
  ✓ tools/media_tools.py               退出码=0    0.7s
  ✓ tools/trend_radar_client.py        退出码=0    0.4s
  ✓ tools/voice_clone.py               退出码=0    0.3s
  ✓ tools/avatar_client.py             退出码=0    0.4s
  ✓ tools/douyin_client.py             退出码=0    4.5s
  ✓ workflows/positioning.py           退出码=0    1.0s
  ✓ workflows/hot_topic.py             退出码=0    1.1s
  ✓ workflows/replicate.py             退出码=0    1.0s
  ✓ workflows/video.py                 退出码=0    1.0s
  ✓ workflows/mashup.py                退出码=0    1.0s
  ✓ workflows/review.py                退出码=0    1.1s
```

每个自检都是**纯逻辑断言，不联网、不花钱**，覆盖：

- **`audio_transcriber`**：毫秒→SRT 时间格式化、`sentence` 字段校验（含时间倒挂/空文本应被拒）、
  词级时间戳按标点聚合为句级（移植自课案 `_split_sentences`）、无标点兜底、失败路径返回结构化结果、SRT 文件写出。
- **`trend_radar_client`**：热度估算单调递减、平台映射表完整。
- **`voice_clone`**：音色指纹稳定性（文件变化后指纹必须变）、失败路径结构化返回。
- **`avatar_client`**：`audio_path` 与 `tts_text` 互斥校验、失败路径结构化返回。
- **`mashup`**：`normalize_path()` 的 4 种畸形路径修复（`C:\c\Users` / `C:/c/Users` / `C:\d\Data` / `/c/Users`）、
  **SKILL.md frontmatter 格式合法**、system_prompt 含 5 条关键约束（cmd.exe / SubtitlesClip 子模块路径 /
  `h + h//3` / `transcribe_to_srt` / 中文字体）。
- **`video`**：提词器模式原样返回、数字人缺模特给中文提示、
  **课案的 `state['optimized']` KeyError 已修复**（用字节码常量断言，不受注释干扰）。
- **`positioning`**：给 `llm_call` 打桩把整张图真跑一遍，断言 3 个节点依次串接且 prompt 里带上了上游产出。
- **`review`**：图结构 6 节点 5 条边、`parse_manual_json` 三种输入格式。

---

## 3. 环境契约检查（第 3 层）

这一层是「升级依赖前先跑一下」的护栏，验证代码对第三方库的假设仍然成立：

```
文本模型: grok-4.6
编排模型: grok-4.6
百炼端点: https://dashscope.aliyuncs.com/api/v1
百炼密钥: 已配置
环境契约检查全部通过
```

覆盖内容：

1. **配置字段存在性**：`settings.media.*` 的 10 个字段、根 `settings` 的 6 个字段。
2. **目录方法返回绝对路径且已自动创建**（避免课案 `.cache/videos` 随 CWD 漂移的坑）。
3. **模型名回退链**：`MEDIA_LLM_MODEL` 留空时必须能回落到根 `MODEL_NAME`（实测回落成功）。
4. **DeepAgents API 形状**（`deepagents 0.7.13`）：
   `LocalShellBackend.__init__` 含 `root_dir / virtual_mode / inherit_env / timeout / max_output_bytes / env`；
   `_resolve_path` 存在（`mashup.py` 的路径补丁依赖它）；`create_deep_agent` 含 `model / backend / system_prompt / skills`。
5. **dashscope SDK 形状**（`1.27.4`）：
   `VoiceEnrollmentService.create_voice` 含 `target_model`；
   **`VideoSynthesis.async_call` 含 `media` 参数**（PixVerse 对口型依赖，这是关键假设）；
   `SpeechSynthesizer.call` 存在；`OssUtils.upload` 含 `model` 参数。
6. **关键文件**：`.skills/video-use/SKILL.md` 存在且 frontmatter 合法。
7. **缺密钥降级**：`transcribe()` / `clone_voice()` / `submit_lipsync()` 在缺参/缺密钥时返回结构化结果而不是抛异常。

---

## 4. 真实联网检查（`--live`）

```
    [热点] 微博 获取成功（最新，30 条）

      ✓ LLM 文本推理: 收到
      ✓ 热点抓取（NewsNow）: 微博热榜 30 条，第 1 条: 影视飓风Tim反掰iPhoneDuo被质疑
      ✓ 百炼 ASR 接线: 参数校验通过（真实转写需要一段音频，见 README 手动验证）
      ✓ 数字人接线: 模型 pixverse/pixverse-lipsync；真实提交需要一段人脸视频，且需先在百炼控制台开通 PixVerse
```

- **LLM 文本推理**：真实调用中转站，返回「收到」。
- **热点抓取**：真实抓到微博热榜 30 条，第 1 条标题已记录（可复现）。
- 另有一次独立的自检跑了 `--net`，抓到同样的 30 条，说明接口稳定。

---

## 5. 端到端 LLM 链路

### 5.1 账号定位（真实 LLM，3 节点串行）

**命令**：

```powershell
Set-Location F:\ProGram\Python_Base\Media_Agent
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -u -c "
import sys; sys.path.insert(0, '.')
from workflows.positioning import run_positioning
r = run_positioning('职业：Python后端开发\n技能：Python、Docker、K8s、LangGraph\n兴趣：AI工具、效率提升、自动化\n目标平台：B站')
for k in ('profile','competitors','plan'): print(k, len(r[k]))
"
```

**结果**（exit 0，耗时 **324.9s**）：

```
[账号定位] ① 画像分析中...
[账号定位] ② 搜索对标账号中...
[账号定位] ③ 生成定位方案中...
profile     1793 字
competitors 1223 字
plan        2160 字
```

实际产出片段：

- **profile**：`**1. 专业优势（核心竞争力）** 你的核心竞争力非常清晰且稀缺：**"Python后端 + LangGraph + 容器化"** 三者高度融合的复合型能力。…`
- **competitors**：`**2个头部大号（天花板参考）** **1. 账号：硅基流动** 粉丝量级：200万+ 内容特色：深度技术向…`
- **plan**：`**1. 一句话定位** "把复杂LangGraph Agent真正跑在生产的K8s集群上…"`

链路验证点：三个节点确实依次串接，后一个节点的 prompt 带上了前一个的产出。

### 5.2 热点监控（真实抓取 + LLM 筛选）

**结果**（exit 0，耗时 **177.8s**）：

```
[热点] 抖音 获取成功（最新，30 条）
[热点] 微博 获取成功（缓存，30 条）
[热点] 知乎 获取成功（缓存，20 条）
[热点] B站热搜 获取成功（缓存，30 条）
[热点] 小红书 失败（500 Server Error），重试后仍失败 → 抓取完成: 0 条
[热点] LLM 筛选中（110 条）...
[热点] LLM 生成选题建议中...
```

- **多平台并行抓取生效**：4 个平台共 **110 条**热点（30+30+20+30）。
- **失败降级生效**：小红书接口 500，自动重试 2 次后返回 0 条，
  **没有中断整条链路**，其余平台结果照常合并进 LLM 筛选。
- **LLM 筛选产出**（`filtered` 473 字），实际输出片段：

  ```
  **1. 豆包AI手机来了**（抖音）
  相关度：10
  切入角度：实测豆包AI手机核心功能，哪些真正提升日常办公/生活效率，附避坑清单。
  创作难度：中等

  **2. 儿子用豆包选下葬母亲黄道吉日，因葬礼后亲戚遇车祸重伤决定起诉豆包**（知乎）
  相关度：9
  ```

  可以看到 LLM 确实按「赛道相关度」而不是按热榜排名在筛 —— 抖音榜第一名是「美联储宣布加息25个基点」，
  但它没进筛选结果，进的是与「AI工具 / 效率提升」赛道相关的条目。

> ⚠️ 小红书 500 是上游 NewsNow 公共 API 的问题（不是本项目代码问题），已通过重试 + 降级正确处理。

### 5.3 数据复盘（四节点 LLM 链路，降级入口）

**命令**（手动粘贴作品数据的降级入口，不需要 Docker）：

```powershell
& '...\python.exe' -u -c "
import sys, json; sys.path.insert(0, '.')
from workflows.review import run_review_from_json
data = json.dumps([{'标题':'AI工具提效实测：一周省下8小时','发布时间':'2026-09-01','时长':'45秒','点赞':1200,'评论':35,'分享':12,'收藏':180},
                   {'标题':'三个Python技巧，同事看了直呼内行','发布时间':'2026-08-25','时长':'60秒','点赞':320,'评论':8,'分享':3,'收藏':45},
                   {'标题':'Docker入门：10分钟搞懂容器','发布时间':'2026-08-18','时长':'120秒','点赞':88,'评论':2,'分享':1,'收藏':12}], ensure_ascii=False)
r = run_review_from_json(data)
for k in ('funnel_diagnosis','content_assessment','suggestions'): print(k, len(r[k]))
"
```

**结果**（exit 0，耗时 **112.7s**，`error_msg` 为空）：

```
[复盘] 手动数据解析完成: 3 条作品
[复盘] 手动数据 3 条作品，跳过采集节点直接诊断
[复盘] 漏斗诊断中 ...
[复盘] 内容评估中 ...
[复盘] 生成优化策略中 ...
```

| 节点 | 产出 |
|---|---|
| `funnel_diagnosis` | **1247 字** |
| `content_assessment` | **1388 字** |
| `suggestions` | **2671 字** |

实际产出片段：

- **漏斗诊断**：`高表现作品为第1条（"AI工具提效实测：一周省下8小时"）：点赞1200、评论35、分享12、收藏180，全面领先。… 高表现共同特征（数据支撑）：时长最短（45秒 vs 60秒/120秒），互动量最高。`
- **内容评估**：`表现最好：AI工具提效类。1200赞、180收藏，远超另外两篇。… 收藏远高于评论/分享，说明用户把它当工具来存，而不是单纯看热闹。`
- **优化策略**：`立即执行核心是复制第1条模式（45秒 + 具体收益数字 + "实测"承诺 + 工具提效）…`
  —— 给出了带具体标题与脚本骨架的可执行方案。

### 5.4 百炼 ASR 真实转写（**本轮发现并修复了一个真 bug**）

**素材**：`yt-dlp` 从 `https://www.bilibili.com/video/BV1GJ411x7h7/` 下载的 208 秒音频
（`.cache/videos/downloads/BV1GJ411x7h7_audio.mp3`，4.86 MB）。

**命令**：

```powershell
& '...\python.exe' -u -c "
import sys; sys.path.insert(0, '.')
from tools.audio_transcriber import transcribe, sentences_to_srt
audio = r'.cache\videos\downloads\BV1GJ411x7h7_audio.mp3'
r1 = transcribe(audio, want_timestamps=False)
print('纯文本', len(r1['text']), '字')
r2 = transcribe(audio, want_timestamps=True)
print('时间戳', len(r2['text']), '字 /', len(r2['sentences']), '句')
sentences_to_srt(r2['sentences'], 'out.srt')
"
```

**结果**（exit 0）：

| 模式 | 耗时 | 产出 |
|---|---|---|
| 纯文本（非流式） | **34~57s** | 1793~2132 字 |
| 带句级时间戳 | **34~81s** | 64 句 + 可用 SRT |

最终 SRT 片段（**每句独立、时间戳单调递增、无重叠**）：

```
1
00:00:19,010 --> 00:00:22,010
We're no strangers to love,

2
00:00:22,970 --> 00:00:23,410
you know,

3
00:00:23,530 --> 00:00:31,730
the rules and so do i feel commitments while i'm thinking of
```

#### ⚠️ 这次验证推翻了第一版的实现假设（重要，值得单独记）

第一版按官方文档的字面理解实现：以为 `output.sentence` 是「当前这一句」，
于是走 `X-DashScope-SSE: enable` 并**按 `sentence_id` 收集每一帧**。
实测打出来的 SRT 是这样的：

```
1
00:00:19,010 --> 00:00:34,770
We're no strangers to love, you know, the rules and so do i feel commitments...
2
00:00:19,010 --> 00:00:56,010
We're no strangers to love, ... Never gonna give you up, ...     ← 前面内容的累加
3
00:00:19,010 --> 00:01:16,950
We're no strangers to love, ... （更长的累加）
```

**dump 原始 SSE 帧后真相清楚了**：`output.sentence` 是「到目前为止的全量快照」，
不是逐句递进 ——

```
帧 1: sentence_id=1, begin_time=19010, end_time=34770,  words=36,  text 长 140
帧 2: sentence_id=2, begin_time=19010, end_time=56010,  words=71,  text 长 318
帧 3: sentence_id=3, begin_time=19010, end_time=76950,  words=106, text 长 496
                   ↑ begin_time 恒定，text / words 单调增长
```

**由此得到三条实测结论，并据此重写了模块**：

1. **SSE 给不出逐句切分** —— 按 `sentence_id` 收集只会得到逐级变长的重复文本。
2. **根本不需要 SSE**：非流式一次调用就返回覆盖全音频的完整 `words[]`
   （208 秒音频 → 401 个词，跨 19010~207970ms），且**比 SSE 快一倍**（34s vs 81s）。
3. **句级切分自己做**：验证过 `"".join(w["text"] + w["punctuation"] for w in words)`
   与 `output.text` **完全一致**，所以 `words[]` 完整可用。

**同时修掉第二个 bug**：`words[]` 里有独立的空格 token
（`{"text": " ", "punctuation": ""}`，418 个词里有 22 个）。
第一版的切句函数写了 `if not token.strip(): continue`，把空格丢了，
于是拼出 `so doi feel` 这种粘连文本。修正为「原样拼接、只在记录时间时跳过空白词」，
并在自检里加了回归断言。

**顺带补了一个课案没有的兜底**：单条字幕的**时长上限 8 秒**。
实测 ASR 在连读/唱歌片段会长时间不吐标点，只按标点切会切出跨 16 秒、140 字符的
巨型字幕行 —— 画面上没法看。超过 8 秒就在当前词处收束一句。

> 这三条都已写进 `tools/audio_transcriber.py` 的文件头 docstring，
> 避免后来者再按文档的字面印象重踩一遍。

### 5.5 三条链路汇总

```
  ✓ 账号定位  324.9s
  ✓ 热点监控  177.8s
  ✓ 数据复盘  112.7s
  ✓ 内容复刻  287.9s  （下载 → ffmpeg 抽音频 → 百炼 ASR 转写 2132 字 → 3 次 LLM）
```

四条链路的**输入都是真实数据**（真实 LLM、真实热点 API、真实视频、真实作品数据），
输出都是具体可读的中文内容，不是占位文本。

> 内容复刻用的是课案自带的示例链接，而它是 Rick Astley 的《Never Gonna Give You Up》
> （课案作者留的是个 rickroll），所以转写出来是英文歌词 ——
> **中文口播的 ASR 质量没验到**，这是本节唯一的缺口。

---

## 6. Streamlit 应用验证

**命令**：

```powershell
Set-Location F:\ProGram\Python_Base\Media_Agent
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run main.py --server.headless true --server.port 8599
```

**结果**：

| 检查 | 结果 |
|---|---|
| 服务启动 | `Uvicorn server started on :::8599` |
| `GET /` | HTTP **200**，10951 字节 |
| `GET /_stcore/health` | HTTP **200** → `ok` |
| 页面标题（脚本执行后） | **自媒体AI创作平台**（= `st.set_page_config(page_title=...)` 生效，证明 `main.py` 真的执行成功） |
| 浏览器 console | **0 errors** |
| `stException` 元素 | **0 个** |

**逐页点击验证**（真实浏览器触发 Streamlit 会话）：

| 页面 | 点击 | 渲染 |
|---|---|---|
| 🏠 首页 | ✓ | ✓ 7 个导航项 + 6 张功能卡 + 运行环境面板 |
| 🎯 账号定位 | ✓ | ✓ 4 个输入项 + 生成按钮 |
| 🔥 热点监控 | ✓ | ✓ |
| 📝 内容复刻 | ✓ | ✓ |
| 🎥 口播视频 | ✓ | ✓ 双模式单选 + 文案输入 + 开始生成 |
| 🎬 视频剪辑 | ✓ | ✓ |
| 📊 数据复盘 | ✓ | ✓ Cookie 配置 + 主页链接 + **降级入口面板** |

**降级行为在界面上可见**（数据复盘页实际渲染出的文案）：

```
⚠️ 采集服务不可达（http://127.0.0.1:8080）—— 可先用下面的降级入口粘贴数据跑诊断
📋 手动粘贴作品数据（采集服务不可达时的降级入口）
```

这正是设计意图：**没有 Docker 服务时页面不报错，而是给出可用的替代路径**。

---

## 7. 课案 bug 修复验证

课案原文有多处照抄会崩的地方，本项目逐条修掉并做了断言：

| 位置 | 课案问题 | 本项目的处理 | 验证方式 |
|---|---|---|---|
| `workflows/video.py` | 读 `state['optimized']`，但 `VideoState` 无此字段 → 数字人模式必 `KeyError` | 改用 `state["raw_script"]` | 字节码常量断言（`"optimized" not in co_consts`） |
| `workflows/hot_topic.py` | 文件末尾模块级 `graph.draw_png(...)` → import 就崩（需 pygraphviz），且图名张冠李戴 | 整段删除，`get_graph()` 移入 `__main__` 供断言用 | 模块 import 成功 + 自检通过 |
| `workflows/positioning.py` | 调未 import 的 `get_session()` / `PositioningRecord` | 整段删除（结果由 Streamlit `session_state` 承载） | 自检通过 |
| `views/replicate.py` | 三引号 f-string 内层复用外层 `"`（PEP 701 之前是 SyntaxError） | 先取变量再插值 | 视图导入通过 |
| `mashup.py` / `views/mashup.py` | 两边 `CACHE_DIR` 不一致（`.cache/videos` vs `.cache/mashup`），兜底查找扫不到 | 统一取 `settings.media.get_mashup_work_dir()` | 兜底查找自检通过 |
| `mashup.py` / `views/mashup.py` | `DEFAULT_BGM` 硬编码 `C:\Users\13261\...wav` | 改为 `MEDIA_BGM_PATH`；留空则不混音 | 全项目 grep 无 `13261` |
| `review.py` / `views/review.py` | 硬编码 `C:/Users/13261/Documents/project/douyin` | 改为 `settings.media.douyin_api_base` | 全项目 grep 无 `13261` |
| `media_tools.py` | `extract_audio_text()` 里两段完全相同的 URL 判断 | 去重 | 自检通过 |

**本仓库实测补充的两条**（课案没有，不加会翻车）：

| 补充 | 为什么 | 落在哪 |
|---|---|---|
| 把 `.venv\Scripts` 顶到子进程 `PATH` 最前 | 本机 PATH 里的 `python` 是 Windows Store 占位符，**执行后静默无输出**，deepagent 跑 `python script.py` 会「成功但什么也没发生」 | `workflows/mashup.py` 的 `_get_editor_agent()` |
| system_prompt 里说明 `execute` 跑的是 `cmd.exe` 不是 bash | 不写，模型会反复敲 `ls`/`pwd`/`cat`，实测一路撞到 `GraphRecursionError` | `_build_editor_system_prompt()` |

**moviepy API 实测修正**（`moviepy 2.1.2`）：

```python
# 课案的 system_prompt 让 agent 从顶层导入 SubtitlesClip —— 实测 ImportError
from moviepy.video.tools.subtitles import SubtitlesClip   # ← 正确路径
```

已验证 `TextClip(font=, text=, font_size=, size=, method='caption')`、
`CompositeVideoClip(clips, size=)`、`with_position / resized / subclipped / with_effects` 在 2.x 全部可用。

---

## 8. 未验证项（如实说明）

以下是**本轮没有真实验证**的部分，不要当成已验证：

1. **声音克隆（CosyVoice）的真实创建音色** —— 未验证。需要：
   一段人声清晰的参考音频 + 百炼已开通 CosyVoice + 音色配额。
   （ASR 那一段已经真跑通了，两者不是一回事：ASR 用的是 `Fun-ASR-Flash`，
   声音复刻要另外在控制台开通 `CosyVoice`。）
2. **数字人对口型（PixVerse）的真实出片** —— 未验证。需要：
   百炼控制台**手动开通 PixVerse**、一段 10~30 秒人脸视频、`DASHSCOPE_WORKSPACE_ID`（如需）。
3. **视频剪辑（DeepAgents）的真实出片** —— 未验证。只验证了：
   图结构、路径清洗、SKILL.md 格式、system_prompt 约束完整性、DeepAgents API 形状。
   真实跑一次要几分钟且会消耗不少 token。另：**HyperFrames 未安装**，会走 moviepy 降级分支。
4. **抖音数据采集的真实链路** —— 未验证。本机**没有 Docker 服务**，
   只验证了「服务不可达 → 返回带修复命令的中文错误 → 降级入口可用」。
   响应体的真实层级、Cookie 是否够用、`/user/self` 解析是否有效，全都需要真跑服务才能确认。
   ⚠️ 另有一个上游版本风险：`Evil0ctal/Douyin_TikTok_Download_API` 的 `main` 已是 **v5**
   （改为 `/api/v1/...` 且需 API Key，端口 80 → 8000），本项目代码按 **v4** 形态实现，
   docker 命令里 pin 的是 `:V4.1.2`。若部署 v5 需要改端点常量与鉴权。
5. **`_read_edge_cookies()`（从 Edge 读抖音 Cookie）** —— 未实跑。
   它会 `taskkill` 掉所有 Edge 进程再起无头实例，副作用大，不适合在验证阶段触发。
   仅验证了端口探活函数不可达时返回 `False` 且不抛异常。
6. **中文口播的 ASR 质量** —— 转写链路本身已真跑通，但用的素材是英文歌曲
   （课案示例链接是个 rickroll），**中文语音的识别质量与标点准确度没有验证过**。

### 环境限制（影响验证方式，不是代码问题）

**本机跑长时间的单个 Python 进程会被环境杀掉**（无 traceback、无 stderr、退出码 1）。
实测：账号定位 3 次串行 LLM 调用合计 324.9s，重试多次才完整跑完；
数据复盘四节点约 121s，同样前几次被中断。

这是**执行环境**的限制而非代码缺陷 —— 各节点单独跑（32~43s）每次都成功，
链路本身的正确性由「每节点一个进程 + 状态落盘」交叉验证过。
在正常终端里手动跑不会有这个问题。

---

## 9. 遗留事项

| 项 | 说明 |
|---|---|
| Cookie 落点 | 页面上粘贴的抖音 Cookie 存在**运行时会话覆写点** `os.environ["MEDIA_DOUYIN_COOKIE"]`（重启即失效），持久值仍在根 `.env` 的 `MEDIA_DOUYIN_COOKIE`。这是为了让「界面粘 Cookie」这个动作不产生忘记清理的持久凭据。若希望它直接写 `.env`，需要改 `config.py` 加一个写入辅助函数。 |
| `MEDIA_IMAGE_*` 未配置 | 图片生成走占位图分支（首页环境面板已明示）。 |
| HyperFrames 未安装 | 剪辑会走 moviepy 降级分支（SKILL.md 里已写明 fallback）。 |
| `docs/` 下的课案提取文件 | 由 `docs/html提取脚本.py` 从课案 HTML 生成，课案更新后可重跑。 |
