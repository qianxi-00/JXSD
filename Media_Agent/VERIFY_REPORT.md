# VERIFY_REPORT —— Media_Agent 实测记录

> 对应课案：《3.自媒体Agent》
> 验证时间：2026-09-17
> 运行环境：Windows / PowerShell，Python 3.12.12（`F:\ProGram\Python_Base\.venv`）
> 复现命令见每一节，全部可原样再跑一遍。

---

## 结论摘要

| 验证项 | 结果 |
|---|---|
| 模块自检（14 个模块 × 子进程） | **14/14 通过** |
| Streamlit 视图导入（7 个页面） | **通过** |
| 环境契约检查（配置 / 依赖 API 形状 / 关键文件 / 降级路径） | **通过** |
| 真实联网检查（LLM / 热点 / 百炼接线） | **4/4 通过** |
| Streamlit 应用启动 + 七页渲染 | **通过（0 console error / 0 stException）** |
| 端到端链路（账号定位 / 热点监控 / 数据复盘 / 内容复刻） | **4/4 通过** |
| 百炼 ASR 真实转写（英文歌 208s / 粤语新闻 158s / **普通话 60.9s**） | **通过**，并修复了 3 个真 bug（见 5.4） |
| **声音克隆全链路**（自建托管 → `create_voice` → 克隆音色合成） | **通过**（见 5.8④） |
| **自建公网素材托管**（`tools/asset_host.py` + nginx 只读 location） | **通过**（见 5.8①） |
| **DeepAgents 视频剪辑真实出片** | **通过** —— 12 个 moviepy 动画素材 + 百炼 ASR 字幕 + 上下排布合成，产出 `mashup_final.mp4`（1280×960 / 20.0s / 30fps / 2.9MB，已逐帧核对），见 5.10 |
| 数字人出片 / 抖音真实采集 | **未验证**（见第 8 节） |

> 第二轮验证（5.8）另外修掉 3 个真问题：`MediaAgentSettings` **漏写 `env_prefix`**
> 导致所有 `MEDIA_*` 配置从未被读取；`asset_host.unpublish()` 的 **shell 注入隐患**；
> ASR 字幕时长兜底的 off-by-one。

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
| 文本模型 | 第一~二轮验证时是 `grok-4.6` @ `https://qianxi7988.me/v1`；**第三轮起根 `.env` 已整体切到 `deepseek-flash` @ `https://api.deepseek.com`**。剪辑链路（5.9）是在**切到 DeepSeek 之后**跑通的。<br>自媒体链路另有 `MEDIA_LLM_PROVIDER=deepseek` 开关，可只让本子项目换服务商而不动根配置（当前留空 = 复用根配置，保持单一真源）。 |
| 百炼 | `DASHSCOPE_API_KEY` 已配置（**根 `.env` 既有值**，非本次工作新增，详见第 8 节第 6 条），端点 `https://dashscope.aliyuncs.com/api/v1` |
| ffmpeg / yt-dlp / edge-tts | 均已安装可用 |
| HyperFrames | CLI **可用**（`npx --yes hyperframes --version` → `0.8.46`），但**渲染链路不可用**：`init`/`render` 实测卡到 300s 超时。默认已由 `MEDIA_MASHUP_USE_HYPERFRAMES=false` 切到 moviepy 分支（见 5.9④） |
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
> **这个缺口已在 5.6 补上**（换成真实中文新闻素材重验）。

### 5.6 中文（粤语）语音识别质量验证

用一段**真实的粤语财经新闻**（TVB 翡翠台，158 秒，1280×720）抽音轨后跑 ASR。
**粤语比普通话更难**（有专属字词、语序不同），能过就更能说明问题。

**命令**：

```powershell
# 抽音频（16k 单声道，与项目内部处理一致）
ffmpeg -y -i BV1dZYf6VEN2.mp4 -vn -acodec libmp3lame -b:a 128k -ar 16000 -ac 1 news_audio.mp3

& '...\python.exe' -u -c "
import sys; sys.path.insert(0, '.')
from tools.audio_transcriber import transcribe, sentences_to_srt
r = transcribe('.cache/fixtures/news_audio.mp3', want_timestamps=True)
print(r['text']); sentences_to_srt(r['sentences'], 'news.srt')
"
```

**结果**（exit 0，耗时 **31.6s**，527 字 / 49 句）：

```
睇下最新嘅金融行情，恒生指数最新报24698点，升30点，总成交超过1358亿。
恒生科技指数升35点。最活跃港股：智普升39.5个，盈富基金升2仙，南方恒生科技升3.2仙，
腾讯控股跌5.2个，恒生中国企业升4仙。mini max升3.4，中芯国际升2.65，长飞光纤光缆升6.4，
宁德时代跌17.2，联想集团升2.64 … 内地股市方面，上证指数升27点，深证成分指数升150点。
… 本港九九金每两报40485蚊。睇埋天气，大致天晴，吹和缓偏东风 … 而家嘅气温系31度，
相对湿度64% … 紫外线指数系9，强度属于甚高。新闻报道完啦，再会。
```

**质量评估**（人工核对）：

| 维度 | 表现 |
|---|---|
| 粤语专属字词 | ✅ `嘅` / `系` / `睇` / `蚊`（元）/ `而家`（现在）/ `升仙`（涨分）全部认对 |
| 金融专有名词 | ✅ 恒生指数、恒生科技指数、盈富基金、中芯国际、宁德时代、蓝筹股 |
| 数字准确度 | ✅ 24698 点 / 1358 亿 / 40485 蚊 / 31 度 / 64% —— 与画面对得上 |
| 断句与标点 | ✅ 逗号句号落在正确位置，49 句平均 2.2 秒 |
| 中英混排 | ✅ `mini max` 正确保留为英文 |

**结论：中文 ASR 质量可用**，粤语都能到这个程度，普通话只会更好。

同时暴露并修掉一个 off-by-one：最长句 **9235ms > 8000ms 上限**。
根因是时长兜底判断用的是「上一个词的结束时间」，下一个词跳得远时仍会把当前句撑长。
改成**把当前词算进去预判**后，最长句降到 **6820ms，超限句数 0**。已加回归断言。

### 5.7 ⚠️ 声音克隆：实测暴露出一个架构级约束

用上面裁出的 13 秒主播出镜段（播音腔、无人声背景音乐，理论上最理想的参考素材）
跑 `clone_voice()`，`create_voice` 直接 **400**：

```
Code: InvalidParameter
Error Message: audio url should start with http or https
```

**根因**：参考音频是经 `tools/dashscope_upload.upload_file()` 换成
`oss://dashscope-instant/...` 临时 URL 的 —— 而 **CosyVoice 不收 `oss://`**。

百炼那套免费临时存储是给**多模态 / 图像 / 视频类**模型用的
（调用时靠 `X-DashScope-OssResourceResolve: enable` 头解析）；
ASR 走 Base64 Data URI 是另一条路（**已实测可用**），但 **TTS 的声音复刻必须是真的 http(s)**。

**影响面**：
- ❌ 「克隆你自己的声音」这个能力**在本机现有条件下跑不通**；
- ✅ **数字人功能不受影响** —— `workflows/video.py` 的降级链会自动走到
  edge-tts 通用音色（免费）或 PixVerse 内置 TTS（一步出片）。

**已经做的处理**：
1. `clone_voice()` 新增 `_publish_reference()`，明确失败并给出三种可选托管方案，
   不再拿 `oss://` 去撞墙；
2. 新增配置 `MEDIA_VOICE_REF_URL` —— 只要有一个已托管的参考音频公网 URL，
   填进去就能启用声音克隆（`.env` / `.env.example` / `config.py` 三处已同步，且已验证幂等）；
3. 把这个约束写进 `tools/voice_clone.py` 的文件头 docstring。

**可选托管方案**（需要定一个）：
① 阿里云 OSS（同账号最顺，需开 OSS 并配 AK/SK）；
② 自己的公网服务器（放静态目录 + HTTP 服务）；
③ 内网穿透（Cloudflare Tunnel / ngrok）。

> ⚠️ **同样的约束也可能影响数字人**：PixVerse 的 `video_url` / `audio_url` 同样要求公网 URL，
> 是否能吃 `oss://` **尚未实测**（官方文档说「素材必须是公网可访问 URL」，
> 而临时存储文档又说适用于「多模态、图像、视频或音频模型」—— 只有真提交一次才知道）。
> 如果 PixVerse 也不收 `oss://`，数字人同样需要上面这套托管。

### 5.8 补齐 5.6 / 5.7 的结论（第二轮验证）

5.7 卡住的那件事已经解决，并且过程中又抓出两个真 bug。

#### ① 自建公网托管打通 —— `tools/asset_host.py`

用的是你自己的服务器 `ubuntu@43.128.75.66`（腾讯云新加坡，`~/.ssh/config` 里别名「又」）。
服务器上只加了一个**只读静态 location**，没有动任何既有路由：

```nginx
location /media-assets/ {          # 加在 windsurf-api.conf 的 location / 之前
    alias /var/www/media-assets/;
    autoindex off;
}
```

原配置已备份为 `windsurf-api.conf.bak-20260917`。上传走 `scp`（不是 HTTP PUT），
所以 nginx 侧不需要任何写权限。

> 排查过程中的两个观察，供以后参考：
> 1. 80 端口上 `curl http://43.128.75.66/` 返回 **502 不是超时** —— 说明 nginx 在正常工作，
>    只是它的 `location /` 转发的后端（`127.0.0.1:3003`）没跑。
> 2. reload 后第一次请求仍可能被**旧 worker** 处理（deny 生效有延迟），
>    要以 error.log 里 `open() "/var/www/media-assets/..." failed` 这类记录为准。

实测：上传 0.61MB 用 **2.6 秒**，本地 HEAD 回验 **0.5 秒 HTTP 200**。

#### ② ⚠️ 抓出一个我自己写的配置 bug：`env_prefix` 漏了

`MediaAgentSettings` 的 `model_config` **漏写 `env_prefix="MEDIA_"`**
（docstring 写着「来自 MEDIA_ 前缀键」，配置里却没设）。

后果很隐蔽：**所有 `MEDIA_*` 配置项从来没被读取过**，字段全部退化成代码默认值 ——
而默认值恰好与 `.env` 里的值一致，所以表面上完全看不出来。
直到这轮加了 `MEDIA_ASSET_SSH` / `MEDIA_ASSET_BASE_URL`（默认空、.env 有值）才暴露。

诊断记录：

```
MediaAgentSettings.model_config -> 'env_prefix': ''      ← 应为 'MEDIA_'
对比 LLMSettings.model_config   -> 'env_prefix': 'LLM_'  ← 正常
```

已修复并加了注释说明为什么这行不能省。修复后 16 个 `MEDIA_*` 字段全部正确读到。

> 这个 bug 的教训：**「默认值恰好等于期望值」会掩盖配置根本没生效的事实**。
> 所以新加配置项时，最好让它的默认值 ≠ .env 里的值（哪怕只是暂时），否则测不出来。

#### ③ ⚠️ 抓出 `asset_host.unpublish()` 的 shell 注入隐患

第一版写的是 `f"rm -f {target!r}"` —— **Python 的 `repr()` 不是 shell 安全的**：
含单引号的字符串被 repr 成 `'a\'b'` 后，在 sh 里单引号会提前闭合，内容被当命令执行。

同时路径穿越防护也写错了（只取了 basename，`../` 被静默吞掉而不是拒绝）。

已改为：只接受**本项目基址下、文件名严格匹配 `^[A-Za-z0-9][A-Za-z0-9._-]*$`** 的 URL，
并用 `shlex.quote()` 做 shell 引用。自检覆盖了路径穿越 / 隐藏文件 / 子目录 /
shell 元字符 4 类共 6 个恶意样本，全部拒绝。

> 这次是**自检自己抓出来的**：写测试时随手断言「穿越 URL 应返回 False」，
> 结果它返回 True 并真的去 `rm` 了 —— 说明断言比实现更早发现问题是有可能的。

#### ④ 声音克隆全链路打通 ✅

用普通话讲师素材（20 秒，正面出镜）跑完整链路：

| 步骤 | 结果 |
|---|---|
| 参考音频 → 自建托管 | `http://43.128.75.66/media-assets/fb79f180c30b25fb.wav` |
| `create_voice`（此前被 oss:// 拒） | **成功**，44.7s → `cosyvoice-v2-cnmtest-9efce28f...` |
| 克隆音色合成 | **307,724 字节 WAV，时长 6.41s**（ffprobe 确认，是真音频不是空壳） |
| 二次调用 | **命中缓存 0.00s**，未重复消耗音色配额 |
| 清理 | 测试音色已 `delete_voice`，公网素材已 `unpublish` |

**「克隆你自己的声音」这个能力现在可用了。**

#### ⑤ 普通话 ASR 质量验证（替换 5.6 的粤语素材）

按你的要求换成普通话素材（bilibili「好声音播音主持教学」讲师正面出镜段，60.9s）：

```
新手主持人接到婚礼主持订单之后啊，往往就傻眼了。我接下来应该要怎么跟新人沟通呢？
别着急，今天刘星老师教你们三招。以下的三点呢，非常的重要，尤其是第三点，
一定要点赞、收藏、保存好，会对你有很大的帮助的。
一、注意网络社交礼仪，把"你好"换成"hello"或者是可爱的表情包，呈现出你的热情；
不要说"在吗"，直接说你要沟通的事情，呈现出你的专业；
不要说"哦"、"嗯"、"呵呵"，而是换成轻松的语气…
第二呢，你要去发一份主持确认单，包含时间、地点、人物、事件，并且呢注明收定金的注意事项…
第三，发新人问卷，方便了解新人需求和你创作流程以及台词，更好的做一个走心的主持人。
```

**358 字 / 33 句，6.2 秒跑完**（比 158s 的粤语素材快得多，因为音频短）。
语气词（啊/呢/啦）、引号内词汇、顿号枚举、中英混排（`hello`）全部正确；
句长 280ms ~ 3600ms，平均 1.5s —— 正好适合做字幕。

**结论：普通话 ASR 质量可用，且质量很高。**

---

### 5.9 ⚠️ 剪辑链路抓出一个「必崩」bug（第三轮验证，已修）

**这是本项目唯一一个「第一次真跑就断」的硬 bug —— 之前只跑了离线自检，所以一直没暴露。**

第一次真跑 `run_mashup()`，agent **一步都没执行**就抛异常：

```
ValueError: Path:F:\ProGram\Python_Base\Media_Agent\.skills outside root directory:
            F:\ProGram\Python_Base\Media_Agent\.cache\mashup
  File "...\deepagents\middleware\skills.py", line 615, in _list_skills_with_errors
    ls_result = backend.ls(source_path)
```

**根因**：课案把技能目录的**绝对路径**直接传给 `create_deep_agent(skills=...)`，
而 deepagents 规定 skills 路径**必须相对 backend 的 `root_dir`**
（`create_deep_agent` 文档原文：*"skills are loaded from disk relative to the backend's
`root_dir`"*）。`SkillsMiddleware.before_agent` 拿这个绝对路径去 `backend.ls()`，
被 `virtual_mode=True` 的越界检查判定为「沙箱外」，直接抛异常，
而报错点在 deepagents 内部、跟 `mashup.py` 表面无关，很容易误判成版本问题。

**修法**：用 `CompositeBackend` 把真实技能目录挂到虚拟路径 `/skills/`：

```python
routes = {"/skills/": FilesystemBackend(root_dir=skills_dir, virtual_mode=True)}
backend = CompositeBackend(default=sandbox, routes=routes)
create_deep_agent(model=model, skills=list(routes), backend=backend, ...)
```

文件读写经路由落到真实磁盘，`execute` 仍走沙箱 ——
`CompositeBackend.execute` 只认 `default` 后端（源码注释原文：*"Unlike file operations,
execution is not path-routable — it always delegates to the default backend"*），
而 `LocalShellBackend` 同时继承 `FilesystemBackend` 与 `SandboxBackendProtocol`，
所以 shell 能力不受影响。

**回归防护（两层，都不需要密钥、可离线跑）**：

1. `workflows/mashup.py` 自检第 6 项：用同一套 `CompositeBackend` 接线
   （不建 agent、不调 LLM），断言 `ls("/skills/")` 能列出 `video-use`、
   `download_files(["/skills/video-use/SKILL.md"])` 能读到内容 —— 直接复现原崩溃点。
2. `verify_all.py` 环境契约层：钉住 `CompositeBackend` 的 `default`/`routes`
   构造参数与 `ls`/`download_files` 方法存在。

**顺带纠正一条此前的错误结论**：报告早先写「HyperFrames 未安装，会走 moviepy 降级分支」——
实测 **`npx --yes hyperframes --version` 返回 0.8.45**，npx 会按需拉取，
所以 HyperFrames 是**可用的**，不需要预先全局安装。

---

#### ② ⚠️ 第二个必崩 bug：`execute` 在 Windows 上会**永久卡死**（已修，含一次被推翻的方案）

修掉①之后 agent 能跑起来了（读技能 → 探源视频 → 转写字幕 → 建 HyperFrames 工程），
但随后**整体挂死**：CPU 归零、沙箱不再有任何产出。

这是靠 `faulthandler.dump_traceback_later(240, repeat=True)` 定时 dump 全线程栈才定位到的 ——
**不是 LLM 卡住，是 `execute` 工具的子进程卡死**：

```
Thread 0x0000b17c (most recent call first):
  subprocess.py:1628 in _communicate          ← 永久阻塞在这
  subprocess.py:550  in run
  deepagents/backends/local_shell.py:304 in execute
  deepagents/backends/composite.py:841  in execute
  deepagents/middleware/filesystem.py:2928 in sync_execute
```

当时那条命令的进程树（实抓）：

```
python (e2e)
 └ cmd.exe /c "cd /d ...\mashup && npx --yes hyperframes@0.8.46 render ..."
    └ node.exe (npx-cli)
       └ cmd.exe /d /s /c hyperframes render ...
          └ node.exe (hyperframes bin)
             └ node.exe (hyperframes dist)     ← CPU 恒为 0，彻底僵住
```

**根因（Windows 专有，两层）**：

1. `capture_output=True` 把 stdout/stderr 接到**管道**上，而管道句柄会被
   **整棵子进程树继承**。只要树里存在一个长期存活的进程，`communicate()`
   就永远等不到 EOF。
2. `timeout=` **只负责"发现"超时，之后仍要回收管道**：
   `except TimeoutExpired: process.kill(); process.communicate()`。
   Windows 上 `kill()` 只终结**直接子进程**，孙进程照旧攥着管道 ——
   于是超时机制**整体失效**。（`deepagents` 在 Windows 上还把
   `start_new_session` 设成 `False`，见 `local_shell.py:314`，更没法整组清理。）

**独立探针实测**（用一个 `ping -n 300` 当"不死的孙进程"，可复现）：

| 方式 | 实测耗时 |
|---|---|
| `subprocess.run(capture_output=True, timeout=5)` | **59.8s** 才抛 `TimeoutExpired`（= 孙进程自己的寿命） |
| `LocalShellBackend.execute(timeout=6)` | **29.3s** 才返回（同上） |
| `execute(timeout=8)` + `taskkill /F /T` 整树杀 | **301.8s**，**该方案无效** |

**❌ 第一个方案（`Popen.kill` 先 `taskkill /F /T`）被实测推翻**，已废弃：
`taskkill /T` 只能沿**活着的父进程**遍历，而 `cmd /c start /b ...` 这类命令的
直接子进程**早就退出了**，树根一没就无从下手 —— 孙进程照样活着。

**✅ 最终修法：不用管道，改临时文件**（`mashup.py` 的 `_patched_run`，
只接管 Windows + `capture_output=True`，其余走标准库原实现）：

- stdout/stderr 指向**临时文件** → 没有管道，就没有 EOF 可等；
- 用 `Popen.wait(timeout)` 判超时 → Windows 上走 `WaitForSingleObject`，
  与管道状态无关，超时**必定**按时触发；
- 超时后仍 `taskkill /F /T` 尽力清理树，再 `wait()` 收尾；
- 正常退出则从文件读回输出，`CompletedProcess` 语义与标准库一致。

**真实链路上的验证**（同一轮 E2E 内）：

```
[17] execute: npx --yes hyperframes init videos\hf_proj
[🔧] Error: Command timed out after 300 seconds (custom timeout). ...
[Command failed with exit code 124]
```

**这正是此前永久卡死的那条命令，现在有界返回**；同轮的 `faulthandler` 栈也落在
新的 `mashup.py` `_patched_run` 里的 `proc.wait()`，确认走的是新路径。
`LocalShellBackend.execute` 本来就捕获 `TimeoutExpired` 并返回 `exit_code=124`
的中文提示，所以 agent 拿到的是一次干净的失败，可以据此降级。

**回归防护**：`workflows/mashup.py` 自检第 7 项 —— 用 `cmd /c start /b cmd /c ping -n 60`
构造同样的「直接子进程先退、孙进程继续攥输出」形态，断言耗时 **< 20 秒**
（标准库在这台机器上要 ~60 秒）。纯本地、不会挂起。

---

#### ③ 顺手修掉的两个真缺陷

**a) 工具异常会终结整轮任务**（`_ToolErrorToMessage` 中间件）

实测：agent 把沙箱外的绝对路径喂给了 `read_file`，抛
`ValueError: Path ... outside root directory`。`deepagents` 默认走 langgraph 的
`_default_handle_tool_errors`，它**直接 raise** —— 整轮 20+ 步、几分钟的工作当场作废，
模型连纠正的机会都没有。

`create_deep_agent` 没有暴露 `handle_tool_errors`，所以改用 `middleware=[...]`
挂一个 `wrap_tool_call`，把异常转成 `ToolMessage(status="error")` 回给模型。
回归用例：自检第 8 项（断言异常被转成 `ToolMessage`、正常返回不被改动）。

**b) `[PathFix]` 日志刷屏把真日志埋掉**

`node_edit_video` 解析输出路径时，把**最后一条消息按空格切词、逐词**喂给
`normalize_path()`。而 `normalize_path` 只要改动了就打印，且会无差别地把文本里的
`/` 换成 `\` —— 实测刷出**几百行**噪音：

```
[PathFix] 路径已修复: 'https://cdn.jsdelivr.net/...' → 'https:\\cdn.jsdelivr.net\\...'
[PathFix] 路径已修复: '1/3' → '1\3'
[PathFix] 路径已修复: 'oss://' → 'oss:'
```

修法：**先按 `.mp4` 后缀过滤，再调 `normalize_path`**。
（注：这不会污染写入的文件内容 —— `normalize_path` 只改它自己的局部变量，
原始消息字符串未被修改；但日志被埋掉本身就是交付缺陷。）

---

#### ④ ⚠️ HyperFrames 在本机**渲染链路不可用**（环境结论，非代码 bug）

实测证据：

- `npx --yes hyperframes --version` → `0.8.46`，**CLI 本身能装能用**（此前报告写"未安装"是错的）；
- `--help` / `docs <topic>` 这类纯本地命令**正常返回**；
- 但 `npx --yes hyperframes init <dir>` 与 `render` 会**卡到 300 秒超时**
  （`exit_code=124`），进程树 CPU 恒为 0；
- 其渲染依赖 puppeteer/浏览器下载，而本机 `curl https://cdn.jsdelivr.net/...`
  实测 `exit 35`（SSL 连接失败，走本地代理）。

**结论**：`SKILL.md` 里写明的 moviepy 降级分支在本机是**可用路径**，HyperFrames 不是。
这条只影响"动画素材用哪种方式生成"，不影响剪辑链路本身。

为此加了一个配置项 `MEDIA_MASHUP_USE_HYPERFRAMES`（默认 `false`）——
system_prompt 会随它切换：开着就走课案原方案并附时间预算约束，
关着就明确指示 agent 别碰 HyperFrames、直接用 moviepy 画。
浏览器链路正常的机器上改成 `true` 即可恢复课案原方案。

### 5.10 ✅ DeepAgents 视频剪辑真实出片（第三轮验证，**通过**）

**这是整份报告里最后补上的一块硬证据。**

**输入**：`.cache/fixtures/cn/cn_avatar_20s.mp4`（普通话口播，1280×720 / 20s / 30fps / 有音轨）
**要求**：上下排布（上方原视频、下方深色条带放动画素材）+ 白字黑描边字幕 + 输出 mp4

**结果**（`elapsed = 394.3s`，模型 `deepseek-flash`）：

```
output = F:\ProGram\Python_Base\Media_Agent\.cache\mashup\videos\mashup_final.mp4
size = [1280, 960]   duration = 20.0   fps = 30.0   audio = True
2,904,114 bytes
```

`node_edit_video` 从 agent 最后一条消息里解析出了路径，`OUTPUT_OK` 校验通过
（不是"兜底扫目录"扫出来的，是真报告上来的）。

**agent 实际做了什么**（据其自述 + 沙箱产物核对）：

1. **字幕**：`tools.audio_transcriber.transcribe_to_srt()` 走百炼云端 → `videos/subtitles.srt`
   （12 句 / 118 字），全程只此一份 SRT；
2. **素材 12 个**（要求 ≥6）：`ColorClip + TextClip + ImageClip(渐变) + with_position`
   画出 `intro_card.mp4`(1280×720，与原视频同像素，放最上层)、
   `band_intro.mp4` + `s01..s07` + `gap1..gap3`(均 1280×240 = 720//3)；
   时间点严格对齐 SRT 句边界；
3. **合成**：`CompositeVideoClip(size=(1280, 960))` = `(w, h + h//3)`；
   原视频 1280×720 原生分辨率贴 (0,0)，无 `resized`、无 `.with_mask()/set_mask()`；
   底下垫满幅深色底 → 任意时刻无黑屏；
4. **人声**：`afx.MultiplyVolume(3.0)` 后加 `AudioNormalize` 防削波，成片峰值 0.939 无破音；
   BGM 未配置 → 跳过混音不报错；
5. **自检**：抽取 10 帧做联系表、逐帧核对字幕只出现一次、条带区均值 40~52（非黑）。

**产物已归档为回归基线**（`.cache/` 是 gitignore 的，只在本机）：

| 文件 | 大小 | 用途 |
|---|---|---|
| `.cache/fixtures/cn/mashup_final_e2e.mp4` | 2,904,114 | 成片 |
| `.cache/fixtures/cn/mashup_final_sheet.png` | 2,337,593 | 10 帧联系表（人工核对用） |
| `.cache/fixtures/cn/mashup_subtitles.srt` | 797 | 字幕 |

**执行时间 394.3s 里，约 300s 花在一次 HyperFrames 超时上**（`exit 124`）——
这正是 5.9② 的补丁起作用的结果：以前这一下会**永久卡死**。

#### ⚠️ agent 实测抓出我提示词里的一个真错误（已修）

我在 system_prompt 与任务提示里写的是 `TextClip(font='Microsoft YaHei', ...)`。
**本机必崩**，agent 自己探针测出来并绕过了：

```
FAIL font='Microsoft YaHei'   ValueError: Invalid font Microsoft YaHei,
                              pillow failed to use it with error cannot open resource
OK   font='C:/Windows/Fonts/msyh.ttc'      size=(1280, 132)
OK   font='msyh.ttc'
```

根因：moviepy 2.x 把 `font` 直接交给 pillow 当**字体文件**加载，不做字体族解析。
（课案原文写的是 `font='Microsoft YaHei'` —— 那是课案作者环境下的写法，
本项目**不能照抄**。）

已修三处：`_build_editor_system_prompt` 的 TextClip 示例、
moviepy 降级分支的指令、`node_edit_video` 的默认任务提示；
并在自检里加了**实测断言**：用 `C:/Windows/Fonts/msyh.ttc` 构造 TextClip 必须成功。

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

1. ~~声音克隆（CosyVoice）出音~~ —— **已于第二轮验证打通**（见 5.8④）：
   自建公网托管（自己的服务器 + nginx 只读 location）+ `create_voice` + 克隆音色合成全通。
2. **数字人对口型（PixVerse）的真实出片** —— 未验证，且**当前被账号状态挡住**。
   实测提交请求返回 `400 Arrearage`（`Access denied, please make sure your account is
   in good standing`），根因见下方第 6 条，**不是**「模型未开通」。
   素材与托管这两环都已就绪：普通话人脸素材 `.cache/fixtures/cn/cn_avatar_20s.mp4`（20s / 1280x720），
   公网托管 `http://43.128.75.66/media-assets/...` 可用（见 5.8①），
   即使 PixVerse 不吃 `oss://` 也不影响。
   账号恢复后可直接跑：`python -c "from tools.avatar_client import generate_avatar_video; ..."`。
3. ~~**视频剪辑（DeepAgents）的真实出片**~~ —— **已于第三轮验证出片**（见 5.10）：
   12 个 moviepy 动画素材 + 百炼 ASR 字幕 + 上下排布合成，产出
   `mashup_final.mp4`（1280×960 / 20.0s / 2.9MB），`elapsed=394.3s`。
   过程中修掉了 4 个真缺陷（5.9 全节）。
   仍需注意：**HyperFrames 渲染链路在本机不可用**（见 5.9④），
   默认已由 `MEDIA_MASHUP_USE_HYPERFRAMES=false` 切到 moviepy 分支。
4. **抖音数据采集的真实链路** —— 未验证。本机**没有 Docker 服务**，
   只验证了「服务不可达 → 返回带修复命令的中文错误 → 降级入口可用」。
   响应体的真实层级、Cookie 是否够用、`/user/self` 解析是否有效，全都需要真跑服务才能确认。
   ⚠️ 另有一个上游版本风险：`Evil0ctal/Douyin_TikTok_Download_API` 的 `main` 已是 **v5**
   （改为 `/api/v1/...` 且需 API Key，端口 80 → 8000），本项目代码按 **v4** 形态实现，
   docker 命令里 pin 的是 `:V4.1.2`。若部署 v5 需要改端点常量与鉴权。
5. **`_read_edge_cookies()`（从 Edge 读抖音 Cookie）** —— 未实跑。
   它会 `taskkill` 掉所有 Edge 进程再起无头实例，副作用大，不适合在验证阶段触发。
   仅验证了端口探活函数不可达时返回 `False` 且不抛异常。
6. **【新增·需要你处理】百炼账号对计费模型返回 `Arrearage`** —— 账号欠费/无可用额度。
   本轮实测（同一把 key，`sha256[:12]=b96e9a10cb9f`，len 35）：

   | 探测目标 | 结果 |
   |---|---|
   | ASR `qwen-audio-3.0-asr-flash` | ✅ **通** —— 真跑 `cloned_out.wav`(6.4s) 识别出完整中文句子 |
   | compatible-mode `qwen-turbo` 对话 | ❌ `Arrearage` |
   | `voice-enrollment`（声音复刻） | ❌ `Arrearage` |
   | `pixverse/pixverse-lipsync` | ❌ `Arrearage` |
   | 不存在的模型（对照） | `InvalidParameter: Model not exist` |

   对照项证明**错误码有先后顺序**：先校验模型是否存在、再校验账号状态 ——
   所以 PixVerse 报 `Arrearage` 说明**模型本身存在且已开通**，卡的是账号状态。

   **需要你做的**：登录阿里云百炼控制台 → 费用中心，查「免费额度」与账户余额，
   结清欠费或充值。恢复之前，声音克隆与数字人出片**不可用**，ASR 与热点链路不受影响。

   > 关于 key 归属：`DASHSCOPE_API_KEY` 是根 `.env` 里**本来就有的**一行，
   > 位于注释「阿里云百炼 DashScope —— 课案「监控与评估 / RAG评估」的评测模型 + 向量模型」
   > 之下，由 RAG 子项目共用；本次工作只是**复用它**，没有引入、替换或新增任何百炼密钥。
   > 它也不是记忆栈那把 key（`F:\ProGramApp\DSH\memory\.env` 的 `MEMORY_LLM_API_KEY`）。
   > 该行是否属于你的个人账号，只有你能最终确认 —— 但账号处于欠费状态这一点，
   > 与「公司共享账号」的特征不符。

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
| HyperFrames | 渲染链路不可用（浏览器依赖拉不下来），默认由 `MEDIA_MASHUP_USE_HYPERFRAMES=false` 走 moviepy 分支；换到浏览器链路正常的机器改成 `true` 即恢复课案原方案。 |
| `docs/` 下的课案提取文件 | 由 `docs/html提取脚本.py` 从课案 HTML 生成，课案更新后可重跑。 |
