# Agent 课案（Jupyter Notebook 版）

课案来源：`G:\笔记\LLM\课件导出\Agent.html`

本项目把课案全部代码整理成 **32 个 Jupyter Notebook**。相比早先的 `.py` 版本，
Notebook 化带来的是「**能渲染的讲解**」：知识点用 Markdown 的标题层级展开、
对比用表格、流程用 mermaid 图、出处用可点击的官方链接，代码按「一格只讲一件事」切分，
每个有输出的格子后面都跟着一段 **实测的「预期输出」**。

> 原来的 162 个课程 `.py` **没有删除**，已归档到 `_py_source/`（保留 git 历史，仍可运行、可 grep）。
> 见下文「源脚本归档」。

## 先做一次：注册项目专用内核

⚠️ **这一步不做，notebook 会跑到错误的解释器上**，症状是 `import langchain` 报 pydantic 版本冲突、
或者干脆什么都不输出（静默失败）。原因：本机默认的 `python3` 内核指向 Anaconda，
而 `.venv` 自带的 kernelspec 用的是裸 `python`（会落到 Windows Store 占位符）。

```powershell
cd F:\ProGram\Python_Base
& .\.venv\Scripts\python.exe -m ipykernel install --user --name python-base-agent --display-name "Python (Python_Base .venv)"
```

装完确认一下：

```powershell
& .\.venv\Scripts\python.exe -m jupyter kernelspec list      # 应能看到 python-base-agent
```

之后用 JupyterLab（`& .\.venv\Scripts\jupyter-lab.exe`）或 VS Code 打开 notebook，
**内核选 `Python (Python_Base .venv)`**（所有 notebook 的 metadata 已经指向它）。

## 怎么读

| 你的目的 | 走哪条路 |
|---|---|
| 第一次接触，想按顺序学 | `01_langgraph` → `02_langchain` → `03_deepagents` → `04_function_call` |
| 只想知道「Agent 是怎么写出来的」 | `02_langchain/02_智能体与工具` → `04_function_call/01_工具定义与参数类型` → `02_langchain/04_中间件_钩子与人工审核` |
| 想接自己的工具/服务 | `05_mcp`（MCP 协议）→ `02_langchain/06_管道_MCP进阶_RAG与测试` |
| 想上线 | `09_aegra_deploy` → `07_protocols` |
| 想做评估与质量闭环 | `06_langfuse` 全章 |
| 想对照官方文档补课 | `官方文档缺口对照.md`（21 个补充篇的缺口清单） |
| 只想找某段代码 | 直接看 `_py_source/<章>/` 下的原脚本，或在各 notebook 里搜索 |

**运行条件分三档**，每本 notebook 开头的表格里都标了是哪一档：

| 标记 | 含义 | 要准备什么 |
|---|---|---|
| 🟢 | 离线可跑 | 什么都不用，装好依赖即可 |
| 🟡 | 需模型 | 根目录 `.env` 里的 `API_KEY` / `BASE_URL` / `MODEL_NAME` |
| 🔴 | 需外部服务 | Langfuse / PostgreSQL / Docker / 自起的 MCP 服务……表格里逐条写明 |

## 目录结构（32 个 Notebook）

| 目录 | notebook |
|---|---|
| `01_langgraph/` | `01_基础图与状态`、`02_记忆_短期与长期`、`03_流式与中断`、`04_时间旅行_子图与容错` |
| `02_langchain/` | `01_模型_消息与结构化输出`、`02_智能体与工具`、`03_记忆与流式`、`04_中间件_钩子与人工审核`、`05_多Agent`、`06_管道_MCP进阶_RAG与测试` |
| `03_deepagents/` | `01_智能体与流式`、`02_七种后端`、`03_人工审核_记忆_子智能体_Skills`、`04_上下文治理与Rubric评分`、`05_解释器PTC与异步子代理` |
| `04_function_call/` | `01_工具定义与参数类型`、`02_三种Agent实现对比` |
| `05_mcp/` | `01_服务端与客户端`、`02_资源与提示词`、`03_三种Agent调用`、`04_权限_JWT认证`、`05_部署与调试` |
| `06_langfuse/` | `01_追踪_会话与提示词`、`02_评估与打分`、`03_标注与数据` |
| `07_protocols/` | `01_ACP协议`、`02_A2A协议` |
| `08_skills/` | `01_概念原理与SKILL示例`、`02_三种技能载体` |
| `09_aegra_deploy/` | `01_为什么需要部署平台与项目骨架`、`02_本地开发_客户端调用_Langfuse` |
| `10_workflow_platform/` | `01_可视化平台与Langflow` |
| `_py_source/` | **归档**：162 个课程 `.py`，与原章节目录结构一一对应 |
| `_tools/` | 改造与维护工具（见下） |
| `_nb_template.md` | Notebook 编写规范 —— **改 notebook 之前先看它** |

各 notebook 运行的临时文件落在同目录的 `tmp_nb_work/`（已 gitignore）；
`08_skills/skills/`、`09_aegra_deploy/aegra_project/` 是被 notebook 引用的**真实资源目录**，不要删。

## 每个 Notebook 长什么样

顺序固定，不要自行调整：

| 区块 | 内容 |
|---|---|
| 标题 | 一句话定位 + 全节概念表 + 「由哪几个源文件合并而成」 |
| **运行条件** | 三档标记 🟢/🟡/🔴 + 依赖 / 密钥 / 前置服务 / 预计耗时 |
| 本节地图 | mermaid 流程图 **＋ 等价表格**（裸 JupyterLab 不渲染 mermaid，靠表格兜底） |
| 0. 环境引导 | 固定一格：向上找到仓库根 → `chdir` + 塞 `sys.path`（少了它必 `ModuleNotFoundError: config`） |
| 1. 课案原版 | 最少代码先跑通，看骨架 |
| 2. 完整版 | 对照课案原文的完整实现，讲「为什么这么做」 |
| 3. 官方文档补充 | 该节独有（没有则整节省略） |
| 小结 / 常见坑 / 官方链接 | 收口 |

**每个有输出的 code cell 后面都跟着 `### 预期输出`**，内容是真跑出来的（不是猜的）。
输出里含模型措辞、时间戳、随机值的那几格，格子里会明确写「这段每次不同，别逐字比对」——
工具 `nbtool.py verify` 认这类声明并跳过比对。

## 源脚本归档（`_py_source/`）

162 个课程 `.py` 按原章节目录结构归档在此。搬运用的是 `git mv`，所以**每个文件的历史都还在**。
它们仍然可运行——凡是「notebook 里被改写过、想对照原文」的地方，都回这里查。

命名沿用旧约定：`NN_主题.py` 是课案精简版、`NN_主题_jxsd.py` 是完整版、
`NN_主题_官方补充.py` 是官方文档补充篇 —— 三者现在都被合并进同一个 notebook。
（`_py_source/03_deepagents/create_txt.py` 例外：它是课程演示**运行时生成**的产物，
不是课案文件，保留归档但不参与合并，已在 `notebook_manifest.json` 里登记为排除项。）

## 工具链（`_tools/`）

| 工具 | 用途 |
|---|---|
| `nbtool.py` | `py2nb`（percent 源→ipynb）/ `nb2py`（还原成文本审阅）/ `strip`（清输出）/ `check`（结构检查）/ `coverage`（源脚本零丢失审计）/ `verify`（核对「预期输出」与实跑是否一致） |
| `run_notebooks.py` | 批量无头执行：分串行/并行车道，判定 PASS / PASS-降级 / TRACEBACK / TIMEOUT |
| `notebook_manifest.json` | 32 个 notebook ↔ 159 个源脚本的权威映射，也是 `coverage` 的输入 |

**为什么不直接手写 `.ipynb`**：它是 JSON，手改极易出错、diff 也几乎不可读。
所以这里的做法是「作者写 percent 格式的 `.py` → 工具转成 `.ipynb`」，往返**逐字节一致**。
改 notebook 的推荐流程：

```powershell
cd F:\ProGram\Python_Base
& .\.venv\Scripts\python.exe Agent\_tools\nbtool.py nb2py Agent\01_langgraph\01_基础图与状态.ipynb F:\temp\改我.py
#  ……编辑 F:\temp\改我.py……
& .\.venv\Scripts\python.exe Agent\_tools\nbtool.py py2nb F:\temp\改我.py Agent\01_langgraph\01_基础图与状态.ipynb
& .\.venv\Scripts\python.exe Agent\_tools\nbtool.py check Agent\01_langgraph\01_基础图与状态.ipynb
& .\.venv\Scripts\python.exe Agent\_tools\run_notebooks.py Agent\01_langgraph\01_基础图与状态.ipynb
```

> `run_notebooks.py` 的位置参数**两种写法都认**：`Agent\01_langgraph\xx.ipynb`（相对仓库根）
> 或 `01_langgraph\xx.ipynb`（相对 `Agent/`）；不传则跑全部，传目录名则跑整章。

## 官方文档缺口补充（非课案，`_官方补充` 系列）

课案讲完之后，我们逐页读完了三个框架的官方 Python 文档并做了比对，
结论汇总在 **`官方文档缺口对照.md`**（三框架各 12 项缺口 + 已覆盖对照 + 8 条实测踩坑）。
其中**已补成可运行代码**的有：

| 文件 | 补的缺口 | 是否离线 |
|---|---|---|
| `01_langgraph/10_控制流与函数式API_官方补充.py` | Send 并行扇出、`Command(goto)`、`@entrypoint/@task` 函数式 API | ✅ 0 次模型调用 |
| `01_langgraph/11_容错与测试_官方补充.py` | `RetryPolicy`、节点超时、官方测试三模式 | ✅ 0 次模型调用 |
| `01_langgraph/12_记忆_持久化与中断进阶_官方补充.py` | 记忆裁剪/删除/摘要、durability 三档、多中断并行、工具内中断 | ✅ 0 次模型调用 |
| `01_langgraph/13_长期记忆_官方补充.py` | 三类记忆分类、Store 语义搜索对照实验、记忆进 agent、TTL 限制 | ⚠️ 需 bge-m3 |
| `01_langgraph/14_子图持久化_官方补充.py` | 子图持久化三档对照（计数/中断/下钻子图快照） | ✅ 0 次模型调用 |
| `02_langchain/16_测试与护栏_官方补充.py` | 假模型单测、轨迹断言、确定性护栏、Runtime Context 注入 | ✅ 假模型 |
| `02_langchain/17_Skills渐进披露_官方补充.py` | Skills 工具化加载 + 引用感知（多 Agent 第 4 种模式） | ⚠️ 需真实模型 |
| `02_langchain/18_自定义工作流_官方补充.py` | agent 当节点、三节点 RAG 工作流、条件边回炉循环（第 5 种模式） | ⚠️ 需真实模型 |
| `02_langchain/19_事件流v3_官方补充.py` | v3 类型化投影、interleave 消费规则、工具失败可见性 | ⚠️ 需真实模型 |
| `02_langchain/20_上下文工程_官方补充.py` | 上下文工程 3×3 总纲索引 + 动态工具集/响应格式/三数据源/生命周期打点 | ⚠️ 需真实模型 |
| `02_langchain/21_MCP进阶_官方补充.py` | MCP 连接生命周期、多服务端命名空间、三原语取法、接 deepagents | ⚠️ 需真实模型 |
| `02_langchain/22_自己组装harness_官方补充.py` | 五步手装 harness + 与 create_deep_agent 默认栈对照 | ⚠️ 需真实模型 |
| `02_langchain/23_模型配置进阶_官方补充.py` | 模型参数与响应元数据、限流器、token 核算、超时验证、多模态现实 | ⚠️ 需真实模型 |
| `02_langchain/24_RAG知识库_官方补充.py` | RAG 全链路：bge-m3 向量化 → 召回 → bge-reranker 精排 → 作答 → agentic RAG | ⚠️ 需真实模型 + SiliconFlow |
| `02_langchain/11_内置中间件_官方补充.py` | ToolError / ModelFallback / ToolCallLimit / PII / LLMToolEmulator | ✅ 假模型 |
| `03_deepagents/14_上下文治理_官方补充.py` | 内置上下文压缩（卸载）、FilesystemPermission、write_todos opt-in | ✅ 剧本模型 |
| `03_deepagents/15_自定义后端_官方补充.py` | 从零实现 BackendProtocol、只读后端、审计与限流/校验策略钩子 | ✅ 剧本模型 |
| `03_deepagents/16_Rubric评分循环_官方补充.py` | RubricMiddleware 自评迭代、评分器带工具取证、迭代上限保护 | ⚠️ 需真实模型 |
| `03_deepagents/17_RAG_检索卸载委派_官方补充.py` | 三种 RAG 架构 + 检索-卸载-委派（全文落盘、子代理读） | ⚠️ 需真实模型 + SiliconFlow |
| `03_deepagents/18_解释器与PTC_官方补充.py` | eval 沙箱（QuickJS）、沙箱边界、PTC、动态子代理扇出 | ⚠️ 需真实模型 + `uv add "deepagents[quickjs]"` |
| `03_deepagents/19_异步子代理_官方补充.py` | 自助起 Agent Protocol 服务（langgraph dev）、异步任务 start/check/list | ⚠️ 需真实模型 + `uv add "langgraph-cli[inmem]"` |

这些文件**不是课案内容**，所以不带 `_jxsd` 后缀；共同特点：全部可离线复现
（用继承 `ChatOpenAI` 的剧本模型替代真模型），注释里标了官方文档路径与本地实测结论。

## 使用前准备

1. **配好根目录 `.env`** —— 全仓库只此一份配置，notebook 里统一 `from config import settings`：
   大模型三件套 `API_KEY` / `BASE_URL` / `MODEL_NAME`、向量化 `EMBEDDING_*`、重排 `RERANK_*`，
   以及数据库与 Langfuse 的连接信息。
2. **装依赖**（已执行过）：`uv sync`。
3. **注册内核**（见开头「先做一次」，只做一次）。
4. PostgreSQL（持久化记忆用）：
   ```powershell
   docker run -e POSTGRES_PASSWORD=<你的口令> -d --name postgres -p 5432:5432 postgres:18
   docker exec -it postgres psql -U postgres -c "CREATE DATABASE langgraph;"
   ```
   连接串写在 `.env` 的 `PG_URI`（形如 `postgresql://<用户名>:<口令>@127.0.0.1:5432/langgraph`）。
5. **冒烟**：打开任意 🟢 档 notebook 从头跑一遍；或命令行无头执行某一本：
   ```powershell
   & .\.venv\Scripts\python.exe Agent\_tools\run_notebooks.py 01_langgraph\01_基础图与状态.ipynb
   ```

## 建议阅读顺序

1. `01_langgraph` 4 本 → `02_langchain` 6 本 → `03_deepagents` 5 本 → `04_function_call` 2 本（基础能力）
2. `05_mcp` 5 本：**服务端在 notebook 内部自己起**，不用再另开窗口（端口已分配好，互不冲突）
3. `06_langfuse` 3 本：先在 Langfuse 控制台建好项目，把密钥填进根目录 `.env`
4. `07_protocols` 2 本：需额外安装 `deepagents-acp` / `a2a_auto_wrapper`；缺包时 notebook 会打印中文提示并走降级路径
5. `08_skills` 2 本：引用仓库里 `skills/` 与 `.claude/` 下的**真实技能目录**（不会另生成一份）
6. `09_aegra_deploy` 2 本：需要 `aegra` CLI 才能真跑部署那段，缺了就降级演示
7. `10_workflow_platform` 1 本：Langflow 未启动时自动降级为 dry-run

## 三个模型端点（都在根目录 `.env`，代码统一 `from config import settings`）

| 用途 | 配置项 | 当前值 |
|---|---|---|
| 作答 / Agent 推理 | `API_KEY` / `BASE_URL` / `MODEL_NAME`（Agent 课案扁平字段） | `deepseek-flash` @ DeepSeek 官方端点 |
| 向量化 | `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` / `EMBEDDING_SIZE` | `BAAI/bge-m3`（**1024 维**）@ SiliconFlow |
| 重排序 | `RERANK_API_KEY` / `RERANK_BASE_URL` / `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` @ SiliconFlow |

两点实测提醒：
- **rerank 分数按相对差距读**：有答案时 top1 能到 0.95、top2 只有 0.003（差几百倍）；
  语料里没有答案时最高分也可能低到 0.03。所以别用固定阈值（`RERANK_RELEVANCE_P=0.65`
  在这种尺度下会误判），看 top1 与 top2 的差距更可靠。
- 走第三方 OpenAI 兼容端点做向量化时，`OpenAIEmbeddings` 要设
  `check_embedding_ctx_length=False`，否则它会按 OpenAI 的 tiktoken 规则预切分文本。

### 换端点的代价：结构化输出（实测三组对照，2026-09）

扁平字段决定 **166 个文件跑得快不快**，也决定 **结构化输出类示例能不能跑通** ——
两者在本机是矛盾的，换端点前先看这张表：

| 端点 / 模型 | 纯聊天延迟 | 原生 `json_schema` | 强制 `tool_choice` | 后果 |
|---|---|---|---|---|
| 课案网关 `grok-4.6` | 单次可达 50 s+，整仓复核会拖到小时级 | ✅ | ✅ | 结构化输出全部正常，但慢 |
| DeepSeek `deepseek-flash` / `deepseek-v4-pro` | **1~10 s**（整仓复核 1.5 分钟级） | ❌ `This response_format type is unavailable now` | ❌ `Thinking mode does not support this tool_choice` | 快，但结构化输出跑不了 |

DeepSeek 是**思考模型**（两个模型都是），两条结构化输出的路都被拒；`method="json_mode"`
虽然能通 HTTP，但它**不约束字段名** —— 实测把 `goal/steps` 输出成了 `目标/步骤`，
解析照样失败。所以：

- 当前 `.env` 指向 DeepSeek 时，`02_langchain/08_结构化输出.py`、
  `20_上下文工程_官方补充.py` Demo 2、`03_deepagents/16_Rubric评分循环_官方补充.py`
  会打印「[跳过] / grader_error」的中文说明，**这是端点的限制，不是代码有问题**；
- 要完整跑这三处，把 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 换回支持
  `json_schema` 或强制 `tool_choice` 的端点即可（代码不用改）；
- 完整对照与原因见 `02_langchain/20_上下文工程_官方补充.py` 文末「实测结论」第 5 条。


## `_jxsd` 版的几处设计约定

- **缺包不报错**：课案里用到但本项目没装的第三方库（`ragas` / `crewai` / `a2a` /
  `deepagents-acp` / `langsmith` 沙箱 等）一律 `try/except ImportError` 兜住，
  模块层永远能 import 成功；`__main__` 里检测前置条件，缺什么就打印中文提示
  （要装什么包、要起什么服务），**不抛 traceback**。
- **缺密钥不报错**：Langfuse / 百度千帆 / Gitee / DashScope 的密钥为空时走降级演示，
  密钥填上即自动切回真实链路。
- **交互式 `input()`**：保留课案的「按回车继续」演示效果，但加了 `isatty()` +
  `EOFError` 兜底，非交互环境自动继续（本机实测 `isatty()` 返回 True 但读取会直接 EOF）。
- **不硬编码凭据**：数据库连接串一律 `settings.pg_uri`；课案原文里的连接串字面量
  已脱敏为 `<用户名>` / `<口令>@127.0.0.1`，只作对照保留。

## 课案原文的已知问题（`_jxsd` 版已修正并在注释里标注）

| 位置 | 课案写法 | 实际情况 |
|---|---|---|
| 多处 | `from conf import settings` / `setting.MODEL_NAME` | 课案自己有 `conf.py`，但函数名与变量名前后不一致；本项目统一 `from config import settings` |
| `01_langgraph` 短期记忆 | `DELETE FROM checkpoint WHERE created_at < ...` | 实测表名是 `checkpoints`（复数）且**没有 `created_at` 列**，该 SQL 跑不通 |
| `01_langgraph` 长期记忆 | `store.search(ns, query=...)` | 没配向量索引时 `query` 被**静默忽略**，退化成按时间倒序取 N 条 |
| `03_deepagents` 记忆 | `store.put(ns, "/memories/AGENTS.md", ...)` | `CompositeBackend` 会剥掉路由前缀，写成这样 Agent 会在 `/memories/memories/` 找文件 |
| `03_deepagents` 子智能体 | 三个 SubAgent 都叫 `researcher` | 直接 `ValueError: Duplicate subagent name` |
| `04_function_call` 参数类型 | `{"type": "number", "enum": ["celsius", "fahrenheit"]}` | 枚举值却是字符串，类型标错 |
| `08_skills` 云技能 | `store.put(ns, "/skills/code-review/SKILL.md", ...)` | 同上前缀问题，会导致「发现技能数 = 0」 |
| `08_skills` deepagents 技能 | `skills=["/"]`（扫根目录） | 参数要指向**技能的父目录**；指向具体技能目录会 0 发现 |
| `09_aegra_deploy` | `aegra.json` 里 graph 名是 `agent`，调用段写 `bushu` | 调用段会 404 |
| `05_mcp` 提示词 | `@mcp.prompt` 返回 `list[dict]` | FastMCP 3.4.7 会报 `messages[0] must be Message or str`，要用 `Message` 对象 |
| `05_mcp` 权限 | `create_token` 只签 `sub/exp/iat` | 配 `required_scopes=["read"]` 会全部 401 —— 令牌里必须有 `scope` |
| `05_mcp` 部署 | 内置工具含 `write_todos` | deepagents 0.7.13 已移除该工具，实际内置 9 个（`ls/read_file/write_file/edit_file/delete/glob/grep/execute/task`） |

## 排障补充：脚本「跑到一半什么都没输出」多半是重定向的锅

跑长脚本（尤其要调模型、跑 agent 的）时，**不要用 PowerShell 的 `*>` 或管道重定向收集输出**：
它会把子进程的 stdout 缓冲住，一旦脚本中途被中断（Ctrl+C、网关抖动、超时），
**缓冲区直接丢失，日志为空**，看起来像"代码崩了"，实际代码根本没报错。

- ❌ `python script.py *> out.log`（中断后 out.log 可能是空的）
- ❌ `python script.py 2>&1 | Select-Object -Last 20`（管道提前关闭还会**终止**上游进程）
- ✅ 让 Python 自己写日志：
  ```powershell
  $code = 'import runpy, sys; sys.stdout = sys.stderr = open(r"out.log", "w", buffering=1, encoding="utf-8"); runpy.run_path(r"script.py", run_name="__main__")'
  & .\.venv\Scripts\python.exe -u -X utf8 -c $code
  ```
- ✅ 或者干脆直接跑、让输出实时打到终端（不加任何重定向）

这条踩坑在本项目的补充篇开发中真实发生过：一次 `03_deepagents/17_RAG_检索卸载委派_官方补充.py`
的"崩溃"其实已经跑通（临时目录里留着落盘文件、日志却为空），换用上面第 3 种方式后
一次通过。**结论：日志为空 ≠ 代码有问题，先怀疑缓冲。**

## 验证方式与结果

每个 `_jxsd.py` 交付前都按同一套标准过检：

| 检查 | 做法 | 结果 |
|---|---|---|
| 语法 | `python -m py_compile` | 78/78 通过 |
| 真跑 | 独立子进程实跑，记录退出码与输出 | 78/78 退出码 0、无顶层 traceback |
| 配置引用 | AST 级扫描「真代码」（注释/docstring 不计） | 78 个文件里，用到 `settings.*` 的**全部** `from config import settings`；不存在的字段 0 处；非白名单环境变量 0 处；硬编码凭据 0 处 |
| 脱敏 | 正则扫描连接串 / Key / 内网 IP | 0 处命中（课案原文里的 `postgres:postgres@localhost` 已改成 `<用户名>:<口令>@127.0.0.1`） |
| 讲解密度 | （注释行 + docstring 行 + 含中文的字符串行）/ 总行数 | 平均 **57%**，最低 42% |

### 全仓库运行复核（最新一轮 2026-09-17：162 个 `.py`，16.6 分钟）

用独立子进程逐个真跑（`run_all.py`：串行/并行分道、端口类文件串行、超时保护、喂空行给
`input()`），结果：**160 个 PASS/SERVER-OK + 2 个异常**，两个都已定位清楚并处理：

| 异常 | 真相 | 处理 |
|---|---|---|
| `05_mcp/09_权限_客户端.py` 连接被拒 | 它是**客户端**，要求先手工起认证服务(9000) + 权限服务(8000) | 编排两个前置服务后实跑通过（见下表末行）；单独跑必然连不上 |
| `03_deepagents/19_异步子代理_官方补充.py` `PermissionError: [WinError 32]` | Windows **不会立刻**释放刚被 taskkill 的进程持有的文件句柄，紧接着删临时目录就撞锁 | 清理改成「小步重试 + 容忍失败」的显式清理；连跑 2 次均 PASS |

> 文件数口径：`Agent/` 下共 **169 个 `.py`**，harness 实跑 162 个 —— 另外 7 个在
> `03_deepagents/tmp_jxsd_deepagents_*/` 里，那是课案脚本**运行时自己生成**的工作目录
> （已被 `.gitignore` 的 `Agent/*/tmp_*/` 覆盖，删掉后重跑会重新生成，已实测）。
> 组成：78 个课案 `_jxsd` + 63 个课案精简版 + 21 个官方补充篇。

### 历史轮次记录（2026-09-16，142 个 `.py`）

那一轮：**137 个 PASS + 5 个环境类异常，全部定位清楚、无一是代码 bug**。

| 现象 | 真相 | 处理 |
|---|---|---|
| `05_mcp/02_客户端_jxsd.py` 报 TRACEBACK | 课案**故意**演示的 `div(1,0)` 服务端报错（进程退出码其实是 0） | 判定改为「rc=0 优先」，标注 `PASS-LOGTB` |
| `01_langgraph/07_中断_接口版.py` EXIT3 | 该文件是**常驻 FastAPI（8000 端口）**，与同批的 05_mcp 抢端口 | 归入「常驻服务」类，独占运行、判启动成功 |
| `05_mcp/08_权限_服务端.py` TIMEOUT | 同上：`mcp.run` 常驻服务，本就永不退出 | 同上（启动无 Traceback 即通过） |
| `05_mcp/09_权限_客户端.py` 连接被拒/502 | 课案原版要求**先手工启动**认证服务(9000)+权限服务(8000) | 编排两个前置服务后实跑通过（拿到 JWT→列工具→调用成功） |
| `03_deepagents/05_后端_Filesystem_jxsd.py` 超时 | 当时模型网关抽风挂住了调用；重跑 93.6 秒正常完成 | 无需改动 |

## 排障：跑不起来时先看这 5 条

1. **必须用仓库自带的 venv**，别用系统 Python：
   - ✅ `uv run Agent/02_langchain/01_模型_jxsd.py`（或 `& .\.venv\Scripts\python.exe Agent\...`）
   - ❌ PATH 里的 `python` 是 Windows Store 占位符（**静默失败、什么都不输出**）；
   - ❌ `F:\ProGramApp\Anaconda\python.exe` 里装的是另一套老版本依赖，`import langchain`
     会因 pydantic 版本冲突直接报错 —— **「大量导包报错」几乎都是解释器用错**，
     仓库代码本身的导入是干净的（282 个文件 AST 级扫描：语法错误 0、未兜底导入错误 0）。
2. **命令必须在项目根目录 `F:\ProGram\Python_Base` 下执行**：代码统一 `from config import settings`，
   换目录会 `ModuleNotFoundError: config`。
3. **本机开着 Clash 等系统代理时**，127.0.0.1 的回环请求会被代理接管，表现为
   `McpError: Session terminated` 或莫名的 `502 Bad Gateway`。跑 MCP/HTTP 类示例前设：
   ```powershell
   $env:NO_PROXY = "127.0.0.1,localhost"
   ```
4. **05_mcp 目录内不要并发跑**：多个文件都要占 8000 端口，必须一个一个来
   （`01_服务端` / `08_权限_服务端` / `01_langgraph/07_中断_接口版` 属于常驻服务，
   会一直挂着，验证「能否启动」即可，别等它退出）。
5. **免费模型额度**：若 `.env` 用的是 OpenRouter 免费档模型，跑一天几十次后会返回
   `429 ... free-models-per-day`（北京时间 0 点重置）。批量跑示例前先确认额度，
   或换成不计入免费额度的模型。

## 关于本机模型的实测提示

`_jxsd` 版里凡是要调大模型的地方都用 `.env` 的 `MODEL_NAME`。
**最新实测（2026-09）**：

| 项 | 网关 `grok-4.6` | DeepSeek `deepseek-flash`（当前） |
|---|---|---|
| 单次请求 | 网关繁忙时可达 50 s+，早期实测 2/2 超时（120 s 上限） | 1~10 s 稳定返回 |
| 工具调用 | 单个工具的 function call 3/3 稳定；工具多 + 中间件多时也能正常发起 `tool_calls` | 正常 |
| 结构化输出 | ✅ | ❌（思考模型，两条路都被 400 拒绝，见上文「换端点的代价」） |

- 代码里为早期不稳定版本加的中文兜底提示（「本轮模型没有发起工具调用」）无需删除 ——
  换个端点时它们仍然有用。
- 网关的 `grok-4.6` 回答风格偏「有个性」（会在正文里吐槽），演示时注意别当成 bug；
  它走的是 OpenRouter 免费额度，批量跑会触发 429（见排障第 5 条）。
- 换端点只需改 `.env` 那三个扁平字段，**代码一行都不用动** —— 这也是课案统一走
  `from config import settings` 的意义。

