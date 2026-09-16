# Agent 课案代码整理

课案来源：`G:\笔记\LLM\课件导出\Agent.html`
本项目把课案中全部代码按章节整理为可独立运行的 Python 文件，附详尽中文注释。

目录下有**两套**代码，并存不冲突：

| 版本 | 文件命名 | 特点 |
|---|---|---|
| 精简版 | `01_基础图.py` | 早期整理，一节能跑通的最小实现，注释偏「知道怎么用」 |
| **课案版** | `01_基础图_jxsd.py` | **对照课案原文的完整实现**，代码量与讲解都更足，注释偏「知道为什么」 |

`_jxsd` = 课案（江西师大课件）版本。两套文件**除后缀外同名**，可以左右对照阅读：
精简版看骨架，`_jxsd` 版看细节、边界情况、课案原文与实测差异。

## 环境

```bash
# 创建虚拟环境并安装全部依赖（已执行）
uv sync

# Windows 控制台中文乱码时，先执行（脚本内已做 reconfigure 处理）
$env:PYTHONUTF8 = "1"
```

## 目录结构

| 目录 | 内容 |
|---|---|
| `../config.py` / `../.env` | 全局统一配置（Python_Base 根目录），所有示例统一 `from config import settings` |
| `01_langgraph/` | 状态图、短期/长期记忆、流式、中断、时间旅行、子图 |
| `02_langchain/` | 模型/消息/智能体/工具、记忆、流式、结构化输出、人工审核、中间件、多 Agent、管道 |
| `03_deepagents/` | create_deep_agent、7 种后端、人工审核、记忆、子智能体、Skills |
| `04_function_call/` | 原生 OpenAI / LangChain / DeepAgents 三种实现对比 |
| `05_mcp/` | FastMCP 服务端/客户端、资源、提示词、JWT 权限、Docker 部署、调试工具 |
| `06_langfuse/` | 追踪、会话与用户、提示词管理、打分、RAG/Agent 评估、标注与数据 |
| `07_protocols/` | ACP（编辑器↔Agent）、A2A（Agent↔Agent）协议 |
| `08_skills/` | 课案「skills」章：概念/原理/SKILL.md/渐进式加载/云技能（`_jxsd` 版新增目录） |
| `09_aegra_deploy/` | 课案「部署」章：LangSmith Deployments 的 license 坑 + Aegra 开源替代（`_jxsd` 版新增目录） |
| `10_workflow_platform/` | 课案「工作流 → 可视化平台」：Coze/Dify/n8n/Langflow 与 Langflow API（`_jxsd` 版新增目录） |

## 课案版（`_jxsd`）覆盖范围

共 78 个 `_jxsd.py`，按课案章节一一对应：

| 目录 | 个数 | 课案来源 |
|---|---|---|
| `01_langgraph/` | 10 | 智能体框架概览 + langgraph 基本概念/核心组件 |
| `02_langchain/` | 14 | langChain 核心组件 + 中间件 + 多 Agent |
| `03_deepagents/` | 13 | deepAgents 智能体/流式/运行环境/人工审核/记忆/子智能体 |
| `04_function_call/` | 5 | 工具调用 → function call（概念/实例/参数类型/langchain/deepagents） |
| `05_mcp/` | 12 | MCP 协议：快速开始/资源/提示词/四种协议/调试/Agent 调用/权限/部署 |
| `06_langfuse/` | 7 | 监控与评估：追踪/会话/提示词/打分/RAG 评估/Agent 指标/标注与数据 |
| `07_protocols/` | 5 | 协议：ACP（实操/原理）、A2A（CrewAI 端/DeepAgents 端/互相通信） |
| `08_skills/` | 5 | skills 全章 |
| `09_aegra_deploy/` | 5 | 部署全章 |
| `10_workflow_platform/` | 2 | 可视化平台 + Langflow API |

对照关系：`01_基础图.py` ↔ `01_基础图_jxsd.py`；课案独有的小节则用
`NN_主题_jxsd.py` 命名（如 `08_时间旅行_jxsd.py` 在精简版里没有单独文件）。
`02_langchain/15_管道.py` 课案没有对应章节，因此没有 `_jxsd` 版本。
`02_langchain/11_内置中间件_官方补充.py` 同样不带 `_jxsd`：它是**官方文档补充篇**
（课案没讲的 5 个内置中间件 Demo + 课案 7 个中间件的签名核对结论），全部用脚本模型
离线演示，0 次真实模型调用，输出逐字节稳定。
另：`10_中间件_钩子_jxsd.py` 在官方核对时补上了课案精简版有、但 HTML 六钩子表漏列的
`dynamic_prompt`（第 7 个官方装饰器，课案精简版把它当「钩子 3」用）。

## 官方文档缺口补充（非课案，`_官方补充` 系列）

课案讲完之后，我们逐页读完了三个框架的官方 Python 文档并做了比对，
结论汇总在 **`官方文档缺口对照.md`**（三框架各 12 项缺口 + 已覆盖对照 + 8 条实测踩坑）。
其中**已补成可运行代码**的有：

| 文件 | 补的缺口 | 是否离线 |
|---|---|---|
| `01_langgraph/10_控制流与函数式API_官方补充.py` | Send 并行扇出、`Command(goto)`、`@entrypoint/@task` 函数式 API | ✅ 0 次模型调用 |
| `01_langgraph/11_容错与测试_官方补充.py` | `RetryPolicy`、节点超时、官方测试三模式 | ✅ 0 次模型调用 |
| `01_langgraph/12_记忆_持久化与中断进阶_官方补充.py` | 记忆裁剪/删除/摘要、durability 三档、多中断并行、工具内中断 | ✅ 0 次模型调用 |
| `02_langchain/16_测试与护栏_官方补充.py` | 假模型单测、轨迹断言、确定性护栏、Runtime Context 注入 | ✅ 假模型 |
| `02_langchain/11_内置中间件_官方补充.py` | ToolError / ModelFallback / ToolCallLimit / PII / LLMToolEmulator | ✅ 假模型 |
| `03_deepagents/14_上下文治理_官方补充.py` | 内置上下文压缩（卸载）、FilesystemPermission、write_todos opt-in | ✅ 剧本模型 |
| `03_deepagents/15_自定义后端_官方补充.py` | 从零实现 BackendProtocol、只读后端、审计与限流/校验策略钩子 | ✅ 剧本模型 |

这些文件**不是课案内容**，所以不带 `_jxsd` 后缀；共同特点：全部可离线复现
（用继承 `ChatOpenAI` 的剧本模型替代真模型），注释里标了官方文档路径与本地实测结论。

## 使用前准备

1. 配置在 Python_Base 根目录 `.env`（已就绪，含大模型 API Key、数据库、Langfuse 等）。
   **全仓库只此一份配置**，`_jxsd` 代码不另建 `conf.py`。
2. PostgreSQL（持久化记忆用）：
   ```bash
   docker run -e POSTGRES_PASSWORD=<你的口令> -d --name postgres -p 5432:5432 postgres:18
   docker exec -it postgres psql -U postgres -c "CREATE DATABASE langgraph;"
   ```
   连接串写在 `.env` 的 `PG_URI`（形如 `postgresql://<用户名>:<口令>@127.0.0.1:5432/langgraph`）。
3. 运行任意示例：`uv run Agent/01_langgraph/01_基础图_jxsd.py`
   或 `& '.\.venv\Scripts\python.exe' 'Agent\01_langgraph\01_基础图_jxsd.py'`

## 运行顺序建议

1. `01_langgraph` → `02_langchain` → `03_deepagents` → `04_function_call`（基础能力）
2. `05_mcp`：先启动 `01_服务端_jxsd.py http`，再运行客户端
3. `06_langfuse`：先在 Langfuse 控制台建好项目并把密钥填入根目录 `.env`
4. `07_protocols`：需额外安装 `deepagents-acp` / `a2a_auto_wrapper`（见各文件头部说明）
5. `08_skills`：`02_SKILL示例_jxsd.py` 会先在磁盘上生成一个真实 skill 目录
6. `09_aegra_deploy`：`02_项目骨架_jxsd.py` 会生成一套 Aegra 项目骨架
7. `10_workflow_platform`：Langflow 未启动时自动降级为 dry-run

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

## 验证方式与结果

每个 `_jxsd.py` 交付前都按同一套标准过检：

| 检查 | 做法 | 结果 |
|---|---|---|
| 语法 | `python -m py_compile` | 78/78 通过 |
| 真跑 | 独立子进程实跑，记录退出码与输出 | 78/78 退出码 0、无顶层 traceback |
| 配置引用 | AST 级扫描「真代码」（注释/docstring 不计） | 78 个文件里，用到 `settings.*` 的**全部** `from config import settings`；不存在的字段 0 处；非白名单环境变量 0 处；硬编码凭据 0 处 |
| 脱敏 | 正则扫描连接串 / Key / 内网 IP | 0 处命中（课案原文里的 `postgres:postgres@localhost` 已改成 `<用户名>:<口令>@127.0.0.1`） |
| 讲解密度 | （注释行 + docstring 行 + 含中文的字符串行）/ 总行数 | 平均 **57%**，最低 42% |

### 全仓库运行复核（`_jxsd` + 精简版 + 官方补充，共 142 个 `.py`）

用独立子进程逐个真跑（`run_all.py`：串行/并行分道、端口类文件串行、超时保护、喂空行给
`input()`），结果：**137 个 PASS + 5 个环境类异常，全部定位清楚、无一是代码 bug**。

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

`_jxsd` 版里凡是要调大模型的地方都用 `.env` 的 `MODEL_NAME`（当前 `grok-4.6`）。
**最新实测（2026-09）**：该端点在单个工具的 function call 上 3/3 稳定，
在「工具多、中间件多、system prompt 复杂」的 Agent 里也能正常发起 `tool_calls`；
代码里为早期不稳定版本加的中文兜底提示（「本轮模型没有发起工具调用」）无需删除 ——
换个端点时它们仍然有用。
两点观察：① 该模型回答风格偏「有个性」（会在正文里吐槽），演示时注意别当成 bug；
② 它走的是 OpenRouter 免费额度，批量跑会触发 429（见排障第 5 条）。
