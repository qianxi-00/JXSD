# Python_Base · 核心项目分支（`subprojects`）

> 本分支是 `master` 的**独立分支**（orphan，与 master 没有共同提交历史），
> 树上只保留**四个核心项目目录**，用于单独查看、归档或分享这几个项目。
>
> 四份课案实现的完整工作区（`Back_End` / `Machine_Learning` / `Deep_Learning` / `Front_End`）
> 在 `master` 分支上。

## 目录结构

| 目录 | 内容 |
|---|---|
| `Agent/` | Agent 课案代码：LangGraph / LangChain / DeepAgents / Function Call / MCP / Langfuse / ACP·A2A |
| `RAG/` | 财务 RAG 智能问答系统：Milvus 向量召回 + Redis 缓存 + BM25 关键词召回 + Chainlit 界面 |
| `Crawler/` | 亚马逊商品采集示例：Scrapling 隐身抓取 + BeautifulSoup 解析 + LLM 结构化抽取 |
| `Data_Analysis/` | NumPy / Pandas / Matplotlib 数据分析示例（含 Jupyter 笔记本） |
| `config.py` | **全局统一配置入口**，四个项目都通过 `from config import settings` 读取 `.env` |
| `.env.example` | 环境变量**脱敏模板**（复制成 `.env` 再填真实值） |
| `pyproject.toml` / `uv.lock` / `.python-version` | uv 环境配置与依赖锁（Python 3.12） |

---

## 一、环境准备（uv）

### 0. 前置：目录名必须叫 `Python_Base`

`RAG/app/main.py` 会从自身位置逐级向上寻找名为 `Python_Base` 的目录来定位项目根
（这样才能 `from config import settings`、`from core.xxx import ...`）。
所以仓库落地目录要保持这个名字：

```bash
git clone -b subprojects https://github.com/qianxi-00/JXSD.git Python_Base
cd Python_Base
```

### 1. 安装 uv，同步依赖

```bash
# 安装 uv（macOS / Linux）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 uv（Windows PowerShell）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 按 pyproject.toml + uv.lock 建虚拟环境并装依赖
uv sync
```

`uv sync` 会自动创建 `.venv`（Python 3.12），安装约 300 个包。
本分支的依赖清单**只包含这四个项目真正用到的库**（用静态扫描四个目录的全部
`import` 反推出来），因此比 `master` 轻很多：不含 PyTorch、Streamlit、
scikit-learn、XGBoost、LightGBM、Flask、Celery 等。

验证：

```bash
uv run python config.py     # 打印一行行配置即表示 .env 读取正常
```

### 2. 让 `from config import settings` 在任何目录下都能用（重要）

`Agent/` 与 `RAG/` 里有几十个脚本写的是 `from config import settings`，
但直接 `python Agent/01_langgraph/01_基础图.py` 时，`sys.path[0]` 是脚本所在子目录，
找不到根目录的 `config.py`。解决办法是把项目根加入虚拟环境的 `sys.path`：

```bash
# Windows PowerShell（在项目根目录执行）
Set-Content -Path .venv\Lib\site-packages\python_base_root.pth `
            -Value (Get-Location).Path -Encoding ascii

# macOS / Linux（在项目根目录执行）
echo "$PWD" > .venv/lib/python3.12/site-packages/python_base_root.pth
```

> 重建虚拟环境（重新 `uv sync` 后删过 `.venv`）需要重新执行这一步。
> 另外 `RAG/app/main.py` 自带路径引导，从它启动时不受影响。

### 3. 配置环境变量

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

然后按需填写 `.env`。最少要配这几项才能跑起来：

| 用途 | 必填项 |
|---|---|
| Agent 示例调用大模型 | `API_KEY` / `BASE_URL` / `MODEL_NAME` |
| RAG 问答 | `LLM_*`（问答模型）、`EMBEDDING_*`（向量化）、`RERANK_*`（重排） |
| Crawler 智能抽取 | `API_KEY`（`Crawler/conf.py` 里的 `api_key` 字段） |
| Milvus / Redis / PostgreSQL | 对应的 `*_HOST` / `*_PORT` / `*_PASSWORD` |

> `.env` 已被 `.gitignore` 排除，**不要提交**；`.env.example` 是脱敏模板，可以提交。

### 4. Crawler 需要额外装一次隐身浏览器内核

`scrapling[fetchers]` 只装了取件器的 Python 依赖，浏览器内核要再执行一次：

```bash
uv run scrapling install
```

---

## 二、外部服务

RAG 与 Agent 的「生产级记忆」依赖下面这些服务，用 Docker 起最省事：

```bash
# Milvus 向量库（RAG 的向量召回）
# 参考官方 compose：https://milvus.io/docs/install_standalone-docker.md

# Redis（RAG 的问答缓存）
docker run -d --name redis -p 6379:6379 redis:7

# PostgreSQL（Agent 的 LangGraph 短期/长期记忆）
docker run -e POSTGRES_PASSWORD=postgres -d --name postgres -p 5432:5432 postgres:18
docker exec -it postgres psql -U postgres -c "CREATE DATABASE langgraph;"
```

初始化 Milvus（建库 / 建用户 / 授权，按你的 Milvus 部署方式调整）：

```bash
uv run python RAG/core/milvus_init.py
```

> 服务没起时，用到它们的脚本会明确报连接失败——这是预期行为，不是本分支的问题。

---

## 三、各项目运行方式

### Agent（`Agent/`）

按课案章节编号组织，每个文件头部都有「对应课案章节 / 本节知识点 / 运行方式」：

```bash
uv run python Agent/01_langgraph/01_基础图.py       # LangGraph 基础
uv run python Agent/02_langchain/03_智能体.py       # LangChain Agent
uv run python Agent/03_deepagents/01_智能体.py      # DeepAgents
uv run python Agent/04_function_call/agent_openai.py
uv run python Agent/05_mcp/01_服务端.py             # MCP 服务端（另开终端跑客户端）
uv run python Agent/06_langfuse/01_追踪.py          # 需要 .env 里配好 LANGFUSE_*
```

章节：`01_langgraph` / `02_langchain` / `03_deepagents` / `04_function_call` /
`05_mcp` / `06_langfuse` / `07_protocols`。

### RAG（`RAG/`）

```bash
# 1) 票据 OCR -> Markdown（支持 --limit 调试）
uv run python RAG/data_process/paddle_ocr.py --categories flight --limit 5

# 2) 字段抽取 -> Milvus
uv run python RAG/data_process/tick_extract.py --limit 5 --insert

# 3) 向量化入库 / 预置问答缓存
uv run python RAG/data_process/embed_tickets.py
uv run python RAG/data_process/seed_qa_cache.py

# 4) 启动服务（FastAPI + Chainlit，默认 127.0.0.1:8099）
uv run python RAG/app/main.py
#    接口：POST /api/chat   {"question": "..."}
#    界面：http://127.0.0.1:8099
```

### Crawler（`Crawler/`）

```bash
# 动态采集：StealthyFetcher 渲染 + CSS 锚点提取 -> amazon_kitchen_products.json
uv run python Crawler/dynamic_crawler.py

# 智能采集：抓取 + 卡片转 Markdown + LLM 抽 JSON -> amazon_kitchen_products_ai.json
uv run python Crawler/ai_crawler.py
```

目标 URL 在 `Crawler/conf.py` 的 `start_url`。
亚马逊风控较重，频繁抓取可能触发验证码；详细的选择器策略与反爬说明见 `Crawler/README.md`。

### Data_Analysis（`Data_Analysis/`）

```bash
uv run python Data_Analysis/demo.py          # NumPy 基础

# 打开 Jupyter 笔记本（demo1_np.ipynb / demo02_matplotlib.ipynb）
uv run jupyter lab
```

---

## 四、这个分支与 `master` 的差异

| 项目 | `master` | `subprojects`（本分支） |
|---|---|---|
| 提交历史 | 完整历史 | **独立 orphan 分支**，自成一条历史 |
| 目录 | 全部（含四份课案实现） | 只有 `Agent` / `RAG` / `Crawler` / `Data_Analysis` |
| 根目录文件 | 全部 | 只保留 `config.py`、`.env.example`、`.gitignore`、`pyproject.toml`、`uv.lock`、`.python-version`、`README.md` |
| 依赖清单 | 整个工作区（含 torch / streamlit / xgboost 等） | **只含这四个项目用到的库**（约 300 个包，无 PyTorch） |
| 用途 | 日常开发、跑四份课案 | 单独查看 / 归档 / 分享这几个项目 |

需要在同一个工作区里同时用两边时：

```bash
git switch master        # 回到完整工作区
git switch subprojects   # 只留四个核心项目目录
```

注意：`master` 与本分支没有共同祖先，`git merge` 两者会产生大量冲突，一般不需要合并。
