# Agent 课案代码整理

课案来源：`G:\笔记\LLM\课件导出\Agent.html`
本项目把课案中全部代码按章节整理为可独立运行的 Python 文件，附详尽中文注释。

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
| `05_mcp/` | FastMCP 服务端/客户端、资源、提示词、JWT 权限、Docker 部署 |
| `06_langfuse/` | 追踪、会话与用户、提示词管理、打分、RAG/Agent 评估 |
| `07_protocols/` | ACP（编辑器↔Agent）、A2A（Agent↔Agent）协议 |

## 使用前准备

1. 配置在 Python_Base 根目录 `.env`（已就绪，含大模型 API Key、数据库、Langfuse 等）。
2. PostgreSQL（持久化记忆用）：
   ```bash
   docker run -e POSTGRES_PASSWORD=postgres -d --name postgres -p 5432:5432 postgres:18
   docker exec -it postgres psql -U postgres -c "CREATE DATABASE langgraph;"
   ```
3. 运行任意示例：`uv run 01_langgraph/01_基础图.py`

## 运行顺序建议

1. `01_langgraph` → `02_langchain` → `03_deepagents` → `04_function_call`（基础能力）
2. `05_mcp`：先启动 `01_服务端.py http`，再运行客户端
3. `06_langfuse`：先在 Langfuse 控制台建好项目并把密钥填入根目录 `.env`
4. `07_protocols`：需额外安装 `deepagents-acp` / `a2a_auto_wrapper`（见各文件头部说明）
