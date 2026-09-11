# Python_Base — Python 学习与项目工作区

统一使用根目录的 **uv 虚拟环境**（`.venv`）与 **全局配置**（`config.py` + `.env`）。
所有子项目一律通过 `from config import settings` 读取配置。

## 目录结构

| 目录 | 内容 |
|---|---|
| `config.py` / `.env` | 全局统一配置（pydantic-settings），合并 RAG / Agent 全部配置项 |
| `RAG/` | 财务 RAG 智能问答系统（Milvus + Redis + Elasticsearch + SiliconFlow） |
| `Agent/` | Agent 课案代码（LangGraph / LangChain / DeepAgents / MCP / Langfuse / ACP / A2A） |
| `Back_End/` | 后端课案实现（Linux/uv/日志/配置文件/Flask/FastAPI/pytest/容器部署） |
| `Machine_Learning/` | 机器学习课案实现（sklearn / 集成学习 / 无监督 / 评估与保存） |
| `Deep_Learning/` | 深度学习课案实现（PyTorch / CNN / RNN / Attention / Transformer / 训练组件） |
| `Front_End/` | 前端课案实现（HTML / CSS / JavaScript / Streamlit 共 18 个应用） |
| `Crawler/` `Data_Analysis/` `Data_Structure/` `Py_Advanced/` | 其他学习项目 |

## 环境准备

```bash
uv sync                       # 按 pyproject.toml + uv.lock 安装依赖
uv run python config.py       # 验证配置加载
```

数据库等外部服务（见 `.env`）：

```bash
# PostgreSQL（LangGraph 记忆 / RAG 业务库）
docker run -e POSTGRES_PASSWORD=postgres -d --name postgres -p 5432:5432 postgres:18
docker exec -it postgres psql -U postgres -c "CREATE DATABASE langgraph;"

# Milvus / Redis / MySQL 按需用 docker 启动
```

注意：`.pth` 文件（`.venv/Lib/site-packages/python_base_root.pth`）把项目根目录
加入 `sys.path`，重建 venv 后需重新创建，内容为一行：`F:\ProGram\Python_Base`。

同理，`.venv/Lib/site-packages/sitecustomize.py` 是控制台编码兜底（Windows 默认 GBK，
脚本打印 `✓ ² ∂` 等符号时若输出被重定向到管道会抛 `UnicodeEncodeError`）。
它由解释器启动时自动导入，重建 venv 后也需要重新放回，缺了不影响功能，只是少数脚本在
管道/重定向场景可能报编码错。

## 课案实现（四个课案目录）

四份课案（后端 / 机器学习 / 深度学习 / 前端）的全部知识点都已落地为**可运行、带详细中文注释与讲解**
的代码，每个目录都有 `README.md`（知识点索引）、`verify_all.py`（一键自检）与 `VERIFY_REPORT.md`（实测记录）。

```bash
# 一键自检（各自独立，互不影响）
uv run python Back_End/verify_all.py
uv run python Machine_Learning/verify_all.py
uv run python Deep_Learning/verify_all.py
uv run python Front_End/verify_all.py

# 单独运行示例
uv run python Back_End/06_FastAPI/01_最小应用.py --check     # --check 用 TestClient 自检，不起服务
uv run python Machine_Learning/04_集成学习/02_XGBoost.py
uv run python Deep_Learning/02_网络架构/06_Transformer完整实现.py
uv run python -m streamlit run Front_End/04_Streamlit/15_图书管理系统/app.py
```

说明：
- 外部服务（MySQL / Redis / PostgreSQL / Milvus）本机未启动，相关示例会自动降级
  （SQLite / Celery eager 模式 / try-except 探活），**默认运行零报错、不阻塞**。
- 绘图统一使用 `matplotlib` 的 `Agg` 后端并保存到各自目录的 `output/`，不调用 `plt.show()`。
- 深度学习为纯 CPU 版 PyTorch，单个脚本运行时间均 < 40 秒，不依赖网络下载数据集。

## RAG 项目

```bash
# 初始化 Milvus（建库/建用户/授权）
uv run python RAG/core/milvus_init.py

# OCR 票据 -> Markdown（支持 --limit 调试）
uv run python RAG/data_process/paddle_ocr.py --categories flight --limit 5

# 字段抽取 -> Milvus
uv run python RAG/data_process/tick_extract.py --limit 5 --insert

# 向量化入库 / 预置问答缓存
uv run python RAG/data_process/embed_tickets.py
uv run python RAG/data_process/seed_qa_cache.py

# 启动服务（FastAPI + Chainlit，默认 127.0.0.1:8099）
uv run python RAG/app/main.py
# 接口：POST /api/chat  {"question": "..."}
```

依赖 Elasticsearch 的关键词检索需要先启动 ES（`ES_HOSTS`）。

## Agent 课案

```bash
uv run python Agent/01_langgraph/01_基础图.py     # 示例
uv run python Agent/03_deepagents/01_智能体.py    # DeepAgents 示例
```

章节：`01_langgraph` / `02_langchain` / `03_deepagents` / `04_function_call` /
`05_mcp` / `06_langfuse` / `07_protocols`，每个文件头部有详细说明与运行方式。
沙箱示例（`08_后端_Sandbox.py`）需 LangSmith 或 Opensandbox 服务端。
