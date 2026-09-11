# Back_End —— 后端开发基础 · 课案实现总览

> 本目录把「后端开发基础」课案的**全部知识点**落地为**可运行、带详细中文注释与讲解**的代码与文档。
> 工作区：`F:\ProGram\Python_Base`　｜　本目录：`F:\ProGram\Python_Base\Back_End`

---

## 一、快速开始

```powershell
# 1) 跑一遍全部自检（推荐第一步，约 20 秒）
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Back_End\verify_all.py'

# 2) 跑测试
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m pytest 'F:\ProGram\Python_Base\Back_End\07_测试框架' -q

# 3) 启动某个 FastAPI 示例（浏览器打开 /docs 交互文档）
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Back_End\06_FastAPI\01_最小应用.py'
```

> **统一约定**：所有脚本都用仓库虚拟环境解释器运行，**不使用 `uv run`**（避免触发 sync 改动环境）。
> 路径含中文，PowerShell 里必须加引号并用 `&` 调用。

---

## 二、目录树

```
Back_End/
├── README.md                      本文件：总览 / 知识点索引 / 运行命令
├── verify_all.py                  一键运行本目录所有脚本并汇总（含 pytest 与接口校验）
├── VERIFY_REPORT.md               verify_all.py 的真实输出（交付证据）
├── pytest.ini                     pytest 配置（asyncio_mode=auto / marker 注册）
│
├── 01_基础环境/                   【纯文档】Linux 与服务器基础
│   ├── Linux常用命令速查.md
│   ├── Docker常用命令.md
│   ├── Ubuntu与AlmaLinux对比.md
│   └── 服务器部署与安全组.md
│
├── 02_包管理/
│   └── uv使用指南.md              【纯文档】uv 全流程（Python 版本/虚拟环境/pip 兼容/项目管理/uv run）
│
├── 03_配置文件/
│   ├── 01_pydantic_settings配置.py   5 个演示：基础用法 / 优先级 / 校验 / 多环境 / 根配置
│   ├── .env.example                  环境变量模板（可入库）
│   └── 配置文件讲解.md
│
├── 04_日志/
│   ├── 01_loguru基础.py              级别 / format / 结构化 / 自定义 sink
│   ├── 02_loguru文件与轮转.py         rotation / retention / compression / serialize / filter
│   └── 03_loguru异常捕获.py           logger.exception / @logger.catch / 耗时装饰器
│
├── 05_Flask/
│   ├── 01_最小应用.py                 路由与 TestClient 自检
│   ├── 02_路由与请求.py               参数来源 / 响应四种写法 / 错误处理
│   ├── 03_模板与静态文件.py            Jinja2 继承 / 过滤器 / 自动转义 / static
│   ├── 04_RESTful接口.py              完整 CRUD + Blueprint + 统一响应
│   ├── templates/                     base.html / index.html / about.html
│   └── static/                        style.css / logo.svg
│
├── 06_FastAPI/                    【核心】12 个示例 + 全量接口校验
│   ├── 01_最小应用.py                 自动文档 / 路径参数 / 422
│   ├── 02_GET与POST.py                路径·查询·请求体·表单·请求头·Cookie
│   ├── 03_数据模型与验证.py            Pydantic v2 全部 Field 与校验器
│   ├── 04_模板与静态文件.py            Jinja2Templates / StaticFiles
│   ├── 05_异步处理.py                 async vs 线程池 vs 阻塞事件循环（并发实测）
│   ├── 06_数据库集成.py               SQLAlchemy 2.0 + SQLite + MySQL 探活降级
│   ├── 07_依赖注入.py                 Depends 七种用法 + dependency_overrides
│   ├── 08_认证与授权.py               PyJWT HS256 + PBKDF2 密码 + 角色授权
│   ├── 09_中间件.py                   CORS / GZip / 自定义中间件（洋葱模型实测）
│   ├── 10_错误处理.py                 统一错误格式 + 四类异常处理器
│   ├── 11_Session与Cookie.py          SessionMiddleware / Cookie / 签名 Cookie
│   ├── 12_Celery后台任务.py           Celery + Redis（无 Redis 时 eager 模式）
│   ├── verify_api.py                  用 TestClient 校验上面全部示例的接口
│   ├── templates/、static/            04 示例用到的模板与静态资源
│
├── 07_测试框架/
│   ├── calculator.py                  被测模块（含冒烟自测）
│   ├── test_calculator.py             pytest 全套用法（59 passed / 1 skipped / 1 xfailed）
│   ├── conftest.py                    fixture：隔离 / yield 清理 / 参数化 / autouse
│   └── 测试说明.md
│
├── 08_代码格式化/
│   ├── README.md                       ruff / black / pre-commit 讲解与命令
│   ├── ruff.toml                       规则集与格式化配置（带中文注释）
│   └── .editorconfig                   编辑器统一约定
│
├── 09_容器部署/
│   ├── Dockerfile                      多阶段构建 + uv + 非 root + HEALTHCHECK
│   ├── docker-compose.yml              api + mysql + redis + worker 编排
│   ├── .dockerignore
│   ├── 检查YAML.py                     用 PyYAML 校验全部 YAML/Dockerfile（静态检查）
│   └── k8s/
│       ├── README.md                   K8S 概念 / Minikube / kuboard / 排障
│       ├── deployment.yaml             Deployment + PVC
│       ├── service.yaml                ClusterIP + NodePort + LoadBalancer
│       ├── configmap.yaml              ConfigMap + Secret
│       └── ingress.yaml                域名路由 + TLS + 限流
│
├── 10_综合实战/book_api/              小而完整的 FastAPI 项目
│   ├── main.py / database.py / models.py / schemas.py / crud.py
│   ├── routers/books.py                路由分层
│   ├── test_book_api.py                pytest + TestClient 测试
│   ├── conftest.py                     client / app_module / sample_book fixture
│   └── README.md                       运行与接口说明
│
└── data/                        运行时产物（自动创建）：SQLite / 日志 / JSON
```

---

## 三、知识点索引（课案章节 → 文件）

| 课案章节 | 落点文件 |
| --- | --- |
| **Linux**：什么是 Linux / 最小镜像 / 发行版家族 | `01_基础环境/Linux常用命令速查.md` |
| **Linux**：基础命令（文件、软硬链接、权限、进程、网络、系统管理） | `01_基础环境/Linux常用命令速查.md` |
| **Linux**：Ubuntu、AlmaLinux、两者对比 | `01_基础环境/Ubuntu与AlmaLinux对比.md` |
| **API 开发部署**：ECS 租用 / SSH 远程连接 / 安装 Docker / 安全组与防火墙 | `01_基础环境/服务器部署与安全组.md` |
| **包管理**：uv（版本、虚拟环境、pip 兼容、项目管理、uv run、完整工作流） | `02_包管理/uv使用指南.md` |
| **配置文件**：pydantic-settings、.env、多环境、优先级 | `03_配置文件/01_pydantic_settings配置.py`、`配置文件讲解.md` |
| **日志**：loguru 初始化 / 分级 / 文件与轮转 / 异常捕获 / 耗时装饰器 | `04_日志/01~03*.py` |
| **Web 框架**：框架概览（Flask/Django/FastAPI 对比） | `06_FastAPI/01_最小应用.py` 顶部说明 + `05_Flask/01_最小应用.py` |
| **Web 框架 / Flask**：最小应用 | `05_Flask/01_最小应用.py` |
| **Web 框架 / Flask**：路由与请求 | `05_Flask/02_路由与请求.py` |
| **Web 框架 / Flask**：模板与静态文件 | `05_Flask/03_模板与静态文件.py` + `templates/`、`static/` |
| **Web 框架 / Flask**：RESTful 接口 | `05_Flask/04_RESTful接口.py` |
| **FastAPI**：什么是 FastAPI / 安装与运行 | `06_FastAPI/01_最小应用.py` |
| **FastAPI**：GET vs POST / 参数来源 | `06_FastAPI/02_GET与POST.py` |
| **FastAPI**：数据模型与验证（Field 表 / 常用类型表） | `06_FastAPI/03_数据模型与验证.py` |
| **FastAPI**：模板与静态文件 | `06_FastAPI/04_模板与静态文件.py` + `templates/`、`static/` |
| **FastAPI**：异步处理 | `06_FastAPI/05_异步处理.py` |
| **FastAPI**：数据库集成（同步 ORM / 异步 ORM 说明） | `06_FastAPI/06_数据库集成.py` |
| **FastAPI**：依赖注入（为什么用 yield、不用会怎样） | `06_FastAPI/07_依赖注入.py`、`06_数据库集成.py` |
| **FastAPI**：认证与授权（认证/授权/JWT/bcrypt/OAuth2PasswordBearer） | `06_FastAPI/08_认证与授权.py` |
| **FastAPI**：中间件（CORS / 自定义 / GZip / 执行顺序） | `06_FastAPI/09_中间件.py` |
| **FastAPI**：错误处理（HTTPException / 状态码 / 统一处理） | `06_FastAPI/10_错误处理.py` |
| **FastAPI**：Celery + Redis 后台任务 | `06_FastAPI/12_Celery后台任务.py` |
| **FastAPI**：Session 和 Cookie | `06_FastAPI/11_Session与Cookie.py` |
| **测试框架**：pytest 基本用法 / fixture / 参数化 / 测试类 / 异步 | `07_测试框架/*`、`06_FastAPI/verify_api.py`、`10_综合实战/book_api/test_book_api.py` |
| **代码格式化**：Git Hook / ruff / black / pre-commit | `08_代码格式化/README.md`、`ruff.toml`、`.editorconfig` |
| **容器部署 / Docker**：Dockerfile / 构建运行 | `09_容器部署/Dockerfile`、`docker-compose.yml`、`01_基础环境/Docker常用命令.md` |
| **容器部署 / K8S**：核心概念 / 工作负载 / Namespace | `09_容器部署/k8s/README.md`、`k8s/deployment.yaml`、`k8s/service.yaml` |
| **容器部署 / K8S**：Minikube / kuboard / 应用部署 | `09_容器部署/k8s/README.md` |
| **容器部署**：配置校验 | `09_容器部署/检查YAML.py` |
| **综合实战** | `10_综合实战/book_api/` |

---

## 四、每个 FastAPI 示例的端口分配

示例脚本直接运行会 `uvicorn.run(...)` 启动服务，端口从 **8101** 起逐个递增，互不冲突：

| 文件 | 端口 | 文件 | 端口 |
| --- | --- | --- | --- |
| 01_最小应用.py | 8101 | 07_依赖注入.py | 8107 |
| 02_GET与POST.py | 8102 | 08_认证与授权.py | 8108 |
| 03_数据模型与验证.py | 8103 | 09_中间件.py | 8109 |
| 04_模板与静态文件.py | 8104 | 10_错误处理.py | 8110 |
| 05_异步处理.py | 8105 | 11_Session与Cookie.py | 8111 |
| 06_数据库集成.py | 8106 | 12_Celery后台任务.py | 8112 |
| Flask 示例 | 5001~5004 | book_api/main.py | 8113 |

**两种运行模式：**

```powershell
# 模式一：快速自检（不启动服务器、不阻塞，verify_all.py 用的就是这个）
& '...\.venv\Scripts\python.exe' '...\06_FastAPI\01_最小应用.py' --check

# 模式二：真正启动服务（阻塞，Ctrl+C 停止）
& '...\.venv\Scripts\python.exe' '...\06_FastAPI\01_最小应用.py'
# Flask 示例用 --serve 参数启动服务器
& '...\.venv\Scripts\python.exe' '...\05_Flask\01_最小应用.py' --serve
```

---

## 五、注意事项（重要）

### 1. 环境约定

- 只用 `F:\ProGram\Python_Base\.venv\Scripts\python.exe`（Python 3.12.12）运行；
- **不要**执行 `uv run` / `uv sync` / `pip install`，避免改动现有环境；
- 脚本内部一律用 `pathlib.Path(__file__).resolve().parent` 计算目录，
  **不依赖当前工作目录**（verify_all.py 特意在 `Back_End` 目录下运行所有脚本以验证这一点）；
- 运行产物（SQLite、日志、JSON）统一写入 `Back_End/data/`，自动创建。

### 2. 外部服务全部未启动（这是设计要求，不是故障）

| 服务 | 地址 | 本目录的处理方式 |
| --- | --- | --- |
| MySQL | 127.0.0.1:3306 | `06_数据库集成.py` 的 `probe_mysql()` 用 try/except 探测，失败只打印中文提示，默认用 SQLite |
| Redis | 127.0.0.1:6379 | `12_Celery后台任务.py` 配置 `task_always_eager=True`，任务在当前进程同步执行，**绝不阻塞等待** |
| PostgreSQL | 127.0.0.1:5432 | 未使用（本目录无依赖） |
| Milvus | 127.0.0.1:19530 | 未使用（本目录无依赖） |

**因此：所有脚本都能在"零外部服务"的情况下 0 报错跑完。**

### 3. 未安装的库（本目录用替代方案）

| 未安装 | 替代方案 |
| --- | --- |
| `flask-sqlalchemy` | 直接用 SQLAlchemy 2.0（Flask 的 RESTful 示例用内存存储演示语义） |
| `alembic` | `Base.metadata.create_all`（文档中说明了生产应使用迁移工具） |
| `gunicorn` | 开发用 uvicorn；Dockerfile 中说明生产可用 `--workers` 或多副本 |
| `passlib` / `bcrypt` | 标准库 `hashlib.pbkdf2_hmac` + 随机盐 + `hmac.compare_digest` |
| `python-jose` | `PyJWT`（API 更简洁，行为一致） |
| `ruff` / `black` / `isort` / `pre-commit` | 只提供配置文件与中文命令说明（`08_代码格式化/`） |
| `sqlmodel` | 课案示例用 SQLModel，本目录统一换成 SQLAlchemy 2.0 原生风格（更通用，且 SQLModel 未安装） |

### 4. 不会写坏环境的行为约定

- 不修改仓库根目录的 `pyproject.toml` / `uv.lock` / `config.py` / `.env` / `README.md`；
- 不触碰 `Machine_Learning` / `Deep_Learning` / `Front_End` 目录；
- 只在 `Back_End/` 内创建与修改文件。

### 5. 交付前自检

```powershell
# 全部脚本 + pytest + 接口校验，一步到位
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Back_End\verify_all.py'
```

期望结尾输出：

```
共 31 个脚本，成功 31 个，失败 0 个。
pytest 附加验证：通过 ✓
verify_api 接口校验：通过 ✓
全部自检通过 ✓
```

真实运行记录见 `VERIFY_REPORT.md`。

---

## 六、学习路径建议

1. **先看文档**：`01_基础环境/` → `02_包管理/`（建立环境与部署的整体认知）；
2. **再打基础**：`03_配置文件/` → `04_日志/`（每个后端项目都要用的两件事）；
3. **Web 入门**：`05_Flask/`（理解路由/请求/响应/模板这些通用概念）；
4. **主力框架**：`06_FastAPI/01 → 12` 按顺序读，每个文件都先讲原理再写代码；
5. **工程质量**：`07_测试框架/` → `08_代码格式化/`；
6. **上线部署**：`09_容器部署/`；
7. **综合演练**：`10_综合实战/book_api/`，自己动手加一个 `/authors` 接口试试。
