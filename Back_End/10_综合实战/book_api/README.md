# book_api —— 图书管理 API（综合实战）

> 对应课案章节：后端开发基础 → Web 框架（综合运用）
> 这是一个"小而完整"的 FastAPI 项目：分层清晰、能跑、能测、有文档。

---

## 一、这个项目演示了什么

| 课案知识点 | 在本项目中的落点 |
| --- | --- |
| FastAPI 应用创建与路由 | `main.py`、`routers/books.py` |
| Pydantic 数据模型与验证 | `schemas.py`（入参严格、出参精简） |
| 依赖注入 `Depends` 管理会话 | `database.py::get_db` + 路由里的 `DbSession` |
| SQLAlchemy 2.0 数据库集成 | `models.py`、`database.py`（SQLite，可换 MySQL） |
| RESTful 语义 | `/api/books` 上的 GET/POST/PUT/PATCH/DELETE |
| 中间件 | `main.py` 中的耗时统计与 `X-Request-Id` |
| 统一错误处理 | `main.py` 的三个 `exception_handler` |
| 测试框架 | `test_book_api.py`（pytest + TestClient，含参数化与异步用例） |
| 日志 | 关键位置的中文 `print`，可无缝换成 `04_日志` 里的 loguru |

**分层结构（重要）：**

```
HTTP 请求
   ↓
routers/books.py   路由层：解析请求 → 调 crud → 包装响应（不写 SQL）
   ↓
crud.py            数据层：只做数据库增删改查（不认识 HTTP）
   ↓
models.py          模型层：表结构（SQLAlchemy 2.0：Mapped + mapped_column）
schemas.py         契约层：接口收什么、吐什么（Pydantic）
database.py        基础设施：引擎、会话工厂、get_db、建表
```

分层的收益：换数据库只改 `database.py` + `crud.py`；测业务不用起 HTTP；
多人协作各改各的文件，冲突少。

---

## 二、目录结构

```
book_api/
├── __init__.py          包标识与版本号
├── database.py          引擎 / SessionLocal / get_db / init_db
├── models.py            ORM 模型（Book）
├── schemas.py           BookCreate / BookUpdate / BookPatch / BookOut / PageOut
├── crud.py              数据访问层
├── routers/
│   ├── __init__.py
│   └── books.py         图书 CRUD 路由（APIRouter，prefix=/api/books）
├── main.py              应用入口（中间件 + 异常处理 + 路由装配 + lifespan）
├── conftest.py          pytest fixture（client / app_module / sample_book）
├── test_book_api.py     pytest 接口测试
└── README.md            本文件
```

数据库文件：`Back_End/data/book_api.db`（首次启动自动创建）

---

## 三、运行方式

> 统一使用本仓库的虚拟环境解释器，路径含中文必须加引号，并用 `&` 调用。

### 1. 启动服务

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Back_End\10_综合实战\book_api\main.py'
```

启动后：

- 交互文档（Swagger UI）：http://127.0.0.1:8113/docs
- 另一种文档（ReDoc）：http://127.0.0.1:8113/redoc
- OpenAPI JSON：http://127.0.0.1:8113/openapi.json

也可以用 uvicorn 命令行（在 `book_api` 目录下执行）：

```powershell
uvicorn main:app --reload --port 8113
```

### 2. 快速自检（不启动服务、不阻塞）

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Back_End\10_综合实战\book_api\main.py' --check
```

会完整跑一遍 CRUD：新增 → 重复 ISBN(409) → 参数非法(422) → 详情 → 分页 → 搜索 →
排序 → PUT → PATCH → 删除 → 404 → 文档 → 中间件。

### 3. 跑测试

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m pytest 'F:\ProGram\Python_Base\Back_End\10_综合实战\book_api' -q
```

---

## 四、接口清单

| 方法 | 路径 | 说明 | 成功状态码 |
| --- | --- | --- | --- |
| GET | `/` | 服务信息与接口清单 | 200 |
| GET | `/health` | 健康检查（含数据库可用性） | 200 |
| GET | `/api/books` | 列表：`?page=&size=&keyword=&order_by=&desc=` | 200 |
| POST | `/api/books` | 新增图书 | 201 |
| GET | `/api/books/{id}` | 详情 | 200 / 404 |
| PUT | `/api/books/{id}` | 全量更新 | 200 / 404 / 409 |
| PATCH | `/api/books/{id}` | 局部更新 | 200 / 400 / 404 / 409 |
| DELETE | `/api/books/{id}` | 删除 | 200 / 404 |

**统一响应格式：**

```json
{ "code": 0, "message": "ok", "data": { "...": "..." } }
```

**统一错误格式：**

```json
{
  "code": 404,
  "message": "图书 1 不存在",
  "detail": { "path": "/api/books/1", "method": "GET" },
  "request_id": "2b5a0e346ff2"
}
```

`request_id` 与响应头 `X-Request-Id` 一致：用户报障时报这个号，运维直接在日志里定位。

---

## 五、用 curl / PowerShell 试接口

```bash
# 新增
curl -X POST http://127.0.0.1:8113/api/books \
  -H "Content-Type: application/json" \
  -d '{"title":"流畅的 Python","author":"Luciano Ramalho","isbn":"978-7-115-45415-7","price":139.0,"stock":10}'

# 列表（分页 + 搜索 + 排序）
curl "http://127.0.0.1:8113/api/books?page=1&size=5&keyword=python&order_by=price&desc=true"

# 详情
curl http://127.0.0.1:8113/api/books/1

# 局部更新（只改库存）
curl -X PATCH http://127.0.0.1:8113/api/books/1 \
  -H "Content-Type: application/json" -d '{"stock":3}'

# 删除
curl -X DELETE http://127.0.0.1:8113/api/books/1
```

---

## 六、常见问题

**Q：数据库文件在哪里？想重置数据怎么办？**
A：`Back_End/data/book_api.db`。删除该文件后重启服务即可重新建表。

**Q：怎么换成 MySQL？**
A：启动前设置环境变量（不要改代码）：
```powershell
$env:BOOK_API_DATABASE_URL = "mysql+pymysql://user:password@127.0.0.1:3306/book_api?charset=utf8mb4"
```
然后重启服务。注意目标库必须已存在（`create_all` 只建表不建库），
且本机 MySQL 未启动时连接会失败 —— 默认的 SQLite 不会受此影响。

**Q：表结构改了怎么办？**
A：本项目用 `Base.metadata.create_all`（只创建缺失的表，不改已存在的表）。
生产环境应使用迁移工具（Alembic）；本环境未安装 alembic，所以这里只做演示。

**Q：金额为什么用 `Numeric(10, 2)` 而不是 `Float`？**
A：浮点数无法精确表示小数（`0.1 + 0.2 != 0.3`）。金额必须用精确小数类型，
SQLite 里会以 NUMERIC 亲和类型存储，MySQL 里对应 `DECIMAL(10,2)`。

**Q：为什么路由是 `/api/books` 而不是 `/books`？**
A：统一加 `/api` 前缀便于将来与静态页面、`/admin` 后台区分，
也方便 Nginx 按前缀做路由分流（`location /api/ { proxy_pass ... }`）。
