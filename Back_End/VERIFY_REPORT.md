# Back_End 验证报告（VERIFY_REPORT.md）

> 本文件是**真实执行**的记录，由 `Back_End/verify_all.py` 的完整输出 + pytest 输出 + 越界检查组成。
> 生成时间：2026-09-11 16:21:18
> 解释器：`F:\ProGram\Python_Base\.venv\Scripts\python.exe`（Python 3.12.12）
> 工作区：`F:\ProGram\Python_Base`　｜　交付目录：`F:\ProGram\Python_Base\Back_End`

---

## 一、实际执行的命令

全部命令都在 PowerShell 中执行，工作目录为 `F:\ProGram\Python_Base`；
路径含中文，因此统一加引号并用 `&` 调用；**未使用 `uv run`**（避免触发 sync 改动环境）。

```powershell
# ① 一键运行 Back_End 下全部脚本（FastAPI 示例走 --check 非阻塞模式）+ pytest + 接口校验
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Back_End\verify_all.py'

# ② 只跑测试框架目录
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m pytest 'F:\ProGram\Python_Base\Back_End\07_测试框架' -q

# ③ 只跑综合实战项目的测试
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m pytest 'F:\ProGram\Python_Base\Back_End\10_综合实战\book_api' -q

# ④ 跑整个 Back_End 的测试
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m pytest 'F:\ProGram\Python_Base\Back_End' -q

# ⑤ FastAPI 全量接口校验（被 ① 自动调用，也可单独执行）
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Back_End\06_FastAPI\verify_api.py'

# ⑥ 越界检查：确认只改动了 Back_End 目录
git status --short
```

---

## 二、`verify_all.py` 完整输出

```text
##############################################################################
# Back_End 全量脚本自检（verify_all.py）
##############################################################################
解释器      ：F:\ProGram\Python_Base\.venv\Scripts\python.exe
Python 版本 ：3.12.12
扫描目录    ：F:\ProGram\Python_Base\Back_End
工作目录    ：F:\ProGram\Python_Base\Back_End（所有子脚本都在这个「中立」目录下运行，用于验证它们不依赖当前工作目录）
单脚本超时  ：180 秒

发现 31 个可运行脚本：
  · 03_配置文件/01_pydantic_settings配置.py                    [直接运行]
  · 04_日志/01_loguru基础.py                                 [直接运行]
  · 04_日志/02_loguru文件与轮转.py                              [直接运行]
  · 04_日志/03_loguru异常捕获.py                               [直接运行]
  · 05_Flask/01_最小应用.py                                  [直接运行]
  · 05_Flask/02_路由与请求.py                                 [直接运行]
  · 05_Flask/03_模板与静态文件.py                               [直接运行]
  · 05_Flask/04_RESTful接口.py                             [直接运行]
  · 06_FastAPI/01_最小应用.py                                [FastAPI --check]
  · 06_FastAPI/02_GET与POST.py                            [FastAPI --check]
  · 06_FastAPI/03_数据模型与验证.py                             [FastAPI --check]
  · 06_FastAPI/04_模板与静态文件.py                             [FastAPI --check]
  · 06_FastAPI/05_异步处理.py                                [FastAPI --check]
  · 06_FastAPI/06_数据库集成.py                               [FastAPI --check]
  · 06_FastAPI/07_依赖注入.py                                [FastAPI --check]
  · 06_FastAPI/08_认证与授权.py                               [FastAPI --check]
  · 06_FastAPI/09_中间件.py                                 [FastAPI --check]
  · 06_FastAPI/10_错误处理.py                                [FastAPI --check]
  · 06_FastAPI/11_Session与Cookie.py                      [FastAPI --check]
  · 06_FastAPI/12_Celery后台任务.py                          [FastAPI --check]
  · 06_FastAPI/verify_api.py                             [直接运行]
  · 07_测试框架/calculator.py                                [直接运行]
  · 07_测试框架/test_calculator.py                           [pytest]
  · 09_容器部署/检查YAML.py                                    [直接运行]
  · 10_综合实战/book_api/crud.py                             [FastAPI --check]
  · 10_综合实战/book_api/database.py                         [FastAPI --check]
  · 10_综合实战/book_api/main.py                             [FastAPI --check]
  · 10_综合实战/book_api/models.py                           [FastAPI --check]
  · 10_综合实战/book_api/routers/books.py                    [FastAPI --check]
  · 10_综合实战/book_api/schemas.py                          [FastAPI --check]
  · 10_综合实战/book_api/test_book_api.py                    [pytest]

==============================================================================
[1/31] 03_配置文件/01_pydantic_settings配置.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\03_配置文件\01_pydantic_settings配置.py
  返回码  ：0   成功 ✓   耗时 0.31 秒
  ---- 标准输出最后 15 行 ----
  |   settings.milvus_uri     = http://127.0.0.1:19530
  |   settings.openai_api_key = sk-x******qw4y   <- 脱敏打印
  |   settings.llm.model      = deepseek-ai/DeepSeek-V4-Flash   <- 分组配置（RAG 项目用）
  | 
  | 注意：根配置文件在这套机制上和本示例完全一样，
  |       只是它同时提供了'扁平字段'和'分组字段'两种访问方式。
  | 
  | ########################################################################
  | # 全部演示完成：配置管理要点回顾
  | ########################################################################
  | 1. 配置与代码分离：改配置不改代码，不用重新打包镜像
  | 2. 类型即校验：字段注解 int/bool/SecretStr 自动完成转换与保护
  | 3. 优先级明确：环境变量 > .env 文件 > 默认值
  | 4. 敏感信息用 SecretStr，且 .env 永不入库
  | 5. 多环境用不同的 env_file 或环境变量区分，绝不在代码里 if 判断硬编码

==============================================================================
[2/31] 04_日志/01_loguru基础.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\04_日志\01_loguru基础.py
  返回码  ：0   成功 ✓   耗时 0.17 秒
  ---- 标准输出最后 15 行 ----
  |   2026-09-11 16:20:14.895 | DEBUG    | __main__:demo_levels:79 | 【DEBUG】开发调试用，例如「查询到 3 条记录」，生产环境通常不开
  |   2026-09-11 16:20:14.895 | INFO     | __main__:demo_levels:80 | 【INFO】正常业务流程，例如「用户 alice 登录成功」
  |   2026-09-11 16:20:14.896 | SUCCESS  | __main__:demo_levels:81 | 【SUCCESS】明确成功的关键操作，例如「订单 1001 支付完成」
  |   2026-09-11 16:20:14.896 | WARNING  | __main__:demo_levels:82 | 【WARNING】不影响运行但要关注，例如「接口耗时 2.3s 偏慢」
  |   2026-09-11 16:20:14.896 | ERROR    | __main__:demo_levels:83 | 【ERROR】明确出错但服务没挂，例如「第三方支付回调验签失败」
  |   （文件共 49 行，路径：F:\ProGram\Python_Base\Back_End\data\logs\01_loguru基础.log）
  | 
  | ########################################################################
  | # 要点回顾
  | ########################################################################
  | 1. logger.remove() 之后重新 add，是 loguru 的标准起手式
  | 2. 不同 sink 用不同 level：终端少而精，文件全而细
  | 3. format 决定日志的信息量，{name}/{function}/{line} 是排障刚需
  | 4. logger.bind(...) 给日志加业务字段，是链路追踪的起点
  | 5. 自定义 sink 可以把 ERROR 日志送到告警通道
  ---- 标准错误最后 8 行 ----
  ! 16:20:14 | WARNING  | 【WARNING】不影响运行但要关注，例如「接口耗时 2.3s 偏慢」
  ! 16:20:14 | ERROR    | 【ERROR】明确出错但服务没挂，例如「第三方支付回调验签失败」
  ! 16:20:14 | CRITICAL | 【CRITICAL】严重故障，例如「数据库连接池耗尽，服务不可用」
  !   时间=2026-09-11 16:20:14 级别=INFO 模块=__main__ 函数=demo_format_fields 行号=112 消息=这一行展示了日志里可以带上的全部定位信息
  !   [req-8f3a91] user=10086 path=/api/books/1 -> 开始处理请求
  !   [req-8f3a91] user=10086 path=/api/books/1 -> 命中缓存，耗时 3ms
  ! 16:20:14 | WARNING  | 这条只是警告，不会触发告警
  ! 16:20:14 | ERROR    | 这条会触发告警：数据库查询超时

==============================================================================
[3/31] 04_日志/02_loguru文件与轮转.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\04_日志\02_loguru文件与轮转.py
  返回码  ：0   成功 ✓   耗时 0.23 秒
  ---- 标准输出最后 15 行 ----
  |     backtrace=True,                  # 记录调用栈
  |     diagnose=False,                  # 关闭变量值，防止敏感信息落盘
  |     format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
  | )
  | 
  | ########################################################################
  | # 要点回顾
  | ########################################################################
  | 1. rotation 决定什么时候换文件，retention 决定留多久，compression 决定怎么省空间
  | 2. 三者必须一起配，否则不是写满磁盘就是丢了排障线索
  | 3. serialize=True 让日志变成结构化数据，是接入日志平台的前提
  | 4. diagnose 生产必须关：它会记录每层栈帧的局部变量，可能把密码写进日志
  | 5. 进程退出前 logger.remove() 可以把 enqueue 队列刷干净
  | 
  | （本脚本总耗时 0.069 秒）
  ---- 标准错误最后 2 行 ----
  ! 16:20:15 | WARNING  | WARNING 也进主日志
  ! 16:20:15 | ERROR    | ERROR 会额外写进 only_error.log，运维只需要盯这一个文件

==============================================================================
[4/31] 04_日志/03_loguru异常捕获.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\04_日志\03_loguru异常捕获.py
  返回码  ：0   成功 ✓   耗时 0.16 秒
  ---- 标准输出最后 15 行 ----
  |   含 ValueError         的行数：8
  |   含 ConnectionError    的行数：4
  | 
  | 结论：日志文件里有完整可追溯的异常链，而屏幕上没有任何堆栈刷屏 ——
  |       这就是「终端干净 + 文件完整」的两个 handler 分工。
  | 
  | ########################################################################
  | # 要点回顾
  | ########################################################################
  | 1. logger.exception() 只在 except 块里有意义，它自带当前异常信息
  | 2. loguru 会自动给「带异常」的记录追加堆栈 —— 用 filter 把它挡在终端之外，
  |    再用一个自定义 sink 输出一行摘要，才能做到「终端干净 + 文件完整」
  | 3. backtrace=True 展开被吞掉的调用链；diagnose 生产必须 False
  | 4. @logger.catch 是函数级兜底，reraise 决定要不要继续往上抛
  | 5. 耗时装饰器要配 functools.wraps，并用 finally 保证异常时也记时间
  ---- 标准错误最后 10 行 ----
  ! 16:20:15 | ERROR    | 计算失败：除数为零 | 异常类型=ZeroDivisionError | 位置=__main__:demo_logger_exception:161
  !           └─ 完整堆栈已写入 03_异常记录.log（不在此刷屏）
  ! 16:20:15 | ERROR    | 字段缺失：'price'，当前记录内容={'name': '苹果'} | 异常类型=KeyError | 位置=__main__:demo_opt_exception:176
  !           └─ 完整堆栈已写入 03_异常记录.log（不在此刷屏）
  ! 16:20:15 | ERROR    | 处理订单时发生未预期异常 | 异常类型=ValueError | 位置=__main__:demo_logger_catch:197
  !           └─ 完整堆栈已写入 03_异常记录.log（不在此刷屏）
  ! 16:20:15 | ERROR    | An error has been caught in function 'demo_logger_catch', process 'MainProcess' (39672), thread 'MainThread' (53092): | 异常类型=ConnectionError | 位置=__main__:demo_logger_catch:205
  !           └─ 完整堆栈已写入 03_异常记录.log（不在此刷屏）
  ! 16:20:15 | ERROR    | broken_calculation 调用失败 | 异常类型=ValueError | 位置=__main__:demo_cost_time:227
  !           └─ 完整堆栈已写入 03_异常记录.log（不在此刷屏）

==============================================================================
[5/31] 05_Flask/01_最小应用.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\05_Flask\01_最小应用.py
  返回码  ：0   成功 ✓   耗时 0.24 秒
  ---- 标准输出最后 15 行 ----
  | 
  | [405 演示]
  |   请求：DELETE /
  |   状态码：405（期望 405，因为 / 只注册了 GET）
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. Flask(__name__) 决定模板与静态文件的位置
  | 2. @app.route 把 URL 绑定到视图函数，函数返回值就是 HTTP 响应
  | 3. 返回 dict / jsonify 得到 JSON 响应，返回 str 得到 HTML 响应
  | 4. 404 = 路径不存在；405 = 路径存在但方法不允许（框架自动处理）
  | 5. app.test_client() 是写自动化测试的正确姿势，正式测试见 07_测试框架
  | 6. 想真正启服务：给本脚本加 --serve 参数

==============================================================================
[6/31] 05_Flask/02_路由与请求.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\05_Flask\02_路由与请求.py
  返回码  ：0   成功 ✓   耗时 0.24 秒
  ---- 标准输出最后 15 行 ----
  |     Set-Cookie = demo_cookie=cookie-value; Expires=Fri, 11 Sep 2026 09:20:15 GMT; Max-Age=3600; H…
  | 
  | [自定义 404 处理器]
  |     状态码 404，响应体 {"error":"资源不存在","path":"/who-am-i","提示":"请检查 URL 是否正确"}
  | 
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. <int:id> 这类转换器在“路由匹配阶段”就完成校验，不合法直接 404
  | 2. request.args / form / json / headers / cookies 是取参数的五个入口
  | 3. 响应的四种写法：字符串、元组、make_response、Response
  | 4. errorhandler 可以把框架默认的 HTML 错误页换成统一 JSON 格式
  | 5. url_for 反向生成 URL，避免在代码里硬编码路径

==============================================================================
[7/31] 05_Flask/03_模板与静态文件.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\05_Flask\03_模板与静态文件.py
  返回码  ：0   成功 ✓   耗时 0.34 秒
  ---- 标准输出最后 15 行 ----
  |     状态码 200（期望 200）
  | 
  | [8] GET /items/99  数据不存在时 abort(404)
  |     状态码 404（期望 404）
  |     页面提示：<h1 style='font-family:sans-serif'>404 · 商品 99 不存在</h1><p st…
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. render_template('x.html', 变量=值) 把数据填充进模板
  | 2. {% extends %} + {% block %} 实现模板继承，公共结构只写一次
  | 3. Jinja2 默认开启自动转义，这是防 XSS 的第一道防线
  | 4. static/ 由 Web 服务器直接返回，不消耗 Python 视图函数的处理时间
  | 5. 页面型接口用模板、数据型接口用 JSON，两者按场景选择

==============================================================================
[8/31] 05_Flask/04_RESTful接口.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\05_Flask\04_RESTful接口.py
  返回码  ：0   成功 ✓   耗时 0.23 秒
  ---- 标准输出最后 15 行 ----
  |     GET /api/nothing
  |     状态码 404（期望 404）
  |     响应体 {"code":404,"data":null,"message":"接口不存在：/api/nothing"}
  | 
  | 
  | ========================================================================
  | 自检全部通过 ✓（共 18 个请求）
  | ========================================================================
  | 知识点回顾：
  | 1. RESTful = URL 定位资源 + HTTP 方法表达动作 + 状态码表达结果
  | 2. Blueprint 把路由按业务分文件，url_prefix 统一管理版本前缀
  | 3. 请求体必须校验并「清洗」，防止客户端注入额外字段
  | 4. PUT 全量替换、PATCH 局部修改，语义不要混用
  | 5. 统一响应格式 + 全局错误处理，让前端只写一次拦截器
  | 6. 内存存储仅用于演示；真实项目必须落库（见 06_FastAPI/06_数据库集成.py）

==============================================================================
[9/31] 06_FastAPI/01_最小应用.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\01_最小应用.py --check
  返回码  ：0   成功 ✓   耗时 0.49 秒
  ---- 标准输出最后 15 行 ----
  | [OpenAPI 自动生成的接口清单]
  |     GET      /
  |     GET      /health
  |     GET      /items
  |     GET      /items/{item_id}
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. 类型注解 = 校验规则 + 文档 + 编辑器补全，一份代码三份收益
  | 2. 参数不合法返回 422，响应体里有 loc/type/msg，前端能精确提示
  | 3. /docs、/redoc、/openapi.json 全部由类型注解自动生成，无需手写文档
  | 4. 模块导入时不监听端口，所以可以被测试与校验脚本安全导入
  | 5. 启动服务：python 本文件（默认端口 8101）

==============================================================================
[10/31] 06_FastAPI/02_GET与POST.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\02_GET与POST.py --check
  返回码  ：0   成功 ✓   耗时 0.47 秒
  ---- 标准输出最后 15 行 ----
  |     响应 {"User-Agent":"MyClient/1.0","X-Token":"abc-123","session_id":"s-999"}
  |     ✓ 请求头 User-Agent / X-Token 与 Cookie session_id 都正确注入到函数参数
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. 参数位置由声明决定：路径 / 查询 / 请求体 / 表单 / 请求头 / Cookie
  | 2. BaseModel 参数 = 请求体；Form() = 表单；Header()/Cookie() = 隐式参数
  | 3. status_code=201 表达「已创建」，response_model 负责过滤输出字段
  | 4. GET 只读且幂等，POST 用于创建与提交；语义错了将来很难改
  | 5. 用 requests/httpx 调用真实服务的示例：
  |    import requests
  |    requests.get('http://127.0.0.1:8102/items/5', params={'page-size': 50})
  |    requests.post('http://127.0.0.1:8102/items', json={'name':'苹果','price':3.99})

==============================================================================
[11/31] 06_FastAPI/03_数据模型与验证.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\03_数据模型与验证.py --check
  返回码  ：0   成功 ✓   耗时 0.48 秒
  ---- 标准输出最后 15 行 ----
  | [自动生成的 JSON Schema 摘要]
  |     模型标题：User
  |     字段数量：11
  |     必填字段：['userName', 'email', 'salary', 'phone']
  |     字段清单：['address', 'age', 'birthday', 'email', 'hobbies', 'internal_note', 'phone', 'role', 'salary', 'userName', 'website']
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. Field 是「校验 + 文档 + 默认值」三合一，能用声明就别写 if 判断
  | 2. v1 → v2 的坑：regex 改名 pattern；列表的 min_items/max_items 改名 min_length/max_length
  | 3. field_validator 管单个字段，model_validator(mode='after') 管字段之间的关系
  | 4. ConfigDict：populate_by_name / str_strip_whitespace / extra='forbid' 是生产常用三件套
  | 5. 请求模型负责「收得严」，响应模型负责「吐得少」，两者不要混用一个类

==============================================================================
[12/31] 06_FastAPI/04_模板与静态文件.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\04_模板与静态文件.py --check
  返回码  ：0   成功 ✓   耗时 0.61 秒
  ---- 标准输出最后 15 行 ----
  |     状态码 404（期望 404）
  | 
  | [7] GET /api/products  同一个应用里的 JSON 接口
  |     状态码 200（期望 200），Content-Type=application/json
  |     响应：{"total":3,"items":[{"name":"机械键盘","price":399.0,"stock":26},{"name":"人体工学椅","price":1299.0,"stock":3},{"name":"显示器支架","…
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. Jinja2Templates(directory=...) + TemplateResponse 完成服务端渲染
  | 2. StaticFiles 挂载后，静态资源由 Starlette 直接返回，不消耗业务逻辑时间
  | 3. response_class=HTMLResponse 让文档正确标注返回类型
  | 4. 同一应用里 HTML 页面与 JSON 接口完全可以共存，用路径区分
  | 5. 模板目录/静态目录一定要用绝对路径，避免工作目录变化导致 500

==============================================================================
[13/31] 06_FastAPI/05_异步处理.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\05_异步处理.py --check
  返回码  ：0   成功 ✓   耗时 3.24 秒
  ---- 标准输出最后 15 行 ----
  |     后台任务执行记录（共 2 条）：
  |       16:20:21 创建订单：机械键盘
  |       16:20:21 通知已发送：订单 机械键盘 已创建
  | 
  | [异步依赖验证]
  |     GET /with-dep  状态码 200，响应 {"request_id":"req-21601","说明":"依赖函数是 async def，FastAPI 会 await 它"}
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. async def 由事件循环执行；def 由线程池执行，两者都并发，机制不同
  | 2. async def 里写 time.sleep / requests / 同步驱动 = 卡死整个服务
  | 3. 必须用同步库时，用 await run_in_threadpool(阻塞函数, ...) 兜底
  | 4. 异步依赖（async def + Depends）不会阻塞事件循环
  | 5. BackgroundTasks 在响应之后执行轻量任务；重任务交给 Celery

==============================================================================
[14/31] 06_FastAPI/06_数据库集成.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\06_数据库集成.py --check
  返回码  ：0   成功 ✓   耗时 2.96 秒
  ---- 标准输出最后 15 行 ----
  |     目标：MySQL
  |     连接串：mysql+pymysql://***:***@127.0.0.1:3306/***?charset=utf8mb4
  |            （用户名 / 口令 / 库名已打码：数据库信息不进入日志与报告）
  |     可用：False
  |     说明：MySQL 未连接（OperationalError）。本机没有启动 MySQL 服务属于正常情况，示例已自动降级为 SQLite。需要真实连接时：启动 MySQL → 建库 → 在根目录 .env 里配置 MYSQL_* → 重启本脚本。
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. 2.0 风格：DeclarativeBase + Mapped + mapped_column + select()
  | 2. 写入三步：add → commit → refresh；查询用 db.execute(stmt).scalars()
  | 3. 依赖注入 + yield 是管理 Session 的标准姿势：解耦、复用、自动清理
  | 4. SQLite 要加 check_same_thread=False，因为同步端点跑在线程池里
  | 5. 外部服务（MySQL）不可用时要降级而不是崩溃，日志里给出明确中文提示
  | 6. 异步 ORM 把 Session 换成 AsyncSession、把语句用 await 执行即可，写法一致

==============================================================================
[15/31] 06_FastAPI/07_依赖注入.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\07_依赖注入.py --check
  返回码  ：0   成功 ✓   耗时 0.49 秒
  ---- 标准输出最后 15 行 ----
  |     GET /need-db
  |     状态码 200（期望 200）
  |     响应 {"dsn":"mysql+pymysql://real-db:3306/prod","说明":"返回值来自 real_dsn 依赖，可被测试覆盖"}
  |     ✓ 覆盖已清理，恢复真实依赖
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. Depends 解决的是「重复创建前置资源」的问题，核心收益是解耦 + 复用 + 可测试
  | 2. yield 依赖负责资源生命周期：Setup 在前、Teardown 在后，用 finally 保证释放
  | 3. 类依赖适合封装一组参数；子依赖可以搭出依赖树
  | 4. 同一请求内默认缓存；use_cache=False 可关闭
  | 5. APIRouter(dependencies=[...]) 做路由组统一鉴权/审计
  | 6. dependency_overrides 是写测试的利器，能彻底替换掉真数据库

==============================================================================
[16/31] 06_FastAPI/08_认证与授权.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\08_认证与授权.py --check
  返回码  ：0   成功 ✓   耗时 0.91 秒
  ---- 标准输出最后 15 行 ----
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. 认证=你是谁（401），授权=你能做什么（403），两者分开实现
  | 2. 密码用 PBKDF2 + 随机盐哈希存储，比对用 hmac.compare_digest 防时序攻击
  | 3. JWT = Base64(header).Base64(payload).签名；payload 可读不可改
  | 4. OAuth2PasswordBearer 只提取 token，验证必须自己做（jwt.decode）
  | 5. 用依赖注入串联：get_current_user（认证）→ require_role（授权）
  | 6. 查询数据时，用户身份只从 token 取，绝不信前端传的参数
  | 7. 前端登录示例（OAuth2 密码模式要求表单格式）：
  |    fetch('/token', {method:'POST',
  |      headers:{'Content-Type':'application/x-www-form-urlencoded'},
  |      body:new URLSearchParams({username:'alice', password:'alice123'})})
  ---- 标准错误最后 2 行 ----
  ! F:\ProGram\Python_Base\.venv\Lib\site-packages\jwt\api_jwt.py:147: InsecureKeyLengthWarning: The HMAC key is 14 bytes long, which is below the minimum recommended length of 32 bytes for SHA256. See RFC 7518 Section 3.2.
  !   return self._jws.encode(

==============================================================================
[17/31] 06_FastAPI/09_中间件.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\09_中间件.py --check
  返回码  ：0   成功 ✓   耗时 0.46 秒
  ---- 标准输出最后 15 行 ----
  |     3) CORS（要在鉴权之前，保证 401/403 响应也带跨域头）
  |     4) 会话 / 认证
  |     5) GZip（最靠近路由，只压缩业务响应）
  |     6) 自定义日志 / 计时中间件
  |     —— 生产上压缩常交给 Nginx，减少应用 CPU 开销
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. 中间件 = 请求/响应的公共处理层，顺序是「洋葱模型」，后添加的在最外层
  | 2. CORS 是浏览器的同源策略要求，服务端用响应头放行；curl/requests 不受影响
  | 3. GZip 只压缩大于 minimum_size 的响应，生产上常由 Nginx 承担
  | 4. 自定义中间件适合做耗时统计、请求 ID、访问日志、限流
  | 5. 中间件里要用 try/finally 保证异常路径也能收尾

==============================================================================
[18/31] 06_FastAPI/10_错误处理.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\10_错误处理.py --check
  返回码  ：0   成功 ✓   耗时 0.46 秒
  ---- 标准输出最后 15 行 ----
  |     422: Unprocessable Entity —— 语法正确但语义校验失败
  |     429: Too Many Requests —— 触发限流
  |     500: Internal Server Error —— 服务器内部错误
  |     503: Service Unavailable —— 服务暂时不可用（如依赖故障）
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. 统一错误格式 = 前端只写一次拦截器；request_id 是排障的钥匙
  | 2. 业务层抛自定义异常，表现层用 exception_handler 映射成 HTTP 状态码
  | 3. 用 RequestValidationError 处理器把 422 整理成前端友好的结构
  | 4. 用 StarletteHTTPException 处理器接管框架自带的 404/405
  | 5. 兜底 Exception 处理器绝不返回堆栈，细节只进服务端日志
  | 6. 能明确用 400/404/409 就别用 500 —— 500 代表「我们出 bug 了」

==============================================================================
[19/31] 06_FastAPI/11_Session与Cookie.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\11_Session与Cookie.py --check
  返回码  ：0   成功 ✓   耗时 0.49 秒
  ---- 标准输出最后 15 行 ----
  |     水平扩展      天然支持，无需共享存储                   需要共享存储（Redis）或粘性会话
  |     主动撤销      困难（要靠黑名单/短过期）                 容易（删掉会话即可）
  |     跨域使用      方便（放 Authorization 头）         受同源策略限制
  |     典型场景      前后端分离、微服务、开放 API              传统 Web、后台管理系统
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. HTTP 无状态 → 需要 Cookie/Session 或 JWT 来「记住」用户
  | 2. Cookie 必设 HttpOnly（防 XSS 窃取）、SameSite（防 CSRF）、Secure（HTTPS）
  | 3. Starlette 的 SessionMiddleware 是签名 Cookie 方案，内容可读不可改
  | 4. 要「服务器只存 session id」就得配合 Redis 等服务端存储
  | 5. itsdangerous 的 URLSafeSerializer 可以给任意 Cookie 加签名校验
  | 6. 登录时轮换 session（clear 后重写）可防会话固定攻击

==============================================================================
[20/31] 06_FastAPI/12_Celery后台任务.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\12_Celery后台任务.py --check
  返回码  ：0   成功 ✓   耗时 2.65 秒
  ---- 标准输出最后 15 行 ----
  |     celery -A 12_Celery后台任务.celery_app worker --loglevel=info --pool=threads
  |     # 一份代码部署到 N 台机器，每台执行同一个 worker 命令，
  |     # 它们会自动从 Redis 竞争任务执行 —— 这就是「水平扩展」
  |     # 3) 定时任务再加一个 beat：celery -A ... beat --loglevel=info
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 知识点回顾：
  | 1. Celery 三件套：broker（队列）+ backend（结果）+ worker（执行者）
  | 2. delay() 是 apply_async() 的简写；复杂参数（队列/延迟/过期）用后者
  | 3. 任务必须可 JSON 序列化、必须幂等（worker 可能重试）
  | 4. chain 串行、group 并行、chord 汇总，是任务编排的三块积木
  | 5. 本机无 Redis 时用 task_always_eager=True，代码写法不变、绝不阻塞
  | 6. 重任务不要塞进 Web 进程同步跑；依赖故障要有明确的降级策略

==============================================================================
[21/31] 06_FastAPI/verify_api.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\06_FastAPI\verify_api.py
  返回码  ：0   成功 ✓   耗时 3.17 秒
  ---- 标准输出最后 15 行 ----
  |         ✓ GET     /docs                                          -> 200（期望 200）
  |             响应 <!DOCTYPE html>     <html>     <head>     <meta name="viewport" content="width=device-width, initial-scale=1.0">     <link type="text/css" rel="styles…
  |         ✓ GET     /openapi.json                                  -> 200（期望 200）
  |             响应 {"openapi":"3.1.0","info":{"title":"图书管理 API（Back_End 综合实战）","description":"后端开发基础课案的综合实战项目：SQLite + SQLAlchemy 2.0 + Pydantic + APIRouter 分层 + 依赖注入 +…
  |         ✓ OpenAPI 文档生成成功，共 4 个路径
  | [book_api] 连接池已释放，应用退出
  | 
  | ==============================================================================
  | 接口校验汇总
  | ==============================================================================
  | 参与校验的应用：13 / 13
  | 实际发出的请求：96
  | 失败项：0
  | 
  | 结论：所有示例接口校验通过 ✓（每个示例的详细讲解见对应脚本的 --check 输出）

==============================================================================
[22/31] 07_测试框架/calculator.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\07_测试框架\calculator.py
  返回码  ：0   成功 ✓   耗时 0.03 秒
  ---- 标准输出最后 15 行 ----
  |   avg([1,2,3]) = 2.0
  |   is_prime(97) = True
  |   fizzbuzz(15) = FizzBuzz
  | 
  | 运算历史：
  |     3 + 4 = 7
  |     10 - 3.5 = 6.5
  |     2.5 * 4 = 10.0
  |     10 / 4 = 2.5
  |     2 ** 10 = 1024
  |     sqrt(144) = 12.0
  |     avg([1, 2, 3]) = 2.0
  | 
  | 冒烟测试完成。运行正式测试：
  |   & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m pytest 'F:\ProGram\Python_Base\Back_End\07_测试框架' -q

==============================================================================
[23/31] 07_测试框架/test_calculator.py    （模式：pytest）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe -m pytest F:\ProGram\Python_Base\Back_End\07_测试框架\test_calculator.py -q --no-header
  返回码  ：0   成功 ✓   耗时 1.23 秒
  ---- 标准输出最后 5 行 ----
  | .................................................s.x.........            [100%]
  | =========================== short test summary info ===========================
  | SKIPPED [1] 07_测试框架\test_calculator.py:249: 演示用：这个用例故意被跳过，不会计入失败
  | XFAIL 07_测试框架/test_calculator.py::test_xfail_demo - 演示用：这是一个已知会失败的用例，xfail 表示'预期失败'
  | 59 passed, 1 skipped, 1 xfailed in 0.40s

==============================================================================
[24/31] 09_容器部署/检查YAML.py    （模式：直接运行）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\09_容器部署\检查YAML.py
  返回码  ：0   成功 ✓   耗时 0.08 秒
  ---- 标准输出最后 15 行 ----
  |     · K8S 的 YAML 可以一个文件多个文档（用 --- 分隔），kubectl apply 会依次创建
  | 
  | ==============================================================================
  | 校验汇总
  | ==============================================================================
  |   ✓ 5 个 YAML 文件语法全部正确
  |   ✓ docker-compose.yml：服务/端口/卷/网络/depends_on/build 引用一致
  |   ✓ K8S：Deployment / Service / ConfigMap / Secret / Ingress 字段齐备，跨文件引用一致
  |   ✓ Dockerfile：多阶段构建、非 root、健康检查、0.0.0.0 监听等 11 项检查通过
  |   ✓ .dockerignore：排除了 .venv / __pycache__ / .env / data
  | 
  |   说明：本机未安装 Docker / kubectl，因此这里做的是**静态校验**。
  |         要真正部署，请按 09_容器部署/k8s/README.md 里的流程操作。
  | 
  | 全部检查通过 ✓

==============================================================================
[25/31] 10_综合实战/book_api/crud.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\10_综合实战\book_api\crud.py --check
  返回码  ：0   成功 ✓   耗时 0.62 秒
  ---- 标准输出最后 15 行 ----
  |   【4】patch_book（局部更新）
  |       stock 改为 99，title 保持 'crud 自检用书'
  | 
  |   【5】update_book（全量更新）
  |       更新后：'crud 自检用书（第2版）' 价格 69.90 库存 3
  | 
  |   【6】delete_book（并清理测试数据）
  |       删除前 25 条 → 删除后 24 条
  | 
  |   为什么要单独测这一层？
  |     · 不依赖 HTTP 与网络，跑得飞快；
  |     · 能覆盖路由层难测的分支（排序白名单、空结果、边界值）；
  |     · 换数据库/换 ORM 时，只测这一层就够了。
  | 
  | crud.py 自检完成 ✓

==============================================================================
[26/31] 10_综合实战/book_api/database.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\10_综合实战\book_api\database.py --check
  返回码  ：0   成功 ✓   耗时 0.48 秒
  ---- 标准输出最后 15 行 ----
  |   当前数据库中已有的表：['books']
  |     · books：id:INTEGER, title:VARCHAR(120), author:VARCHAR(60), isbn:VARCHAR(20), price:NUMERIC(10, 2), stock:INTEGER, created_at:DATETIME
  | 
  |   验证会话工厂与 get_db 依赖 ...
  |   SessionLocal() 可用，books 当前 24 行
  | 
  |   get_db() 是生成器函数（含 yield），必须由 FastAPI 的 Depends 驱动：
  |       def get_db():
  |           db = SessionLocal()
  |           try:
  |               yield db        # 请求处理期间使用
  |           finally:
  |               db.close()      # 请求结束后自动关闭，绝不泄漏连接
  | 
  | database.py 自检完成 ✓

==============================================================================
[27/31] 10_综合实战/book_api/main.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\10_综合实战\book_api\main.py --check
  返回码  ：0   成功 ✓   耗时 0.93 秒
  ---- 标准输出最后 15 行 ----
  |     GET /openapi.json -> 200 (8.40ms)
  |     GET /health -> 200 (0.68ms)
  | [book_api] 连接池已释放，应用退出
  | 
  | ========================================================================
  | 自检全部通过 ✓
  | ========================================================================
  | 项目结构回顾：
  |     main.py       应用入口：中间件 + 异常处理 + 路由装配 + 生命周期
  |     database.py   引擎 / 会话工厂 / get_db 依赖 / 建表
  |     models.py     ORM 模型（SQLAlchemy 2.0 风格）
  |     schemas.py    Pydantic 请求与响应模型（入参严格、出参精简）
  |     crud.py       数据访问层（纯数据库操作，不认识 HTTP）
  |     routers/books.py  路由层（只解析请求、调用 crud、包装响应）
  | 分层的好处：换库只改 database+crud；测业务不用起 HTTP；多人协作冲突少。

==============================================================================
[28/31] 10_综合实战/book_api/models.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\10_综合实战\book_api\models.py --check
  返回码  ：0   成功 ✓   耗时 0.48 秒
  ---- 标准输出最后 15 行 ----
  |     stock       INTEGER         非空
  |     created_at  DATETIME        非空、数据库端默认值
  | 
  |   建模要点回顾：
  |     1. Mapped[int] + mapped_column(...) 是 2.0 的推荐写法（类型即列定义）
  |     2. 金额用 Numeric(10,2) 而不是 Float —— 浮点数无法精确表示小数
  |     3. created_at 用 server_default=func.now()，由数据库生成，避免多机时钟不一致
  |     4. Book.__repr__ 让调试输出更可读
  | 
  |   建表后数据库中的列：['id', 'title', 'author', 'isbn', 'price', 'stock', 'created_at']
  |   与模型定义一致    ：True
  | 
  |   实例化一个 Book 对象（未提交）：<Book id=None title='流畅的 Python' price=139.0>
  | 
  | models.py 自检完成 ✓

==============================================================================
[29/31] 10_综合实战/book_api/routers/books.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\10_综合实战\book_api\routers\books.py --check
  返回码  ：0   成功 ✓   耗时 0.73 秒
  ---- 标准输出最后 15 行 ----
  | 
  |   路由清单（方法 / 完整路径 / 端点函数 / 说明）：
  |     GET     /api/books                list_books  图书列表（分页 / 搜索 / 排序）
  |     GET     /api/books/{book_id}      get_book    图书详情
  |     POST    /api/books                create_book 新增图书
  |     PUT     /api/books/{book_id}      update_book 全量更新图书（PUT）
  |     PATCH   /api/books/{book_id}      patch_book  局部更新图书（PATCH）
  |     DELETE  /api/books/{book_id}      delete_book 删除图书
  | 
  |   分层约定回顾：
  |     · 本文件只做：解析请求 → 调用 crud → 包装响应；
  |     · 数据库会话由 Depends(get_db) 注入，路由里不出现 SessionLocal()；
  |     · 业务错误统一 raise HTTPException，由 main.py 的处理器转成统一 JSON。
  | 
  | routers/books.py 自检完成 ✓

==============================================================================
[30/31] 10_综合实战/book_api/schemas.py    （模式：FastAPI --check）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe F:\ProGram\Python_Base\Back_End\10_综合实战\book_api\schemas.py --check
  返回码  ：0   成功 ✓   耗时 0.19 秒
  ---- 标准输出最后 15 行 ----
  |       ✓ ISBN 含字母：字段 isbn -> String should match pattern '^[0-9\-]{10,20}$'
  |       ✓ 库存为负：字段 stock -> Input should be greater than or equal to 0
  | 
  |   【3】BookPatch：所有字段可选（PATCH 局部更新的关键）
  |       只传 stock：{'stock': 5}
  |       exclude_unset=True 只会带上客户端真正提交过的字段 ——
  |       这正是 PATCH 不会把未提交字段覆盖成 None 的原因。
  | 
  |   【4】BookOut：from_attributes=True 可直接从 ORM 对象构造
  |       BookOut.model_validate(ORM 对象) -> {'id': 1, 'title': '流畅的 Python', 'author': 'Luciano Ramalho', 'isbn': '978-7-115-45415-7', 'price': 139.0, 'stock': 10, 'created_at': datetime.datetime(2026, 1, 1, 12, 0)}
  | 
  |   【5】PageOut：统一分页结构
  |       {'total': 1, 'page': 1, 'size': 10, 'items': [{'id': 1, 'title': '流畅的 Python', 'author': 'Luciano Ramalho', 'isbn': '978-7-115-45415-7', 'price': 139.0, 'stock': 10, 'created_at': datetime.datetime(2026, 1, 1, 12, 0)}]}
  | 
  | schemas.py 自检完成 ✓

==============================================================================
[31/31] 10_综合实战/book_api/test_book_api.py    （模式：pytest）
==============================================================================
  运行方式：F:\ProGram\Python_Base\.venv\Scripts\python.exe -m pytest F:\ProGram\Python_Base\Back_End\10_综合实战\book_api\test_book_api.py -q --no-header
  返回码  ：0   成功 ✓   耗时 1.61 秒
  ---- 标准输出最后 2 行 ----
  | ...................                                                      [100%]
  | 19 passed in 0.67s

##############################################################################
# 脚本自检汇总
##############################################################################
脚本                                                      模式                 返回码       耗时
------------------------------------------------------------------------------
03_配置文件/01_pydantic_settings配置.py                       直接运行                 0    0.31s
04_日志/01_loguru基础.py                                    直接运行                 0    0.17s
04_日志/02_loguru文件与轮转.py                                 直接运行                 0    0.23s
04_日志/03_loguru异常捕获.py                                  直接运行                 0    0.16s
05_Flask/01_最小应用.py                                     直接运行                 0    0.24s
05_Flask/02_路由与请求.py                                    直接运行                 0    0.24s
05_Flask/03_模板与静态文件.py                                  直接运行                 0    0.34s
05_Flask/04_RESTful接口.py                                直接运行                 0    0.23s
06_FastAPI/01_最小应用.py                                   FastAPI --check      0    0.49s
06_FastAPI/02_GET与POST.py                               FastAPI --check      0    0.47s
06_FastAPI/03_数据模型与验证.py                                FastAPI --check      0    0.48s
06_FastAPI/04_模板与静态文件.py                                FastAPI --check      0    0.61s
06_FastAPI/05_异步处理.py                                   FastAPI --check      0    3.24s
06_FastAPI/06_数据库集成.py                                  FastAPI --check      0    2.96s
06_FastAPI/07_依赖注入.py                                   FastAPI --check      0    0.49s
06_FastAPI/08_认证与授权.py                                  FastAPI --check      0    0.91s
06_FastAPI/09_中间件.py                                    FastAPI --check      0    0.46s
06_FastAPI/10_错误处理.py                                   FastAPI --check      0    0.46s
06_FastAPI/11_Session与Cookie.py                         FastAPI --check      0    0.49s
06_FastAPI/12_Celery后台任务.py                             FastAPI --check      0    2.65s
06_FastAPI/verify_api.py                                直接运行                 0    3.17s
07_测试框架/calculator.py                                   直接运行                 0    0.03s
07_测试框架/test_calculator.py                              pytest               0    1.23s
09_容器部署/检查YAML.py                                       直接运行                 0    0.08s
10_综合实战/book_api/crud.py                                FastAPI --check      0    0.62s
10_综合实战/book_api/database.py                            FastAPI --check      0    0.48s
10_综合实战/book_api/main.py                                FastAPI --check      0    0.93s
10_综合实战/book_api/models.py                              FastAPI --check      0    0.48s
10_综合实战/book_api/routers/books.py                       FastAPI --check      0    0.73s
10_综合实战/book_api/schemas.py                             FastAPI --check      0    0.19s
10_综合实战/book_api/test_book_api.py                       pytest               0    1.61s
------------------------------------------------------------------------------

##############################################################################
# 附加验证：pytest（07_测试框架 + 10_综合实战/book_api）
##############################################################################
  | .................................................s.x.................... [ 90%]
  | ........                                                                 [100%]
  | =========================== short test summary info ===========================
  | SKIPPED [1] 07_测试框架\test_calculator.py:249: 演示用：这个用例故意被跳过，不会计入失败
  | XFAIL 07_测试框架/test_calculator.py::test_xfail_demo - 演示用：这是一个已知会失败的用例，xfail 表示'预期失败'
  | 78 passed, 1 skipped, 1 xfailed in 0.98s
  pytest 返回码：0   全部通过 ✓   耗时 1.98 秒

##############################################################################
# 附加验证：06_FastAPI/verify_api.py（TestClient 逐个示例校验接口）
##############################################################################
  | ==============================================================================
  | 接口校验汇总
  | ==============================================================================
  | 参与校验的应用：13 / 13
  | 实际发出的请求：96
  | 失败项：0
  | 
  | 结论：所有示例接口校验通过 ✓（每个示例的详细讲解见对应脚本的 --check 输出）
  verify_api 返回码：0   接口全部通过 ✓   耗时 3.20 秒

##############################################################################
# 最终结论
##############################################################################
共 31 个脚本，成功 31 个，失败 0 个。
pytest 附加验证：通过 ✓
verify_api 接口校验：通过 ✓

全部自检通过 ✓（脚本 0 失败，pytest 全绿，96 个接口请求全部符合预期）
提示：把本文件的完整输出保存到 Back_End/VERIFY_REPORT.md 即可作为交付证据。
```

**关键汇总行（原样摘录）：**

```text
共 31 个脚本，成功 31 个，失败 0 个。
pytest 附加验证：通过 ✓
verify_api 接口校验：通过 ✓
全部自检通过 ✓（脚本 0 失败，pytest 全绿，96 个接口请求全部符合预期）
```

---

## 三、pytest 结果

### 3.1 `Back_End/07_测试框架`（命令 ②）

```text
.................................................s.x.........            [100%]
=========================== short test summary info ===========================
SKIPPED [1] Back_End\07_测试框架\test_calculator.py:249: 演示用：这个用例故意被跳过，不会计入失败
XFAIL Back_End\07_测试框架\test_calculator.py::test_xfail_demo - 演示用：这是一个已知会失败的用例，xfail 表示'预期失败'
59 passed, 1 skipped, 1 xfailed in 0.44s
```

### 3.2 `Back_End/10_综合实战/book_api`（命令 ③）

```text
...................                                                      [100%]
19 passed in 0.67s
```

### 3.3 整个 `Back_End`（命令 ④）

```text
........................................................................ [ 92%]
.......                                                                  [100%]
=========================== short test summary info ===========================
SKIPPED [1] Back_End\07_测试框架\test_calculator.py:249: 演示用：这个用例故意被跳过，不会计入失败
XFAIL Back_End\07_测试框架/test_calculator.py::test_xfail_demo - 演示用：这是一个已知会失败的用例，xfail 表示'预期失败'
78 passed, 1 skipped, 1 xfailed in 1.04s
```

**结论：**

| 范围 | 结果 |
| --- | --- |
| `07_测试框架` | `59 passed, 1 skipped, 1 xfailed` |
| `10_综合实战/book_api` | `19 passed` |
| 整个 `Back_End` | `78 passed, 1 skipped, 1 xfailed` |

> `1 skipped` 与 `1 xfailed` 是**故意设计**的教学用例（演示 `@pytest.mark.skip` 与 `@pytest.mark.xfail`），不是缺陷。

---

## 四、越界检查（`git status --short`）

```text
AM "RAG/\346\265\213\350\257\225\351\227\256\351\242\230.txt"
 M pyproject.toml
 M uv.lock
?? .course_extract/
?? Back_End/
?? Deep_Learning/
?? Front_End/
?? Machine_Learning/
```

**逐条说明：**

| 条目 | 是否本 agent 造成 | 说明 |
| --- | --- | --- |
| `?? Back_End/` | ✅ 是 | 本次任务的全部交付物，全部位于 `Back_End/` 内 |
| `?? Deep_Learning/`、`?? Front_End/`、`?? Machine_Learning/` | ❌ 否 | 由并行处理的其他 agent 创建，本 agent 未写入任何文件 |
| `?? .course_extract/` | ❌ 否 | 课案抽取产物（任务开始前已存在），本 agent 只读取 |
| ` M pyproject.toml`、` M uv.lock` | ❌ 否 | 任务开始前就存在（内容是新增 celery / flask / lightgbm / pytest / pytest-asyncio 等依赖声明，即环境预装依赖所致）。本 agent **从未执行** `uv add` / `uv sync` / `pip install`，也未编辑过这两个文件 |
| `AM "RAG/测试问题.txt"` | ❌ 否 | 任务开始前已存在 |

**结论：本 agent 只在 `Back_End/` 内创建文件，没有修改/删除该目录之外的任何文件。**

---

## 五、交付物清单

- Python 脚本：**36** 个（其中 **31** 个是可独立运行的"脚本"，其余为包标识 `__init__.py`、pytest 的 `conftest.py` 与工具入口 `verify_all.py`）
- Markdown 文档：**12** 个
- 另有 Dockerfile / docker-compose.yml / .dockerignore / K8S YAML / ruff.toml / .editorconfig / .env.example / pytest.ini 等配置文件

```text
Back_End/01_基础环境/
Back_End/01_基础环境/Docker常用命令.md    (11414 字节)
Back_End/01_基础环境/Linux常用命令速查.md    (17140 字节)
Back_End/01_基础环境/Ubuntu与AlmaLinux对比.md    (8428 字节)
Back_End/01_基础环境/服务器部署与安全组.md    (9412 字节)
Back_End/02_包管理/
Back_End/02_包管理/uv使用指南.md    (12295 字节)
Back_End/03_配置文件/
Back_End/03_配置文件/.env.example    (1916 字节)
Back_End/03_配置文件/01_pydantic_settings配置.py    (15348 字节)
Back_End/03_配置文件/配置文件讲解.md    (7438 字节)
Back_End/04_日志/
Back_End/04_日志/01_loguru基础.py    (9218 字节)
Back_End/04_日志/02_loguru文件与轮转.py    (11113 字节)
Back_End/04_日志/03_loguru异常捕获.py    (12470 字节)
Back_End/05_Flask/
Back_End/05_Flask/01_最小应用.py    (8218 字节)
Back_End/05_Flask/02_路由与请求.py    (13674 字节)
Back_End/05_Flask/03_模板与静态文件.py    (10670 字节)
Back_End/05_Flask/04_RESTful接口.py    (16404 字节)
Back_End/05_Flask/static/
Back_End/05_Flask/static/logo.svg    (734 字节)
Back_End/05_Flask/static/style.css    (2024 字节)
Back_End/05_Flask/templates/
Back_End/05_Flask/templates/about.html    (1320 字节)
Back_End/05_Flask/templates/base.html    (1890 字节)
Back_End/05_Flask/templates/index.html    (2911 字节)
Back_End/06_FastAPI/
Back_End/06_FastAPI/01_最小应用.py    (9376 字节)
Back_End/06_FastAPI/02_GET与POST.py    (14360 字节)
Back_End/06_FastAPI/03_数据模型与验证.py    (20491 字节)
Back_End/06_FastAPI/04_模板与静态文件.py    (10636 字节)
Back_End/06_FastAPI/05_异步处理.py    (14001 字节)
Back_End/06_FastAPI/06_数据库集成.py    (20647 字节)
Back_End/06_FastAPI/07_依赖注入.py    (18450 字节)
Back_End/06_FastAPI/08_认证与授权.py    (22242 字节)
Back_End/06_FastAPI/09_中间件.py    (16410 字节)
Back_End/06_FastAPI/10_错误处理.py    (18314 字节)
Back_End/06_FastAPI/11_Session与Cookie.py    (19618 字节)
Back_End/06_FastAPI/12_Celery后台任务.py    (21421 字节)
Back_End/06_FastAPI/static/
Back_End/06_FastAPI/static/a.svg    (879 字节)
Back_End/06_FastAPI/static/style.css    (1986 字节)
Back_End/06_FastAPI/templates/
Back_End/06_FastAPI/templates/index.html    (3038 字节)
Back_End/06_FastAPI/verify_api.py    (21233 字节)
Back_End/07_测试框架/
Back_End/07_测试框架/calculator.py    (5676 字节)
Back_End/07_测试框架/conftest.py    (5998 字节)
Back_End/07_测试框架/test_calculator.py    (13350 字节)
Back_End/07_测试框架/测试说明.md    (5816 字节)
Back_End/08_代码格式化/
Back_End/08_代码格式化/.editorconfig    (2655 字节)
Back_End/08_代码格式化/README.md    (8909 字节)
Back_End/08_代码格式化/ruff.toml    (4692 字节)
Back_End/09_容器部署/
Back_End/09_容器部署/.dockerignore    (1882 字节)
Back_End/09_容器部署/docker-compose.yml    (7165 字节)
Back_End/09_容器部署/Dockerfile    (4228 字节)
Back_End/09_容器部署/k8s/
Back_End/09_容器部署/k8s/configmap.yaml    (4206 字节)
Back_End/09_容器部署/k8s/deployment.yaml    (6040 字节)
Back_End/09_容器部署/k8s/ingress.yaml    (4042 字节)
Back_End/09_容器部署/k8s/README.md    (13007 字节)
Back_End/09_容器部署/k8s/service.yaml    (3499 字节)
Back_End/09_容器部署/检查YAML.py    (21127 字节)
Back_End/10_综合实战/
Back_End/10_综合实战/book_api/
Back_End/10_综合实战/book_api/__init__.py    (1936 字节)
Back_End/10_综合实战/book_api/conftest.py    (2617 字节)
Back_End/10_综合实战/book_api/crud.py    (7922 字节)
Back_End/10_综合实战/book_api/database.py    (5482 字节)
Back_End/10_综合实战/book_api/main.py    (14234 字节)
Back_End/10_综合实战/book_api/models.py    (5053 字节)
Back_End/10_综合实战/book_api/README.md    (6558 字节)
Back_End/10_综合实战/book_api/routers/
Back_End/10_综合实战/book_api/routers/__init__.py    (334 字节)
Back_End/10_综合实战/book_api/routers/books.py    (9185 字节)
Back_End/10_综合实战/book_api/schemas.py    (6239 字节)
Back_End/10_综合实战/book_api/test_book_api.py    (8403 字节)
Back_End/data/
Back_End/data/book_api.db    (16384 字节)
Back_End/data/demo.db    (28672 字节)
Back_End/data/env_demo/
Back_End/data/env_demo/.env.dev    (231 字节)
Back_End/data/env_demo/.env.prod    (278 字节)
Back_End/data/logs/
Back_End/data/logs/01_loguru基础.log    (7080 字节)
Back_End/data/logs/03_异常记录.log    (6499 字节)
Back_End/data/logs/rotation/
Back_End/data/logs/rotation/app.2026-09-11_16-20-15_099715.log.zip    (599 字节)
Back_End/data/logs/rotation/app.2026-09-11_16-20-15_106859.log.zip    (598 字节)
Back_End/data/logs/rotation/app.2026-09-11_16-20-15_113555.log.zip    (591 字节)
Back_End/data/logs/rotation/app.log    (6179 字节)
Back_End/data/logs/rotation/only_error.log    (142 字节)
Back_End/data/logs/rotation/structured.json    (1418 字节)
Back_End/pytest.ini    (1644 字节)
Back_End/README.md    (14414 字节)
Back_End/verify_all.py    (12211 字节)
Back_End/VERIFY_REPORT.md    (66083 字节)
```

---

## 六、外部服务降级验证（本机零外部服务）

| 服务 | 预期连接结果 | 实际表现 |
| --- | --- | --- |
| MySQL 127.0.0.1:3306 | 连接失败（服务未启动） | `06_数据库集成.py::probe_mysql()` 捕获 `OperationalError`，打印中文提示，程序继续用 SQLite，**退出码 0** |
| Redis 127.0.0.1:6379 | 连接失败（服务未启动） | `12_Celery后台任务.py` 用 socket 探测报告 `available=False`，Celery 处于 `task_always_eager=True`，`delay()/apply_async()/chain` 全部在当前进程跑通，**不阻塞、退出码 0** |
| PostgreSQL 5432 / Milvus 19530 | 未使用 | 本目录无依赖，不涉及 |

---

## 七、结论

1. `Back_End` 下 **31 个可运行脚本全部返回码 0**，输出中 **0 处 Traceback**；
2. pytest 全绿（`78 passed, 1 skipped, 1 xfailed`，skip/xfail 为教学用例）；
3. `verify_api.py` 对 **13 个 FastAPI 应用发出 96 个请求，全部符合预期状态码**；
4. `09_容器部署/检查YAML.py` 对 5 个 YAML + Dockerfile + .dockerignore 完成静态校验，全部通过；
5. 所有脚本在**不启动 MySQL / Redis / PostgreSQL / Milvus** 的前提下零报错跑完；
6. 只在 `Back_End/` 内创建文件，未触碰边界外的任何文件。
