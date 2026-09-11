"""book_api —— 一个"小而完整"的 FastAPI 图书管理项目
================================================================
对应课案章节：后端开发基础 → Web框架（综合运用）+ 后端工程结构最佳实践

本项目把前面各节的零散知识点组装成一个真实可用的应用，目录结构如下：

    book_api/
    ├── __init__.py         包标识
    ├── database.py         数据库连接、会话工厂、依赖注入的 get_db、建表
    ├── models.py           ORM 模型（SQLAlchemy 2.0 风格）
    ├── schemas.py          Pydantic 请求/响应模型
    ├── crud.py             数据访问层（只关心数据库操作）
    ├── routers/
    │   ├── __init__.py
    │   └── books.py        路由层（只关心 HTTP 语义）
    ├── main.py             应用入口：装配路由、中间件、异常处理
    ├── test_book_api.py    pytest + TestClient 自测
    └── README.md           运行说明

**为什么要分层？**（这是本节最重要的工程知识）
    路由层（routers）只做三件事：解析请求、调用业务、包装响应；
    业务/数据层（crud）只关心"数据怎么存怎么取"；
    模型层（models/schemas）负责结构定义。
    这样做的收益：
        · 换数据库只改 database.py + crud.py，路由一行不动；
        · 写单元测试可以直接测 crud，不需要起 HTTP；
        · 多人协作时各改各的文件，冲突少。

**关于 import**：为了让 `python main.py` 与"被其他脚本按路径加载"两种方式都能工作，
每个模块都会把自己所在目录加入 sys.path，然后使用**扁平 import**（import models），
而不是包内相对 import（from . import models）—— 后者要求必须以包的方式被加载。
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
