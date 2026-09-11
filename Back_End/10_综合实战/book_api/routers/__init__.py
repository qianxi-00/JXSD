"""routers 包：把不同业务的路由分文件管理。

随着项目变大，main.py 里塞几百个路由会无法维护。
标准做法是每个业务域一个模块（books.py、users.py、orders.py……），
每个模块导出一个 APIRouter，最后由 main.py 用 include_router 装配起来。
"""

__all__ = ["books"]
