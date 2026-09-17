"""Milvus 初始化:创建业务数据库与业务用户(需使用 root 账号连接),可重复执行"""

# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
# 为什么必须手写这段（三个理由，缺一个都会 ImportError）：
#   1) RAG 不是一个已安装的包，`from config import settings`（仓库根 Python_Base\config.py）
#      与 `from core... / from pipeline...`（RAG 根下的包）都只能靠 sys.path 才解析得到；
#   2) 用 `python RAG\core\milvus_init.py` 直接跑时，sys.path[0] 是**脚本自己所在的目录**
#      （RAG\core），既没有仓库根也没有 RAG 根；
#   3) 被别的脚本 import（例如复用 ROOT_USER 常量）、或从任意工作目录被调用时，
#      CWD 不可控 —— 本段的效果必须与"从哪启动、CWD 是什么"无关。
# 代价：同一段样板在 app\main.py、app\chat_ui.py 里各有一份（逻辑复制三处），改动要同步。
import sys as _sys
from pathlib import Path as _Path

# 从本文件位置逐级向上找名为 Python_Base 的祖先目录。
# 循环条件里的 `_BASE.parent != _BASE` 是**到顶保护**：Path 到达盘符根后 parent 等于自己，
# 不写它就成了死循环；写了的话，仓库万一被改名，_BASE 会停在盘符根而不是卡死。
_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
# 两次 insert(0) 之后列表顺序是 [RAG 根, Python_Base 根, ...]：RAG 根在最前，
# 保证 `core` / `llm` / `pipeline` 这些通用词优先解析到本项目的目录。
# 用 _sys / _Path 这种带下划线的别名，是为了不给模块命名空间留下 sys / Path
# 这两个极易与业务变量重名的名字。
# （这份 docstring 原先写在这段引导**之后**，因而 `core.milvus_init.__doc__` 是 None；
# 已挪到文件首个语句 —— 原先那句"要让它真正生效，需把字符串挪到 import 之前"已经做了。）

# 本脚本只做 Milvus 的「库 / 账号」级初始化，三件事，且都可重复执行（幂等）：
#   1) 建业务数据库（.env 的 MILVUS_DB_NAME，本机是 finance_rag）；
#   2) 建业务账号（MILVUS_USER / MILVUS_PASSWORD）—— 业务代码连的是这个账号；
#   3) 给业务账号授 admin 角色（Milvus 的两个**内置角色**：admin=有管理权限，public=只读；
#      见 pymilvus\client\types.py::UserItem 的示例 `<roles:('admin', 'public')>`）。
# 它**不**负责建集合 tick、建索引、灌数据 —— 那些在 data_process\tick_extract.py
# （--insert --recreate）与 data_process\embed_tickets.py 里；
# 也不负责启服务：Milvus 跑在本机 docker（连不上时先按 _error_reply("milvus") 的提示起库）。
#
# 为什么用 root 连接（⚠ 理由要说准，下面两条都是实测过的）：
#   - 这里读写的是**集群级**资源（库清单 / 用户清单），不是某个库里的数据，所以客户端
#     刻意不带 db_name —— 本脚本的职责就是先把那个库建出来，"指向还不存在的库"没有意义；
#   - 用 root 是"先有鸡还是先有蛋"：业务账号本身要由本脚本创建并授权，建好之前谈不上用它。
#     ⚠ 但**不要**写成"业务账号没有权限" —— 实测业务账号拿到的就是 admin 角色，
#     用它一样能 create_database；差别只在于那时业务账号还不存在。
# ROOT_USER / ROOT_PASSWORD 是 Milvus 单机版 docker 的**出厂默认值**，所以直接写死 ——
# 它只用于本地开发环境；业务账号的口令一律走 .env，不进代码。

from pymilvus import MilvusClient

from config import settings

ROOT_USER = "root"
ROOT_PASSWORD = "Milvus"
BUSINESS_ROLE = "admin"


def main() -> None:
    """幂等地初始化业务库与业务账号：已存在就只打印一行，不报错、不重复创建。"""
    cfg = settings.milvus
    # 业务库名 / 账号 / 口令全部来自根 .env（MILVUS_* 前缀，见 config.py::RagMilvusSettings）。
    # 这里刻意**不带 db_name**：下面读/写的是集群级的「库清单 / 用户清单」，而业务库正是
    # 本脚本要创建的对象 —— 指向一个还不存在的库没有意义。
    # （实测：带上不存在的 db_name 构造 MilvusClient 并不会报错、list_databases 也照常，
    #   所以理由是"语义上不该带"，不是"会连不上"。）
    client = MilvusClient(uri=cfg.uri, user=ROOT_USER, password=ROOT_PASSWORD)

    # 先查再建：create_database 对已存在的库会直接抛异常，那样脚本就不能重复跑了。
    if cfg.db_name not in client.list_databases():
        client.create_database(db_name=cfg.db_name)
        print(f"数据库 {cfg.db_name} 已创建")
    else:
        print(f"数据库 {cfg.db_name} 已存在")

    # 同样先 list_users 判断存在性，理由同上（脚本要能被反复执行）。
    if cfg.user not in client.list_users():
        client.create_user(user_name=cfg.user, password=cfg.password)
        print(f"用户 {cfg.user} 已创建")
    else:
        print(f"用户 {cfg.user} 已存在")

    # 下面三行是对 describe_user 返回值的**形态兼容**：roles 里的元素既可能是字符串，
    # 也可能是 {"role_name": ...} 这类字典（两种形态都存在过），所以统一归一化后再判存在性，
    # 免得取不到角色名 → 每次跑都重复授权。
    info = client.describe_user(user_name=cfg.user)
    # 不是 dict 就按「没有角色」处理：初始化脚本宁可多试一次授权，也不要在读角色时抛异常中断。
    # 代价要说清楚：若角色其实已存在，这一岔路会让 grant_role 报错（脚本不再幂等）——
    # 但那种情况说明返回值形态超出预期，报错比静默跳过更容易被发现。
    raw_roles = info.get("roles", ()) if isinstance(info, dict) else ()
    roles = {r if isinstance(r, str) else r.get("role_name") for r in raw_roles}
    # 去重成集合后判存在性：这样重复执行时不会重复调用 grant_role、日志也干净。
    # ⚠ 但"grant_role 对已有角色会报错"这个常见说法**不成立**：实测（pymilvus 3.0.1）
    #   对已授予的角色再次 grant_role 不报错、是幂等的 —— 所以这个守卫不是"防报错的关键"，
    #   别把它当成必要保护（真正必须要守卫的是上面的 create_database / create_user）。
    if BUSINESS_ROLE not in roles:
        client.grant_role(user_name=cfg.user, role_name=BUSINESS_ROLE)
        print(f"角色 {BUSINESS_ROLE} 已授予 {cfg.user}")
    else:
        print(f"用户 {cfg.user} 已拥有角色 {BUSINESS_ROLE}")

    # 显式关闭：MilvusClient 持有 gRPC 通道，不关会留下连接（脚本短命，但习惯要立住）。
    client.close()
    print("Milvus 初始化完成")
    # 这里的输出故意用 print 而不是 core.logger：本脚本是**给人手动跑的一次性运维脚本**，
    # 结果直接打在终端上更直观，不需要落进 logs 目录（也就没引 core.logger）。


if __name__ == "__main__":
    # 自检 / 入口：只有**以脚本方式直接运行**时才执行初始化 ——
    # 被 import（例如被别的运维脚本复用 ROOT_USER 常量）时不会误建库、误建账号。
    #
    # 它验证什么：Milvus 的库、业务账号、角色三件套是否齐备。判据是**幂等性** ——
    # 连跑两次，第二次必须打印「已存在 / 已拥有角色」而不是报错，说明真做成了。
    # 想进一步确认，可连上后复核 client.list_databases() / list_users() / describe_user()。
    main()
