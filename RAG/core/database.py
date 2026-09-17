"""数据库连接管理

本模块目前只真正实现了 **Milvus(向量库)** 的连接;MySQL 那个函数是**占位桩**,
保留是为了名字/接口不丢(见下方 get_mysql_engine 的注释)。
PostgreSQL(票据关系表)与 Neo4j(图谱)的连接不在本模块,分别在
`script/import_tickets_to_pg.py` 与 `graph_rag/models.py` 里。所以别指望
"统一连接层"在这里 —— 本文件的定位是"向量库客户端工厂"。
"""

from pymilvus import MilvusClient

from config import settings


def get_milvus_client(db_name: str | None = None) -> MilvusClient:
    """创建向量数据库 (Milvus) 客户端

    为什么是**每次调用新建**而不是模块级单例(与 redis_client 的写法相反):
    MilvusClient 在构造时就会按 uri 建连并做一次连通性握手,而本项目的调用方
    遍布一次性脚本(data_process/*、script/*)与常驻服务(app/*),生命周期差异大;
    更重要的是各脚本经常需要在不同的 db_name 之间切换(建库/清理/回填)。
    所以这里选择"工厂"语义,由调用方决定持有多久 —— 想复用就在调用方缓存。
    代价是:在常驻服务里若每次请求都新建,会反复握手(见报告)。

    `db_name or cfg.db_name`:入参优先于配置。传 None(默认)时用 .env 里的默认库;
    传空字符串也会退回默认库 —— 空串在 Milvus 里不是合法库名,当默认处理更安全。

    ⚠ 客户端在这里**不做任何校验**:库名写错、Milvus 没起,都是后续第一次真正
    发命令时才报错(构造期只做 TCP 握手级别的事)。所以别把"能拿到 client"
    当成"Milvus 可用"的证据。
    """
    cfg = settings.milvus
    return MilvusClient(
        uri=cfg.uri,
        user=cfg.user,
        password=cfg.password,
        db_name=db_name or cfg.db_name,
    )


def get_mysql_engine():
    """创建 MySQL 连接引擎,用于元数据管理

    ⚠ **当前是未实现的占位**(调用即 `raise NotImplementedError`),而且这个仓库
    实际用的是 **PostgreSQL** 而不是 MySQL(见 script/import_tickets_to_pg.py)。

    为什么保留而不是删掉:它是从课案骨架抄下来时留下的名字,**删掉这个函数**
    等于改动公共接口(可能有外部/历史脚本 `from core.database import get_mysql_engine`),
    属于"顺手清理"的范畴。本次只加注释、不动代码 ——
    真要清理应当在确认零调用方之后单独做,并同步更新 README(见报告)。

    写 `raise NotImplementedError` 而不是 `pass` 或 `return None` 是**正确**的:
    调用方会立刻拿到明确异常,而不是拿着 None 去 `.connect()` 得到一个
    "NoneType has no attribute"的误导性报错。
    """
    raise NotImplementedError
