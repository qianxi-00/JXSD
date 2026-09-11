"""数据库连接管理"""

from pymilvus import MilvusClient

from config import settings


def get_milvus_client(db_name: str | None = None) -> MilvusClient:
    """创建向量数据库 (Milvus) 客户端"""
    cfg = settings.milvus
    return MilvusClient(
        uri=cfg.uri,
        user=cfg.user,
        password=cfg.password,
        db_name=db_name or cfg.db_name,
    )


def get_mysql_engine():
    """创建 MySQL 连接引擎,用于元数据管理"""
    raise NotImplementedError
