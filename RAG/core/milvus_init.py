# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""Milvus 初始化:创建业务数据库与业务用户(需使用 root 账号连接),可重复执行"""

from pymilvus import MilvusClient

from config import settings

ROOT_USER = "root"
ROOT_PASSWORD = "Milvus"
BUSINESS_ROLE = "admin"


def main() -> None:
    cfg = settings.milvus
    client = MilvusClient(uri=cfg.uri, user=ROOT_USER, password=ROOT_PASSWORD)

    if cfg.db_name not in client.list_databases():
        client.create_database(db_name=cfg.db_name)
        print(f"数据库 {cfg.db_name} 已创建")
    else:
        print(f"数据库 {cfg.db_name} 已存在")

    if cfg.user not in client.list_users():
        client.create_user(user_name=cfg.user, password=cfg.password)
        print(f"用户 {cfg.user} 已创建")
    else:
        print(f"用户 {cfg.user} 已存在")

    info = client.describe_user(user_name=cfg.user)
    raw_roles = info.get("roles", ()) if isinstance(info, dict) else ()
    roles = {r if isinstance(r, str) else r.get("role_name") for r in raw_roles}
    if BUSINESS_ROLE not in roles:
        client.grant_role(user_name=cfg.user, role_name=BUSINESS_ROLE)
        print(f"角色 {BUSINESS_ROLE} 已授予 {cfg.user}")
    else:
        print(f"用户 {cfg.user} 已拥有角色 {BUSINESS_ROLE}")

    client.close()
    print("Milvus 初始化完成")


if __name__ == "__main__":
    main()
