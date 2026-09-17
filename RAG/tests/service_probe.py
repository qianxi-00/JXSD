"""本地服务可用性探针 + 集成测试的跳过守卫。

设计约定(与 GraphRAG 的 `requires_neo4j` 保持一致):
- 服务不可用时,**快速跳过**集成用例,不要让套件挂死或跑红;
- 探测只做 TCP 连通性(不做鉴权),超时给足短(默认 0.5 秒);
- 跳过原因写成中文,直接告诉使用者该起什么服务。

为什么要这层守卫:数据库栈的容器是 `restart: "no"`,不随 Docker 自启;
而 libpq 之类的客户端默认连接超时要等约 130 秒才失败 —— 实测整套测试因此
卡满 600 秒超时。用 TCP 探针把"服务没起"从"等两分钟再报错"变成"立即跳过"。
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from config import settings


def service_available(host: str, port: int, timeout: float = 0.5) -> bool:
    """TCP 探一次端口是否可连(不鉴权,只判断服务在不在)。"""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _host_port(uri: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(uri if "://" in uri else f"//{uri}")
    return parsed.hostname or "127.0.0.1", parsed.port or default_port


def pg_available() -> bool:
    return service_available(settings.postgres_host, settings.postgres_port)


def neo4j_available() -> bool:
    host, port = _host_port(settings.neo4j.uri, 7687)
    return service_available(host, port)


def redis_available() -> bool:
    return service_available(settings.redis.host, settings.redis.port)


def milvus_available() -> bool:
    host, port = _host_port(settings.milvus.uri, 19530)
    return service_available(host, port)


requires_pg = pytest.mark.skipif(
    not pg_available(),
    reason=f"PostgreSQL 不可用（{settings.postgres_host}:{settings.postgres_port}），跳过集成测试；"
    "启动服务：powershell -File F:\\DockerDesktopData\\start-dbs.ps1",
)
requires_neo4j = pytest.mark.skipif(
    not neo4j_available(),
    reason=f"Neo4j 不可用（{settings.neo4j.uri}），跳过集成测试；"
    "启动服务：powershell -File F:\\DockerDesktopData\\start-dbs.ps1",
)
requires_redis = pytest.mark.skipif(
    not redis_available(),
    reason=f"Redis 不可用（{settings.redis.host}:{settings.redis.port}），跳过集成测试；"
    "启动服务：powershell -File F:\\DockerDesktopData\\start-dbs.ps1",
)
requires_milvus = pytest.mark.skipif(
    not milvus_available(),
    reason=f"Milvus 不可用（{settings.milvus.uri}），跳过集成测试；"
    "启动服务：powershell -File F:\\DockerDesktopData\\start-dbs.ps1",
)
