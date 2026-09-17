"""测试基建守卫:本地服务不可用时必须"快速跳过",不能把套件挂死或跑红。

背景(实测):
- 数据库栈的容器是 `restart: "no"`,不随 Docker 自启,机器重启后常常是停着的;
- 某次服务未启动时 `uv run pytest RAG` 卡满 600 秒超时,根因是 Text-to-SQL 的
  integration 用例直接连 PostgreSQL,而 libpq 默认连接超时要等约 130 秒才失败;
- GraphRAG 的 integration 用例做对了(带 `requires_neo4j` 守卫,不可用即跳过),
  说明约定已有,只是没覆盖到所有模块。

本文件既覆盖"探针 + 连接超时"的行为,也用一条元测试锁住
"凡是 integration 用例都必须带可用性守卫"这条约定。
"""

import ast
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


class TestServiceProbe:
    def test_closed_port_reported_unavailable_quickly(self):
        from service_probe import service_available

        started = time.perf_counter()
        assert service_available("127.0.0.1", 1, timeout=0.3) is False
        assert time.perf_counter() - started < 3, "探测必须快速返回,不能继承长超时"

    def test_probes_return_bool(self):
        from service_probe import (
            milvus_available,
            neo4j_available,
            pg_available,
            redis_available,
        )

        for probe in (pg_available, neo4j_available, redis_available, milvus_available):
            assert isinstance(probe(), bool), probe.__name__

    def test_pg_probe_uses_configured_host_and_port(self, monkeypatch):
        import service_probe

        seen = {}

        def fake(host, port, timeout=0.5):
            seen["host"] = host
            seen["port"] = port
            return True

        monkeypatch.setattr(service_probe, "service_available", fake)
        assert service_probe.pg_available() is True
        assert seen["port"] == service_probe.settings.postgres_port

    def test_skip_markers_exist(self):
        import service_probe

        for name in ("requires_pg", "requires_neo4j", "requires_redis", "requires_milvus"):
            assert hasattr(service_probe, name), name


class TestPgEngineFailsFast:
    def test_engine_sets_short_connect_timeout(self, monkeypatch):
        """回归:引擎必须带短连接超时,否则服务不可用时要卡满默认超时才报错"""
        from agentic import text_to_sql

        captured: dict = {}

        def fake_create_engine(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return "fake-engine"

        monkeypatch.setattr(text_to_sql, "create_engine", fake_create_engine)
        monkeypatch.setattr(text_to_sql, "_engine", None)

        assert text_to_sql.get_engine() == "fake-engine"
        assert captured["connect_args"]["connect_timeout"] <= 5
        assert captured["pool_pre_ping"] is True


def _decorator_names(node: ast.AST) -> set[str]:
    """收集装饰器里出现的名字(pytest.mark.integration / requires_pg / skipif ...)"""
    names: set[str] = set()
    for decorator in getattr(node, "decorator_list", []):
        names.update(part for part in ast.dump(decorator).split("'") if part.isidentifier())
    return names


def _is_integration(names: set[str], source: str) -> bool:
    return "integration" in names or "pytestmark" in names and "integration" in source


def _is_guarded(names: set[str], node: ast.AST) -> bool:
    if any(name == "skipif" or "skipif" in name for name in names):
        return True
    if any(name.startswith("requires_") for name in names):
        return True
    return "pytest.skip(" in ast.unparse(node)


class TestIntegrationGuardConvention:
    """元测试:integration 用例必须带可用性守卫,否则本机服务没起时套件会挂死"""

    def test_every_integration_test_is_guarded(self):
        offenders: list[str] = []

        for path in sorted(TESTS_DIR.glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            module_source = path.read_text(encoding="utf-8")
            module_names = _decorator_names(tree)

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names = _decorator_names(node) | module_names
                    if _is_integration(names, module_source) and not _is_guarded(names, node):
                        offenders.append(f"{path.name}::{node.name}")
                elif isinstance(node, ast.ClassDef):
                    class_names = _decorator_names(node) | module_names
                    for child in node.body:
                        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        names = _decorator_names(child) | class_names
                        if _is_integration(names, module_source) and not _is_guarded(names, child):
                            offenders.append(f"{path.name}::{node.name}::{child.name}")

        assert not offenders, (
            "以下 integration 用例缺少可用性守卫(加 requires_pg/requires_neo4j 等 skipif 守卫): "
            + ", ".join(offenders)
        )
