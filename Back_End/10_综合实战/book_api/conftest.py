"""conftest.py —— book_api 的 pytest 公共配置与 fixture
================================================================
对应课案章节：后端开发基础 → 测试框架（pytest fixture）

pytest 会自动发现同目录及其子目录下的 conftest.py，
里面定义的 fixture 可以被同目录的测试文件直接使用（不需要 import）。

本文件的 fixture：
    client  —— 基于 TestClient 的接口测试客户端（with 语法触发 lifespan 建表）
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# ------------------------------------------------------------
# 路径引导：让测试能 import 到 main / database 等同级模块
# 本文件位于 Back_End/10_综合实战/book_api/conftest.py
# ------------------------------------------------------------
BOOK_API_DIR = pathlib.Path(__file__).resolve().parent
BACK_END = BOOK_API_DIR.parents[1]
ROOT = BOOK_API_DIR.parents[2]
for _p in (str(BOOK_API_DIR), str(BACK_END), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def app_module():
    """整个测试会话只导入一次应用模块。

    scope="session"：模块导入（含引擎创建）只做一次，测试更快；
    同时避免 SQLite 引擎被反复创建。
    """
    import main

    return main


@pytest.fixture()
def client(app_module):
    """每个测试函数一个独立的 TestClient。

    用 `with` 而不是直接构造：
        with TestClient(app) as c:
            ...
    会触发 lifespan 的启动/关闭逻辑（建表、释放连接池），
    这样测试环境和真实运行环境保持一致。
    """
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture()
def sample_book(client):
    """准备一条测试数据，并保证测试结束后清理掉。

    yield 之前 = Setup（创建数据），之后 = Teardown（删除数据）。
    这就是课案讲的 fixture 清理模式。
    """
    import uuid

    payload = {
        "title": "fixture 创建的书",
        "author": "pytest",
        "isbn": f"978-7-111-{uuid.uuid4().int % 100000:05d}-0",
        "price": 88.0,
        "stock": 3,
    }
    resp = client.post("/api/books", json=payload)
    assert resp.status_code == 201, "fixture 创建测试数据失败"
    book = resp.json()["data"]

    yield book

    # Teardown：删掉它，保证测试互不影响（幂等：可能已被用例自己删除）
    client.delete(f"/api/books/{book['id']}")
