"""端点自检脚本的行为测试（不真连任何端点）。

盯的是**脚本自己**的骨架：异常要变成"失败"而不是把脚本崩掉、退出码要能被 CI 用。
真端点连通性由脚本真机跑（见 README §7.1 的证据行），单测不替它背书。
"""

import pytest

from script import check_endpoints


def test_unknown_endpoint_is_rejected():
    # argparse.error 会 SystemExit(2)：拼错名字必须立刻停，不能"看起来在探但什么都没探"
    with pytest.raises(SystemExit) as excinfo:
        check_endpoints.main(["--only", "embedding,不存在的端点"])
    assert excinfo.value.code == 2


def test_exception_becomes_a_failed_check_not_a_crash(monkeypatch):
    """一个检查抛异常时，其余检查要照跑完，整体退出码为 1。

    这条不是假想：写这个脚本的第一版把配置字段名写成了 `settings.embedding.embedding_model`
    （真名是 `settings.embedding.model`），两个检查直接 AttributeError。
    当时正是这段兜底把错误记成了 FAIL 明细 —— 不然看到的就是一串 traceback。
    """
    monkeypatch.setitem(
        check_endpoints.CHECKS, "embedding", ("假 Embedding", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    monkeypatch.setitem(check_endpoints.CHECKS, "rerank", ("假 Reranker", lambda: {"ok": True, "detail": "ok"}))

    results = check_endpoints.run_checks(["embedding", "rerank"])
    assert [item["ok"] for item in results] == [False, True]
    assert "RuntimeError: boom" in results[0]["detail"]
    assert results[0]["seconds"] >= 0
    assert check_endpoints.main(["--only", "embedding,rerank"]) == 1
    assert check_endpoints.main(["--only", "rerank"]) == 0


def test_json_output_is_parseable(monkeypatch, capsys):
    monkeypatch.setitem(check_endpoints.CHECKS, "llm", ("假生成", lambda: {"ok": True, "detail": "ok"}))
    assert check_endpoints.main(["--only", "llm", "--json"]) == 0
    payload = check_endpoints.json.loads(capsys.readouterr().out)
    assert payload == [
        {"name": "llm", "label": "假生成", "ok": True, "detail": "ok", "seconds": payload[0]["seconds"]}
    ]
