"""RAG 测试公共引导：把 Python_Base 根目录与 RAG 根目录挂进 sys.path。

- Python_Base 根：提供根目录唯一配置入口 config.py（读根目录 .env）
- RAG 根：提供 core / llm / retrieval / pipeline 等包
"""

import sys
import tempfile
from pathlib import Path

import pytest

_PYTHON_BASE = Path(__file__).resolve().parent.parent.parent  # Python_Base
_RAG = _PYTHON_BASE / "RAG"

for _p in (str(_PYTHON_BASE), str(_RAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def tmp_dir():
    """临时目录，**替代 pytest 自带的 `tmp_path`**。

    为什么不用 `tmp_path`：本机 `%TEMP%\\pytest-of-QianXi` 被某次沙箱进程创建成了
    "创建它的进程自己能读写、别的进程一律 Access denied"的目录（连删都要提权），
    于是 `tmp_path` 直接抛 `PermissionError: [WinError 5]`，而且**每次新进程都复现**
    （`test_build_ocr_json.py` 的 7 个用例就是这么红的）。
    `tempfile.TemporaryDirectory` 走 `%TEMP%\\rag-test-*` 另一个前缀，实测可写。

    顺带一个好处：临时目录在用例结束即删，不依赖 pytest 的 `--basetemp` 保留策略。
    """
    with tempfile.TemporaryDirectory(prefix="rag-test-") as tmp:
        yield Path(tmp)
