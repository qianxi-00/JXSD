"""RAG 测试公共引导：把 Python_Base 根目录与 RAG 根目录挂进 sys.path。

- Python_Base 根：提供根目录唯一配置入口 config.py（读根目录 .env）
- RAG 根：提供 core / llm / retrieval / pipeline 等包
"""

import sys
from pathlib import Path

_PYTHON_BASE = Path(__file__).resolve().parent.parent.parent  # Python_Base
_RAG = _PYTHON_BASE / "RAG"

for _p in (str(_PYTHON_BASE), str(_RAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
