# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""将预设问答对及其问题向量写入 Redis 缓存

用法: uv run python -m data_process.seed_qa_cache [--force]
"""

import sys
from pathlib import Path

try:
    from core.cache import AnswerCache
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.cache import AnswerCache


def main() -> None:
    force = "--force" in sys.argv
    cache = AnswerCache()
    count = cache.seed_preset(force=force)
    if count:
        print(f"seeded {count} preset QA pairs into redis")
    else:
        print("preset QA pairs already exist in redis (use --force to refresh)")


if __name__ == "__main__":
    main()
