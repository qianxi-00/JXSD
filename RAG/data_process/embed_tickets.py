# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""批量向量化 tick 集合中的 semantic_text,并将向量回填到 Milvus 的 vec 字段

用法: uv run python -m data_process.embed_tickets
"""

import time

from config import settings
from core.database import get_milvus_client
from retrieval.embedding import embed_texts

FIELDS = [
    "id",
    "ticket_type",
    "ticket_no",
    "person",
    "date_int",
    "amount_fen",
    "route",
    "counterparty",
    "semantic_text",
    "ocr_text",
    "source_file",
]


def main() -> None:
    client = get_milvus_client()
    name = settings.milvus.collection

    rows = client.query(
        collection_name=name,
        filter='ticket_type in ["flight", "invoice", "train"]',
        output_fields=FIELDS,
        limit=10000,
    )
    todo = [r for r in rows if (r.get("semantic_text") or "").strip()]
    skipped = len(rows) - len(todo)
    print(f"total {len(rows)} rows, to embed {len(todo)}, skipped(empty semantic_text) {skipped}")

    batch_size = settings.embedding.batch_size
    done = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        vectors = embed_texts([r["semantic_text"] for r in batch])
        for row, vec in zip(batch, vectors):
            row["vec"] = vec
        client.upsert(collection_name=name, data=batch)
        done += len(batch)
        print(f"upserted {len(batch)} rows (progress {done}/{len(todo)})")
        time.sleep(0.2)

    print(f"embedding done, {done} rows updated in collection '{name}'")
    client.close()


if __name__ == "__main__":
    main()
