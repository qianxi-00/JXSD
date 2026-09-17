"""探针：重排 query 用"追问原话"还是"改写后的检索文本"，分数差多少？

背景（真机踩到的多轮追问失败）：即使过滤条件正确（只召回乐艳那 1 张票），
用追问原话「那乐艳的呢？」做重排 query 时得分只有 **0.098 < 0.22**，一条都留不下；
换成改写出的检索文本才过阈值。**结论：要修的是重排 query，不是过滤条件**
（我第一版只修了过滤条件，探针一量就发现没用）。

对同一张票比四种 query 的重排分：
    A. "那乐艳的呢？"（失败时的实际行为）
    B. "乐艳 火车票 票号"（改写出的检索文本）
    C. "乐艳的火车票票号是多少？"（agent 有时会改写成的独立问题）
    D. "那乐艳的呢？ 乐艳 火车票 票号"（`retrieve_evidence` 兜底实际采用的拼接）

用法：.venv\\Scripts\\python.exe RAG\\script\\probe_rerank_query.py
（只读探针：真连 Milvus + reranker 打分，不写任何数据）
"""

import sys
from pathlib import Path

BASE = Path(r"F:\ProGram\Python_Base")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "RAG"))

from retrieval.rerank import rerank  # noqa: E402
from retrieval.vector_retrieval import vector_search  # noqa: E402

EXPR = 'ticket_type == "train" and person == "乐艳"'

rows: dict[str, dict] = {}
for query in ("乐艳 火车票 票号", "乐艳的火车票票号是多少？"):
    for row in vector_search(query, top_n=10, filter_expr=EXPR):
        rows.setdefault(str(row.get("id")), row)
candidates = list(rows.values())
print(f"候选 {len(candidates)} 条（过滤 {EXPR}）")
for row in candidates:
    print(f"  - {row.get('source_file')} 票号={row.get('ticket_no')} 人={row.get('person')}")

for label, query in [
    ("A. 追问原话", "那乐艳的呢？"),
    ("B. 改写检索文本", "乐艳 火车票 票号"),
    ("C. 独立问题", "乐艳的火车票票号是多少？"),
    ("D. 原话+改写拼接", "那乐艳的呢？ 乐艳 火车票 票号"),
]:
    kept = rerank(query, candidates, top_k=5)
    top = max((r.get("rerank_score") or 0) for r in candidates) if candidates else 0
    print(
        f"{label:<16} query={query!r:<26} → 过阈值保留 {len(kept)} 条 "
        f"（候选里最高分需 ≥ 0.22；保留条数即判据）"
    )
