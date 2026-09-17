# =============================================================================
# 向量回填脚本（RAG 基础篇「向量化入库」的第二步）
#
# 为什么要有这个脚本：tick_extract.py 写入 Milvus 时 `vec` 是 None（只写结构化字段和
# 正文），真正的向量是在这里补的。分成两步的原因有两条：
#   1. 建库/重建时不需要调 embedding 接口，先把数据落库、字段抽取结果肉眼可查；
#   2. 换 embedding 模型（本项目是 BAAI/bge-m3，1024 维）时只需重跑本脚本，
#      不必重新 OCR、重新抽字段——这也是"换模型要重算向量库"那条清单里的动作。
#
# 幂等性：用 upsert（主键 id 相同就覆盖）而不是 insert，所以可以反复重跑；
# 重跑会覆盖 vec 字段而不会产生重复行。
# =============================================================================

# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
# 必须位于业务 import 之前：直接以脚本路径运行时 sys.path[0] 是 RAG/data_process，
# 看不到仓库根的 config 与 RAG 下的 core / retrieval 包。
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

# 回填写 upsert 时要带上的**全部标量字段**：Milvus 的 upsert 语义是"整实体覆盖"，
# 不是"部分更新"。只传 id + vec 会把其余字段一起抹成默认值（person/date_int 变空），
# 检索结果里就会出现一堆无法过滤也无法展示的空条目。所以这里把建表时的 11 个标量字段
# 全部列出——有意**不含** vec 本身（vec 是本次要写的结果，从库里读回来没有意义）。
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
    # 集合名来自配置（settings.milvus.collection），不写死：换集合/开新实验库时只改 .env。
    name = settings.milvus.collection

    # 一次性把待回填的行全捞出来，在进程内分批处理。
    # 坑 1：这里靠 limit=10000 兜底（Milvus 的 query 自身也有个上限，量级在 16384，
    #   以所用版本的文档为准）。数据量超过 1 万行时会**静默截断**——不报错，
    #   只是后半部分永远没有向量；届时必须改成分页或按主键迭代查询。
    # 坑 2：过滤条件写死三种票据类型，等于假设库里只有这三类。tick_extract 按 ticket_type
    #   落库时并不会校验这一点，若出现第四类，那些行不会进 todo，vec 一直是 NULL，
    #   向量检索也就永远搜不到它们（结构化 SQL 查得到、语义检索查不到，这种不一致最难查）。
    rows = client.query(
        collection_name=name,
        filter='ticket_type in ["flight", "invoice", "train"]',
        output_fields=FIELDS,
        limit=10000,
    )
    # semantic_text 才是检索/重排用的文本（rerank.py 发给接口的也是它），所以空文本的行
    # 直接跳过：embedding 空串要么报错要么得到一个无意义的向量，两种都会污染召回结果。
    # 这些被跳过的行 vec 保持 NULL——Milvus 建了 vec 索引后，NULL 向量不参与检索，
    # 相当于该行只在结构化查询里可见。skipped 计数就是为了让这种情况在日志里现形。
    todo = [r for r in rows if (r.get("semantic_text") or "").strip()]
    skipped = len(rows) - len(todo)
    print(f"total {len(rows)} rows, to embed {len(todo)}, skipped(empty semantic_text) {skipped}")

    # 批大小走配置（settings.embedding.batch_size，默认 32）：由网关/接口能接受的最大
    # 批量决定，换供应商时改 .env 即可，不用动代码。
    batch_size = settings.embedding.batch_size
    done = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        # embed_texts 内部会按批调用 embedding 接口，顺序与入参严格一一对应，
        # 因此下面的 zip 是安全的；一旦实现改成并发乱序返回，这里就会张冠李戴。
        vectors = embed_texts([r["semantic_text"] for r in batch])
        for row, vec in zip(batch, vectors):
            row["vec"] = vec
        # 逐批 upsert 而不是攒到最后一次性写：单批失败时前面的批次已经落库，
        # 重跑（幂等）即可补齐，不必从头再来。
        client.upsert(collection_name=name, data=batch)
        done += len(batch)
        print(f"upserted {len(batch)} rows (progress {done}/{len(todo)})")
        # 轻微限速，避免连续打满 embedding 网关触发限流（本机走的是外部 API）。
        time.sleep(0.2)

    print(f"embedding done, {done} rows updated in collection '{name}'")
    # 显式关闭连接。注意这个 print 与 close 都不在 try/finally 里：中途异常时连接不会
    # 被关掉，脚本进程退出时才回收——一次性脚本可以接受，改成常驻服务则需要修。
    client.close()


if __name__ == "__main__":
    main()
