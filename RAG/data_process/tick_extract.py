# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""从 data/ocr_results_clean.json 提取三类票据字段,并可批量写入 Milvus tick 集合

用法:
  uv run python -m data_process.tick_extract                 # 干跑:统计 + 样本报告(data/tick_extract_report.json)
  uv run python -m data_process.tick_extract --type invoice --limit 5
  uv run python -m data_process.tick_extract --insert        # 建 tick 集合并全量入库
  uv run python -m data_process.tick_extract --insert --recreate
  uv run python -m data_process.tick_extract --save data/tick_extracted.jsonl
"""

import argparse
import hashlib
import json
import re
import sys
import time
from decimal import Decimal
from pathlib import Path

try:
    from config import settings
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OCR_JSON_PATH = PROJECT_ROOT / "data" / "ocr_results_clean.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "tick_extract_report.json"

FIELDS = ("ticket_no", "person", "date_int", "amount_fen", "route", "counterparty")
CN_NAME = r"[\u4e00-\u9fa5]{2,4}"
AIRPORT_RE = r"[A-Z][A-Z ]*JICHANG"
TRAIN_DATE_RE = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
TRAIN_MONEY_RE = r"[￥¥]\s*([0-9,]+(?:\.[0-9]+)?)\s*元"
INVOICE_AMOUNT_RE = r"总金额\s*([0-9,]+\.[0-9]{2})"


def ticket_id(source_file: str) -> str:
    """根据 source_file 的 SHA-256 前 24 位生成稳定 ID"""
    return "ticket_" + hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:24]


def yuan_to_fen(value: str) -> int:
    return int((Decimal(value.replace(",", "")) * 100).to_integral_value())


def clean_semantic(text: str) -> str:
    """整理检索文本:去掉 HTML 标签、换行、多余空白等影响理解的符号"""
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"[\r\n\t\u3000]+", " ", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


def is_person_name(s: str) -> bool:
    return bool(re.fullmatch(CN_NAME, s)) and not s.endswith(("机场", "站"))


def extract_flight(text: str) -> dict:
    """登机牌:只有 Jan01 无年份与金额,date/amount 恒为 null"""
    out = dict.fromkeys(FIELDS)
    if not text:
        return out

    m = re.search(r"ETKT\s*(\d{6,})", text)
    if m:
        out["ticket_no"] = m.group(1)

    lines = [ln.strip() for ln in text.split("\n")]
    for i, line in enumerate(lines):
        if "姓名" not in line:
            continue
        rest = line.split(":", 1)[1].strip() if ":" in line else ""
        cand = None
        if rest:
            if re.fullmatch(r"[A-Z][A-Z ]{1,15}", rest):
                if i + 1 < len(lines) and is_person_name(lines[i + 1]):
                    cand = lines[i + 1]
            elif is_person_name(rest):
                cand = rest
        elif (
            i + 2 < len(lines)
            and re.fullmatch(r"[A-Z][A-Z ]{1,15}", lines[i + 1])
            and is_person_name(lines[i + 2])
        ):
            cand = lines[i + 2]
        if cand:
            out["person"] = cand
            break
    if out["person"] is None:
        for m in re.finditer(rf"^([A-Z][A-Z ]{{1,15}})\n({CN_NAME})$", text, re.M):
            if is_person_name(m.group(2)):
                out["person"] = m.group(2)
                break

    from_m = re.search(rf"自\s*From[^\n]*?({AIRPORT_RE})", text)
    to_m = re.search(rf"至\s*To[^\n]*?({AIRPORT_RE})", text)
    if from_m and to_m:
        out["route"] = f"{from_m.group(1).replace(' ', '')}-{to_m.group(1).replace(' ', '')}"
    else:
        airports = list(dict.fromkeys(a.replace(" ", "") for a in re.findall(AIRPORT_RE, text)))
        if len(airports) >= 2:
            out["route"] = f"{airports[0]}-{airports[-1]}"

    if re.search(r"中国国[际際]航空公司", text):
        out["counterparty"] = "中国国际航空公司"
    elif re.search(r"AIR CHINA", text):
        out["counterparty"] = "AIR CHINA"
    return out


def _company_after(lines: list[str], label: str):
    for i, line in enumerate(lines):
        if line == label:
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if re.search(r"[\u4e00-\u9fa5]", nxt) and "<" not in nxt and not re.fullmatch(r"[0-9\s]+", nxt):
                    return nxt
            return None
    return None


def extract_invoice(text: str) -> dict:
    """发票:route 固定为空字符串,date 取开票日期,amount 取总金额"""
    out = dict.fromkeys(FIELDS)
    out["route"] = ""
    if not text:
        return out

    m = re.search(r"发票编号[：:]\s*([A-Za-z0-9\-]+)", text)
    if m:
        out["ticket_no"] = m.group(1)

    m = re.search(r"(?:开票日期|开具日期)[：:]\s*(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        y, mo, d = m.groups()
        out["date_int"] = int(y) * 10000 + int(mo) * 100 + int(d)

    m = re.search(INVOICE_AMOUNT_RE, text)
    if m:
        out["amount_fen"] = yuan_to_fen(m.group(1))

    lines = [ln.strip() for ln in text.split("\n")]
    out["counterparty"] = _company_after(lines, "卖方")
    out["person"] = _company_after(lines, "客户")
    return out


def extract_train(text: str) -> dict:
    """火车票:counterparty 固定为 null,person 取证件号后的姓名"""
    out = dict.fromkeys(FIELDS)
    if not text:
        return out

    m = re.search(r"\bT(\d{12,16})\b", text)
    if m:
        out["ticket_no"] = "T" + m.group(1)

    m = re.search(TRAIN_DATE_RE, text)
    if m:
        y, mo, d = m.groups()
        out["date_int"] = int(y) * 10000 + int(mo) * 100 + int(d)

    m = re.search(TRAIN_MONEY_RE, text)
    if m:
        out["amount_fen"] = yuan_to_fen(m.group(1))

    m = re.search(r"[0-9]{6,10}\*{4}[0-9]{3,4}[0-9X]\s*(" + CN_NAME + r")", text)
    if m:
        out["person"] = m.group(1)

    stations = []
    for line in text.split("\n"):
        line = line.strip()
        if re.fullmatch(r"[A-Z][a-z]{2,}", line):
            stations.append(line)
            continue
        m2 = re.search(r"站\s+([A-Z][a-z]{2,})\s*$", line)
        if m2:
            stations.append(m2.group(1))
    stations = list(dict.fromkeys(stations))
    if len(stations) >= 2:
        out["route"] = f"{stations[0]}-{stations[-1]}"
    return out


EXTRACTORS = {"flight": extract_flight, "invoice": extract_invoice, "train": extract_train}


def build_record(item: dict) -> dict:
    source_file = item.get("source_file", "")
    ticket_type = item.get("ticket_type", "")
    text = item.get("ocr_text") or ""
    extract = EXTRACTORS.get(ticket_type, lambda _t: dict.fromkeys(FIELDS))(text)
    return {
        "id": ticket_id(source_file),
        "ticket_type": ticket_type,
        **extract,
        "semantic_text": clean_semantic(text),
        "ocr_text": text,
        "source_file": source_file,
        "vec": None,
    }


def ensure_collection(client, name: str, recreate: bool = False) -> None:
    from pymilvus import DataType

    if client.has_collection(name):
        if not recreate:
            return
        client.drop_collection(name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("ticket_type", DataType.VARCHAR, max_length=16)
    schema.add_field("ticket_no", DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field("person", DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field("date_int", DataType.INT64, nullable=True)
    schema.add_field("amount_fen", DataType.INT64, nullable=True)
    schema.add_field("route", DataType.VARCHAR, max_length=256, nullable=True)
    schema.add_field("counterparty", DataType.VARCHAR, max_length=128, nullable=True)
    schema.add_field("semantic_text", DataType.VARCHAR, max_length=8192)
    schema.add_field("ocr_text", DataType.VARCHAR, max_length=16384)
    schema.add_field("source_file", DataType.VARCHAR, max_length=256)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=1024, nullable=True)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vec", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)


def build_report(records: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = {}
    for r in records:
        by_type.setdefault(r["ticket_type"], []).append(r)
    coverage = {
        t: {f: sum(1 for r in rs if r.get(f) not in (None, "")) for f in FIELDS}
        for t, rs in by_type.items()
    }
    samples = {}
    for t, rs in by_type.items():
        samples[t] = [{**r, "ocr_text": r["ocr_text"][:120], "semantic_text": r["semantic_text"][:120]} for r in rs[:3]]
    return {
        "total": len(records),
        "by_type": {t: len(rs) for t, rs in by_type.items()},
        "coverage": coverage,
        "samples": samples,
    }


def load_items(args) -> list[dict]:
    data = json.loads(OCR_JSON_PATH.read_text(encoding="utf-8"))
    items = data.get("results", [])
    if args.type:
        items = [it for it in items if it.get("ticket_type") == args.type]
    if args.limit:
        items = items[: args.limit]
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="提取票据字段并写入 Milvus tick 集合")
    ap.add_argument("--type", choices=["flight", "invoice", "train"], help="只处理某类票据")
    ap.add_argument("--limit", type=int, help="最多处理多少条,调试用")
    ap.add_argument("--insert", action="store_true", help="写入 Milvus")
    ap.add_argument("--recreate", action="store_true", help="删除并重建 tick 集合")
    ap.add_argument("--save", metavar="PATH", help="把提取结果另存为 jsonl")
    ap.add_argument("--report", metavar="PATH", default=str(DEFAULT_REPORT_PATH))
    args = ap.parse_args()

    records = [build_record(it) for it in load_items(args)]

    report = build_report(records)
    print(f"extracted {report['total']} records: {report['by_type']}")
    for t, cov in report["coverage"].items():
        print(f"  [{t}] " + " ".join(f"{f}={n}" for f, n in cov.items()))
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {args.report}")

    if args.save:
        Path(args.save).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
        )
        print(f"records -> {args.save}")

    if args.insert:
        from core.database import get_milvus_client

        client = get_milvus_client()
        name = settings.milvus.collection
        ensure_collection(client, name, recreate=args.recreate)
        print(f"collection '{name}' ready")

        inserted = 0
        for i in range(0, len(records), 100):
            batch = records[i : i + 100]
            res = client.insert(collection_name=name, data=batch)
            inserted += res.get("insert_count", len(batch))
        print(f"inserted {inserted} rows")

        try:
            client.flush(collection_name=name)
        except Exception:
            pass
        for _ in range(5):
            stats = client.get_collection_stats(collection_name=name)
            if int(stats.get("row_count", 0)) >= len(records):
                break
            time.sleep(2)
        print(f"row_count: {stats.get('row_count')}")

        client.load_collection(collection_name=name)
        sample = client.query(
            collection_name=name,
            filter='ticket_type == "train"',
            limit=2,
            output_fields=["id", "ticket_no", "person", "date_int", "amount_fen", "route", "counterparty"],
        )
        print(f"query sample: {sample}")
        client.close()


if __name__ == "__main__":
    main()
