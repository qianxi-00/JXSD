"""探针：不开 PGGSSENCMODE 时，psycopg 连本机 PG 到底会不会卡住？

用**线程 + join 超时**做有界探测：即使底层卡死，探针也会在 15 秒后给出结论，
不会把整个 shell 挂满 600 秒（这正是上一次复跑踩的坑）。

本机实测结论：默认设置下 15 秒不返回（`connect_timeout` 对它无效）；
设 `PGGSSENCMODE=disable` 后 0.01 秒就返回。
⚠️ 这个坑不只影响测试：**服务进程同样中招** —— 融合线路的 Text-to-SQL 那一路会把整条
问答挂满十分钟，所以 `pipeline/fusion.py` 给 SQL 取证加了 20 秒有界降级。

用法：.venv\\Scripts\\python.exe RAG\\script\\probe_pg_gss.py
"""

import os
import threading
import time

RESULT: dict = {}


def connect_probe(label: str, dsn: str) -> None:
    started = time.perf_counter()
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        RESULT[label] = f"OK ({time.perf_counter() - started:.2f}s)"
    except Exception as exc:  # noqa: BLE001 探针要如实记录
        RESULT[label] = f"{type(exc).__name__}: {exc} ({time.perf_counter() - started:.2f}s)"


DSN = "host=127.0.0.1 port=5432 user=postgres dbname=finance_db"

print(f"当前 PGGSSENCMODE = {os.environ.get('PGGSSENCMODE', '<未设置>')}")
for label in ("默认设置（不设 PGGSSENCMODE）",):
    thread = threading.Thread(target=connect_probe, args=(label, DSN), daemon=True)
    thread.start()
    thread.join(timeout=15)
    if thread.is_alive():
        RESULT[label] = "卡死：15 秒内未返回（线程仍在跑，已放弃等待）"
    print(f"  {label}: {RESULT[label]}")

os.environ["PGGSSENCMODE"] = "disable"
label2 = "设 PGGSSENCMODE=disable 后"
thread = threading.Thread(target=connect_probe, args=(label2, DSN), daemon=True)
thread.start()
thread.join(timeout=15)
if thread.is_alive():
    RESULT[label2] = "卡死：15 秒内未返回"
print(f"  {label2}: {RESULT[label2]}")
