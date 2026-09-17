"""有界等待工具：**不让一次卡死的依赖调用拖垮整个请求**。

为什么单独成模块：本仓库有三处需要它，各自抄一份必然漂移
（`pipeline/fusion.py` 的 SQL 取证、`app/main.py` 的 /api/health 探针、将来别的外部调用）。

典型场景（本机真实踩过）：沙箱里 libpq 的 GSS/SSPI 协商会**永久卡住**，
`connect_timeout` 对它无效 —— 没有这道闸时，问一句聚合问题会把整条融合线路挂满十分钟，
而 /api/health 这种"本该秒回"的接口会直接挂死（比返回 unhealthy 更糟：健康检查自己不可用）。
"""

from __future__ import annotations

import threading


def call_with_timeout(fn, seconds: float):
    """在有界时间内跑 fn；超时返回 (False, None)，异常返回 (False, exc)，成功 (True, 值)。

    实现是线程 + `join(timeout)`：`fn` 卡在系统调用里时无法被安全中断，
    所以超时后那个线程会继续挂着（daemon=True，不阻止进程退出）——
    这是刻意的取舍：要的是"这次请求还能返回"，而不是"杀掉那个卡住的调用"。

    为什么不用 signal.alarm / asyncio.wait_for：前者在 Windows 上不可用，
    后者要求被等待的是协程，而这些阻塞调用（psycopg / pymilvus / neo4j）都是同步的。
    """
    box: dict = {}

    def _worker():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 原样带回给调用方判断
            box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=seconds)
    if thread.is_alive():
        return False, None
    if "error" in box:
        return False, box["error"]
    return True, box.get("value")
