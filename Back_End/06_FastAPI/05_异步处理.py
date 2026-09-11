"""FastAPI 异步处理：async def 与 def 的真实差别
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（异步处理）

先讲原理（这是本课最容易讲错的一块）：
    · uvicorn 启动后是一个 **ASGI 服务器**，内部跑一个 **事件循环（event loop）**，
      默认只有 **1 个 Python 线程** 在处理请求。
    · `async def` 端点：由事件循环直接执行。遇到 `await asyncio.sleep()` 这类
      "可等待的 IO" 时会**让出控制权**，事件循环马上去处理别的请求 ——
      所以单线程也能同时挂着成百上千个请求。
    · `def`（同步）端点：Starlette 认为它可能阻塞，于是把它丢到
      **线程池（threadpool）** 里执行（默认 40 个线程）。
      所以同步端点也是并发的，只是受线程数上限限制，且线程切换有开销。
    · **最危险的情况**：`async def` 里写了阻塞代码（time.sleep、requests.get、
      同步数据库驱动）—— 事件循环被卡住，**所有请求一起变慢**，这是线上最常见的性能事故。

本节用 httpx.ASGITransport 在**进程内**并发调用自己的接口（不启动服务器、不占端口），
通过实测耗时把上面三条结论验证一遍。

本节知识点：
    1. async def 与 def 的执行模型差异（事件循环 vs 线程池）
    2. asyncio.gather 并发发请求，量化对比耗时
    3. 在 async def 里调用阻塞代码的危害（第三个端点专门演示）
    4. 同步阻塞代码的正确处理：run_in_threadpool / anyio.to_thread
    5. 异步依赖与 BackgroundTasks（响应返回后继续跑任务）

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\05_异步处理.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\05_异步处理.py'
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time

import httpx
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, Query
from fastapi.concurrency import run_in_threadpool

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8105

app = FastAPI(
    title="05 异步处理",
    description="后端开发基础 · FastAPI 示例：async/await 与并发模型",
    version="1.0.0",
)

# 每次请求"干活"的时间。课案用的是 10 秒，这里缩到 0.1 秒，
# 是为了让自检在 1 秒内跑完；并发比例与课案完全一致。
WORK_SECONDS = 0.1


# ============================================================
# 一、三种端点：异步 / 同步 / 异步里写阻塞
# ============================================================
@app.get("/async", summary="异步端点（正确写法）", tags=["并发模型"])
async def async_endpoint() -> dict:
    """async def + await asyncio.sleep()：**非阻塞**。

    await 的语义是"我暂时不用 CPU，先把控制权还给事件循环"，
    事件循环立刻去处理别的请求；等 sleep 结束再回来继续。
    因此单线程可以同时"挂着"成千上万个这样的请求。
    """
    await asyncio.sleep(WORK_SECONDS)
    return {"mode": "async", "waited": WORK_SECONDS, "说明": "await 期间事件循环可以处理其他请求"}


@app.get("/sync", summary="同步端点（框架自动放线程池）", tags=["并发模型"])
def sync_endpoint() -> dict:
    """def（同步）+ time.sleep()：**也是并发的**，但靠的是线程池。

    Starlette 检测到同步端点会调用 anyio.to_thread.run_sync，
    把它丢进线程池（默认 40 个线程）。所以：
        - 并发数受线程数限制（40 以上会排队）；
        - 每个请求占用一个线程，内存与切换开销都比协程大。
    类比：同步是"多招几个员工"，异步是"一个员工同时接多通电话"。
    """
    time.sleep(WORK_SECONDS)
    return {"mode": "sync", "waited": WORK_SECONDS, "说明": "由线程池执行，并发上限约等于线程数"}


@app.get("/async-blocking", summary="反面教材：async def 里写阻塞代码", tags=["并发模型"])
async def async_blocking_endpoint() -> dict:
    """**错误示范**：async def 里用 time.sleep()。

    time.sleep() 是同步阻塞调用，它不会让出控制权，
    于是整个事件循环（也就是整个服务）都被卡住 ——
    其他请求只能排队等待，吞吐量直接退化成串行。

    生产中最常见的三种"偷偷阻塞"：
        1) time.sleep()            → 换成 await asyncio.sleep()
        2) requests.get()          → 换成 httpx.AsyncClient
        3) 同步数据库驱动/大文件读写 → 换成 async 驱动，或用 run_in_threadpool 包一层
    """
    time.sleep(WORK_SECONDS)     # ← 故意写错，用来量化危害
    return {"mode": "async-but-blocking", "waited": WORK_SECONDS,
            "说明": "这段代码会阻塞整个事件循环，其他请求被迫排队"}


@app.get("/threadpool", summary="异步端点里安全地跑阻塞代码", tags=["并发模型"])
async def threadpool_endpoint() -> dict:
    """正确做法：用 run_in_threadpool 把阻塞调用丢到线程池。

    `await run_in_threadpool(阻塞函数, 参数...)` 会在别的线程里执行阻塞代码，
    事件循环继续服务其他请求。这是"必须用同步库"时的标准兜底方案。
    """
    started = time.perf_counter()
    await run_in_threadpool(time.sleep, WORK_SECONDS)      # 阻塞代码在线程池里跑
    return {"mode": "async + threadpool", "waited": round(time.perf_counter() - started, 3),
            "说明": "阻塞代码被移到线程池，事件循环仍然畅通"}


# ============================================================
# 二、异步依赖
# ============================================================
async def get_request_id() -> str:
    """异步依赖：可以 await，例如查 Redis、读异步数据库。

    FastAPI 会正确地 await 它，不会阻塞事件循环。
    """
    await asyncio.sleep(0)          # 模拟一次异步 IO（sleep(0) 只让出一次控制权）
    return f"req-{int(time.time() * 1000) % 100000}"


@app.get("/with-dep", summary="使用异步依赖", tags=["并发模型"])
async def with_dep(request_id: str = Depends(get_request_id)) -> dict:
    return {"request_id": request_id, "说明": "依赖函数是 async def，FastAPI 会 await 它"}


# ============================================================
# 三、BackgroundTasks：响应返回后再执行
# ============================================================
BACKGROUND_LOG: list[str] = []


def write_audit_log(message: str) -> None:
    """同步函数也能作为后台任务（FastAPI 会在线程池里跑它）。"""
    BACKGROUND_LOG.append(f"{time.strftime('%H:%M:%S')} {message}")


async def send_notification(message: str) -> None:
    """异步函数作为后台任务（FastAPI 直接 await 它）。"""
    await asyncio.sleep(0)
    BACKGROUND_LOG.append(f"{time.strftime('%H:%M:%S')} 通知已发送：{message}")


@app.post("/orders", summary="下单：立即返回，后台慢慢记账", tags=["后台任务"])
async def create_order(
    background: BackgroundTasks,
    item: str = Query(..., description="商品名"),
) -> dict:
    """BackgroundTasks 的语义：**先把响应发出去，再执行这些任务**。

    适合"轻量、不需重试"的收尾工作（写审计日志、发站内通知）。
    重活、需要重试/可观测的任务应该交给 Celery（见 12_Celery后台任务.py）。

    对比 Flask/Django 的"响应后钩子"：FastAPI 把它做成了一个显式依赖，一目了然。
    """
    background.add_task(write_audit_log, f"创建订单：{item}")
    background.add_task(send_notification, f"订单 {item} 已创建")
    return {"order": item, "status": "created", "说明": "后台任务已在响应之后执行"}


# ============================================================
# 自检：进程内并发压测，量化对比
# ============================================================
async def _request_many(client: httpx.AsyncClient, path: str, count: int) -> tuple[float, int]:
    """并发发起 count 个请求，返回 (总耗时, 成功数)。"""
    started = time.perf_counter()
    tasks = [client.get(path) for _ in range(count)]
    results = await asyncio.gather(*tasks)
    cost = time.perf_counter() - started
    ok = sum(1 for r in results if r.status_code == 200)
    return cost, ok


async def _benchmark() -> None:
    # ASGITransport：直接把请求交给 app 对象处理，不走网络、不启服务器。
    # 这样我们测到的就是"应用的并发处理能力"，而不是网络开销。
    transport = httpx.ASGITransport(app=app)
    concurrency = 20
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 每个端点先预热一次，避免首次导入开销影响数据
        for path in ("/async", "/sync", "/async-blocking", "/threadpool"):
            await client.get(path)

        print("=" * 72)
        print(f"并发实测：{concurrency} 个请求同时发出，单次工作耗时 {WORK_SECONDS}s")
        print("=" * 72)
        print(f"{'端点':<20}{'总耗时(s)':>12}{'理想串行(s)':>14}{'成功数':>8}   结论")
        print("-" * 72)

        ideal = WORK_SECONDS * concurrency
        results: dict[str, float] = {}
        rows = [
            ("/async", "await 让出控制权 → 全部并发，耗时≈单次"),
            ("/sync", "线程池并发 → 也很快，但受线程数限制"),
            ("/async-blocking", "阻塞事件循环 → 退化成串行，耗时≈20 倍"),
            ("/threadpool", "阻塞代码移到线程池 → 恢复并发"),
        ]
        for path, conclusion in rows:
            cost, ok = await _request_many(client, path, concurrency)
            results[path] = cost
            print(f"{path:<20}{cost:>12.3f}{ideal:>14.3f}{ok:>8}   {conclusion}")

        print("-" * 72)
        print("解读（这是本节的核心结论）：")
        print(f"  · /async 只用了 {results['/async']:.3f}s —— 20 个请求几乎同时在等，"
              f"相当于理想串行的 1/{ideal / max(results['/async'], 1e-6):.0f}")
        print(f"  · /sync 用了 {results['/sync']:.3f}s —— 线程池并发，同样很快")
        print(f"  · /async-blocking 用了 {results['/async-blocking']:.3f}s —— "
              f"整整慢 {results['/async-blocking'] / max(results['/async'], 1e-6):.0f} 倍，"
              "因为事件循环被阻塞了")
        print(f"  · /threadpool 用了 {results['/threadpool']:.3f}s —— 把阻塞代码挪走就恢复正常")
        print()

        # ---------- 断言：验证上述结论成立 ----------
        assert results["/async"] < ideal * 0.5, "异步端点应当并发执行"
        assert results["/async-blocking"] > results["/async"] * 3, "阻塞端点应显著更慢"
        assert results["/sync"] < ideal * 0.5, "同步端点由线程池并发执行，也应较快"

        # ---------- 后台任务 ----------
        BACKGROUND_LOG.clear()
        resp = await client.post("/orders", params={"item": "机械键盘"})
        print("[BackgroundTasks 验证]")
        print(f"    POST /orders  状态码 {resp.status_code}，响应体 {resp.text}")
        assert resp.status_code == 200
        # 响应返回后，后台任务已经执行完（TestClient/ASGI 传输会等待后台任务结束）
        print(f"    后台任务执行记录（共 {len(BACKGROUND_LOG)} 条）：")
        for line in BACKGROUND_LOG:
            print(f"      {line}")
        assert len(BACKGROUND_LOG) == 2, "两个后台任务都应当执行"

        # ---------- 异步依赖 ----------
        resp = await client.get("/with-dep")
        print("\n[异步依赖验证]")
        print(f"    GET /with-dep  状态码 {resp.status_code}，响应 {resp.text}")
        assert resp.status_code == 200 and resp.json()["request_id"].startswith("req-")


def run_self_check() -> None:
    print()
    print("#" * 72)
    print("# FastAPI 异步处理 · 自检（httpx.ASGITransport 进程内并发实测）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"计划监听端口：{PORT}")
    print()
    asyncio.run(_benchmark())
    print("=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. async def 由事件循环执行；def 由线程池执行，两者都并发，机制不同")
    print("2. async def 里写 time.sleep / requests / 同步驱动 = 卡死整个服务")
    print("3. 必须用同步库时，用 await run_in_threadpool(阻塞函数, ...) 兜底")
    print("4. 异步依赖（async def + Depends）不会阻塞事件循环")
    print("5. BackgroundTasks 在响应之后执行轻量任务；重任务交给 Celery")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI 异步处理（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")

    if "--check" in sys.argv:
        run_self_check()
    else:
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}/docs")
        print("可以用压测工具对比：")
        print(f"  ab -n 100 -c 20 http://127.0.0.1:{PORT}/async")
        print(f"  ab -n 100 -c 20 http://127.0.0.1:{PORT}/sync")
        print("按 Ctrl+C 停止。")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
