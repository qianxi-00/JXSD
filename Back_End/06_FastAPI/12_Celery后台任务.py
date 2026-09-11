"""FastAPI + Celery + Redis 后台任务
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（Celery + Redis 后台任务）

**为什么需要后台任务？** 有些操作很慢（发邮件、生成报表、处理图片、调用第三方接口），
如果在 HTTP 请求里同步做完，用户要一直转圈，接口还会超时。
正确做法是把任务"丢进队列"，立刻返回一个任务 ID，用户之后拿 ID 查进度。

**架构（课案的四个角色）**：
    客户端 → FastAPI（接收请求，把任务丢队列，立即返回任务ID）
                     ↓ 投递
                  Redis（消息队列 broker + 结果后端 backend）
                     ↓ 竞争消费
              Celery Worker（真正干活的进程，可以部署 N 台）

**本机没有 Redis 怎么办？**（本环境实测 127.0.0.1:6379 连接失败）
    把 Celery 配成 **task_always_eager = True**（eager 模式 / 饥饿模式）：
    调用 `.delay()` / `.apply_async()` 时**不投递到 broker**，而是在当前进程里
    立即同步执行并返回一个 EagerResult。这样：
        · 代码写法完全一样（delay / apply_async / chain / 查结果都能跑通）
        · 不需要 Redis、不需要 worker、不会阻塞等待连接
        · 代价是"失去了异步"，仅用于开发/教学/单测

本节知识点：
    1. Celery 应用配置：broker / backend / 序列化 / 时区
    2. @app.task 定义任务；任务必须能被序列化（参数尽量用 JSON 友好类型）
    3. delay()（简写）与 apply_async()（可带 countdown/queue/retry 等参数）
    4. 任务结果：task.id / task.status / task.get() / AsyncResult
    5. chain 链式任务：上一个任务的返回值传给下一个
    6. eager 模式的启用方式与适用边界
    7. 没有 Redis 时的降级：直接调用任务函数（task(...) 就是普通函数调用）
    8. 生产部署：worker 启动命令、多机扩展、幂等与重试

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\12_Celery后台任务.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\12_Celery后台任务.py'
"""

from __future__ import annotations

import pathlib
import socket
import sys
import time

import uvicorn
from celery import Celery, chain
from celery.result import AsyncResult
from fastapi import FastAPI, Query

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8112

# ============================================================
# 一、Celery 配置（对应课案的 celery_config.py）
# ============================================================
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0

# broker_url：消息代理（任务队列），存放"待执行的任务"
# result_backend：结果后端，存放"任务执行完的返回值"
# 数据库编号 0-15，可用不同编号把队列与结果分开存放
BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
RESULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

celery_app = Celery(
    "backend_demo_tasks",       # 应用名，会显示在 worker 日志与监控里
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    # include=["celery_tasks"]  # 生产上把任务放在独立模块 celery_tasks.py 里，用 include 自动导入
)

celery_app.conf.update(
    # ---------- 序列化 ----------
    task_serializer="json",         # 任务参数序列化格式，json 跨语言兼容、最安全
    accept_content=["json"],        # 只接受 json，防止 pickle 反序列化漏洞
    result_serializer="json",       # 返回值序列化格式
    # ---------- 时间 ----------
    timezone="Asia/Shanghai",       # 时区，影响定时任务（beat）的触发时间
    enable_utc=True,                # 内部统一用 UTC 存储，避免夏令时问题
    # ---------- 多机部署关键配置（默认注释掉，生产按需开启） ----------
    # task_acks_late=True,          # 任务执行完才确认，worker 崩溃时任务重回队列
    # worker_prefetch_multiplier=1, # 每次只取 1 个任务，避免任务堆积在一台机器上
    # ---------- 本机没有 Redis 时的关键配置 ----------
    task_always_eager=True,         # eager 模式：.delay() 不投递 broker，直接在当前进程同步执行
    task_eager_propagates=False,    # eager 下任务抛异常时不向上冒泡（方便演示失败分支）
    task_store_eager_result=False,  # 不把 eager 结果写进 backend（否则会尝试连 Redis）
)

# 本地任务登记表：eager 模式下没有 backend，我们自己记一份，模拟"按任务 ID 查结果"
TASK_REGISTRY: dict[str, dict] = {}


def redis_available(host: str = REDIS_HOST, port: int = REDIS_PORT, timeout: float = 0.5) -> bool:
    """探测 Redis 是否可连接。

    用标准库 socket 直接探测端口，比实例化 redis 客户端更轻量，
    也避免客户端在连接失败时打印一堆重试日志。
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def record_result(task) -> None:
    """把任务结果记到本地登记表（eager 模式下 backend 不可用，用它来查状态）。"""
    TASK_REGISTRY[task.id] = {
        "task_id": task.id,
        "name": getattr(task, "name", "unknown"),
        "status": "SUCCESS" if task.successful() else "FAILURE",
        "result": task.result if task.successful() else str(task.result),
    }


# ============================================================
# 二、定义任务
# ============================================================
@celery_app.task(name="tasks.send_email")
def send_email_task(email: str, content: str) -> dict:
    """模拟发送邮件（耗时操作）。

    任务函数的写法要求：
        · 参数与返回值必须是可 JSON 序列化的（dict / list / str / 数字）；
        · 不要传 ORM 对象、文件句柄、连接对象 —— 它们无法跨进程传输；
        · **任务必须幂等**：worker 可能因为超时/崩溃重试，同一个任务被执行两次
          （例如"给账户加 100 元"要改成"把订单标记为已发放"）。
    """
    print(f"      [任务执行中] 正在发送邮件到：{email}")
    time.sleep(0.05)                     # 模拟真实的网络耗时（课案里是 5 秒）
    print(f"      [任务执行中] 邮件已发送：{content}")
    return {"status": "success", "email": email, "content": content}


@celery_app.task(name="tasks.generate_report")
def generate_report_task(rows: int) -> dict:
    """模拟生成报表（CPU/IO 混合的耗时任务）。"""
    print(f"      [任务执行中] 开始生成报表，数据量 {rows} 行")
    total = sum(i * 2 for i in range(rows))
    return {"status": "success", "rows": rows, "total": total, "report_url": f"/reports/{rows}.xlsx"}


@celery_app.task(name="tasks.add")
def add_task(x: int, y: int) -> int:
    """最简单的任务：用于演示链式调用（chain）。"""
    return x + y


@celery_app.task(name="tasks.maybe_fail")
def maybe_fail_task(should_fail: bool = True) -> str:
    """会失败的任务，用来演示"失败状态如何查询"。

    eager 模式下我们把 task_eager_propagates 设为 False，
    所以异常会被 Celery 捕获并记录为 FAILURE 状态，而不是直接抛给调用方。
    """
    if should_fail:
        raise ValueError("模拟任务执行失败：下游服务不可用")
    return "任务成功"


# ============================================================
# 三、FastAPI 应用
# ============================================================
app = FastAPI(
    title="12 Celery 后台任务",
    description="后端开发基础 · FastAPI 示例：Celery + Redis（本机无 Redis 时用 eager 模式）",
    version="1.0.0",
)


def submit_email(email: str, content: str) -> dict:
    """提交邮件任务的统一入口：有 Redis 就投递队列，没有就本地直调（降级）。

    这一段是"零报错跑完"的关键：
        · 正常情况下 `.delay()` 把任务丢进队列，接口立刻返回任务 ID；
        · 本机没有 Redis 时（eager 模式），`.delay()` 会在当前进程直接执行；
        · 万一 Celery 因为环境问题抛异常，我们捕获后改成"直接调用任务函数"，
          业务功能不中断，只在响应里标注降级。
    """
    try:
        task = send_email_task.delay(email, content)      # 投递（eager 下为同步执行）
        record_result(task)
        return {
            "task_id": task.id,
            "status": task.status,
            "mode": "eager（本机未连接 Redis，任务在当前进程同步执行）"
            if celery_app.conf.task_always_eager else "broker（已投递到 Redis 队列）",
            "result": task.result if task.successful() else None,
        }
    except Exception as exc:                              # 极端情况下的兜底降级
        # 直接调用任务函数（对 Celery 任务来说，task(...) 就是执行函数体）
        result = send_email_task(email, content)
        return {
            "task_id": f"local-{int(time.time() * 1000)}",
            "status": "SUCCESS",
            "mode": f"降级为本地直接调用（原因：{type(exc).__name__}）",
            "result": result,
        }


@app.post("/send-email", summary="触发发送邮件任务，立即返回任务 ID", tags=["任务"])
async def send_email(
    email: str = Query(..., description="收件人邮箱"),
    content: str = Query(..., description="邮件内容"),
) -> dict:
    """课案原型的接口：`send_email_task.delay(email, content)` 然后返回任务 ID。"""
    return submit_email(email, content)


@app.get("/task-status/{task_id}", summary="按任务 ID 查询状态与结果", tags=["任务"])
async def get_task_status(task_id: str) -> dict:
    """查询任务状态。

    真实环境（有 Redis）里这样查：
        task_result = AsyncResult(task_id, app=celery_app)
        return {"status": task_result.status, "result": task_result.result}
    状态取值：PENDING（等待中）/ STARTED（执行中）/ SUCCESS / FAILURE / RETRY

    本机没有 Redis 时，AsyncResult 需要连 backend 才能读结果，
    所以优先查我们自己的 TASK_REGISTRY（eager 模式下的本地登记表）。
    """
    if task_id in TASK_REGISTRY:
        info = dict(TASK_REGISTRY[task_id])
        info["来源"] = "本地登记表（eager 模式）"
        return info

    if not redis_available():
        return {
            "task_id": task_id,
            "status": "UNKNOWN",
            "result": None,
            "说明": "本机未启动 Redis，无法从结果后端读取该任务；"
                    "eager 模式下的任务请用创建时返回的 task_id 查询（本进程内有效）",
        }

    # 有 Redis：走标准的 AsyncResult 查询路径
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.result if task_result.ready() else None,
            "来源": "Celery 结果后端（Redis）",
        }
    except Exception as exc:
        return {"task_id": task_id, "status": "UNKNOWN", "result": None,
                "说明": f"查询失败：{type(exc).__name__}（不影响其他功能）"}


@app.post("/report", summary="触发报表生成任务", tags=["任务"])
async def create_report(rows: int = Query(1000, ge=1, le=100000)) -> dict:
    task = generate_report_task.delay(rows)
    record_result(task)
    return {"task_id": task.id, "status": task.status, "result": task.result}


@app.get("/health", summary="健康检查（含 Redis 可用性）", tags=["运维"])
async def health() -> dict:
    """健康检查里带上依赖服务的状态，方便运维快速定位。

    注意：依赖不可用时是否应该返回"不健康"，要按业务决定 ——
    如果 Redis 只是可选加速，就返回 ok + 提示；如果它是必需依赖，则应返回 503。
    """
    redis_ok = redis_available()
    return {
        "status": "ok",
        "redis": {"host": f"{REDIS_HOST}:{REDIS_PORT}", "available": redis_ok,
                  "提示": "可用" if redis_ok else "未启动：Celery 处于 eager 模式，代码仍可正常运行"},
        "celery": {"task_always_eager": celery_app.conf.task_always_eager,
                   "broker": BROKER_URL},
    }


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    from fastapi.testclient import TestClient

    print("=" * 72)
    print("FastAPI + Celery 后台任务 · 自检")
    print("=" * 72)
    print(f"broker / backend：{BROKER_URL}")
    redis_ok = redis_available()
    print(f"Redis 可用性探测（socket 连接 {REDIS_HOST}:{REDIS_PORT}）：{'可用' if redis_ok else '不可用'}")
    if redis_ok:
        print("  说明：检测到 Redis 已启动。不过本示例为了可复现，仍以 eager 模式运行。")
    else:
        print("  说明：本机没有启动 Redis —— 这是本环境的正常状态。")
        print("        Celery 已配置 task_always_eager=True，任务会在当前进程同步执行，")
        print("        因此下面的 delay()/apply_async()/chain 全部可以正常跑通，且不会阻塞等待。")
    print()

    # ---------- 1. delay() ----------
    print("[1] task.delay()：最简单的触发方式")
    started = time.perf_counter()
    task = send_email_task.delay("alice@example.com", "您的订单已发货")
    cost = time.perf_counter() - started
    print(f"    task.id     = {task.id}")
    print(f"    task.status = {task.status}")
    print(f"    耗时        = {cost * 1000:.1f} ms（eager 模式下等待任务执行完）")
    print(f"    task.get()  = {task.get()}")
    assert task.successful(), "eager 模式下任务应当已成功"
    assert task.result["status"] == "success"
    print("    ✓ delay() 的写法与真实投递队列时完全一致，切换环境不用改代码")

    # ---------- 2. apply_async() ----------
    print("\n[2] task.apply_async()：可以带更多参数")
    task2 = send_email_task.apply_async(
        args=("bob@example.com", "月度账单已生成"),
        kwargs={},
        countdown=0,            # 延迟 N 秒后执行（eager 模式下会忽略）
        expires=3600,           # 任务过期时间
        queue="email",          # 指定队列（生产上可按队列分配不同 worker）
    )
    print(f"    task.id     = {task2.id}")
    print(f"    task.status = {task2.status}")
    print(f"    task.get()  = {task2.get()}")
    assert task2.successful()

    # ---------- 3. 结果登记与查询 ----------
    print("\n[3] 任务状态查询")
    record_result(task)
    resp = TestClient(app).get(f"/task-status/{task.id}")
    print(f"    GET /task-status/{task.id}")
    print(f"    响应：{resp.json()}")
    assert resp.status_code == 200 and resp.json()["status"] == "SUCCESS"

    client = TestClient(app)
    resp = client.get("/task-status/00000000-0000-0000-0000-000000000000")
    print(f"    查询不存在的 task_id：{resp.json()['status']}（不会报错，返回 UNKNOWN）")
    assert resp.status_code == 200

    # ---------- 4. 失败任务 ----------
    print("\n[4] 任务失败的处理")
    print("    说明：eager 模式下若不想让异常冒泡，需设 task_eager_propagates=False（本示例已设）")
    failed = maybe_fail_task.delay(True)
    print(f"    task.status = {failed.status}")
    print(f"    task.result = {failed.result}（类型：{type(failed.result).__name__}）")
    record_result(failed)
    resp = client.get(f"/task-status/{failed.id}")
    print(f"    查询接口返回：{resp.json()}")
    print("    ✓ 失败被记录为 FAILURE 状态，生产上应配置重试与告警")
    print("      重试写法：@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)")

    ok = maybe_fail_task.delay(False)
    print(f"    传 should_fail=False：status={ok.status}，result={ok.result}")
    assert ok.successful()

    # ---------- 5. 链式任务 ----------
    print("\n[5] 链式任务 chain：上一个任务的返回值传给下一个")
    # s() 是 signature（签名）的简写，描述"要调用哪个任务、带什么参数"，但不执行
    pipeline = chain(add_task.s(1, 2), add_task.s(10), add_task.s(100))
    print(f"    任务链：add(1,2) → add(结果,10) → add(结果,100)")
    chain_result = pipeline.apply_async()
    print(f"    最终结果：{chain_result.get()}（(1+2)+10+100 = 113）")
    assert chain_result.get() == 113
    print("    ✓ chain 在 eager 模式下同样可以跑通，生产上它是一条串行执行的任务流水线")
    print("      其他常用组合：group（并行）、chord（并行后汇总）、chain(group) 混合编排")

    # ---------- 6. 降级分支 ----------
    print("\n[6] 没有 Redis 时的降级分支")
    print("    ① Celery 层降级：task_always_eager=True → .delay() 就地执行（本节示例）")
    print("    ② 应用层降级：直接调用任务函数（不经过 Celery 任何机制）")
    direct = send_email_task("carol@example.com", "直接调用任务函数")
    print(f"       send_email_task('carol@example.com', ...) = {direct}")
    assert direct["status"] == "success"
    print("       —— 对 Celery 任务来说，task(...) 就是执行函数体，这是最好用的兜底")
    print("    ③ 生产上的正确做法：Redis 挂掉时应该让接口返回 503 或写入本地队列表，")
    print("       而不是把重任务塞进请求线程里同步跑（会拖垮 Web 进程）")

    # ---------- 7. 接口自检 ----------
    print("\n[7] FastAPI 接口自检")
    resp = client.post("/send-email", params={"email": "dave@example.com", "content": "欢迎注册"})
    print(f"    POST /send-email  状态码 {resp.status_code}")
    body = resp.json()
    print(f"    响应：task_id={body['task_id'][:20]}…  mode={body['mode']}")
    assert resp.status_code == 200 and body["status"] == "SUCCESS"

    resp = client.get("/health")
    print(f"\n    GET /health  状态码 {resp.status_code}")
    print(f"    响应：{resp.json()}")
    assert resp.status_code == 200

    # ---------- 8. 生产部署说明 ----------
    print("\n[8] 生产环境怎么跑（本机没有 Redis，以下命令仅作示例，请勿在无 Redis 时执行）]")
    print("    # 1) 先启动 Redis（Docker 一条命令）")
    print("    docker run -d --name redis -p 6379:6379 redis:7-alpine")
    print("    # 2) 把 task_always_eager 改成 False，然后启动 FastAPI 与 worker")
    print("    uvicorn 12_Celery后台任务:app --host 0.0.0.0 --port 8112")
    print("    celery -A 12_Celery后台任务.celery_app worker --loglevel=info --pool=threads")
    print("    # 一份代码部署到 N 台机器，每台执行同一个 worker 命令，")
    print("    # 它们会自动从 Redis 竞争任务执行 —— 这就是「水平扩展」")
    print("    # 3) 定时任务再加一个 beat：celery -A ... beat --loglevel=info")

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. Celery 三件套：broker（队列）+ backend（结果）+ worker（执行者）")
    print("2. delay() 是 apply_async() 的简写；复杂参数（队列/延迟/过期）用后者")
    print("3. 任务必须可 JSON 序列化、必须幂等（worker 可能重试）")
    print("4. chain 串行、group 并行、chord 汇总，是任务编排的三块积木")
    print("5. 本机无 Redis 时用 task_always_eager=True，代码写法不变、绝不阻塞")
    print("6. 重任务不要塞进 Web 进程同步跑；依赖故障要有明确的降级策略")


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI + Celery + Redis 后台任务（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"计划监听端口：{PORT}")

    if "--check" in sys.argv:
        print()
        print("提示：--check 模式，执行自检，不启动服务器、不阻塞。")
        run_self_check()
    else:
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}/docs")
        print("注意：当前为 eager 模式（本机未启动 Redis），任务会在请求里同步执行完成。")
        print("要变成真正的异步：启动 Redis → 把 task_always_eager 改为 False → 另外起一个 celery worker。")
        print("按 Ctrl+C 停止。")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
