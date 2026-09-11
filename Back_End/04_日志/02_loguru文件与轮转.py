"""Loguru 日志文件管理与轮转（生产环境必备）
================================================================
对应课案章节：后端开发基础 → 日志（logger.add 文件输出 / rotation 轮转）

本节知识点：
    1. 为什么必须轮转：日志文件只增不减，磁盘迟早被写满（服务器最常见的"莫名宕机"原因之一）
    2. rotation：什么时候切新文件 —— 按大小 "10 MB" / 按时间 "00:00"、"1 week" / 按自定义函数
    3. retention：旧文件保留策略 —— 保留 N 个 / "7 days" / 回调函数删除
    4. compression：切出来的旧文件自动压缩（zip / gz / bz2 / xz），省磁盘
    5. serialize=True：输出 JSON 结构化日志，方便 ELK / Loki / 日志服务采集
    6. filter：不同内容分流到不同文件（如"只把 ERROR 单独存一份"）
    7. backtrace / diagnose：是否打印完整调用栈与变量值（生产要关掉 diagnose，防止泄露数据）
    8. enqueue=True：多线程/多进程安全写入（FastAPI + Gunicorn 多 worker 时必须开）

运行方式（PowerShell，必须用本仓库的虚拟环境解释器）：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\04_日志\\02_loguru文件与轮转.py'
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

from loguru import logger

# ------------------------------------------------------------
# 路径：本文件位于 Back_End/04_日志/02_loguru文件与轮转.py
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
LOG_DIR = BACK_END / "data" / "logs"
ROTATE_DIR = LOG_DIR / "rotation"
ROTATE_DIR.mkdir(parents=True, exist_ok=True)

APP_LOG = ROTATE_DIR / "app.log"                     # 主日志
# 文件名也可以写时间占位符，例如 "app_{time:YYYY-MM-DD}.log"，loguru 会按天生成新文件
ERROR_LOG = ROTATE_DIR / "only_error.log"            # 只收 ERROR 及以上的独立文件
JSON_LOG = ROTATE_DIR / "structured.json"            # JSON 结构化日志

# 演示用的轮转阈值：故意设得很小（10 KB），这样几十行日志就能触发轮转，便于观察
ROTATION_SIZE = "10 KB"


def clean_old_files() -> None:
    """清理上一次运行留下的日志，保证每次演示结果清晰可复现。"""
    for path in ROTATE_DIR.glob("*"):
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass


def build_logger():
    """配置三套文件处理器：主日志（轮转）/ 错误日志（只收 ERROR）/ JSON 日志。"""
    logger.remove()

    # ---------- 1. 主日志：按大小轮转 + 只保留最近 3 个 + 旧文件压缩 ----------
    logger.add(
        APP_LOG,
        level="DEBUG",
        encoding="utf-8",
        rotation=ROTATION_SIZE,     # 文件超过 10 KB 就切一个新人（生产常用 "500 MB" 或 "00:00"）
        retention=3,                # 最多保留 3 个历史文件，更旧的自动删除
        compression="zip",          # 切出去的旧文件自动压成 .zip，磁盘占用能降 80%+
        enqueue=True,               # 多线程/多进程安全
        backtrace=True,             # 记录完整调用栈（排错更快）
        diagnose=False,             # 关闭变量值快照：生产必须 False，否则可能把密码写进日志
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    # ---------- 2. 错误专用日志：只收 ERROR 及以上 ----------
    # filter 也可以写成 lambda record: record["level"].no >= 40
    logger.add(
        ERROR_LOG,
        level="ERROR",
        encoding="utf-8",
        rotation="50 KB",
        retention=5,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )

    # ---------- 3. JSON 结构化日志：一行一个 JSON 对象 ----------
    logger.add(
        JSON_LOG,
        level="INFO",
        encoding="utf-8",
        serialize=True,             # 关键参数：把整条日志记录序列化成 JSON
        format="{message}",         # serialize 时 format 只影响 text 字段
        rotation="100 KB",
        # filter：只把"绑定了 task_id 的日志"写进这个文件。
        # 好处：日志文件不掺噪音，日志平台采集后可以直接按 extra 字段检索。
        filter=lambda record: "task_id" in record["extra"],
    )

    # ---------- 4. 终端仍然保留一份（保持交互友好） ----------
    # level="WARNING"：演示里要写 400 条 INFO，如果终端也收会刷屏；
    #                  真实项目通常设 INFO，生产设 WARNING。
    logger.add(
        sys.stderr,
        level="WARNING",
        colorize=True,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
    )
    return logger


def demo_rotation() -> int:
    """演示 1：写足够多的日志把文件"撑爆"，观察自动轮转。"""
    print("=" * 72)
    print("演示 1：日志轮转（rotation / retention / compression）")
    print("=" * 72)
    print(f"配置：文件超过 {ROTATION_SIZE} 就切新文件，只保留最近 3 个，旧文件自动 zip 压缩")
    print(f"目录：{ROTATE_DIR}\n")

    # 每行大约 130 字节，写 400 行 ≈ 50 KB，足以触发 4 次以上轮转
    message = "这是一条用于触发日志轮转的测试日志，请忽略内容，只关注文件被切开的效果。"
    for i in range(1, 401):
        logger.info(f"第 {i:03d} 条：{message}")
    print("已写入 400 条 INFO 日志。\n")
    return 400


def demo_filter_and_levels() -> None:
    """演示 2：同一个 logger，不同级别进不同文件。"""
    print("=" * 72)
    print("演示 2：分级分流（ERROR 单独一个文件，方便快速定位）")
    print("=" * 72)
    logger.debug("DEBUG 只进主日志，不进 only_error.log")
    logger.info("INFO 只进主日志和 JSON 日志")
    logger.warning("WARNING 也进主日志")
    logger.error("ERROR 会额外写进 only_error.log，运维只需要盯这一个文件")
    print("已写入 1 条 ERROR。\n")


def demo_serialize() -> None:
    """演示 3：JSON 结构化日志，机器友好。"""
    print("=" * 72)
    print("演示 3：serialize=True 输出 JSON 日志（日志平台可以直接索引字段）")
    print("=" * 72)

    task_logger = logger.bind(task_id="task-20240501", step="generate_report")
    task_logger.info("报表生成任务开始")
    task_logger.info("报表生成任务完成")

    # 先把队列里的日志刷盘（enqueue=True 时是异步写的），再读取文件
    logger.complete()

    example = None
    if JSON_LOG.exists():
        for line in JSON_LOG.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            record = obj.get("record", {})
            if record.get("extra", {}).get("task_id") == "task-20240501":
                example = obj
                break

    if example is None:
        print("（暂时没有采到示例 JSON，通常是因为日志还异步在队列里，不影响理解）")
    else:
        print("一条 JSON 日志的字段说明：")
        print(f"  text      = {example.get('text')}")
        print(f"  record.time = {example['record']['time']['repr']}")
        print(f"  record.level = {example['record']['level']['name']}")
        print(f"  record.name/function/line = {example['record']['name']} / "
              f"{example['record']['function']} / {example['record']['line']}")
        print(f"  record.extra = {example['record']['extra']}")
        print("\n  在日志平台里就能直接按 extra.task_id 检索「一次任务的全部日志」。")
    print()


def list_files() -> None:
    """列出轮转目录里的文件，展示轮转 + 压缩 + 清理的结果。"""
    print("=" * 72)
    print("演示 4：轮转结果（文件名、大小、类型）")
    print("=" * 72)
    files = sorted(p for p in ROTATE_DIR.glob("*") if p.is_file())
    if not files:
        print("目录为空。")
        return
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        kind = "压缩包" if path.suffix == ".zip" else "文本"
        print(f"  {path.name:<48}{size:>8} 字节   {kind}")
    print(f"\n  共 {len(files)} 个文件，合计 {total} 字节。")
    print("  观察点：")
    print("  1) 主日志被切成了多个文件，每个都不超过 10 KB；")
    print("  2) 历史文件变成了 .zip —— 这就是 compression='zip' 的效果；")
    print("  3) 文件数量被 retention 限制住，不会无限增长把磁盘写满。")
    print()


def show_production_config() -> None:
    """打印一份生产环境推荐配置，便于对照记忆。"""
    print("=" * 72)
    print("演示 5：生产环境推荐配置（可直接抄）")
    print("=" * 72)
    print("""
logger.add(
    "logs/app.log",                  # 日志文件
    level="INFO",                    # 生产不看 DEBUG
    rotation="500 MB",               # 也可以写 "00:00" 表示每天零点切一个
    retention="30 days",             # 保留 30 天，到期自动删
    compression="zip",               # 旧文件压缩
    encoding="utf-8",                # 中文不乱码
    enqueue=True,                    # 多进程/多线程安全
    backtrace=True,                  # 记录调用栈
    diagnose=False,                  # 关闭变量值，防止敏感信息落盘
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
)
""")


def main() -> None:
    print()
    print("#" * 72)
    print("# loguru 文件输出与轮转（后端开发基础 · 日志章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"日志目录：{ROTATE_DIR}")
    print()

    clean_old_files()
    build_logger()

    demo_rotation()
    demo_filter_and_levels()
    demo_serialize()
    list_files()
    show_production_config()

    # 移除所有 handler：确保日志队列刷盘，文件内容完整（生产环境进程退出前也应做这一步）
    logger.remove()

    print("#" * 72)
    print("# 要点回顾")
    print("#" * 72)
    print("1. rotation 决定什么时候换文件，retention 决定留多久，compression 决定怎么省空间")
    print("2. 三者必须一起配，否则不是写满磁盘就是丢了排障线索")
    print("3. serialize=True 让日志变成结构化数据，是接入日志平台的前提")
    print("4. diagnose 生产必须关：它会记录每层栈帧的局部变量，可能把密码写进日志")
    print("5. 进程退出前 logger.remove() 可以把 enqueue 队列刷干净")
    print()


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"（本脚本总耗时 {time.time() - start:.3f} 秒）")
