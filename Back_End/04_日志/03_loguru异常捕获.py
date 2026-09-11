"""Loguru 异常捕获：把堆栈写进文件，让终端保持干净
================================================================
对应课案章节：后端开发基础 → 日志（logger.exception 捕获异常 / 函数耗时装饰器）

本节知识点：
    1. logger.exception()：在 except 块里记录异常，自动带上完整堆栈信息
    2. {exception} 占位符与"loguru 会自动追加堆栈"的行为：
       只要记录里有异常，loguru 就会把堆栈拼到格式化结果后面（即使 format 不写 {exception}），
       因此想让终端干净，必须用 **filter 过滤** + 自定义 sink，而不是改 format
    3. backtrace=True：展开被 try/except 吞掉的中间调用栈（看真正的出错源头）
    4. diagnose=False：不记录每层栈帧的局部变量值（生产必须关，防止密码泄露）
    5. @logger.catch 装饰器：整个函数的兜底捕获，不用自己写 try/except
    6. logger.opt(exception=True)：显式告诉 loguru "本次日志要附上当前异常"
    7. 函数耗时装饰器 @cost_time：用 functools.wraps 保留原函数元信息
    8. 为什么"不允许把堆栈直接打到终端"：终端日志会被采集、会展示给用户，
       堆栈里可能含路径、SQL、密钥；正确做法是终端只提示一句，细节全部落盘。

运行方式（PowerShell，必须用本仓库的虚拟环境解释器）：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\04_日志\\03_loguru异常捕获.py'
"""

from __future__ import annotations

import functools
import pathlib
import sys
import time

from loguru import logger

# ------------------------------------------------------------
# 路径：本文件位于 Back_End/04_日志/03_loguru异常捕获.py
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
LOG_DIR = BACK_END / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

EXCEPTION_LOG = LOG_DIR / "03_异常记录.log"


def build_logger():
    """配置三个 handler，实现"终端一行、文件一堆"的效果。

    为什么需要三个？这是本节最重要的机制：

        loguru 只要发现记录里带了异常（record["exception"] 不为空），
        就会在格式化结果后面**自动追加完整堆栈** —— 哪怕 format 里没写 {exception}。
        所以想让终端保持干净，唯一可靠的办法是：用 filter 把"带异常的记录"挡在终端之外，
        再单独用一个小 handler 把这类记录压缩成一行提示。

    三个 handler 分工：
        1) console_plain   只收"没有异常"的普通日志 —— 正常信息
        2) console_error   只收"有异常"的日志 —— 自己拼一行摘要，绝不输出堆栈
        3) 文件 handler    全部收下，并且带上完整堆栈（带 {exception}）
    """
    logger.remove()

    # ---------- 1. 终端：普通日志 ----------
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=False,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        filter=lambda record: record["exception"] is None,   # 放行"没有异常"的记录
    )

    # ---------- 2. 终端：异常日志压缩成一行 ----------
    def console_exception_sink(message) -> None:
        """自定义 sink：把一条带异常的日志压缩成一行。

        参数 message 是 loguru 的 Message（str 子类），它的 .record 里有全部原始信息。
        我们**不使用** message 本身（里面已经被 loguru 追加了完整堆栈），
        而是自己用 record 拼出想要的一行 —— 这就是"格式完全自己说了算"的写法。
        """
        record = message.record
        exc_type = record["exception"].type
        exc_name = exc_type.__name__ if exc_type is not None else "未知异常"
        where = f"{record['name']}:{record['function']}:{record['line']}"
        sys.stderr.write(
            f"{record['time']:%H:%M:%S} | {record['level'].name:<8} | "
            f"{record['message']} | 异常类型={exc_name} | 位置={where}\n"
        )
        sys.stderr.write(f"          └─ 完整堆栈已写入 {EXCEPTION_LOG.name}（不在此刷屏）\n")

    logger.add(
        console_exception_sink,
        level="ERROR",
        format="{message}",
        filter=lambda record: record["exception"] is not None,   # 只放行"有异常"的记录
    )

    # ---------- 3. 文件：完整堆栈 ----------
    # backtrace=True：把被 except 吞掉的中间调用链也展开，方便找到真正源头
    # diagnose=False：不打印局部变量，避免把密码/Token 写进日志
    logger.add(
        EXCEPTION_LOG,
        level="INFO",
        encoding="utf-8",
        backtrace=True,
        diagnose=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}\n{exception}"
        ),
    )
    return logger


# ============================================================
# 课案原版：函数耗时装饰器
# ============================================================
def cost_time(func):
    """统计函数耗时的装饰器。

    原理：
        - 用 functools.wraps 把原函数的 __name__ / __doc__ 复制到 inner 上，
          否则日志里只会看到一堆 "inner"，而且 FastAPI 之类的框架会认不出函数；
        - 在函数执行前后各取一次 time.perf_counter()，差值就是耗时；
        - 用 try/finally 保证"即使函数抛异常，耗时也会被记录"。
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            cost = time.perf_counter() - start
            # 用 debug 级别：终端（INFO）看不到，文件（INFO/DEBUG）能看到
            logger.debug(f"函数 {func.__name__} 耗时 {cost * 1000:.2f} ms")

    return inner


@cost_time
def slow_calculation(n: int) -> int:
    """模拟一个耗时计算，返回值必须保留（验证装饰器没有吞掉返回值）。"""
    total = 0
    for i in range(n):
        total += i * i
    return total


@cost_time
def broken_calculation() -> int:
    """这个函数会被装饰器记录耗时后正常抛出异常，用于演示 finally 的作用。"""
    raise ValueError("模拟业务校验失败：金额不能为负数")


def demo_logger_exception() -> None:
    """演示 1：课案写法 logger.exception()。"""
    print("=" * 72)
    print("演示 1：try/except + logger.exception（终端一行，文件完整堆栈）")
    print("=" * 72)
    try:
        1 / 0
    except ZeroDivisionError:
        # logger.exception 等价于 logger.opt(exception=True).error(...)
        # 它必须在 except 块内部调用，才能拿到 sys.exc_info()
        logger.exception("计算失败：除数为零")
    print("↑ 终端只出现一行 ERROR（压缩过的摘要），堆栈已经写进日志文件。")
    print("  这正是生产环境想要的：值班同学看一行结论，排障时再翻文件看堆栈。\n")


def demo_opt_exception() -> None:
    """演示 2：logger.opt(exception=True) 显式附加异常信息。"""
    print("=" * 72)
    print("演示 2：logger.opt(exception=True) 与 except ... as exc 组合")
    print("=" * 72)
    data = {"name": "苹果"}
    try:
        price = data["price"]  # KeyError
    except KeyError as exc:
        # 把业务上下文一起记下来：出错时"现场信息"往往比堆栈更值钱
        logger.opt(exception=True).error(f"字段缺失：{exc!s}，当前记录内容={data}")
    print("↑ 依然只有一行；文件里这条记录带有完整堆栈。\n")


def demo_logger_catch() -> None:
    """演示 3：@logger.catch 装饰器（函数级兜底）。"""
    print("=" * 72)
    print("演示 3：@logger.catch 装饰器 —— 不写 try/except 也能兜底")
    print("=" * 72)

    @logger.catch(reraise=False, message="处理订单时发生未预期异常")
    def process_order(order_id: int) -> str:
        # reraise=False（默认）：异常被记录后吞掉，函数返回 None
        # reraise=True：记录完继续往上抛，交给上层（如 FastAPI 全局异常处理器）
        if order_id < 0:
            raise ValueError("订单号不能为负数")
        return f"订单 {order_id} 处理完成"

    ok = process_order(1001)
    print(f"正常调用返回：{ok!r}")

    bad = process_order(-1)
    print(f"异常调用返回：{bad!r}（异常已被记录并吞掉，程序继续运行）")

    # 也可以给一个兜底返回值，让调用方拿到"安全默认值"
    @logger.catch(default={"status": "failed"})
    def fetch_remote_config() -> dict:
        raise ConnectionError("模拟远程配置中心连接失败")

    print(f"带默认值的兜底返回：{fetch_remote_config()}")
    print()


def demo_cost_time() -> None:
    """演示 4：课案的耗时装饰器（并验证返回值、异常、元信息都没被破坏）。"""
    print("=" * 72)
    print("演示 4：耗时装饰器 @cost_time")
    print("=" * 72)
    result = slow_calculation(200_000)
    print(f"slow_calculation(200000) = {result}")
    print(f"装饰后函数名仍然是：{slow_calculation.__name__!r}（functools.wraps 的功劳）")

    try:
        broken_calculation()
    except ValueError as exc:
        # 装饰器没有吞掉异常（finally 只负责记时间），业务逻辑照常向上抛
        print(f"broken_calculation() 照常抛出异常：{exc}")
    print("技巧：把报错函数的 exec_info 也记进日志，定位更省力：")
    try:
        broken_calculation()
    except ValueError:
        logger.opt(exception=True).error("broken_calculation 调用失败")
    print()


def show_log_file() -> None:
    """演示 5：证明堆栈确实写进了文件（只做统计，不把堆栈打到屏幕）。"""
    print("=" * 72)
    print("演示 5：检查异常日志文件（统计而不刷屏）")
    print("=" * 72)
    if not EXCEPTION_LOG.exists():
        print("日志文件不存在，跳过。")
        return

    lines = EXCEPTION_LOG.read_text(encoding="utf-8").splitlines()
    marker = "Trace" + "back (most recent call last)"      # 拆分拼接，避免本文件自身被误搜
    stack_blocks = sum(1 for line in lines if marker in line)
    keywords = ["ZeroDivisionError", "KeyError", "ValueError", "ConnectionError"]

    print(f"文件：{EXCEPTION_LOG}")
    print(f"总行数：{len(lines)}")
    print(f"完整堆栈块数：{stack_blocks} 条（每一条对应一次异常记录）")
    for word in keywords:
        count = sum(1 for line in lines if word in line)
        print(f"  含 {word:<18} 的行数：{count}")
    print("\n结论：日志文件里有完整可追溯的异常链，而屏幕上没有任何堆栈刷屏 ——")
    print("      这就是「终端干净 + 文件完整」的两个 handler 分工。\n")


def main() -> None:
    print()
    print("#" * 72)
    print("# loguru 异常捕获（后端开发基础 · 日志章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"异常日志文件：{EXCEPTION_LOG}")
    print()

    # 清掉上一次运行的日志，保证每次演示的统计结果可复现
    EXCEPTION_LOG.unlink(missing_ok=True)

    build_logger()

    demo_logger_exception()
    demo_opt_exception()
    demo_logger_catch()
    demo_cost_time()
    logger.complete()        # 刷盘，保证统计准确
    show_log_file()

    logger.remove()

    print("#" * 72)
    print("# 要点回顾")
    print("#" * 72)
    print("1. logger.exception() 只在 except 块里有意义，它自带当前异常信息")
    print("2. loguru 会自动给「带异常」的记录追加堆栈 —— 用 filter 把它挡在终端之外，")
    print("   再用一个自定义 sink 输出一行摘要，才能做到「终端干净 + 文件完整」")
    print("3. backtrace=True 展开被吞掉的调用链；diagnose 生产必须 False")
    print("4. @logger.catch 是函数级兜底，reraise 决定要不要继续往上抛")
    print("5. 耗时装饰器要配 functools.wraps，并用 finally 保证异常时也记时间")
    print()


if __name__ == "__main__":
    main()
