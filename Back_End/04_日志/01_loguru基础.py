"""Loguru 日志基础：从 print 到专业日志
================================================================
对应课案章节：后端开发基础 → 日志（初始化 Loguru / 分级记录）

本节知识点：
    1. 为什么不能用 print 记日志：没有级别、没有时间、不能重定向、无法按大小切割
    2. logger.remove()：先清空默认处理器，再自己定义输出目标（sink）
    3. sink 可以是 stderr、文件路径、甚至是自定义函数
    4. level 参数：每个 sink 可以有独立的最低级别（终端只看 WARNING，文件全都要）
    5. format 占位符：{time} {level} {message} {name} {function} {line} {extra}
    6. 分级记录：debug / info / success / warning / error / critical
    7. 结构化日志：logger.bind() 附加业务字段，便于日志平台检索
    8. 日志写入 Back_End/data/logs/ 目录（脚本自动创建，不依赖当前工作目录）

运行方式（PowerShell，必须用本仓库的虚拟环境解释器）：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\04_日志\\01_loguru基础.py'
"""

from __future__ import annotations

import pathlib
import sys

from loguru import logger

# ------------------------------------------------------------
# 路径：本文件位于 Back_End/04_日志/01_loguru基础.py
#   parents[0] = 04_日志，parents[1] = Back_End
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
LOG_DIR = BACK_END / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)     # exist_ok=True：已存在也不报错

LOG_FILE = LOG_DIR / "01_loguru基础.log"


def build_logger():
    """按课案思路重新配置 loguru 的全局 logger。

    loguru 默认就已经带一个 handler（输出到 stderr、级别 DEBUG、带颜色）。
    真实项目里我们要"关掉默认的，自己定义"，所以第一步永远是 logger.remove()。

    返回值：新配置的 logger 对象（其实就是 loguru 的全局单例，只是为了书写方便）
    """
    # ---------- 第 1 步：清空所有已有处理器 ----------
    logger.remove()          # 不传参数 = 移除所有 handler；传 0 = 只移除默认那个

    # ---------- 第 2 步：终端处理器（人看的，只看重要信息） ----------
    # level="WARNING"：DEBUG/INFO 级别的内容不会出现在终端，保持终端干净
    # colorize=True：按级别自动着色（ERROR 红、WARNING 黄）
    logger.add(
        sys.stderr,
        level="WARNING",
        colorize=True,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
    )

    # ---------- 第 3 步：文件处理器（排查问题用的，全都要） ----------
    # level="DEBUG"：所有级别都落盘，出问题时能完整回看
    # encoding="utf-8"：中文不乱码（Windows 上尤其重要）
    # enqueue=True：多线程/多进程写日志时先入队列，避免内容交差
    logger.add(
        LOG_FILE,
        level="DEBUG",
        encoding="utf-8",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    return logger


def demo_levels() -> None:
    """演示 1：六个日志级别，以及"为什么终端只看到一半"。"""
    print("=" * 72)
    print("演示 1：日志级别 —— 终端只显示 WARNING 及以上，文件里全都有")
    print("=" * 72)

    logger.debug("【DEBUG】开发调试用，例如「查询到 3 条记录」，生产环境通常不开")
    logger.info("【INFO】正常业务流程，例如「用户 alice 登录成功」")
    logger.success("【SUCCESS】明确成功的关键操作，例如「订单 1001 支付完成」")
    logger.warning("【WARNING】不影响运行但要关注，例如「接口耗时 2.3s 偏慢」")
    logger.error("【ERROR】明确出错但服务没挂，例如「第三方支付回调验签失败」")
    logger.critical("【CRITICAL】严重故障，例如「数据库连接池耗尽，服务不可用」")

    print("↑ 上面你可能只看到 WARNING / ERROR / CRITICAL 三行，这是故意的：")
    print("  终端给人看，要简洁；文件给机器和排障看，要完整。")
    print(f"  完整日志已写入：{LOG_FILE}")
    print()


def demo_format_fields() -> None:
    """演示 2：format 可用的占位符（日志"信息量"从哪来）。"""
    print("=" * 72)
    print("演示 2：format 常用占位符")
    print("=" * 72)

    # 临时加一个"全字段"handler 到 stderr，演示完立刻移除（用 handler_id 精确移除）
    handler_id = logger.add(
        sys.stderr,
        level="INFO",
        colorize=False,
        format=(
            "  时间={time:YYYY-MM-DD HH:mm:ss} "
            "级别={level} "
            "模块={name} "
            "函数={function} "
            "行号={line} "
            "消息={message}"
        ),
    )
    logger.info("这一行展示了日志里可以带上的全部定位信息")
    logger.remove(handler_id)      # 演示完必须移除，避免污染后续输出

    print("为什么需要这些字段：线上出问题时，日志要能回答")
    print("  「哪个文件、哪个函数、哪一行、什么时间、什么级别」——缺一个都难定位。")
    print()


def demo_structured() -> None:
    """演示 3：结构化日志（logger.bind 附加业务字段）。"""
    print("=" * 72)
    print("演示 3：结构化日志 logger.bind()，让日志能被检索与统计")
    print("=" * 72)

    # bind 会返回一个"绑定了额外字段"的新 logger，原 logger 不受影响
    # extra 里的字段可以在 format 中用 {extra[字段名]} 取出（也可序列化成 JSON，见下一个脚本）
    request_logger = logger.bind(request_id="req-8f3a91", user_id=10086, path="/api/books/1")
    handler_id = logger.add(
        sys.stderr,
        level="INFO",
        colorize=False,
        format="  [{extra[request_id]}] user={extra[user_id]} path={extra[path]} -> {message}",
    )
    request_logger.info("开始处理请求")
    request_logger.info("命中缓存，耗时 3ms")
    logger.remove(handler_id)

    print("好处：同一次请求的所有日志都带同一个 request_id，")
    print("      在日志平台里搜这个 id 就能还原整条调用链（分布式追踪的基础）。")
    print()


def demo_custom_sink() -> None:
    """演示 4：自定义 sink —— 把日志送进自己的函数（可用于告警）。"""
    print("=" * 72)
    print("演示 4：自定义 sink 函数（把 ERROR 及以上日志「推」到告警通道）")
    print("=" * 72)

    alerts: list[str] = []

    def alert_sink(message) -> None:
        """message 是 loguru 的 Message 对象（str 子类），这里模拟发送告警。

        真实项目里这里可以：requests.post(钉钉机器人)、发邮件、写 Kafka……
        注意：sink 内部千万不要再抛异常，否则会打乱日志系统。
        """
        record = message.record
        alerts.append(f"[告警]{record['time']:%H:%M:%S} {record['level'].name}: {record['message']}")

    # level="ERROR"：只有 ERROR 及以上才进这个 sink
    handler_id = logger.add(alert_sink, level="ERROR", format="{message}")
    logger.warning("这条只是警告，不会触发告警")
    logger.error("这条会触发告警：数据库查询超时")
    logger.remove(handler_id)

    print("告警通道收到的内容：")
    for line in alerts:
        print("  " + line)
    print(f"（模拟共触发 {len(alerts)} 条告警）")
    print()


def show_file_tail(lines: int = 8) -> None:
    """把日志文件最后几行读出来，证明"终端没显示的 DEBUG 其实落盘了"。"""
    print("=" * 72)
    print(f"演示 5：回看日志文件最后 {lines} 行（证明 DEBUG 也落盘了）")
    print("=" * 72)
    if not LOG_FILE.exists():
        print("日志文件还不存在，跳过。")
        return
    content = LOG_FILE.read_text(encoding="utf-8").splitlines()
    for line in content[-lines:]:
        print("  " + line)
    print(f"  （文件共 {len(content)} 行，路径：{LOG_FILE}）")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# loguru 日志基础（后端开发基础 · 日志章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"日志目录：{LOG_DIR}")
    print()

    build_logger()
    demo_levels()
    demo_format_fields()
    demo_structured()
    demo_custom_sink()
    show_file_tail()

    print("#" * 72)
    print("# 要点回顾")
    print("#" * 72)
    print("1. logger.remove() 之后重新 add，是 loguru 的标准起手式")
    print("2. 不同 sink 用不同 level：终端少而精，文件全而细")
    print("3. format 决定日志的信息量，{name}/{function}/{line} 是排障刚需")
    print("4. logger.bind(...) 给日志加业务字段，是链路追踪的起点")
    print("5. 自定义 sink 可以把 ERROR 日志送到告警通道")
    print()


if __name__ == "__main__":
    main()
