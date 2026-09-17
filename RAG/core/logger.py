"""loguru 日志配置:控制台 INFO,文件 DEBUG 按天轮转

这个模块是**配置即副作用**的写法:import 它的时候就把 loguru 的 handler 装好了,
任何模块只要 `from core.logger import logger` 就自动获得统一的日志格式。
这么做的理由:loguru 默认 handler 是全局的,若每个模块各自 add handler,
日志会重复输出 N 遍(全仓几十个模块调用 logger 的话就是灾难)。

⚠ 由此带来两个必须知道的坑:
1. **import 顺序会影响谁先配置**:本模块先 `logger.remove()` 清掉 loguru 的默认
   stderr handler,再装自己的两个。如果别的模块在 import 本模块之前就调了
   `logger.add`,那个 handler 会被这里的 remove 一起清掉(remove 无参 = 移除全部);
2. **`import core.logger` 会在磁盘上建目录**:下面的 `LOG_DIR.mkdir(exist_ok=True)`
   在 import 期执行。所以只要有任何代码 import 了 logger,`RAG/logs/` 就会出现 ——
   写只读环境/沙箱下的测试时要注意这一点(它会在导入期尝试创建目录)。
"""

import sys
from pathlib import Path

from loguru import logger

# 日志目录相对**本文件**定位(不是 CWD):这样不管从仓库根、RAG/ 还是别处启动,
# 日志都落在同一个 RAG/logs/ 下。用 CWD 相对路径的话,从不同目录启动会各写一份,
# 排查时找不到日志。
# `exist_ok=True` 而不是 try/except:目录已存在是正常情况(第二次启动)
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 先清空 loguru 的默认 handler(它输出到 stderr、无格式),避免同一条日志打印两遍
logger.remove()
# 控制台 handler:给**人看**的,所以只到 INFO、带颜色、时间只到秒(HH:mm:ss)。
# 写 stderr 而不是 stdout,是为了不干扰管道里的正常输出(如脚本产出的 JSON)
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
)
# 文件 handler:给**排查**用的,所以到 DEBUG、带毫秒与 模块:函数:行号。
# - rotation="00:00":每天零点切分,文件名里的 {time:YYYY-MM-DD} 会在切分时重新求值;
# - retention="14 days":只留两周,防止长期运行把磁盘写满;
# - encoding="utf-8":必须显式指定 —— Windows 默认编码(GBK)写中文日志会乱码,
#   更糟的是遇到 emoji/生僻字会直接抛 UnicodeEncodeError 把业务代码带崩;
# - 格式串里没有颜色标记:文件里出现 ANSI 转义码会让 grep/less 看到一堆乱码
logger.add(
    LOG_DIR / "rag_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",
    retention="14 days",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{function}:{line} - {message}",
)

# 显式声明导出面:让 `from core.logger import *` 只带出 logger,
# 不把 sys / Path / LOG_DIR 一起漏出去(LOG_DIR 会暴露路径结构)
__all__ = ["logger"]
