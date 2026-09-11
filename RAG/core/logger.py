"""loguru 日志配置:控制台 INFO,文件 DEBUG 按天轮转"""

import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
)
logger.add(
    LOG_DIR / "rag_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",
    retention="14 days",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{function}:{line} - {message}",
)

__all__ = ["logger"]
