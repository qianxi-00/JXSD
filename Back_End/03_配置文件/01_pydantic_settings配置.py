"""pydantic-settings 配置管理完整示例
================================================================
对应课案章节：后端开发基础 → 配置文件（Pydantic Settings）

本节知识点：
    1. 为什么需要配置管理：把"会变的东西"（数据库地址、密钥、端口）从代码里抽出来
    2. BaseSettings 基础用法：字段即配置项，类型注解即校验规则
    3. .env 文件加载：SettingsConfigDict(env_file=...)
    4. 环境变量优先级：环境变量 > .env 文件 > 代码默认值
    5. 类型自动转换与校验：PORT 从字符串 "8000" 变成 int 8000，写错直接报错
    6. SecretStr 敏感字段保护：打印时自动变成 **********
    7. field_validator 自定义校验：如 app_env 只能是固定几个值
    8. 多环境配置：.env.dev / .env.prod 的切换思路
    9. 与本仓库根目录 config.py 的关系：根配置也是用同一套机制实现的

运行方式（PowerShell，必须用本仓库的虚拟环境解释器）：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\03_配置文件\\01_pydantic_settings配置.py'

运行结果：一段纯中文讲解式输出，并把演示用的 .env.dev / .env.prod 写入 Back_End/data/env_demo/。
"""

from __future__ import annotations

import os
import sys
import pathlib

# ------------------------------------------------------------
# 路径计算：脚本内部一律用 __file__ 推算目录，绝不依赖"当前工作目录"
# 本文件层级：Back_End/03_配置文件/01_pydantic_settings配置.py
#   parents[0] = 03_配置文件
#   parents[1] = Back_End
#   parents[2] = Python_Base（仓库根目录，config.py 所在处）
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]

# sys.path 兜底：根目录本来已通过 .pth 加入 sys.path，这里再兜一层，保证任何运行方式都能 import config
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = BACK_END / "data"
ENV_DEMO_DIR = DATA_DIR / "env_demo"
ENV_EXAMPLE = HERE / ".env.example"

# 避免 pydantic-settings 在 Windows 控制台输出 ANSI 颜色干扰阅读（只是标记，无副作用）
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from pydantic import Field, SecretStr, field_validator  # noqa: E402
from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402


# ============================================================
# 一、定义自己的 Settings 类
# ============================================================
class AppSettings(BaseSettings):
    """应用配置类。

    核心思想：**把配置声明成一个 Pydantic 模型**。
    - 每个字段的**类型注解**就是校验规则；
    - 每个字段的**默认值**就是"没配置时的兜底值"；
    - 实例化时 pydantic-settings 会按固定优先级去"找值"。

    取值优先级（从高到低）：
        1) 直接传给构造函数的参数（AppSettings(app_name="x")）
        2) 操作系统的环境变量（os.environ）
        3) env_file 指定的 .env 文件
        4) 字段默认值
    """

    # SettingsConfigDict 就是"这个类怎么读配置"的说明
    model_config = SettingsConfigDict(
        env_file=str(ENV_EXAMPLE),   # 从 .env.example 读取（真实项目里应指向 .env）
        env_file_encoding="utf-8",   # 必须写，否则中文值在 Windows 上可能乱码
        case_sensitive=False,        # 不区分大小写：字段 app_name ↔ 环境变量 APP_NAME
        extra="ignore",              # .env 里有、类里没声明的键直接忽略，不报错
    )

    # ---------- 基础字段：带默认值 ----------
    app_name: str = Field(default="我的后端应用", description="应用名称，显示在日志与文档标题里")
    app_env: str = Field(default="development", description="运行环境：development/testing/production")
    debug: bool = Field(default=True, description="调试开关，生产环境必须为 False")

    # ---------- 类型校验：字符串 "8000" 会被自动转成 int ----------
    port: int = Field(default=8000, ge=1, le=65535, description="服务监听端口")

    # ---------- 数据库连接串 ----------
    database_url: str = Field(
        default="sqlite:///./data/app.db",
        description="数据库连接串。SQLite 无需服务；MySQL 形如 mysql+pymysql://user:pwd@host:3306/db",
    )

    # ---------- 安全性：SecretStr ----------
    # 用 SecretStr 而不是 str 的原因：不小心 print / 打印日志时只会显示 **********，
    # 需要真实值时显式调用 .get_secret_value()，防止密钥泄露进日志。
    jwt_secret: SecretStr = Field(
        default=SecretStr("please-change-me-to-a-random-secret"),
        description="JWT 签名密钥，生产环境必须换成随机长串",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT 签名算法")

    admin_email: str = Field(default="admin@example.com", description="管理员邮箱")

    # ---------- 外部服务（本机可能没启动，程序内部会自行降级） ----------
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", description="Redis 连接串")

    # ---------- 自定义校验器 ----------
    @field_validator("app_env")
    @classmethod
    def _check_app_env(cls, value: str) -> str:
        """只允许三种运行环境，写错立刻报错，避免把生产配置当开发用。

        field_validator 在"类型转换完成之后"执行，此时 value 已经是 str。
        """
        allowed = {"development", "testing", "production"}
        if value not in allowed:
            raise ValueError(f"app_env 只能是 {sorted(allowed)} 之一，当前为 {value!r}")
        return value

    @field_validator("port")
    @classmethod
    def _check_port(cls, value: int) -> int:
        """端口范围再次确认（Field 的 ge/le 已经做了，这里演示校验器写法）。"""
        if value in (3306, 6379, 19530):
            print(f"  [提示] 端口 {value} 是数据库/缓存默认端口，生产环境建议改掉以防被扫描。")
        return value


# ============================================================
# 二、如何把 .env.dev / .env.prod 写出来（多环境演示）
# ============================================================
DEV_ENV_CONTENT = """# 开发环境配置（由示例脚本自动生成，仅用于演示多环境切换）
APP_NAME="后端示例-开发环境"
APP_ENV=development
DEBUG=true
PORT=8001
DATABASE_URL="sqlite:///./data/app_dev.db"
LOG_LEVEL=DEBUG
"""

PROD_ENV_CONTENT = """# 生产环境配置（由示例脚本自动生成，仅用于演示多环境切换）
APP_NAME="后端示例-生产环境"
APP_ENV=production
DEBUG=false
PORT=8099
DATABASE_URL="mysql+pymysql://user:password@127.0.0.1:3306/my_database?charset=utf8mb4"
LOG_LEVEL=WARNING
"""


def prepare_env_files() -> tuple[pathlib.Path, pathlib.Path]:
    """把两个环境的 .env 文件写到 data/env_demo/ 下并返回路径。

    真实项目中这两个文件通常放在项目根目录，命名为 .env.dev / .env.prod，
    并且**不入 Git**（因为可能含密码），团队里通过密钥管理平台或 CI 变量下发。
    """
    ENV_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    dev_file = ENV_DEMO_DIR / ".env.dev"
    prod_file = ENV_DEMO_DIR / ".env.prod"
    dev_file.write_text(DEV_ENV_CONTENT, encoding="utf-8")
    prod_file.write_text(PROD_ENV_CONTENT, encoding="utf-8")
    return dev_file, prod_file


def mask(value: str, keep: int = 4) -> str:
    """把敏感字符串打码，只保留头尾少量字符，用于安全地打印配置。"""
    if not value:
        return "(空)"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * 6}{value[-keep:]}"


def show_basic_usage() -> None:
    """演示 1：基础用法（从 .env.example 读取 + 默认值兜底）。"""
    print("=" * 72)
    print("演示 1：BaseSettings 基础用法（数据来源 = .env.example + 代码默认值）")
    print("=" * 72)
    print(f"配置文件路径：{ENV_EXAMPLE}")
    print(f"配置文件存在：{ENV_EXAMPLE.exists()}\n")

    settings = AppSettings()

    print(f"app_name       = {settings.app_name}")
    print(f"app_env        = {settings.app_env}")
    print(f"debug          = {settings.debug}          <- 注意：.env 里写的是 true，被转成了 bool")
    print(f"port           = {settings.port}            <- 类型注解是 int，底层其实是字符串，已被自动转换")
    print(f"database_url   = {settings.database_url}")
    print(f"admin_email    = {settings.admin_email}")
    print(f"log_level      = {getattr(settings, 'log_level', '(未声明，被 extra=ignore 忽略)')}")
    print(f"jwt_secret     = {settings.jwt_secret}      <- SecretStr 自动打码，日志里不会泄露")
    print(f"jwt_algorithm  = {settings.jwt_algorithm}")
    print(f"redis_url      = {settings.redis_url}")
    print("\n小结：字段类型注解 = 自动校验 + 自动转换；字段默认值 = 没配置时的兜底。\n")


def show_env_priority() -> None:
    """演示 2：环境变量优先级高于 .env 文件。"""
    print("=" * 72)
    print("演示 2：取值优先级（环境变量 > .env 文件 > 代码默认值）")
    print("=" * 72)

    before = AppSettings().app_name
    print(f"修改前 app_name = {before!r}（来自 .env.example）")

    # 临时设置一个环境变量，模拟"生产环境用容器注入环境变量覆盖配置文件"
    os.environ["APP_NAME"] = "被环境变量覆盖的应用名"
    after = AppSettings().app_name
    print(f"设置环境变量 APP_NAME 后 app_name = {after!r}")
    assert after == "被环境变量覆盖的应用名", "环境变量应当覆盖 .env 文件"
    print("结论：环境变量优先级最高 —— 这正是 Docker/K8S 注入配置的原理。")

    # 还原，避免影响后续演示
    os.environ.pop("APP_NAME", None)
    print(f"已还原环境变量，app_name 回到 {AppSettings().app_name!r}\n")


def show_validation() -> None:
    """演示 3：类型校验与自定义校验器（配置写错要"启动即失败"，而不是运行到一半才炸）。"""
    print("=" * 72)
    print("演示 3：类型校验与自定义校验器")
    print("=" * 72)

    # 3.1 正确的类型转换
    s = AppSettings(port="8123")
    print(f"传入字符串 port='8123' -> 得到 {s.port!r}，类型 {type(s.port).__name__}")

    # 3.2 非法值会被 pydantic 拦截
    for bad_port in ("abc", 70000):
        try:
            AppSettings(port=bad_port)
        except Exception as exc:  # 捕获后只打印第一行，保持输出干净
            first_line = str(exc).strip().splitlines()[0]
            print(f"传入非法 port={bad_port!r} -> 被拦截：{first_line}")

    # 3.3 自定义校验器
    try:
        AppSettings(app_env="prod")  # 只写了 prod，不在允许集合里
    except Exception as exc:
        first_line = str(exc).strip().splitlines()[0]
        print(f"传入非法 app_env='prod' -> 被拦截：{first_line}")

    # 3.4 演示"默认端口提醒"校验器的副作用输出
    print("传入 port=6379（Redis 默认端口）触发自定义提示：")
    AppSettings(port=6379)
    print("小结：配置错误应该在启动阶段就暴露，这是 pydantic-settings 最大的价值。\n")


def show_multi_env() -> None:
    """演示 4：多环境配置切换（.env.dev / .env.prod）。"""
    print("=" * 72)
    print("演示 4：多环境配置（开发 / 生产）")
    print("=" * 72)

    dev_file, prod_file = prepare_env_files()
    print(f"已生成演示用环境文件：\n  {dev_file}\n  {prod_file}\n")

    # 关键点：用 _env_file 参数在实例化时临时指定文件，不需要改类定义
    dev = AppSettings(_env_file=str(dev_file))
    prod = AppSettings(_env_file=str(prod_file))

    print(f"{'配置项':<16}{'开发环境':<28}{'生产环境'}")
    print("-" * 72)
    rows = [
        ("app_name", dev.app_name, prod.app_name),
        ("app_env", dev.app_env, prod.app_env),
        ("debug", dev.debug, prod.debug),
        ("port", dev.port, prod.port),
        ("database_url", dev.database_url, prod.database_url),
    ]
    for name, dv, pv in rows:
        print(f"{name:<16}{str(dv):<28}{pv}")

    print("\n真实项目常见做法：")
    print("  · pyproject.toml 里区分依赖组，CI 里按分支注入不同环境变量")
    print("  · 本地开发用 .env（不入库），生产用容器环境变量/密钥管理平台下发")
    print("  · 用 APP_ENV 决定加载哪个文件：env_file=f'.env.{os.getenv(\"APP_ENV\", \"dev\")}'\n")


def show_root_config() -> None:
    """演示 5：读取本仓库根目录 config.py 的统一配置。"""
    print("=" * 72)
    print("演示 5：与本仓库根目录 config.py 打通（from config import settings）")
    print("=" * 72)

    try:
        from config import settings as root_settings  # noqa: PLC0415

        print("成功导入根配置：from config import settings")
        print(f"  settings.app_env        = {root_settings.app_env}")
        print(f"  settings.log_level      = {root_settings.log_level}")
        print(f"  settings.mysql_url      = {root_settings.mysql_url}")
        print(f"  settings.redis_url      = {root_settings.redis_url}")
        print(f"  settings.milvus_uri     = {root_settings.milvus_uri}")
        print(f"  settings.openai_api_key = {mask(root_settings.openai_api_key)}   <- 脱敏打印")
        print(f"  settings.llm.model      = {root_settings.llm.model}   <- 分组配置（RAG 项目用）")
        print("\n注意：根配置文件在这套机制上和本示例完全一样，")
        print("      只是它同时提供了'扁平字段'和'分组字段'两种访问方式。")
    except Exception as exc:  # 极少数情况下根配置导入失败也不允许抛错
        print(f"未能导入根配置（不影响本节学习）：{type(exc).__name__}: {exc}")
    print()


def main() -> None:
    """按顺序跑完 5 个演示。"""
    print()
    print("#" * 72)
    print("# pydantic-settings 配置管理演示（后端开发基础 · 配置文件章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"Python 版本：{sys.version.split()[0]}")
    print(f"脚本目录：{HERE}")
    print(f"输出目录：{DATA_DIR}")
    print()

    show_basic_usage()
    show_env_priority()
    show_validation()
    show_multi_env()
    show_root_config()

    print("#" * 72)
    print("# 全部演示完成：配置管理要点回顾")
    print("#" * 72)
    print("1. 配置与代码分离：改配置不改代码，不用重新打包镜像")
    print("2. 类型即校验：字段注解 int/bool/SecretStr 自动完成转换与保护")
    print("3. 优先级明确：环境变量 > .env 文件 > 默认值")
    print("4. 敏感信息用 SecretStr，且 .env 永不入库")
    print("5. 多环境用不同的 env_file 或环境变量区分，绝不在代码里 if 判断硬编码")
    print()


if __name__ == "__main__":
    main()
