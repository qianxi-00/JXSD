"""项目全局统一配置模块（Python_Base 根目录唯一配置入口）
================================================================
本文件合并了三个项目的全部配置：
    1. Python_Base 原有配置（deepseek/openai/mysql/postgres/redis/milvus）
    2. RAG 项目分组配置（llm/embedding/rerank/retrieval/redis/milvus/es/mysql/paddleocr）
    3. Agent 课案扁平配置（api_key/base_url/model_name/pg_uri/langfuse）

所有子项目（RAG/Agent/...）一律通过根目录的这套配置读取：
    from config import settings

用法：
    settings.api_key            # Agent 课案扁平字段
    settings.llm.api_key        # RAG 分组字段
    settings.mysql_url          # 拼接好的连接串（property）
    settings.openai_api_key     # Python_Base 原有字段

配置值统一存放在根目录 `.env`（不入 Git），由 pydantic-settings 自动加载。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py 所在目录 (项目根目录), 确保无论从哪里运行都能找到 .env
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


# ============================================================
# 一、扁平配置：Python_Base 原有字段 + Agent 课案字段
# ============================================================
class _CoreSettings(BaseSettings):
    """扁平配置类，自动从 .env 文件/环境变量中读取"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ============ LLM 模型配置（Python_Base 原有） ============
    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # OpenAI 兼容接口
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # ============ Agent 课案扁平字段（原 Agent/conf.py） ============
    # 课案代码统一使用这三个字段初始化大模型
    api_key: str = ""
    base_url: str = ""
    model_name: str = ""

    # LangGraph 持久化（短期记忆/长期记忆）使用的 PostgreSQL 连接串
    # 形如 postgresql://<用户>:<口令>@<主机>:<端口>/<库名>，由 .env 的 PG_URI 提供。
    # 这里刻意**不写默认值**：连接串内嵌账号口令，写进代码就等于把凭据提交进仓库。
    pg_uri: str = ""

    # Langfuse 可观测平台
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---- Agent 课案里额外用到的第三方平台密钥（全部留空，只在本地 .env 提供）----
    # 百度千帆：课案「多 Agent / 子Agent」用它的联网搜索 MCP
    baidu_qfan_api_key: str = ""
    # Gitee：课案「多 Agent / 路由与合并」用它的代码仓库 MCP
    gitee_api_key: str = ""
    # 阿里云百炼（DashScope）：课案「监控与评估 / RAG评估」用它跑评测模型
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ============ 数据库配置（Python_Base 原有） ============
    # 说明：下面这些字段一律只保留「主机 / 端口」这类无敏感的默认值，
    #       用户名、口令、库名全部留空，统一由根目录 .env 提供。
    #       口令写进代码等于把凭据提交进仓库，任何情况下都不要这样做。

    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = ""
    mysql_password: str = ""
    mysql_db: str = ""

    # PostgreSQL
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_user: str = ""
    postgres_password: str = ""
    postgres_db: str = ""

    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    redis_url: str = "redis://localhost:6379/0"

    # Milvus 向量数据库
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_db: str = "default"

    # ============ 通用配置 ============
    app_env: str = "development"
    log_level: str = "INFO"

    # ============ 拼接好的连接串 ============
    @property
    def mysql_url(self) -> str:
        """MySQL 连接 URL (PyMySQL/SQLAlchemy 格式)"""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
            f"?charset=utf8mb4"
        )

    @property
    def postgres_dsn(self) -> str:
        """PostgreSQL 连接串 (psycopg3 格式)"""
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"user={self.postgres_user} password={self.postgres_password} "
            f"dbname={self.postgres_db}"
        )

    @property
    def postgres_url(self) -> str:
        """PostgreSQL 连接 URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# ============================================================
# 二、RAG 项目分组配置（原 RAG/config.py，逐组保留）
# ============================================================
class AppSettings(BaseSettings):
    """Web 应用配置"""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="APP_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8099


class LLMSettings(BaseSettings):
    """大模型调用配置"""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="LLM_", extra="ignore")

    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 4096
    enable_thinking: bool = False


class EvalLLMSettings(BaseSettings):
    """评估/评判用的大模型配置（**与生成模型分家**）。

    为什么要有这一段（优化篇课案明确要求，本项目此前是"自述分开、实则同一个"）：
    Ragas 四指标与 `evaluation/llm_judge.py` 的 LLM 评分，如果用的是**生成回答的那个模型**，
    就存在**自我偏好**风险 —— 模型给自己的回答打分偏高，评估结果系统性乐观。
    分开配置（至少换成同网关的另一个模型）能显著削弱这一偏差。

    三项都要可独立配置：`model` 是必须不同的那一项；`api_key` / `base_url` 留空则
    **回退到生成侧的 `LLM_*`**（同一家网关时不必重复填密钥，也避免把密钥复制两份）。

    ⚠ 未配置 `EVAL_LLM_MODEL` 时行为与以前一致（回退生成模型），但调用方会打 WARNING
    说明"这次评判用的就是生成模型"，不让差异静默。
    """

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="EVAL_LLM_", extra="ignore")

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    # 评判要可复现：温度 0（生成侧默认 0.2，是另一个刻意的差异）
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 180.0

    @property
    def configured(self) -> bool:
        """是否显式配置了评判模型（只看 model：密钥/网关允许回退）。"""
        return bool(self.model)


class EmbeddingSettings(BaseSettings):
    """向量化模型配置"""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="EMBEDDING_", extra="ignore")

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    embedding_size: int = 1024
    batch_size: int = 32


class RerankSettings(BaseSettings):
    """重排序模型配置"""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="RERANK_", extra="ignore")

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    top_k: int = 8
    relevance_p: float = 0.65


class RetrievalSettings(BaseSettings):
    """检索配置：双路召回合并"""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="RETRIEVAL_", extra="ignore")

    top_n: int = 10


class RagRedisSettings(BaseSettings):
    """Redis 缓存配置"""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="REDIS_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 6379
    password: str = ""
    db: int = 0
    exact_ttl: int = 600
    sim_threshold: float = 0.85


class RagMilvusSettings(BaseSettings):
    """向量数据库 (Milvus) 配置"""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="MILVUS_", extra="ignore")

    # 只保留「本机 IP + 默认端口」这类无敏感信息；用户名 / 口令 / 库名 / 集合名
    # 一律留空，只在本地 .env 里配置（对应 MILVUS_USER / MILVUS_PASSWORD /
    # MILVUS_DB_NAME / MILVUS_COLLECTION）。
    uri: str = "http://127.0.0.1:19530"
    user: str = ""
    password: str = ""
    db_name: str = ""
    collection: str = ""


class PaddleOCRSettings(BaseSettings):
    """OCR 服务 (PaddleOCR) 配置"""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="PADDLEOCR_", extra="ignore")

    job_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    token: str = ""
    model: str = "PaddleOCR-VL-1.6"


# ============================================================
# 三、聚合对象：settings 同时支持扁平字段与分组字段
# ============================================================
class Settings:
    """配置聚合对象：

    - settings.api_key            → 扁平字段（Agent 课案）
    - settings.llm.api_key        → RAG 分组字段
    - settings.mysql_url          → 拼接好的连接串
    未在实例上定义的属性一律转发给扁平配置 _CoreSettings。
    """

    def __init__(self) -> None:
        # 扁平配置（Python_Base 原有 + Agent 课案）
        self._core = _CoreSettings()
        # RAG 分组配置
        self.app = AppSettings()
        self.llm = LLMSettings()
        # 评估/评判模型（与生成模型分家，见 EvalLLMSettings 的说明）
        self.eval_llm = EvalLLMSettings()
        self.embedding = EmbeddingSettings()
        self.rerank = RerankSettings()
        self.retrieval = RetrievalSettings()
        self.redis = RagRedisSettings()
        self.milvus = RagMilvusSettings()
        self.paddleocr = PaddleOCRSettings()

    def __getattr__(self, name: str):
        # 只在实例属性里找不到时才会走到这里：转发给扁平配置
        return getattr(self._core, name)


@lru_cache
def get_settings() -> Settings:
    return Settings()


# 全局唯一配置实例（所有代码统一 from config import settings）
settings = get_settings()


def eval_llm_target() -> dict:
    """评判模型**实际生效**的配置（未配置 `EVAL_LLM_*` 时回退到生成模型）。

    返回 `{model, api_key, base_url, temperature, max_tokens, timeout, from_fallback}`：
        - `from_fallback=True` 表示这次评判用的就是生成模型（有自我偏好风险），
          **调用方负责打 WARNING** —— 配置层不 import 日志模块（core.logger 反过来依赖
          config，在 config 里 import 它会成环）。
        - `api_key` / `base_url` 允许**逐项**回退：只配 `EVAL_LLM_MODEL` 是最常见的用法
          （同一家网关换模型），此时密钥与地址沿用 `LLM_*`。

    为什么放在 config 里而不是各调用点各写一遍：Ragas 与 `evaluation/llm_judge.py`
    是两处独立实现，回退规则写两遍必然漂移（一处改了另一处忘）。
    """
    ev = settings.eval_llm
    if ev.configured:
        return {
            "model": ev.model,
            "api_key": ev.api_key or settings.llm.api_key,
            "base_url": ev.base_url or settings.llm.base_url,
            "temperature": ev.temperature,
            "max_tokens": ev.max_tokens,
            "timeout": ev.timeout,
            "from_fallback": False,
        }
    return {
        "model": settings.llm.model,
        "api_key": settings.llm.api_key,
        "base_url": settings.llm.base_url,
        "temperature": settings.llm.temperature,
        "max_tokens": settings.llm.max_tokens,
        "timeout": settings.llm.timeout,
        "from_fallback": True,
    }



if __name__ == "__main__":
    # 启动验证配置加载
    print("配置集加载成功")
    print(f"  APP_ENV: {settings.app_env}")
    print(f"  Agent 模型: {settings.model_name} @ {settings.base_url}")
    print(f"  RAG LLM: {settings.llm.model} @ {settings.llm.base_url}")
    _ev = eval_llm_target()
    print(
        f"  评判模型: {_ev['model']}"
        + ("（未配置 EVAL_LLM_MODEL，回退生成模型 —— 有自我偏好风险）"
           if _ev["from_fallback"] else " @ " + str(_ev["base_url"]))
    )
    print(f"  Embedding: {settings.embedding.model}")
    print(f"  PostgreSQL URL: {settings.postgres_url}")
    print(f"  LangGraph PG_URI: {settings.pg_uri}")
    print(f"  Redis URL: {settings.redis_url}")
    print(f"  Milvus URI: {settings.milvus_uri}")
