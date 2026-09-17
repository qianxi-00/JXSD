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
    # RAG 评估用的 Langfuse 数据集名(优化篇 script/langfuse_evaluation.py)
    langfuse_dataset_name: str = "finance-rag-validation"

    # ---- Agent 课案里额外用到的第三方平台密钥（全部留空，只在本地 .env 提供）----
    # 百度千帆：课案「多 Agent / 子Agent」用它的联网搜索 MCP
    baidu_qfan_api_key: str = ""
    # Gitee：课案「多 Agent / 路由与合并」用它的代码仓库 MCP
    gitee_api_key: str = ""
    # 阿里云百炼（DashScope）：课案「监控与评估 / RAG评估」用它跑评测模型
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ---- 百炼「原生」API 基址（与上面的 compatible-mode 是两套不同的接口）----
    # ⚠️ 这两个不能混用，混用会 404：
    #   dashscope_base_url = OpenAI 兼容模式，路径 /compatible-mode/v1，
    #                        给 langchain_openai.ChatOpenAI 这类 OpenAI 协议客户端用；
    #   dashscope_api_base = 原生 DashScope API，路径 /api/v1，
    #                        给语音识别(Fun-ASR-Flash) / 声音复刻(CosyVoice) /
    #                        视频合成(VideoSynthesis，数字人对口型) / 临时文件上传 用。
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/api/v1"
    # 可选：百炼「业务空间专属域名」的 Workspace ID。
    # 填了就切到 https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1
    # （部分模型——如爱诗 PixVerse 视频对口型——文档只给这个专属域名形式；
    #   不填则用上面的通用域名 dashscope.aliyuncs.com）。
    dashscope_workspace_id: str = ""

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
    # RAG 优化篇 Text-to-SQL 的票据库(与 LangGraph 记忆用的默认库分开)
    finance_db: str = "finance"

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
    def dashscope_api_endpoint(self) -> str:
        """百炼「原生」API 的实际基址。

        配了 DASHSCOPE_WORKSPACE_ID 就用业务空间专属域名
        （部分模型如爱诗 PixVerse 视频对口型只给这种形式），否则用通用域名。
        语音识别 / 声音复刻 / 视频合成 / 临时文件上传都从这里取基址。
        """
        if self.dashscope_workspace_id:
            return (
                f"https://{self.dashscope_workspace_id}"
                ".cn-beijing.maas.aliyuncs.com/api/v1"
            )
        return self.dashscope_api_base

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

    @property
    def finance_pg_url(self) -> str:
        """Text-to-SQL 票据库的连接 URL(SQLAlchemy + psycopg3 驱动)。

        必须显式带 `+psycopg`:环境里装的是 psycopg 3,SQLAlchemy 默认的
        `postgresql://` 会去找未安装的 psycopg2 并报 ModuleNotFoundError。
        """
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.finance_db}"
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


class EmbeddingSettings(BaseSettings):
    """向量化模型配置"""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="EMBEDDING_", extra="ignore")

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    embedding_size: int = 1024
    batch_size: int = 32
    # 是否把 embedding_size 作为 dimensions 参数下发。
    # 默认 False:不少模型（如 BAAI/bge-m3）不接受该参数，传了直接 400；
    # 需要 MRL 截断的模型（如 Qwen/Qwen3-Embedding-4B，原生 2560 维）才开成 True。
    send_dimensions: bool = False


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


class Neo4jSettings(BaseSettings):
    """RAG 优化篇 GraphRAG 的 Neo4j 图数据库配置"""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="NEO4J_", extra="ignore")

    # 用户名/口令只在 .env 提供(.env 里是 NEO4J_USER / NEO4J_PASSWORD)
    uri: str = "bolt://127.0.0.1:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    # 原生向量索引名(Neo4j 5.11+)
    entity_index: str = "entity_embedding_index"
    community_index: str = "community_embedding_index"


# ============================================================
# 二·补、自媒体 Agent（Media_Agent 子项目）分组配置
# ============================================================
# 对应课案《3.自媒体Agent》。与 RAG 的分组配置同构：前缀 MEDIA_，通过
# settings.media.xxx 访问。
#
# 设计要点（重要）：
#   1. LLM 复用根配置的扁平字段 api_key / base_url / model_name —— 课案代码
#      用的就是 settings.api_key / base_url / model_name，因此零改动。
#      这里只提供一个「可选覆盖」llm_model，留空即跟随根配置。
#   2. 百炼相关统一复用 dashscope_api_key，不重复配置密钥。
#   3. 路径默认值一律给「以本文件所在目录为基准的绝对路径」，
#      避免课案里 ".cache/videos" 这种相对路径随 CWD 漂移的坑。
MEDIA_AGENT_DIR = BASE_DIR / "Media_Agent"


class MediaAgentSettings(BaseSettings):
    """自媒体 Agent（Media_Agent）配置：全部来自根 .env 的 MEDIA_ 前缀键。"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        # ⚠️ 这个 env_prefix 是必须的，漏了它 **所有 MEDIA_* 配置项都不会被读取**。
        #    而且不会报错：字段全部退化成代码里的默认值，
        #    因为默认值恰好与 .env 里的值一致，表面上完全看不出问题。
        #    （实测踩到：MEDIA_ASSET_SSH / MEDIA_ASSET_BASE_URL / MEDIA_VOICE_REF_URL
        #      配了却读不到，排查半天才发现是这里漏了一行。）
        env_prefix="MEDIA_",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM（留空 = 复用根配置的 model_name / get_model 逻辑）----
    llm_model: str = ""
    # DeepAgents 编排层用的模型，留空 = 复用 llm_model（照抄课案 get_deepagent_model 思路）。
    # 视频剪辑那一步链路很长，通常用更稳的模型单独跑。
    deepagent_model: str = ""

    # ---- 语音识别 ASR（替代课案的本地 FunASR）----
    # 百炼 Fun-ASR-Flash / Qwen-Audio-3.0-ASR-Flash，同一接口同时提供
    # 纯文本转写与「句级+词级时间戳」，一个接口替掉课案的两个 FunASR 模型。
    asr_model: str = "qwen-audio-3.0-asr-flash"
    # 语种提示（zh / en / ...）。Fun-ASR 系列只取第一个值。
    asr_language: str = "zh"
    # 音频转 Base64 内嵌的上限（字节）。超过就走百炼临时存储换 oss:// URL。
    asr_inline_max_bytes: int = 10 * 1024 * 1024

    # ---- TTS 语音合成（替代课案的本地 Fish-Speech）----
    tts_enabled: bool = True
    # 声音复刻驱动模型（CosyVoice 系列）。要与 create_voice 时传的 target_model 一致。
    tts_model: str = "cosyvoice-v2"
    # edge-tts 兜底音色（免费、无需密钥、但无法克隆音色）。
    tts_fallback_voice: str = "zh-CN-XiaoxiaoNeural"
    # 克隆音色缓存文件（CosyVoice 建音色有配额，必须复用，不能每次重建）。
    voice_cache_file: str = str(MEDIA_AGENT_DIR / ".cache" / "voices.json")
    # 声音克隆的参考音频公网 URL（可选）。
    # ⚠️ 实测：create_voice 只收**真正的 http(s)**，百炼临时存储的 oss:// 会被拒
    #    （400 InvalidParameter: audio url should start with http or https）。
    # 本项目没有内置公网托管，所以要么在这里给一个已托管好的参考音频 URL，
    # 要么不启用声音克隆（会自动降级到 edge-tts 通用音色 / PixVerse 内置 TTS）。
    voice_ref_url: str = ""

    # ---- 数字人对口型（替代课案的本地 HeyGem）----
    # 爱诗 PixVerse 视频对口型：video + audio（或 video + TTS 文本）→ 对口型视频。
    avatar_model: str = "pixverse/pixverse-lipsync"
    # 数字人任务最长等待秒数（异步任务，官方说 1~5 分钟）。
    avatar_timeout: int = 900

    # ---- 公网素材托管（声音克隆 / 数字人的硬前提，见 tools/asset_host.py）----
    # 为什么需要：百炼的 CosyVoice 声音复刻**不收 oss:// 临时存储**（实测 400），
    # 只认真正的 http(s)；PixVerse 的 video_url/audio_url 同样要求公网 URL。
    # 做法：把本地素材 scp 到自己的服务器静态目录，nginx 只读分发。
    # 三项齐备 is_configured() 才为真；不配则声音克隆自动降级到 edge-tts。
    asset_ssh: str = ""                                    # 形如 ubuntu@1.2.3.4
    asset_remote_dir: str = "/var/www/media-assets"        # 服务器上的静态目录
    asset_base_url: str = ""                               # 形如 http://1.2.3.4/media-assets

    # ---- 图片生成（可选；三项任一为空则降级为本地占位图）----
    image_api_key: str = ""
    image_base_url: str = ""
    image_model: str = ""

    # ---- 热点抓取（NewsNow 聚合 API，公共接口无需密钥，可自部署）----
    trendradar_api_url: str = "https://newsnow.busiyi.world/api/s"

    # ---- 数据复盘：抖音作品数据采集 ----
    # 课案原用本地 erma0/douyin 项目（该项目已因合规审查清空），
    # 现改为自托管 Evil0ctal/Douyin_TikTok_Download_API 的 REST 接口。
    douyin_api_base: str = "http://127.0.0.1:8080"
    # 抖音登录 Cookie（采集本人主页数据需要）。留空则只能走「手动粘贴数据」降级入口。
    douyin_cookie: str = ""

    # ---- 运行时输出目录 / 素材 ----
    video_output_dir: str = str(MEDIA_AGENT_DIR / ".cache" / "videos")
    image_output_dir: str = str(MEDIA_AGENT_DIR / ".cache" / "images")
    # 数字人模特视频目录（课案里是 HeyGem 的 Docker 挂载目录，这里改成本地目录）。
    avatar_input_dir: str = str(MEDIA_AGENT_DIR / ".cache" / "avatars")
    # 视频剪辑的背景音乐路径。留空则不混音（课案在这里硬编码了作者本机路径）。
    bgm_path: str = ""
    # 视频剪辑的 deepagent 沙箱根目录（文件工具与 execute 的工作目录）。
    mashup_work_dir: str = str(MEDIA_AGENT_DIR / ".cache" / "mashup")

    # ---------------- 方法 ----------------
    # 说明：模型名的「回退到根配置」逻辑放在下面的 Settings 聚合类里实现
    # （只有聚合类同时能看到根字段 model_name 和这里的 media.llm_model）。

    @staticmethod
    def _resolve(p: str, fallback_dir: Path) -> str:
        """把配置里的目录解析成绝对路径并确保存在。

        相对路径一律相对 Media_Agent 目录解析 —— 避免课案里
        ".cache/videos" 随当前工作目录漂移、找不到文件的坑。
        """
        if not p:
            return ""
        path = Path(p)
        if not path.is_absolute():
            path = fallback_dir / path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def get_video_output_dir(self) -> str:
        return self._resolve(self.video_output_dir or ".cache/videos", MEDIA_AGENT_DIR)

    def get_image_output_dir(self) -> str:
        return self._resolve(self.image_output_dir or ".cache/images", MEDIA_AGENT_DIR)

    def get_avatar_input_dir(self) -> str:
        return self._resolve(self.avatar_input_dir or ".cache/avatars", MEDIA_AGENT_DIR)

    def get_mashup_work_dir(self) -> str:
        return self._resolve(self.mashup_work_dir or ".cache/mashup", MEDIA_AGENT_DIR)


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
        self.embedding = EmbeddingSettings()
        self.rerank = RerankSettings()
        self.retrieval = RetrievalSettings()
        self.redis = RagRedisSettings()
        self.milvus = RagMilvusSettings()
        self.paddleocr = PaddleOCRSettings()
        self.neo4j = Neo4jSettings()
        # 自媒体 Agent（Media_Agent 子项目）分组配置
        self.media = MediaAgentSettings()

    def __getattr__(self, name: str):
        # 只在实例属性里找不到时才会走到这里：转发给扁平配置
        return getattr(self._core, name)

    # ---------- Media_Agent 的模型名解析 ----------
    # 放这里而不是 MediaAgentSettings 内部：只有聚合类同时看得到
    # 根字段 model_name（扁平）和 media.llm_model（分组）。
    def media_llm_model(self) -> str:
        """自媒体链路文本模型：MEDIA_LLM_MODEL 优先，留空则复用根 MODEL_NAME。"""
        return self.media.llm_model or self.model_name

    def media_deepagent_model(self) -> str:
        """DeepAgent 用模型：MEDIA_DEEPAGENT_MODEL → MEDIA_LLM_MODEL → 根 MODEL_NAME。"""
        return self.media.deepagent_model or self.media_llm_model()


@lru_cache
def get_settings() -> Settings:
    return Settings()


# 全局唯一配置实例（所有代码统一 from config import settings）
settings = get_settings()


if __name__ == "__main__":
    # Windows 控制台默认 GBK，这里打印的是中文，不重配置会乱码
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 启动验证配置加载
    print("配置集加载成功")
    print(f"  APP_ENV: {settings.app_env}")
    print(f"  Agent 模型: {settings.model_name} @ {settings.base_url}")
    print(f"  RAG LLM: {settings.llm.model} @ {settings.llm.base_url}")
    print(f"  Embedding: {settings.embedding.model}")
    print(f"  PostgreSQL URL: {settings.postgres_url}")
    print(f"  LangGraph PG_URI: {settings.pg_uri}")
    print(f"  Redis URL: {settings.redis_url}")
    print("  Milvus URI: " + settings.milvus_uri)
    print(f"  自媒体 Agent 模型: {settings.media_llm_model()} @ {settings.base_url}")
    print(f"  自媒体 Agent 编排模型: {settings.media_deepagent_model()}")
    print(f"  百炼原生 API: {settings.dashscope_api_endpoint}")
    print(f"  百炼密钥已配置: {bool(settings.dashscope_api_key)}")
