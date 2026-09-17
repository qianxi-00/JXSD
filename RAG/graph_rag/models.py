"""GraphRAG 的图模型定义：neomodel 节点/关系 + Neo4j 连接与统一 Cypher 出口。

与课案写法的差异（安装的是 neomodel 7.0.0，课案按 neomodel 5.x 写）：
1. 连接配置
   课案：`from neomodel import config; config.DATABASE_URL = "bolt://user:pwd@host:7687"`
   7.0：模块级 `config.DATABASE_URL` 只是向后兼容的过渡属性，赋值会发
        DeprecationWarning 并提示改用 `get_config()`。这里统一走
        `get_config().database_url / database_name`（现代 API），
        并且**不使用 `db.set_connection()` 在导入期建连**——那样会让"只导入模型"
        也去连数据库，离线单测根本没法跑。连接延迟到第一次查询前由 `connect()` 配置。
2. 向量索引
   7.0 新增了 `neomodel.VectorIndex(dimensions, similarity_function)`，可以写在
       属性上：`embedding = ArrayProperty(FloatProperty(), vector_index=VectorIndex(...))`。
       但它**不允许指定索引名**（名字由 neomodel 自己生成），而课案要求索引名取
       `settings.neo4j.entity_index / community_index`，所以这里仍按课案用
       `CREATE VECTOR INDEX ... IF NOT EXISTS` 原生 DDL 建索引（幂等）。
       两者效果一致：都是 Neo4j 5.11+ 的原生向量索引。
3. `StructuredNode.get_or_create()` 在 7.0 已标记 deprecated（提示改用
   `MyNode.nodes.bulk_get_or_create()`），但本模块需要的语义是"存在则按业务规则合并、
   不存在则新建并写向量"，MERGE 表达不了"累加 mentions / 加权平均 confidence"，
   所以 builder 里用 `nodes.get_or_none()` + 显式合并，语义更清楚也可离线单测。
4. `ArrayProperty` 的默认值用 `default=list`（可调用对象）而不是 `default=[]`：
   neomodel 对非 callable 的 default 会直接返回同一个对象，多个实例会共享一个列表。
5. 两处向量的**写入时机不同**，这决定了两条向量检索路径的语义：
   `Entity.embedding` 只在**新建节点**时算一次（文本 = "名称 + 描述"），
   之后描述被合并变长也**不会**重算（见 builder.upsert_entity）；
   `Community.embedding` 则在每次生成社区摘要时重算（摘要换了向量就换）。

本模块在整个 GraphRAG 里的位置（谁写、谁读）：
    写：`init_schema()` 建唯一性约束 + 两个原生向量索引 →
        `builder.upsert_entity` 写实体与实体向量 →
        `builder.detect_communities_louvain` 写 `Entity.community_id` →
        `builder.generate_community_summaries` 写 `Community.summary / embedding`
    读：`retriever.match_communities` 查社区向量索引 →
        `retriever.vector_match_entities` / `builder.match_entity` 查实体向量索引 →
        `retriever.retrieve_by_entities` 沿 `RELATES_TO` 多跳遍历
注意这里是**两条并存**的数据库访问路径：Cypher 走本模块的 `run_cypher`，
节点/关系的增删改走 neomodel ORM（builder 里的 `Entity` / `Community`）。
两条路依赖的都是 `connect()` 灌进 neomodel 的那一份**进程级**配置。
"""

from urllib.parse import quote, urlparse, urlunparse

from neomodel import (
    ArrayProperty,
    FloatProperty,
    IntegerProperty,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    StructuredRel,
    db,
    get_config,
)

from config import settings
from core.logger import logger

# 默认相似度阈值等常量放在 builder/retriever 里会更贴近用法，这里只放模型相关的东西。


class RelatesTo(StructuredRel):
    """实体之间的关系（带属性）。

    关系的唯一性由 Neo4j 保证"同一对节点之间同类型关系只可能有一条"，
    重复抽取到同一条关系时按 `weight` 做加权平均（见 builder.merge_relation_props）。
    """

    # 关系语义，例如"转账""报销""属于"
    relation_type = StringProperty(default="RELATES_TO")
    # 置信度：多次抽取同一条关系时按历史权重做加权平均
    confidence = FloatProperty(default=1.0)
    # 该关系被观察到多少次（加权平均的权重）
    weight = IntegerProperty(default=1)
    # 支撑该关系的文档 id 列表（去重）
    doc_ids = ArrayProperty(StringProperty(), default=list)


class Entity(StructuredNode):
    """图谱实体节点。

    `embedding` 存的是「名称 + 描述」的向量，用于实体级向量召回；
    这也是 Neo4j 原生向量索引的落点（见 ensure_vector_indexes）。
    """

    # 实体名，全局唯一（也是增量入库时的精确匹配键）
    name = StringProperty(unique_index=True, required=True)
    # 实体类型，例如 人物 / 公司 / 发票 / 航班
    entity_type = StringProperty(default="UNKNOWN")
    # 实体描述，多次抽取同一实体时逐段合并
    description = StringProperty(default="")
    # 支撑该实体的文档 id 列表（去重）
    doc_ids = ArrayProperty(StringProperty(), default=list)
    # 被提及次数，每抽取到一次 +1
    mentions = IntegerProperty(default=1)
    # 所属社区（Louvain 检测写回）；-1 表示还没分配社区
    #
    # ⚠ 已知语义冲突（历史遗留，注释说明而不是改默认值）："-1 = 未分配" 这条约定
    #   只在 builder.group_entities_by_community 里被遵守（它按 `< 0` 丢弃），
    #   而查社区用的 Cypher（如 _CYPHER_ALL_ENTITIES_WITH_COMMUNITY、社区限定遍历）
    #   用的是 `community_id IS NOT NULL` —— 对它们来说 -1 也算"已分配社区"。
    #   现状不会出错：新建节点是 -1，而 Louvain 写回的编号恒 >= 0，
    #   所以"值为 -1 的实体"要么还没被分区（那时两条判据都不该放行）、要么已经被改写。
    #   但任何新写的 Cypher 都要先想清楚自己属于哪一套判据。
    community_id = IntegerProperty(default=-1)
    # 「名称 + 描述」的向量，维度 = settings.embedding.embedding_size
    embedding = ArrayProperty(FloatProperty(), default=list)

    # 实体之间的有向关系，关系属性用 RelatesTo
    relates_to = RelationshipTo("Entity", "RELATES_TO", model=RelatesTo)


class Community(StructuredNode):
    """社区节点：一个社区一段 LLM 摘要 + 摘要向量。

    分层检索的第一步就是拿查询向量去社区摘要向量索引上召回语义区域。

    ⚠ 生命周期（已知缺陷，见"发现的问题"）：`generate_community_summaries()`
    只**更新/新建**当前划分里存在的社区，**不删除**已经不存在的 Community 节点。
    而 Louvain 的社区编号只是内部分区顺序号，重算后同一个 `community_id`
    可能对应完全不同的实体集合 ⇒ 残留节点的摘要既过时、又照样会被社区向量检索召回
    （`retriever.match_communities` 只查 `Community` 标签，不校验该社区是否还有实体）。
    要彻底重来请走 `builder.create_graph()`（清空整图）或手动删 `Community` 节点。
    """

    # Louvain 产出的社区编号，全局唯一
    community_id = IntegerProperty(unique_index=True, required=True)
    # 100~200 字的社区摘要
    summary = StringProperty(default="")
    # 摘要向量
    embedding = ArrayProperty(FloatProperty(), default=list)


# ============================================================
# 连接与 Cypher 出口
# ============================================================
# 只连一次；bool 而非 driver 句柄，避免在模块里持有连接对象
#
# ⚠ 这里省掉锁是刻意的：neomodel 的配置是**进程级全局**（`get_config()` 返回单例），
#   不按请求/按线程隔离。FastAPI 的同步视图在线程池里跑，多线程首次并发调用
#   `connect()` 只会重复写同样的值（幂等、无副作用），加锁反而把建图脚本的
#   串行流程复杂化。真正的连接由 neomodel 自己惰性创建并由它管理，不需要我们保护。
_connected = False


def build_database_url() -> str:
    """把 settings.neo4j 拼成 neomodel 需要的 `bolt://user:pwd@host:port` 形式。

    口令必须百分号编码：口令里一旦出现 `@` / `:` / `/`，不编码会直接把 URL 解析坏
    （neomodel 按最后一个 `@` 切分，用户名按第一个 `:` 切分）。

    `settings.neo4j.uri` 必须带 scheme（本项目 `.env` 里是 `bolt://127.0.0.1:7687`）：
    `urlparse("127.0.0.1:7687")` 会把 `127.0.0.1` 当成 scheme、`hostname` 变成 None，
    于是 host 静默退化成兜底值 —— 看起来"连上了"，其实连的是本机默认端口。
    同理 uri 里**不要**塞账号口令：本函数只从 uri 取 scheme/host/port，
    账号口令一律取自 `settings.neo4j.user / password` 两个独立字段。
    """
    cfg = settings.neo4j
    parsed = urlparse(cfg.uri)
    scheme = parsed.scheme or "bolt"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 7687
    user = quote(cfg.user, safe="")
    password = quote(cfg.password, safe="")
    netloc = f"{user}:{password}@{host}:{port}"
    # path 留空：库名交给 config.database_name 控制，不写进 URL（URL 里的库名优先级更高）
    return urlunparse((scheme, netloc, "", "", "", ""))


def connect() -> None:
    """把 Neo4j 配置灌进 neomodel；幂等，可重复调用。

    只写配置、不建连接：真正的 driver 由 neomodel 在第一次查询时惰性创建，
    所以"服务没起"这件事只会在查询时报错，不会在 import 时报错。

    调用点约定：本模块里只有 `ensure_constraints()` **不自带** connect()，它依赖调用方
    先把配置灌好（`init_schema()` 和 builder 的各个入口都已经先连）。单独调用
    `ensure_constraints()` 之前要自己先 connect()，否则 `run_cypher` 会在没有配置的
    情况下建连并报错。

    ⚠ 日志里**不要**打印 `build_database_url()` 的返回值：那串里带明文口令。
    这里只打 `settings.neo4j.uri`（本项目该字段不含账号口令）用于排错。
    """
    global _connected
    if _connected:
        return
    cfg = get_config()
    cfg.database_url = build_database_url()
    cfg.database_name = settings.neo4j.database
    _connected = True
    logger.info(f"[GraphRAG] Neo4j 配置就绪: {settings.neo4j.uri}/{settings.neo4j.database}")


def run_cypher(query: str, params: dict | None = None) -> list[dict]:
    """执行 Cypher 并把结果整形为 `[{列名: 值}, ...]`。

    这是本模块**唯一的数据库出口**，builder / retriever 都走它：
    既统一了结果整形，也让离线单测可以整体替换掉数据库访问。

    离线单测的替换点（写错会"patch 了但没生效"，实测约定）：
    builder / retriever 用的是 `from graph_rag.models import ... run_cypher`，
    `from ... import` 绑定的是**函数对象本身**，所以必须 patch **使用方模块**里的名字
    （`monkeypatch.setattr(builder, "run_cypher", fake)`）；
    只 patch `graph_rag.models.run_cypher` 对它们无效 —— 只有 models 自己调用时才走后者。

    整形取舍：`dict(zip(columns, row))` 丢掉了列顺序、也丢掉了同名重复列
    （后一个覆盖前一个）。现有 Cypher 都不返回重名列，够用；要返回两列同名就得改这里。
    `params or {}` 而不是默认参数 `{}`：避免可变默认值被跨调用共享。
    """
    rows, columns = db.cypher_query(query, params or {})
    return [dict(zip(columns, row)) for row in rows]


# ============================================================
# 向量索引（Neo4j 5.11+ 原生向量索引）
# ============================================================
# 为什么用 .format() 拼字符串而不是 Cypher 参数（$name / $dims）：
#   Neo4j 的 DDL（CREATE INDEX）**不接受查询参数**，索引名与 indexConfig 的值必须是字面量。
#   所以只能拼 —— 但 name / dims 全部来自 settings（不是用户输入），不存在注入面。
#   模板里的 `{{ }}` 是 str.format 的转义写法，渲染后就是 Cypher 的 `{ }` 映射。
# 两个索引的 DDL 只差节点标签（Entity / Community），故意不合并成一个函数：
#   以后某一边要改（比如换相似度函数、加过滤属性）时不会互相牵连。
_ENTITY_VECTOR_INDEX_DDL = """
CREATE VECTOR INDEX {name} IF NOT EXISTS
FOR (n:Entity) ON (n.embedding)
OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, `vector.similarity_function`: '{similarity}'}}}}
"""

_COMMUNITY_VECTOR_INDEX_DDL = """
CREATE VECTOR INDEX {name} IF NOT EXISTS
FOR (n:Community) ON (n.embedding)
OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, `vector.similarity_function`: '{similarity}'}}}}
"""


def ensure_vector_indexes() -> None:
    """在两个 embedding 属性上建原生向量索引（幂等）。

    课案写法（Neo4j 5.11+）：
        CREATE VECTOR INDEX <name> IF NOT EXISTS FOR (n:Entity) ON (n.embedding)
        OPTIONS {indexConfig: {`vector.dimensions`: N, `vector.similarity_function`: 'cosine'}}
    维度取自 settings.embedding.embedding_size（本项目 1024），
    相似度固定 cosine（与 embedding 模型的归一化语义一致）。

    建完等索引 ONLINE 再返回：索引刚创建时可能还在 POPULATING，
    此时 `db.index.vector.queryNodes` 会报 "no such index"，awaitIndexes 可以避免这个坑。

    `CALL db.awaitIndexes(300)` 里的 300 是**秒**（超时上限）；索引都已 ONLINE 时
    立即返回，几乎零成本，只有首次创建（要建 1024 维向量索引）才真的等。
    幂等但**非零成本**：每调用一次就多等一次，别放进高频路径。
    维度取自配置 ⇒ **换 embedding 模型（维度变了）必须重建索引**，
    `IF NOT EXISTS` 会跳过已存在的旧索引，导致新旧维度不匹配、写入直接失败。
    """
    connect()
    dims = settings.embedding.embedding_size
    for name, template in (
        (settings.neo4j.entity_index, _ENTITY_VECTOR_INDEX_DDL),
        (settings.neo4j.community_index, _COMMUNITY_VECTOR_INDEX_DDL),
    ):
        run_cypher(
            template.format(name=name, dims=dims, similarity="cosine").strip()
        )
        logger.info(f"[GraphRAG] 向量索引就绪: {name} (dim={dims}, cosine)")
    run_cypher("CALL db.awaitIndexes(300)")


def ensure_constraints() -> bool:
    """确保节点唯一性约束存在（幂等）。

    为什么要先查再建：neomodel 的 `install_all_labels()` 是"无脑建一遍、撞到
    EquivalentSchemaRuleAlreadyExists 就吞掉"，每次调用都会往 stdout 打一堆
    "Setting up indexes and constraints..." 和红字报错。约束已经在时直接跳过，
    输出干净，也不再依赖 neomodel 的异常吞并行为。

    返回是否真的执行了安装。

    前置条件：调用方必须先 `connect()`（本函数直接 `run_cypher`，不自带连接）——
    这是它与 `ensure_vector_indexes()` 的唯一差别，后者自己会 connect。
    `SHOW CONSTRAINTS YIELD labelsOrTypes, properties` 是 Neo4j 5 的语法
    （4.x 叫 `SHOW CONSTRAINT` 且列名不同），本项目跑的是 5.11+ 才敢这么写。
    判据是"两条约束都在就整体跳过"：只要缺一条就整份重装（反正 neomodel 会吞掉已存在的）。
    """
    rows = run_cypher(
        "SHOW CONSTRAINTS YIELD labelsOrTypes, properties RETURN labelsOrTypes, properties"
    )
    existing = {
        (tuple(row.get("labelsOrTypes") or []), tuple(row.get("properties") or []))
        for row in rows
    }
    if (("Entity",), ("name",)) in existing and (("Community",), ("community_id",)) in existing:
        return False
    db.install_all_labels()
    logger.info("[GraphRAG] 节点唯一性约束就绪")
    return True


def init_schema() -> None:
    """初始化图 schema：唯一性约束 + 向量索引。

    幂等，可重复调用。建图脚本启动时调用一次即可。
    """
    connect()
    ensure_constraints()
    ensure_vector_indexes()
