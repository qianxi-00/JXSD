"""四条 RAG 线路的**纯常量**（键名、中文名、缓存作用域、是否支持多轮）。

为什么单独放一个模块（而不是塞进 `pipeline/modes.py`）：
    `core/` 是最底层，`cache.py` 需要知道"缓存要按哪些线路分键"，
    而 `pipeline/modes.py` 需要知道"有哪几条线路、叫什么名字"。
    把常量放这里，两边都能 import，且**不会出现 core → pipeline 的反向依赖**
    （那会绕成循环：pipeline 依赖 core，core 又依赖 pipeline）。

    这里刻意**只放常量与文案**，不放 runner（那必然要 import pipeline/agentic/graph）。
    真正的线路注册表在 `pipeline/modes.py`，它的 runner 一律**函数内惰性 import**。

命名有两套，别混：
    - **线路键**（`ROUTE_*`）：给前端/HTTP 用，四个都好记；
    - **缓存作用域**（`CACHE_SCOPES`）：给 Redis 键用。基础线路的作用域是
      **空串**，这样它的缓存键与本功能上线前逐字节相同 —— Redis 里已有的
      运行时缓存和 `seed_details` 播下的 60 条明细继续有效（换键会让它们
      变成不可达的孤儿，且**不报错**，只表现为"怎么全 miss 了"）。
"""

from __future__ import annotations

# 线路键：HTTP 参数、前端选择器、日志里的 mode 值都用它。
ROUTE_BASIC = "basic"
ROUTE_AGENTIC = "agentic"
ROUTE_GRAPH = "graph"
ROUTE_FUSION = "fusion"

# 顺序即前端选择器的显示顺序（从简单到复杂），改动要同步 README 与前端。
ROUTE_KEYS: tuple[str, ...] = (ROUTE_BASIC, ROUTE_AGENTIC, ROUTE_GRAPH, ROUTE_FUSION)

# 中文短名：前端下拉框、路由事件、README 表格共用同一份文案。
ROUTE_LABELS: dict[str, str] = {
    ROUTE_BASIC: "① 基础单轮检索",
    ROUTE_AGENTIC: "② Agentic RAG",
    ROUTE_GRAPH: "③ GraphRAG",
    ROUTE_FUSION: "④ 融合线路（三合一）",
}

# 一句话说明：前端在选择器下面展示，让用户知道"这条线路会做什么"。
ROUTE_SUMMARIES: dict[str, str] = {
    ROUTE_BASIC: "路由 → 双路召回（向量+BM25）→ 重排门控 → 生成。单轮，最快。",
    ROUTE_AGENTIC: "Deep Agents 主控 + 证据子代理核验 + 工具调用（Text-to-SQL）。支持多轮。",
    ROUTE_GRAPH: "知识图谱社区摘要 + 实体多跳遍历。擅长关系型问题，覆盖票据有限。",
    ROUTE_FUSION: "三路证据融合（票据/图谱/结构化统计）+ 数字核验。最慢，信息最全。",
}

# 是否吃会话历史（多轮）。基础线路是**单轮**的 —— 它只把当前问题喂给链路，
# 追问里的代词（"那乐艳的呢？"）无人解析，实测会抽不到过滤条件、退化成保守回复。
ROUTE_SUPPORTS_HISTORY: dict[str, bool] = {
    ROUTE_BASIC: False,
    ROUTE_AGENTIC: True,
    ROUTE_GRAPH: False,
    ROUTE_FUSION: True,
}

# 缓存作用域：基础线路是空串（见模块 docstring 里的兼容说明）。
CACHE_SCOPE_BASIC = ""
CACHE_SCOPES: tuple[str, ...] = (CACHE_SCOPE_BASIC, ROUTE_AGENTIC, ROUTE_GRAPH, ROUTE_FUSION)


def cache_scope(route: str) -> str:
    """线路键 → 缓存作用域。未知线路按"专属作用域"处理（每个拼错的键各占一份，
    避免拼错时静默共用基础线路的缓存而看起来"配置没生效"）。"""
    if route == ROUTE_BASIC:
        return CACHE_SCOPE_BASIC
    return route
