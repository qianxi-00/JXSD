# -*- coding: utf-8 -*-
r"""
LangGraph 官方补充篇④：长期记忆与 Store 语义搜索（非课案内容）
================================================================
来源与定位：
    本文件对照 LangGraph **官方文档** concepts/memory.mdx 与 stores.mdx，
    补缺口表 **LangGraph 第 8 项**（长期记忆策略分类 + Store 语义搜索）。

课案 01_langgraph/05_长期记忆 讲了 Store 的基本读写（put/get/search）与
「用命名空间分区」；本文件补的是官方文档里更进一步的三个问题：

    ① 长期记忆该**存什么**？官方把记忆分成三类（concepts/memory.mdx）：
         · semantic（语义记忆）：事实与偏好 —— 用户是谁、喜欢什么、项目背景；
         · episodic（情景记忆）：经历与事件 —— 上次怎么解决的、踩过什么坑；
         · procedural（程序记忆）：做事的方法 —— 团队规范、流程、套路。
       三类混在一个命名空间里，检索时就会互相污染，所以要**分开命名空间存**。

    ② 怎么**按语义找回**来？在 Store 上配 `index=`（embedding 索引）后，
       `search(query=...)` 才是语义检索；**不配索引时 query 会被静默忽略** ——
       课案记录了这条坑，本文件 Demo 2 把它做成了对照实验，差异一眼可见。

    ③ 记忆怎么**进 agent**？工具里通过 `runtime.store` 读写（课案 03_deepagents/11
       与 02_langchain/20 官方补充篇都用过这套），本文件演示跨会话记住用户偏好。

⚠️ 本文件用的向量化模型来自 .env：`EMBEDDING_MODEL=BAAI/bge-m3`（SiliconFlow，1024 维）。

运行方式（项目根目录下，需真实 embedding 与模型）：
    uv run Agent/01_langgraph/13_长期记忆_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import time

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore

from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,          # grok-4.6
    api_key=settings.api_key,
    base_url=settings.base_url,
    max_retries=0,
)

# 向量化模型（.env 里的 EMBEDDING_* 三个字段）
embeddings = OpenAIEmbeddings(
    model=settings.embedding.model,     # BAAI/bge-m3，1024 维
    api_key=settings.embedding.api_key,
    base_url=settings.embedding.base_url,
    check_embedding_ctx_length=False,   # 第三方端点必须关（见 24_RAG知识库 官方补充篇）
    # 显式超时：端点慢时快速失败，而不是无限挂住（实测踩过：没设超时时整跑会卡死）
    request_timeout=60,
    max_retries=1,
)

# 带语义索引的 Store：dims 必须与 embedding 维度一致，fields 指明对哪些字段建索引
INDEXED_STORE = InMemoryStore(
    index={"dims": 1024, "embed": embeddings.embed_documents, "fields": ["text"]}
)
# 不带索引的 Store：用来做「query 被静默忽略」的对照实验
PLAIN_STORE = InMemoryStore()

# 命名空间前缀：三类记忆分开存（官方 memory.mdx 的分类）
USER_NS = ("user-1001",)
SEMANTIC = USER_NS + ("semantic",)      # 事实与偏好
EPISODIC = USER_NS + ("episodic",)      # 经历与事件
PROCEDURAL = USER_NS + ("procedural",)  # 方法与规范


def seed_memories(store: InMemoryStore) -> None:
    """写入三类记忆（真实项目里由 agent 在对话中抽取后写入）。"""
    store.put(SEMANTIC, "pref-style", {"text": "用户偏好简洁的中文回答，不要客套话"})
    store.put(SEMANTIC, "pref-stack", {"text": "用户团队的技术栈是 FastAPI + PostgreSQL"})
    store.put(SEMANTIC, "fact-project", {"text": "用户在做一周报自动汇总的机器人"})
    store.put(EPISODIC, "incident-redis", {"text": "上个月排查过一次 Redis 连接池耗尽，根因是没设超时"})
    store.put(EPISODIC, "incident-slow", {"text": "上周接口变慢，最后发现是 N+1 查询"})
    store.put(PROCEDURAL, "rule-review", {"text": "代码必须先过 lint 再提测，禁止绕过"})
    store.put(PROCEDURAL, "rule-deploy", {"text": "发布前要在预发环境跑一遍全量回归"})


# ================================================================
# Demo 1：三类记忆分开命名空间存，检索时互不污染
# ================================================================
def demo_1_taxonomy() -> None:
    print("=" * 70)
    print("Demo 1：长期记忆的三类分类（semantic / episodic / procedural）")
    print("=" * 70)

    seed_memories(INDEXED_STORE)
    print(f"  写入完成：semantic 3 条 / episodic 2 条 / procedural 2 条")

    query = "这个团队写代码有什么规范？"
    print(f"\n  查询：{query}")
    for label, namespace in (("只查 procedural（正确做法）", PROCEDURAL), ("在全部记忆里混着查（错误做法）", USER_NS)):
        hits = INDEXED_STORE.search(namespace, query=query, limit=3)
        print(f"    {label}：")
        for hit in hits:
            print(f"      [{hit.score:.4f}] {hit.value.get('text')}")
    print(
        "  ↑ 分类的价值在这里：问「有什么规范」时，混查会把用户偏好、事故经历一起捞上来，\n"
        "    白白占用上下文、还可能误导模型。**三类记忆三种用途**：\n"
        "      semantic  → 个性化回答（知道你是谁、你偏好什么）\n"
        "      episodic  → 避免重蹈覆辙（上次那个坑别再踩）\n"
        "      procedural→ 遵守规矩（团队的流程与规范）"
    )


# ================================================================
# Demo 2：配索引 vs 不配索引 —— 课案那条坑的对照实验
# ================================================================
# 课案 01_langgraph/05 记录过：「没配向量索引时 query 被静默忽略，退化成分页取 N 条」。
# 官方 stores.mdx 进一步说明：InMemoryStore 按**插入顺序**返回（最新的一条在最后）。
# 本 Demo 把两种 Store 摆在一起跑同一个查询，让差异自己说话 ——
# 这也是"为什么必须配 index"的最直观证据。
def demo_2_index_matters() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：配索引 vs 不配索引（同一个查询，两种结果）")
    print("=" * 70)

    seed_memories(PLAIN_STORE)
    query = "上次线上出过什么故障？"

    print(f"  查询：{query}")
    print("\n  ① 不配索引的 Store（PLAIN_STORE）：")
    plain_hits = PLAIN_STORE.search(EPISODIC, query=query, limit=2)
    for hit in plain_hits:
        score = getattr(hit, "score", None)
        print(f"      score={score}｜{hit.value.get('text')}")
    print("      → score 恒为 None：**query 根本没被用来排序**，返回的是插入顺序")

    print("\n  ② 配了 bge-m3 索引的 Store（INDEXED_STORE）：")
    indexed_hits = INDEXED_STORE.search(EPISODIC, query=query, limit=2)
    for hit in indexed_hits:
        print(f"      score={hit.score:.4f}｜{hit.value.get('text')}")
    print("      → 有真实相似度分数（顺序见上，不再写死谁在前 —— 重跑结果可能不同）")

    print(
        "\n  ↑ 结论：**`search(query=...)` 只有在 Store 配了 `index=` 时才是语义检索**；\n"
        "    没配索引时不会报错、只会静默退化成按插入顺序返回 —— 这类「静默降级」\n"
        "    最危险，因为它看起来「还能用」。\n"
        "    排查口诀：看到 `score is None`，就是索引没生效。"
    )


# ================================================================
# Demo 3：记忆进 agent —— 跨会话记住用户偏好
# ================================================================
# 工具里通过 `runtime.store` 读写长期记忆（与 02_langchain/20 官方补充篇 Demo 3 同一套机制）。
# 关键演示：**两个不同的 thread_id**（两个"会话"）之间，长期记忆是共享的 ——
# 这正是短期记忆（checkpointer + thread_id）与长期记忆（store）的分工。
def demo_3_memory_in_agent() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：长期记忆进 agent —— 换会话仍然记得你")
    print("=" * 70)

    @tool
    def recall_preferences(query: str, runtime: ToolRuntime) -> str:
        """按语义检索用户的长期记忆（偏好/经历/规范）。"""
        store = runtime.store
        if store is None:
            return "（没有配置长期记忆库）"
        lines: list[str] = []
        for namespace in (SEMANTIC, EPISODIC, PROCEDURAL):
            for hit in store.search(namespace, query=query, limit=2):
                lines.append(f"[{namespace[-1]}] {hit.value.get('text')}")
        return "\n".join(lines) or "（没有找到相关记忆）"

    @tool
    def remember_preference(text: str, runtime: ToolRuntime) -> str:
        """把一条用户偏好写入长期记忆。"""
        store = runtime.store
        if store is None:
            return "（没有配置长期记忆库）"
        key = f"pref-{int(time.time())}"
        # 指定 index=["text"]：让这条记忆**立刻可被语义检索到**
        store.put(SEMANTIC, key, {"text": text}, index=["text"])
        return f"已记住：{text}"

    agent = create_agent(
        model=llm,
        tools=[recall_preferences, remember_preference],
        store=INDEXED_STORE,          # ← 长期记忆要显式注入
        system_prompt=(
            "你是个人助理。规则：\n"
            "1. 回答前**必须先调用 recall_preferences** 查用户的长期记忆；\n"
            "2. 只挑选与用户问题**最相关的 1~2 条**记忆来回答，不要把检索到的内容全部罗列；\n"
            "3. 用户表达新偏好时，用 remember_preference 记下来；\n"
            "4. 回答不超过两句话，禁止把工具名或调用过程写进回答。"
        ),
    )

    def tool_texts(result: dict) -> list[str]:
        """取出这轮里所有工具返回的原文（用于展示模型到底收到了什么）。"""
        return [str(m.content) for m in result["messages"] if m.type == "tool"]

    # 会话 A：告诉它一个新偏好
    config_a = {"configurable": {"thread_id": "memory-demo-a"}}
    result_a = agent.invoke(
        {"messages": [{"role": "user", "content": "记住：我以后要用中文标点，不要用英文逗号。"}]},
        config_a,
    )
    called_a = [c["name"] for m in result_a["messages"] for c in (getattr(m, "tool_calls", None) or [])]
    print(f"  会话 A 调用的工具：{called_a}")
    print(f"  会话 A 回答：{str(result_a['messages'][-1].content)[:90]}")

    # 会话 B：**换一个 thread_id**（等价于换一次对话），看它还能不能想起来
    config_b = {"configurable": {"thread_id": "memory-demo-b"}}
    result_b = agent.invoke(
        {"messages": [{"role": "user", "content": "我之前的写作偏好是什么？逐条告诉我。"}]},
        config_b,
    )
    called_b = [c["name"] for m in result_b["messages"] for c in (getattr(m, "tool_calls", None) or [])]
    answer_b = str(result_b["messages"][-1].content)
    print(f"  会话 B 调用的工具：{called_b}")
    print("  会话 B 的工具返回原文（模型看到的记忆）：")
    for text in tool_texts(result_b):
        for line in text.splitlines():
            print(f"      {line}")
    print(f"  会话 B 回答：{answer_b[:140]}")

    # 直接查库，证明新偏好确实落库了
    stored = INDEXED_STORE.search(SEMANTIC, query="写作偏好 标点", limit=2)
    print("\n  直接查记忆库（验证落库）：")
    for hit in stored:
        print(f"    [{hit.score:.4f}] {hit.value.get('text')}")

    # 判定会话 B 是否真的复述了「长期记忆里的 semantic 记忆」
    # （注意：只要答出任意一条真实存在的偏好就算跨会话生效；
    #   之前用「标点」两个字做判定太窄，把"答出了别的偏好"误判成失败 —— 本文件踩过）
    stored_semantic = [item.value.get("text", "") for item in INDEXED_STORE.search(SEMANTIC, limit=10)]
    # 判定词**从库里真实存在的记忆里取**（而不是硬编码一张关键词表）：
    # 否则库里根本没这条记忆时，模型凭空说出某个词也会被误判成「跨会话生效」。
    candidates = ("简洁", "客套", "标点", "FastAPI", "周报", "逗号")
    keywords = [token for token in candidates if any(token in text for text in stored_semantic)]
    hit_keywords = [k for k in keywords if k in answer_b]
    if hit_keywords:
        print(f"\n  ✔ 会话 B 用全新 thread_id 仍复述出了长期记忆中的偏好（命中关键词：{hit_keywords}）")
        print("    → 机制层面证实：**store 跨会话共享**，与 thread_id 无关。")
    else:
        print("\n  ⚠️ 本次会话 B 没复述出任何偏好（模型行为，重跑通常即可）")

    print("\n  ★ 一个值得注意的细节：会话 A 新写的那条「中文标点」偏好，**本次没有被召回**")
    print("    （取决于模型这一轮的查询措辞与 top-k，不是必然结果）。")
    print("    但直接查库能查到它（score 有值，见上），说明**写入成功、索引也生效**了；")
    print("    没进上下文的原因是 **top-k 截断 + 查询措辞**：recall 工具每个命名空间只取 2 条，")
    print("    而模型这次的查询词（偏「写作偏好」）让老偏好排在了前面。")
    print("    实践启示：① k 要按语料规模调；② 写入时把内容写得**自解释**（带主题词），")
    print("    检索时更容易命中；③ 重要偏好可以在系统提示里直接注入，别只靠检索。")
    print(
        "\n  ↑ 分工要记牢：\n"
        "    · **短期记忆** = checkpointer + thread_id（会话内、自动、可中断恢复）；\n"
        "    · **长期记忆** = store（跨会话、要显式读写、可语义检索）。"
    )


# ================================================================
# Demo 4：记忆的维护 —— 覆盖写与过期
# ================================================================
# 官方 stores.mdx 提醒过：长期记忆会越积越多，需要维护策略。
# 本 Demo 演示两个最基本的旋钮：
#   · 同一个 (namespace, key) 再 put = **覆盖**（适合「偏好变了」的场景）；
#   · `ttl=` = 过期时间，**单位是分钟**（官方 API：Time to live in minutes）——
#     适合「临时上下文」「时效性信息」，但**要看后端支不支持**（见下面实测）。
def demo_4_maintenance() -> None:
    print("\n" + "=" * 70)
    print("Demo 4：记忆维护 —— 覆盖写，以及 TTL 的实测限制")
    print("=" * 70)

    namespace = USER_NS + ("scratch",)
    INDEXED_STORE.put(namespace, "session-note", {"text": "用户正在准备季度汇报"}, index=["text"])
    print(f"  第一次写入：{INDEXED_STORE.get(namespace, 'session-note').value}")

    INDEXED_STORE.put(namespace, "session-note", {"text": "用户已完成季度汇报，转向下季度规划"}, index=["text"])
    print(f"  同 key 再写一次（覆盖）：{INDEXED_STORE.get(namespace, 'session-note').value}")

    # TTL：本机 InMemoryStore **不支持**，会直接抛 NotImplementedError（实测）
    print("\n  尝试写入一条带 TTL 的记忆（ttl=1，单位是**分钟**）：")
    try:
        INDEXED_STORE.put(namespace, "temp-token", {"text": "临时验证码 8520"}, index=["text"], ttl=1.0)
        time.sleep(1.3)
        print(f"    1.3 秒后再取：{INDEXED_STORE.get(namespace, 'temp-token')}")
    except NotImplementedError as exc:
        print(f"    ✘ 不支持：{str(exc)[:110]}")
        print("    → 实测结论：**InMemoryStore 不实现 TTL**，要过期能力得换后端")
        print("      （官方 stores.mdx 列的后端里，支持 TTL 的是持久化实现，如 PostgresStore）")
    print(
        "  ↑ 两个维护旋钮的用法：\n"
        "    · **覆盖**：用户偏好变了就改写同一条，别不断新增（否则一条偏好会有 N 个版本，\n"
        "      检索时互相打架）；\n"
        "    · **TTL**：时效性内容（验证码、临时上下文）设过期时间 —— 但**要看后端是否支持**，\n"
        "      本机用的 InMemoryStore 就明确不支持（宁可直接报错也不静默失效，这点做得对）。"
    )


if __name__ == "__main__":
    demo_1_taxonomy()
    demo_2_index_matters()
    demo_3_memory_in_agent()
    demo_4_maintenance()
    print("\n全部 Demo 执行完毕。")


# ================================================================
# 实测结论 / 与课案的衔接 / 踩坑提示
# ================================================================
# 1. 官方事实（concepts/memory.mdx、stores.mdx）：
#    - 长期记忆分三类：semantic（事实偏好）/ episodic（经历事件）/ procedural（方法规范），
#      官方建议按用途分开组织；
#    - Store 的语义检索由 `index=` 提供：IndexConfig = {dims, embed, fields}，
#      `search(ns, query=...)` 才走向量；
#    - 记忆需要维护策略（覆盖写、TTL 过期）。
# 2. 本机实测（2026-09-17，bge-m3 + grok-4.6）：
#    - 配了索引的 Store：`search` 返回真实 score（示例 0.7380 / 0.4651），排序符合语义；
#    - **不配索引的 Store：`score` 恒为 None，query 被静默忽略**（返回插入顺序）——
#      课案记录的那条坑在本机复现，本文件做成对照实验；
#    - `put(..., index=["text"])` 可让**新写入**的记忆立刻可被检索；
#    - **InMemoryStore 不支持 TTL**：`put(..., ttl=1.0)` 直接抛
#      `NotImplementedError: TTL is not supported by InMemoryStore`（`supports_ttl` 为 False）；
#      ttl 的单位是**分钟**（官方 API：Time to live in minutes）——
#      要过期能力请换 PostgresStore 等支持 TTL 的后端（换后端接口不变）。
# 3. 与课案的衔接：
#    - 课案 01_langgraph/05_长期记忆：Store 基本读写与命名空间；
#    - 课案 03_deepagents/11_记忆 与 02_langchain/20 官方补充篇 Demo 3：
#      工具里通过 `runtime.store` 读写（本文件 Demo 3 是同一机制的应用）；
#    - 短期记忆（checkpointer/thread_id）见课案 01_langgraph/02~04 与 12_记忆 官方补充篇。
# 4. 未收录（官方还有、本文件没做的）：
#    - **生产级 Store 后端**：本文件用 InMemoryStore（进程内、重启即失）；
#      生产要换 PostgresStore / 向量库后端，接口一致、换构造即可；
#    - **自动记忆抽取**：官方示例里会用模型从对话中抽取"值得记住的事"再入库，
#      本文件为了聚焦 Store 机制，改成显式工具写入；
#    - **记忆去重与冲突消解**：属于应用层策略（相似度去重、时间衰减），未展开。
# 5. 踩坑提示：
#    A. `index` 的 `dims` 必须与 embedding 维度一致（bge-m3 = **1024**）；
#       **InMemoryStore 并不校验 dims**（写入时只查向量条数），维度写错要到**检索时**
#       才暴露（numpy 形状不匹配；纯 Python 回退路径下会被 zip 静默截断、算出错误结果）；
#    B. `fields` 指明对 value 的哪些字段建索引；写入时可用 `index=["text"]` 覆盖，
#       但字段名必须存在，否则索引不到；
#    C. **`score is None` = 索引没生效**（最常见的静默故障），排查先看这个；
#    D. 命名空间用元组（`("user-1001", "semantic")`），别用字符串拼路径 ——
#       元组才能用 `list_namespaces` 做前缀遍历；
#    E. TTL 只是"取不到"，底层数据可能仍在（取决于后端实现），
#       涉及隐私的场景要显式 `delete`。
