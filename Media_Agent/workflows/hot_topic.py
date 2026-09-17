# -*- coding: utf-8 -*-
"""热点监控工作流 —— Send 并行抓取 → LLM 筛选 → 选题建议

课案出处：《3.自媒体Agent》→ 热点监控 → workflows/hot_topic.py

本节要讲什么
    1. **LangGraph 的并行写法**：`add_conditional_edges(START, route_fetch, [...])`
       的返回不是节点名，而是一串 ``Send("节点名", 该节点要看到的 state)``。
       一个 Send 就是一支并行分支，它们同时跑、各自往同一个 state 字段写。
    2. **并发合并靠 reducer**，不靠锁：
       ``raw_topics: Annotated[list, operator.add]`` 声明了「这个通道用列表相加合并」，
       所以 5 个抓取节点各写各的，LangGraph 会把它们 ``+`` 到一起。
       没有这个 Annotated，后写的会把先写的**覆盖掉** —— 这是本模块最容易踩的坑。
    3. **join 是隐式的**：5 个抓取节点都 `add_edge(..., "filter")`，
       LangGraph 自动等它们全部结束才跑一次 filter（不会跑 5 次）。
       实测确认：filter 看到的 `raw_topics` 已经是合并后的完整列表。
    4. 抓取是 I/O 密集且经常失败的（公共热榜 API 会限流），
       所以每个抓取节点都单独兜底：**某个平台挂了，其它平台照常出结果**。

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 文件末尾模块级代码 | `graph = hot_topic_graph.get_graph()` + `graph.draw_png("positioning_workflow.png")` | **整段删除** | import 就执行：`draw_png` 要 pygraphviz（未装）直接崩；而且文件名是 `positioning_workflow.png`，图名张冠李戴 |
    | 5 个抓取节点 | 5 个内容几乎一样的函数复制粘贴 | 一个 `_make_fetch_node()` 工厂 + `FETCH_SOURCES` 表 | 逻辑完全相同，只有「平台 ID / 中文名」两个参数不同；节点名保持不变，图结构一致 |
    | `PLATFORM_SENDS` | `{中文名: [Send(...)]}` | `{中文名: [节点名]}`，Send 在路由里现造 | 课案那份 Send 对象只是拿来读 `.node`，绕了一层；语义完全等价 |
    | 节点容错 | 每个 fetch 有 try（好），filter/suggest 没有 | 四个环节全部 try（含 filter/suggest） | 与其余模块统一：失败写进 state 让链路走完 |
    | 无数据处理 | filter 已判空 | 同左（保留课案的判定字符串） | 课案这里是对的：无数据就不该白烧 LLM 额度 |
    | 自检 | 无 | 末尾 `__main__` 离线自检（打桩抓取 + LLM） | 验证并行合并、单平台路由、平台挂掉不影响整体 |
    | 平台清单 | 图里 5 个平台 | 同左 | 与课案一致 |

踩过的坑
    · `Annotated[list, operator.add]` 的初始值是空列表：`invoke` 时不传 `raw_topics`
      也能跑（实测 5 个 Send 分支的结果会 merge 成 5 条）。
    · 并行分支的落地顺序**不保证**（谁先回来谁先写），
      所以下游不能假设「抖音一定在第 1 条」；自检里用集合而不是列表比顺序。
    · 课案的 `draw_png` 只要 import 就会执行 —— 这也是「模块级不要放副作用」的经典反例。
    · 直接 `python workflows/hot_topic.py` 时 `sys.path[0]` 是 `workflows/`，
      需要顶部那段 path 引导才能 `from workflows import llm_call`。

运行方式
    离线自检（不联网）::

        Set-Location F:\\ProGram\\Python_Base\\Media_Agent
        & ..\\.venv\\Scripts\\python.exe workflows\\hot_topic.py

    真实联网抓热榜 + 调 LLM::

        & ..\\.venv\\Scripts\\python.exe workflows\\hot_topic.py --live
"""

import json
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

# Windows 控制台默认 GBK，本模块会打印中文日志
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 直接以脚本方式运行时，sys.path[0] 是 workflows/ 目录，项目根不在里面。
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Send  # noqa: E402

from tools.trend_radar_client import PLATFORM_IDS, fetch_platform_hot  # noqa: E402
from workflows import llm_call  # noqa: E402

# 筛选是判断题（要稳），选题建议是创作题（可以放一点）
_TEMP_FILTER = 0.4
_TEMP_SUGGEST = 0.6

# 课案里那个「没数据就短路」的判定串，filter / suggest 两边要一致，提成常量
_NO_TOPIC_HINT = "暂无热点数据，请检查网络连接或稍后重试。"


class HotTopicState(TypedDict):
    """热点监控工作流的共享状态（契约已冻结，字段名不要改）。"""

    platform: str                                   # 输入：中文平台名，或 "全部"
    account_field: str                              # 输入：赛道描述
    raw_topics: Annotated[list, operator.add]       # 并行结果合并（reducer = 列表相加）
    filtered: str                                   # 节点产出：赛道相关热点筛选
    suggestions: str                                # 节点产出：具体选题建议


# ==========================================================================
# 抓取源表：(图节点名, NewsNow 平台 ID, 展示用中文名)
# 课案是 5 个几乎一模一样的函数，这里收成一张表 + 一个工厂。
# ==========================================================================
FETCH_SOURCES = [
    ("fetch_douyin", "douyin", "抖音"),
    ("fetch_weibo", "weibo", "微博"),
    ("fetch_zhihu", "zhihu", "知乎"),
    ("fetch_xiaohongshu", "xiaohongshu", "小红书"),
    ("fetch_bilibili", "bilibili-hot-search", "B站"),
]


def _make_fetch_node(platform_id: str, label: str):
    """造一个「抓某个平台热榜」的节点函数。

    每个节点都自带兜底：抓不到就返回空列表，绝不影响其它并行分支。

    Args:
        platform_id: NewsNow / TrendRadar 的平台 ID，如 ``"douyin"``。
        label: 中文展示名，只用于日志和 ``source`` 字段。

    Returns:
        签名 ``(state) -> dict`` 的 LangGraph 节点函数。
    """

    def _fetch(state: HotTopicState) -> dict:
        try:
            topics = fetch_platform_hot(platform_id)
            print(f"[热点] {label} 抓取完成: {len(topics)} 条")
            # source 用中文名，前端表格直接展示
            return {"raw_topics": [{"source": label, **t} for t in topics]}
        except Exception as exc:  # noqa: BLE001 —— 单平台失败不能拖垮整条图
            print(f"[热点] {label} 抓取失败: {exc}")
            return {"raw_topics": []}

    # 让日志/调试里能看出是哪个平台的节点
    _fetch.__name__ = f"fetch_{platform_id.replace('-', '_')}"
    return _fetch


# 中文平台名 → 要跑的抓取节点；不在表里的一律回退抖音（与课案一致）
PLATFORM_SENDS = {
    "抖音": ["fetch_douyin"],
    "小红书": ["fetch_xiaohongshu"],
    "微博": ["fetch_weibo"],
    "知乎": ["fetch_zhihu"],
    "B站": ["fetch_bilibili"],
}


def route_fetch(state: HotTopicState):
    """路由：根据 platform 决定并行抓哪几个源。

    Returns:
        ``[Send(节点名, state), ...]`` —— 每个 Send 是一支并行分支。
    """
    platform = state.get("platform", "全部")
    if platform == "全部":
        return [Send(name, state) for name, _, _ in FETCH_SOURCES]

    nodes = PLATFORM_SENDS.get(platform)
    if not nodes:
        print(f"[热点] 未识别的平台「{platform}」，回退到抖音")
        nodes = ["fetch_douyin"]
    return [Send(name, state) for name in nodes]


# ==========================================================================
# LLM 节点：筛选 + 选题建议
# ==========================================================================
def node_filter(state: HotTopicState) -> dict:
    """LLM 筛选与赛道相关的热点（相关度评分 / 切入角度 / 创作难度）。"""
    try:
        topics = state.get("raw_topics", []) or []
        if not topics:
            print("[热点] 没有抓到任何热点，跳过 LLM 筛选")
            return {"filtered": _NO_TOPIC_HINT}

        topics_json = json.dumps(topics, ensure_ascii=False)
        account_field = state.get("account_field", "") or ""
        prompt = f"""你是「{account_field}」赛道的内容策划。

当前热点（{len(topics)}条）：
{topics_json}

筛选出与你赛道相关度高的热点，按相关度排序。对每个热点给出：
- 相关度评分(1-10)
- 推荐切入角度（50字以内）
- 创作难度（简单/中等/困难）
"""
        print(f"[热点] LLM 筛选中（{len(topics)} 条）...")
        return {"filtered": llm_call(prompt, temperature=_TEMP_FILTER)}
    except Exception as exc:  # noqa: BLE001
        print(f"[热点] 筛选节点异常: {exc}")
        return {"filtered": f"[热点筛选失败: {exc}]"}


def node_suggest(state: HotTopicState) -> dict:
    """LLM 生成具体选题建议（标题模板 / 内容框架 / 预估播放量）。"""
    try:
        filtered = state.get("filtered", "") or ""
        # 课案的短路判断保留：上游没拿到数据就别再白烧一次额度
        if not filtered or _NO_TOPIC_HINT in filtered:
            print("[热点] 上游没有可用的筛选结果，跳过选题建议")
            return {"suggestions": "暂无选题建议，请先确保热点数据获取成功。"}

        prompt = f"""基于筛选出的热点：
{filtered}

为每个热点生成2个具体选题，每个选题包含：
1. 标题模板（可直接套用）
2. 内容框架（开头钩子-核心内容-结尾引导）
3. 预估播放量区间
"""
        print("[热点] LLM 生成选题建议中...")
        return {"suggestions": llm_call(prompt, temperature=_TEMP_SUGGEST)}
    except Exception as exc:  # noqa: BLE001
        print(f"[热点] 选题建议节点异常: {exc}")
        return {"suggestions": f"[选题建议生成失败: {exc}]"}


# ==========================================================================
# 构建并行工作流图
# ==========================================================================
builder = StateGraph(HotTopicState)

for _name, _pid, _label in FETCH_SOURCES:
    builder.add_node(_name, _make_fetch_node(_pid, _label))
builder.add_node("filter", node_filter)
builder.add_node("suggest", node_suggest)

_fetch_node_names = [name for name, _, _ in FETCH_SOURCES]
builder.add_conditional_edges(START, route_fetch, _fetch_node_names)
for _name in _fetch_node_names:
    # 5 个抓取节点全部指向 filter —— LangGraph 会等它们都结束再跑一次 filter
    builder.add_edge(_name, "filter")
builder.add_edge("filter", "suggest")
builder.add_edge("suggest", END)

hot_topic_graph = builder.compile()

# 课案在这里还有两行模块级代码：
#     graph = hot_topic_graph.get_graph()
#     graph.draw_png("positioning_workflow.png")
# 已按任务要求整段删除 —— import 即执行，且需要未安装的 pygraphviz。


# ==========================================================================
# 对外接口
# ==========================================================================
def run_hot_topic(platform: str, account_field: str) -> dict:
    """运行热点监控工作流。

    Args:
        platform: 平台名（``"抖音"/"小红书"/"微博"/"知乎"/"B站"/"全部"``）。
        account_field: 赛道描述（如 ``"科技测评 / AI工具分享"``）。

    Returns:
        ``{"platform","account_field","raw_topics","filtered","suggestions"}``。
        抓取失败时 ``raw_topics`` 为 ``[]``，``filtered`` 是中文提示文本。
    """
    try:
        result = hot_topic_graph.invoke({
            "platform": platform or "全部",
            "account_field": account_field or "",
        })
    except Exception as exc:  # noqa: BLE001 —— 图本身出错也不往上抛
        print(f"[热点] 工作流执行失败: {exc}")
        result = {"filtered": f"[热点工作流失败: {exc}]"}

    return {
        "platform": result.get("platform", platform or "全部"),
        "account_field": result.get("account_field", account_field or ""),
        "raw_topics": result.get("raw_topics", []) or [],
        "filtered": result.get("filtered", ""),
        "suggestions": result.get("suggestions", ""),
    }


# ==========================================================================
# 离线自检
# ==========================================================================
def _live_check() -> None:
    """真实联网抓热榜 + 调 LLM —— 只在 ``--live`` 时执行。"""
    print("\n=== 真实联网跑一次（--live）===")
    for platform in ("抖音", "全部"):
        result = run_hot_topic(platform, "AI工具 / 科技测评")
        topics = result["raw_topics"]
        print(f"\n--- 平台：{platform} ---")
        print(f"抓到热点: {len(topics)} 条")
        for t in topics[:3]:
            print(f"  #{t.get('rank')} [{t.get('source')}] {t.get('title')}")
        print(f"filtered 前 200 字: {result['filtered'][:200]}")
        print(f"suggestions 前 200 字: {result['suggestions'][:200]}")


if __name__ == "__main__":
    print("=== 热点监控工作流 自检（离线，不联网）===")

    # ---- 1) 图结构 ----
    graph = hot_topic_graph.get_graph()
    print(f"图节点: {sorted(graph.nodes)}")
    print(f"图边:   {graph.edges}")

    expected_nodes = sorted(
        ["__end__", "__start__", "filter", "suggest"] + _fetch_node_names
    )
    assert sorted(graph.nodes) == expected_nodes, sorted(graph.nodes)

    edge_pairs = sorted((e.source, e.target) for e in graph.edges)
    assert edge_pairs == sorted(
        [("__start__", n) for n in _fetch_node_names]
        + [(n, "filter") for n in _fetch_node_names]
        + [("filter", "suggest"), ("suggest", "__end__")]
    ), edge_pairs
    print(f"  ✓ {len(_fetch_node_names)} 路并行抓取 → filter → suggest 接对了")

    # ---- 2) State 契约冻结 ----
    assert list(HotTopicState.__annotations__) == [
        "platform", "account_field", "raw_topics", "filtered", "suggestions",
    ], HotTopicState.__annotations__
    print("  ✓ HotTopicState 字段与契约一致")

    # ---- 3) 打桩：抓取 + LLM 全部换成本地假数据，把图真跑一遍 ----
    _real_fetch = fetch_platform_hot
    _real_llm = llm_call
    prompts: list[tuple[str, float]] = []
    fetched: list[str] = []

    def _stub_llm(prompt: str, temperature: float = 0.5, fallback: str = "") -> str:
        prompts.append((prompt, temperature))
        return f"[STUB-{len(prompts)}]"

    def make_stub_fetch(fail_ids: set[str] | None = None):
        fail_ids = fail_ids or set()

        def _stub_fetch(platform_id: str) -> list:
            fetched.append(platform_id)
            if platform_id in fail_ids:
                raise RuntimeError(f"模拟 {platform_id} 抓取失败")
            if platform_id == "empty-platform":
                return []
            return [{
                "title": f"{platform_id} 的热点",
                "url": f"https://example.com/{platform_id}",
                "mobile_url": f"https://m.example.com/{platform_id}",
                "platform": PLATFORM_IDS.get(platform_id, platform_id),
                "platform_id": platform_id,
                "rank": 1,
                "heat": 1000,
            }]

        return _stub_fetch

    llm_call = _stub_llm  # noqa: F841

    # 3a) 全部平台
    fetch_platform_hot = make_stub_fetch()  # noqa: F841
    try:
        all_result = run_hot_topic("全部", "科技测评")
    finally:
        fetch_platform_hot = _real_fetch

    assert len(all_result["raw_topics"]) == 5, len(all_result["raw_topics"])
    assert {t["source"] for t in all_result["raw_topics"]} == {
        "抖音", "微博", "知乎", "小红书", "B站",
    }, {t["source"] for t in all_result["raw_topics"]}
    assert set(fetched) == {pid for _, pid, _ in FETCH_SOURCES}, fetched
    assert all_result["filtered"] == "[STUB-1]"
    assert all_result["suggestions"] == "[STUB-2]"
    assert set(all_result) == {
        "platform", "account_field", "raw_topics", "filtered", "suggestions",
    }, set(all_result)
    # 赛道的值要进到筛选 prompt 里
    assert "科技测评" in prompts[0][0], prompts[0][0]
    assert [t for _, t in prompts] == [0.4, 0.6], prompts
    print(f"  ✓ 「全部」并行抓到 {len(all_result['raw_topics'])} 条并合并成功")

    # 3b) 单平台：只该抓一个源
    fetched.clear()
    prompts.clear()
    fetch_platform_hot = make_stub_fetch()  # noqa: F841
    try:
        one_result = run_hot_topic("抖音", "科技测评")
    finally:
        fetch_platform_hot = _real_fetch

    assert fetched == ["douyin"], fetched
    assert len(one_result["raw_topics"]) == 1, one_result["raw_topics"]
    assert one_result["raw_topics"][0]["source"] == "抖音"
    print("  ✓ 单平台只跑一路分支")

    # 3c) 未知平台回退抖音（与课案行为一致）
    fetched.clear()
    fetch_platform_hot = make_stub_fetch()  # noqa: F841
    try:
        fallback_result = run_hot_topic("不存在的平台", "科技测评")
    finally:
        fetch_platform_hot = _real_fetch

    assert fetched == ["douyin"], fetched
    assert len(fallback_result["raw_topics"]) == 1
    print("  ✓ 未识别的平台回退到抖音")

    # 3d) 某个平台挂了，其它平台照常出结果（并行分支互不影响）
    fetched.clear()
    prompts.clear()
    fetch_platform_hot = make_stub_fetch(fail_ids={"weibo"})  # noqa: F841
    try:
        partial = run_hot_topic("全部", "科技测评")
    finally:
        fetch_platform_hot = _real_fetch

    assert len(partial["raw_topics"]) == 4, len(partial["raw_topics"])
    assert "微博" not in {t["source"] for t in partial["raw_topics"]}
    assert partial["filtered"] == "[STUB-1]"
    print("  ✓ 单平台抓取失败不影响其它 4 路")

    # 3e) 一条都没抓到：不许调 LLM，两级短路都要走通
    fetched.clear()
    prompts.clear()
    fetch_platform_hot = make_stub_fetch()  # noqa: F841
    llm_call = _stub_llm  # noqa: F841
    try:
        # 用一个假平台 ID 也拿不到数据：直接打桩成「永远返回空」
        def _empty_fetch(platform_id: str) -> list:
            return []

        fetch_platform_hot = _empty_fetch  # noqa: F841
        empty_result = run_hot_topic("全部", "科技测评")
    finally:
        fetch_platform_hot = _real_fetch
        llm_call = _real_llm

    assert empty_result["raw_topics"] == []
    assert empty_result["filtered"] == _NO_TOPIC_HINT, empty_result["filtered"]
    assert empty_result["suggestions"] == "暂无选题建议，请先确保热点数据获取成功。"
    assert prompts == [], f"无数据时不该调 LLM，实际调了 {len(prompts)} 次"
    print("  ✓ 无热点数据时短路：一次 LLM 都没调")

    print("\n全部自检通过")

    if "--live" in sys.argv:
        _live_check()
    else:
        print("（跳过联网验证；加 --live 可真实抓一次热榜）")
