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
       注意「兜底」与「守卫」是两件事：兜底是**请求失败之后**的善后，
       守卫是**请求之前**判定这个源已经失效、干脆不发（见下表最后一行）。
    5. 为什么「无数据就短路」写了两处（filter 判 `raw_topics`、suggest 判 `filtered`）：
       热点抓取失败是常态，不能让后面两级白烧 LLM 额度；两处共用常量
       ``_NO_TOPIC_HINT``，改文案时不会只改一半。
与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 文件末尾模块级代码 | `graph = hot_topic_graph.get_graph()` + `graph.draw_png("positioning_workflow.png")` | **整段删除** | import 就执行：`draw_png` 要 pygraphviz（未装）直接崩；而且文件名是 `positioning_workflow.png`，图名张冠李戴 |
    | 5 个抓取节点 | 5 个内容几乎一样的函数复制粘贴 | 一个 `_make_fetch_node()` 工厂 + `FETCH_SOURCES` 表 | 逻辑完全相同，只有「平台 ID / 中文名」两个参数不同；节点名保持不变，图结构一致 |
    | `PLATFORM_SENDS` | `{中文名: [Send(...)]}` | `{中文名: [节点名]}`，Send 在路由里现造 | 课案那份 Send 对象只是拿来读 `.node`，绕了一层；语义完全等价 |
    | 节点容错 | 每个 fetch 有 try（好），filter/suggest 没有 | 四个环节全部 try（含 filter/suggest） | 与其余模块统一：失败写进 state 让链路走完 |
    | 无数据处理 | filter 已判空 | 同左（保留课案的判定字符串） | 课案这里是对的：无数据就不该白烧 LLM 额度 |
    | 失效平台 | 无此概念，5 个源一律发请求 | `_make_fetch_node` 里加守卫：`_platform_unavailable(label)` 命中就返回空列表，**一个请求都不发** | 上游 `xiaohongshu` 实测 HTTP 500，而 `fetch_platform()` 内含 2 次重试 + sleep ⇒ 选默认的「全部」白等 12.0 秒、白发 3 次注定失败的请求；图结构不动（节点照旧注册），守卫在节点函数内部 |
    | 合并结果 | 课案:310 称 `operator.add` 合并时「去重 + 按热度降序」 | `node_filter` 里显式按 heat 降序 + 按 title 保序去重 | `operator.add` 只拼接；排序只在 `tools/trend_radar_client.py` 的 `fetch_for_workflow()`，工作流走 `fetch_platform_hot` 不经过它；同名热搜重复进 prompt 白烧 token |
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

本文件在图里干了什么（节点 / 边 / 状态流向）
    一张「扇出 → 汇聚 → 串行两跳」的图。前 1 条边是**条件边**，
    它不返回节点名而返回一串 ``Send``，所以运行时展开成 1~5 支并行分支::

        START ──(route_fetch)──┬──> fetch_douyin ──────┐
                               ├──> fetch_weibo ───────┤
                               ├──> fetch_zhihu ───────┼──> filter ──> suggest ──> END
                               ├──> fetch_xiaohongshu ─┤
                               └──> fetch_bilibili ────┘
                                    （5 路并行，各自写 raw_topics）

    · 五个 fetch 节点都写同一个键 ``raw_topics``，靠 ``Annotated[list, operator.add]``
      这个 reducer 相加合并；**汇聚点（join）是隐式的** —— 五个节点全都
      ``add_edge(..., "filter")``，LangGraph 等它们全部结束才跑**一次** filter。
    · 字段流向：``platform`` / ``account_field`` 是入口输入，一路透传到 filter；
      ``raw_topics``（并行写）→ ``filtered``（filter 写）→ ``suggestions``（suggest 写）。
    · ``filter`` / ``suggest`` 之后是普通串行边，与 ``positioning.py`` 的写法完全一致。
    · 图对象 ``hot_topic_graph`` 在模块级别 compile 一次，``run_hot_topic()`` 复用。

``Send`` 与 reducer 是怎么配合的（本模块的核心机制）
    ``route_fetch`` 返回的 ``Send("fetch_douyin", state)`` 有两个语义：
      ① **扇出**：LangGraph 按返回的 Send 个数并行起分支，分支数量在运行时才定；
      ② **传参**：每个分支拿到的是 ``Send`` 第二个参数那份 state 快照
         （这里直接透传原 state），所以每个 fetch 节点都能读到 ``platform``。
    分支的结果怎么回来？ 靠 state schema 上的 reducer：
      没有 ``Annotated[..., operator.add]`` 时，5 个分支各写一次 ``raw_topics``，
      后写者**覆盖**先写者 —— 最终只剩 1 条，这是本模块最容易踩的坑；
      有了 reducer，LangGraph 把 5 次写入 ``+`` 到一起，才是 5 条。
    ⚠️ ``operator.add`` **只拼接**：既不排序也不去重，这两件事在 ``node_filter`` 里显式做。
    ⚠️ 并行分支的**落地顺序不保证**（谁先回来谁先写），所以下游不能假设
       「抖音一定在第 1 条」；自检里用集合而不是列表来比对。

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

from tools.trend_radar_client import (  # noqa: E402
    NAME_TO_IDS,
    PLATFORM_IDS,
    UNAVAILABLE_PLATFORMS,
    fetch_platform_hot,
)
from workflows import llm_call  # noqa: E402

# 筛选是判断题（要稳），选题建议是创作题（可以放一点）
_TEMP_FILTER = 0.4
_TEMP_SUGGEST = 0.6

# 课案里那个「没数据就短路」的判定串，filter / suggest 两边要一致，提成常量
_NO_TOPIC_HINT = "暂无热点数据，请检查网络连接或稍后重试。"

# 「没数据」时 suggest 给出的文案，同样提成常量：节点与自检共用一份，改文案不会只改一处。
_NO_SUGGEST_HINT = "暂无选题建议，请先确保热点数据获取成功。"

# 「上游产出不可用」的失败前缀 —— 两类都要认，否则 suggest 会拿着报错串白调一次 LLM：
#   · `llm_call` 的失败提示（口径与 `workflows/replicate.py` 的 `_LLM_FAIL_PREFIXES`
#     一致，**刻意各留一份**：一个 2 元组不值得为它 import 另一个工作流模块，
#     真要收口也该落在 `workflows/__init__.py` 那个 `llm_call` 的家）；
#   · 本模块 `node_filter` 自己出错时回填的 `[热点筛选失败: …]`。
# 为什么不带 `[热点工作流失败`：那一串是 `run_hot_topic` 在图**外层**兜住异常后写的，
# 那时图已经停了、suggest 根本没跑，到不了这条判据。
_FILTER_FAIL_PREFIX = "[热点筛选失败"
_UPSTREAM_FAIL_PREFIXES = ("[LLM调用失败", "[LLM未配置", _FILTER_FAIL_PREFIX)


class HotTopicState(TypedDict):
    """热点监控工作流的共享状态（契约已冻结，字段名不要改）。

    五个字段分别由谁写、又被谁读::

        platform / account_field  ← 入口输入（route_fetch 用它选分支，filter 用它做赛道限定）
        raw_topics                ← 5 个 fetch 节点**并行**写，filter 读
        filtered                  ← filter 写，suggest 读
        suggestions               ← suggest 写，`run_hot_topic` 原样返回给页面

    ⚠️ ``raw_topics`` 上那个 ``Annotated[list, operator.add]`` **不是装饰、是语义**：
      它把该字段声明成「带 reducer 的通道」，LangGraph 于是把 5 支并行分支写进来的
      列表**相加**而不是**覆盖**。改写这个注解（哪怕只是去掉 ``Annotated``）会让
      5 个平台的结果只剩最后回来的那一个，而**图照样能编译、能跑、不报错**；
      图结构自检也不会红（它比的是 ``graph.nodes`` / ``graph.edges``），
      只有 3a 那条「抓到 5 条」的断言会挂 —— 这是本文件最需要盯住的一行。
      标了 reducer 的字段不需要初始值：``invoke`` 不传 ``raw_topics`` 时从 ``[]`` 起算。

    另外四个字段是普通通道（无 reducer），语义是「后写的覆盖先写的」——
    它们都只由单个节点写，所以不存在竞争。
    """

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
    # 三元组顺序是 (图节点名, 平台 ID, 中文展示名)，别调换 —— `_make_fetch_node` 按位置取参。
    # ⚠️ B站 的平台 ID 是 `"bilibili-hot-search"` 而不是 `"bilibili"`：
    #    上游 NewsNow 只有 `bilibili-hot-search` 这个键（见 tools/trend_radar_client.py 的
    #    PLATFORM_IDS），写成 `"bilibili"` 不会报错，只是永远拿回 0 条。
    ("fetch_douyin", "douyin", "抖音"),
    ("fetch_weibo", "weibo", "微博"),
    ("fetch_zhihu", "zhihu", "知乎"),
    ("fetch_xiaohongshu", "xiaohongshu", "小红书"),
    ("fetch_bilibili", "bilibili-hot-search", "B站"),
]

# 本表仍是 5 条（图结构不动），但其中可能有**已失效**的平台 —— 小红书就是：
# `NAME_TO_IDS["小红书"]` 的 id 列表已被 tools 层清空（上游 `xiaohongshu` HTTP 500），
# 而 `fetch_platform()` 内部带 2 次重试 + 失败后 sleep，实测白等 12.0 秒才回空列表。
# 所以守卫必须加在**节点函数里**（见 `_make_fetch_node`），而不是靠"少注册一个节点"：
# 后者会改图结构，连带打翻图结构断言与 `_fetch_node_names`。


def _platform_unavailable(label: str) -> str:
    """平台当前是否抓不到；返回中文原因，可用时返回空串。

    判据一律取自 tools 层的**单一真源**，本文件不另抄一份平台清单 ——
    「白等 12 秒」这个坑的成因就是两层各维护一份清单：tools 层把「小红书」摘掉了，
    工作流层却仍在照着自己的表发请求。

    Args:
        label: 中文平台名（``FETCH_SOURCES`` 里的第三个元素）。

    Returns:
        ``UNAVAILABLE_PLATFORMS`` 里登记的原因；没登记但 ``NAME_TO_IDS`` 的 id
        列表为空的，给一句通用说明；两者都不命中（可抓）时返回空串。
    """
    if label in UNAVAILABLE_PLATFORMS:
        return UNAVAILABLE_PLATFORMS[label]
    if label in NAME_TO_IDS and not NAME_TO_IDS[label]:
        return "上游该源当前不可用（tools 层未登记具体原因）"
    return ""


def _make_fetch_node(platform_id: str, label: str):
    """造一个「抓某个平台热榜」的节点函数。

    这是**闭包工厂**：课案里是 5 个几乎逐字重复的函数，只有平台 ID 与中文名不同，
    这里收成一张 ``FETCH_SOURCES`` 表 + 一个工厂，节点名与图结构完全不变。

    每个节点都自带兜底：抓不到就返回空列表，绝不影响其它并行分支。
    失效平台（如小红书）在这里就被拦下：**一个请求都不发**（见下方守卫）。

    Args:
        platform_id: NewsNow / TrendRadar 的平台 ID，如 ``"douyin"``。
        label: 中文展示名，只用于日志和 ``source`` 字段。

    Returns:
        签名 ``(state) -> dict`` 的 LangGraph 节点函数；
        它返回的 dict 形如 ``{"raw_topics": [...]}``，失败时是 ``{"raw_topics": []}``
        （**空列表而不是缺键**：reducer 相加时 ``[]`` 是单位元，安全）。
    """

    def _fetch(state: HotTopicState) -> dict:
        # ⚠️ 守卫必须在**调 `fetch_platform_hot` 之前**：失效平台的 id 列表是空的，
        #    放它进工具层虽然「空 id 直接返回空列表、不发请求」，但那只是下游兜底 ——
        #    本节点照旧会留下一行「抓取完成: 0 条」，把「上游源死了」和「今天没热搜」
        #    混成一回事。守卫命中只打日志，原因取 tools 层的登记（单一真源）。
        reason = _platform_unavailable(label)
        if reason:
            print(f"[热点] {label} 当前不可用，跳过抓取（不发请求）: {reason}")
            return {"raw_topics": []}
        try:
            # 工具层 `fetch_platform_hot` 自己就带 2 次重试，失败时**静默返回 []**（不抛）。
            # 所以这里的 `except` 兜的是更外层的意外（连接被拒、import 期错误等），
            # 保留它是为了「单平台无论怎么挂都不能拖垮整条图」这一条硬约束。
            topics = fetch_platform_hot(platform_id)
            print(f"[热点] {label} 抓取完成: {len(topics)} 条")
            # source 用中文名，前端表格直接展示。
            # `{"source": label, **t}` 的键序是有意的：`t` 在后，**它若也带 source 会覆盖中文名**。
            # 当前工具层返回的键是 title/url/mobile_url/platform/platform_id/rank/heat，不含 source，
            # 所以不会被覆盖 —— 以后给工具层加 `source` 字段时要留意这里。
            return {"raw_topics": [{"source": label, **t} for t in topics]}
        except Exception as exc:  # noqa: BLE001 —— 单平台失败不能拖垮整条图
            print(f"[热点] {label} 抓取失败: {exc}")
            # 返回空列表而不是抛异常，也不把错误写进 state：
            # 这是并行分支，错误信息没有「归属地」能放（5 支共用 raw_topics），
            # 真正的用户提示由下游「一条都没抓到」的统一文案负责。
            return {"raw_topics": []}

    # 让日志/调试里能看出是哪个平台的节点。
    # 工厂造出来的函数都叫 `_fetch`，不设 `__name__` 的话调试时看不出是谁；
    # 顺带把平台 ID 里的 `-` 换成 `_`（Python 标识符不能带连字符）。
    _fetch.__name__ = f"fetch_{platform_id.replace('-', '_')}"
    return _fetch


# 中文平台名 → 要跑的抓取节点；不在表里的一律回退抖音（与课案一致）
# 这里存的是**节点名字符串**而不是课案那种 `Send` 对象：
# `Send` 的构造需要 state，而 state 只有进 `route_fetch` 才拿得到 ——
# 课案那份 `{中文名: [Send(...)]}` 其实只是拿来读 `.node`，绕了一层。
PLATFORM_SENDS = {
    "抖音": ["fetch_douyin"],
    "小红书": ["fetch_xiaohongshu"],
    "微博": ["fetch_weibo"],
    "知乎": ["fetch_zhihu"],
    "B站": ["fetch_bilibili"],
}


def route_fetch(state: HotTopicState):
    """路由：根据 platform 决定并行抓哪几个源。

    这是图上的**条件边**函数：`add_conditional_edges(START, route_fetch, [...])`
    调它一次，拿到的返回值决定「START 之后跑哪些节点」。
    它与普通节点的关键区别 —— 返回值里是 ``Send`` 而不是节点名，
    于是「选一个节点」变成了「**起 N 支并行分支**」，N 在运行时才定。

    读 ``state["platform"]``；不写任何 state 键。

    Args:
        state: 看 ``platform`` 一项就够了；其余字段只是被原样透传给分支。

    Returns:
        ``[Send(节点名, state), ...]`` —— 每个 Send 是一支并行分支：

        · ``platform == "全部"`` → 5 支（``FETCH_SOURCES`` 全表）；
        · ``platform`` 在 ``PLATFORM_SENDS`` 里 → 1 支；
        · 其它（含空串、未识别平台）→ 回退 1 支抖音（与课案行为一致，
          宁可给一份不相关的热榜，也不要给用户一片空白）。
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
    """LLM 筛选与赛道相关的热点（相关度评分 / 切入角度 / 创作难度）。

    图上的位置：**并行扇出的汇聚点**。它是 5 个 fetch 节点共同的出边目标，
    LangGraph 等 5 支分支全部结束才跑它一次 —— 所以它第一次读到的
    ``raw_topics`` 就已经是 reducer 合并后的完整列表，不需要自己等或加锁。

    读 ``state["raw_topics"]``（并行合并结果）+ ``state["account_field"]``（赛道）
    → 排序去重 → ``llm_call(temperature=0.4)`` → 写 ``state["filtered"]``。

    两条短路/兜底：
        · ``raw_topics`` 为空 → 直接返回 ``_NO_TOPIC_HINT``，**一次 LLM 都不调**
          （热点接口限流是常态，没数据时烧额度纯属浪费）；
        · ``llm_call`` 失败 → 返回 ``[LLM调用失败: ...]`` 文本，不抛异常；
          本节点自己的 ``except`` 只兜排序/序列化等意外，回填 ``[热点筛选失败: ...]``。
    """
    try:
        topics = state.get("raw_topics", []) or []
        if not topics:
            # 「一条都没抓到」有两种成因，提示要分开：平台本身就失效（如小红书），
            # 还是网络/上游偶发失败。前者带上 tools 层登记的原因，用户才知道该换平台，
            # 而不是对着「请检查网络连接」反复重试。
            # ⚠️ 提示里**必须保留 `_NO_TOPIC_HINT` 这个子串**：`node_suggest` 正是靠
            #    `_NO_TOPIC_HINT in filtered` 判短路的 —— 换成一句不含它的原因串，
            #    suggest 会拿着一句「平台不可用」去白调一次 LLM（A3 那类病）。
            reason = _platform_unavailable(state.get("platform", "") or "")
            if reason:
                print(f"[热点] 平台「{state.get('platform')}」当前不可用: {reason}")
                return {"filtered": f"{_NO_TOPIC_HINT}\n\n原因：{reason}"}
            print("[热点] 没有抓到任何热点，跳过 LLM 筛选")
            return {"filtered": _NO_TOPIC_HINT}

        # ① 降序：operator.add 合并出来的顺序是「哪支并行分支先回来」，不是热度序 ——
        #    降序排序只存在于 tools/trend_radar_client.py 的 fetch_for_workflow()，
        #    而工作流走的是 fetch_platform_hot（见上面的抓取节点），不经过那条路。
        #    所以在送进 LLM 之前在这里补一次，让模型先看到最热的热点。
        topics = sorted(topics, key=lambda t: t.get("heat") or 0, reverse=True)
        # ② 去重：同一条热搜常常同时挂在多个平台，重复送进 prompt 就是白烧 token。
        #    按 title 保序去重（setdefault 只认第一次出现），上一步已降序，
        #    因此留下的那一条天然就是同名里 heat 最大的那条。
        #    ⚠️ **两步的顺序不能换**：先按热度降序、再 setdefault 去重，才等价于
        #    「同名只留最热的那条」。反过来的话第一次遇到的可能是 heat=3000 的那条，
        #    真正的爆款（heat=999xxx）会被丢掉 —— 而且代码照样跑、不报错。
        unique: dict = {}
        for t in topics:
            unique.setdefault(t.get("title", ""), t)
        # `dict` 保插入顺序（Py3.7+），所以这里转回 list 后仍是降序，prompt 里最热的在前。
        topics = list(unique.values())

        # `ensure_ascii=False` 是必须的：默认 True 会把中文转成 \uXXXX 转义，
        # 模型仍能读，但 prompt 变得又长又难排查；这里要的是人可读的 JSON。
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
        # 温度 0.4：筛选是「判断题」，要的是稳定可复现的取舍，不是发散。
        return {"filtered": llm_call(prompt, temperature=_TEMP_FILTER)}
    except Exception as exc:  # noqa: BLE001
        print(f"[热点] 筛选节点异常: {exc}")
        # 失败也写 `filtered`，且用**另一条**前缀（`[热点筛选失败`）与「无数据」文案区分开：
        # 前者说明链路真的出错了，后者只是上游没抓到热点，两者对用户意味着不同的处置。
        return {"filtered": f"[热点筛选失败: {exc}]"}


def node_suggest(state: HotTopicState) -> dict:
    """LLM 生成具体选题建议（标题模板 / 内容框架 / 预估播放量）。

    图上的位置：链条末端（``filter`` → ``suggest`` → ``END``），它写出的
    ``suggestions`` 就是页面上「具体选题建议」那一段。

    读 ``state["filtered"]`` → ``llm_call(temperature=0.6)`` → 写 ``state["suggestions"]``。

    短路与兜底：
        · ``filtered`` 为空或含 ``_NO_TOPIC_HINT`` → 返回「没数据」的中文提示，**不调 LLM**
          （上游已经明确说了"没数据"，再问一次模型只会得到一段凭空编的选题）；
        · ``filtered`` 以任一 ``_UPSTREAM_FAIL_PREFIXES`` 开头（``node_filter`` 出错回填的
          ``[热点筛选失败: …]``，或筛选那一步 ``llm_call`` 失败返回的
          ``[LLM调用失败: …]`` / ``[LLM未配置] …``）→ 同样不调 LLM，但提示要带上
          **出错原文**：这跟"没数据"是两件事，用户该去查的地方也不同；
        · ``llm_call`` 失败 → 返回 ``[LLM调用失败: ...]`` 文本；
          本节点的 ``except`` 回填 ``[选题建议生成失败: ...]``，两者都不抛异常。
    """
    try:
        filtered = state.get("filtered", "") or ""
        # 两条短路判据分开放（不是 OR 到一行），因为**提示语必须不同**：
        # 「没数据」是常态（热榜限流），用户等会儿再试；「筛选出错」是异常，
        # 用户该去查接口或模型配置。一律回一句"请确保热点数据获取成功"会指错方向。
        # 「没数据」这条用**子串**匹配（`in`）而不是 `startswith`：`_NO_TOPIC_HINT`
        # 可能被 filter 包在自己的输出里；而失败前缀只应出现在**开头**，所以用 `startswith`。
        if not filtered or _NO_TOPIC_HINT in filtered:
            print("[热点] 上游没有可用的筛选结果，跳过选题建议")
            return {"suggestions": _NO_SUGGEST_HINT}
        # ⚠️ 判据里**必须**含 `[热点筛选失败`：`node_filter` 的 except 回填的是它，
        #    而它不含 `_NO_TOPIC_HINT` —— 修之前这里会放行，suggest 便拿着一整段
        #    报错去白调一次 LLM 生成选题（错的输入、编出来的输出、还花了额度）。
        #    `lstrip()` 先吃掉上游可能留下的前导空白，判据用 `startswith` 与
        #    `workflows/replicate.py` 的 `_is_usable()` 保持同一口径。
        if filtered.lstrip().startswith(_UPSTREAM_FAIL_PREFIXES):
            print(f"[热点] 上游筛选失败，跳过选题建议: {filtered[:200]}")
            return {"suggestions": (
                "暂无选题建议：热点筛选阶段出错（不是没抓到数据）——"
                "请检查热榜接口或模型配置后重试。\n"
                f"原始信息：{filtered.strip()}"
            )}

        prompt = f"""基于筛选出的热点：
{filtered}

为每个热点生成2个具体选题，每个选题包含：
1. 标题模板（可直接套用）
2. 内容框架（开头钩子-核心内容-结尾引导）
3. 预估播放量区间
"""
        print("[热点] LLM 生成选题建议中...")
        # 温度 0.6：选题是「创作题」，比筛选（0.4）放松一档，但仍低于定位模块的 0.7 ——
        # 这里要输出可套用的标题模板，太放开会跑成散文。
        return {"suggestions": llm_call(prompt, temperature=_TEMP_SUGGEST)}
    except Exception as exc:  # noqa: BLE001
        print(f"[热点] 选题建议节点异常: {exc}")
        return {"suggestions": f"[选题建议生成失败: {exc}]"}


# ==========================================================================
# 构建并行工作流图
# ==========================================================================
# 整张图的形状（箭头即 add_edge；`START` 那一跳是 add_conditional_edges）：
#     START ══(route_fetch)→ 5 个 fetch ══→ filter → suggest → END
# 5 条出边全指向 filter，LangGraph 据此把 filter 当作汇聚点：等 5 支都结束再跑一次。
builder = StateGraph(HotTopicState)

# 用循环注册 5 个抓取节点：节点名与函数体都来自 `FETCH_SOURCES` 同一张表，
# 表和图不可能对不上（课案是手写 5 遍 add_node，改一处漏一处）。
# 注意 `_make_fetch_node(_pid, _label)` 是**当场造一个新函数**再注册，
# 不是注册工厂本身 —— 闭包把 `platform_id` / `label` 各自固定住了。
for _name, _pid, _label in FETCH_SOURCES:
    builder.add_node(_name, _make_fetch_node(_pid, _label))
builder.add_node("filter", node_filter)
builder.add_node("suggest", node_suggest)

# `add_conditional_edges(起点, 路由函数, 可能的目标列表)`：
# 前两个参数决定「谁来选」，第三个参数只是**候选集声明**（LangGraph 用它画图/做校验），
# 真正跑哪几个由 `route_fetch` 的返回值说了算。
_fetch_node_names = [name for name, _, _ in FETCH_SOURCES]
builder.add_conditional_edges(START, route_fetch, _fetch_node_names)
for _name in _fetch_node_names:
    # 5 个抓取节点全部指向 filter —— LangGraph 会等它们都结束再跑一次 filter
    builder.add_edge(_name, "filter")
builder.add_edge("filter", "suggest")
builder.add_edge("suggest", END)

# 模块级编译一次、全进程复用；`run_hot_topic()` 不会重复建图。
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
        五个键**无论成败都会齐全**（LangGraph 只返回被写过的键，这里逐键补齐）。

    Raises:
        不抛异常。图级异常会被收成 ``[热点工作流失败: ...]`` 写进 ``filtered`` ——
        Streamlit 页面直接取字段渲染，抛出去就是整页 traceback。
        ``views/hot_topic.py`` 认得这串前缀（它把前缀收进失败判据、渲染成红条并贴原文），
        所以图级失败在页面上是看得见的。
    """
    try:
        # 只传两个入口字段：`raw_topics` 由 reducer 从 `[]` 起算，无需初始值。
        # `platform or "全部"` 是兜底 None/空串 —— 空串会让 route_fetch 落到"未识别平台"
        # 分支去回退抖音，而 None 会直接让 `PLATFORM_SENDS.get(None)` 落空，行为更难解释。
        result = hot_topic_graph.invoke({
            "platform": platform or "全部",
            "account_field": account_field or "",
        })
    except Exception as exc:  # noqa: BLE001 —— 图本身出错也不往上抛
        # 与 `positioning.py` 同一套契约：失败以返回值表达，不以异常表达。
        print(f"[热点] 工作流执行失败: {exc}")
        result = {"filtered": f"[热点工作流失败: {exc}]"}

    # 逐键补齐 + 类型兜底：`raw_topics` 用 `or []` 把 None 也收成空列表 ——
    # 页面无条件对它做 `len()` / 列表推导，给 None 过去就是 TypeError。
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
    """真实联网抓热榜 + 调 LLM —— 只在 ``--live`` 时执行。

    为什么跑「抖音」和「全部」两个平台：前者验**单支分支**，后者验**5 路并行 + 合并**，
    这两条路在离线自检里是打桩出来的，联网跑一次才算真验证了上游接口还活着。
    非确定输出（热榜内容每次都变、模型措辞每次都不同）只打印、不断言。
    """
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
    # 自检总原则：默认路径**离线可过**（不联网、不花钱）；真抓热榜 + 真调模型留在 `--live`。
    print("=== 热点监控工作流 自检（离线，不联网）===")

    # ---- 1) 图结构 ----
    # 这里能看出「条件边」在图视图里的样子：`get_graph()` 会把它展开成
    # `__start__ → 每个候选节点` 的多条边（候选集就是 add_conditional_edges 的第三个参数），
    # 而不是运行时真实跑的那几条 —— 真实分支数由 `route_fetch` 决定，3b/3c 才验得到。
    graph = hot_topic_graph.get_graph()
    print(f"图节点: {sorted(graph.nodes)}")
    print(f"图边:   {graph.edges}")

    expected_nodes = sorted(
        ["__end__", "__start__", "filter", "suggest"] + _fetch_node_names
    )
    assert sorted(graph.nodes) == expected_nodes, sorted(graph.nodes)

    # 关键断言 1：`__start__ → 每个 fetch` 与 `每个 fetch → filter` 必须齐全。
    # 后者正是「隐式 join」的声明方式 —— 漏掉任何一个 fetch 的出边，
    # 那支分支的结果就永远到不了 filter（且图仍能编译、不报错）。
    edge_pairs = sorted((e.source, e.target) for e in graph.edges)
    assert edge_pairs == sorted(
        [("__start__", n) for n in _fetch_node_names]
        + [(n, "filter") for n in _fetch_node_names]
        + [("filter", "suggest"), ("suggest", "__end__")]
    ), edge_pairs
    print(f"  ✓ {len(_fetch_node_names)} 路并行抓取 → filter → suggest 接对了")

    # ---- 2) State 契约冻结 ----
    # 与 positioning 同理，`__annotations__` 的键序被锁住；
    # 但这里更要紧的是**那个值**（`Annotated[list, operator.add]`）——
    # 自检没法靠这段断言验 reducer 行为，它是靠 3a 实际的「5 条合并成 5 条」验的。
    assert list(HotTopicState.__annotations__) == [
        "platform", "account_field", "raw_topics", "filtered", "suggestions",
    ], HotTopicState.__annotations__
    print("  ✓ HotTopicState 字段与契约一致")

    # ---- 3) 打桩：抓取 + LLM 全部换成本地假数据，把图真跑一遍 ----
    # 打的是**本模块的全局名字**（`fetch_platform_hot` / `llm_call`）：
    # 节点函数体里写的是裸名字，运行时才去本模块 globals 里查，所以直接赋值即可生效。
    # 不用 mock 库，是因为要验的是「图 + reducer + 节点」的真行为，
    # 桩越薄越不容易自欺；每个用例结束都用 `finally` 还原真身。
    _real_fetch = fetch_platform_hot
    _real_llm = llm_call
    prompts: list[tuple[str, float]] = []
    fetched: list[str] = []

    def _stub_llm(prompt: str, temperature: float = 0.5, fallback: str = "") -> str:
        # 签名与 `workflows.llm_call` 一致（含 `fallback`），否则调用点会 TypeError。
        prompts.append((prompt, temperature))
        return f"[STUB-{len(prompts)}]"

    def make_stub_fetch(fail_ids: set[str] | None = None):
        """造一个假的抓取函数；``fail_ids`` 里的平台 ID 让它抛异常。

        走的是**抛异常**而不是「返回空列表」：工具层真实的失败形态是静默返回 ``[]``，
        但节点里的 ``except`` 也得有人验 —— 用一个必定抛的桩把它逼出来。
        """
        fail_ids = fail_ids or set()

        def _stub_fetch(platform_id: str) -> list:
            # 记下真实被调到的平台 ID，3b/3c 靠它断言「只跑了一路 / 回退到抖音」。
            fetched.append(platform_id)
            if platform_id in fail_ids:
                raise RuntimeError(f"模拟 {platform_id} 抓取失败")
            if platform_id == "empty-platform":
                return []
            # 形状必须与 `tools/trend_radar_client.fetch_platform` 一致（7 个键、无 `source`），
            # 这样才验得到节点里 `{"source": label, **t}` 拼出来的 `source` 是中文名。
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

    # 由 tools 层的真源算出「哪几支真的会发请求」与「哪几支被守卫拦下」——
    # 不在这里手抄平台清单：抄一份就会漂移，而「白等 12 秒」正是这么来的。
    _guarded = [(name, pid, label) for name, pid, label in FETCH_SOURCES
                if _platform_unavailable(label)]
    _fetchable_ids = {pid for _, pid, label in FETCH_SOURCES
                      if not _platform_unavailable(label)}
    for _name, _pid, _label in _guarded:
        # 守卫命中时节点**连工具函数都不该调**：桩被调到就会记进 `fetched`。
        fetched.clear()
        fetch_platform_hot = make_stub_fetch()  # noqa: F841
        _guard_result = _make_fetch_node(_pid, _label)({"platform": "全部"})
        assert _guard_result == {"raw_topics": []}, _guard_result
        assert fetched == [], f"失效平台「{_label}」不该发请求，实际调了 {fetched}"
    print(f"  ✓ {len(_guarded)} 个失效平台（{', '.join(l for _, _, l in _guarded)}）"
          f"一个请求都不发")

    # 3a) 全部平台 —— 验的最重要一条：多支并行写 `raw_topics` 后**没有互相覆盖**
    fetch_platform_hot = make_stub_fetch()  # noqa: F841
    try:
        all_result = run_hot_topic("全部", "科技测评")
    finally:
        fetch_platform_hot = _real_fetch

    # 条数 > 1 而不是 1 ⇒ reducer（operator.add）真的生效了；
    # 这一条要是变成 1，说明 `Annotated[list, operator.add]` 被去掉了 —— 而图不会报错。
    assert len(all_result["raw_topics"]) == len(_fetchable_ids), all_result["raw_topics"]
    # 用集合比 source：并行分支落地顺序不保证，不能假设抖音在第 1 条。
    assert {t["source"] for t in all_result["raw_topics"]} == {
        label for _, _, label in FETCH_SOURCES if not _platform_unavailable(label)
    }, {t["source"] for t in all_result["raw_topics"]}
    # 「全部」必须把**可抓的**平台都调到（且只调这些）—— 失效平台那支一次都没调，
    # 这正是 B1b 的判据：改前这里会多出 `xiaohongshu`（白等 12 秒、白发 3 次请求）。
    assert set(fetched) == _fetchable_ids, fetched
    assert all_result["filtered"] == "[STUB-1]"
    assert all_result["suggestions"] == "[STUB-2]"
    assert set(all_result) == {
        "platform", "account_field", "raw_topics", "filtered", "suggestions",
    }, set(all_result)
    # 赛道的值要进到筛选 prompt 里
    assert "科技测评" in prompts[0][0], prompts[0][0]
    # 温度顺序也被锁住：[filter, suggest] = [0.4, 0.6]（模块级两个常量）。
    # 调换这两个值不会报错，但会让"判断题"变飘、"创作题"变死板。
    assert [t for _, t in prompts] == [0.4, 0.6], prompts
    print(f"  ✓ 「全部」并行抓到 {len(all_result['raw_topics'])} 条并合并成功")

    # 3b) 单平台：只该抓一个源
    # `fetched` 是精确相等而不是包含：多抓一路既是白等网络，也说明路由表串了。
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
    # 走的是 `route_fetch` 里 `PLATFORM_SENDS.get(platform)` 拿不到值那个分支；
    # 断言 `fetched == ["douyin"]` 是在验「回退真的发生了」，不只是「没崩」。
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
    # 让 weibo 分支抛异常：如果哪个 fetch 节点忘了 try，这里会顺着图冒上来，
    # 断言就不是「少一条」而是整个 `run_hot_topic` 抛异常 —— 正是要验的东西。
    # （条数按 `_fetchable_ids` 减 1 算：失效平台那支本来就不出数据，不能算进来。）
    fetched.clear()
    prompts.clear()
    fetch_platform_hot = make_stub_fetch(fail_ids={"weibo"})  # noqa: F841
    try:
        partial = run_hot_topic("全部", "科技测评")
    finally:
        fetch_platform_hot = _real_fetch

    assert len(partial["raw_topics"]) == len(_fetchable_ids) - 1, partial["raw_topics"]
    assert "微博" not in {t["source"] for t in partial["raw_topics"]}
    assert partial["filtered"] == "[STUB-1]"
    print(f"  ✓ 单平台抓取失败不影响其它 {len(partial['raw_topics'])} 路")

    # 3e) 一条都没抓到：不许调 LLM，两级短路都要走通
    # 断言 `prompts == []` 是本用例的核心 —— 它同时验了 filter 层的判空短路
    # 和 suggest 层对 `_NO_TOPIC_HINT` 的识别（只要有一层漏判，这里就会记到 prompt）。
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
    assert empty_result["suggestions"] == _NO_SUGGEST_HINT, empty_result["suggestions"]
    assert prompts == [], f"无数据时不该调 LLM，实际调了 {len(prompts)} 次"
    print("  ✓ 无热点数据时短路：一次 LLM 都没调")

    # 3f) 筛选阶段**出错**时也必须短路（本轮修的 bug：旧判据只认 `_NO_TOPIC_HINT`）
    #     刻意走真图触发，而不是直接给 `node_suggest` 喂一个假串 ——
    #     这样连「node_filter 出错时到底往 state 里写了什么」一起验了：
    #     抓取桩返回一条**带不可 JSON 序列化字段**的热点，`node_filter` 里的
    #     `json.dumps` 抛 TypeError，被它自己的 except 收成 `[热点筛选失败: …]`。
    #     改前：这串不含 `_NO_TOPIC_HINT` ⇒ 不短路 ⇒ suggest 拿着报错白调一次 LLM。
    fetched.clear()
    prompts.clear()
    # `llm_call` 必须换成**计数桩**：真身不记调用，用它的话「一次都没调」这条断言
    # 即使 bug 还在也会通过（那才是真正的假绿）。
    llm_call = _stub_llm  # noqa: F841

    def _bad_payload_fetch(platform_id: str) -> list:
        """返回一条 json 序列化不了的记录：``{"坏字段"}`` 是 set，`json.dumps` 必抛。"""
        return [{"title": f"{platform_id} 的热点", "heat": 1, "url": {"坏字段"}}]

    fetch_platform_hot = _bad_payload_fetch  # noqa: F841
    try:
        broken = run_hot_topic("抖音", "科技测评")
    finally:
        fetch_platform_hot = _real_fetch
        llm_call = _real_llm

    assert broken["filtered"].startswith(_FILTER_FAIL_PREFIX), broken["filtered"]
    # 核心断言：筛选出错时一次 LLM 都不许调 —— 旧代码在这是 1。
    assert prompts == [], f"筛选出错时不该调 LLM，实际调了 {len(prompts)} 次"
    assert broken["suggestions"].startswith("暂无选题建议"), broken["suggestions"]
    # 提示必须与「一条都没抓到」区分开：否则用户会去查网络，而真正的问题在筛选阶段。
    assert broken["suggestions"] != _NO_SUGGEST_HINT, broken["suggestions"]
    print("  ✓ 筛选出错时短路：0 次 LLM，提示与「无数据」区分开")

    # 3g) 选中一个**已失效的平台**（小红书）：不发请求、不调 LLM，且提示要说明原因。
    #     这条是 B1b 的端到端判据：改前「全部」会为它白等约 12 秒（2 次重试 + sleep），
    #     单点选它更是每次都白等；改后 `_make_fetch_node` 的守卫生效，一支请求都不发。
    #     用工具层的真源挑出「哪个平台当前失效」，不在自检里手抄平台名。
    assert _guarded, "没有失效平台可测（tools 层的 UNAVAILABLE_PLATFORMS 变了？）"
    for _name, _pid, _label in _guarded:
        fetched.clear()
        prompts.clear()
        llm_call = _stub_llm  # noqa: F841
        fetch_platform_hot = make_stub_fetch()  # noqa: F841
        try:
            dead = run_hot_topic(_label, "科技测评")
        finally:
            fetch_platform_hot = _real_fetch
            llm_call = _real_llm

        assert fetched == [], f"「{_label}」已失效，不该发请求，实际调了 {fetched}"
        assert dead["raw_topics"] == [], dead["raw_topics"]
        # 提示必须**同时**含 `_NO_TOPIC_HINT`（suggest 靠它短路）与原因（用户靠它换平台）。
        assert _NO_TOPIC_HINT in dead["filtered"], dead["filtered"]
        assert "原因：" in dead["filtered"], dead["filtered"]
        assert prompts == [], f"平台不可用时不该调 LLM，实际调了 {len(prompts)} 次"
        assert dead["suggestions"] == _NO_SUGGEST_HINT, dead["suggestions"]
        print(f"  ✓ 失效平台「{_label}」：0 次请求、0 次 LLM，提示带原因")

    print("\n全部自检通过")

    if "--live" in sys.argv:
        _live_check()
    else:
        print("（跳过联网验证；加 --live 可真实抓一次热榜）")
