# -*- coding: utf-8 -*-
"""数据复盘工作流 —— 抖音作品数据采集 + LLM 多维诊断

课案出处：自媒体课案 → 数据复盘 → ``workflows/review.py``

课案的四个节点（本项目**结构原样保留**，prompt 也照抄）
    ① fetch    从抖音个人主页采集真实作品数据
    ② funnel   播放/互动量分析 + 转化率 + 流量漏斗 + 发布时间（temperature 0.3）
    ③ content  内容方向 / 标题技巧 / 节奏建议 / 竞争分析（temperature 0.4）
    ④ suggest  立即执行 / 短期策略 / 长期规划 / A/B 测试方案（temperature 0.4）
    串行连成一条链，全部走 ``workflows.llm_call``（绝不抛异常的统一 LLM 入口）。

与课案的落地差异
    | 项 | 课案 | 本项目 |
    |---|---|---|
    | 采集节点 | ``DOUYIN_PROJECT`` 硬编码绝对路径 + ``sys.path.insert`` + ``from backend.lib.douyin import Douyin`` | 全部删掉，改调 ``tools.douyin_client.fetch_user_videos()``（HTTP，走自托管服务） |
    | Cookie | 读本地 douyin 项目的 ``config/settings.json`` | ``settings.media.douyin_cookie``，由 ``tools/douyin_client`` 内部解析 |
    | ``/user/self`` | 节点内用 ``niquests`` 抓 HTML 抠 sec_uid | 下移到 ``tools/douyin_client``，本文件不再碰网络细节 |
    | 采集失败 | 返回 error_msg，但只此一条路 | 保留 error_msg，**另加** ``run_review_from_json()`` 降级入口（跳过采集节点） |
    | 返回值 | ``review_graph.invoke(...)`` 原样返回（只含被写过的键） | ``run_review()`` 补齐 6 个字段，契约稳定 |

为什么要有降级入口
    自托管采集服务（Docker）不一定在跑，而 agent 也起不动 Docker。
    没有它，「四节点链路」在离线环境里就完全没法验证。
    ``run_review_from_json()`` 走的是同一套 funnel → content → suggest 节点，
    只是换了个入口把 ``video_data`` 灌进去。

本文件在图里干了什么（节点 / 边 / 状态流向）
    这里是**两张同源的图**，只差第一个节点：主入口带采集，降级入口直接从诊断开始::

        主入口   START ──> fetch ──> funnel ──> content ──> suggest ──> END
        降级入口 START ─────────────> funnel ──> content ──> suggest ──> END
                  （video_data 由调用方灌进来，跳过 fetch）

    · 字段流向：``profile_url`` 是入口输入；``video_data`` 由 ``fetch``（或调用方）写，
      后面三个节点**都读它**；``funnel_diagnosis`` → ``content_assessment`` 是
      ``suggest`` 的两个额外输入（它一次读三样）；``error_msg`` 由 ``fetch`` 写，出口读。
    · 四个节点全是普通串行边，没有条件边、没有并行、没有 reducer ——
      与 ``positioning.py`` 同一种形态，复杂度全在 prompt 上。
    · 两张图都在模块级别 ``compile()`` 一次（``review_graph`` / ``json_review_graph``），
      ``run_review*()`` 复用它俩。
    · ``run_review()`` / ``run_review_from_json()`` 出口统一走 ``_filled()``：
      **补齐 6 个键**是硬要求 —— LangGraph 的 ``invoke`` 只返回被写过的键，
      而 `views/review.py` 无条件取 ``result["video_data"]`` 等字段，缺键就是 KeyError。

自检怎么验的（离线，服务没起来也必须全绿）
    全部断言都落在「图结构 + 失败路径」上，**默认一次网络和 LLM 都不碰**：
    · 图结构：节点的集合关系与 5 条边的集合关系（降级图额外断言「没有 fetch」）；
    · 失败路径：给 ``run_review_from_json`` 喂 ``""`` / ``"   "`` / ``"不是 JSON"``
      / ``"[]"`` / ``"{}"`` 五种垃圾输入，每次都要求「6 字段齐全 + ``error_msg`` 非空
      + 诊断字段为空串」（不能被当成正常结果）；
    · 真调模型的那条链路放在 ``--llm`` 分支里，由人显式触发；
      该分支用 ``_LLM_FAIL_PREFIXES`` **逐个**校验三个产出，命中就以非 0 退出码收场 ——
      只判非空会把「没配 key / 额度耗尽」当成通过（失败提示文本同样是真值）。

踩过的坑
    · 直接 ``python workflows/review.py`` 时 ``sys.path[0]`` 是 ``workflows/`` 目录，
      ``from workflows import llm_call`` 会 ModuleNotFoundError ——
      所以**路径引导必须在文件顶部、所有项目内 import 之前**（放在 ``__main__`` 里没用，
      模块级 import 早跑完了）。
    · LangGraph 的 ``invoke`` 只返回状态里**被写过的键**；前端直接取
      ``result["suggestions"]`` 会 KeyError，所以对外函数统一补齐字段。

运行方式
    离线自检（不联网、不调模型）::

        Set-Location F:\\ProGram\\Python_Base\\Media_Agent
        & ..\\.venv\\Scripts\\python.exe workflows\\review.py

    真调模型跑完整链路（会消耗额度）::

        & ..\\.venv\\Scripts\\python.exe workflows\\review.py --llm
"""

import sys
from pathlib import Path

# ---- 路径引导：必须在 import 项目内模块（workflows/*、tools/*）之前执行 ----
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json  # noqa: E402
from typing import TypedDict  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402

from workflows import llm_call  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 没数据时各节点的占位文案（课案原文如此，前端据此判断"没跑出东西"）
NO_DATA = "暂无数据"

# `llm_call` 的失败串前缀 —— 见 `workflows/__init__.py`：它**绝不抛异常**，
# 失败时返回 `[LLM调用失败: …]` / `[LLM未配置] …` 这两种**提示文本**。
# 这份元组与 `workflows/replicate.py` 的 `_LLM_FAIL_PREFIXES` 口径一致、**刻意各留一份**：
# 它只是个 2 元组，重复的成本远低于「为一个常量去 import 另一个工作流模块」——
# 那会把「读懂本文件要哪些判据」变成「顺着 import 去别处找定义」。
# 真要收口，也该落在 `workflows/__init__.py`（`llm_call` 的家），而不是某个具体工作流里。
_LLM_FAIL_PREFIXES = ("[LLM调用失败", "[LLM未配置")


def _llm_fail_reason(text: str) -> str:
    """判断一段 LLM 产出能不能用；不能用时给出中文原因，能用返回空串。

    两种坏形态都要抓，**漏一种就是假绿**：

    * 空 / 纯空白 —— 模型没吐东西（原来那个 ``and`` 断言覆盖的就是这一种）；
    * 以 ``_LLM_FAIL_PREFIXES`` 开头 —— ``llm_call`` 失败时返回的提示文本。
      它是**真值**，所以「只判非空」的断言会放它过去；本文件的自检 ``--llm``
      分支就是这么假绿过的（没配 key 也报「通过」）。

    Args:
        text: 待判定的产出。允许空串，也容忍 ``None``（调用点多取自 state）。

    Returns:
        空串表示可用；否则是一句能直接打进日志的中文说明。
    """
    value = (text or "").strip()
    if not value:
        return "为空"
    if value.startswith(_LLM_FAIL_PREFIXES):
        return "是 LLM 失败提示文本，不是模型正文"
    return ""


class ReviewState(TypedDict):
    """复盘工作流的状态（字段与课案一致）。

    六个字段里只有第一个是入口输入，其余都由节点写::

        profile_url                                  ← 入口输入（`run_review` 传进来）
        video_data / error_msg                       ← fetch 节点写（降级入口由调用方直接给）
        funnel_diagnosis → content_assessment → suggestions  ← 三个 LLM 节点各写一个

    ⚠️ 三条与 `positioning.py` 不同的约定（读代码时最容易踩这里）::
        ① ``error_msg`` 非空表示「采集没成」，页面据此**提前 return**，
           后面三个诊断节点即便跑了也不展示 —— 所以 fetch 失败时没必要再让它短路一遍；
        ② ``video_data`` 是 **JSON 字符串**（不是 list）：`[]` 这个字面量表示"没数据"，
           `_data_of()` 专门把 `""` / `"[]"` / `"{}"` 三种形态都归一成空；
        ③ 前两个 LLM 节点**没有 try/except**（其余模块的节点都有）。
           这是安全的，前提是 ``llm_call`` 的「不抛异常」契约成立 ——
           它失败时返回 ``[LLM调用失败: ...]`` 文本，节点会把它当真值写进字段。
           一旦哪天 ``llm_call`` 改成抛异常，这里会直接让整张图挂掉。
    """

    profile_url: str        # 抖音主页 URL
    video_data: str         # JSON: 作品列表 + 统计数据
    funnel_diagnosis: str
    content_assessment: str
    suggestions: str
    error_msg: str


# ============================================================
# Node 1: 采集真实数据
# ============================================================

def node_fetch_data(state: ReviewState) -> dict:
    """从抖音个人主页采集作品数据（课案走本地爬虫，本项目走自托管 REST 服务）。

    图上的位置：主入口图的第一个节点；降级入口图（``json_review_graph``）
    **没有它**，那时 ``video_data`` 由调用方 ``run_review_from_json()`` 直接灌进来。

    输入: 抖音用户主页 URL（如 https://www.douyin.com/user/xxx）
    输出: JSON 格式的作品列表，含点赞/评论/分享/收藏（本人主页另含播放）

    读 ``state["profile_url"]``；写 ``state["video_data"]`` 与 ``state["error_msg"]``。
    **一次网络都不自己发**：真正的 HTTP、Cookie、sec_uid 解析都在
    ``tools/douyin_client.fetch_user_videos()`` 里，这个节点只负责"翻译"它的返回值。

    Returns:
        带 ``video_data`` + ``error_msg`` 的 dict：

        · 链接为空 → ``error_msg="请输入抖音用户主页链接"``、``video_data="[]"``；
        · 采集失败 → ``error_msg`` 是工具层给的中文原文（可能多行）、``video_data="[]"``；
        · 采集到 0 条 → ``error_msg=`` 一句固定的排查提示、``video_data="[]"``；
        · 成功 → ``video_data`` 是缩进过的 JSON、``error_msg=""``。

    Raises:
        不抛异常：工具层 ``fetch_user_videos`` 自带全套兜底（任何失败都返回
        ``{"items": [], "error": "中文说明"}``），所以本节点不需要 try。
    """
    # `(x or "").strip()` 两步都不能省：`or ""` 兜住 None，`strip()` 兜住
    # 「用户只敲了空格」—— 那种输入不该被当成合法链接发去请求采集服务。
    url = (state.get("profile_url") or "").strip()
    if not url:
        # 空链接不走网络：采集服务没起时若还去请求，用户等到的是一段
        # "服务不可达"的长文案，而不是「你还没填链接」这个真正的原因。
        return {"error_msg": "请输入抖音用户主页链接", "video_data": "[]"}

    # 函数内 import 而不是文件顶部：`tools/douyin_client` 会连带加载 requests +
    # Cookie 解析等一堆东西，而本模块的**图结构自检**只想验拓扑、不想被工具层的
    # 导入期依赖牵连（与 `views/review.py`、`workflows/__init__.py` 里
    # `from langchain_core.messages import HumanMessage` 同一个理由）。
    from tools.douyin_client import fetch_user_videos

    print(f"[复盘] 正在采集: {url}")
    # limit=20：一次诊断 20 条作品足够看出内容偏好与转化率趋势；
    # 调大只会让 prompt 变长、token 更贵，诊断质量并不会线性提升。
    result = fetch_user_videos(url, limit=20)

    if result.get("error"):
        # 只打印第一行：工具层的 error 常常是多行排查指引（怎么重新拿 Cookie、怎么换链接），
        # 全打到日志里会淹掉本次运行的其他输出；完整原文仍然原样进 state 给页面展示。
        print(f"[复盘] 采集失败: {result['error'].splitlines()[0]}")
        return {"error_msg": result["error"], "video_data": "[]"}

    # 工具层"没报错"不等于"拿到了数据"：抓取成功但页面为空同样是失败，
    # 必须和 error 分开给文案 —— 否则用户会以为是自己链接写错了。
    items = result.get("items") or []
    if not items:
        return {
            "error_msg": "未获取到作品数据，请检查链接是否正确或 Cookie 是否有效",
            "video_data": "[]",
        }

    print(f"[复盘] 采集完成: {len(items)} 条作品")
    # `ensure_ascii=False` 保住中文（作品标题、发布时间都是中文）；
    # `indent=2` 是为了 prompt 可读 —— 缩进的 JSON 对模型的"逐条比对"更友好，
    # 而且页面「原始数据」展开区里人也能直接看。
    return {"video_data": json.dumps(items, ensure_ascii=False, indent=2), "error_msg": ""}


# ============================================================
# Node 2~4: LLM 诊断（prompt 与课案逐字一致）
# ============================================================

def _data_of(state: ReviewState) -> str:
    """取作品数据；没数据时返回空串，让各节点短路成「暂无数据」。

    这是三个 LLM 节点共用的**降级判据**，把「没数据」的所有写法归一成空串：
    没用过 ``_data_of`` 的写法是每个节点各自 ``if not state.get("video_data")``，
    那样 ``"[]"``（fetch 失败时写的值）会被当成"有数据"，模型就对着一个空数组
    编出一整套诊断结论。

    读 ``state["video_data"]``；不写任何键。

    Args:
        state: 复盘工作流的 state。

    Returns:
        非空时返回原 JSON 文本；``""`` / ``"[]"`` / ``"{}"`` 三种"空值"都返回空串。
    """
    data = (state.get("video_data") or "").strip()
    # 三种空形态都要认：`""` 是降级入口没给数据，`"[]"` 是 fetch 失败时写的，
    # `"{}"` 是上游（或用户粘贴）给了一个空对象。少认一种就会白烧一次 LLM。
    return "" if data in ("", "[]", "{}") else data


def node_funnel(state: ReviewState) -> dict:
    """Node 2: LLM 流量漏斗诊断。

    读 ``state["video_data"]``（经 ``_data_of()`` 归一）→ ``llm_call(temperature=0.3)``
    → 写 ``state["funnel_diagnosis"]``。

    温度 0.3：这一节点要**算转化率、比大小**，属于分析诊断，模型越稳越好；
    后两个节点（内容评估 / 优化方案）分别是 0.4 / 0.4，只有定位模块那种创作题才用 0.7。

    Returns:
        带 ``funnel_diagnosis`` 的 dict。没数据时是占位文案 ``NO_DATA``；
        ``llm_call`` 失败时是 ``[LLM调用失败: ...]`` 文本（不会抛异常）。
    """
    data = _data_of(state)
    if not data:
        # 短路成 NO_DATA 而不是空串：页面用 `or "（无）"` 兜空串，
        # 给空串就分不清"没跑"和"跑了没结果"；常量文案让两条路可区分。
        print("[复盘] 无作品数据，跳过漏斗诊断")
        return {"funnel_diagnosis": NO_DATA}

    prompt = f"""你是短视频运营诊断专家。以下是抖音个人主页的真实数据：

{data}

请诊断：
1. **播放/互动量分析** — 哪些作品播放/点赞/评论/分享/收藏高？高表现作品的共同特征是什么？
  （注：如果播放量为0或不含播放字段，说明是非本人主页，只基于互动数据做分析即可）
2. **转化分析** — 计算点赞→评论转化率、点赞→收藏转化率（如有播放量，加上播放→点赞转化率），找出各指标最优/最差作品
3. **流量漏斗** — 分析从曝光到互动的转化情况
4. **发布时间** — 哪个时间段发布效果最好？
每个结论用数据支撑。
"""
    print("[复盘] 漏斗诊断中 ...")
    # 契约：`llm_call` 失败返回 `[LLM调用失败: ...]` 文本而非抛异常，
    # 这一节没有 try/except 正是依赖它 —— 拿到失败串也照样写进 `funnel_diagnosis`，
    # 让 `suggest` 继续跑（会话结果里能看到失败原因，页面不会白屏）。
    return {"funnel_diagnosis": llm_call(prompt, temperature=0.3)}


def node_content(state: ReviewState) -> dict:
    """Node 3: LLM 内容质量评估。

    读 ``state["video_data"]``（经 ``_data_of()`` 归一）→ ``llm_call(temperature=0.4)``
    → 写 ``state["content_assessment"]``。

    它与 ``node_funnel`` 是**并列关系而非上下游**：两者都只吃原始数据，
    互相不读对方的产出（所以图上的 ``funnel → content`` 只是串行执行顺序，
    不代表数据依赖）。真正同时吃两者的是 ``node_suggest``。

    Returns:
        带 ``content_assessment`` 的 dict；没数据时 ``NO_DATA``，
        ``llm_call`` 失败时是失败提示文本。
    """
    data = _data_of(state)
    if not data:
        print("[复盘] 无作品数据，跳过内容评估")
        return {"content_assessment": NO_DATA}

    prompt = f"""基于这些抖音作品真实数据，评估内容质量：

{data}

从以下维度评估：
1. **内容方向** — 哪些主题/选题表现最好？有没有明显的内容偏好？
2. **标题技巧** — 高播放作品的标题有什么共同特征？
3. **节奏建议** — 根据播放/互动数据，建议最佳视频时长
4. **竞争分析** — 对比同赛道平均水平，这组数据的优劣势在哪里
"""
    print("[复盘] 内容评估中 ...")
    # 温度 0.4：比漏斗诊断（0.3）略松 —— 内容方向 / 标题技巧带一点主观判断，
    # 但整体仍是"评估"而非"创作"，所以不放到 0.7。
    return {"content_assessment": llm_call(prompt, temperature=0.4)}


def node_suggest(state: ReviewState) -> dict:
    """Node 4: LLM 生成可执行优化方案（吃 Node 2/3 的结论）。

    图上唯一读三样东西的节点：``state["video_data"]`` + ``state["funnel_diagnosis"]``
    + ``state["content_assessment"]`` → ``llm_call(temperature=0.4)``
    → 写 ``state["suggestions"]``。

    ⚠️ 它的短路判据只用 ``_data_of()``（看原始数据），**不看**上游两个诊断字段：
    这是可接受的设计 —— 漏斗/内容节点失败时写进的是失败提示文本，让模型
    "基于一份读不通的诊断"仍然给出一版通用优化方案，比直接跳过更有价值；
    用户能否看出上游失败由页面负责（本模块的失败串带 ``[LLM调用失败`` 前缀）。
    代价是：诊断失败时这个节点一定会真调一次 LLM。

    Returns:
        带 ``suggestions`` 的 dict；没数据时 ``NO_DATA``（此时**一次 LLM 都不调**），
        ``llm_call`` 失败时是失败提示文本。
    """
    data = _data_of(state)
    if not data:
        print("[复盘] 无作品数据，跳过优化策略")
        return {"suggestions": NO_DATA}

    # 两个上游结论用 `.get(..., "")` 取：`""` 进 f-string 就是一个空段落，
    # 模型会自己说"缺少诊断信息"，比传字面量 "None" 或 KeyError 都好。
    funnel = state.get("funnel_diagnosis", "")
    content = state.get("content_assessment", "")

    prompt = f"""基于抖音账号真实数据分析，生成可执行优化方案：

真实数据:
{data}

诊断结果:
{funnel}

内容评估:
{content}

请给出：
1. **立即执行**（3条 — 下一条视频就能用的改进）
2. **短期策略**（3条 — 1周内迭代方向）
3. **长期规划**（3条 — 1个月优化路径）
4. **A/B 测试方案** — 设计一个对比实验：改什么、怎么测、看什么指标
"""
    print("[复盘] 生成优化策略中 ...")
    # 与 content 同为 0.4：这里要输出「3 条立即执行 / 3 条短期 / 3 条长期」的
    # 结构化清单，太发散会跑成散文而失去可执行性。
    return {"suggestions": llm_call(prompt, temperature=0.4)}


# ============================================================
# 图：四节点串行（与课案的节点名、边完全一致）
# ============================================================
# 主图（`review_graph`）：START → fetch → funnel → content → suggest → END。
# 节点名与课案完全一致（`fetch` / `funnel` / `content` / `suggest`），
# 自检里对节点名与边做了精确断言 —— 改名等于改契约。

builder = StateGraph(ReviewState)
# 四个节点里的 `fetch` 与其余三个不同源：它是唯一碰外部服务（抖音采集 REST 接口）的，
# 也正因如此才会有下面那张「没有 fetch」的降级图。
builder.add_node("fetch", node_fetch_data)
builder.add_node("funnel", node_funnel)
builder.add_node("content", node_content)
builder.add_node("suggest", node_suggest)

# 五条边一条不能少：`fetch` 失败时把 `error_msg`/`video_data` 写进 state，
# 后面三个节点照常被**串行执行**（图里没有条件边，失败也不改道）——
# 但因为 `video_data` 是 `"[]"`，`_data_of()` 会让它们全部短路成 NO_DATA、不调 LLM。
# 也就是说「采集失败就不烧 token」是靠节点内的判空实现的，不是靠图的拓扑。
builder.add_edge(START, "fetch")
builder.add_edge("fetch", "funnel")
builder.add_edge("funnel", "content")
builder.add_edge("content", "suggest")
builder.add_edge("suggest", END)

review_graph = builder.compile()

# 降级入口用的图：与上面同源，只是少了采集节点（video_data 由调用方灌进来）
# ⚠️ 这是**第二张真的图**，不是同一张图的某个参数分支；两张图各自 compile 一次。
# 业务上两条路走的是同一批节点函数（`node_funnel` / `node_content` / `node_suggest`），
# 所以 prompt 与温度只有一份实现，不会两边跑偏。
_json_builder = StateGraph(ReviewState)
_json_builder.add_node("funnel", node_funnel)
_json_builder.add_node("content", node_content)
_json_builder.add_node("suggest", node_suggest)
_json_builder.add_edge(START, "funnel")
_json_builder.add_edge("funnel", "content")
_json_builder.add_edge("content", "suggest")
_json_builder.add_edge("suggest", END)

json_review_graph = _json_builder.compile()


# ============================================================
# 对外入口
# ============================================================

def _filled(state: dict, profile_url: str = "", error_msg: str = "") -> dict:
    """补齐 6 个字段 —— LangGraph 只返回被写过的键，前端不该为此兜底。

    ``invoke`` 的返回值里只含**被节点写过的键**：例如 ``run_review("")`` 会在
    ``node_fetch_data`` 的第一行就返回，`funnel` 等键根本不存在，
    而 ``views/review.py`` 无条件取 ``result["video_data"]``。
    这个函数就是那道「出口归一化」的闸门，**所有对外入口都必须经过它**。

    Args:
        state: ``invoke`` 的原始返回（可能缺键）。
        profile_url: 调用方知道的链接；state 里没有时用它兜底。
        error_msg: 调用方自己产生的错误说明（如「请输入抖音用户主页链接」）；
            优先级低于 state 里由节点写的 ``error_msg``。

    Returns:
        恰好 6 个键的 dict，``video_data`` 至少是 ``"[]"``（页面靠它判空）。
    """
    # 注意这里用的是 `or` 而不是 `.get(key, default)`：`state["error_msg"]` 在采集成功时
    # 是**空字符串**（正常值，不是缺键），`or` 会让它落到调用方给的兜底说明上；
    # 两个都为空时结果仍是 `""`，成功判定不受影响。
    return {
        "profile_url": state.get("profile_url") or profile_url,
        "video_data": state.get("video_data") or "[]",
        "funnel_diagnosis": state.get("funnel_diagnosis") or "",
        "content_assessment": state.get("content_assessment") or "",
        "suggestions": state.get("suggestions") or "",
        "error_msg": state.get("error_msg") or error_msg,
    }


def run_review(profile_url: str) -> dict:
    """运行数据复盘工作流（采集 → 漏斗诊断 → 内容评估 → 优化策略）。

    图：``review_graph``（四节点串行，见文件上半部分）。
    它**不判链接格式**：真正的解析、sec_uid 提取、Cookie 校验都在
    ``tools/douyin_client`` 里，本函数只拦「一个字都没填」。

    Args:
        profile_url: 抖音用户主页链接。

    Returns:
        ``{"profile_url", "video_data", "funnel_diagnosis",
           "content_assessment", "suggestions", "error_msg"}``
        **失败也只写 error_msg，不抛异常。**
        采集失败时 ``error_msg`` 非空、``video_data == "[]"``、
        三个诊断字段为空串或 ``NO_DATA``（页面看到 ``error_msg`` 就提前 return）。

    Raises:
        不抛异常。图级异常被收成 ``error_msg="复盘工作流异常: ..."``；
        这与 ``positioning.py`` 把失败写进**产出字段**的做法不同 ——
        本模块有独立的 ``error_msg`` 通道，语义更清楚（页面据此弹红条）。
    """
    # 先 `or ""` 兜 None 再 `strip()`：只敲了空格的输入不该被当成合法链接发出去。
    profile_url = (profile_url or "").strip()
    if not profile_url:
        # 提前返回，连图都不进 —— 省掉一次「采集服务不可达」的长文案，
        # 直接告诉用户真正的原因（没填链接）。
        return _filled({}, profile_url, "请输入抖音用户主页链接")

    try:
        # 入口 state 只带 `profile_url`，其余 5 个键由节点写。
        result = review_graph.invoke({"profile_url": profile_url})
    except Exception as exc:  # noqa: BLE001 —— 工作流出错不该把 Streamlit 页面带崩
        # Streamlit 的渲染代码不套 try：异常冒到这里就是整页 traceback。
        print(f"[复盘] 工作流异常: {exc}")
        return _filled({}, profile_url, f"复盘工作流异常: {exc}")
    return _filled(result, profile_url)


def run_review_from_json(raw_json: str) -> dict:
    """降级入口：跳过采集节点，用粘贴的作品数据跑 漏斗诊断 → 内容评估 → 优化策略。

    存在的理由：自托管采集服务（Docker）经常不在跑，而 agent 起不动 Docker。
    没有这个入口，「三个诊断节点的 prompt 对不对」在离线环境里就完全没法验证。
    **它跑的是同一批节点函数**，只是换成 ``json_review_graph``（无 ``fetch``）、
    由调用方把 ``video_data`` 直接灌进去。

    Args:
        raw_json: 用户粘贴的 JSON 数组 / 抖音原始响应 / CSV / TSV 文本。

    Returns:
        与 ``run_review()`` 同构的 dict；解析失败时只写 error_msg。
        ``profile_url`` 固定写成 ``"（手动粘贴数据）"`` —— 页面拿它做结果抬头，
        要能一眼看出这次不是采集来的。

    Raises:
        不抛异常。解析交给 ``tools.douyin_client.parse_manual_json()``
        （它内部自带全套兜底，失败返回 ``{"items": [], "error": "中文说明"}``）。
    """
    # 函数内 import：与 `node_fetch_data` 同一个理由 —— 让本模块的图结构自检
    # 不被工具层的导入期依赖牵连。
    from tools.douyin_client import parse_manual_json

    parsed = parse_manual_json(raw_json)
    if parsed.get("error"):
        # 工具层的 error 是多行排查指引（支持哪三种格式、每行至少要有什么字段），
        # 原样交给页面；这里不再包一层自己的文案，否则会把真正的用法说明盖掉。
        return _filled({}, "（手动粘贴数据）", parsed["error"])

    items = parsed.get("items") or []
    if not items:
        # 工具层没报错但列表为空：这一层也要拦，不能把 `[]` 灌进图 ——
        # 那会让三个节点都短路成 NO_DATA，用户看到的是"暂无数据"而不是"你粘贴的内容没解析出东西"。
        return _filled({}, "（手动粘贴数据）", "粘贴的数据里没有作品记录")

    data_json = json.dumps(items, ensure_ascii=False, indent=2)
    print(f"[复盘] 手动数据 {len(items)} 条作品，跳过采集节点直接诊断")
    try:
        # 两个入口字段一起给：`video_data` 是本入口的"命门"，
        # `profile_url` 只用于页面抬头与 `_filled` 的兜底。
        result = json_review_graph.invoke(
            {"profile_url": "（手动粘贴数据）", "video_data": data_json}
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[复盘] 工作流异常: {exc}")
        return _filled({}, "（手动粘贴数据）", f"复盘工作流异常: {exc}")
    return _filled(result, "（手动粘贴数据）")


# ============================================================
# 离线自检（服务没起来也必须全绿；加 --llm 才真调模型）
# ============================================================

if __name__ == "__main__":
    # 自检总原则：默认路径**离线可过**（采集服务没起来、LLM 额度为 0 也要全绿）；
    # 会真实消耗额度的那条链路放在 `--llm` 分支里，由人显式触发。
    print("=== 数据复盘工作流自检 ===")

    # 1) 图结构：节点与边必须与课案一致（这是「四节点串行」的硬约束）
    # 这里用的是「集合包含」（`<=`）而不是精确相等：LangGraph 未来若多出别的
    # 哨兵节点，这条断言不该跟着红；真正要锁死的是**这 5 条边都在**。
    graph = review_graph.get_graph()
    nodes = set(graph.nodes)
    assert {"fetch", "funnel", "content", "suggest"} <= nodes, nodes
    edges = {(e.source, e.target) for e in graph.edges}
    expected = {
        ("__start__", "fetch"),
        ("fetch", "funnel"),
        ("funnel", "content"),
        ("content", "suggest"),
        ("suggest", "__end__"),
    }
    assert expected <= edges, f"缺少边: {expected - edges}\n实际: {sorted(edges)}"
    print(f"  review_graph 结构        OK  {len(nodes)} 节点 / {len(edges)} 条边，串行顺序正确")

    # 2) 降级图：只有三个诊断节点，没有 fetch
    # `assert "fetch" not in json_nodes` 是这一节的要害：降级入口若误把采集节点也加进来，
    # 就会在离线环境下又去连采集服务，整个"离线可验"的前提就没了。
    json_nodes = set(json_review_graph.get_graph().nodes)
    assert {"funnel", "content", "suggest"} <= json_nodes, json_nodes
    assert "fetch" not in json_nodes, "降级入口不该有采集节点"
    print("  json_review_graph 结构   OK  无 fetch，仅三个诊断节点")

    # 3) 空输入 / 无效输入：返回结构化错误，不抛异常
    # 五种输入覆盖两条不同的失败路径：`""` / `"   "` 在工具层第一行就被拦，
    # `"不是 JSON"` 走「JSON 解析失败 → 改按 CSV/TSV 解析 → 仍然空」，
    # `"[]"` / `"{}"` 是**合法 JSON 但里面没有记录**。三条断言都要求它们被当成失败。
    for bad in ("", "   ", "不是 JSON", "[]", "{}"):
        result = run_review_from_json(bad)
        # `set(result) == {...}` 精确比集合（不看顺序）：`_filled` 的输出是六键契约，
        # 既不能少（页面 KeyError）也不能多（说明有人往出口塞了东西）。
        assert set(result) == {
            "profile_url", "video_data", "funnel_diagnosis",
            "content_assessment", "suggestions", "error_msg",
        }, result.keys()
        assert result["error_msg"], f"{bad!r} 应该给出 error_msg"
        # 失败时三个诊断字段必须是**空串**（而不是 NO_DATA、"分析中..."）：
        # 页面用 `error_msg` 判失败并提前 return，这里再确认它没被当成正常结果。
        assert result["suggestions"] == "", result["suggestions"]
        assert result["video_data"] == "[]"
    print("  run_review_from_json 失败路径 OK  6 字段齐全 + error_msg 非空")

    # 空链接走的是 `run_review` 自己的提前返回分支（压根不进图）。
    result = run_review("")
    assert result["error_msg"] and result["video_data"] == "[]"
    print("  run_review 空链接         OK")

    # 4) 真实 LLM 链路（默认跳过，保证离线全绿）
    if "--llm" in sys.argv:
        # 样例数据来自 `tools/douyin_client.SAMPLE_MANUAL_JSON`（两份假作品，带中文键），
        # 用它才能保证这条用例不依赖网络、只依赖额度。
        from tools.douyin_client import SAMPLE_MANUAL_JSON

        print("  --llm：真调模型跑完整链路（漏斗诊断 → 内容评估 → 优化策略）...")
        result = run_review_from_json(SAMPLE_MANUAL_JSON)
        assert not result["error_msg"], result["error_msg"]
        # 三个产出**逐个**过可用性判据，并把原文打出来：
        # 一眼能分清是「没配密钥」（`[LLM未配置]`）还是「调用报错/额度耗尽」（`[LLM调用失败: …]`）。
        # ⚠️ 旧版本这里只判"非空"，**不认失败前缀** —— 而 `llm_call` 失败返回的
        # 提示文本同样是真值，于是没配 key 也会打印「LLM 链路 OK」（假绿，本轮修掉）。
        _outputs = (
            ("漏斗诊断", "funnel_diagnosis"),
            ("内容评估", "content_assessment"),
            ("优化策略", "suggestions"),
        )
        _bad = [
            (label, key, _llm_fail_reason(result[key]), result[key])
            for label, key in _outputs
            if _llm_fail_reason(result[key])
        ]
        for _label, _key, _why, _text in _bad:
            print(f"    ✗ {_label}（{_key}）{_why}；原文：{_text!r}")
        # 这里必须是 `assert`（退出码非 0）而不是只 `print`：
        # `verify_all.py` 这类**子进程**场景只看退出码，只打印的话失败照样是绿。
        assert not _bad, "--llm 链路没拿到可用正文：" + "；".join(
            f"{label}={why}: {text[:200]!r}" for label, _, why, text in _bad
        )
        print(f"    漏斗诊断: {result['funnel_diagnosis'][:120]}...")
        print(f"    优化策略: {result['suggestions'][:120]}...")
        print("  LLM 链路                  OK")
    else:
        print("  LLM 链路                  跳过（加 --llm 才跑，会真实消耗额度）")

    print("\n自检完成")
