# -*- coding: utf-8 -*-
"""账号定位工作流 —— 3 步串行：画像分析 → 对标搜索 → 方案生成

课案出处：《3.自媒体Agent》→ 账号定位 → workflows/positioning.py

本节要讲什么
    1. LangGraph 的最小可用形态。整张图就是一条直线：
       START → analyze → competitors → plan → END。
       没有分支、没有并行、没有 checkpointer —— 定位模块的价值全在 prompt 上，
       先用最简单的图把「前端 views → 工作流 workflows → LLM」三层串通，
       后面热点监控（Send 并行）、内容复刻（带工具的节点）才有对照物。
    2. 节点返回的是 **差量 dict**，不是完整 state。
       `node_analyze` 只 `return {"profile": ...}`，剩下三个字段由 LangGraph 合并。
       这是和「普通函数传 dict」最容易搞混的地方：写惯了 `state["x"] = ...; return state`
       的人会在这里把并发/合并语义搞坏。
    3. 三个节点依次「吃上一步的产出」：competitors 的 prompt 里是 profile，
       plan 的 prompt 里是 profile + competitors。链路一旦断在中间，
       后面拿到的就是空串 —— 所以每个节点都要能容忍空输入。

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 数据库落库 | `run_positioning` 里 `with get_session()` 写 `PositioningRecord` | **整段删除** | 课案没给 DB 层实现，`get_session` / `PositioningRecord` 从未 import，照抄必 `NameError`，又被 `except` 吞成一行日志 |
    | 节点容错 | 直接 `state["user_info"]`，无 try | `state.get(...)` + 每个节点 try/except | 单点失败不该让整条图挂掉；失败也要把字段写回 state 让链路走完 |
    | import 路径 | 直接 `from workflows import llm_call` | 文件顶部先做 path 引导再 import | 直接 `python workflows/positioning.py` 时 `sys.path[0]` 是 `workflows/`，`.venv` 的 `.pth` 只加了仓库根，会 `ModuleNotFoundError` |
    | 自检 | 无 | 末尾 `__main__` 离线自检（给 `llm_call` 打桩，把整张图真跑一遍） | 必须能脱离 Streamlit 和网络验证「图接对了、字段串起来了」 |
    | 温度 | 0.7 / 0.7 / 0.7 | 同左 | 创作类任务是高温度场景，不需要调 |
    | prompt | 见课案 | **逐字保留** | prompt 是这个模块的全部价值，不改 |

踩过的坑
    · `from workflows import llm_call` 在「脚本方式」下会失败：`python workflows/positioning.py`
      的 `sys.path[0]` 是脚本所在目录 `workflows/`，不是项目根。
      直接跑 `workflows/base.py` 复现过 `ModuleNotFoundError: No module named 'workflows'`。
    · `positioning_graph.get_graph().draw_ascii()` 需要额外的 `grandalf` 包（未装），
      自检里只看 `nodes` / `edges`，不依赖这个可选依赖。
    · Windows 控制台默认 GBK，中文/emoji 打印会炸 —— 顶部做了 UTF-8 兜底。

本文件在图里干了什么（节点 / 边 / 状态流向）
    三个节点串行成一条链，四条边就是全部拓扑，没有任何分支与并行::

        START ──> analyze ──> competitors ──> plan ──> END
                  读 user_info  读 profile    读 profile + competitors
                  写 profile    写 competitors 写 plan

    · ``analyze`` 读 ``user_info``，写 ``profile``；
      ``competitors`` 读 ``profile``，写 ``competitors``；
      ``plan`` 读 ``profile`` + ``competitors``，写 ``plan``。
      每个节点只认自己那一前一后 —— 这就是「差量 dict」在链路里的实际含义。
    · 编译器是 ``builder.compile()``，结果 ``positioning_graph`` 在模块级别创建一次；
      ``run_positioning()`` 每次都复用它，不会重复建图。

自检怎么验的（离线，不联网、不花钱）
    自检**给 ``llm_call`` 打桩**，把整张图真跑一遍（不是 mock 掉 ``invoke``）：
    假 LLM 每次返回 ``[STUB-n]`` 并记下 prompt，于是三件事可证：
    「3 次调用」「后一个节点的 prompt 里带着前一个节点的产出」「字段按
    user_info → profile → competitors → plan 的次序流下去」。
    另外还验了空 state（``invoke({})`` 走 ``state.get`` 兜底，不 KeyError）
    与两条失败路径（LLM 全挂 / 图级 ``invoke`` 抛异常）都必须返回完整四字段。

运行方式
    离线自检（不联网、不花钱）::

        Set-Location F:\\ProGram\\Python_Base\\Media_Agent
        & ..\\.venv\\Scripts\\python.exe workflows\\positioning.py

    真实联网跑一次（会调 LLM）::

        & ..\\.venv\\Scripts\\python.exe workflows\\positioning.py --live
"""

import sys
from pathlib import Path

# Windows 控制台默认 GBK，本模块会打印中文日志
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 直接以脚本方式运行时，sys.path[0] 是 workflows/ 目录，
# 项目根（Media_Agent）不在里面 —— 必须赶在 `from workflows import ...` 之前补上。
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import TypedDict  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402

from workflows import llm_call  # noqa: E402

# 创作类任务用高温度：定位方案要有人味，不要四平八稳的模板话
_TEMP = 0.7


class PositioningState(TypedDict):
    """账号定位工作流的共享状态（契约已冻结，字段名不要改）。

    四个字段就是三个节点之间的全部数据通道，依次往下流::

        user_info ──(analyze)──> profile ──(competitors)──> competitors ──(plan)──> plan

    · 顺序约定：节点只读上游已写好的键，写的键给下游用；
      所以任一节点失败时**必须照样把输出键写回**（哪怕只写一句失败提示），
      否则下游读到空串，会拿着空 prompt 去问模型。
    · ``str`` 全部是**纯文本**：链路里不传结构化对象，
      LLM 的原始输出（含 Markdown）直接落进字段，前端按 Markdown 渲染。
    · 这是 LangGraph 的 **schema**，不是运行时容器：三点说明 ——
      ① ``invoke`` 传进来的初始值只需带 ``user_info``，其余键由节点写入；
      ② 节点里读键统一用 ``state.get("key", "")``，缺键不会 ``KeyError``；
      ③ ``total`` 默认为 ``True``，即四个键都是「必填」，
         但 LangGraph 不做运行期校验，真正兜底的是节点里的 ``.get`` 与 ``run_positioning`` 的补齐。
    """

    user_info: str    # 输入：职业 / 技能 / 兴趣 / 目标平台拼成的多行文本
    profile: str      # 节点①产出：四维画像分析
    competitors: str  # 节点②产出：5 个对标账号
    plan: str         # 节点③产出：完整定位策划方案


# ==========================================================================
# 节点
# ==========================================================================
def node_analyze(state: PositioningState) -> dict:
    """节点①：分析用户画像（专业优势 / 内容风格 / 推荐赛道 / 差异化）。

    图上的位置：``START`` 的第一跳，链路上游，没有更早的产出可用。

    读 ``state["user_info"]`` → 构造「四维分析」prompt → ``llm_call(temperature=0.7)``
    → 写 ``state["profile"]``。

    失败时的返回契约：``llm_call`` 本身**不抛异常**（失败会返回
    ``[LLM调用失败: ...]`` 这类提示文本），所以这里的 ``try`` 只兜
    「prompt 拼装 / 读 state 等意外异常」；无论走哪条路都返回带 ``profile`` 的
    dict、绝不往图外抛 —— 节点一抛，整张图就断在第一步，连 ``competitors`` /
    ``plan`` 都不会跑。

    ⚠️ 失败分支**不能返回空串**：``views/positioning.py`` 把空串渲染成
    「分析中...」，用户分不清「在跑」和「挂了」；必须回填以 ``[画像分析失败``
    开头的文案，页面才认得出并弹红条。
    """
    try:
        user_info = state.get("user_info", "") or ""
        prompt = f"""你是自媒体定位分析师。分析以下用户的画像：

【用户信息】
{user_info}

从4个维度分析：
1. 专业优势（核心竞争力）
2. 内容风格匹配（性格适合什么风格）
3. 3个推荐赛道（按优先级，说明理由）
4. 差异化切入点（与同类账号的区别）
"""
        print("[账号定位] ① 画像分析中...")
        # 只写 `profile` 这一个键，其余三个由 LangGraph 合并 —— 节点返回的是**差量**，
        # 不是完整 state（对照：直接 `state["profile"] = x; return state` 在并行图里语义会坏掉）。
        # 调用契约：`llm_call` 失败时返回 `[LLM调用失败: ...]` 文本，不抛异常，
        # 所以这里拿到的一定是 str，可以直接落进字段。
        return {"profile": llm_call(prompt, temperature=_TEMP)}
    except Exception as exc:  # noqa: BLE001 —— 节点绝不抛，失败也要把字段填上
        # 走到这里说明是「读 state / 拼 prompt / print 编码」这类意外异常，不是 LLM 报错。
        # 仍然不往上抛：`run_positioning()` 与 Streamlit 调用方都按「返回值」而不是
        # 「捕获异常」来判定失败，抛出去会绕过页面那套前缀匹配的告警逻辑。
        print(f"[账号定位] ① 画像分析失败: {exc}")
        return {"profile": f"[画像分析失败: {exc}]"}


def node_competitors(state: PositioningState) -> dict:
    """节点②：基于画像推荐 5 个对标账号（2 头部 + 3 腰部）。

    图上的位置：链条中段，上游是 ``analyze``。

    读 ``state["profile"]`` → 构造「推荐对标账号」prompt →
    ``llm_call(temperature=0.7)`` → 写 ``state["competitors"]``。

    失败时的返回契约：与节点① 同构 —— ``llm_call`` 不抛异常，``try`` 只兜意外，
    失败回填 ``[对标账号分析失败: ...]`` 而**不是**抛给 LangGraph；
    空串会被页面当成「正在分析」，同样是必须避开的写法。
    """
    try:
        # ``or ""`` 是必需的兜底：上游 ``profile`` 可能是 None（比如调用方显式传了 None），
        # 直接进 f-string 会渲染出字面量 "None"，模型就会去分析一个叫 None 的人；
        # 传空串至少得到一个诚实的空段落，模型会自行说明信息不足，而不是照着 "None" 幻觉。
        # 另外 `state.get(...)` 而非 `state[...]`：初始 state 里本来就没有 `profile`
        # 这个键（它由节点①写入），缺键不应 KeyError。
        profile = state.get("profile", "") or ""
        prompt = f"""基于以下用户画像，推荐5个对标账号：

{profile}

要求：
- 2个头部大号（天花板参考），3个腰部账号（可追赶目标）
- 每个账号说明：账号名、粉丝量级、内容特色、可借鉴点
"""
        print("[账号定位] ② 搜索对标账号中...")
        # 这里没有 `if not profile` 的短路：空画像时「让模型自己说信息不足」比
        # 「直接跳过这一步、留个空字段」体验更好；真正的兜底在节点①的失败文案上。
        return {"competitors": llm_call(prompt, temperature=_TEMP)}
    except Exception as exc:  # noqa: BLE001
        print(f"[账号定位] ② 对标账号失败: {exc}")
        # 失败文案的前缀是有契约的：`views/positioning.py` 的 `_FAIL_PREFIXES`
        # 逐条列了这三个节点各自的前缀，改文案等于改契约，页面就不弹红条了。
        return {"competitors": f"[对标账号分析失败: {exc}]"}


def node_plan(state: PositioningState) -> dict:
    """节点③：汇总画像 + 对标，产出完整定位策划方案。

    图上的位置：链条末端，``END`` 的最后一跳；它的产出就是用户看到的那份方案。

    读 ``state["profile"]`` + ``state["competitors"]``（**两个上游产出一起读**，
    这是本模块唯一的多输入节点）→ 构造「六段式方案」prompt →
    ``llm_call(temperature=0.7)`` → 写 ``state["plan"]``。

    失败时的返回契约：同节点①②，回填 ``[定位方案生成失败: ...]``，不抛异常。
    注意它**不做上游判空短路** —— 前两个节点已经把失败文案写进字段了，
    即便此刻 ``profile``/``competitors`` 是失败提示，也让模型照常输出一版方案，
    用户至少能拿到点东西；真正的告警由页面对三个字段做前缀匹配来完成。
    """
    try:
        profile = state.get("profile", "") or ""
        competitors = state.get("competitors", "") or ""
        prompt = f"""基于以下信息，生成完整的账号定位策划方案：

【用户画像】
{profile}

【对标账号】
{competitors}

输出结构化方案：
1. **一句话定位**（你是谁，提供什么价值）
2. **目标受众**（3类核心粉丝画像）
3. **内容版图**（5个内容系列，每个说明主题+形式+频率）
4. **人设打造**（语言风格、视觉风格、记忆点）
5. **变现路径**（3条商业模式，从近到远排列）
6. **30天起号计划**（每周运营重点和内容数量）
"""
        print("[账号定位] ③ 生成定位方案中...")
        # 温度维持 0.7（模块级 `_TEMP`）：策划方案属于创作类输出，
        # 低温度会退化成四平八稳的模板话，反而没有可用性。
        return {"plan": llm_call(prompt, temperature=_TEMP)}
    except Exception as exc:  # noqa: BLE001
        print(f"[账号定位] ③ 定位方案失败: {exc}")
        return {"plan": f"[定位方案生成失败: {exc}]"}


# ==========================================================================
# 构建图：START → analyze → competitors → plan → END
# ==========================================================================
# `StateGraph(schema)` 只是拿到一个「可加节点/边的构建器」，此时还没跟 LangGraph 绑定；
# 图的编译（compile）、执行（invoke）都由下面与 `run_positioning` 负责。
builder = StateGraph(PositioningState)
# 节点名（"analyze"）是图里的标识，与 Python 函数名解耦 ——
# 自检里的 `sorted(graph.nodes)` 断言比对的就是这三个名字，改名等于改契约。
builder.add_node("analyze", node_analyze)
builder.add_node("competitors", node_competitors)
builder.add_node("plan", node_plan)

# 四条边一条不能少，这就是整张图的拓扑（无分支、无并行）：
# `START` / `END` 是 LangGraph 的两个哨兵节点，编译后会变成 `__start__` / `__end__`；
# `add_edge(a, b)` 的语义是「a 这个节点返回后，必然执行 b」——
# 顺序由边的连接方式决定，跟 `add_node` 的书写次序无关。
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "competitors")
builder.add_edge("competitors", "plan")
builder.add_edge("plan", END)

# 编译一次、全进程复用：每次 `run_positioning` 都重新 compile 会白白重建图对象。
positioning_graph = builder.compile()


# ==========================================================================
# 对外接口
# ==========================================================================
def run_positioning(user_info: str) -> dict:
    """运行定位工作流，返回含四个字段的完整 state。

    课案在这里多了一段「把结果存数据库」的代码，
    调用了从未 import 的 ``get_session()`` / ``PositioningRecord`` —— **已整段删除**。
    本仓库这一版不落库：结果由 Streamlit 页面存进 ``st.session_state``。

    Args:
        user_info: 用户背景信息（职业 / 技能 / 兴趣 / 目标平台的拼接文本）。

    Returns:
        ``{"user_info","profile","competitors","plan"}``。
        即使 LLM 全失败也会返回这四个键（值是中文提示文本，不会抛异常）；
        图级异常时三个产出字段都回填 ``[工作流执行失败: ...]``，四个键依然齐全。

    Raises:
        不抛异常。**这是与 ``views/`` 的约定** —— 页面按「返回文本的前缀」
        判定成功/失败（``_FAIL_PREFIXES``），抛出去反而绕过了那套告警。
    """
    try:
        # 初始 state 只塞 `user_info`，另外三个键留给节点写；
        # `user_info or ""` 是为了拦住 None —— 传 None 进去 f-string 会渲染成 "None"。
        result = positioning_graph.invoke({"user_info": user_info or ""})
    except Exception as exc:  # noqa: BLE001 —— 图本身出错也不往上抛
        # 能落到这里的通常是「图还没编译好 / 状态 schema 对不上 / 检查点损坏」这类
        # 结构性错误，而不是普通的业务失败。仍然收成返回值：
        # Streamlit 的渲染代码不会套 try，抛出去就是整页 traceback（白屏）。
        print(f"[账号定位] 工作流执行失败: {exc}")
        # 不能只留空串：页面把空串渲染成「分析中...」，用户分不清「在跑」和「挂了」。
        # 三个产出字段统一回填失败文案，前端 `_FAIL_PREFIXES` 才认得出、才弹红条。
        failed = f"[工作流执行失败: {exc}]"
        result = {"profile": failed, "competitors": failed, "plan": failed}

    # 出口再包一层 `.get` 补齐：正常路径下四键齐全，但这层保证「契约不依赖节点的正确性」——
    # 只要 LangGraph 只返回「被写过的键」（节点被跳过、图被改动），这里也不会漏键。
    return {
        "user_info": result.get("user_info", user_info or ""),
        "profile": result.get("profile", ""),
        "competitors": result.get("competitors", ""),
        "plan": result.get("plan", ""),
    }


# ==========================================================================
# 离线自检
# ==========================================================================
def _live_check() -> None:
    """真实联网跑一次（会调 LLM，花钱）—— 只在 ``--live`` 时执行。

    与离线自检的区别：这里**不打桩**，走真实的 ``llm_call``，
    因此会消耗额度、且结果每次不同（0.7 温度）；只打印前 300 字供人眼确认，
    不做断言 —— 非确定输出没法做断言，硬断言只会变成 flaky 测试。
    """
    print("\n=== 真实联网跑一次（--live）===")
    # 用一份贴近真实用户的样例输入；必须有值，空输入会走各节点的 `state.get(..., "")` 兜底，
    # 那验证的就不是「prompt 对不对」而是「空跑能不能活」了。
    demo = (
        "职业：Python后端开发\n"
        "技能：Python、Docker、K8s\n"
        "兴趣：AI工具、效率提升\n"
        "目标平台：B站"
    )
    result = run_positioning(demo)
    for key in ("profile", "competitors", "plan"):
        text = result.get(key, "")
        print(f"\n--- {key}（{len(text)} 字，前 300 字）---")
        print(text[:300])


if __name__ == "__main__":
    # 自检总原则：默认路径**必须离线可过**（不联网、不花钱、不依赖 Streamlit）。
    # 真调模型单独留在 `--live` 分支里，由人显式触发。
    print("=== 账号定位工作流 自检（离线，不联网）===")

    # ---- 1) 图结构 ----
    # `get_graph()` 拿的是 Mermaid/ASCII 用的只读视图（含 `__start__` / `__end__` 两个哨兵），
    # 不是可执行对象；`.nodes` 是 dict，所以 `sorted()` 排的是节点名。
    # 注意别顺手改成 `draw_ascii()`：它需要额外的 `grandalf` 包（本机未装），会直接报错。
    graph = positioning_graph.get_graph()
    print(f"图节点: {sorted(graph.nodes)}")
    print(f"图边:   {graph.edges}")

    assert sorted(graph.nodes) == [
        "__end__", "__start__", "analyze", "competitors", "plan",
    ], sorted(graph.nodes)

    # 边的顺序不保证（`graph.edges` 是 set），所以先归一化成有序的 (源, 目标) 再比 ——
    # 直接比 `graph.edges` 会时不时假失败。
    edge_pairs = sorted((e.source, e.target) for e in graph.edges)
    assert edge_pairs == [
        ("__start__", "analyze"),
        ("analyze", "competitors"),
        ("competitors", "plan"),
        ("plan", "__end__"),
    ], edge_pairs
    print("  ✓ 3 节点串行链路接对了")

    # ---- 2) State 契约冻结 ----
    # `__annotations__` 保留**声明顺序**，所以这里能顺带锁住字段的书写次序；
    # 前端 `views/positioning.py` 直接按这三个名字取结果，改名/加字段都会在这里被拦下。
    assert list(PositioningState.__annotations__) == [
        "user_info", "profile", "competitors", "plan",
    ], PositioningState.__annotations__
    print("  ✓ PositioningState 字段与契约一致")

    # ---- 3) 给 llm_call 打桩，把整张图真跑一遍（不联网）----
    # 节点内部查的是模块全局的 llm_call，所以这里替换模块级名字就能生效。
    # ⚠️ 之所以「覆盖模块全局」而不是用 mock 库：节点函数体里写的是裸名字 `llm_call`，
    #    它在**调用时**才去本模块的 globals 里查，所以直接赋值就是最轻的打桩方式。
    captured: list[tuple[str, float]] = []

    def _stub_llm(prompt: str, temperature: float = 0.5, fallback: str = "") -> str:
        # 签名必须与 `workflows.llm_call` 完全一致（含 `fallback`）：节点是按位置/关键字传参的，
        # 少一个参数就会在调用点 TypeError —— 这类坑不报在打桩处、报在节点里，很难查。
        captured.append((prompt, temperature))
        return f"[STUB-{len(captured)}]"

    _real_llm = llm_call
    llm_call = _stub_llm  # noqa: F841 —— 故意覆盖模块全局，给节点用
    try:
        result = run_positioning("职业：测试工程师")
    finally:
        # 必须还原：本文件后续用例还要继续覆盖它，但收尾一定要把真身放回去，
        # 否则 `--live`（同一个进程里往下走）会拿着桩跑，看起来"通过"其实什么都没调。
        llm_call = _real_llm

    # 三条断言分别锁住：调用次数（没有多余 LLM 往返）、温度（模块级 `_TEMP`）、
    # 返回值被正确落进对应字段（`profile`←第 1 次、`competitors`←第 2 次、`plan`←第 3 次）。
    assert len(captured) == 3, f"应该只调 3 次 LLM，实际 {len(captured)}"
    assert all(t == 0.7 for _, t in captured), [t for _, t in captured]
    assert list(result) == ["user_info", "profile", "competitors", "plan"], list(result)
    assert result["profile"] == "[STUB-1]", result["profile"]
    assert result["competitors"] == "[STUB-2]", result["competitors"]
    assert result["plan"] == "[STUB-3]", result["plan"]

    # 关键：后一个节点的 prompt 里必须带上前一个节点的产出
    assert "职业：测试工程师" in captured[0][0], captured[0][0]
    assert "[STUB-1]" in captured[1][0], captured[1][0]
    assert "[STUB-2]" in captured[2][0], captured[2][0]
    print("  ✓ 打桩跑通：3 个节点依次串接，prompt 里带上了上游产出")

    # ---- 4) 空输入不能把图搞崩（state.get 兜底）----
    # 走 `invoke({})` 而不是 `run_positioning("")`：这里要验的是**图本身**能接受空 state，
    # 三个节点各自的 `state.get("key", "")` 就是为它准备的（缺键不 KeyError）。
    captured.clear()
    llm_call = _stub_llm  # noqa: F841
    try:
        empty_result = positioning_graph.invoke({})
    finally:
        llm_call = _real_llm

    # 即使输入为空，三个节点也必须各跑一次（不做「空输入就跳过 LLM」的短路）：
    # 短路的代价是用户只能看到一句"没输入"，而让模型说明「信息不足」体验更好。
    assert len(captured) == 3, len(captured)
    # 用 `>=` 而不是 `==`：`invoke({})` 的返回里不会带没写过的 `user_info`，
    # 所以这里只断言「产出三键都在」，逐键补齐是 `run_positioning` 的职责。
    assert set(empty_result) >= {"profile", "competitors", "plan"}, set(empty_result)
    print("  ✓ 空 state 也能走完整条链路（不 KeyError）")

    # ---- 5) run_positioning 的失败兜底：LLM 全挂也返回四个键 ----
    def _boom_llm(prompt: str, temperature: float = 0.5, fallback: str = "") -> str:
        # 模拟 `llm_call` 的真实失败形态：**返回提示文本而不是抛异常**。
        # 用它才能验出「节点拿到失败串后照旧写进字段、链路继续往下走」这条路径。
        return "[LLM调用失败: 模拟]"

    llm_call = _boom_llm  # noqa: F841
    try:
        bad_result = run_positioning("x")
    finally:
        llm_call = _real_llm

    # 这条断言锁的是「键序 + 键数」：`list(dict)` 给的是插入顺序，
    # 也正是页面的取值顺序；少一个键页面就会 KeyError（所以 `run_positioning` 才要补齐）。
    assert list(bad_result) == ["user_info", "profile", "competitors", "plan"], list(bad_result)
    assert bad_result["plan"].startswith("[LLM调用失败"), bad_result["plan"]
    print("  ✓ LLM 失败时仍返回完整四字段")

    # ---- 6) 图级异常：三个产出字段必须回填失败文案，不能是空串 ----
    # 空串会被页面渲染成「分析中...」，用户分不清「在跑」和「挂了」。
    class _BoomGraph:
        """假的图对象：invoke 必抛异常。

        用鸭子类型的最小替身（只需 `invoke`）而不是 mock 库：
        要验的就是 `run_positioning` 里那个 `except` 分支，越简单越不容易自欺。
        """

        def invoke(self, *args, **kwargs):
            raise RuntimeError("模拟图构造失败")

    _real_graph = positioning_graph
    positioning_graph = _BoomGraph()  # noqa: F841 —— 故意覆盖模块全局，给 run_positioning 用
    try:
        boom_result = run_positioning("职业：测试工程师")
    finally:
        positioning_graph = _real_graph

    # 三条断言依次是：键齐全 → 三块产出都是同一条失败文案 → 用户输入仍被原样带回来
    # （`user_info` 不能在失败路径里被丢掉，页面要拿它做「本次输入」的说明）。
    assert list(boom_result) == ["user_info", "profile", "competitors", "plan"], list(boom_result)
    assert boom_result["profile"].startswith("[工作流执行失败"), boom_result["profile"]
    assert boom_result["profile"] == boom_result["competitors"] == boom_result["plan"]
    assert boom_result["user_info"] == "职业：测试工程师", boom_result["user_info"]
    print("  ✓ 图级异常时回填失败文案，不再是三个空串")

    print("\n全部自检通过")

    if "--live" in sys.argv:
        _live_check()
    else:
        print("（跳过联网验证；加 --live 可真实调用一次 LLM）")
