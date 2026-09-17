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
    """账号定位工作流的共享状态（契约已冻结，字段名不要改）。"""

    user_info: str    # 输入：职业 / 技能 / 兴趣 / 目标平台拼成的多行文本
    profile: str      # 节点①产出：四维画像分析
    competitors: str  # 节点②产出：5 个对标账号
    plan: str         # 节点③产出：完整定位策划方案


# ==========================================================================
# 节点
# ==========================================================================
def node_analyze(state: PositioningState) -> dict:
    """节点①：分析用户画像（专业优势 / 内容风格 / 推荐赛道 / 差异化）。"""
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
        return {"profile": llm_call(prompt, temperature=_TEMP)}
    except Exception as exc:  # noqa: BLE001 —— 节点绝不抛，失败也要把字段填上
        print(f"[账号定位] ① 画像分析失败: {exc}")
        return {"profile": f"[画像分析失败: {exc}]"}


def node_competitors(state: PositioningState) -> dict:
    """节点②：基于画像推荐 5 个对标账号（2 头部 + 3 腰部）。"""
    try:
        profile = state.get("profile", "") or ""
        prompt = f"""基于以下用户画像，推荐5个对标账号：

{profile}

要求：
- 2个头部大号（天花板参考），3个腰部账号（可追赶目标）
- 每个账号说明：账号名、粉丝量级、内容特色、可借鉴点
"""
        print("[账号定位] ② 搜索对标账号中...")
        return {"competitors": llm_call(prompt, temperature=_TEMP)}
    except Exception as exc:  # noqa: BLE001
        print(f"[账号定位] ② 对标账号失败: {exc}")
        return {"competitors": f"[对标账号分析失败: {exc}]"}


def node_plan(state: PositioningState) -> dict:
    """节点③：汇总画像 + 对标，产出完整定位策划方案。"""
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
        return {"plan": llm_call(prompt, temperature=_TEMP)}
    except Exception as exc:  # noqa: BLE001
        print(f"[账号定位] ③ 定位方案失败: {exc}")
        return {"plan": f"[定位方案生成失败: {exc}]"}


# ==========================================================================
# 构建图：START → analyze → competitors → plan → END
# ==========================================================================
builder = StateGraph(PositioningState)
builder.add_node("analyze", node_analyze)
builder.add_node("competitors", node_competitors)
builder.add_node("plan", node_plan)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "competitors")
builder.add_edge("competitors", "plan")
builder.add_edge("plan", END)

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
        即使 LLM 全失败也会返回这四个键（值是中文提示文本，不会抛异常）。
    """
    try:
        result = positioning_graph.invoke({"user_info": user_info or ""})
    except Exception as exc:  # noqa: BLE001 —— 图本身出错也不往上抛
        print(f"[账号定位] 工作流执行失败: {exc}")
        result = {}

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
    """真实联网跑一次（会调 LLM，花钱）—— 只在 ``--live`` 时执行。"""
    print("\n=== 真实联网跑一次（--live）===")
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
    print("=== 账号定位工作流 自检（离线，不联网）===")

    # ---- 1) 图结构 ----
    graph = positioning_graph.get_graph()
    print(f"图节点: {sorted(graph.nodes)}")
    print(f"图边:   {graph.edges}")

    assert sorted(graph.nodes) == [
        "__end__", "__start__", "analyze", "competitors", "plan",
    ], sorted(graph.nodes)

    edge_pairs = sorted((e.source, e.target) for e in graph.edges)
    assert edge_pairs == [
        ("__start__", "analyze"),
        ("analyze", "competitors"),
        ("competitors", "plan"),
        ("plan", "__end__"),
    ], edge_pairs
    print("  ✓ 3 节点串行链路接对了")

    # ---- 2) State 契约冻结 ----
    assert list(PositioningState.__annotations__) == [
        "user_info", "profile", "competitors", "plan",
    ], PositioningState.__annotations__
    print("  ✓ PositioningState 字段与契约一致")

    # ---- 3) 给 llm_call 打桩，把整张图真跑一遍（不联网）----
    # 节点内部查的是模块全局的 llm_call，所以这里替换模块级名字就能生效。
    captured: list[tuple[str, float]] = []

    def _stub_llm(prompt: str, temperature: float = 0.5, fallback: str = "") -> str:
        captured.append((prompt, temperature))
        return f"[STUB-{len(captured)}]"

    _real_llm = llm_call
    llm_call = _stub_llm  # noqa: F841 —— 故意覆盖模块全局，给节点用
    try:
        result = run_positioning("职业：测试工程师")
    finally:
        llm_call = _real_llm

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
    captured.clear()
    llm_call = _stub_llm  # noqa: F841
    try:
        empty_result = positioning_graph.invoke({})
    finally:
        llm_call = _real_llm

    assert len(captured) == 3, len(captured)
    assert set(empty_result) >= {"profile", "competitors", "plan"}, set(empty_result)
    print("  ✓ 空 state 也能走完整条链路（不 KeyError）")

    # ---- 5) run_positioning 的失败兜底：LLM 全挂也返回四个键 ----
    def _boom_llm(prompt: str, temperature: float = 0.5, fallback: str = "") -> str:
        return "[LLM调用失败: 模拟]"

    llm_call = _boom_llm  # noqa: F841
    try:
        bad_result = run_positioning("x")
    finally:
        llm_call = _real_llm

    assert list(bad_result) == ["user_info", "profile", "competitors", "plan"], list(bad_result)
    assert bad_result["plan"].startswith("[LLM调用失败"), bad_result["plan"]
    print("  ✓ LLM 失败时仍返回完整四字段")

    print("\n全部自检通过")

    if "--live" in sys.argv:
        _live_check()
    else:
        print("（跳过联网验证；加 --live 可真实调用一次 LLM）")
