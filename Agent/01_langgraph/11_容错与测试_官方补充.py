# -*- coding: utf-8 -*-
r"""
LangGraph 官方补充篇②：容错与测试（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangGraph **官方文档**补课案完全空白的两个工程化维度：
      - fault-tolerance.mdx / use-graph-api.mdx（Add retry policies / Set node timeouts /
        Handle node errors）→ 节点级重试、节点超时、错误处理
      - test.mdx（三种 pytest 模式）→ 整图测试、单节点测试、部分执行测试
    姊妹篇：10_控制流与函数式API_官方补充.py（Send / Command / 函数式 API）。

为什么这两个维度重要（课案缺口表第 5、6 项，优先级都为「高」）：
    课案把「怎么把图跑起来」讲透了，但没讲「跑不稳怎么办」和「怎么证明它是对的」。
    生产里 agent 卡死、第三方接口抖动、模型偶尔超时都是常态 —— 容错和测试就是
    从 demo 到可用系统的分界线。

官方给出的**错误处理四分类**（fault-tolerance.mdx 的核心表，照抄在这里当索引）：

    # | 错误类型            | 谁来修 | 策略                                    | 本仓库对应
    1 | 瞬时错误（网络/限流）| 系统   | 节点级 RetryPolicy（本文件 Demo 1/2）     | 本文件
    2 | LLM 可修复（工具失败）| 模型  | 错误转 ToolMessage 回灌模型               | 02_langchain/11 官方补充 Demo 1
    3 | 用户可修复（缺信息） | 人工  | interrupt() 问人                          | 课案 01_langgraph 06/07 + deepagents 10
    4 | 意外错误            | 开发者| 让它抛出来，别吞                          | ——

本地版本的三条硬约束（**都是实测出来的，官方文档没有明说**）：
    A. `timeout=` **只支持异步节点**：同步节点上写 timeout 直接抛
       `ValueError: Node timeouts are only supported for async nodes ...`，
       必须 `async def` 节点 + `asyncio.run(graph.ainvoke(...))` 异步入口。
    B. 想让**超时**被重试，`retry_on` 必须写 `NodeTimeoutError`（或干脆用默认 RetryPolicy）；
       写 `(TimeoutError,)` **不会**重试 —— 尽管 NodeTimeoutError 在 Python 里是
       TimeoutError 的子类，实测 retry_on 走的是精确类型匹配。
    C. 默认 RetryPolicy **不重试 ValueError 这类业务异常**（实测 1 次即抛）——
       这是特性不是 bug：参数错误重试一万次也还是错。

运行方式（项目根目录下，全离线、0 次模型调用）：
    uv run Agent/01_langgraph/11_容错与测试_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import asyncio
import time

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, TimeoutPolicy


# ================================================================
# Demo 1：RetryPolicy —— 谁会重试，谁不会
# ================================================================
# 节点级重试挂在 add_node(..., retry_policy=RetryPolicy(...)) 上，字段（实测签名）：
#     max_attempts     最多尝试几次（含首次）
#     initial_interval 首次重试前等多久（秒）
#     backoff_factor   每轮等待时间翻几倍（2.0 = 1s→2s→4s）
#     max_interval     单次等待上限
#     jitter           是否加随机抖动（防止大量节点同时重试打垮下游）
#     retry_on         哪些异常值得重试：异常类/元组，或「接收异常返回 bool」的函数
class SimpleState(TypedDict):
    log: list


# ---- Part A：ValueError —— 默认策略**不**重试 ----
value_error_attempts = {"n": 0}


def raise_value_error(state: SimpleState) -> dict:
    """模拟「参数写错了」这类业务异常：重试没有意义。"""
    value_error_attempts["n"] += 1
    raise ValueError(f"第 {value_error_attempts['n']} 次：参数不合法")


graph_value_error = (
    StateGraph(SimpleState)
    .add_node("bad", raise_value_error, retry_policy=RetryPolicy(max_attempts=3))
    .add_edge(START, "bad")
    .add_edge("bad", END)
    .compile()
)


# ---- Part B：自定义瞬时异常 —— 会按策略重试并最终成功 ----
class TransientError(Exception):
    """模拟「网络抖了一下」：这种才值得重试。"""


transient_attempts = {"n": 0}


def flaky_call(state: SimpleState) -> dict:
    transient_attempts["n"] += 1
    if transient_attempts["n"] < 3:
        raise TransientError(f"第 {transient_attempts['n']} 次：下游服务暂时不可用")
    return {"log": [f"第 {transient_attempts['n']} 次调用成功"]}


graph_retry = (
    StateGraph(SimpleState)
    .add_node(
        "call_api",
        flaky_call,
        # 只重试我们指定的异常；initial_interval 给小值，教学时不必等
        retry_policy=RetryPolicy(
            max_attempts=3,
            retry_on=(TransientError,),
            initial_interval=0.05,
            backoff_factor=2.0,
            jitter=False,
        ),
    )
    .add_edge(START, "call_api")
    .add_edge("call_api", END)
    .compile()
)


# ================================================================
# Demo 2：节点超时 —— 只支持异步节点（实测硬约束）
# ================================================================
# 为什么必须异步：同步 Python 代码没法在中途被安全打断（官方原话大意如此），
# 所以 langgraph 只在异步执行路径上实现节点级超时。
async def slow_async_node(state: SimpleState) -> dict:
    """模拟一个卡住的异步调用（真实场景：模型/接口迟迟不返回）。"""
    await asyncio.sleep(1.5)
    return {"log": ["慢节点居然跑完了（说明没被超时拦住）"]}


# ---- Part A：裸数字超时 ----
graph_timeout = (
    StateGraph(SimpleState)
    .add_node("slow", slow_async_node, timeout=0.4)   # 单位：秒（也接受 timedelta / TimeoutPolicy）
    .add_edge(START, "slow")
    .add_edge("slow", END)
    .compile()
)


# ---- Part B：TimeoutPolicy 精细控制（run_timeout 与 idle_timeout 谁先到算谁）----
graph_timeout_policy = (
    StateGraph(SimpleState)
    .add_node(
        "slow",
        slow_async_node,
        timeout=TimeoutPolicy(run_timeout=1.0, idle_timeout=0.25),
        # run_timeout ：单次尝试的总时长上限
        # idle_timeout：连续多久没有产出就算卡死（本例节点一直在 sleep，所以它先触发）
    )
    .add_edge(START, "slow")
    .add_edge("slow", END)
    .compile()
)


# ---- Part C：超时 + 重试的正确/错误写法对照 ----
timeout_retry_attempts = {"n": 0}


async def slow_always(state: SimpleState) -> dict:
    timeout_retry_attempts["n"] += 1
    await asyncio.sleep(1.0)      # 永远比 timeout 长 → 每次尝试都超时
    return {"log": ["不可能走到这里"]}


def build_timeout_retry_graph(retry_policy: RetryPolicy):
    return (
        StateGraph(SimpleState)
        .add_node("slow", slow_always, timeout=0.2, retry_policy=retry_policy)
        .add_edge(START, "slow")
        .add_edge("slow", END)
        .compile()
    )


# ================================================================
# Demo 3：测试 LangGraph 应用 —— 官方 test.mdx 的三种模式
# ================================================================
# 官方推荐的三种粒度（本文件用 assert 直接演示，放进 pytest 就是现成的 test_*.py）：
#     模式 1 整图测试   ：invoke 一遍，断言最终 state
#     模式 2 单节点测试 ：graph.nodes["节点名"].invoke(...)，绕过图和 checkpointer
#     模式 3 部分执行   ：compile(interrupt_after=[...]) + update_state(as_node=...)
#                         —— 只测图中间某一段的输入输出，不用跑完整流程
class CalcState(TypedDict):
    value: int


def double(state: CalcState) -> dict:
    return {"value": state["value"] * 2}


graph_calc = (
    StateGraph(CalcState)
    .add_node("double", double)
    .add_edge(START, "double")
    .add_edge("double", END)
    .compile()
)


class TwoStepState(TypedDict):
    x: int
    y: int


def step_a(state: TwoStepState) -> dict:
    return {"x": state.get("x", 0) + 1}


def step_b(state: TwoStepState) -> dict:
    return {"y": state.get("x", 0) * 10}


# interrupt_after 是**静态断点**：图跑到 step_a 之后自动停下（课案 06 章讲过静态断点）
graph_two_step = (
    StateGraph(TwoStepState)
    .add_node("step_a", step_a)
    .add_node("step_b", step_b)
    .add_edge(START, "step_a")
    .add_edge("step_a", "step_b")
    .add_edge("step_b", END)
    .compile(checkpointer=MemorySaver(), interrupt_after=["step_a"])
)


if __name__ == "__main__":
    # ---------- Demo 1 ----------
    print("=" * 70)
    print("Demo 1：RetryPolicy —— 谁会重试，谁不会")
    print("=" * 70)

    print("\nPart A：ValueError（业务异常）")
    try:
        graph_value_error.invoke({"log": []})
        print("  没抛异常？（不符合预期）")
    except ValueError as exc:
        print(f"  抛出 ValueError：{exc}")
        print(
            f"  实际尝试了 {value_error_attempts['n']} 次 ← 虽然配了 max_attempts=3，"
            "但默认策略**不重试** ValueError 这类业务异常（重试也没用，早失败早修）"
        )

    print("\nPart B：TransientError（瞬时异常）")
    result = graph_retry.invoke({"log": []})
    print(f"  最终状态：{result['log']}")
    print(
        f"  实际尝试了 {transient_attempts['n']} 次 ← 前两次抛异常被策略吸收，"
        "第三次成功；retry_on 显式指定了要重试的异常类型"
    )

    # ---------- Demo 2 ----------
    print("\n" + "=" * 70)
    print("Demo 2：节点超时 —— 只支持异步节点（本地实测）")
    print("=" * 70)

    print("\nPart A：timeout=0.4（节点实际要 1.5 秒）")
    started = time.time()
    try:
        # 必须走异步入口：同步 invoke 会因为「timeout 只支持异步节点」直接报 ValueError
        asyncio.run(graph_timeout.ainvoke({"log": []}))
        print("  居然没超时？（不符合预期）")
    except NodeTimeoutError as exc:
        print(f"  NodeTimeoutError（{time.time() - started:.2f}s）：{str(exc)[:70]}")
    print("  注意上面用的是异步入口 asyncio.run(graph.ainvoke(...))；")
    print("  同步节点写 timeout 会直接报 ValueError: Node timeouts are only supported for async nodes")

    print("\nPart B：TimeoutPolicy(run_timeout=1.0, idle_timeout=0.25)")
    started = time.time()
    try:
        asyncio.run(graph_timeout_policy.ainvoke({"log": []}))
        print("  居然没超时？（不符合预期）")
    except NodeTimeoutError:
        print(
            f"  NodeTimeoutError（{time.time() - started:.2f}s）← 约 0.25s 就炸了："
            "idle_timeout 先于 run_timeout 触发（节点一直没产出）"
        )

    print("\nPart C：超时能不能被重试？三种 retry_on 写法的实测对照")
    for label, policy in (
        ("retry_on=(TimeoutError,)     ", RetryPolicy(max_attempts=3, retry_on=(TimeoutError,), initial_interval=0.05)),
        ("retry_on=(NodeTimeoutError,) ", RetryPolicy(max_attempts=3, retry_on=(NodeTimeoutError,), initial_interval=0.05)),
        ("默认 RetryPolicy()           ", RetryPolicy(max_attempts=3, initial_interval=0.05)),
    ):
        timeout_retry_attempts["n"] = 0
        try:
            asyncio.run(build_timeout_retry_graph(policy).ainvoke({"log": []}))
            outcome = "成功"
        except NodeTimeoutError:
            outcome = "NodeTimeoutError"
        except Exception as exc:  # noqa: BLE001
            outcome = f"{type(exc).__name__}"
        print(f"  {label} → {outcome}，共尝试 {timeout_retry_attempts['n']} 次")
    print(
        "  ↑ 关键坑：NodeTimeoutError 在 Python 里是 TimeoutError 的子类，\n"
        "    但 retry_on 走**精确类型匹配** —— 写 (TimeoutError,) 一次都不重试；\n"
        "    要重试超时就得写 (NodeTimeoutError,) 或者直接用默认 RetryPolicy()。"
    )

    # ---------- Demo 3 ----------
    print("\n" + "=" * 70)
    print("Demo 3：测试三模式（官方 test.mdx）")
    print("=" * 70)

    print("\n模式 1：整图测试 —— invoke 后断言最终 state")
    whole = graph_calc.invoke({"value": 21})
    assert whole == {"value": 42}, whole
    print(f"  graph_calc.invoke({{'value': 21}}) == {{'value': 42}}  ✔ 断言通过（{whole}）")

    print("\n模式 2：单节点测试 —— graph.nodes['名字'].invoke(...)，绕过图与 checkpointer")
    node_output = graph_calc.nodes["double"].invoke({"value": 21})
    assert node_output == {"value": 42}, node_output
    print(f"  graph_calc.nodes['double'].invoke({{'value': 21}}) == {{'value': 42}}  ✔（{node_output}）")
    print("  ↑ 节点函数是普通函数，能脱离图单独测 —— 复杂图排错时先这样定位到具体节点")

    print("\n模式 3：部分执行 —— 静态断点 + update_state(as_node=...)，只测中间一段")
    config = {"configurable": {"thread_id": "test-two-step"}}
    first = graph_two_step.invoke({"x": 0, "y": 0}, config)
    print(f"  interrupt_after=['step_a'] 第一次 invoke 停在 step_a 之后：{first}")
    # as_node="step_a" 表示「假装这些状态是 step_a 刚产出的」，图会从 step_a 的下游继续
    graph_two_step.update_state(config, {"x": 100}, as_node="step_a")
    second = graph_two_step.invoke(None, config)
    assert second == {"x": 100, "y": 1000}, second
    print(f"  注入 x=100 后继续跑：{second}  ✔ step_b 按注入值算出 y=1000")
    print(
        "  ↑ 这样就**跳过了 step_a** 直接测 step_b —— 上游很慢/很贵时（比如要调模型）\n"
        "    这一招能省掉大部分测试成本。放进 pytest 的写法见文件末尾注释。"
    )

    print("\n全部 Demo 执行完毕（0 次模型调用，离线可复现）。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 实测结论（langgraph 1.2.11，本机）：
#    - Demo 1 Part A：ValueError + RetryPolicy(max_attempts=3) → 实际只尝试 1 次（默认不重试业务异常）；
#      Part B：TransientError + retry_on 指定 → 尝试 3 次后成功。
#    - Demo 2 Part A：异步节点 timeout=0.4 → 0.4s 抛 NodeTimeoutError（错误文案含
#      "exceeded its run timeout of 0.400s"）；Part B：idle_timeout=0.25 先于 run_timeout=1.0 触发；
#      Part C：retry_on=(TimeoutError,) 只尝试 1 次，换成 (NodeTimeoutError,) 或默认策略则尝试 3 次。
#    - Demo 3：graph.nodes['double'] 是 dict，.invoke() 直接返回节点更新字典；
#      interrupt_after + update_state(as_node=...) + invoke(None, config) 三个断言全部通过。
# 2. 未收录（官方还有、本文件没做的）：
#    - durability modes（"exit"/"async"/"sync" 三档持久化粒度）与自定义 checkpointer 契约
#      → checkpointers.mdx，属于进阶话题，见 Agent/官方文档缺口对照.md 表格；
#    - 错误处理器里用 Command 做降级路由、drain 优雅停机 → fault-tolerance.mdx 后半段；
#    - 可观测性（LangSmith trace / Studio 可视化调试）→ 需要外部服务，课案 06_langfuse 章
#      已用 Langfuse 讲了同类思想。
# 3. 踩坑提示：
#    A. timeout 只支持**异步节点**，而且必须走异步入口（ainvoke）；同步节点上写 timeout 直接
#       抛 ValueError: Node timeouts are only supported for async nodes ...。
#    B. 超时的重试匹配是**精确类型**：retry_on=(TimeoutError,) 无效，要用 NodeTimeoutError
#       或默认 RetryPolicy()（官方文档只说"可以重试 TimeoutError 或 NodeTimeoutError"，实测更严）。
#    C. 默认策略排除 ValueError 等业务异常 —— 想连业务异常也重试，必须显式 retry_on 声明。
#    D. 静态断点用 compile(interrupt_after=[...])，恢复用 invoke(None, config)；
#       update_state(..., as_node=...) 的 as_node 必须写**已经跑过的节点名**，语义是
#       "这些状态由它产出"，图才会从它的下游继续。
#    E. 本文件把三种测试模式用 assert 写成可执行脚本；搬进 pytest 时的对应写法：
#           def test_whole_graph():
#               assert graph_calc.invoke({"value": 21}) == {"value": 42}
#           def test_single_node():
#               assert graph_calc.nodes["double"].invoke({"value": 21}) == {"value": 42}
#           def test_middle_segment():
#               cfg = {"configurable": {"thread_id": "t"}}
#               graph_two_step.invoke({"x": 0, "y": 0}, cfg)
#               graph_two_step.update_state(cfg, {"x": 100}, as_node="step_a")
#               assert graph_two_step.invoke(None, cfg) == {"x": 100, "y": 1000}
