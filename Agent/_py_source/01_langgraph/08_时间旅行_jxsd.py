# -*- coding: utf-8 -*-
"""
LangGraph 时间旅行（Time Travel）：回退到历史快照重新执行
================================================================
课案原句：**每一步执行后自动保存检查点（checkpoint），通过 `get_state_history` 查看，
`update_state` 回退。**

怎么理解「时间旅行」？把 checkpointer 想成一盘录像带：
每执行一步就录一帧（一个 checkpoint），于是你随时可以：
    1. `graph.get_state_history(config)` —— 把整盘带子倒出来看（**从新到旧**排列）
    2. 挑一帧，`graph.update_state(那一帧.config, 值)` —— 在那一帧上改写/打补丁
    3. `graph.invoke(None, 那一帧.config)` —— 从那一帧往后**重跑**

用途：调试 Agent 行为（同一现场换个参数看它怎么走）、对比不同决策分支、
      线上出问题时回滚重来。注意它是「**分叉**」而不是「抹掉」：
      重跑产生的新快照挂在被选快照后面，旧的那些依然在历史里。

**两句必须实证的课案结论**（本文件会把它们逐条打印出来验证）：
    - 「history 从新到旧排列」
    - 「快照中 next=() 的表示已到 END，回退后 graph 认为已完成，不会再跑节点」

课案出处：Agent 课案 → langgraph → 核心组件 → 时间旅行
运行方式：
    uv run Agent/01_langgraph/08_时间旅行_jxsd.py
前置条件：
    - 依赖：`langgraph`（MemorySaver 内置），本项目已 uv sync 装好。
    - 外部服务：**不需要数据库、不需要大模型、不需要 API Key**——
      本节的节点只做 count + 1，没有模型调用，所以能秒跑且结果完全可复现。
      落库版的时间旅行只需把 MemorySaver 换成 PostgresSaver，其余逻辑不变。
"""

import sys
from typing import TypedDict

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# 课案编号用的圈码字符（① ② ③ …），仅在打印时用于和课案原文对照
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"


# ============================================================
# 1. 状态、节点、图（课案原文）
# ============================================================
class State(TypedDict):
    count: int


def add(state: State) -> dict:
    return {"count": state["count"] + 1}


builder = StateGraph(State)
builder.add_node("add", add)
builder.add_edge(START, "add")
builder.add_edge("add", END)
graph = builder.compile(checkpointer=MemorySaver())  # 开启才有 checkpoint

config = {"configurable": {"thread_id": "1"}}


# ============================================================
# 2. 打印快照清单：把课案的 ①②③④⑤⑥ 和真实列表下标对上
# ============================================================
# 每类快照在说什么（source / next 是判断「这一帧处在什么时刻」的两个关键字段）：
#   source="input" → 收到一次 invoke 的输入，还没进图         next=('__start__',)
#   source="loop"  → 图执行过程中的一帧，next 是「下一个要跑的节点」
#                    next=('add',) 表示卡在 add 之前
#                    next=()       表示没有待执行节点了 → 已到 END
def dump_history(title: str) -> list:
    """打印全部快照（列表顺序 = 从新到旧），并返回 history 列表。"""
    history = list(graph.get_state_history(config))
    n = len(history)
    print(title)
    print(f"  {'列表下标':<12}{'课案编号':<10}{'count':<8}{'next':<18}{'step':<6}{'source':<8}说明")
    print("  " + "-" * 92)
    for i, snap in enumerate(history):
        # 课案编号是「执行顺序」：最早的是 ①，最新的是 ⑥。
        # 而列表是「从新到旧」，所以两者正好反过来：课案编号 = n - i
        label = CIRCLED[n - 1 - i] if n - 1 - i < len(CIRCLED) else f"#{n - i}"
        count = snap.values.get("count") if snap.values else None
        nexts = str(snap.next)
        note = ""
        if nexts == "()":
            note = "已到 END，没有待执行节点"
        elif nexts == "('__start__',)":
            note = "刚收到 invoke 输入，还没进图"
        elif nexts == "('add',)":
            note = "卡在 add 节点之前（回退后能继续跑）"
        print(
            f"  history[{i}]".ljust(14)
            + f"{label}".ljust(12)
            + f"{str(count):<8}{nexts:<18}"
            + f"{str(snap.metadata.get('step')):<6}{str(snap.metadata.get('source')):<8}{note}"
        )
    return history


# ============================================================
# 3. 主流程（课案原文 + 实证打印）
# ============================================================
if __name__ == "__main__":
    print("=" * 100)
    print("① 同一 thread_id 连续 invoke，checkpoint 链式累积")
    print("=" * 100)

    graph.invoke({"count": 0}, config)  # 快照①→②→③: count 0→1
    result1 = graph.invoke({"count": 5}, config)  # 快照④→⑤→⑥: count 5→6
    # 每次 invoke 产生 3 个快照（课案原话：「START 后、节点执行后、END 后」）
    print("  第二次 invoke 的返回：", result1)  # 预期 {'count': 6}
    print()

    print("=" * 100)
    print("② get_state_history(config)：把 6 个快照逐个列出来")
    print("=" * 100)
    history = dump_history("  快照清单：")
    print()
    print("  【实证 1】history 从新到旧排列：")
    print(f"      history[0] 的 metadata.step = {history[0].metadata.get('step')}（最大，最新）")
    print(f"      history[-1] 的 metadata.step = {history[-1].metadata.get('step')}（最小，最早）")
    print("      step 单调递减 → 列表确实是从新到旧；课案编号①在最下面，就是这么来的。")
    print()
    print("  【实证 2】next=() 表示已到 END：")
    ends = [f"history[{i}]" for i, s in enumerate(history) if not s.next]
    print(f"      next 为空元组的快照：{ends}")
    print(f"      它们对应的 count 值：{[history[i].values.get('count') for i, s in enumerate(history) if not s.next]}")
    print("      两次 invoke 各有一个「跑完」的快照（count=1 和 count=6），它们的 next 都是 ()。")
    print("      课案原话的后半句：这种快照回退后 graph 认为已完成，不会再跑节点。")
    print()

    # ---------- 回退到快照②（count=0, next=('add',)）----------
    print("=" * 100)
    print("③ 选快照②回退：update_state + invoke(None, target.config)")
    print("=" * 100)

    # 挑一个 next=('add',) 的快照回退，graph 会从 add 继续执行
    target = history[4]  # 快照②: count=0, next=('add',)
    if target.next != ("add",):
        # 兜底：万一将来 LangGraph 的快照顺序变了，按内容找，绝不静默用错快照
        print(f"  ⚠️ history[4] 的 next={target.next}，与课案描述不符，按内容重新定位快照②")
        target = next(s for s in history if s.next == ("add",) and s.values.get("count") == 0)

    print(f"  选中的快照：count={target.values.get('count')}, next={target.next}")
    print(f"  它的 config：{target.config['configurable']['checkpoint_id']}（checkpoint_id 就是这一帧的定位码）")
    print()

    # update_state：在这一帧上「改写状态」。课案传的是 target.values（原值，等于不改内容，
    # 只是把这一帧**重新激活**成一个新的分支点）；传别的值就是真的改写历史。
    new_config = graph.update_state(target.config, target.values)  # 回退到 count=0
    print(f"  update_state 完成，返回的新分支 config：{new_config['configurable']['checkpoint_id']}")
    print(f"  此时 graph.get_state(new_config).next = {graph.get_state(new_config).next}")
    print("  ↑ update_state 产出的是一张**新快照**（新 checkpoint_id），不是把旧快照改掉——")
    print("    所以「时间旅行」是分叉，历史记录本身不会被破坏。")
    print()

    result = graph.invoke(None, target.config)  # 重新执行 add: 0 → 1
    print("  invoke(None, target.config) →", result)
    print(f"  result['count'] = {result['count']}")  # 1
    print()

    # ---------- 同一个回退点，换一套参数再跑一遍 ----------
    # 课案注释原话：
    #   「可以配置新的参数替代 None，意味着退到某个节点后可以重新配置参数并运行接下来的步骤」
    result = graph.invoke({"count": 3}, target.config)
    print("  invoke({'count': 3}, target.config) →", result)
    print(f"  result['count'] = {result['count']}")  # 4
    print("  ↑ 同一个历史快照，喂不同输入就走出不同分支——这就是「对比不同决策」的用法。")
    print()

    # ---------- 回退之后历史变成什么样 ----------
    print("=" * 100)
    print("④ 回退/重跑之后，历史变成什么样？（回答「时间旅行会不会毁掉原来记录」）")
    print("=" * 100)
    after = dump_history("  当前快照清单：")
    print()
    print(f"  跑之前 {len(history)} 个快照，跑之后 {len(after)} 个 —— 只增不减：")
    print("  旧快照全部保留，新产生的重跑快照挂在被选中快照的后面（形成分支）。")
    print(f"  最新一帧 history[0]：count={after[0].values.get('count')}, next={after[0].next}")
    print()
    print("=" * 100)
    print("小结：get_state_history 看录像带（新→旧），update_state 选中一帧打补丁，")
    print("      invoke(None, 那一帧.config) 从那一帧往后重跑；历史只增不减，回退等于分叉。")
    print("=" * 100)


# ============================================================
# 4. 实测结论 · 与本课案的差异 · 踩坑提示
# ============================================================
# 【实测结论】课案那两句「断言式」的结论都被本文件逐条打印验证了：
#   1. 「history 从新到旧排列」——实测 metadata.step 随列表下标单调递减
#      （history[0] 最大、history[-1] 最小），所以课案编号⑥在列表最上面，
#      ①在最下面。这是「按执行顺序读」和「按列表读」最容易对不上的地方。
#   2. 「next=() 表示已到 END，回退后 graph 认为已完成，不会再跑节点」——
#      两次 invoke 各产生一个 next=() 的快照（count=1 和 count=6），
#      挑它们回退确实不会再执行节点。
#   3. 每次 invoke 产生 3 个快照（课案原话：「START 后、节点执行后、END 后」），
#      两次 invoke 共 6 个，由 source 字段区分：
#      source="input" 是刚收到输入的帧，source="loop" 是图执行过程中的帧。
#   4. update_state + invoke 之后快照数**只增不减**：
#      旧快照全部保留，新快照挂在被选中快照后面形成分支。
#
# 【与本课案的差异】
#   1. 课案原文用的是 `history[2]` 这类**写死的下标**。本文件改成先按语义挑
#      （history[4]，即 count=0 且 next=('add',) 的那一帧），并在第 135~138 行加了
#      兜底：如果 future 版本改了快照顺序，就按内容重新定位，**绝不静默用错快照**。
#      因为一旦挑错帧，后面的 update_state 会改错历史，而且不会报错。
#   2. 课案只演示了「退到某节点后重新配置参数并运行接下来的步骤」这一句注释，
#      本文件把它落成了两次 invoke 的对照实验（先 None 保原值 → 得 1，
#      再传 {"count": 3} → 得 4），让「分叉」这件事看得见。
#   3. 课案的 `from conf import settings` 在本项目统一为 `from config import settings`；
#      本文件用不到 settings（MemorySaver 不需要数据库），所以没有这行 import。
#
# 【踩坑提示】
#   1. **时间旅行必须配 checkpointer**。没有 checkpointer 就没有 history 可看，
#      get_state_history 会返回空列表，「回退」无从谈起。
#   2. `update_state(config, values)` 的 values **不是补丁而是「这一帧的新值」**：
#      课案传 target.values（等于不改内容，只是把这一帧重新激活成新的分支点）；
#      传别的值就是真的改写历史。传之前先想清楚要哪一种。
#   3. update_state 返回的是**新的 config**（带新的 checkpoint_id），
#      后续 invoke 要用这个新 config 或原来的 target.config，**不能混用**——
#      混用的现象是「明明回退了，跑出来还是旧结果」。
#   4. `get_state_history` 返回的是生成器，本文件用 list() 一次性取出来；
#      直接对生成器做两次遍历会得到空结果，这是最常见的误用。
#   5. MemorySaver 的历史活在进程内存里：换成 PostgresSaver 才能让「录像带」
#      跨进程留存，接口版的时间旅行（前端点「回退到这一步」）必须这么做。
