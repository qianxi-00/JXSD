# -*- coding: utf-8 -*-
"""
LangGraph 子图（Subgraph）：把一个编译好的图当成另一个图的节点
================================================================
课案原句：**将一个已编译好的图作为另一个图的节点，父图和子图各自独立编译。**

为什么要把图拆开？因为复杂 Agent 的图会长到没法看。拆成子图之后：
  - 每个子图只负责一件事（规划 / 检索 / 执行），可以**单独编译、单独运行、单独测试**
  - 父图只看「大流程」，不用被几十个内部节点淹没
  - 子图可以复用：同一份「检索子图」挂到多个父图上，不用复制粘贴

**三个必须记住的点**：
  1. 子图先 `compile()`，再像普通节点一样 `add_node("名字", 子图对象)`——
     区别只是「节点函数」换成了「一张已经编译好的图」。
  2. **父图与子图共享同名 state 键**时，数据是自动透传的：
     父图的 state 原样喂给子图，子图返回的增量再合并回父图 state。
     所以课案里父图只声明了 `text: str`，子图写完 `text`，父图的 `suffix` 节点立刻就能读到。
  3. 子图只看得见**自己 schema 里声明过的键**（后面第 3 节会实测这一点）——
     父图独有的字段在子图内部是「不存在」的，别指望在子图里随手读写。

**本节要讲什么**（四件事，正文按顺序对应 4.1~4.4）：
  ① 课案原文的最小示例：子图当节点用，同名 state 键自动透传；
  ② 子图能独立 compile → 也就能独立 invoke，这是「单独测试」的工程价值；
  ③ 父子 schema 不同名时的实测行为：子图看不见父图独有的键；
  ④ `stream(subgraphs=True)`：命名空间元组怎么标出「哪一层图的哪个节点」。

课案出处：Agent 课案 → langgraph → 核心组件 → 子图
运行方式：
    uv run Agent/01_langgraph/09_子图_jxsd.py
前置条件：
    - 依赖：`langgraph`，本项目已 uv sync 装好。
    - 外部服务：**不需要数据库、不需要大模型、不需要 API Key**——
      本节的节点只做字符串拼接，没有模型调用，秒跑且结果完全可复现。
"""

import sys
from typing import TypedDict

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from langgraph.graph import END, START, StateGraph


# ============================================================
# 1. 课案原文：父图与子图共享同一个 State
# ============================================================
class State(TypedDict):
    text: str


def prefix_node(state: State) -> dict:
    """子图里的节点：给文本加个前缀。

    课案原文这一步写的是 lambda：
        child.add_node("prefix", lambda s: {"text": "[子图]" + s["text"]})
    这里改成具名函数，只为能在函数里加打印观察点看 state 怎么流动，逻辑完全一致。
    """
    print(f"      [子图 prefix 节点] 收到的 state：{state}")
    return {"text": "[子图]" + state["text"]}


# 子图：加前缀
child = StateGraph(State)
child.add_node("prefix", prefix_node)
child.add_edge(START, "prefix")
child.add_edge("prefix", END)
child_graph = child.compile()  # 子图先独立编译


def suffix_node(state: State) -> dict:
    """父图自己的节点：注意它读到的是子图刚写进去的 text。

    课案原文同样是 lambda：
        parent.add_node("suffix", lambda s: {"text": s["text"] + " → 父图"})
    """
    print(f"      [父图 suffix 节点] 收到的 state：{state}")
    return {"text": state["text"] + " → 父图"}


# 父图：调用子图 + 自己的节点
parent = StateGraph(State)
parent.add_node("child", child_graph)  # 子图作为普通节点使用
parent.add_node("suffix", suffix_node)
parent.add_edge(START, "child")  # 先走子图
parent.add_edge("child", "suffix")  # 再走父图节点
parent.add_edge("suffix", END)
parent_graph = parent.compile()


# ============================================================
# 2. 补充一：子图「独立编译」意味着它也能独立运行
# ============================================================
# 这是子图最重要的工程价值：**单独测试**。
# 调试子图时不需要把整张父图跑起来，直接 invoke 子图即可，定位问题快得多。
# （顺带一提：子图自己也可以挂 checkpointer / 设 interrupt，粒度比父图细。）


# ============================================================
# 3. 补充二：父图与子图共享同名 state 键 —— 以及「不共享」时会怎样
# ============================================================
class ChildOnlyText(TypedDict):
    """子图 schema：只声明 text。"""

    text: str


class ParentWithNote(TypedDict):
    """父图 schema：比子图多一个 note。"""

    text: str
    note: str


def note_child_node(state: ChildOnlyText) -> dict:
    """故意在子图里打印「它到底能看见哪些键」。"""
    print(f"      [子图] 收到的键：{sorted(state.keys())}")
    return {"text": "[子图]" + state["text"]}


def note_after_node(state: ParentWithNote) -> dict:
    print(f"      [父图 after] 收到：{state}")
    return {"note": state["note"] + "（父图改过）"}


nc = StateGraph(ChildOnlyText)
nc.add_node("prefix", note_child_node)
nc.add_edge(START, "prefix")
nc.add_edge("prefix", END)
note_child_graph = nc.compile()

np_builder = StateGraph(ParentWithNote)
np_builder.add_node("child", note_child_graph)
np_builder.add_node("after", note_after_node)
np_builder.add_edge(START, "child")
np_builder.add_edge("child", "after")
np_builder.add_edge("after", END)
note_parent_graph = np_builder.compile()


# ============================================================
# 4. 主流程
# ============================================================
if __name__ == "__main__":
    # ---------- 4.1 课案原文示例 ----------
    print("=" * 74)
    print("① 课案原文：父图 invoke({'text': 'hello'})")
    print("=" * 74)
    print("  执行过程：")
    result = parent_graph.invoke({"text": "hello"})
    print(f"\n  完整返回：{result}")
    print(f"  result['text'] = {result['text']}")  # 预期 [子图]hello → 父图
    print("  ↑ 父图只声明了 text 一个字段，子图写进去的值直接就被父图节点读到了：")
    print("    这就是「共享同名 state 键」的效果——不需要任何胶水代码。")
    print()

    # ---------- 4.2 子图独立运行 ----------
    print("=" * 74)
    print("② 补充：子图能独立运行，所以能独立测试")
    print("=" * 74)
    child_result = child_graph.invoke({"text": "单独测试"})
    print(f"  child_graph.invoke({{'text': '单独测试'}}) → {child_result}")
    print("  ↑ 调试子图时不用把整张父图跑起来；同一个子图也能挂到别的父图上复用。")
    print()

    # ---------- 4.3 子图只看得到自己 schema 里的键 ----------
    print("=" * 74)
    print("③ 补充：父子 schema 不同名时会怎样（实测）")
    print("=" * 74)
    print("  父图 schema：text + note；子图 schema：只有 text")
    print("  执行过程：")
    res = note_parent_graph.invoke({"text": "hello", "note": "初始 note"})
    print(f"\n  完整返回：{res}")
    print("  ↑ 观察两点：")
    print("    1. 子图里打印出来的键**只有 ['text']** —— 父图的 note 在子图内部根本看不见，")
    print("       因为子图只按自己的 schema 取键。别在子图里读写父图独有的字段。")
    print("    2. 子图改的 text 照常合并回父图，父图独有的 note 原封不动地保留下来，")
    print("       最后由父图自己的 after 节点修改。")
    print()

    # ---------- 4.4 用 stream(subgraphs=True) 看子图内部 ----------
    print("=" * 74)
    print("④ 补充：stream(subgraphs=True) —— 连子图内部的执行都能看见")
    print("=" * 74)
    print("  每个事件的第一个元素是**命名空间元组**：() 表示父图，('child:xxx',) 表示进了子图。")
    for namespace, chunk in parent_graph.stream({"text": "hello"}, subgraphs=True):
        where = "父图" if not namespace else f"子图 {namespace[0].split(':')[0]}"
        print(f"      [{where}] {chunk}")
    print()
    print("  ↑ 节点名相同也不会混：命名空间把「哪一层图的哪个节点」标得清清楚楚。")
    print()
    print("=" * 74)
    print("小结：子图 = 先 compile、再当节点用；同名键自动透传，异名键互不可见；")
    print("      拆分后每块都能独立跑、独立测，这是复杂 Agent 唯一可控的组织方式。")
    print("=" * 74)


# ============================================================
# 5. 实测结论 · 与本课案的差异 · 踩坑提示
# ============================================================
# 【实测结论】四节的输出与课案描述一致：
#   ① parent_graph.invoke({"text": "hello"}) → text = "[子图]hello → 父图"。
#      子图节点打印的 state 就是父图的 state，**没有任何胶水代码**——
#      这是「共享同名 state 键」最直接的证据。
#   ② child_graph.invoke({"text": "单独测试"}) → "[子图]单独测试"，
#      父图完全不参与，证明子图确实是一个独立可运行的图。
#   ③ 父图 schema 是 text + note，子图 schema 只有 text 时，
#      子图内部打印出来的键**只有 ['text']**（note 在子图里根本不存在）；
#      但子图改的 text 照常合并回父图，父图独有的 note 原封不动保留到最后，
#      由父图自己的 after 节点修改。两条结论一次跑全验证了。
#   ④ stream(subgraphs=True) 每个事件的第一个元素是命名空间元组：
#      () 表示父图，('child:<uuid>',) 表示进了 child 子图内部。
#
# 【与本课案的差异】
#   1. 课案子图只给了一个最小示例（第 1 节）；第 2~4 节（独立运行 / 异名键实测 /
#      subgraphs 命名空间）都是本文件补的——课案原句「父图和子图各自独立编译」
#      只说了结论，没演示「独立编译带来什么好处」，第 2 节补的就是这个「所以呢」。
#   2. 课案原文的 prefix / suffix 节点是 lambda；本文件改成具名函数，
#      只为能在函数里加打印观察点看 state 怎么流动，**逻辑完全一致**。
#   3. 课案的 `from conf import settings` 在本项目统一为 `from config import settings`；
#      本文件根本用不到 settings（没有模型/数据库），所以没有这行 import。
#
# 【踩坑提示】
#   1. 子图必须**先 compile** 再 add_node。传一个没编译的 StateGraph 进去会直接报错。
#   2. 父子 schema 同名键的字段类型 / reducer 必须能对上：
#      父图字段是 Annotated[list, add] 而子图字段是普通 list 时，
#      合并行为按**父图**的 reducer 走，子图里看不出来，容易在深层子图上踩到。
#   3. 子图看不见父图独有的键 → 想在子图里读写父图字段，只有两条路：
#      把字段加进子图 schema（共享同名键），或显式通过参数传进去。
#   4. 用 `graph.get_graph(xray=True)` 才能把子图内部节点展开画出来；
#      默认 `get_graph()` 里子图只是一个方框，排查问题时要记得加 xray。
#   5. 命名空间里的 uuid 每次运行都不同，**不要把它写进断言或快照对比**，
#      要比较就取 `namespace[0].split(':')[0]`（本文件第 ④ 节就是这么做的）。
