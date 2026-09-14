# -*- coding: utf-8 -*-
"""
LangGraph 中断（一）：静态断点 interrupt_before + 人工审核
================================================================
课案原句：**`interrupt_before` 让图在指定节点前暂停，`Command(resume=...)` 恢复执行。**

这句话里其实藏着**两种中断机制**，别混：

  # | 机制                  | 怎么设置                              | 怎么恢复                        |
  # |-----------------------|---------------------------------------|---------------------------------|
  # | 静态断点（本节）      | compile(interrupt_before=["step"])    | graph.invoke(None, config)      |
  # | 动态中断（见 06 精简版 | 节点函数里调 interrupt(payload)       | graph.invoke(Command(resume=...), config) |
  #    / 07_中断_接口版）    |                                       |                                 |
  #
  # | 对比项   | 静态断点 interrupt_before        | 动态中断 interrupt()                 |
  # |----------|----------------------------------|--------------------------------------|
  # | 断在哪   | 编译时写死，节点执行**之前**     | 运行时决定，节点执行到那一行才停     |
  # | 带数据吗 | 不带（只是「停一下」）           | 能把「待审核的问题」抛给人类         |
  # | 恢复方式 | invoke(None, config)             | invoke(Command(resume=值), config)   |
  # | 典型场景 | 固定的人工审核关卡               | 按内容决定要不要问人（如审批金额）   |
  #
  # 另有 interrupt_after=["节点名"]：在节点执行**之后**暂停。三者都需要 checkpointer。

**为什么中断必须配 checkpointer？**
因为「暂停」要实现成「把现场存下来，然后退出执行」。
没有 checkpointer 就没有地方保存这个现场，图停下就真的丢了，也就无从恢复。

**怎么知道它停在哪？**
`snapshot = graph.get_state(config)` 之后看 `snapshot.next`：
    - `next=('step',)` → 下一个要跑的是 step 节点（说明正好卡在它前面）
    - `next=()`        → 已经跑到 END，没有待执行节点了

课案出处：Agent 课案 → langgraph → 核心组件 → 中断（第一段：脚本版）
运行方式：
    uv run Agent/01_langgraph/06_中断_人工审核_jxsd.py
前置条件：
    - 依赖：`langgraph`（MemorySaver 是内置的），本项目已 uv sync 装好。
    - 外部服务：**不需要数据库、不需要大模型、不需要 API Key**——
      本节的图里根本没有模型节点，只演示「暂停 / 恢复」这一套调度机制，
      所以它是 01_langgraph 里跑得最快、最适合反复试验的文件之一。
    - 交互：课案原样要求「按回车继续」。非交互环境（管道 / CI / IDE 运行窗口）
      会被 wait_for_enter() 自动兜住，不会卡死。
"""

import sys
from typing import TypedDict

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


# ============================================================
# 1. 状态与节点（课案原文）
# ============================================================
class State(TypedDict):
    text: str


def step(state: State) -> dict:
    """被中断挡在门外的那个节点。

    打印「step节点开始执行」是课案刻意留的观察点：
    第一次 invoke 时**看不到**它，恢复之后才会出现——一眼就能看出中断生效了。
    """
    print("step节点开始执行")

    return {"text": state["text"] + " → 已执行"}


# ============================================================
# 2. 组装图（课案原文）
# ============================================================
builder = StateGraph(State)

builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_edge("step", END)

# 在 step 节点执行之前设置静态断点
# 注意：断点写在 compile() 里，属于「图的结构特性」，编译后不可更改；
#       运行时想动态决定要不要停，就得用 interrupt()（见 07_中断_接口版_jxsd.py）。
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["step"],
)

# config 里的 thread_id 就是「这次中断现场的编号」。
# 恢复时必须用**同一个 config**，否则 LangGraph 找不到刚才存下的现场。
config = {
    "configurable": {
        "thread_id": "1"
    }
}


# ============================================================
# 3. 主流程（课案原文，加了非交互环境的兜底）
# ============================================================
def wait_for_enter() -> None:
    """课案原文：input("\\n图已经暂停，按回车继续执行step节点……")

    加两层防护是工程上的必要处理：
      1. sys.stdin.isatty()：在 CI / 重定向输入 / 被其他程序调用时，
         stdin 不是终端，按回车无从谈起，直接自动继续；
      2. except EOFError：有些「伪终端」（IDE 运行窗口、任务编排器）里
         isatty() 会返回 True，但真正读的时候仍然立刻 EOF。
         单靠 isatty() 判断会漏掉这种情况，必须把读取动作本身也兜住。

    课案的演示效果（暂停 + 打印提示）完整保留，只是不会再把程序打断。
    """
    prompt = "\n图已经暂停，按回车继续执行step节点……"
    if not sys.stdin.isatty():
        print(prompt)
        print("（检测到当前是非交互环境 stdin 不是终端，自动继续，避免卡死）")
        return
    try:
        input(prompt)
    except EOFError:
        print("\n（stdin 已到结尾，读不到回车，自动继续，避免卡死）")


if __name__ == "__main__":
    print("=" * 74)
    print("① 第一次调用图：会被静态断点挡在 step 节点之前")
    print("=" * 74)
    print("第一次调用图")

    first_result = graph.invoke(
        {"text": "hello"},
        config=config,
    )

    print("  第一次返回结果：", first_result)
    # 预期输出：{'text': 'hello'} —— 输入原样返回，step 节点没跑
    print("  ↑ 注意 text 还是 'hello'，没有 ' → 已执行'，说明 step 确实没执行。")
    print()

    # ---------- 查看暂停位置 ----------
    print("=" * 74)
    print("② 查看暂停位置：get_state(config) 看「现场」")
    print("=" * 74)
    snapshot = graph.get_state(config)

    print("  当前状态：", snapshot.values)
    print("  等待执行的节点：", snapshot.next)
    # 预期输出：('step',) —— 下一个待执行节点是 step
    print("  ↑ next=('step',) 就是「卡在 step 前面」的书面证据；")
    print("    如果已经跑完，next 会是空元组 ()。")
    print()

    # ---------- 手动等待，便于观察中断 ----------
    wait_for_enter()

    # ---------- 恢复执行 ----------
    print("=" * 74)
    print("③ 恢复执行：静态中断用 None 恢复（不是 Command！）")
    print("=" * 74)
    # 「传 None」的语义是：不提供新输入，从上次保存的现场**接着往下跑**。
    # 传 Command(resume=...) 是给**动态中断** interrupt() 传返回值的，用在静态断点上不对。
    second_result = graph.invoke(
        None,
        config=config,
    )

    print("\n  恢复后的结果：", second_result)
    # 预期输出：{'text': 'hello → 已执行'}，且上一行会出现「step节点开始执行」
    print()

    # ---------- 再次查看图状态 ----------
    print("=" * 74)
    print("④ 再次查看图状态：已到 END，next 变成空元组")
    print("=" * 74)
    snapshot = graph.get_state(config)

    print("  最终状态：", snapshot.values)
    print("  后续节点：", snapshot.next)
    # 预期输出：next=() —— 没有待执行节点，图已到达 END
    print()

    # ---------- 补充：同一个断点可以反复用 ----------
    print("=" * 74)
    print("⑤ 补充：换一个 thread_id，同一个断点又是一次全新审核")
    print("=" * 74)
    config2 = {"configurable": {"thread_id": "2"}}
    r = graph.invoke({"text": "第二次 hello"}, config=config2)
    s = graph.get_state(config2)
    print(f"  新会话第一次 invoke 返回：{r}")
    print(f"  它的 next：{s.next}   ← 同样卡在 step 前面")
    print("  ↑ 断点是图的结构属性，对每个 thread_id 都生效；")
    print("    这也说明了为什么 thread_id 是「中断现场」的定位键。")
    print()
    print("=" * 74)
    print("小结：静态断点 = 编译时写死的人工关卡，用 invoke(None, config) 放行。")
    print("      实际项目里这个「按回车」的动作会变成一个 HTTP 接口 → 见 07_中断_接口版_jxsd.py。")
    print("=" * 74)


# ============================================================
# 4. 实测结论 · 与本课案的差异 · 踩坑提示
# ============================================================
# 【实测结论】本机跑一遍，五步的现象与课案描述完全一致：
#   ① 第一次 invoke 返回 {'text': 'hello'}，且**没有**打印「step节点开始执行」
#      —— text 里没有 " → 已执行"，证明 step 真的被挡在门外了。
#   ② snapshot.next = ('step',)，这是「卡在 step 前面」的书面证据。
#   ③ invoke(None, config) 之后才出现「step节点开始执行」，返回
#      {'text': 'hello → 已执行'}。
#   ④ 再次 get_state，next 变成空元组 ()，表示已到 END。
#   ⑤ 换 thread_id="2" 再 invoke，同样卡在 step 前 —— 断点是图的结构属性，
#      对每个 thread_id 都生效。
#
# 【与本课案的差异】
#   1. 课案原样是 `input("...按回车继续...")` 直接等人。本文件包了一层
#      wait_for_enter()：非交互环境（管道输入 / CI / IDE 运行窗口）自动继续，
#      不会把脚本永久挂住。**演示效果保留，只是不再会卡死**。
#   2. 课案只演示了一遍 invoke + resume。本文件补了第 ④ 步（恢复后看 next）
#      和第 ⑤ 步（换 thread_id 再来一次），把「断点作用域」和「next 的两种取值」
#      这两个概念补全。
#   3. 课案的 `from conf import settings` 在本项目统一为 `from config import settings`；
#      本文件根本用不到 settings（没有模型/数据库），所以没有这行 import。
#
# 【踩坑提示】
#   1. **静态断点必须配 checkpointer**。不传 checkpointer 时 compile 不会报错，
#      但「暂停」会退化成「直接停止」——没有地方保存现场，invoke(None) 也无从恢复。
#      记住因果：能恢复不是断点的功劳，是 checkpointer 的功劳。
#   2. 恢复要用 `invoke(None, config)`，**不是** `Command(resume=...)`。
#      Command(resume=) 是给动态中断 interrupt() 传返回值的；
#      用在静态断点上不会报错，但语义不对，很容易写出「看起来跑通了其实没恢复」的代码。
#   3. 恢复必须用**同一个 config**（同一个 thread_id）。换一个 thread_id 就找不到现场，
#      invoke(None) 会拿一个空状态往下跑，行为与预期完全不同且不报错——
#      这是本节最容易踩的坑。
#   4. `interrupt_before` 的取值是**节点名列表**，写错名字不会报错，
#      只会「静默不生效」。症状是「怎么没停？」——先核对节点名拼写。
#   5. `interrupt_after=["节点名"]` 是在节点**执行之后**暂停，别与 before 混用记反。
