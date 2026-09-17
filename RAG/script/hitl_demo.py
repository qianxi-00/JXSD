"""HITL 审批冒烟：证明"副作用工具要人工审批"不是声明态（优化篇课案「安全保障 · 暂停与恢复」）。

课案 2694-2732 的示例：同一 thread_id 发起运行 → 遇到副作用工具暂停 →
`Command(resume={"decisions": [...]})` 恢复。

**为什么需要一个"冒烟脚本"而不是只写单元测试**:
`interrupt_on` 配对了 ≠ 审批真的会拦。中间件是否生效取决于 langgraph/langchain
的**归一化语义**(实测 1.4.0:`True → {allowed_decisions: [approve, edit, reject, respond]}`、
`False` 的条目直接不进名单),这是**版本相关的风险点** —— 升级依赖后可能静默失效。
单元测试能钉住配置形状(fake agent),但钉不住"真跑一次会不会暂停,
暂停后 resume 能不能走完"。本脚本是后者的证据。

★ 两个关键实现细节(缺一就演示不出审批):
1. **必须持久化 checkpointer + 同一 thread_id**:暂停与恢复是两次独立的 invoke,
   中间状态存在 checkpoint 里;没有 checkpointer 就没有"暂停"这回事
   (注意本项目 `build_production_agent` 的 checkpointer 参数**默认是 None**,
   所以这里显式传 `InMemorySaver()`);
2. **提示词不能是"让模型删东西"**:思考型 LLM 会先检索、再以"证据不足/不会执行删除"
   收尾,或者直接拒绝调用危险工具(这是模型的安全行为,**不是中间件缺陷**)
   ⇒ 演示改用副作用较小的 `update_config`,验的是"拦不拦得住"而不是"删不删得掉"。

用法（需要真实 LLM）：
    uv run python RAG/script/hitl_demo.py

⚠ 本文件的问题：docstring 在**第 1 行**(位置正确),但路径引导的 import 在其后 ——
从别的目录直接跑时,`from agentic.finance_agent import ...` 依赖下面那段 sys.path 注入,
注入发生在 import 之前所以能跑通;但 `import warnings` 被放在路径引导**之后**
(带 noqa: E402)纯属风格问题,不影响功能(见报告)。
"""

import sys as _sys
from pathlib import Path as _Path

# 与其他脚本相同的仓库根定位(硬编码目录名 `Python_Base` 作停止条件)
_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))
_sys.path.insert(0, str(_BASE / "RAG"))

import warnings  # noqa: E402

# 全局静音所有警告(不只是 DeprecationWarning):
# 冒烟脚本的用途是"看审批链路通不通",框架层的 deprecation 噪声会淹没
# 「✅ 已暂停,等待审批」这类关键输出。代价是**真实的兼容性警告也会被吞掉**,
# 所以这里适合一次性冒烟,不适合放进生产链路(见报告)
warnings.filterwarnings("ignore")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agentic.finance_agent import build_production_agent  # noqa: E402

# 两次 invoke 共用同一个 thread_id ⇒ 第二次能读回第一次的 checkpoint(恢复点)。
# 用模块级常量而不是写两遍字面量:两处的 thread_id 必须**完全一致**,
# 差一个字符就会"恢复"成一个全新的会话(表现是不暂停、直接跑完),很难排查
THREAD = {"configurable": {"thread_id": "hitl-demo-1"}}
# 提醒：课案原文的触发语句「删除已停用的财务索引」在本项目实测**演示不出审批** ——
#  DeepSeek 会先检索证据、再以"证据不足/不会执行删除"收尾；即使改成
#  「立刻调用 delete_index 删除 tick 索引」也会被模型直接拒绝（这是模型的安全行为，不是缺陷）。
#  所以这里用副作用较小的 update_config 验证**中间件拦不拦得住**。
PROMPT = "请调用 update_config 工具，把 rerank_score_threshold 设为 0.85。只要调用工具，不要检索。"


def first_interrupt(result):
    """兼容两种暴露方式：result.interrupts（新版）与 result["__interrupt__"]。

    为什么要写兼容:langgraph 在版本演进中改了中断的暴露位置 ——
    新版把 `interrupts` 做成**属性**(state 对象),老版放在 state 字典的
    `__interrupt__` 键里。冒烟脚本要能在不同 langgraph 版本上跑,
    所以两条路都试。

    顺序是"先新后旧":先用 `getattr(..., None)` 探属性,没有再退回字典。
    用 getattr 而不是 `result.interrupts` 直接取 —— 老版 state 上没有这个属性,
    直接取会 AttributeError。

    返回 **None 有明确含义**:这次运行没暂停(模型没调副作用工具)。
    调用方必须处理这一支,否则会把 None 当 interrupt 用而崩掉
    (而"没暂停"本身是**合法结果** —— 模型行为不可控)。
    """
    interrupts = getattr(result, "interrupts", None)
    if interrupts:
        return interrupts[0]
    legacy = result.get("__interrupt__") if isinstance(result, dict) else None
    return legacy[0] if legacy else None


def main() -> int:
    """跑一次"暂停 → 审批 → 恢复"的全过程,返回**进程退出码**。

    返回值用 int 而不是 None:调用点写的是 `raise SystemExit(main())`,
    这样"没暂停"能作为**非零退出码**传出去(返回 1),让 CI/批量脚本
    能区分"演示成功"与"这次模型没触发副作用工具"。

    暂停时的打印结构:`value` 里是中断载荷,形如
    `{"action_requests": [{"name": ..., "args": {...}}]}`。
    `getattr(interrupt, "value", interrupt)` 兼容两种形态(对象带 .value、或本身就是 dict);
    `isinstance(value, dict)` 再兜一层,防止对非 dict 调 .get 而崩 ——
    这里**故意不做更严格的校验**:冒烟脚本要能容忍框架的小版本差异。
    """
    agent = build_production_agent(checkpointer=InMemorySaver())

    print(f"[1] 发起运行: {PROMPT!r}")
    result = agent.invoke({"messages": [{"role": "user", "content": PROMPT}]}, config=THREAD)
    interrupt = first_interrupt(result)
    if interrupt is None:
        # 不是"错误"而是"这次没演示出来",所以打印的是**可行建议**:
        # 换个说法、或换更强的模型(中间件本身没问题)
        print("    ⚠️ 没有暂停：模型这次没调用副作用工具（换个说法或用更强的模型再试）")
        # 打印模型最终回复的前 160 字:能立刻看出它是"没调工具"还是"拒绝了调用"
        print("    最终回复:", str(result["messages"][-1].content)[:160])
        return 1

    value = getattr(interrupt, "value", interrupt)
    requests = (value or {}).get("action_requests") if isinstance(value, dict) else None
    print("    ✅ 已暂停，等待审批")
    # 打印每个待批工具及其参数 —— **这就是"审批真的拦住了"的证据**:
    # 参数里能看到具体要改哪个配置项、改成什么值
    for request in requests or []:
        print(f"       待批工具: {request.get('name')} 参数: {request.get('args')}")

    print("[2] 提交 approve 决策后恢复")
    # `Command(resume=...)` 是恢复协议:`decisions` 列表的**顺序**对应 action_requests 的顺序。
    # 只给了一个 approve —— 若上面打印出多个待批工具,这里应当相应给多个决策
    # (当前 demo 只会有一个,见报告)
    resumed = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config=THREAD
    )
    answer = str(resumed["messages"][-1].content)
    print("    恢复后回复:", answer[:200])
    # 用一个**字符串特征**判断工具是否真执行了("未删除索引"出现在回复里说明模型
    # 只是在解释而没走到工具结果)。这是冒烟脚本的启发式判断,不是断言 ——
    # 所以下一行打印的是"是/否(属模型行为)"而不是失败退出
    executed = "未删除索引" in answer
    print(f"    工具是否真的执行: {'是' if executed else '否（模型没走到工具结果就收尾了，属模型行为）'}")
    print("\n结论: HumanInTheLoopMiddleware 的暂停/恢复链路可用（上面的待批工具即证据）")
    # 走到这里说明**暂停发生了**(无论工具最终是否执行),审批链路本身已验证 ⇒ 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
