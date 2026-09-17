# -*- coding: utf-8 -*-
"""
DeepAgents 人工审核（Human-in-the-Loop，interrupt_on）
================================================================
让 Agent 在调用**危险工具**之前先停下来，等人点头。
课案给的官方文档地址：
    https://docs.langchain.com/oss/python/langchain/human-in-the-loop

本节要讲什么
    1. 它**解决什么问题**：Agent 拿到 write_file / execute 这类危险工具之后，
       谁来兜底？答案是把「执行前必须有人点头」做成图的执行流程的一部分，
       而不是靠提示词劝它别乱来；
    2. 中断（interrupt）**怎么发生、又怎么恢复**：`__interrupt__` 字段里装什么、
       `Command(resume=...)` 怎么把决策送回被冻结的图；
    3. 两个硬性前提：必须配 checkpointer、恢复必须用同一个 thread_id；
    4. 决策的三种形态与 `interrupt_on` 的简写/完整写法。

一、机制：靠 LangGraph 的 interrupt 实现
    1. `create_deep_agent(..., interrupt_on={"write_file": True})`
       —— 告诉 deepagents：调用 write_file 之前挂起。
       框架会在中间件栈尾部挂上 `HumanInTheLoopMiddleware(interrupt_on=...)`。
    2. 挂起时，`agent.invoke(...)` 的返回值里会多出一个 `__interrupt__` 字段，
       内容是**待审批的请求**：
           {"action_requests": [{"name": 工具名, "args": 参数字典, "description": 说明}, ...],
            "review_configs": [...]}
    3. 人工给出决策列表（与 action_requests **等长**，一项对一个工具调用）：
           {"type": "approve"}                        批准
           {"type": "edit", "args": {...}}            改参数后执行
           {"type": "reject", "message": "拒绝理由"}   拒绝（理由会回传给模型）
    4. 用 `Command(resume={"decisions": [...]})` 把决策喂回去，图从中断处继续跑。

二、两个必须记住的前提
    - **必须配 checkpointer**：中断意味着「图的执行状态被冻结」，这份状态要有地方存。
      不传 checkpointer 会直接报错。
    - 中断后要用**同一个 config**（同一个 thread_id）恢复，否则接不上原来那条执行线。

三、`interrupt_on` 的两种写法
        interrupt_on={"write_file": True}
            —— 简写，等价于允许全部三种决策（approve / edit / reject）。
        interrupt_on={
            "send_email": {
                "allowed_decisions": ["approve", "reject"],   # 只允许批准或拒绝，不给改参数
                "description": "发送邮件前必须经过用户确认",
            },
        }
            —— 完整写法，可以逐工具限制决策类型、补充说明文字。

四、课案原文（本文件实现的就是它）
    backend 用 FilesystemBackend（"文件写到真实磁盘当前目录"），
    interrupt_on 盯住 write_file，两个函数分工：
        review_interrupts(hitl_request) —— 展示待审批调用，读人工决定，返回 decisions
        main()                          —— 跑一轮；没触发中断就直接输出，触发了就走审批

五、本文件相对课案的两处工程化改动
    1. `root_dir` 从课案的 `"."` 改成脚本同级的
       `Agent/03_deepagents/tmp_jxsd_deepagents_hitl/`，避免把 hello.txt 写到仓库根目录。
    2. 课案的 `input()` 在非交互终端（管道/重定向）会 EOFError。
       按本项目规范加 `sys.stdin.isatty()` 判断：**交互时行为与课案完全一致**；
       非交互时打印提示并自动批准，保证脚本能无人值守跑完。

课案出处：Agent 课案 → deepAgents → 人工审核

运行方式：
    uv run Agent/03_deepagents/10_人工审核_jxsd.py

前置条件：.env 里的 api_key / base_url / model_name 可调通（实测可用）。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend   # 改用真实磁盘
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 文件写到这个真实目录（课案是 "."）
WORKDIR = Path(__file__).resolve().parent / "tmp_jxsd_deepagents_hitl"

agent = create_deep_agent(
    model=llm,
    backend=FilesystemBackend(root_dir=str(WORKDIR), virtual_mode=True),  # 文件写到真实磁盘
    # 盯住 write_file：只要模型要写文件就挂起等审批。
    # True 是简写，等价于 allowed_decisions=["approve","edit","reject"]。
    interrupt_on={"write_file": True},
    # 中断状态要有地方存 —— 不配 checkpointer 直接报错
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}


def review_interrupts(hitl_request: dict) -> list[dict]:
    """展示待审批的工具调用，读取人工决定并返回 decisions 列表。

    :param hitl_request: HumanInTheLoopMiddleware 中断携带的审批请求，
        结构为 {"action_requests": [...], "review_configs": [...]}。
    :return: 与 action_requests 等长的决策列表，每项为 approve 或 reject。
    """
    action_requests = hitl_request["action_requests"]
    for action in action_requests:
        print(f"待审批工具: {action['name']}")
        print(f"工具参数: {action['args']}")
        if action.get("description"):
            print(f"说明: {action['description']}")

    # ---------- 本文件新增：非交互终端自动批准 ----------
    # 课案这里是裸的 input()。在管道/重定向/CI 里 input() 会抛 EOFError 把脚本打断，
    # 所以加一层判断：不是交互终端就自动批准，并把「本该问什么」打印出来。
    if not sys.stdin.isatty():
        print("（非交互终端：跳过 input()，自动批准本次操作）")
        return [{"type": "approve"}] * len(action_requests)

    try:
        choice = input("是否批准执行上述操作？[y/n]: ").strip().lower()
    except EOFError:
        # 有些环境 isatty() 为真但 stdin 已关闭，这里兜一下，避免脚本崩掉
        print("（stdin 已关闭，自动批准本次操作）")
        return [{"type": "approve"}] * len(action_requests)

    # 批准 → approve；其他任何输入（含直接回车）都按拒绝处理，并且把理由回传给模型
    if choice == "y":
        decision = {"type": "approve"}
    else:
        try:
            reason = input("拒绝原因（可选，直接回车跳过）: ").strip()
        except EOFError:
            # 同一层兜底：拿不到理由就用空串，后面会给一个默认文案
            reason = ""
        # message 会作为工具反馈交回模型，所以即便用户没写理由也要给一句人话
        decision = {"type": "reject", "message": reason or "人工审核拒绝"}
    # 一次中断可能攒了多个待审批调用，用同一个决定铺满，长度必须与 action_requests 对齐
    return [decision] * len(action_requests)


def main() -> None:
    """运行代理；若触发审批中断则等待人工输入，否则直接输出结果。"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "写一个hello.txt"}]}, config
    )

    interrupts = result.get("__interrupt__")
    if not interrupts:
        # 模型未调用 write_file，无需审批
        print("未触发审批，最终结果：", result["messages"][-1].content)
        return

    # deepagents 0.7.x 中 state.next 为 ('HumanInTheLoopMiddleware.after_model',)
    # get_state 能看到「图停在哪一步」，是排查中断问题最直接的手段。
    state = agent.get_state(config)
    print(f"审批中断点: {state.next}")

    # interrupts[0].value 就是上面 review_interrupts 的入参；
    # 一次可能攒了多个待审批调用，所以 decisions 要按数量对齐。
    decisions = review_interrupts(interrupts[0].value)
    print(f"人工决策: {decisions}")

    # Command(resume=...) 把决策送回被冻结的图，从断点继续执行
    result = agent.invoke(Command(resume={"decisions": decisions}), config)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    WORKDIR.mkdir(parents=True, exist_ok=True)
    print(f"工作目录：{WORKDIR}\n")

    main()

    # ---------- 事后核对：审批通过后文件才真的落盘 ----------
    print("\n===== 磁盘结果 =====")
    files = sorted(p.name for p in WORKDIR.rglob("*") if p.is_file())
    print(f"  {WORKDIR.name}/ 下的文件：{files if files else '（空）'}")
    print(
        "\n要点回顾：\n"
        "  - 中断发生在工具真正执行**之前**，所以被拒绝时磁盘上不会留下任何痕迹；\n"
        "  - decisions 必须和 action_requests 等长，一次中断可能攒了多个调用；\n"
        "  - 拒绝时带上 message，模型会收到这条反馈并自行调整方案。"
    )
