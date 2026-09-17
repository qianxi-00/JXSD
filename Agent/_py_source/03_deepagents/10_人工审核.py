# -*- coding: utf-8 -*-
"""
DeepAgents 人工审核（interrupt_on）
================================================================
给工具配置 interrupt_on，调用前挂起等待人类批准：

    interrupt_on={
        "工具名": {"allowed_decisions": ["approve", "edit", "reject"]}
    }

    approve：批准        edit：改参数后执行
    reject：拒绝（模型收到反馈后自行调整）

必须配 checkpointer（中断状态要持久化）。

运行方式：
    uv run 03_deepagents/10_人工审核.py
"""

import sys

from deepagents.backends import FilesystemBackend

sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


@tool
def send_email(to: str, subject: str) -> str:
    """发送邮件（危险操作，需要人工批准）"""
    return f"已发送邮件给 {to}，主题：{subject}"


agent = create_deep_agent(
    model=llm,
    # tools=[send_email],
    # system_prompt="你是邮件助手，帮用户发送邮件。",
    checkpointer=MemorySaver(),
    interrupt_on={"write_file": True},
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    # interrupt_on={
    #     "send_email": {
    #         "allowed_decisions": ["approve", "reject"],
    #         "description": "发送邮件前必须经过用户确认",
    #     },
    # },
)

# if __name__ == "__main__":
#     config = {"configurable": {"thread_id": "email-1"}}
#
#     # 第一次执行：模型要发邮件 → 挂起
#     result = agent.invoke(
#         {"messages": [("user", "给 boss@company.com 发一封邮件，主题：请假")]},
#         config,
#     )
#     print("挂起等待审核：", result.get("__interrupt__"))
#
#     # 批准（演示）。拒绝的话：
#     # Command(resume={"decisions": [{"type": "reject", "message": "拒绝理由"}]})
#     result = agent.invoke(
#         Command(resume={"decisions": [{"type": "approve"}]}),
#         config,
#     )
#     print("AI：", result["messages"][-1].content)

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

    # 非交互环境（stdin 被重定向 / 后台运行）下 input() 会直接抛 EOFError。
    # 审批是「有副作用的关卡」，读不到人就不放行 —— 默认拒绝并打印中文提示，
    # 而不是让整个脚本 traceback。要手工审批请在真实终端里运行本文件。
    try:
        choice = input("是否批准执行上述操作？[y/n]: ").strip().lower()
    except EOFError:
        print("\n[非交互环境] 读不到控制台输入，无人审批 → 默认拒绝执行。")
        print("             想手工审批请在真实终端里运行本文件。")
        return [{"type": "reject", "message": "非交互环境，未获得人工批准"}] * len(action_requests)

    if choice == "y":
        decision = {"type": "approve"}
    else:
        try:
            reason = input("拒绝原因（可选，直接回车跳过）: ").strip()
        except EOFError:
            reason = ""
        decision = {"type": "reject", "message": reason or "人工审核拒绝"}
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
    state = agent.get_state(config)
    print(f"审批中断点: {state.next}")

    decisions = review_interrupts(interrupts[0].value)
    result = agent.invoke(Command(resume={"decisions": decisions}), config)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
