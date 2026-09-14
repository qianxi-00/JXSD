# -*- coding: utf-8 -*-
r"""
LangChain 智能体：人工审核（Human-in-the-loop）
================================================================
课案原文（本节把课案那 89 行完整保留，只改了配置来源与交互环境的兜底）：

    from copy import deepcopy
    from pathlib import Path
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from config import setting

    @tool
    def write_file(content: str) -> str:
        \"\"\"将内容写入 output.txt。\"\"\"
        Path("output.txt").write_text(content, encoding="utf-8")
        return "已写入 output.txt"

    def review(action_request: dict) -> dict:
        \"\"\"允许多次修改，最后只提交一次审核决定。\"\"\"
        ...（见下方 review 函数，逐字保留）...

    agent = create_agent(
        model=model,
        tools=[write_file],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={...})],
        system_prompt="调用 write_file 前必须等待审核。被拒绝后不要再次调用。",
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "1"}}

    while True:
        result = agent.invoke({"messages": [("user", input("\\n用户: "))]},
                              config=config, version="v2")
        while result.interrupts:
            requests = result.interrupts[0].value["action_requests"]
            decisions = [review(request) for request in requests]
            print(decisions)
            result = agent.invoke(Command(resume={"decisions": decisions}),
                                  config=config, version="v2")
        print("AI:", result.value["messages"][-1].content)

课案原文的说明：`HumanInTheLoopMiddleware` 会在指定工具执行前暂停。
审核需要 `checkpointer` 保存状态，并通过同一个 `thread_id` 恢复。

--------------------------------------------------------------
一次「暂停 → 审批 → 恢复」的完整链路（这是本节的核心）
--------------------------------------------------------------
    1. 模型产出 tool_call（比如 write_file(content="你好世界")）；
    2. HITL 中间件在**工具真正执行之前**调用 LangGraph 的 interrupt()，
       整个图在这一步**冻结**成一份检查点（所以没有 checkpointer 就没法审核）；
    3. `invoke` 立刻返回一个 GraphOutput 对象（`version="v2"` 的产物），
       `result.interrupts` 非空，里面装着 `action_requests`（待审核的动作清单）；
    4. 人类看到清单后做决定，组装成 decisions 列表；
    5. `agent.invoke(Command(resume={"decisions": decisions}), config=...)`
       让图从**同一个 thread_id 的检查点原地复活**，带着决定继续往下跑；
    6. 恢复后如果还有别的工具要审核，会再次 interrupt（所以课案写了 while 循环），
       全部处理完才走到 `result.value["messages"][-1]` 拿到最终回复。

三种决策的区别（课案原文）：

    - `approve`：按原参数执行工具
    - `reject` ：跳过本次工具调用，并将拒绝信息返回给 Agent；不会直接终止整个 Agent
    - `edit`   ：修改工具名或参数后执行工具

    `edit` 会恢复并继续执行，不会再次暂停。示例先在本地完成多次修改，
    最后只提交一次决定。生产环境应使用持久化 checkpointer。

本项目适配说明（都在注释里标了「适配」二字）：
    - 课案的 `from config import setting` → 本项目 `from config import settings`，
      `setting.MODEL_NAME/API_KEY/BASE_URL` → `settings.model_name/api_key/base_url`；
    - 课案的 `version="v2"` 与 `result.interrupts` / `result.value` 全部保留；
    - 课案最后是一个死循环 + `input()`，在非交互环境（CI、重定向输入）会直接
      EOFError 崩掉。按本项目规范加 `sys.stdin.isatty()` 判断：交互时照课案跑，
      非交互时改用「脚本化输入」驱动**同一个 review() 函数**，
      三种决策各演示一次，效果等价且不会卡死。

前置条件：`settings.model_name` 可用（实测可用）。
输出副作用：write_file 会往**当前工作目录**写 output.txt（课案原文行为，未改动）。
    - 建议在项目根目录 `F:\ProGram\Python_Base` 下运行（否则 import 不到 `config`，
      且 output.txt 会落在别的目录里，不好找）；
    - 不需要 PostgreSQL：课案这里用的就是内存版 `InMemorySaver`；
    - 交互环境会停下来等你输入 approve / reject / edit；
      非交互环境（重定向输入 / CI）自动跑三种决策，见文末「本项目适配说明」。

课案出处：Agent 课案 → langChain → 核心组件 → 人工审核

运行方式：
    uv run Agent/02_langchain/09_人工审核_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import builtins
from copy import deepcopy                      # ← 课案原文：深拷贝待审核参数，避免改坏原对象
from pathlib import Path                       # ← 课案原文：写文件的工具需要

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from config import settings

# 课案写的是 ChatOpenAI(model=setting.MODEL_NAME, api_key=setting.API_KEY, base_url=setting.BASE_URL)
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 1. 定义工具（课案原文） ----------
@tool
def write_file(content: str) -> str:
    """将内容写入 output.txt。"""
    # 课案原文：相对路径，落点是「运行时的当前工作目录」
    Path("output.txt").write_text(content, encoding="utf-8")
    return "已写入 output.txt"


# ---------- 2. 人工审核函数（课案原文，逐字保留） ----------
def review(action_request: dict) -> dict:
    """允许多次修改，最后只提交一次审核决定。"""
    action = {
        "name": action_request["name"],
        # ActionRequest 的参数字段名为 "args"（langchain 1.x，旧版为 "arguments"）
        "args": deepcopy(action_request["args"]),
    }
    edited = False

    while True:
        print(f"\n待审核：{action['name']} {action['args']}")
        choice = input("approve / reject / edit 补充内容: ").strip()

        # 三个分支对应课案说的三种决策，注意这里有一个容易忽略的设计：
        # 一旦用过 edit，后面的 approve 就不能再返回 {"type": "approve"}，
        # 否则前面改的参数会被丢掉 —— 所以 approve 分支要判断 edited 并返回 edited_action。
        if choice == "approve":
            return {"type": "edit", "edited_action": action} if edited else {"type": "approve"}
        if choice == "reject":
            # reject 只是「跳过本次工具调用 + 把拒绝信息回传给 Agent」，
            # 不是终止整个 Agent —— 模型收到拒绝后还会继续说话。
            return {"type": "reject"}
        if choice.startswith("edit"):
            # edited_action 的 args 就是 write_file 的入参，extra 直接拼到 content 后面；
            # 拼完 continue 回到循环顶部再问一次，这就是 docstring 里「允许多次修改」的实现。
            extra = choice[4:].strip()
            if extra:
                action["args"]["content"] += extra
                edited = True
            continue
        print("输入无效")


# ---------- 3. 创建智能体（课案原文） ----------
model = llm

agent = create_agent(
    model=model,
    tools=[write_file],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": {
                    "allowed_decisions": ["approve", "reject", "edit"]
                }
            }
        )
    ],
    system_prompt="调用 write_file 前必须等待审核。被拒绝后不要再次调用。",
    checkpointer=InMemorySaver(),
)

# 课案原文的会话配置：审核必须靠 checkpointer + 同一个 thread_id 才能恢复
config = {"configurable": {"thread_id": "1"}}


# ---------- 4. 一轮对话（课案 while True 循环体，抽成函数便于复现三种决策） ----------
def run_turn(user_text: str, turn_config: dict) -> None:
    """课案主循环体的一轮：invoke → 处理中断 → 恢复 → 打印 AI 回复。"""
    result = agent.invoke(
        {"messages": [("user", user_text)]},
        config=turn_config,
        version="v2",
    )

    # 只要有中断就继续处理：一次 invoke 可能挂起多轮（每轮审核一个工具调用）
    if not result.interrupts and not any(
        getattr(m, "tool_calls", None) for m in result.value["messages"]
    ):
        # 本机模型偶发「把工具调用说成一段文本」，此时不会产生中断；
        # 这不是审核机制失效，给一句中文提示，重跑一次即可。
        print("  ⚠ 本轮模型没有真正发起 write_file 工具调用（本机模型偶发行为），故无中断。")
        print("    模型输出：", str(result.value["messages"][-1].content)[:80])

    while result.interrupts:
        requests = result.interrupts[0].value["action_requests"]
        print(f"  [中断] 待审核动作 {len(requests)} 个，工具尚未执行")
        decisions = [review(request) for request in requests]
        print("  提交的决定：", decisions)
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=turn_config,
            version="v2",
        )

    print("  AI:", result.value["messages"][-1].content)


# ---------- 5. 适配：用预设答案替换 input()，让 review() 在非交互环境也能跑 ----------
def scripted_input(answers: list[str]):
    """把 builtins.input 临时换成「按顺序吐预设答案」，用于非交互演示。

    这样做的价值：跑的是**课案原样的 review() 函数**（同一条代码路径），
    而不是另写一段假逻辑，演示与真实交互的效果一致。
    """
    answers = list(answers)

    def fake_input(prompt: str = "") -> str:
        print(f"{prompt}{answers[0]}")     # 回显，让日志看起来像真人输入的
        return answers.pop(0)

    return fake_input


def demo_non_interactive() -> None:
    """非交互环境：三种决策各跑一轮（各自独立 thread_id，互不干扰）。"""
    print("检测到非交互环境（stdin 不是终端），自动演示三种审核决策。\n")
    scenarios = [
        # (说明, 用户输入, review() 里要喂的答案序列, thread_id)
        ("决策 A：approve —— 直接批准执行",
         "把「你好世界」写进 output.txt",
         ["approve"],
         "1-demo-approve"),
        ("决策 B：edit → approve —— 先改内容再批准（课案「允许多次修改」的效果）",
         "把「你好世界」写进 output.txt",
         ["edit （人工补充：请附带日期）", "edit 再补一句", "approve"],
         "1-demo-edit"),
        ("决策 C：reject —— 拒绝执行，拒绝信息回传给 Agent",
         "把「你好世界」写进 output.txt",
         ["reject"],
         "1-demo-reject"),
    ]

    for title, user_text, answers, thread_id in scenarios:
        # 三个场景各自用一个独立 thread_id：审核状态挂在 thread 上，
        # 共用 id 会让上一场景的检查点串进下一场景，三种决策就分不清了。
        print("=" * 60)
        print(title)
        print(f"用户: {user_text}")
        # 替换 builtins.input 而不是重写 review()：这样跑的还是课案原来的那条代码路径。
        original_input = builtins.input
        builtins.input = scripted_input(answers)    # 喂给 review() 的预设答案
        try:
            run_turn(user_text, {"configurable": {"thread_id": thread_id}})
        finally:
            builtins.input = original_input          # 一定要还原，避免影响后续代码

    # 审核通过时，工具真的执行了，磁盘上应当出现 output.txt
    target = Path("output.txt")
    print("\n" + "=" * 60)
    print(f"output.txt 是否生成：{target.exists()}"
          f"{'，内容=' + repr(target.read_text(encoding='utf-8')) if target.exists() else ''}")
    print("（approve / edit 两种情况会写文件；reject 不会。）")


def demo_interactive() -> bool:
    """交互环境：完全按课案原文跑，输入 exit / quit 结束。

    返回值：True = 正常结束；False = 读不到输入（无可用控制台）需走降级路径。
    注意：本机某些执行环境里 `sys.stdin.isatty()` 会返回 True，
    但真正 read 的时候立刻 EOF（比如被 IDE / 任务调度器接管的标准输入），
    所以这里还要再兜一层 EOFError —— 这正是规范第 5.4 条要防的坑。
    """
    print("交互模式：输入内容让 Agent 写文件，中途会停下来等你审核。")
    print("审核时可用：approve / reject / edit 补充内容；主循环输入 exit 退出。")
    while True:
        try:
            user_text = input("\n用户: ")
        except EOFError:
            return False                      # 读不到输入，交给调用方降级
        if user_text.strip().lower() in ("exit", "quit"):
            print("=== 对话结束 ===")
            break
        run_turn(user_text, config)
    return True


if __name__ == "__main__":
    # 课案是死循环 + input()，重定向输入时会 EOFError；
    # 这里按规范做 isatty 判断（外加 EOF 兜底）：终端里走交互，否则走脚本化演示。
    interactive_ok = False
    if sys.stdin.isatty():
        interactive_ok = demo_interactive()
    if not interactive_ok:
        if sys.stdin.isatty():
            print("[提示] 标准输入被判为终端但读不到内容（无可用控制台），改用脚本化演示。\n")
        demo_non_interactive()
