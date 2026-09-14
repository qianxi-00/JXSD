# -*- coding: utf-8 -*-
r"""
LangChain 智能体：结构化输出（response_format）
================================================================
课案原文（注意第一行那句注释，它是本节最容易踩的坑）：

    # 需要使用没有思考模式的模型，比如 MODEL_NAME="deepseek-chat"

    from pydantic import BaseModel, Field
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from conf import settings

    class ContactInfo(BaseModel):
        \"\"\"个人联系信息\"\"\"
        name: str = Field(description="姓名")
        email: str = Field(description="邮箱")
        phone: str = Field(description="电话")

    model = ChatOpenAI(model=settings.model_name, api_key=settings.api_key, base_url=settings.base_url)
    agent = create_agent(model=model, response_format=ContactInfo)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "李四，lisi@example.com，13900139000"}]
    })
    print(result["structured_response"])
    # name='李四' email='lisi@example.com' phone='13900139000'

**为什么必须用「没有思考模式」的模型？**（课案那句注释的原因）

    结构化输出的底层是 Function Calling：框架把 Pydantic 模型转成一份 JSON Schema，
    再通过 `tool_choice`（强制调用某个函数）让模型「只能」按这份 schema 输出参数。
    而思考模式（reasoning / thinking）的模型在给出最终答案前会先产出一段思维链，
    部分供应商在这种情况下**不允许强制 tool_choice**，请求会直接报错
    （典型报错：`tool_choice` 与 thinking 不兼容 / 返回空的结构化结果）。
    课案因此指定 `deepseek-chat` 这类非思考模型。

本项目 `settings.model_name` 就是当前可用的对话模型（实测能正常返回结构化结果），
所以下面直接用 settings 里的模型；如果换成带思考模式的模型而报错，
把注释里那句「需要使用没有思考模式的模型」当成排查第一站即可。

与「裸模型 + with_structured_output」的分工（本项目 08_结构化输出.py 讲的另一种写法）：

    # | 写法                                   | 谁来做结构化   | 适合场景                       |
    # |---------------------------------------|--------------|-------------------------------|
    # | create_agent(..., response_format=模型) | 智能体（可先调工具再输出） | 需要「先查资料/调工具，再产出结构化结果」 |
    # | llm.with_structured_output(模型)         | 裸模型        | 纯信息抽取、分类，不需要工具     |

两种写法的结果都在「结构化对象」里：智能体放 `result["structured_response"]`，
裸模型直接把对象返回给你。

课案出处：Agent 课案 → langChain → 核心组件 → 结构化输出

前置条件与运行方式：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好；
    - 关键前提是**模型不能带思考模式**（课案原话），否则第 4~7 步会命中
      `tool_choice` 不兼容的报错 —— 本文件已用 try/except 兜成中文提示，不会崩；
    - 需要 `pydantic`（本项目 venv 已装）；不需要数据库；
    - 在项目根目录下执行（否则 import 不到 `config`）：
    uv run Agent/02_langchain/08_结构化输出_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 1. 课案原文的输出模型 ----------
class ContactInfo(BaseModel):
    """个人联系信息"""
    # Field(description=...) 会原样进入 JSON Schema 的 description，
    # 模型就是靠这几句话知道「哪个字段该填什么」——所以描述要写清楚。
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")


# ---------- 2. 更复杂一点：嵌套列表 + 枚举 ----------
class Ticket(BaseModel):
    """一张待办工单"""
    title: str = Field(description="工单标题，不超过 10 个字")
    priority: str = Field(description="优先级，只能是 高/中/低 三者之一")
    tags: list[str] = Field(description="标签，如 ['后端', '线上']")


class TicketList(BaseModel):
    """工单列表（演示嵌套结构）"""
    items: list[Ticket] = Field(description="拆解出来的工单列表")


# ---------- 3. 带工具的智能体 + 结构化输出：先查再答 ----------
@tool
def lookup_user(user_id: str) -> str:
    """按用户编号查询用户资料。user_id：用户编号，如 u1"""
    # 演示用：真实项目里这里会查数据库
    return "u1：李四，lisi@example.com，13900139000"


def run_case(tag: str, agent, question: str) -> None:
    """统一的调用 + 取结果 + 异常兜底，保证脚本不会因为模型不支持而崩掉。"""
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"===== {tag} =====")
        print("structured_response：", result["structured_response"])
        print("类型：", type(result["structured_response"]).__name__)
        # 结构化结果的本质：可以直接按属性取值，不再需要正则去抠字符串
        print("模型 dump 出来的字典：", result["structured_response"].model_dump())
    except Exception as exc:
        # 最常见的失败原因就是课案注释里说的「模型带思考模式，不支持强制 tool_choice」
        print(f"[跳过] {tag} 结构化输出失败：{type(exc).__name__}")
        print("       若报错与 tool_choice / thinking 有关，请把 settings.model_name")
        print("       换成没有思考模式的对话模型（课案用的是 deepseek-chat）。")
        print("       详情：", str(exc)[:200])


if __name__ == "__main__":
    # ---------- 4. 课案原文：最简结构化输出 ----------
    # response_format 传 Pydantic 类，智能体就会把最终答案「塞」成这个类的实例，
    # 存放在 result["structured_response"] 里（messages 里仍有一次普通回复）。
    contact_agent = create_agent(model=llm, response_format=ContactInfo)
    run_case("4. 课案原文：抽取联系人信息", contact_agent, "李四，lisi@example.com，13900139000")

    # ---------- 5. 课案原样的嵌套结构 ----------
    print()
    ticket_agent = create_agent(model=llm, response_format=TicketList)
    run_case(
        "5. 嵌套结构：把需求拆成工单",
        ticket_agent,
        "登录页报 500 要马上修；顺便把文档补一下，不急。",
    )

    # ---------- 6. 智能体 + 工具 + 结构化输出 ----------
    # 这是 create_agent 版比 with_structured_output 强的地方：
    # 模型可以先调 lookup_user 拿资料，再把结果整理成 ContactInfo。
    print()
    tool_agent = create_agent(model=llm, tools=[lookup_user], response_format=ContactInfo)
    run_case("6. 先查工具再结构化", tool_agent, "帮我查一下 u1 这个用户的联系方式")

    # ---------- 7. 对照：裸模型的 with_structured_output ----------
    # 不经过智能体，少一次图调度，速度更快；但没法「先调工具再输出」。
    # 两种写法用的是同一套底层机制（都是把 Pydantic 转 JSON Schema + 强制 tool_choice），
    # 所以第 6 步报错时这一步大概率也报错 —— 排查方向是同一个。
    print("\n===== 7. 对照写法：llm.with_structured_output =====")
    try:
        reviewer = llm.with_structured_output(ContactInfo)
        info = reviewer.invoke("李四，lisi@example.com，13900139000")
        print("直接拿到对象：", info)
        print("直接取属性：name =", info.name)   # 不再需要 json.loads / 正则
    except Exception as exc:
        print("[跳过] with_structured_output 失败：", type(exc).__name__, str(exc)[:200])
