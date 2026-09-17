# -*- coding: utf-8 -*-
r"""
LangChain 核心组件（二）：消息（Message）
================================================================
课案原文（注意 import 路径是**新版**的 `langchain.messages`，
旧教程里写的是 `langchain_core.messages`，两者是同一批类，新版路径更短）：

    from langchain.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    from config import setting

    model = ChatOpenAI(model=setting.MODEL_NAME, api_key=setting.API_KEY, base_url=setting.BASE_URL)
    messages = [
        SystemMessage("你是一位唐诗专家"),
        HumanMessage("写一首关于春天的七言绝句"),
    ]
    response = model.invoke(messages)
    print(response.content)

模型这一层没有「记忆」：它每次只看到你这次传进去的**整个消息列表**。
所以「对话历史」= 一个不断变长的 List[BaseMessage]，
每轮把新的输入 append 进去，再把整个列表重新喂给模型。

四种消息的角色与顺序：

    # | 顺序 | 类             | type 值 | 谁产生   | 作用                                       | 是否必需       |
    # |  1   | SystemMessage  | system  | 开发者   | 人设 / 规则 / 输出格式约束，放最前，通常一条 | 否（但强烈建议） |
    # |  2   | HumanMessage   | human   | 用户     | 本轮用户输入                                | 是             |
    # |  3   | AIMessage      | ai      | 模型     | 模型回复；要调工具时里面带 tool_calls        | 由模型产生      |
    # |  4   | ToolMessage    | tool    | 工具执行层 | 工具结果，必须用 tool_call_id 与请求配对     | 调工具时必需    |

顺序铁律：
    system 必须在最前（且只放最前），human / ai / tool 按时间顺序交替追加；
    ToolMessage 必须紧跟在「发起该次 tool_call 的 AIMessage」之后，
    并且 tool_call_id 要对得上，否则供应商接口直接报 400。

本节还会演示一个反例：把裸字符串 `str` 混进消息列表——LangChain 会把它
当成 HumanMessage 处理，虽然能跑，但角色语义就丢了。

课案出处：Agent 课案 → langChain → 核心组件 → 消息

前置条件：
    - 项目根目录 `.env` 里 `API_KEY` / `BASE_URL` / `MODEL_NAME` 三项已填好
      （`config.py` 自动加载，本文件用 `settings.xxx` 读取）；
    - 只调大模型，不碰数据库，**不需要** PostgreSQL / Redis。

运行方式（在项目根目录 `F:\ProGram\Python_Base` 下执行，否则 import 不到 `config`）：
    uv run Agent/02_langchain/02_消息_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

# 新版导入路径（课案原文用的就是这条）；等价于 from langchain_core.messages import ...
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from config import settings

# 课案写的是 ChatOpenAI(model=..., api_key=..., base_url=...)；
# 本项目统一走 init_chat_model，参数同样全部来自 settings。
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 1. 消息对象体检：看清四种消息各自长什么样 ----------
def dump(tag: str, msg) -> None:
    """打印一条消息的「身份信息」，方便对照上面的表格。"""
    print(f"  {tag:<10} 类={type(msg).__name__:<14} type={msg.type:<7} content={msg.content!r}")
    # ToolMessage 才有 tool_call_id；AIMessage 要调工具时才有 tool_calls
    if isinstance(msg, ToolMessage):
        print(f"  {'':<10} tool_call_id={msg.tool_call_id}")
    if isinstance(msg, AIMessage) and msg.tool_calls:
        print(f"  {'':<10} tool_calls={msg.tool_calls}")


if __name__ == "__main__":
    # ---------- 2. 课案原文：SystemMessage + HumanMessage 写七言绝句 ----------
    messages = [
        SystemMessage("你是一位唐诗专家"),
        HumanMessage("写一首关于春天的七言绝句"),
    ]
    print("===== 2. 课案原文：两条消息 =====")
    # 打印的是「我们即将发出去的东西」，不是模型的返回 —— 对照上面那张四行表看：
    # 这里刚好是「system 在最前、human 随后」的合规顺序。
    for index, msg in enumerate(messages, start=1):
        dump(f"[{index}]", msg)

    # 为什么可以整个列表丢进去：模型这一层无状态，每次请求都要重新带上全部消息，
    # 所以「怎么拼这个列表」就是 LangChain 消息体系的全部工作量所在。
    response = llm.invoke(messages)
    print("\n模型返回：", response.content)
    dump("[模型返回]", response)          # 模型返回的是 AIMessage

    # ---------- 3. 多轮对话 = 不断 append 消息 ----------
    # 注意：不是「模型记住了」，而是「我们把上一轮它说的话又原样递回去了」。
    print("\n===== 3. 多轮对话：把 AIMessage 追加进列表 =====")
    messages.append(response)                          # 追加模型刚才的回复（AIMessage）
    messages.append(HumanMessage("把这首诗改成五言"))    # 追加新一轮用户输入（HumanMessage）
    print(f"  当前列表共 {len(messages)} 条消息，角色顺序：{[m.type for m in messages]}")

    response = llm.invoke(messages)
    print("模型返回：", response.content)

    # ---------- 4. ToolMessage：工具结果的载体 ----------
    # 只有在模型发起工具调用之后才会出现，必须与 tool_call_id 配对。
    # 这里手工造一对「AIMessage(带 tool_calls) + ToolMessage」展示结构，
    # 真正由智能体自动产生的过程见 03_智能体_jxsd.py / 04_工具_jxsd.py。
    print("\n===== 4. ToolMessage 的结构（手工构造，仅示意） =====")
    # 手工拼这两条消息，是为了让你看清楚「一次工具调用」在消息列表里长什么样：
    # AIMessage 用 tool_calls[].id 提出问题，ToolMessage 用 tool_call_id 回答 —— 一问一答。
    # 框架自动做这件事时，就是 create_agent 内部那个 tool 节点。
    ai_with_tool_call = AIMessage(
        content="",                                   # 要调工具时，正文通常为空
        tool_calls=[
            {
                "name": "get_weather",                # 工具名
                "args": {"city": "北京"},              # 参数（由模型按 schema 填）
                "id": "call_1",                       # 本次调用的唯一编号
                "type": "tool_call",
            }
        ],
    )
    tool_message = ToolMessage(
        content="北京：晴，25℃",                        # 工具真正返回的内容
        name="get_weather",
        tool_call_id="call_1",                        # 必须等于上面那条 tool_call 的 id
    )
    dump("[AI]", ai_with_tool_call)
    dump("[TOOL]", tool_message)

    # ---------- 5. 元组写法 / 裸字符串：语法糖对照 ----------
    # 三种写法等价，LangChain 内部都会归一化成消息对象：
    #   ("user", "你好")            → HumanMessage("你好")
    #   {"role": "user", "content": "你好"} → HumanMessage("你好")
    #   "你好"                       → HumanMessage("你好")   ← 丢了角色语义，不推荐
    print("\n===== 5. 三种等价写法 =====")
    # 下面两行只列「元组」和「字典」两种，第三种（裸字符串）刻意没放进来 ——
    # 它就是上面注释里那个「角色语义丢失」的反例，单独跑一遍看不出差别，容易误导。
    sugar_forms = [
        ("元组写法", [("system", "只回答一个字"), ("user", "中国的首都是？")]),
        ("字典写法", [{"role": "system", "content": "只回答一个字"}, {"role": "user", "content": "中国的首都是？"}]),
    ]
    for tag, msgs in sugar_forms:
        # 归一化发生在 invoke 内部：LangChain 把元组/字典都转成 Message 对象再发给接口，
        # 所以两种写法对模型而言完全等价（差别只在 Python 侧的写法习惯）。
        normalized = llm.invoke(msgs)
        print(f"  {tag}：{normalized.content}")
