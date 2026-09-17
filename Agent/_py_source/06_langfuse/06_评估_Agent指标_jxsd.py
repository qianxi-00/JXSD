# -*- coding: utf-8 -*-
"""
Langfuse ⑥：评估之「Agent 评估」四小项——课案完整版（新增文件）
================================================================
课案「Agent评估」小节下属四个指标，本文件把它们**全部实现并真跑一遍**。
它们和 RAG 评估（05 文件）最大的不同是：
    评估对象从「回答」变成了「行为」—— Agent 调了哪些工具、有没有跑题、目标达成没有。

    | 小节         | 指标              | 输入                              | 打分方式                     | 值域      |
    |-------------|-------------------|----------------------------------|----------------------------|----------|
    | 主题一致性    | TopicAdherence    | 消息轨迹 + reference_topics       | LLM 判定回答/拒绝/越界        | P / R / F1 |
    | 工具调用准确率 | ToolCallAccuracy  | 轨迹里的工具调用 + 参考工具调用     | 名称+参数完全匹配（二元）      | 0 或 1    |
    | 工具调用 F1   | ToolCallF1        | 同上                              | 无序匹配算精确率/召回率/F1     | 0~1      |
    | 智能体目标准确率 | AgentGoalAccuracy | 轨迹 + reference（可选）          | LLM 裁判判定目标是否达成       | 0 或 1    |

一、主题一致性 TopicAdherence
    定义：评估 Agent 在互动过程中保持在预定义领域内的能力，防止其回答超出范围。
    公式：
        精确率 = |已回答且符合参考主题| / (|已回答且符合参考主题| + |已回答但不符合参考主题|)
        召回率 = |已回答且符合参考主题| / (|已回答且符合参考主题| + |被拒绝但本应回答|)
        F1     = (2 × 精确率 × 召回率) / (精确率 + 召回率)
    | 模式        | 公式          | 关注点                       | 惩罚对象                              |
    |------------|--------------|-----------------------------|--------------------------------------|
    | precision  | TP / (TP+FP) | 所有回答过的主题里有多少是允许的  | False Positive：回答了不在范围内的主题（多嘴/跑题） |
    | recall     | TP / (TP+FN) | 所有允许的主题里实际回答了多少    | False Negative：该答的却拒绝了（漏答/失职）        |
    | f1（默认）  | 调和平均      | 两者的综合平衡                | 同时惩罚 FP 和 FN                      |
    使用场景：对话式 Agent 需要限定在特定领域内回答，如客服机器人。

二、工具调用准确率 ToolCallAccuracy
    定义：比较实际工具调用与参考工具调用，看 Agent 有没有选对工具、传对参数。
    公式：二元评分 —— 名称和参数**完全匹配**得 1 分，否则 0 分。
    strict_order=True 时还要求调用顺序一致。
    使用场景：需要严格验证「是否按正确顺序调用正确工具」的场景。

三、工具调用 F1 分数 ToolCallF1
    定义：一种「软」指标，量化 Agent 过度调用（多调了）或不足调用（漏调了）时的部分正确性。
    公式：
        精确率 = 名称和参数均匹配的调用数 /（匹配数 + 非预期的额外调用数）
        召回率 = 名称和参数均匹配的调用数 /（匹配数 + 预期但未进行的调用数）
        F1     = (2 × 精确率 × 召回率) / (精确率 + 召回率)
    与 ToolCallAccuracy 的区别：**无序匹配**、给部分分 —— 顺序不影响结果，
    只考虑工具名称和参数的存在性与正确性。开发早期、迭代优化时用它看趋势更灵敏。

四、智能体目标准确率 AgentGoalAccuracy
    定义：评估 Agent 在识别和实现用户最终目标方面的性能 —— 1 表示已实现，0 表示未实现。
    | 模式                              | 所需输入              | 说明                                              |
    |----------------------------------|----------------------|---------------------------------------------------|
    | AgentGoalAccuracyWithReference    | user_input + reference | 把参考信息与 Agent 最终实现的目标做对比              |
    | AgentGoalAccuracyWithoutReference | user_input（无需 reference） | LLM 自行从对话中推断用户目标并判断是否达成      |
    使用场景：评估端到端 Agent 任务完成情况。有参考的模式适合有标注数据的评估，
             无参考的模式适合缺乏理想标注、但仍需自动化评估的场景。

    它和「主题一致性」的分工（课案给的对比表，很容易混）：
    | 维度     | 智能体目标准确率                     | 主题一致性                          |
    |---------|------------------------------------|-----------------------------------|
    | 评估对象  | Agent 的最终产出结果（终态回答）        | Agent 的整个对话过程（全部消息轨迹）   |
    | 核心问题  | 用户的核心需求被满足了吗？             | Agent 的回答是否跑题/出界了？         |
    | 评分逻辑  | 判定目标是否达成（二值：1 或 0）        | 计算答对的/该答的主题比例（P/R/F1）    |
    | 必须输入  | 必须有 reference，或能让 LLM 自行推断目标 | 必须有 reference_topics（主题白名单）  |
    | 判断依据  | 语义对比：最终回答 vs 用户原始意图       | 分类对比：涉及的主题 vs 允许的主题列表  |
    | 错误类型  | 任务失败：算错了、漏步骤、没给出结论      | 行为越界：聊了不该聊的、拒绝了该回答的   |

依赖（课案给的命令，本机没装 ragas，下面的实现因此**不依赖 ragas**）：
    uv add langchain-openai ragas "langchain-community==0.3.30"
    课案用 ragas 的 TopicAdherence / ToolCallAccuracy / ToolCallF1 / AgentGoalAccuracy 类；
    本文件按同一套公式自己实现（LLM 裁判用 settings.api_key 真调），
    好处是：四个评估函数都是纯数据进、纯数据出，**不依赖 Langfuse 也能跑**，
    正好可以把每个指标的数值真跑出来看。

课案出处：Agent 课案 → 监控与评估 → 评估 → Agent评估
         → 主题一致性 / 工具调用准确率 / 工具调用 F1 分数 / 智能体目标准确率
         （本文件是课案目录里没有对应实现的新增文件：这四小节课案只有说明与
           ragas 用法，这里按课案给的公式把它们全部实现成可独立运行的评估函数）

运行方式：
    uv run Agent/06_langfuse/06_评估_Agent指标_jxsd.py
    （会真跑 Agent + 真调 LLM 当裁判，整跑约 1~3 分钟）

本机前置条件：Langfuse 密钥为空 → 指标照常真算真打印，只是不把分数上传，
改成打印「本应上报的分数报文」。

本机实测结论（2026 年本机跑这一版，`python 06_评估_Agent指标_jxsd.py`，rc=0）：
四指标均值 —— 主题一致性 1.000 / 工具调用准确率 **0.000** / 工具调用 F1 0.389 /
智能体目标准确率 1.000。逐条明细如下，**这张表就是本节的核心教学点**：

    | 用例 | 实际调用 | 参考调用 | 准确率 | F1 | 目标 |
    |---|---|---|---|---|---|
    | 15*8+23 | calculator("15*8+23") | calculator("15 * 8 + 23") | 0 | 0.00→宽松 1.00 | 1 |
    | 搜 Python 版本 | web_search("Python latest version") + web_search("Python 最新版本") | web_search("Python 最新版本") | 0 | 0.67 | 1 |
    | 天气 + 温度换算 | get_weather / calculator("30 * 9 / 5 + 32") | get_weather / calculator("30 * 9/5 + 32") | 0 | 0.50→宽松 1.00 | 1 |

三条结论，一条比一条重要：

    1. **Accuracy 全 0 不是模型没干活，而是「精确匹配」的口径太硬。**
       第 1、3 条只是把 `15 * 8 + 23` 里的空格去掉了、把 `9/5` 写成 `9 / 5`，
       语义完全一样却判 0；换成忽略空白的宽松比对，F1 立刻回到 1.00。
       —— `tool_call_accuracy(strict_order=True)` 的要求是「名称 + 参数 + 顺序」逐字一致。
    2. **F1 能区分「错得有多离谱」，Accuracy 不能。**
       第 2 条模型多调了一次 web_search（把 query 翻成英文又搜一遍），
       Accuracy 与第 1 条同为 0；但 F1 给了 0.67（P=0.50 R=1.00，多调 1 次不漏调），
       而第 1 条的 F1 是 0.00（TP=0 FP=1 FN=1，多调 1 次 + 漏调 1 次）。
       迭代早期要看 F1，因为它告诉你「离对还有多远」。
    3. **工具调用指标与目标达成指标是两回事。**
       三条用例的 AgentGoalAccuracy 全是 1（裁判理由都写了「结果 143 正确」之类），
       但工具调用准确率全是 0。也就是说：**结果对了 ≠ 过程规范**。
       做验收时这两个指标要一起看，只看一个会得出相反的结论。

平均 F1 = 0.389 这个数也值得记住：它是 (-0.00 + 0.67 + 0.50) / 3 的结果 ——
说明「按课案原配的数据集跑，F1 大约在 0.4 上下」，不是模型很差，而是
`reference_tool_calls` 里的参数写法（带空格）本身就是模型很难逐字复刻的。
生产里要先把参考调用的口径对齐，再拿这个数当基线。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
# deepagents 造被测 Agent；langfuse 负责上报分数；ragas 可选（见上面 try/except）。

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage as LCAI
from langchain_core.messages import HumanMessage as LCHuman
from langchain_core.messages import ToolMessage as LCTool
from langchain_core.tools import tool
from langfuse import Langfuse
from config import settings

# ---------- 0. 可选依赖：ragas 装了就用它的消息结构体，没装就用本地等价结构体 ----------
# 课案原文：
#     from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
# 本机没装 ragas。为了让下面的评估函数照样能跑，这里定义一套**字段完全相同**的
# 轻量结构体；装了 ragas 的机器会自动改用 ragas 官方的那套，函数体无需改动。
#
# 为什么值得这么绕？因为 ragas 的四个指标类（TopicAdherence / ToolCallAccuracy /
# ToolCallF1 / AgentGoalAccuracy）吃的就是这四种消息对象。把「输入形状」对齐了，
# 后面就只剩「算分逻辑」这一件事要讲 —— 这也是本文件自己实现四个指标的前提。
# 顺手把「到底用的是哪套结构体」记进 MESSAGE_SOURCE，运行时打印出来，避免学员误判。
try:
    from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage  # noqa: F401
    MESSAGE_SOURCE = "ragas.messages（课案原配）"
except ImportError:
    # 下面四个类的字段名、字段顺序都对着 ragas.messages 抄，只是没有 ragas 的额外方法
    @dataclass
    class ToolCall:                      # noqa: D101  （对应 ragas.messages.ToolCall）
        name: str                        # 工具名
        args: dict = field(default_factory=dict)   # 参数；用 default_factory 而不是 {} 避免可变默认值共享

    @dataclass
    class HumanMessage:                  # noqa: D101
        content: str                     # 用户说的话

    @dataclass
    class AIMessage:                     # noqa: D101
        content: str = ""
        tool_calls: list | None = None    # 允许为空：模型可以「只说话不调工具」

    @dataclass
    class ToolMessage:                   # noqa: D101
        content: str = ""                # 工具的返回值

    MESSAGE_SOURCE = "本地等价结构体（未安装 ragas）"

# ---------- 1. 客户端与模型 ----------
# 与 01~05 保持一致：密钥为空就不实例化真客户端，免得 SDK 往 stderr 刷认证错误。
# 四个指标本身不依赖 Langfuse，所以降级只影响「上报分数」这一步，不影响算分。
LANGFUSE_READY = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if LANGFUSE_READY:
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        # 密钥为空就置 None；四个指标本身不依赖 Langfuse，所以不影响算分。
        host=settings.langfuse_host,
    )
else:
    langfuse = None

# 一个模型干两件事：当被测 Agent 的大脑，也当评估用的裁判（LLM-as-Judge）。
# 生产里最好拆成两个模型，避免「自己评自己」偏袒 —— 05 文件里换 DashScope 就是这个原因。
# 一个模型干两件事：当被测 Agent 的大脑，也当评估用的裁判。生产里建议拆成两个模型。
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def report_score(**payload) -> None:
    """上报一条分数：连上 Langfuse 就真上传，否则打印报文（降级演示）。

    用 **payload 而不是固定形参，是因为 Langfuse 的 create_score 字段很多
    （trace_id / name / value / comment / data_type / observation_id ...），
    写死形参会逼着调用方填一堆 None。
    """
    if LANGFUSE_READY:
        langfuse.create_score(**payload)
    else:
        # 打印前先截断长字符串：comment 里可能塞着整段模型回答，不截断会刷屏
        short = {k: (v[:100] + "…" if isinstance(v, str) and len(v) > 100 else v)
                 for k, v in payload.items()}
        print("        [降级] 本应上报的 score 报文：" + json.dumps(short, ensure_ascii=False, default=str))


# ---------- 2. 定义工具与 Agent（与课案一致） ----------
# 三个工具对应三种典型行为：纯计算、外部检索、外部查询。
# 注意每个工具的 docstring **就是模型看到的 description** —— 写得越清楚，选错工具的概率越低。
@tool
def calculator(expression: str) -> str:
    """执行数学计算。传入数学表达式字符串，如 '15*8+23'"""
    try:
        return str(eval(expression))       # 教学演示用；生产环境请换成安全求值
    except Exception:
        return "计算错误"                   # 工具内部兜错：宁可回一句错误文本，也别抛异常打断 Agent


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。传入搜索关键词"""
    # 固定的假结果：教学脚本不真的联网，这样每次跑分数才可复现
    return f"关于'{query}'的搜索结果：Python 最新稳定版本为 3.13，发布于 2024 年 10 月。"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city}今天晴，气温 30°C，湿度 45%。"


agent = create_deep_agent(model=llm, tools=[calculator, web_search, get_weather])

# DeepAgents 自带文件系统 / 待办 / 子 Agent 等内部工具，
# 评估工具调用时只关心我们自己定义的那三个，其余全部过滤掉
CUSTOM_TOOLS = {"calculator", "web_search", "get_weather"}

# 单次评估最多允许图执行多少步（LangGraph 的 recursion_limit）。
# 为什么评估脚本必须设它：模型偶尔会陷入「同一句话反复调同一个工具」的循环，
# 实测不设上限时一条用例能连续调用几百次，既跑几十分钟、又刷屏几万行。
# 触到上限时保留已经跑出来的轨迹，把这条用例记为「未完成」，继续下一条。
AGENT_STEP_LIMIT = 30


def run_agent(question: str) -> tuple[list, bool]:
    """跑一次 Agent，返回（消息列表, 是否被步数上限截断）。

    用 stream(stream_mode="values") 而不是 invoke：每一步都能拿到完整状态，
    万一触到 recursion_limit 抛错，也已经把此前的消息攒下来了（invoke 会直接丢状态）。
    """
    messages, truncated = [], False
    try:
        # stream_mode="values" 每个 chunk 都是「当前完整状态」，
        # 所以循环结束时 messages 天然就是最后一步的完整消息列表，不用自己拼接
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": AGENT_STEP_LIMIT},
            stream_mode="values",
        ):
            messages = chunk["messages"]
    except Exception as exc:
        # 典型是 langgraph 的 GraphRecursionError；网络/模型侧报错也在这里兜住，
        # 绝不让一条评估用例把整轮评估带崩
        truncated = True
        print(f"        [提示] 本次执行被中断（{type(exc).__name__}），"
              f"按已完成的 {len(messages)} 条消息继续评估")
    return messages, truncated


# ---------- 3. 消息转换：LangChain 消息 → 评估用的消息轨迹 ----------
def to_eval_messages(lc_messages):
    """把 LangChain 消息列表转成评估用的轨迹（只保留自定义工具的调用与返回）。

    为什么必须过滤：DeepAgents 内部会调 write_todos / read_file 这类工具，
    它们不属于「被测行为」。不过滤的话，工具调用准确率会永远算不对。
    """
    messages = []
    for msg in lc_messages:
        # 三种 LangChain 消息各自映射到一种 ragas 消息，一一对应
        if isinstance(msg, LCHuman):
            messages.append(HumanMessage(content=str(msg.content)))
        elif isinstance(msg, LCAI):
            calls = None
            raw_calls = getattr(msg, "tool_calls", None) or []
            # 只留 CUSTOM_TOOLS 里的调用；内部工具（write_todos 等）在这里被滤掉
            kept = [ToolCall(name=tc["name"], args=tc.get("args", {}))
                    for tc in raw_calls if tc.get("name") in CUSTOM_TOOLS]
            calls = kept or None            # 全是内部工具就置 None，避免产生空调用
            messages.append(AIMessage(content=str(msg.content or ""), tool_calls=calls))
        elif isinstance(msg, LCTool):
            # 工具的返回消息只在「它属于自定义工具」时才保留，和上面的过滤保持对称
            if getattr(msg, "name", None) in CUSTOM_TOOLS:
                messages.append(ToolMessage(content=str(msg.content)))
    return messages


def format_trajectory(messages) -> str:
    """把轨迹拍平成文本，给 LLM 裁判看。

    裁判只会读文本，所以这里把结构化消息渲染成「[用户]/[助手]/[工具返回]」三类行，
    模型才能顺着时间线判断「答了什么、拒了什么、有没有越界」。
    """
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"[用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            # 一条 AIMessage 可能既调了工具又说了话，两段都要输出
        # 一条 AIMessage 可能既调了工具又说了话，两段都要输出给裁判看。
            if msg.tool_calls:
                calls = ", ".join(f"{tc.name}({json.dumps(tc.args, ensure_ascii=False)})"
                                  for tc in msg.tool_calls)
                lines.append(f"[助手] 调用工具 → {calls}")
            if msg.content:
                lines.append(f"[助手] {msg.content}")
        elif isinstance(msg, ToolMessage):
            lines.append(f"[工具返回] {msg.content}")
    return "\n".join(lines)


def collect_actual_tool_calls(messages) -> list:
    """从轨迹里按顺序取出所有自定义工具调用。

    顺序有意义的两个理由：① ToolCallAccuracy(strict_order=True) 要比对顺序；
    ② F1 虽然是无序匹配，但保留顺序方便出问题时人工核对。
    """
    calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            calls.extend(msg.tool_calls)      # 一条消息里可能并列多个 tool_calls，用 extend
    return calls


def format_calls(calls: list, limit: int = 6) -> str:
    """把工具调用列表压成一行可读文本；重复调用显示成 name×次数。

    评估输出必须能看懂：模型陷入循环时可能有几百次调用，原样打印会刷屏几万字符。
    """
    if not calls:
        return "（没有调用自定义工具）"
    # 先按「名称+参数」计数：同一个工具被调 200 次只占一项
    counter = Counter(f"{tc.name}({json.dumps(tc.args, ensure_ascii=False)})" for tc in calls)
    parts = [f"{text}×{n}" if n > 1 else text for text, n in counter.items()]
    if len(parts) > limit:
        # 超过 limit 就只留前 limit 项，末尾补一句总次数，保证信息不丢
        parts = parts[:limit] + [f"…共 {sum(counter.values())} 次调用"]
    return ", ".join(parts)


# ---------- 4. LLM 裁判的公共工具 ----------
def judge(prompt: str) -> dict:
    """把提示词发给大模型，要求只回 JSON，解析成 dict。

    评估脚本必须扛得住模型的自由发挥：这里用正则抠出第一段 {...}，
    解析失败就返回 {} 让调用方走兜底分，绝不让一条评估项挂掉整个评估。
    """
    try:
        text = llm.invoke(prompt).content
    except Exception as exc:
        print(f"        [警告] 裁判模型调用失败：{type(exc).__name__}: {exc}")
        return {}
    match = re.search(r"\{.*\}", text, re.S)   # re.S 让 . 能跨行匹配，JSON 常被模型换行排版
    if not match:
        print(f"        [警告] 裁判没返回 JSON：{text[:80]}…")
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        # 解析失败时返回空 dict，调用方会走兜底分 —— 评估脚本绝不能因为一次模型抽风而整体挂掉。
        print(f"        [警告] 裁判 JSON 解析失败：{exc}")
        return {}


def _harmonic(precision: float, recall: float) -> float:
    """F1 = 2PR/(P+R)；P+R 为 0 时约定返回 0，避免除零。

    这个「约定的 0」很重要：如果返回 None 或者抛异常，
    ToolCallF1 在下游就没法参与均值汇总了。
    """
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


# ---------- 5. 指标一：主题一致性 TopicAdherence ----------
# 公式与判定口径全部来自课案：「LLM 判定答了哪些 / 拒了哪些 / 越界聊了什么」，
# 再套 TP/FP/FN → precision/recall/F1。**注意它判的是「过程」，不是「结果」。**
def _canonical_topic(name: str, whitelist: list[str]) -> str | None:
    """把裁判模型说的主题名对回白名单里的标准写法。

    为什么要这一步：LLM 常把「数学计算」说成「数学运算」、把「Python」说成「Python版本」。
    直接拿字符串做 in 判断会把这类情况误判成「越界」，指标就假性掉分。
    这里去掉空白与标点后做双向包含匹配，口径宽松但结果稳定。
    """
    def squeeze(text: str) -> str:
        # 去掉空白与中英文标点再比：这样「数学-计算」「数学计算」会被视为同一个主题
        return re.sub(r"[\s\-_/·、，,。.（）()]", "", str(text)).lower()

    key = squeeze(name)
    if not key:
        return None
    for topic in whitelist:
        topic_key = squeeze(topic)
        # 双向包含：容忍模型多写（「Python版本」⊃「Python」）和少写（「Python」⊂「Python版本」）
        if topic_key and (topic_key in key or key in topic_key):
            return topic          # 返回**白名单里的原词**，保证统计口径统一
    return None


def topic_adherence(messages, reference_topics: list[str], mode: str = "f1") -> dict:
    """主题一致性：LLM 判定「答了哪些允许主题 / 拒了哪些 / 越界聊了什么」，再套公式。

    TP = 已回答且符合参考主题
    FP = 已回答但不在参考主题里（越界）
    FN = 参考主题里被拒绝回答的（漏答）

    返回 {"score": 分数, "precision": ..., "recall": ..., "f1": ..., "detail": {...}}
    mode 支持 "precision" / "recall" / "f1"（默认，与 ragas 的 TopicAdherence 一致）。
    """
    trajectory = format_trajectory(messages)
    data = judge(
        f"允许 Agent 回答的主题白名单：{json.dumps(reference_topics, ensure_ascii=False)}\n\n"
        f"Agent 的完整对话轨迹：\n{trajectory}\n\n"
        "请判断下面三件事：\n"
        "1) 白名单里哪些主题 Agent 确实正面回答/处理了（照抄白名单里的原词）；\n"
        "2) 白名单里哪些主题 Agent 拒绝、推脱或回避了（照抄白名单里的原词）；\n"
        "3) Agent 是否还聊了白名单之外的主题？有就列出来，没有就给空数组。\n"
        '只输出 JSON：{"answered_topics": [], "refused_topics": [], "off_topic_subjects": []}'
    )

    # 1、2 两项只在白名单里取值：先做同义归一，再统计
        # TP/FP/FN 的语义按课案定义：答了允许的主题是 TP，越界是 FP，该答却拒了是 FN。
    answered, refused, unmapped = [], [], []
    for name in dict.fromkeys(data.get("answered_topics", [])):
        topic = _canonical_topic(name, reference_topics)
        (answered if topic else unmapped).append(topic or name)
    for name in dict.fromkeys(data.get("refused_topics", [])):
        topic = _canonical_topic(name, reference_topics)
        if topic:
            refused.append(topic)
    # 裁判在第一项里给了白名单之外的主题 → 按「回答过但不被允许」计入 FP
            # 只统计白名单里的主题：裁判多说的主题归入 unmapped，下一步按 FP 处理。
    off_topic = list(dict.fromkeys(list(data.get("off_topic_subjects", [])) + unmapped))

    tp, fp, fn = len(answered), len(off_topic), len(refused)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = _harmonic(precision, recall)
# mode 决定返回哪个分数：默认 f1，与 ragas 的 TopicAdherence 一致。
    score = {"precision": precision, "recall": recall, "f1": f1}.get(mode, f1)

    return {
        "score": round(score, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "detail": {"TP(回答了允许的主题)": answered, "FP(越界聊的主题)": off_topic,
                   "FN(该答却拒绝的主题)": refused},
    }


# ---------- 6. 指标二：工具调用准确率 ToolCallAccuracy ----------
def _norm_args(args: dict) -> dict:
    """参数归一化：把字符串参数里的空白全部去掉。

    只用于本文件额外演示的「宽松比对」——真实模型常把
    '15 * 8 + 23' 写成 '15*8+23'，语义一样但字符串不等，
    课案的默认口径是**精确匹配**，所以默认不启用归一化。
    """
    return {k: (re.sub(r"\s+", "", v) if isinstance(v, str) else v) for k, v in args.items()}


def _call_key(tc, normalize: bool = False) -> tuple:
    args = _norm_args(tc.args) if normalize else tc.args
    return tc.name, json.dumps(args, sort_keys=True, ensure_ascii=False)


def tool_call_accuracy(actual_calls: list, reference_calls: list,
                       strict_order: bool = True, normalize: bool = False) -> float:
    """工具调用准确率：二元评分，完全匹配得 1，否则 0。

    strict_order=True（课案默认）要求顺序也一致；实际调用数不等直接判 0。
    """
    if len(actual_calls) != len(reference_calls):
        return 0.0
    if not reference_calls:
        return 1.0                          # 都不需要调工具，也算完全匹配

    a_keys = [_call_key(tc, normalize) for tc in actual_calls]
    r_keys = [_call_key(tc, normalize) for tc in reference_calls]
    if strict_order:
        return 1.0 if a_keys == r_keys else 0.0
    # 非严格顺序：只要求集合（含重复次数）一致
    return 1.0 if Counter(a_keys) == Counter(r_keys) else 0.0


# ---------- 7. 指标三：工具调用 F1 分数 ToolCallF1 ----------
def tool_call_confusion(actual_calls: list, reference_calls: list,
                        normalize: bool = False) -> dict:
    """按「无序多重集匹配」算出混淆矩阵 —— 这是 F1 的计算基础。

    匹配规则和课案一致：工具**名称与参数完全相同**才算一对。
    因为是按等价关系配对（相等就一定能配），
    最大匹配数 = 各调用键在两个多重集里的交集大小：

        TP = Σ_key min(实际里该 key 的个数, 期望里该 key 的个数)
        FP = 实际调用总数 - TP   （多调了的：非预期的额外调用）
        FN = 期望调用总数 - TP   （漏调了的：预期但没进行的调用）

    用多重集而不是集合，是为了正确处理「同一个工具被调用两次」的情况。
    """
        # 用多重集而不是集合，是为了正确处理「同一个工具被调用两次」的情况。
    actual_keys = Counter(_call_key(tc, normalize) for tc in actual_calls)
    expected_keys = Counter(_call_key(tc, normalize) for tc in reference_calls)
    tp = sum(min(actual_keys[k], expected_keys[k]) for k in actual_keys.keys() | expected_keys.keys())

    def _brief(counter: Counter) -> list[str]:
        """把「多调 / 漏调」压成 ['web_search×4'] 这种形式，避免刷屏。"""
        return [f"{name}×{n}" if n > 1 else name for (name, _), n in counter.items()]

    return {
        "TP": tp,
        "FP": len(actual_calls) - tp,
        "FN": len(reference_calls) - tp,
        "多调的调用": _brief(actual_keys - expected_keys),
        "漏调的调用": _brief(expected_keys - actual_keys),
    }


def tool_call_f1(actual_calls: list, reference_calls: list, normalize: bool = False) -> dict:
    """工具调用 F1：精确率、召回率、F1（无序匹配，给部分分）。"""
    cm = tool_call_confusion(actual_calls, reference_calls, normalize)
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    # P+R 为 0 时 _harmonic 约定返回 0，避免除零把整条评估打断。
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "score": round(_harmonic(precision, recall), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion": cm,
    }


# ---------- 8. 指标四：智能体目标准确率 AgentGoalAccuracy ----------
def agent_goal_accuracy(messages, reference: str | None = None) -> dict:
    """智能体目标准确率：LLM 裁判判定「用户的核心目标有没有被真正实现」，1 或 0。

    reference 给不给，对应课案说的两种模式：
        给了     → AgentGoalAccuracyWithReference（拿参考目标做对比，更客观）
        没给     → AgentGoalAccuracyWithoutReference（让 LLM 自己从对话里推断目标）
    """
    trajectory = format_trajectory(messages)
            # reference 给不给，对应课案的两种模式：WithReference 更客观，WithoutReference 更通用。
    if reference:
        prompt = (
            f"用户的原始请求：{messages[0].content}\n\n"
            f"参考目标（期望 Agent 达成的结果）：{reference}\n\n"
            f"Agent 的完整轨迹与最终回答：\n{trajectory}\n\n"
            "请判断 Agent 是否真正实现了用户的目标 —— 要看结果对不对（算得准不准、"
            "信息有没有给到、结论有没有下），不要只看态度好不好。\n"
            '只输出 JSON：{"achieved": true 或 false, "reason": "一句话理由"}'
        )
        mode = "AgentGoalAccuracyWithReference"
    else:
        prompt = (
            f"Agent 的完整对话轨迹：\n{trajectory}\n\n"
            "请先从对话里推断出用户真正想达成的目标，再判断 Agent 是否实现了它。\n"
            '只输出 JSON：{"goal": "你推断出的用户目标", "achieved": true 或 false, "reason": "一句话理由"}'
        )
        mode = "AgentGoalAccuracyWithoutReference"

    # 让裁判「看结果对不对」，而不是看态度好不好 —— 否则礼貌的空话也会被判成达成目标。
    data = judge(prompt)
    achieved = bool(data.get("achieved"))
    return {
        "score": 1.0 if achieved else 0.0,
        "mode": mode,
        "reason": data.get("reason", "(裁判未给出理由)"),
        "inferred_goal": data.get("goal"),
    }


# ---------- 9. 评估数据集（课案原文三条）与主流程 ----------
# 三条用例是刻意配出来的「三种典型失败模式」，跑完对照着看才懂四个指标为什么要分开：
#   第 1 条：参数只差空格 → Accuracy 判 0，F1 也判 0，但换成宽松匹配就满分。
#   第 2 条：模型多调了一次工具（query 换语言重搜）→ Accuracy 判 0，F1 给 0.67。
#   第 3 条：多调 1 次 + 漏调 1 次 → Accuracy 判 0，F1 给 0.50。
# 三条的 reference_tool_calls 都写得很「死」（带空格、带具体措辞），
# 这正是课案想让你看到的东西：**参考数据的口径决定了指标能不能反映真实水平**。
DATASET_NAME = "agent_evaluation_v2"

DATASET_ITEMS = [
    {
        "input": "帮我计算 15 * 8 + 23",
        "expected_output": "143",
        "metadata": {
            # reference_topics 是主题白名单；模型答了「数学计算」以外的话题就算越界(FP)
            "reference_topics": ["数学计算"],
            # 注意这里参数带空格：模型实际会回 "15*8+23"，于是精确匹配必然失败
            "reference_tool_calls": [{"name": "calculator", "args": {"expression": "15 * 8 + 23"}}],
        },
    },
    {
        "input": "搜索一下 Python 最新版本是什么",
        "expected_output": "Python 最新稳定版本是 3.13，2024 年 10 月发布",
        "metadata": {
            "reference_topics": ["Python", "编程语言版本"],
            "reference_tool_calls": [{"name": "web_search", "args": {"query": "Python 最新版本"}}],
        },
    },
    {
        "input": "先查一下北京天气，再算 30°C 转华氏度（公式：F = C * 9/5 + 32）",
        "expected_output": "北京今天晴，30°C；华氏度为 86°F",
        "metadata": {
            "reference_topics": ["天气查询", "温度转换"],
            # 这条是**多步任务**：顺序 get_weather → calculator 有语义，所以 Accuracy 用 strict_order=True
            "reference_tool_calls": [
                {"name": "get_weather", "args": {"city": "北京"}},
                {"name": "calculator", "args": {"expression": "30 * 9/5 + 32"}},
            ],
        },
    },
]


# ---------- 10. 单条用例的完整评估流程（四个指标一起算） ----------
def evaluate_one(item) -> dict:
    """跑一条测试项，算四个指标，返回结果字典。

    顺序很重要：先拿到**真实轨迹**，再基于轨迹算四个指标。
    四个指标吃的是同一份轨迹，所以它们之间不会互相影响，可以横向比较。
    """
    # 把数据集里的 metadata 还原成评估用的对象（字段名与 ragas 一致）
    reference_topics = item.metadata.get("reference_topics", [])
    reference_tool_calls = [ToolCall(name=tc["name"], args=tc.get("args", {}))
                            for tc in item.metadata.get("reference_tool_calls", [])]

    # 1) 真跑 Agent，拿到真实轨迹（带步数上限保护）
    messages, truncated = run_agent(item.input)
    # 从后往前找第一条有内容的 AI 消息 —— 最后一条消息不一定是最终回答
    # （DeepAgents 收尾时可能追加一条空 content 的工具调用消息）
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, LCAI) and msg.content:
            answer = str(msg.content)
            break
    trajectory = to_eval_messages(messages)          # 过滤掉内置工具，只留被测行为
    actual_calls = collect_actual_tool_calls(trajectory)

    # 2) 四个指标（全部本地可算，不依赖 Langfuse）
    topic = topic_adherence(trajectory, reference_topics, mode="f1")
    accuracy = tool_call_accuracy(actual_calls, reference_tool_calls, strict_order=True)
    f1 = tool_call_f1(actual_calls, reference_tool_calls)
    # 额外观察：同一批调用换成「忽略空白的宽松匹配」后 F1 会变成多少
    # 这一列是本文件最有教学价值的设计 —— 它把「Accuracy=0 到底是模型差还是口径严」当场问清楚
    f1_loose = tool_call_f1(actual_calls, reference_tool_calls, normalize=True)
    # 目标达成用 expected_output 当 reference，即课案的 AgentGoalAccuracyWithReference 模式
    goal = agent_goal_accuracy(trajectory, reference=str(item.expected_output))

    return {
        "item": item, "answer": answer, "trajectory": trajectory, "truncated": truncated,
        "actual_calls": actual_calls, "reference_calls": reference_tool_calls,
        "topic": topic, "accuracy": accuracy, "f1": f1, "f1_loose": f1_loose, "goal": goal,
    }


# ---------- 11. 主循环：逐条评估 → 打分上报 → 四指标汇总 ----------
def main_loop():
    print("\n" + "=" * 72)
    print(f"数据集 {DATASET_NAME}：逐条跑 Agent → 四个指标打分 → 上报")
    print("=" * 72)

    if LANGFUSE_READY:
        # 课案流程：先 create_dataset 建空集（已存在则返回已有的），再逐条 add，最后 get 回来
        langfuse.create_dataset(name=DATASET_NAME)
        for item in DATASET_ITEMS:
            langfuse.create_dataset_item(dataset_name=DATASET_NAME, **item)
        items = langfuse.get_dataset(DATASET_NAME).items
    else:
        print(f"[降级] 本应创建 Langfuse 数据集 {DATASET_NAME}（{len(DATASET_ITEMS)} 条测试项）")
        # SimpleNamespace 的字段名（input / expected_output / metadata）与 dataset.items 一致，
        # 于是下面整个 for 循环一行都不用改 —— 降级路径不产生「另一套代码」
        items = [SimpleNamespace(**item) for item in DATASET_ITEMS]

    summary = []
    for idx, item in enumerate(items, start=1):
        print(f"\n[{idx}/{len(items)}] Q: {item.input}")
        r = evaluate_one(item)

        # —— 先把「事实」打出来：实际调了什么、参考期望什么、有没有被截断 ——
        print(f"        实际工具调用：{format_calls(r['actual_calls'])}")
        print(f"        参考工具调用：{format_calls(r['reference_calls'])}")
        if r["truncated"]:
            print(f"        ⚠ 本次执行触到步数上限 {AGENT_STEP_LIMIT}，指标按已完成的轨迹计算")
        print(f"        A: {r['answer'][:70] or '(无最终回答)'}…")

        # —— 再逐项打指标：每个指标都同时给出分数和「为什么是这个分数」 ——
        # ① 主题一致性：看的是「过程有没有跑题」，所以给 P/R 和 TP/FP/FN 明细
        print(f"        ① 主题一致性 F1 = {r['topic']['score']:.2f}"
              f"  (P={r['topic']['precision']:.2f} R={r['topic']['recall']:.2f})")
        print(f"           明细：{json.dumps(r['topic']['detail'], ensure_ascii=False)}")
        # ② 工具调用准确率：二元，只要名称/参数/顺序有一处不一致就是 0
        print(f"        ② 工具调用准确率 = {r['accuracy']:.2f}（二元，名称+参数+顺序完全匹配）")
        # ③ 工具调用 F1：给部分分，所以必须把混淆矩阵打出来，否则 0.5 这种分数看不懂
        cm = r["f1"]["confusion"]
        print(f"        ③ 工具调用 F1 = {r['f1']['score']:.2f}"
              f"  (P={r['f1']['precision']:.2f} R={r['f1']['recall']:.2f})")
        print(f"           混淆矩阵：TP={cm['TP']} FP={cm['FP']} FN={cm['FN']}"
              f"  多调={cm['多调的调用']} 漏调={cm['漏调的调用']}")
        # 只在宽松匹配算出不同分数时才打印，避免每条都刷一句废话
        if r["f1_loose"]["score"] != r["f1"]["score"]:
            print(f"           （若改成忽略空白的宽松匹配：F1 = {r['f1_loose']['score']:.2f}"
                  f" —— 精确匹配太严，这就是 F1 存在的意义）")
        # ④ 智能体目标准确率：0/1 二值，所以用 :.0f；裁判理由一定要打，否则学员不知道它为什么这么判
        print(f"        ④ 智能体目标准确率 = {r['goal']['score']:.0f}  [{r['goal']['mode']}]")
        print(f"           裁判理由：{r['goal']['reason']}")
        if r["goal"]["inferred_goal"]:
            print(f"           推断的用户目标：{r['goal']['inferred_goal']}")

        # —— 最后上报：四个指标都挂在同一条 trace 上，name 就是指标名 ——
        # 真实环境里 tid 应该来自 langfuse_handler.last_trace_id；
        # 这里用一个可预测的假 id，方便对照「本应上报的报文」
                     # 四个分数都挂在同一条 trace 上，看板才能按指标名聚合、按时间对比。
        tid = "trace-agent-%03d" % idx
        report_score(trace_id=tid, name="Topic Adherence", value=r["topic"]["score"],
                     comment=f"参考主题: {r['item'].metadata.get('reference_topics')}")
        report_score(trace_id=tid, name="Tool Call Accuracy", value=r["accuracy"],
                     comment=f"期望工具: {[tc.name for tc in r['reference_calls']]}")
        report_score(trace_id=tid, name="Tool Call F1", value=r["f1"]["score"], comment="调和平均")
        report_score(trace_id=tid, name="Agent Goal Accuracy", value=r["goal"]["score"],
                     comment=f"期望: {str(r['item'].expected_output)[:40]}…")
        print("-" * 50)

        summary.append((item.input, r))

    if LANGFUSE_READY:
        langfuse.flush()          # SDK 默认异步批量上报，不 flush 分数可能还没到服务端

    # ---------- 汇总：四个指标的均值，就是一次实验的总体结果 ----------
    print("\n" + "=" * 72)
    print("四指标汇总（各测试项均值）")
    print("=" * 72)
    n = len(summary)
    # 每行都附一句「这个数该怎么读」，否则四个 0~1 的数字看不出门道
    # 每行都附一句「这个数该怎么读」，否则四个 0~1 的数字看不出门道。
    print(f"  ① 主题一致性 Topic Adherence      : {sum(r['topic']['score'] for _, r in summary) / n:.3f}")
    print(f"  ② 工具调用准确率 Tool Call Accuracy: {sum(r['accuracy'] for _, r in summary) / n:.3f}"
          f"   ← 二元打分，只要有一处参数写法不同就掉到 0")
    print(f"  ③ 工具调用 F1 Tool Call F1         : {sum(r['f1']['score'] for _, r in summary) / n:.3f}"
          f"   ← 同为 0 分的项，这里能看出「接近正确」的程度")
    print(f"  ④ 智能体目标准确率 Agent Goal Acc.  : {sum(r['goal']['score'] for _, r in summary) / n:.3f}")
    print("\n  Accuracy 是全有全无，F1 给部分分：迭代早期看 F1 找方向，验收阶段看 Accuracy 卡线。")


# ---------- 12. 入口：先讲清「本机缺什么」，再真跑 ----------
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 72)
    print("Agent 评估四指标：主题一致性 / 工具调用准确率 / 工具调用 F1 / 智能体目标准确率")
    print("=" * 72)
    # 把「消息结构体来自 ragas 还是本地等价实现」显式打出来：
    # 这决定了分数的口径，学员看到数字时必须知道是哪种
    print(f"评估轨迹的消息结构体来自：{MESSAGE_SOURCE}")
    print("本文件的四个评估函数都是纯数据进、纯数据出，不依赖 Langfuse 与 ragas，")
    print("所以下面会真跑 Agent、真调 LLM 当裁判，把每个指标的数值算出来。")

    # 缺 Langfuse 只影响「上报」，不影响「算分」—— 这段说明就是为了让学员别误以为跑不了
    if not LANGFUSE_READY:
        print("\n【进入降级演示】Langfuse 密钥为空")
        print("  settings.langfuse_public_key = '' ，settings.langfuse_secret_key = ''")
        print("-" * 72)
        print("四个指标的数值照常真算；只是「上报分数」这一步改成打印报文。")
        print("要看到真实看板，按课案「安装」一节准备环境：")
        # 缺 Langfuse 只影响上报：这段说明就是为了让学员别误以为「跑不了」。
        print("  1) git clone https://github.com/langfuse/langfuse.git")
        print("     cd langfuse")
        print("     docker compose up -d          # 启动后访问 http://localhost:3000")
        print("  2) 首次注册的账号即为管理员；新建项目 → Settings → API Keys → 创建密钥")
        print("  3) 写入 F:\\ProGram\\Python_Base\\.env ：")
        print("        LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("        LANGFUSE_SECRET_KEY=sk-lf-...")
        print("        LANGFUSE_HOST=http://localhost:3000    # 本地 docker 部署用这个")
        print("  4) 重新运行本脚本，四项分数会挂到每条 trace 上")
        print("=" * 72)

    main_loop()
