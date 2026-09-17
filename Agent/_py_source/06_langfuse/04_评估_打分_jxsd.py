# -*- coding: utf-8 -*-
"""
Langfuse ④：评估之「打分（Score）」——课案完整版
================================================================
课案原话：
    运行数据集，对每个测试项用 Agent 输出对比期望结果，自动打分并上报。

追踪（01 文件）回答「发生了什么」，打分回答「好不好」。
两者配合起来才叫评估闭环：
    trace（链路）  →  能看到每一步的输入输出
    score（分数）  →  能按分数筛出「回答得差」的那些 trace，再点进去看链路

分数从哪来？课案把来源分成两类：

    | 来源       | 谁打分          | 例子                                        |
    |-----------|----------------|--------------------------------------------|
    | 人工打分   | 人在 Web UI 上点 | 点赞/点踩、给客服回答打 1~5 分（标注队列，见 07 文件） |
    | 代码打分   | 程序            | 用户反馈上报、业务指标、字符串匹配、LLM 评审   |

一条 score 的四个要素：

    | 字段      | 说明                                                            |
    |----------|-----------------------------------------------------------------|
    | trace_id | 这个分数打在**哪一次调用**上（也可以用 observation_id 打到某个 span） |
    | name     | 指标名，例如 accuracy / response_length_ok / user_feedback         |
    | value    | 数值或字符串；配合 data_type 决定语义                              |
    | comment  | 说明 / 理由，排查时最关键的一列                                    |

    data_type：NUMERIC（默认，0~1 或任意数值）、BOOLEAN、CATEGORICAL（类别型，
    如 thumb_up / thumb_down）、TEXT、CORRECTION（人工订正后的标准答案）。
    看板上按 name 聚合，就能看到「哪个版本的 accuracy 掉了」。

安装：uv add langfuse

课案出处：Agent 课案 → 监控与评估 → 评估 → 打分

运行方式：
    uv run Agent/06_langfuse/04_评估_打分_jxsd.py

本机前置条件：settings.langfuse_public_key / langfuse_secret_key 为空，
脚本会先打印中文配置指引，再走不依赖 Langfuse 服务的降级演示：
数据集用本地桩数据、分数以「本应上报的报文」形式打印，Agent 与大模型都是真跑的。

本机实测结论（这三条正是「打分能干什么」的答案）：
    ① 子串匹配打分**只能当玩具**：三条测试项里，第 1 条（期望 "2"）能命中、得 1.0 分，
       第 3 条期望值是一整句标准答案，模型只要换个说法（哪怕答对了）必然拿 0 分 ——
       所以本文件末尾特意把这一点写出来：这类「语义正确但字面不同」的情况必须上 LLM 裁判，
       具体做法见 05_评估_RAG与Agent_jxsd.py 与 06_评估_Agent指标_jxsd.py；
    ② `score_current_trace`（v4 写法）不需要 trace_id：在 @observe 函数里直接调用即可，
       本次两条分数都成功生成报文；
    ③ 类别型分数（如点赞/点踩）必须显式写 `data_type="CATEGORICAL"`：
       按 SDK 的行为，不写会按 NUMERIC 解析字符串值而报错 —— 这是 v4 最容易踩的一个坑。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
from types import SimpleNamespace

# deepagents 造被测 Agent；langfuse 的 CallbackHandler / @observe 负责把分数挂到 trace 上。
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langfuse import Langfuse
from langfuse import get_client
from langfuse import observe as langfuse_observe
from langfuse.langchain import CallbackHandler
from config import settings

# ---------- 0. 客户端初始化：先判断密钥是否就绪 ----------
LANGFUSE_READY = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if LANGFUSE_READY:
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
else:
    langfuse = None

# 本文件要真跑 Agent 才能打分，所以业务模型必须就绪；Langfuse 只决定分数往哪去。
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def _observe_local(*dargs, **_dkwargs):
    """降级版 @observe：不上报，只把原函数原样返回（保持课案的代码形状）。"""
    def _deco(func):
        return func

    return dargs[0] if (dargs and callable(dargs[0])) else _deco


observe = langfuse_observe if LANGFUSE_READY else _observe_local


def report_score(**payload) -> None:
    """上报一条分数。

    真连上 Langfuse → 调 create_score 上传；
    密钥为空     → 把「本应发送的报文」打印出来，让学员看清数据结构。
    """
    if LANGFUSE_READY:
        langfuse.create_score(**payload)
    else:
        print("      [降级] 本应上报的 score 报文：" + _payload_text(payload))


def report_score_current_trace(**payload) -> None:
    """上报到「当前所在的 trace」——课案 04 精简版用的就是这个 v4 API：
    在 @observe 函数里不用手动拿 trace_id，SDK 知道你现在在哪个 span 里。
    """
    if LANGFUSE_READY:
        langfuse.score_current_trace(**payload)
    else:
        print("      [降级] 本应上报到「当前 trace」的 score 报文：" + _payload_text(payload))


def _payload_text(payload: dict, limit: int = 120) -> str:
    """把报文压成一行打印；超长的字符串字段（比如整段回答）截断，否则一行几百字看不清。"""
    short = {}
    for key, value in payload.items():
        if isinstance(value, str) and len(value) > limit:
            short[key] = value[:limit] + "…"
        else:
            short[key] = value
    return json.dumps(short, ensure_ascii=False, default=str)


# ---------- 1. 演示用的工具与 Agent ----------
@tool
def calculator(expression: str) -> str:
    """执行数学计算。传入数学表达式字符串，如 '1+1'"""
    try:
        return str(eval(expression))
    except Exception:
        return "计算错误"


# 课案这一节写的是 create_deep_agent(model=model, tools=[])；
# 这里挂一个计算器，让「1+1等于几」这条测试项真的走一次工具调用。
agent = create_deep_agent(model=llm, tools=[calculator])


# ---------- 2. 构造测试数据集 ----------
# 课案：先 create_dataset 建空数据集（数据集已存在时返回已有的那个），
#       再 create_dataset_item 逐条添加测试项。
DATASET_NAME = "qa_accuracy"

DATASET_ITEMS = [
    {"input": "1+1等于几", "expected_output": "2"},
    {"input": "Python的作者是谁", "expected_output": "Guido van Rossum"},
    # 第三条故意用「一整句长标准答案」当期望值：子串匹配必然判 0 分，
    # 用来说明「精确字符串匹配太严格」——这正是 05 / 06 要上 LLM-as-Judge 的原因
    {"input": "用一句话解释什么是 MCP",
     "expected_output": "Model Context Protocol（模型上下文协议），是 Anthropic 提出的让大模型连接外部工具与数据的开放标准"},
]


def build_dataset_items():
    """返回可迭代的数据集条目。

    真连上 Langfuse → 走课案流程：create_dataset → create_dataset_item → get_dataset；
    密钥为空       → 用 SimpleNamespace 造出与 dataset.items 同样字段的本地桩数据，
                     这样下面评估循环的代码一个字都不用改。
    """
    if LANGFUSE_READY:
        langfuse.create_dataset(name=DATASET_NAME)
        # 降级时把每一步的报文都打出来，学员照着就能在 UI 上手工复现同一套数据集。
        for item in DATASET_ITEMS:
            langfuse.create_dataset_item(dataset_name=DATASET_NAME, **item)
        dataset = langfuse.get_dataset(DATASET_NAME)
        return dataset.items

    print(f"[降级] 本应发送的 create_dataset 报文：{json.dumps({'name': DATASET_NAME}, ensure_ascii=False)}")
    for item in DATASET_ITEMS:
        payload = {"dataset_name": DATASET_NAME, **item}
        print("[降级] 本应发送的 create_dataset_item 报文："
              + json.dumps(payload, ensure_ascii=False))
# SimpleNamespace 的字段名与 dataset.items 一致，所以下面评估循环一行都不用改。
    return [SimpleNamespace(**item) for item in DATASET_ITEMS]   # 字段名与 dataset.items 一致


# ---------- 3. 评估循环：跑数据集 → 打分 → 上报 ----------
def demo_dataset_scoring():
    print("\n" + "=" * 72)
    print("一、数据集自动打分：output 里是否包含 expected_output")
    print("=" * 72)

    items = build_dataset_items()
    print()

    for item in items:
        if LANGFUSE_READY:
            langfuse_handler = CallbackHandler()
            # 逐条跑 Agent：真连 Langfuse 时用 handler.last_trace_id，降级时伪造一个可预测的 id。
            output = agent.invoke(
                {"messages": [{"role": "user", "content": item.input}]},
                config={"callbacks": [langfuse_handler]},
            )
            actual = output["messages"][-1].content
            tid = langfuse_handler.last_trace_id          # 这一轮在 Langfuse 里的 trace id
        else:
            output = agent.invoke({"messages": [{"role": "user", "content": item.input}]})
            actual = output["messages"][-1].content
            tid = f"trace-{abs(hash(item.input)) % 100000:05d}"   # 本地伪造一个 id，方便看报文

        # 最朴素的自动打分：期望值是不是回答的子串
        score = 1.0 if str(item.expected_output) in actual else 0.0

            # 打分与上报分开：先算出分数（纯逻辑，好测），再决定是上传还是打印。
        report_score(
            trace_id=tid,
            name="accuracy",
            value=score,
            comment=actual,
        )
        print(f"Q: {item.input} → A: {actual[:40]}... | 得分: {score}   trace={tid}")

    if LANGFUSE_READY:
        langfuse.flush()
    print("\n注意第三条：期望值是一整句标准答案，子串匹配只有模型一字不差复述时才给分；")
    print("模型只要换个说法（哪怕答对了）就拿 0 分 —— 这就是最朴素的自动打分的天花板。")
    print("这类「语义正确但字面不同」的情况需要 LLM 当裁判 —— 见 05 / 06 两个文件。")


# ---------- 4. 在 @observe 里打分：不用手动拿 trace_id ----------
@observe
def answer_with_length_score(question: str) -> str:
    """业务指标打分示例：回复长度是否落在合理区间。"""
    response = llm.invoke(question)
    length = len(response.content)
    report_score_current_trace(
        # score_current_trace 不需要 trace_id：SDK 知道当前代码跑在哪个 span 里。
        name="response_length_ok",
        value=1.0 if 20 <= length <= 200 else 0.0,
        comment=f"回复长度 {length} 字",
    )
    return response.content


@observe
def answer_with_user_feedback(question: str) -> str:
    """用户反馈打分示例：前端点「踩」之后，回调里把反馈上报成类别型分数。"""
    response = llm.invoke(question)
    feedback = "thumb_down"           # 实际来自前端用户的点赞/点踩按钮
    report_score_current_trace(
        name="user_feedback",
        value=feedback,               # 类别型分数直接传字符串
        data_type="CATEGORICAL",      # 不写明 data_type 会被当成 NUMERIC 解析而报错
        comment="用户点了踩：回答太啰嗦",
    )
    return response.content


# 两个业务指标示例：一个数值型（回复长度是否合理）、一个类别型（用户点赞/点踩）。
def demo_observe_scoring():
    print("\n" + "=" * 72)
    print("二、在 @observe 函数里打分：score_current_trace（v4 写法，免 trace_id）")
    print("=" * 72)
    answer_with_length_score("什么是 Function Call？一句话回答")
    print("  业务指标分数已上报（response_length_ok）")
    answer_with_user_feedback("什么是 MCP？一句话回答")
    print("  用户反馈已上报（user_feedback = thumb_down，CATEGORICAL）")
    # 密钥就绪时再 flush 一次，确保刚才两条分数都发出去了。
    if LANGFUSE_READY:
        get_client().flush()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    if not LANGFUSE_READY:
        # 起始检查：先讲清怎么把 Langfuse 配起来，再走降级演示。
        print("=" * 72)
        print("【进入降级演示】Langfuse 密钥为空")
        print("  settings.langfuse_public_key = '' ，settings.langfuse_secret_key = ''")
        print("-" * 72)
        # 这里只讲「缺什么、怎么补」，不重复课案正文；完整的安装说明见 01_追踪_jxsd.py 的文件头。
        print("要看到真实的数据集与分数看板，按课案「安装」一节准备环境：")
        print("  1) git clone https://github.com/langfuse/langfuse.git")
        print("     cd langfuse")
        print("     docker compose up -d          # 启动后访问 http://localhost:3000")
        print("  2) 首次注册的账号即为管理员；新建项目 → Settings → API Keys → 创建密钥")
        print("  3) 写入 F:\\ProGram\\Python_Base\\.env ：")
        print("        LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("        LANGFUSE_SECRET_KEY=sk-lf-...")
        print("        LANGFUSE_HOST=http://localhost:3000    # 本地 docker 部署用这个")
        print("  4) 重新运行本脚本，去 Datasets 看数据集、去 Scores 看分数")
        print("-" * 72)
        # 数据集换成同字段的本地桩数据，所以下面评估循环的代码一个字都不用改。
        print("下面不依赖 Langfuse 服务：数据集换成同字段的本地桩数据，")
        print("Agent 与大模型都真跑，只把「本应上报的分数报文」打印出来。")
        print("=" * 72)
    else:
        print("Langfuse 已配置，分数将上传到：", settings.langfuse_host)

    # 两节依次演示：数据集自动打分 → 在 @observe 里上报业务指标与用户反馈。
    demo_dataset_scoring()
    demo_observe_scoring()

    print("\n小结：打分就是把「主观好坏」变成可筛选的字段 ——")
    print("      trace_id 定位哪一次调用，name 决定看板怎么聚合，comment 留下判断依据。")
