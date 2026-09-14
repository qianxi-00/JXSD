# -*- coding: utf-8 -*-
"""
Langfuse ①：可观测性之「追踪（Tracing）」——课案完整版
================================================================
课案开头的一句话把定位说清楚了：
    LangSmith 是 LangChain 官方提供的智能体调试、监控和评估平台，收费、不开源；
    Langfuse 是开源的、同样功能的平台，本节只讲 Langfuse。

Langfuse 是开源的 LLM 可观测性平台，提供四大能力：
    Tracing（追踪）      —— 每一步做了什么（本文件）
    Monitoring（监控）   —— 成本 / 延迟 / 错误率看板（数据来源就是追踪）
    Evaluation（评估）   —— 回答好不好（04 / 05 / 06 三个文件）
    Prompt Management    —— 提示词托管（03 文件）

Trace（追踪）到底记录了什么？这是本文件要建立的直觉：
    每次模型调用：输入 / 输出 / token 数 / 耗时 / 成本
    每次工具调用：工具名 / 参数 / 返回值
    以及上面这些步骤之间「谁包着谁」的调用链路树（trace tree）

安装 Langfuse 服务端（课案「安装」一节，需要 Docker）：
    git clone https://github.com/langfuse/langfuse.git
    cd langfuse
    docker compose up -d
    启动后访问 http://localhost:3000，首次注册账号即为管理员。

安装 Python SDK：
    uv add langfuse

课案里的三种埋点方式（本文件逐一演示）：
    | 方式                       | 适用对象                              | 自动化程度 |
    |---------------------------|--------------------------------------|-----------|
    | CallbackHandler           | LangChain / LangGraph / DeepAgents    | 全自动     |
    | @observe 装饰器            | 任意普通 Python 函数                  | 一行       |
    | start_as_current_observation | 需要手工控制 span 边界的代码        | 手工       |

课案出处：Agent 课案 → 监控与评估 → 可观测性 → 追踪
         （课案里 Langfuse 的安装、API key 设置、查看追踪链路三张截图就对应本节）

运行方式：
    uv run Agent/06_langfuse/01_追踪_jxsd.py

本机前置条件：settings.langfuse_public_key / langfuse_secret_key 当前为空，
所以脚本会先打印中文配置指引，然后走一条**不依赖 Langfuse 服务**的降级演示：
用 LangChain 自带回调自己收集一遍调用树并打印出来 —— 看到的就是
Langfuse 会收到的那批字段（名字 / 输入 / 输出 / 耗时 / 层级）。

本机实测结论：
    ① 三种埋点方式都能跑通；降级时打印出来的 trace 树与 Langfuse UI 里的形状一致
       （chain → tool → generation 的层级，工具节点还带「入参 / 出参」两行）；
    ② 手工 span 的耗时是真的：本次 rag_retrieve 0.20s（里面只有 time.sleep(0.2)），
       answer_with_context 6.83s（里面是一次真的大模型调用）——
       这就是「非模型调用也要圈出边界」的意义：不圈就没有耗时，看板上查不到瓶颈；
    ③ 本文件没设 session_id / user_id，所以后台只能按单条 trace 看。
       要把多轮对话串成一个会话，见 02_会话和用户_jxsd.py。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import time
from contextlib import contextmanager
from types import SimpleNamespace

# 第三方依赖：deepagents 造 Agent；langfuse 提供 @observe 与 LangChain 回调处理器。
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langfuse import Langfuse
from langfuse import get_client
from langfuse import observe as langfuse_observe
from langfuse.langchain import CallbackHandler
from config import settings

# ---------- 0. 客户端初始化：先判断密钥是否就绪 ----------
# 课案原文是直接 new 一个客户端：
#     langfuse = Langfuse(public_key=..., secret_key=..., host=...)
# 但两个 key 为空时，SDK 会往 stderr 刷 "Authentication error: ...
# Client will be disabled."，虽然不抛异常，但教学输出会很脏。
# 所以这里先算一个开关，空密钥时干脆不实例化真客户端。
LANGFUSE_READY = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if LANGFUSE_READY:
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,          # 云端 https://cloud.langfuse.com；自建 http://localhost:3000
    )
else:
    langfuse = None                            # 降级演示不需要客户端

# 项目统一用 init_chat_model 初始化大模型（课案原文是 ChatOpenAI 直连，
# 参数同样来自 settings，本项目按 CONVENTIONS 第 3 节统一走 init_chat_model）
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def _observe_local(*dargs, **_dkwargs):
    """降级版 @observe：不做任何埋点，仅把原函数原样返回。

    保留这个壳子，是为了让下面的业务函数**保持课案里的代码形状**
    （函数上方照样写着 @observe），学员对照课案时不会觉得写法变了。
    """
    def _deco(func):
        return func

    return dargs[0] if (dargs and callable(dargs[0])) else _deco


# 密钥就绪 → 用 Langfuse 官方装饰器；否则 → 本地空壳
observe = langfuse_observe if LANGFUSE_READY else _observe_local


# ---------- 1. 本地「降级版 Langfuse」：自己收集调用树 ----------
class LocalTraceCollector(BaseCallbackHandler):
    """监听 LangChain 回调，把整条调用链收集成一棵树并打印出来。

    它顶替的就是 Langfuse 的 CallbackHandler。两者监听的是同一批回调事件，
    区别只是：Langfuse 把事件转成 OpenTelemetry span 上传到服务端，
    这里只是攒在内存里打印 —— 所以打印出来的字段，就是「本应上报的数据」。

    事件与 Langfuse 里 observation 类型的对应关系：
        | LangChain 回调   | Langfuse observation 类型 |
        |-----------------|--------------------------|
        | on_chain_*      | span / chain              |
        | on_tool_*       | tool                      |
        | on_llm_*        | generation（带 token 与成本） |
    """

    def __init__(self, trace_name: str):
        super().__init__()
        self.trace_name = trace_name
        self.nodes: dict[str, dict] = {}      # run_id → 节点
        self.roots: list[str] = []            # 没有父节点的根 run

    # ---- 内部工具：登记一个节点的开始 / 结束 ----
    def _begin(self, kind: str, name: str, run_id, parent_run_id, payload=None):
        key = str(run_id)
        parent = str(parent_run_id) if parent_run_id else None
        # 登记节点时顺手记开始时间，_end 里再算耗时 —— 这就是 Langfuse 算 latency 的做法。
        self.nodes[key] = {
            "kind": kind, "name": name, "parent": parent,
            "input": payload, "output": None, "elapsed": None,
            "t0": time.perf_counter(), "children": [],
        }
        if parent and parent in self.nodes:
            self.nodes[parent]["children"].append(key)   # 挂到父节点下面，形成树
        else:
            self.roots.append(key)                       # 否则它就是一条 trace 的根

    def _end(self, run_id, output=None):
        node = self.nodes.get(str(run_id))
        # 结束时补上输出与耗时；找不到节点说明配对事件丢了，静默忽略即可。
        if node is not None:
            node["elapsed"] = time.perf_counter() - node["t0"]
            node["output"] = output

    @staticmethod
    def _node_name(serialized, metadata, kwargs, prefer_serialized=False) -> str:
        """给节点起个可读的名字。

        LangGraph / DeepAgents 里 serialized 常常是 None，真正有用的是
        kwargs['name']（图节点名）和 metadata['langgraph_node']（图节点名）；
        但工具的 serialized 里带着真正的工具名（calculator），要优先用它，
        否则树上只会看到 LangGraph 的节点名 tools。
        """
        meta = metadata or {}
        candidates = [(serialized or {}).get("name"), kwargs.get("name"), meta.get("langgraph_node")]
        if not prefer_serialized:
            candidates = [kwargs.get("name"), meta.get("langgraph_node"), (serialized or {}).get("name")]
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return "chain"

    # ---- LangChain 回调钩子：链 / 工具 / 大模型 ----
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                       metadata=None, **kwargs):
        self._begin("chain", self._node_name(serialized, metadata, kwargs),
                    run_id, parent_run_id, payload=inputs)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id, outputs)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None,
                      metadata=None, **kwargs):
        self._begin("tool", self._node_name(serialized, metadata, kwargs, prefer_serialized=True),
                    # 工具回调要多传 prefer_serialized=True —— 只有它的 serialized 里带真正的工具名。
                    run_id, parent_run_id, payload=input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id, output)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None,
                     metadata=None, **kwargs):
        self._begin("generation", self._node_name(serialized, metadata, kwargs),
                    run_id, parent_run_id, payload=prompts)

    def on_llm_end(self, response, *, run_id, **kwargs):
        # token 用量就藏在 LLMResult 里，Langfuse 用它算成本，这里也顺手取出来
        text, usage = None, None
        try:
            text = response.generations[0][0].text or response.generations[0][0].message.content
        except Exception:
            pass
        try:
            raw = response.llm_output.get("token_usage") or {}
            # 只留三个关键数字，明细（缓存命中/推理 token）太长，塞进一行会看不清
            usage = {k: raw.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                     if raw.get(k) is not None} or None
        except Exception:
            pass
        node = self.nodes.get(str(run_id))
        if node is not None:
            node["usage"] = usage
        self._end(run_id, text)

    # ---- 打印：等价于 Langfuse Web UI 里那棵 trace 树 ----
    def print_tree(self):
        print(f"trace  name={self.trace_name}")
        for key in self.roots:
            self._print_node(key, depth=1)

    # 递归打印：竖线和 ├─ 只是为了让终端里的层级一眼可读。
    def _print_node(self, key: str, depth: int):
        node = self.nodes[key]
        pad = "│  " * (depth - 1) + "├─ "
        elapsed = f"{node['elapsed']:.2f}s" if node["elapsed"] is not None else "-"
        line = f"{pad}[{node['kind']:<10}] {node['name']:<34} {elapsed:>7}"
        if node.get("usage"):
            line += "  tokens=" + "/".join(str(v) for v in node["usage"].values())
        # 递归打印子节点：深度 +1，缩进跟着加，终端里就能看出「谁包着谁」。
        print(line)
        if node["kind"] == "tool":
            print(f"{'│  ' * depth}   入参: {_short(node['input'])}")
            print(f"{'│  ' * depth}   出参: {_short(node['output'])}")
        for child in node["children"]:
            self._print_node(child, depth + 1)

    def as_ingestion_payload(self) -> dict:
        """把收集到的事件拼成「本应上传给 Langfuse 的报文」。

        Langfuse 服务端最终存的就是这种结构：一条 trace + 若干条 observation，
        每条 observation 带 name / type / input / output / start_time / end_time。
        """
        observations = []
        # 逐个事件转成 observation —— 这正是 Langfuse ingestion API 期望的形状。
        for key, node in self.nodes.items():
            observations.append({
                "id": key,
                "traceId": "<由 SDK 生成的 trace_id>",
                "parentObservationId": node["parent"],
                "type": node["kind"].upper(),
                # 顶层是 trace + observations 两段，正是 Langfuse ingestion API 的请求体形状。
                "name": node["name"],
                "input": _short(node["input"], 120),
                "output": _short(node["output"], 120),
                "usage": node.get("usage"),
                "durationSeconds": round(node["elapsed"], 3) if node["elapsed"] else None,
            })
        return {"trace": {"name": self.trace_name}, "observations": observations}


def _short(value, limit: int = 60) -> str:
    """把任意对象压成一行短文本，方便塞进追踪节点里看。"""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


@contextmanager
def manual_span(name: str, **fields):
    """手工 span：课案第三种埋点方式。

    密钥就绪时用官方 API（v4 里叫 start_as_current_observation）：
        with langfuse.start_as_current_observation(name=name, as_type="span") as sp:
            ...业务代码...          # 期间发生的 LLM / 工具调用会自动挂到这个 span 下面
    密钥缺失时退化成「本地计时 + 打印」，保证脚本照样跑得下去。
    """
    if LANGFUSE_READY:
        with langfuse.start_as_current_observation(name=name, as_type="span", input=fields or None) as sp:
            yield sp
    else:
        t0 = time.perf_counter()
        holder = SimpleNamespace(update=lambda **kw: None)   # 空壳，保持调用形状一致
        print(f"  [降级] 进入手工 span：{name}  metadata={_short(fields)}")
        try:
            yield holder
        finally:
            print(f"  [降级] 退出手工 span：{name}  耗时 {time.perf_counter() - t0:.2f}s")


# ---------- 2. 定义演示用的工具与 Agent ----------
# 课案这一节用的是 create_deep_agent(model=model, tools=[])，
# 为了让追踪树里出现 tool 节点，这里挂一个计算器工具，只提问算术题。
@tool
def calculator(expression: str) -> str:
    """执行数学计算。传入数学表达式字符串，如 '5*3+9'"""
    try:
        return str(eval(expression))       # 教学演示，表达式由模型生成，实际项目要换成安全求值
    except Exception:
        return "计算错误"


agent = create_deep_agent(model=llm, tools=[calculator])


# ---------- 3. 方式一：CallbackHandler（LangChain 生态一把梭） ----------
def demo_callback_handler():
    """课案主推方式：给 invoke 的 config 挂一个回调处理器，整条链路自动埋点。"""
    print("\n" + "=" * 72)
    print("方式一：CallbackHandler —— config={'callbacks': [langfuse_handler]}")
    print("=" * 72)

    if LANGFUSE_READY:
        # CallbackHandler 会自动把 LangChain 的每次链/工具/模型调用
        # 转成 Langfuse 的 observation，并串成一条 trace
        langfuse_handler = CallbackHandler()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "帮我算 5*3+9"}]},
            config={"callbacks": [langfuse_handler]},
        )
        print("AI：", result["messages"][-1].content)
        # last_trace_id 就是这次调用在 Langfuse 里的 trace id，后面打分要用它
        print("本次 trace_id：", langfuse_handler.last_trace_id)
        get_client().flush()      # 立刻上传（SDK 默认批量异步刷，不 flush 可能等几秒）
    else:
        collector = LocalTraceCollector("Langfuse 追踪演示")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "帮我算 5*3+9"}]},
            config={"callbacks": [collector]},
        )
        # 上面这行的形状和课案完全一致，只是把 langfuse_handler 换成了本地收集器
        print("AI：", result["messages"][-1].content)
        print("\n[降级] 下面是本次运行收集到的 trace 树（= Langfuse Web UI 里的样子）：")
        collector.print_tree()
        print("\n[降级] 本应上传给 Langfuse 的报文（节选）：")
        payload = collector.as_ingestion_payload()
        print(json.dumps(payload["observations"][:2], ensure_ascii=False, indent=2))


# ---------- 4. 方式二：@observe 装饰器（任意 Python 函数） ----------
@observe
def generate_answer(question: str) -> str:
    """普通业务函数，加个装饰器就有完整追踪。

    @observe 会为每次调用生成一个 span；如果函数内部又调用被 @observe
    装饰的函数，它们会自动嵌套成父子关系。
    """
    return llm.invoke(question).content


# 函数体只有一行，但 @observe 会为它生成 span；内部再调被装饰的函数会自动嵌套成父子。
def demo_observe():
    print("\n" + "=" * 72)
    print("方式二：@observe 装饰器 —— 不限于 LangChain，任何 Python 函数都能追踪")
    print("=" * 72)
    answer = generate_answer("一句话解释什么是 Langfuse")
    print("AI：", answer)

    if LANGFUSE_READY:
        # 当前 span 的 trace_id，函数内部随时可查
        print("当前 trace_id：", get_client().get_current_trace_id())
        get_client().flush()
    else:
        print("  [降级] @observe 变成了空壳（不上报），本次要上报的数据形如：")
        print(json.dumps({
            "type": "span",
            "name": "generate_answer",
            "input": {"question": "一句话解释什么是 Langfuse"},
            "output": answer,
            "metadata": {"sdk": "langfuse", "host": settings.langfuse_host},
        }, ensure_ascii=False, indent=2))

# 降级时把「本应上报的 span」直接打出来，字段名与 Langfuse 一致，方便对照。

# ---------- 5. 方式三：手工 span（细粒度控制） ----------
def demo_manual_span():
    """当一步操作不是「一次模型调用」时（比如检索、rerank、后处理），
    用 span 手工圈出边界，Langfuse 才能算出这一步的耗时。"""
    print("\n" + "=" * 72)
    print("方式三：手工 span —— 给「非模型调用」的步骤也加上耗时统计")
    print("=" * 72)
    with manual_span("rag_retrieve", 检索库="教学知识库", top_k=3):
        time.sleep(0.2)                                   # 假装在做向量检索
        docs = ["Langfuse 是开源的 LLM 可观测性平台。", "它支持追踪、监控、评估、提示词管理。"]
    print("  检索到", len(docs), "条文档")
    with manual_span("answer_with_context", 文档数=len(docs)):
        answer = llm.invoke("根据资料一句话说明 Langfuse 的能力：\n" + "\n".join(docs))
    print("AI：", answer.content)
    if LANGFUSE_READY:
        get_client().flush()


# ---------- 6. 主流程 ----------
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    if not LANGFUSE_READY:
        print("=" * 72)
        print("【进入降级演示】Langfuse 密钥为空")
        print("  settings.langfuse_public_key = '' ，settings.langfuse_secret_key = ''")
        print("-" * 72)
        print("要看到真实的追踪链路，按课案「安装」一节准备环境：")
        # 缺 Langfuse 只影响「数据往哪去」，不影响脚本能否跑 —— 先讲清怎么补，再走降级演示。
        # 四步就是课案「安装」一节的原文顺序；配完密钥再跑本脚本，追踪数据才会真的上传。
        print("  1) git clone https://github.com/langfuse/langfuse.git")
        print("     cd langfuse")
        print("     docker compose up -d          # 启动后访问 http://localhost:3000")
        print("  2) 首次注册的账号即为管理员；新建项目 → Settings → API Keys → 创建密钥，")
        print("     复制 Public Key（pk-lf-...）与 Secret Key（sk-lf-...）")
        print("  3) 写入 F:\\ProGram\\Python_Base\\.env ：")
        print("        LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("        LANGFUSE_SECRET_KEY=sk-lf-...")
        print("        LANGFUSE_HOST=http://localhost:3000    # 本地 docker 部署用这个")
        print("  4) 重新运行本脚本，追踪数据就会出现在 Langfuse 的 Tracing 页面")
        print("-" * 72)
        print("下面不依赖 Langfuse 服务，照样把 Agent 跑起来，")
        print("并把「本应上报给 Langfuse 的数据」打印出来 —— 对照着看数据长什么样。")
        print("=" * 72)
    else:
        print("Langfuse 已配置，追踪数据将上传到：", settings.langfuse_host)

    # 三种埋点方式依次演示：CallbackHandler（全自动）→ @observe（一行）→ 手工 span（细粒度）。
    demo_callback_handler()
    demo_observe()
    demo_manual_span()

    if LANGFUSE_READY:
        get_client().flush()
        # 密钥就绪时再 flush 一次兜底，确保退出前数据都发出去了。
        print("\n已上传，去控制台查看：", settings.langfuse_host)
    else:
        print("\n[降级] 演示结束：以上 trace 树与报文都没有上传，因为密钥未配置。")
