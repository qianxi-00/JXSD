# -*- coding: utf-8 -*-
"""
部署 ⑤：给部署好的 Agent 接上 Langfuse 追踪
================================================================
Agent 上生产之后必须能回答三个问题：它为什么这么答？花了多少 token？慢在哪一步？
这就是可观测（observability）。Aegra 的追踪走 **OpenTelemetry + Langfuse**。

课案给了**两条接入路径**，水平和代价都不一样，别混为一谈：

  路径 A（.env 四行，**零代码改动**）—— 推荐
      OTEL_TARGETS="LANGFUSE"
      LANGFUSE_BASE_URL="http://localhost:3000"   # 自托管 Langfuse 地址
      LANGFUSE_PUBLIC_KEY="pk-lf-xxx"
      LANGFUSE_SECRET_KEY="sk-lf-xxx"
      原理：Aegra 服务自己内置了 OTEL 导出器，读到 OTEL_TARGETS=LANGFUSE 就把
            服务端的 span（一次 run、一次 graph step）推到 Langfuse。
            代码一个字不用改，改完 .env 重启服务即可生效。

  路径 B（graph.py 里手工挂 CallbackHandler）—— 粒度更细
      langfuse = Langfuse(public_key=..., secret_key=..., host=...)
      langfuse_handler = CallbackHandler()
      graph = create_deep_agent(model=model, tools=[], config={"callbacks": [langfuse_handler]})
      原理：LangChain 在执行过程中会到处「喊事件」（chain 开始/结束、LLM 开始/结束、
            工具开始/结束……），CallbackHandler 就是那个一直在旁边听的人，
            它把听到的事件翻译成 OTel span，再发给 Langfuse。
            好处是能看到 LangChain **内部**的层次（哪条链、哪次 LLM、哪个工具），
            代价是代码里多几行、key 要自己传。

⚠️ 容器里的 localhost 陷阱（课案 docker-compose 里专门写了注释）：
      # 启用 Langfuse 追踪；容器内 localhost 指向自身，需走宿主机网关访问 langfuse
      - LANGFUSE_BASE_URL=http://host.docker.internal:3000
   Langfuse 跑在宿主机上，Agent 跑在容器里 —— 容器里的 "localhost" 是**容器自己**，
   不是宿主机。所以必须写 host.docker.internal（Docker Desktop 提供的宿主机别名）。
   这是自托管 Langfuse + 容器化 Agent 最常见的「配了但收不到数据」的原因。

本文件的运行策略：
  1. 从 settings.langfuse_public_key / langfuse_secret_key / langfuse_host 读配置；
  2. 密钥**当前为空**（这是本仓库的实际情况）→ 打印中文提示怎么配，
     然后走降级演示：`langfuse.langchain.CallbackHandler` 用不了，就手写一个
     「打印型 callback handler」，**真的调一次大模型**，把回调事件原样打出来 ——
     看到这些事件，你就知道 Langfuse 上那条 trace 是由什么拼出来的；
  3. 密钥配好了的话，本文件会自动切到真实 CallbackHandler 路径。

课案出处：Agent 课案 → 部署 → 对接Langfuse

本节要讲什么：

    1. 【为什么需要】Agent 上生产后必须能回答「它为什么这么答 / 花了多少 token /
       慢在哪一步」—— 这就是可观测（observability），Aegra 走 OTEL + Langfuse；
    2. 【路径 A】只改 .env 四行、代码零改动：OTEL_TARGETS 是开关，
       另外三个是「地址 + 两把钥匙」，缺一个都推不上去；
    3. 【路径 B】在 graph.py 里手工挂 langfuse 的 CallbackHandler，
       粒度更细（能看到 LangChain 内部的链 / 每次 LLM / 每个工具），代价是多几行代码；
    4. 【原理】LangChain 的「喊事件」机制 + BaseCallbackHandler 的 on_xxx 方法，
       以及 run_id / parent_run_id 是怎么还原出整棵调用树的；
    5. 【容器陷阱】容器里的 localhost 是容器自己 —— 自托管 Langfuse 必须写
       host.docker.internal，这是「配了但收不到数据」最常见的原因；
    6. 【这 8 个依赖】requirements.txt 里每个包各自负责什么；
    7. 【降级演示】没有 Langfuse 密钥时，用一个「打印型」handler 真的调一次模型，
       把回调事件原样打出来 —— 看清 Langfuse 上那条 trace 是由什么拼出来的。

本文件的运行策略：
  1. 从 settings.langfuse_public_key / langfuse_secret_key / langfuse_host 读配置；
  2. 密钥**当前为空**（这是本仓库的实际情况）→ 打印中文提示怎么配，
     然后走降级演示：`langfuse.langchain.CallbackHandler` 用不了，就手写一个
     「打印型 callback handler」，**真的调一次大模型**，把回调事件原样打出来 ——
     看到这些事件，你就知道 Langfuse 上那条 trace 是由什么拼出来的；
  3. 密钥配好了的话，本文件会自动切到真实 CallbackHandler 路径。

运行前置条件：
    · settings.api_key / model_name / base_url 可用（本机实测可用，模型 grok-4.6）——
      第 5 节的降级演示本身就要真调一次模型；
    · Langfuse 密钥可缺：缺了走降级，不报错；配了就自动走真实 CallbackHandler。

运行方式：
    uv run Agent/09_aegra_deploy/05_对接Langfuse_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import unicodedata

# 下面三个模块各自扮演什么角色：
#   init_chat_model      —— 按 provider 建模型（本项目统一写法，替代课案的 ChatOpenAI 直连）
#   BaseCallbackHandler  —— 第 5 节那个「打印型」handler 的基类；它也是理解
#                           Langfuse 的 CallbackHandler 的入口：接口完全一样，
#                           差别只在 on_xxx 里「听到事件之后做什么」
#   settings             —— 根目录统一配置（F:\ProGram\Python_Base\config.py + .env）
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from config import settings


# ================================================================
# 小工具：按显示宽度对齐打印表格
# ================================================================
def _disp_width(text: str) -> int:
    """按东亚字符宽度计算字符串在终端里占的列数"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _disp_width(text))


def print_table(title: str, headers: list, rows: list) -> None:
    widths = [
        max(_disp_width(headers[i]), *(_disp_width(str(r[i])) for r in rows))
        for i in range(len(headers))
    ]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(f"\n【{title}】")
    print(line)
    print("| " + " | ".join(_pad(str(headers[i]), widths[i]) for i in range(len(headers))) + " |")
    print(line)
    for row in rows:
        print("| " + " | ".join(_pad(str(row[i]), widths[i]) for i in range(len(row))) + " |")
    print(line)


# ================================================================
# 1. 路径 A：.env 四行，零代码改动
# ================================================================
# 课案原文：
#     `.env` 里加几个变量即可，代码零改动：
#         OTEL_TARGETS="LANGFUSE"
#         LANGFUSE_BASE_URL="http://localhost:3000"   # 自托管 Langfuse 地址
#         LANGFUSE_PUBLIC_KEY="pk-lf-xxx"
#         LANGFUSE_SECRET_KEY="sk-lf-xxx"
ENV_LINES = [
    'OTEL_TARGETS="LANGFUSE"',
    'LANGFUSE_BASE_URL="http://localhost:3000"   # 自托管 Langfuse 地址',
    'LANGFUSE_PUBLIC_KEY="pk-lf-xxx"',
    'LANGFUSE_SECRET_KEY="sk-lf-xxx"',
]


def section_1_env_only() -> None:
    print("=" * 78)
    print("1. 路径 A：.env 里加四行，代码零改动")
    print("=" * 78)
    for ln in ENV_LINES:
        print("    " + ln)
    print("\n  OTEL_TARGETS 是「开关」：Aegra 服务读到 LANGFUSE 就启用 Langfuse 导出器；")
    print("  剩下三个是「地址 + 钥匙」，缺一个都推不上去。")
    print_table(
        "四个变量的作用",
        ["变量", "作用"],
        [
            ["OTEL_TARGETS", "选择 OTEL 导出目标；值 LANGFUSE 即开启 Langfuse 追踪"],
            ["LANGFUSE_BASE_URL", "Langfuse 服务地址（自托管就填自己的，如 http://localhost:3000）"],
            ["LANGFUSE_PUBLIC_KEY", "公钥，pk-lf- 开头，Langfuse 项目设置里生成"],
            ["LANGFUSE_SECRET_KEY", "私钥，sk-lf- 开头 —— **只在本地 .env，不要提交进仓库**"],
        ],
    )
    print("\n  ⚠️ 密钥只写在本地 .env 里，不要提交进仓库（.env 已在 .gitignore 中）。")
    print("  ⚠️ 容器里的 localhost 是容器自己 —— 见下面第 3 节那个 host.docker.internal 的坑。")


# ================================================================
# 2. 路径 B：graph.py 里手工挂 CallbackHandler
# ================================================================
# 课案原文（graph.py）：
#     from langfuse import Langfuse
#     from langfuse.langchain import CallbackHandler
#     from langchain_openai import ChatOpenAI
#     from deepagents import create_deep_agent
#
#     from config import setting
#
#     langfuse = Langfuse(
#         public_key=setting.LANGFUSE_PUBLIC_KEY,
#         secret_key=setting.LANGFUSE_SECRET_KEY,
#         host=setting.LANGFUSE_BASE_URL,
#     )
#     langfuse_handler = CallbackHandler()
#
#     model = ChatOpenAI(model=setting.MODEL_NAME, api_key=setting.API_KEY, base_url=setting.BASE_URL)
#
#     graph = create_deep_agent(model=model, tools=[], config={"callbacks": [langfuse_handler]})
#
# 这里把课案的 `from config import setting` / `setting.XXX` 统一改成
# `from config import settings` / `settings.xxx`（配置规范第 3 节）。
GRAPH_PY_LANGFUSE = '''from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from config import settings

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)
langfuse_handler = CallbackHandler()

model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

graph = create_deep_agent(model=model, tools=[], config={"callbacks": [langfuse_handler]})
'''

CALLBACK_PRINCIPLE = [
    ("LangChain 在执行时会「喊事件」",
     "每进入一个 Runnable 就喊 on_chain_start，调用模型喊 on_chat_model_start、"
     "模型返回喊 on_llm_end，工具调用喊 on_tool_start / on_tool_end …… "
     "这些喊话是 LangChain 内建的机制，不需要你写任何埋点代码。"),
    ("CallbackHandler 就是「听事件的人」",
     "它继承自 langchain_core.callbacks.BaseCallbackHandler，把 on_xxx 方法实现一遍即可。"
     "Langfuse 的 CallbackHandler 做的事就是：每听到一个事件，就开/关一个 OTel span，"
     "把输入输出、token 数、耗时挂上去，再批量发给 Langfuse。"),
    ("怎么让 handler 听到：config={\"callbacks\": [...]}",
     "把 handler 放进 config 里，LangChain 会在**整棵调用树**上传播它 —— "
     "所以 create_deep_agent(...) 里挂一次，规划、子 Agent、工具调用全都会被记录，"
     "不用逐个节点挂。这是它比手工埋点省事的地方。"),
    ("它的信号源和路径 A 不一样",
     "路径 A 记的是「服务端视角」（一次 run / 一次 graph step）；"
     "路径 B 记的是「LangChain 内部视角」（每一条链、每一次 LLM、每一个工具）。"
     "两条路可以同时开，在 Langfuse 上会看到粗细两级的 span 拼成同一棵 trace。"),
]


def section_2_graph_py() -> None:
    print("\n" + "=" * 78)
    print("2. 路径 B：graph.py 里手工挂 CallbackHandler（逐行讲解）")
    print("=" * 78)
    for ln in GRAPH_PY_LANGFUSE.rstrip("\n").splitlines():
        print("    " + ln)

    print("\n  ◆ CallbackHandler 的原理（四步）")
    for i, (title, detail) in enumerate(CALLBACK_PRINCIPLE, 1):
        print(f"\n    {i}. {title}")
        text = detail
        while text:
            print("       " + text[:68])
            text = text[68:]


# ================================================================
# 3. 容器里的 localhost 陷阱
# ================================================================
# 课案 docker-compose.yml 原文（这几行里最后两行是重点）：
#     environment:
#       ...
#       # 启用 Langfuse 追踪；容器内 localhost 指向自身，需走宿主机网关访问 langfuse
#       - OTEL_TARGETS=LANGFUSE
#       - LANGFUSE_BASE_URL=http://host.docker.internal:3000
def section_3_docker_localhost() -> None:
    print("\n" + "=" * 78)
    print("3. 容器里的 localhost 陷阱（课案 docker-compose 的注释）")
    print("=" * 78)
    print("  课案 docker-compose.yml 里这几行：")
    print("      environment:")
    print("        # 启用 Langfuse 追踪；容器内 localhost 指向自身，需走宿主机网关访问 langfuse")
    print("        - OTEL_TARGETS=LANGFUSE")
    print("        - LANGFUSE_BASE_URL=http://host.docker.internal:3000")
    print("\n  为什么：Langfuse 跑在宿主机，Agent 跑在容器里。")
    print("          容器内的 localhost = **容器自己**，不是宿主机 ——")
    print("          填 http://localhost:3000 的话，容器会去找自己身上根本不存在的 Langfuse，")
    print("          表现是「配置看起来都对，但 Langfuse 上一条数据都没有」。")
    print("          host.docker.internal 是 Docker Desktop 给的宿主机别名，走这个才通。")
    print("\n  另外注意 .env 里的 LANGFUSE_BASE_URL 会被 compose 的 environment **覆盖**：")
    print("      同一个键，environment 优先级高于 env_file。")
    print("      所以本地 .env 填 localhost（宿主机直接跑时用），")
    print("      compose 里再改成 host.docker.internal（容器里跑时用），两边互不干扰。")


# ================================================================
# 4. requirements.txt 那 8 个包，各自干什么
# ================================================================
# 课案原文 requirements.txt：
#     aegra-cli
#     aegra-api
#     langgraph
#     langchain
#     langchain-openai
#     deepagents
#     langfuse
#     pydantic-settings
REQUIREMENTS = [
    "aegra-cli",
    "aegra-api",
    "langgraph",
    "langchain",
    "langchain-openai",
    "deepagents",
    "langfuse",
    "pydantic-settings",
]

REQUIREMENTS_DOC = {
    "aegra-cli": "命令行工具：aegra init / dev / up / down / serve。只在你敲命令时需要（生产镜像里其实可以不要，但留着方便进容器排查）",
    "aegra-api": "服务本体：用 FastAPI 实现的那套 Agent Protocol（threads / runs / assistants / store）",
    "langgraph": "图引擎（MIT 开源）：StateGraph、checkpoint、streaming 都来自它 —— 部署平台收费，它不收费",
    "langchain": "LangChain 主包：Runnable / 回调 / 工具那套公共抽象，也是 langchain-core 的统一入口",
    "langchain-openai": "模型接入层：ChatOpenAI，走 OpenAI 兼容协议（DeepSeek / grok / 通义 都靠它接）",
    "deepagents": "create_deep_agent：课案 graph.py 用它把「规划 + 子 Agent + 工具」组装成一张现成的图",
    "langfuse": "可观测平台 SDK：提供 Langfuse 客户端与 langfuse.langchain.CallbackHandler",
    "pydantic-settings": "从 .env 读配置：课案的 conf.py 就是靠它的 BaseSettings（本项目用根目录 config.py）",
}


def section_4_requirements() -> None:
    print("\n" + "=" * 78)
    print("4. 课案 requirements.txt 那 8 个包，各自干什么")
    print("=" * 78)
    for name in REQUIREMENTS:
        print(f"    {name}")
    print_table(
        "8 个依赖的分工",
        ["包名", "作用"],
        [[name, REQUIREMENTS_DOC[name]] for name in REQUIREMENTS],
    )
    print("\n  一句话：前 2 个是 Aegra 自己的（命令行 + 服务），")
    print("          中间 4 个是你写 Agent 用的，langfuse 是观测，pydantic-settings 是配置。")


# ================================================================
# 5. 降级方案：一个「打印型」CallbackHandler
# ================================================================
class PrintingCallbackHandler(BaseCallbackHandler):
    """把 LangChain 的回调事件直接打印到屏幕上 —— 用来看清「事件长什么样」

    它和 langfuse.langchain.CallbackHandler **是同一个接口**：
    都继承 BaseCallbackHandler，都靠重写 on_xxx 方法来接收事件。
    区别只在于「听到之后做什么」：

        Langfuse 的 handler → 开/关 OTel span，批量发给 Langfuse 服务
        这个 handler        → print 出来，什么也不发

    所以在没有 Langfuse 密钥的环境里，用它来演示「Langfuse 上那条 trace
    是由哪些事件拼出来的」，是完全等价的教学替身。

    实现要点：
      · run_id / parent_run_id 是 LangChain 给每次执行分配的 ID，
        靠 parent 指针就能还原出调用树的层次（下面用缩进表示）；
      · on_llm_new_token 只在**流式**调用时才触发，invoke 不会触发 ——
        这里保留它是为了说明「流式和回调是两套机制」。
    """

    def __init__(self) -> None:
        super().__init__()
        self.events = []        # (事件名, 详情) 供结尾汇总
        self._parents = {}      # run_id -> parent_run_id，用来算缩进层级

    def _record(self, kind: str, run_id, name: str, detail: str = "", parent_run_id=None) -> None:
        """记录并打印一条事件，按父子关系缩进"""
        self._parents[run_id] = parent_run_id
        depth, cur = 0, parent_run_id
        while cur is not None and depth < 10:
            depth += 1
            cur = self._parents.get(cur)
        # 打上 run_id 前 8 位：**同一次执行的 start/end 用的是同一个 run_id**，
        # 这正是 Langfuse 把两个事件合成一个 span 的依据。
        line = f"        {'    ' * depth}└─ [{kind}] {name}  (run={str(run_id)[:8]})"
        if detail:
            line += f"  {detail}"
        print(line)
        self.events.append((kind, detail or name))

    # ---------- 链（Runnable）事件 ----------
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        name = (serialized or {}).get("name") or "Runnable"
        self._record("chain_start", run_id, name, f"输入键={list((inputs or {}).keys())}",
                     parent_run_id)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        keys = list(outputs.keys()) if isinstance(outputs, dict) else type(outputs).__name__
        self._record("chain_end", run_id, "链结束", f"输出={keys}", parent_run_id)

    # ---------- 大模型事件 ----------
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        name = (serialized or {}).get("name") or "ChatModel"
        n = len(messages[0]) if messages else 0
        self._record("chat_model_start", run_id, name, f"发送 {n} 条消息", parent_run_id)

    def on_llm_new_token(self, token, *, run_id, parent_run_id=None, **kwargs):
        # invoke（非流式）不会触发这个事件 —— 只有 stream 才会
        self._record("llm_new_token", run_id, "新 token", repr(token), parent_run_id)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        detail = ""
        try:
            gen = response.generations[0][0]
            text = getattr(getattr(gen, "message", None), "content", None) or getattr(gen, "text", "")
            usage = (response.llm_output or {}).get("token_usage") or {}
            if usage:
                # 只挑三个最关键的字段，整份 usage 里还塞着一堆 None，全打出来没法看
                total = usage.get("total_tokens")
                reason = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                detail = f"total_tokens={total}"
                if reason:
                    detail += f"（其中推理 {reason}）"
                detail += "  "
            detail += f"回答前 20 字={text[:20]!r}"
        except Exception:   # noqa: BLE001 —— 回调里绝不能再抛异常
            detail = "（结构解析失败，跳过）"
        self._record("llm_end", run_id, "模型返回", detail, parent_run_id)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._record("llm_error", run_id, "模型报错", f"{type(error).__name__}", parent_run_id)

    # ---------- 工具事件 ----------
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        name = (serialized or {}).get("name") or "Tool"
        self._record("tool_start", run_id, name, f"参数={input_str[:40]!r}", parent_run_id)

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._record("tool_end", run_id, "工具返回", f"{str(output)[:40]!r}", parent_run_id)

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._record("tool_error", run_id, "工具报错", f"{type(error).__name__}", parent_run_id)


def section_5_keys_and_demo() -> None:
    """读 Langfuse 配置：配好了走真 handler，没配走降级演示"""
    print("\n" + "=" * 78)
    print("5. 本机 Langfuse 配置检查 + 回调事件演示")
    print("=" * 78)

    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    host = settings.langfuse_host
    print(f"  settings.langfuse_public_key : {public_key!r}")
    print(f"  settings.langfuse_secret_key : {'（已设置）' if secret_key else repr(secret_key)}")
    print(f"  settings.langfuse_host       : {host!r}")

    handler = None
    real_langfuse = False
    if public_key and secret_key:
        # 密钥齐全 → 用课案那套真实 CallbackHandler
        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            handler = CallbackHandler()
            real_langfuse = True
            print("\n  ✅ 密钥齐全，使用真实的 langfuse.langchain.CallbackHandler")
        except Exception as exc:   # noqa: BLE001 —— 包缺了/版本不对都降级
            print(f"\n  ⚠️ 初始化 Langfuse 失败（{type(exc).__name__}: {exc}），降级为打印型 handler")

    if handler is None:
        # 密钥为空（本仓库的实际情况）→ 打印中文提示 + 降级。
        # 降级而不是直接退出，是因为本节要教的核心其实是「Langfuse 收到的是什么」，
        # 而这件事用打印型 handler 就能完整演示 —— 没有密钥不该让这一节白跑。
        print("\n  ❌ Langfuse 密钥为空 —— 追踪链路走不通，本次走降级演示（不抛异常）")
        print("\n  怎么配上（两条路任选，细节见本文件第 1 / 2 节）：")
        print("     路径 A（零代码，推荐）：在 aegra_project/.env 里加四行")
        for ln in ENV_LINES:
            print("         " + ln)
        print("         → 然后重启服务（aegra dev / aegra up），不用改任何 .py")
        print("     路径 B（代码级）：把第 2 节那段 graph.py 里的 CallbackHandler 挂到 agent 上")
        print("\n     自托管 Langfuse 的启动（在 Langfuse 自己的仓库目录里）：")
        print("         docker compose up -d      # 默认就是 3000 端口")
        print("     然后在 Langfuse 控制台 → Settings → API Keys 生成 pk-lf- / sk-lf- 一对，")
        print("     填进本地 .env 的 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY。")
        print("\n     ⚠️ 密钥只写在本地 .env，不要提交进仓库、也不要贴进聊天记录。")
        print("\n  ---- 降级：用一个「打印型」CallbackHandler，真的调一次大模型 ----")
        print("      它和 Langfuse 的 handler 接口完全相同，只是把事件 print 出来而不是发出去。\n")
        handler = PrintingCallbackHandler()

    # ---------- 真实调用大模型（这一步会真的发请求）----------
    llm = init_chat_model(
        model_provider="openai",
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    question = "用一句话说明 Langfuse 能观测到 Agent 的哪些东西"
    print(f"  提问：{question}")
    print(f"  模型：{settings.model_name}（{settings.base_url}）\n")
    print("  ---- 回调事件序列（缩进 = 调用树的层次）----")
    try:
        response = llm.invoke(question, config={"callbacks": [handler]})
        print("\n  ---- 事件结束 ----")
        answer = getattr(response, "content", str(response))
        print(f"\n  AI：{answer}")
        print("\n  ◆ 怎么读上面这串事件：")
        print("    · chat_model_start 和 llm_end 的 run= 前缀是**同一个** ——")
        print("      它们描述的是同一次执行的开始与结束，Langfuse 靠 run_id 把两个事件")
        print("      合成一个 generation span（耗时 = 两者时间差，输入输出各挂一头）。")
        print("    · 这里有工具调用的话，会多出 tool_start / tool_end 两层，并用缩进体现父子关系。")
        print("    · ⚠️ 直接 llm.invoke 只会触发模型级事件，**看不到 chain_start/chain_end**；")
        print("      把模型包进链（RunnableSequence）或 Agent 之后，chain 那一层才出现 ——")
        print("      课案 graph.py 里 create_deep_agent 挂 handler，记的就是这一整棵树。")
        if real_langfuse:
            # Langfuse v4：flush() 立即上传，否则要等批量刷新，控制台里可能看不到
            from langfuse import get_client
            get_client().flush()
            print(f"\n  已上传到 Langfuse：{host} → 去控制台看这条 trace")
        else:
            print("\n  （以上事件就是 Langfuse 那条 trace 的原料：")
            print("    chat_model_start → llm_end 这一段会被记成一个 generation span，")
            print("    token 数 / 耗时 / 输入输出都会挂在上面。）")
    except Exception as exc:   # noqa: BLE001 —— 网络/额度问题不该把演示炸掉
        print(f"\n  ❌ 调用模型失败：{type(exc).__name__}: {exc}")
        print("     检查 .env 里的 API_KEY / BASE_URL / MODEL_NAME 是否可用。")
        return

    # 结尾汇总：只有「打印型」handler 会把事件同时存进 `.events`；真实的
    # `langfuse.langchain.CallbackHandler` **没有**这个属性（它把事件直接发去服务端，
    # 本地不留），无条件取会 `AttributeError` —— 实测踩坑，故用 getattr 判空。
    # 事件明细在上面已经边收边打印过了，这里只报个数 + 说明事件与 span 的对应关系。
    events = getattr(handler, "events", None)
    if events is None:
        print("\n  （真实 Langfuse handler 不在本地留事件，去控制台看这条 trace 的 span 树。）")
    else:
        print(f"\n  本次共捕获 {len(events)} 个回调事件")
        print("    · chat_model_start / llm_end 成对出现 → 合成一个 generation span")
        print("    · chain_start / chain_end             → 合成一个 chain span（父节点）")
        print("    · tool_start / tool_end               → 合成一个 tool span")


# ================================================================
# 主流程
# ================================================================
if __name__ == "__main__":
    print("Agent 课案 · 部署 ⑤：对接 Langfuse（.env 零代码 + CallbackHandler）")
    section_1_env_only()
    section_2_graph_py()
    section_3_docker_localhost()
    section_4_requirements()
    section_5_keys_and_demo()
    print("\n小结：追踪有两条路 —— .env 四行（服务端视角，零代码）与")
    print("      graph.py 挂 CallbackHandler（LangChain 内部视角）；")
    print("      容器里访问宿主机的 Langfuse 记得用 host.docker.internal。")
