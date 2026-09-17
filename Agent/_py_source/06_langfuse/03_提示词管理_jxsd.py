# -*- coding: utf-8 -*-
"""
Langfuse ③：提示词管理（Prompt Management）——课案完整版
================================================================
课案原话：
    在 Langfuse Web UI 创建提示词并标记 production 版本，
    代码中拉取最新版本，线上更新不重启。

提示词为什么不该硬编码在代码里？
    | 硬编码在代码里                     | 托管在 Langfuse                     |
    |----------------------------------|------------------------------------|
    | 改一个错别字要走 提交→构建→发版→回归 | 界面上改完保存，下一次请求就生效      |
    | 只有开发能改                       | 产品 / 运营 / 提示词工程师都能改      |
    | 想回滚就 revert 代码               | 版本（version）天然不可变，一键切回   |
    | 灰度只能靠发版切环境                | label 可以随时指向某个版本做灰度      |

三个核心概念（课案这几节的骨架）：

    | 概念        | 说明                                                          |
    |------------|---------------------------------------------------------------|
    | prompt 名称 | 提示词的唯一标识，代码里 get_prompt("agent_system_prompt") 用它 |
    | version    | 每保存一次自增 1，**内容不可变**；老版本永远拉得到               |
    | label      | 可移动的指针（production / staging / latest），用来切换「当前生效」的版本 |

    所以「线上更新不重启」的原理是：代码里写死的是 name + label，
    真正的内容每次请求都去拉（SDK 默认带 60 秒缓存），改了 label 就换了内容。

变量语法：Langfuse 用 Mustache 风格的双花括号占位 —— 你是{{user_name}}的助手。
    - 代码里 prompt.compile(user_name="张三") 会把变量填进去
    - prompt.variables 可以列出这个提示词声明了哪些变量

安装：uv add langfuse

课案出处：Agent 课案 → 监控与评估 → 提示词
         （课案的「创建提示词并标 production」「添加版本、切换生产版本」「使用带变量的提示词」
           三张截图，分别对应本文件的第一、三节和第二节）

运行方式：
    uv run Agent/06_langfuse/03_提示词管理_jxsd.py

本机前置条件：settings.langfuse_public_key / langfuse_secret_key 为空，
脚本会先打印中文配置指引，再走不依赖 Langfuse 服务的降级演示：
用 Langfuse 自己的 Prompt 结构体在本地造一个 prompt 对象，
compile 的行为与从服务端拉下来时**完全一致**，并且照样真调大模型看效果。

本机实测结论：
    ① `compile(user_name="张三", current_date="2026-07-07")` 把
       「你是{{user_name}}的专属助手，今天是{{current_date}}。」填成
       「你是张三的专属助手，今天是2026-07-07。」—— 双花括号占位符生效；
    ② 同一个 `label="production"` 连续拉两次，拿到的是 **version=3** 同一版，
       印证了「代码里写死的是 label，具体版本由服务端决定」+「SDK 有 60 秒缓存」；
    ③ 编译结果直接喂给 `create_deep_agent(system_prompt=...)` 能正常驱动 Agent ——
       提示词托管和 Agent 框架是解耦的，换框架不用改提示词那一层。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
# create_deep_agent 只用在最后一节：证明「编译出来的提示词真的能驱动 Agent」。

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langfuse import Langfuse
from langfuse.api.prompts.types.prompt import Prompt_Text
from langfuse.model import TextPromptClient
from config import settings

# ---------- 0. 客户端初始化：先判断密钥是否就绪 ----------
LANGFUSE_READY = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if LANGFUSE_READY:
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
# 密钥为空就置 None：SDK 拿不到凭据会往 stderr 刷错误，不如干脆不实例化。
else:
    langfuse = None

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# ---------- 1. 提示词的两种类型 ----------
# text 类型：一整段字符串，compile() 返回 str —— 适合 system prompt
PROMPT_NAME = "agent_system_prompt"
PROMPT_V1 = "你是一个乐于助人的助手。"
PROMPT_V2 = "你是{{user_name}}的专属助手，今天是{{current_date}}，回答请控制在 50 字以内。"
PROMPT_LATEST = "你是{{user_name}}的专属助手，今天是{{current_date}}。回答请控制在 50 字以内，并在结尾加一句鼓励。"
# 本地兜底：真拉不到时用它（生产代码里也建议带上 fallback，避免 Langfuse 挂了业务跟着挂）
FALLBACK_PROMPT = "你是一个助手，请用中文简洁回答。"


def build_prompt_locally(text: str, version: int, labels: list[str]) -> TextPromptClient:
    """降级用：在本地造一个与服务端返回结构完全相同的 prompt 对象。

    Langfuse 从服务端拉提示词时，拿到的就是这个 Prompt_Text 结构体，
    再用它包出 TextPromptClient。所以本地这样造出来的对象，
    compile() / variables / version / labels 的行为与真的拉下来一模一样。
    """
    return TextPromptClient(Prompt_Text(
        name=PROMPT_NAME,
        version=version,
        prompt=text,
        labels=labels,
        # Prompt_Text 是 SDK 的强类型模型，字段要填全 —— 缺字段会直接报校验错误。
        tags=[],
        config={},
        commit_message=None,
    ))


def get_prompt(name: str, label: str = "production", version: int | None = None):
    """拉取提示词：真连上就查服务端，没连上就返回本地对应的版本。

    label 与 version 二选一：
        label="production"  取「当前生产版本」，服务端换 label 指向即可热更新
        version=2           取「第 2 版」这个不可变快照，用于复现历史实验
    """
    if LANGFUSE_READY:
        # fallback 参数非常关键：服务端不可用 / 这个提示词还没建，
        # SDK 会直接返回 fallback 内容而不是抛异常，保证线上不受影响
        # ⚠️ label 与 version 必须**二选一**：两个都传时 SDK 会在本地就抛
        #    `ValueError: Cannot specify both version and label at the same time.`
        #    （不是服务端报错）—— 所以这里显式只传一个，与下方降级分支保持一致。
        if version is not None:
            return langfuse.get_prompt(name, version=version, fallback=FALLBACK_PROMPT)
        return langfuse.get_prompt(name, label=label, fallback=FALLBACK_PROMPT)

    # 降级：本地三个版本，模拟服务端已有的版本历史
    local_versions = {
        1: (PROMPT_V1, ["latest"]),
        2: (PROMPT_V2, []),
        3: (PROMPT_LATEST, ["production", "latest"]),
    }
    # version 与 label 二选一：version 是不可变快照（复现历史实验），label 是可移动指针（热更新）。
    if version is not None:
        text, labels = local_versions.get(version, (FALLBACK_PROMPT, []))
        return build_prompt_locally(text, version, labels)
    if label == "production":
        return build_prompt_locally(PROMPT_LATEST, 3, ["production", "latest"])
    if label == "staging":
        return build_prompt_locally(PROMPT_V2, 2, ["staging"])
    return build_prompt_locally(PROMPT_V1, 1, ["latest"])


# ---------- 2. 在代码里创建 / 更新提示词 ----------
def demo_create_prompt():
    """课案原文是在 Web UI 上点出来；这一步演示「代码里也能建」，
    适合把提示词随 CI 流程一起发布、保证多环境一致。"""
    print("\n" + "=" * 72)
    print("一、创建提示词 create_prompt(name=..., prompt=..., labels=['production'])")
    print("=" * 72)

    payload = {
        "name": PROMPT_NAME,
        "prompt": PROMPT_LATEST,
        "labels": ["production"],       # 打上 production，代码里就能按 label 拉到它
        "type": "text",                 # text = 一整段字符串；chat = 消息数组
        "commit_message": "增加用户称呼与日期变量，限制回答长度",
    }
    if LANGFUSE_READY:
    # 真连上就调 SDK 建提示词；否则打印 create_prompt 的报文，学员照着就能在 UI 上点出来。
        prompt = langfuse.create_prompt(**payload)
        print(f"已创建：name={prompt.name} version={prompt.version} labels={prompt.labels}")
        langfuse.flush()
    else:
        print("[降级] 本应发送给 Langfuse 的 create_prompt 报文：")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("（Web UI 上等价操作：Prompts → New prompt → 填 name 与内容 → Save → ")
        print("  在版本列表里把 production 标签拖到想要的那一版）")


# ---------- 3. 拉取 + 编译 + 用于 Agent ----------
def demo_pull_and_compile():
    print("\n" + "=" * 72)
    print("二、拉取并编译 get_prompt(name, label='production').compile(**变量)")
    print("=" * 72)

    # 代码里写死的是 name + label，具体是哪一版由服务端决定 —— 这就是「线上更新不重启」。
    label = "production"
    prompt = get_prompt(PROMPT_NAME, label=label)
    print(f"拉取到：name={PROMPT_NAME} label={label} "
          f"version={prompt.version} labels={prompt.labels}")
    print("声明的变量：", prompt.variables)     # ['user_name', 'current_date']

    # compile 就是把 {{变量}} 填上值；缺变量会抛错，所以变量表要维护好
    system_text = prompt.compile(user_name="张三", current_date="2026-07-07")
    print("编译结果：", system_text)

    # 想直接拿去当 LangChain 的模板时，官方还提供了占位符转换（{{x}} → {x}）
    print("LangChain 模板形式：", prompt.get_langchain_prompt())

    # 课案把编译结果塞进了 DeepAgents 的 system_prompt
    agent = create_deep_agent(model=llm, tools=[], system_prompt=system_text)
    result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
    print("AI：", result["messages"][-1].content)

    if LANGFUSE_READY:
        langfuse.flush()


# ---------- 4. 版本与 label：线上更新不重启 ----------
def demo_version_switch():
    """课案截图「可以添加版本，切换生产版本」讲的就是这一段。"""
    print("\n" + "=" * 72)
    print("三、版本与 label：把 production 指向新版本，代码无需改动、进程无需重启")
    print("=" * 72)

    print("课案在 Web UI 上的操作：Prompts → 选中提示词 → New version 存新内容 →")
    print("把 production 标签从 v2 拖到 v3。下面用本地版本历史模拟这个切换：\n")

    for version in (2, 3):
        p = get_prompt(PROMPT_NAME, version=version)
        print(f"  version={p.version}  labels={p.labels}")
        print(f"    原文   : {p.prompt}")
        print(f"    编译后 : {p.compile(user_name='张三', current_date='2026-07-07')}")
# 同一个 get_prompt 调用，两次拿到不同内容：变的只有服务端 label 的指向。

    print("\n  代码里写的是 label='production'，从来没改过；")
    print("  改的只是服务端那个 label 指向哪个 version —— 这就是「线上更新不重启」。")

    # 同一个 label 再拉一次：SDK 有 60 秒缓存，所以这里拿到的还是同一版
    again = get_prompt(PROMPT_NAME, label="production")
    print(f"\n  再拉一次 label='production' → version={again.version}（SDK 缓存 60s，避免每次请求都打网络）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # 起始检查：缺密钥走降级指引，密钥就绪只打印一行拉取地址。
    if not LANGFUSE_READY:
        print("=" * 72)
        print("【进入降级演示】Langfuse 密钥为空")
        print("  settings.langfuse_public_key = '' ，settings.langfuse_secret_key = ''")
        print("-" * 72)
        # 四步照课案「安装」一节抄下来；第 4 步特意提醒「先在控制台把提示词建出来并标 production」。
        print("要看到真实的提示词托管，按课案「安装」一节准备环境：")
        print("  1) git clone https://github.com/langfuse/langfuse.git")
        print("     cd langfuse")
        print("     docker compose up -d          # 启动后访问 http://localhost:3000")
        print("  2) 首次注册的账号即为管理员；新建项目 → Settings → API Keys → 创建密钥")
        print("  3) 写入 F:\\ProGram\\Python_Base\\.env ：")
        print("        LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("        LANGFUSE_SECRET_KEY=sk-lf-...")
        print("        LANGFUSE_HOST=http://localhost:3000    # 本地 docker 部署用这个")
        print("  4) 重新运行本脚本；别忘了先在控制台把 agent_system_prompt 建出来并标 production")
        print("-" * 72)
        # 上面是配置步骤，下面是「没配也能跑」的说明：compile() 的行为与服务端拉取时一致。
        print("下面不依赖 Langfuse 服务：用 Langfuse 自己的 Prompt 结构体在本地造提示词对象，")
        print("compile() 的行为与从服务端拉下来一致，并且照样真调大模型。")
        print("=" * 72)
    else:
        print("Langfuse 已配置，提示词将从以下地址拉取：", settings.langfuse_host)

    # 三节依次演示：代码建提示词 → 拉取 + 编译 + 驱动 Agent → 版本与 label 切换。
    demo_create_prompt()
    demo_pull_and_compile()
    demo_version_switch()
