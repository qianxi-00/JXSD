# -*- coding: utf-8 -*-
r"""
LangChain 核心组件（一）：模型
================================================================
课案原文用 `ChatOpenAI` 直接 new 出模型对象：

    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(
        api_key=setting.API_KEY,
        base_url=setting.BASE_URL,
        model=setting.MODEL_NAME,
        temperature=0,
    )
    response = model.invoke("你好")
    print(response)

本项目（Python_Base）把配置统一收口在根目录 `config.py`，所以课案里的
`from config import setting` 一律改成 `from config import settings`，
三件套字段名也从大写换成小写：`settings.api_key` / `settings.base_url` /
`settings.model_name`（详见根目录 `config.py` 的注释）。

本节要讲清两件事：

    1. 课案写法 `ChatOpenAI(...)`：直接 new 出某一家供应商的模型类；
    2. 本项目统一写法 `init_chat_model(...)`：只报「供应商 + 模型名」，
       由 LangChain 按名字去查表、反射出对应的模型类。
    两者的返回值都能 `.invoke()`，功能完全一样，区别在「可切换性」，见文件末尾表格。

课案出处：Agent 课案 → langChain → 核心组件 → 模型

前置条件：
    - 项目根目录 `.env` 里 `API_KEY` / `BASE_URL` / `MODEL_NAME` 三项已填好
      （`config.py` 会自动加载，本文件通过 `settings.api_key` 等读取），
      缺任一项时下面的 `invoke` 会在网络层报错；
    - 本节只调大模型、不碰数据库，**不需要** PostgreSQL / Redis。

运行方式（必须在项目根目录 `F:\ProGram\Python_Base` 下执行，
否则 import 不到根目录的 `config`）：
    uv run Agent/02_langchain/01_模型_jxsd.py
    # 或： & '.\.venv\Scripts\python.exe' 'Agent\02_langchain\01_模型_jxsd.py'
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from config import settings


# ---------- 1. 课案写法：ChatOpenAI 直接 new ----------
# 课案演示的就是这种写法，所以这里保留下来（CONVENTIONS 允许：
# 课案原文明确演示 ChatOpenAI 的小节可以用 ChatOpenAI，但参数必须来自 settings，不能硬编码）。
legacy_model = ChatOpenAI(
    api_key=settings.api_key,       # 课案的 setting.API_KEY → settings.api_key
    base_url=settings.base_url,     # 课案的 setting.BASE_URL → settings.base_url
    model=settings.model_name,      # 课案的 setting.MODEL_NAME → settings.model_name
    temperature=0,                  # 课案固定 0：要的是稳定、可复现的输出
)

# ---------- 2. 本项目统一写法：init_chat_model ----------
# 注意这里没有 import 任何供应商的模型类，只给了 model_provider="openai"。
# init_chat_model 内部维护了一张「供应商名 → 模型类」的注册表：
# 它按 model_provider 找到 langchain_openai.ChatOpenAI，
# 再把 model / api_key / base_url 当构造参数传进去。
# 换句话说：init_chat_model 是「工厂」，ChatOpenAI 是「车间」。
llm = init_chat_model(
    model_provider="openai",        # 供应商：openai / anthropic / google_genai ...
    model=settings.model_name,      # 模型名
    api_key=settings.api_key,       # OpenAI 兼容接口的三件套
    base_url=settings.base_url,
    temperature=0.7,                # 采样温度：越高越随机；课案这里是 0
)


# ---------- 3. 两种写法的区别 ----------
# 课案只给了 ChatOpenAI 的写法，实际项目里两种都会遇到，区别如下：
#
# | 维度         | ChatOpenAI(...) 直接 new                | init_chat_model(...) 工厂        |
# |--------------|-----------------------------------------|----------------------------------|
# | 代码依赖     | 强依赖 langchain_openai 这个包和类名     | 只依赖字符串 "openai"，不 import 供应商包 |
# | 换供应商     | 要改 import + 改类名（通义/Claude 各不相同） | 只改 model_provider 一个字符串   |
# | 配置驱动     | 模型名写死在代码里                       | 可从 .env / 数据库读，天然支持多环境 |
# | 私有参数     | 能用到供应商独有的参数（如 logit_bias）  | 只能传通用参数，私有参数要走 **kwargs |
# | 运行期类型   | ChatOpenAI                              | 仍然是 ChatOpenAI（运行时同一对象） |
# | 适合场景     | 只用一家、要压榨该家私有能力             | 多供应商切换、配置化/平台化的项目  |
#
# 结论：本项目 Agent 课案统一用 init_chat_model，
# 只有课案原文明确演示 ChatOpenAI 的小节才保留直接 new 的写法。

if __name__ == "__main__":
    # ---------- 4. 打印两种写法的真实类型，证明「运行时是同一个类」 ----------
    print("legacy_model 的实际类型：", type(legacy_model).__name__)
    print("llm 的实际类型：         ", type(llm).__name__)

    # ---------- 5. 课案原样调用：一句话进，一句话出 ----------
    # 模型对象统一用 invoke() 调用（异步版是 ainvoke）；传进去的字符串会被自动包成
    # 一条 HumanMessage —— 这就是课案里 model.invoke("你好") 能直接跑的原因。
    response = legacy_model.invoke("你好")
    print("\n[课案写法] response 对象：", type(response).__name__)
    print("[课案写法] response.content：", response.content)

    # ---------- 6. 工厂写法同样能调，并演示「消息列表」入参 ----------
    # invoke 既能吃纯字符串，也能吃 (角色, 内容) 元组列表或 Message 对象（见 02_消息_jxsd.py）
    response = llm.invoke(
        [
            ("system", "你是一位惜字如金的助手，回答不超过 20 个字。"),
            ("user", "用一句话解释什么是 Agent"),
        ]
    )
    print("\n[工厂写法] 带角色的回复：", response.content)

    # ---------- 7. 同一个模型对象可以反复 invoke，它是无状态的 ----------
    # 每次调用都是独立的一次 HTTP 请求，模型不会记得上一句，
    # 「记忆」是我们在外面把历史消息拼进列表实现的（见 05/06）。
    again = llm.invoke("我刚才问你的是什么？")
    print("[无状态验证] 再问一次：", again.content)
