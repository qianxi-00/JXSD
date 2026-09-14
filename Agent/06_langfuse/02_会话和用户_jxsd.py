# -*- coding: utf-8 -*-
"""
Langfuse ②：可观测性之「会话（Session）与用户（User）」——课案完整版
================================================================
课案原话：
    通过 session_id 和 user_id 标记每次对话，方便在后台按用户、会话筛选追踪数据。

为什么单有 trace_id 不够？三个 ID 是三个不同的粒度：

    | 字段        | 代表什么                     | 一次对话里有几条 | 后台能干什么               |
    |------------|------------------------------|----------------|--------------------------|
    | trace_id   | 一次请求 / 一次图执行          | 每轮一条        | 看这一轮的完整调用链        |
    | session_id | 一次会话（一个聊天窗口）        | 多条 trace     | 把多轮对话连起来看上下文     |
    | user_id    | 一个用户（跨会话、跨天）        | 跨会话累计      | 看某个用户的所有历史行为     |
    | tags       | 自定义标签（环境 / 版本 / 渠道） | 任意            | 按 灰度版本、生产/测试 筛选  |

课案给 LangChain 系的写法是把这三个值塞进 invoke 的 config["metadata"]，
用三个「魔法 key」（Langfuse 的回调处理器会识别它们，并提升为 trace 的属性）：

    config={
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": "session_001",
            "langfuse_user_id": "user_A",
            "langfuse_tags": ["production", "agent-v2"],
        },
    }

本机装的 SDK 是 langfuse v4，除了上面的 metadata 写法，还提供了更直观的
上下文管理器 propagate_attributes()，两种写法效果完全一致，本文件都演示一遍。

课案出处：Agent 课案 → 监控与评估 → 可观测性 → 会话和用户

运行方式：
    uv run Agent/06_langfuse/02_会话和用户_jxsd.py

本机前置条件：settings.langfuse_public_key / langfuse_secret_key 为空，
脚本会先打印中文配置指引，再走不依赖 Langfuse 服务的降级演示：
照样真调大模型，并把「本应上报的 trace 元数据」打印出来。

本机实测结论：
    ① 密钥为空时 SDK 会往 stderr 刷 "Authentication error: ... Client will be disabled."，
       虽然不抛异常但输出很脏 —— 所以本文件先算 LANGFUSE_READY 再决定要不要实例化客户端；
    ② 两种写法（metadata 三个魔法 key / propagate_attributes 上下文管理器）在
       langfuse v4 上都能用，上报的字段完全一致，差别只在「作用范围」：
         metadata 写法是**一次调用一份**，要写在每个 invoke 的 config 里；
         propagate_attributes 是**一个 with 块一份**，块内所有调用（含嵌套）自动继承。
    ③ 降级演示里打印的分组视图，就是 Langfuse 后台 Sessions / Users 两个页面的口径：
       同一个 session_id 的多条 trace 会被串成一个会话，user_id 则跨会话聚合。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
# from config import settings 是仓库根目录的统一配置，Langfuse 密钥与模型参数都从这里取。

from langchain.chat_models import init_chat_model
from langfuse import Langfuse
from langfuse import get_client
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
from config import settings

# ---------- 0. 客户端初始化：先判断密钥是否就绪 ----------
LANGFUSE_READY = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if LANGFUSE_READY:
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        # 密钥为空时不实例化真客户端：SDK 会往 stderr 刷认证错误，教学输出会很脏。
        host=settings.langfuse_host,
    )
else:
    langfuse = None      # 降级演示：不实例化真客户端，免得 SDK 往 stderr 刷认证错误

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 降级演示用：把「每条 trace 本应携带的元数据」攒起来，最后统一打印
LOCAL_TRACES: list[dict] = []


def _record(trace_name: str, session_id: str, user_id: str, tags: list[str],
            question: str, answer: str) -> None:
    """登记一条「本应上报的 trace」。真连上 Langfuse 后，这些字段就是后台的筛选项。"""
    LOCAL_TRACES.append({
        "id": f"trace-{len(LOCAL_TRACES) + 1:03d}",   # 真实环境里是 SDK 生成的 32 位 hex
        "name": trace_name,
        "sessionId": session_id,                      # 后台按「会话」筛选用它的值
        "userId": user_id,                            # 后台按「用户」筛选用它的值
        "tags": tags,                                 # 后台按标签二次筛选
        "input": question,
        "output": answer,
    })


# ---------- 1. 课案写法：metadata 里的三个魔法 key ----------
def demo_metadata_keys():
    print("\n" + "=" * 72)
    print("写法一（课案）：config['metadata'] 里的 langfuse_session_id / user_id / tags")
    print("=" * 72)

    session_id = "session_001"
    user_id = "user_A"
    tags = ["production", "agent-v2"]

    turns = ["你好，我是小明", "帮我算一下 12*12"]   # 同一会话里的连续两轮
    for i, question in enumerate(turns, start=1):
        if LANGFUSE_READY:
            langfuse_handler = CallbackHandler()
            config = {
                "callbacks": [langfuse_handler],
                # last_trace_id 就是这一轮在 Langfuse 里的 trace id，后面打分要用它。
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": user_id,
                    "langfuse_tags": tags,
                },
            }
            response = llm.invoke(question, config=config)
            answer = response.content
            print(f"  第{i}轮 trace_id = {langfuse_handler.last_trace_id}")
        else:
            # 降级：config 的形状与课案一模一样，只是回调换成了空壳
            answer = llm.invoke(question).content
            _record("chat", session_id, user_id, tags, question, answer)
        print(f"  第{i}轮  Q: {question}")
        print(f"          A: {answer[:60]}")

    # 多轮共用一个 session_id，后台就会把这两条 trace 串成一个会话。
    if LANGFUSE_READY:
        get_client().flush()


# ---------- 2. v4 写法：propagate_attributes 上下文管理器 ----------
def demo_propagate_attributes():
    """同一个会话换个用户再聊一句，用来对比「同会话 / 不同用户」在后台的差异。"""
    print("\n" + "=" * 72)
    print("写法二（langfuse v4）：with propagate_attributes(session_id=..., user_id=...)")
    print("=" * 72)

    session_id, user_id = "session_002", "user_B"
    question = "用一句话说明 session_id 和 user_id 的区别"

    if LANGFUSE_READY:
        # 这个 with 块里发生的**所有**调用都自动带上这些属性，
        # 不用再逐个 invoke 传 metadata，嵌套调用也会继承
        with propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=["staging", "v4-api"],
            trace_name="chat_propagated",
        ):
            langfuse_handler = CallbackHandler()
            answer = llm.invoke(question, config={"callbacks": [langfuse_handler]}).content
        print("  trace_id =", langfuse_handler.last_trace_id)
        get_client().flush()
    else:
        answer = llm.invoke(question).content
        _record("chat_propagated", session_id, user_id, ["staging", "v4-api"], question, answer)
# with 块里发生的所有调用都会继承这些属性，包括嵌套调用 —— 这是它比 metadata 省事的地方。

    print(f"  Q: {question}")
    print(f"  A: {answer[:60]}")


# ---------- 3. 降级演示：把「本应上报的 trace 元数据」打印出来 ----------
def print_local_traces():
    if LANGFUSE_READY:
        return
    print("\n" + "=" * 72)
    print("[降级] 本应上报给 Langfuse 的 trace 记录（后台的「会话 / 用户」筛选就靠这些字段）")
    print("=" * 72)
    print(json.dumps(LOCAL_TRACES, ensure_ascii=False, indent=2))

    # 顺便按 session_id / user_id 分组，模拟 Langfuse 后台的两个视图
    # 这一步是「为什么要这两个 id」最直观的答案：同一批 trace，换一个 key 分组
    # 就得到两种完全不同的视图 —— 一个是「这次聊天说了什么」，一个是「这个人做过什么」。
    print("\n[降级] 换成 Langfuse 后台的视角：")
    by_session: dict[str, list[dict]] = {}
    by_user: dict[str, list[dict]] = {}
    for t in LOCAL_TRACES:
        # setdefault + append = 分组惯用法：key 不存在就建空列表，然后统一 append
        by_session.setdefault(t["sessionId"], []).append(t)
        by_user.setdefault(t["userId"], []).append(t)
    for sid, items in by_session.items():
        # 会话视图：只关心「这一窗里问了哪些问题」，顺序就是对话顺序
        print(f"  会话 {sid:<14} → {len(items)} 条 trace：{[i['input'] for i in items]}")
    for uid, items in by_user.items():
        # 用户视图：同一批 trace 再按 user_id 聚合，能跨会话看到「这个人一共聊了几窗」
        # {i['sessionId'] for i in items} 是集合推导，自动去重 = 会话数
        print(f"  用户 {uid:<14} → {len(items)} 条 trace，跨 "
              f"{len({i['sessionId'] for i in items})} 个会话")


# ---------- 4. 入口：先讲清「缺什么」，再依次跑三段演示 ----------
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # 降级指引写得长一点是故意的：Langfuse 没配好时，学员最容易卡在
    # 「docker 起了但没建项目」「建了项目但没把 key 写进 .env」这两步
    if not LANGFUSE_READY:
        print("=" * 72)
        print("【进入降级演示】Langfuse 密钥为空")
        print("  settings.langfuse_public_key = '' ，settings.langfuse_secret_key = ''")
        # 密钥就绪时这里只打印一行目标地址；降级时下面的指引才是学员真正要看的。
        print("-" * 72)
        print("要看到真实的会话/用户视图，按课案「安装」一节准备环境：")
        print("  1) git clone https://github.com/langfuse/langfuse.git")
        print("     cd langfuse")
        print("     docker compose up -d          # 启动后访问 http://localhost:3000")
        print("  2) 首次注册的账号即为管理员；新建项目 → Settings → API Keys → 创建密钥")
        print("  3) 写入 F:\\ProGram\\Python_Base\\.env ：")
        print("        LANGFUSE_PUBLIC_KEY=pk-lf-...")
        # 给出可执行的补救步骤（git clone / docker compose / 写入 .env），而不是只说「未配置」。
        print("        LANGFUSE_SECRET_KEY=sk-lf-...")
        print("        LANGFUSE_HOST=http://localhost:3000    # 本地 docker 部署用这个")
        print("  4) 重新运行本脚本，去 Langfuse 的 Sessions / Users 页面按会话、用户筛选")
        print("-" * 72)
        print("下面不依赖 Langfuse 服务：照样真调大模型，只把「本应上报的元数据」打印出来。")
        print("=" * 72)
    else:
        print("Langfuse 已配置，数据将上传到：", settings.langfuse_host)

    # 三段依次跑：metadata 写法 → propagate_attributes 写法 → 降级视图（密钥就绪时是空操作）
    demo_metadata_keys()
    demo_propagate_attributes()
    print_local_traces()
# 两段演示各走一种写法，最后一段只在降级时才有输出（密钥就绪时它是空操作）。

    # 小结落在「三个 id = 三个粒度」上，和文件头的表格首尾呼应
    print("\n小结：trace_id 定位「这一轮」、session_id 串起「这一窗」、user_id 归属「这个人」；")
    print("      Langfuse 后台的 Sessions / Users / Traces 三个页面，就是这三种粒度的视图。")
