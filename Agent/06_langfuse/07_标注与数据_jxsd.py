# -*- coding: utf-8 -*-
"""
Langfuse ⑦：评估之「标注」+「数据」——课案完整版（新增文件）
================================================================
课案「标注」与「数据」两节只给了 Web UI 截图，没给代码。本文件把这两个页面
背后的 API 补全成可跑的脚本，并讲清它们在整个评估闭环里的位置：

    自动打分（04/05/06）→ 找出可疑样本 → 人工标注（本节）→ 沉淀成数据集（本节）
        ↑                                                          ↓
        └──────────────── 拿数据集回归评估（04/05/06 循环）  ←────────┘

一、标注（Annotation Queue）—— 人工评估
    自动指标能发现「哪条不对劲」，但不能给出「正确答案是什么」。
    所以需要把可疑的 trace 挑出来，交给人工逐条打分，人打出来的分数就是
    评估自动指标时的**地面真值（ground truth）**。

    工作流（每一步一个 API）：

    | 步骤 | 干什么                        | 对应 API                                          |
    |-----|------------------------------|--------------------------------------------------|
    | 1   | 定义打分维度（评分表）          | api.score_configs.create                         |
    | 2   | 建一个标注队列，挂上打分维度     | api.annotation_queues.create_queue               |
    | 3   | 把要复核的 trace 放进队列       | api.annotation_queues.create_queue_item          |
    | 4   | 把队列指派给标注员              | api.annotation_queues.create_queue_assignment     |
    | 5   | 人在 UI 上逐条打分/写评论        | （Web UI 操作）                                   |
    | 6   | 分数回流到 trace，可按分数筛选   | api.scores.get_many                              |

    打分维度（Score Config）的三种类型：

    | data_type    | 用途           | 必填项                                | 例子                       |
    |-------------|----------------|--------------------------------------|---------------------------|
    | NUMERIC     | 连续评分        | min_value / max_value                 | 准确性 0~1、满意度 1~5      |
    | CATEGORICAL | 单选分类        | categories（每个含 value(数字) + label）| 问题类型：咨询/投诉/其他     |
    | BOOLEAN     | 是/否           | 无                                    | 回答是否有害、是否答非所问    |

二、数据（Datasets）—— 把标注结果沉淀成回归测试集
    数据集 = 一组测试项（input + expected_output + metadata）。
    每次拿它跑一遍系统，就产生一次 **Dataset Run**；
    同一个数据集跑不同版本（v1 / v2 / 换模型 / 换提示词），
    在 UI 里可以并排对比——这就是「改完到底有没有变好」的客观依据。

    | 概念          | 说明                                                   |
    |--------------|--------------------------------------------------------|
    | Dataset      | 测试集本身，可版本化                                      |
    | Dataset Item | 一条测试项：input / expected_output / metadata           |
    | Dataset Run  | 拿这个测试集跑一遍的结果（run_experiment 会产生）            |
    | Run Item     | 一次运行里「某条测试项 → 实际输出 + 分数」的对应关系          |

安装：uv add langfuse

课案出处：Agent 课案 → 监控与评估 → 评估 → 标注（四张截图：标注队列的建立与使用）
         Agent 课案 → 监控与评估 → 数据（三张截图：数据集、数据集条目、数据集运行对比）
         （这两节课案只给了 Web UI 截图，本文件是新增文件：把截图背后的 REST API 补全成代码）

运行方式：
    uv run Agent/06_langfuse/07_标注与数据_jxsd.py

本机前置条件：settings.langfuse_public_key / langfuse_secret_key 为空，
脚本会先打印中文配置指引，再走不依赖 Langfuse 服务的降级演示：
把上面每一步要发的报文原样打印出来，并真跑一次 Agent 产生「待标注的样本」。

本机实测结论（关键在「数据集运行」那一段）：
    ① 三个打分维度都用同一套 `api_call` 封装，降级时打印出的报文就是 SDK 的请求体，
       把里面的 endpoint 与 payload 抄进 Postman / curl 就能手工复现；
    ② 数据集运行真的会产生 run item：每条含 datasetItemId / runName / output / scores，
       其中 `runName` 是并排对比的钥匙 —— 换个 run_name（换提示词或换模型）再跑一次，
       UI 上就能把两次 run 并排看，这是「改完到底有没有变好」的客观依据；
    ③ 人工标注的价值在于它产出的是**标准答案**：自动指标只能告诉你「哪条不对劲」，
       给不出 ground truth。所以本文件把标注结论直接写成 `expected_output` 存进数据集，
       下一轮回归就有了判分基准 —— 这就是课案「标注 → 数据」两节连起来的意思。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
from types import SimpleNamespace

from langchain.chat_models import init_chat_model
from langfuse import Langfuse
# langfuse 客户端只在本文件里用；标注队列与数据集都通过它的 .api.* 调用。
from config import settings

# ---------- 0. 客户端初始化：先判断密钥是否就绪 ----------
LANGFUSE_READY = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if LANGFUSE_READY:
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        # 密钥为空置 None：下面所有 api_call 会走降级分支，打印报文并返回本地桩对象。
        host=settings.langfuse_host,
    )
else:
    langfuse = None

    # 本文件要真跑一次大模型产生「待标注样本」，所以模型必须就绪。
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def api_call(endpoint: str, func, fallback_id: str, **payload):
    """统一封装一次 REST 调用。

    真连上 Langfuse → 调 SDK 对应方法；
    密钥为空       → 打印「本应发送的报文」，并返回一个带 id 的本地桩对象，
                     这样上层代码拿 id 继续往下走的逻辑完全一致。
    """
    if LANGFUSE_READY:
        return func(**payload)
    print(f"    [降级] 本应调用  POST {endpoint}")
    print("           payload = " + json.dumps(payload, ensure_ascii=False, default=str))
    return SimpleNamespace(id=fallback_id, **payload)


# ---------- 1. 定义打分维度（Score Config） ----------
def create_score_configs() -> dict:
    """先把「人工要按什么标准打」定义出来，后面建队列时要挂上这些维度。"""
    print("\n" + "=" * 72)
    print("第一步：定义打分维度 POST /api/public/score-configs")
    print("=" * 72)

    configs = {}

    # ① 数值型：人工给 0~1 的连续分
    configs["回答准确性"] = api_call(
        "/api/public/score-configs",
        (langfuse.api.score_configs.create if LANGFUSE_READY else None),
        "cfg-accuracy-001",
        name="回答准确性",
        data_type="NUMERIC",
        # NUMERIC 必须给 min/max：UI 上才会渲染成滑杆，也才能校验人工输入的范围。
        min_value=0.0,
        max_value=1.0,
        description="人工判断：回答是否正确解决了用户问题（0=完全错误，1=完全正确）",
    )

    # ② 布尔型：是/否，用于红线检查
    configs["回答是否有害"] = api_call(
        "/api/public/score-configs",
        (langfuse.api.score_configs.create if LANGFUSE_READY else None),
        "cfg-harmful-001",
        name="回答是否有害",
        # BOOLEAN 不用额外字段；适合「是/否」的红线检查（是否有害、是否答非所问）。
        data_type="BOOLEAN",
        description="回答中是否包含有害/违规内容",
    )

    # ③ 分类型：给样本归类，方便后续按类型切片分析
    configs["问题类型"] = api_call(
        "/api/public/score-configs",
        (langfuse.api.score_configs.create if LANGFUSE_READY else None),
        "cfg-category-001",
        name="问题类型",
        data_type="CATEGORICAL",
        categories=[
            # ⚠️ 实测踩坑：CATEGORICAL 的 **value 必须是数字**，不能写字符串。
            #    SDK 的 `ConfigCategory.value` 类型是 `float`，写成 "consult" 这种
            #    字符串会被服务端拒掉：
            #      Category must be an array of objects with label value pairs,
            #      where labels and values are unique.
            #    label 才是给人看的显示名 —— 两者分工别搞反。
            {"value": 0, "label": "咨询"},
            {"value": 1, "label": "投诉"},
            {"value": 2, "label": "其他"},
        ],
        description="给用户问题归类，便于分桶看指标",
    )

    for name, cfg in configs.items():
        print(f"  ✓ {name}  id={cfg.id}  data_type={cfg.data_type}")
    return configs


# ---------- 2. 建标注队列 / 放样本 / 指派标注员 ----------
QUEUE_NAME = "agent-回答质量复核"


def build_annotation_queue(configs: dict, trace_ids: list[str]) -> str:
    print("\n" + "=" * 72)
    print("第二~四步：建队列 → 放 trace → 指派标注员")
    print("=" * 72)

    # 1) 建队列，把三个打分维度挂上去（标注员进队列看到的表单就是这几个维度）
    # queue 的 id 要留给后面「放样本」「指派标注员」两步用，所以必须接住返回值。
    #
    # 幂等处理：Langfuse **不允许同名队列** —— 再建会 400
    # `A queue with this name already exists.`（实测踩坑）。
    # 本文件是可以反复跑的演示脚本，所以先按名字查一遍，已有就直接复用；
    # 否则第二次跑必崩，而「重跑一次」在演示/回归里是最常见的动作。
    queue = None
    if LANGFUSE_READY:
        existing = langfuse.api.annotation_queues.list_queues(limit=100)
        queue = next(
            (q for q in getattr(existing, "data", []) if getattr(q, "name", None) == QUEUE_NAME),
            None,
        )
        if queue is not None:
            print(f"  · 已存在同名队列，直接复用：{queue.name}  id={queue.id}")

    if queue is None:
        queue = api_call(
            "/api/public/annotation-queues",
            (langfuse.api.annotation_queues.create_queue if LANGFUSE_READY else None),
            "queue-local-001",
            name=QUEUE_NAME,
            description="把 accuracy 低的 trace 挑出来，人工复核后给出标准答案，沉淀成数据集",
            score_config_ids=[cfg.id for cfg in configs.values()],
        )
        print(f"  ✓ 队列已建：{queue.name}  id={queue.id}")

    # 2) 逐条把要复核的 trace 放进队列（object_type 还能是 OBSERVATION / SESSION）
    for trace_id in trace_ids:
        item = api_call(
            "/api/public/annotation-queues/{queueId}/items",
            (langfuse.api.annotation_queues.create_queue_item if LANGFUSE_READY else None),
            f"queue-item-{trace_id}",
            queue_id=queue.id,
            object_id=trace_id,
            object_type="TRACE",     # 也可以整条会话（SESSION）或某个 span（OBSERVATION）
            status="PENDING",        # PENDING 待处理 / COMPLETED 已完成
        )
        print(f"  ✓ 已入队：{item.object_id}  status={item.status}")

    # 3) 指派标注员（user_id 是 Langfuse **组织成员**的用户 id，必须是项目里真实存在的成员）
    #    ⚠️ 实测踩坑：随便填一个（如 "user_A"）会被服务端拒掉 ——
    #       404 `User not found or not authorized for this project`。
    #       真实 id 在 Langfuse 控制台 → Settings → Members 里看。
    #       这是服务端的**引用完整性校验**，只有拿到真实 id 才能通过；所以这里兜住不崩：
    #       队列与样本都已经建好了，指派失败不影响本节其余演示。
    try:
        assignment = api_call(
            "/api/public/annotation-queues/{queueId}/assignments",
            (langfuse.api.annotation_queues.create_queue_assignment if LANGFUSE_READY else None),
            "queue-assignment-001",
            queue_id=queue.id,
            # 指派之后，标注员登录 Langfuse 就能在 Human Annotation 页面看到这个队列。
            user_id="user_A",
        )
        print(f"  ✓ 已指派标注员：{assignment.user_id}")
    except Exception as exc:   # noqa: BLE001 —— 填的是示例 id，被服务端拒绝属预期情况
        print(f"  ⚠️ 指派标注员失败（{type(exc).__name__}）：{str(exc)[:110]}")
        print("     user_id 必须是本项目里真实存在的成员 id"
              "（控制台 Settings → Members 查看）。")
        print("     队列与样本都已建好，这一步不影响后续演示。")
    return queue.id


# 人工打分在 SDK 侧长什么样：没接服务端、或部署形态不支持取分接口时用它做示例。
EXAMPLE_SCORES = [
    {"traceId": "<trace-id>", "name": "回答准确性", "value": 1.0,
     "dataType": "NUMERIC", "source": "ANNOTATION", "comment": "答对了，但没给出单位"},
    {"traceId": "<trace-id>", "name": "回答是否有害", "value": 0,
     "dataType": "BOOLEAN", "source": "ANNOTATION", "comment": ""},
]


def print_example_scores() -> None:
    """打印「人工分数长什么样」的示例结构（两条降级路径共用）。"""
    print("    人工在 UI 上打完分之后，SDK 侧取到的数据结构形如：")
    print(json.dumps(EXAMPLE_SCORES, ensure_ascii=False, indent=2))


def read_queue(queue_id: str) -> None:
    """标注做完之后，怎么把结果取回来做分析。"""
    print("\n" + "=" * 72)
    print("第六步：取回人工分数（分数回流到 trace，可按 name 聚合看板）")
    print("=" * 72)

    if LANGFUSE_READY:
        # 队列里各条的状态
        items = langfuse.api.annotation_queues.list_queue_items(queue_id=queue_id)
        for item in getattr(items, "data", []):
            print(f"  {item.object_id}  {item.status}")

        # 按分数名取人工打出来的分（source=ANNOTATION 表示来自标注队列）
        # source=ANNOTATION 是「人工打的分」与「代码打的分」唯一的区分字段，筛选时用它。
        # ⚠️ 实测踩坑：Langfuse **v4 的 events_only 模式**下这条 REST 路径不存在 ——
        #    404 `This endpoint is not available on deployments running in Langfuse v4
        #    events_only mode.`（v4 把分数改走事件通道了）。
        #    这里兜住并打印数据结构：让读者知道「是部署形态不支持这条路径」，
        #    而不是让整份演示崩在最后一步。
        try:
            scores = langfuse.api.scores.get_many(name="回答准确性", limit=10)
        except Exception as exc:   # noqa: BLE001 —— 部署形态差异，不该把演示炸掉
            print(f"  ⚠️ 按分数名取分失败（{type(exc).__name__}）：{str(exc)[:120]}")
            print("     本机 Langfuse 是 v4 events_only 模式，/api/public/v2/scores 不可用；")
            print("     分数在 UI 的 Human Annotation 页面能正常看到，程序取数要走 v4 的 metrics 接口。")
            print_example_scores()
        else:
            for score in getattr(scores, "data", []):
                print(f"  trace={score.trace_id}  {score.name}={score.value}  comment={score.comment}")
        langfuse.flush()
    else:
        print("    [降级] 本应调用  GET /api/public/annotation-queues/{queueId}/items")
        print("    [降级] 本应调用  GET /api/public/v2/scores?name=回答准确性")
        print_example_scores()


# ---------- 3. 数据：把标注结果沉淀成数据集 ----------
DATASET_NAME = "agent_golden_set"


def build_dataset_from_annotation() -> list:
    """标注过的高价值样本 → 数据集。数据集是回归测试的唯一事实来源。"""
    print("\n" + "=" * 72)
    print("数据：Dataset（测试集）与 Dataset Run（一次实验），对应课案「数据」页")
    print("=" * 72)

    golden_items = [
        # 把标注结论固化成测试项：input 复现问题，expected_output 是人工确认的标准答案。
        {"input": "帮我计算 15 * 8 + 23", "expected_output": "143",
         "metadata": {"来源": "标注队列", "问题类型": "consult"}},
        {"input": "北京今天天气怎么样？", "expected_output": "北京今天晴，气温 30°C，湿度 45%",
         "metadata": {"来源": "标注队列", "问题类型": "other"}},
    ]

    api_call(
        "/api/public/datasets",
        (langfuse.create_dataset if LANGFUSE_READY else None),
        "dataset-local-001",
        name=DATASET_NAME,
        description="人工标注沉淀下来的黄金测试集，用于每次改动的回归评估",
    )
    print(f"  ✓ 数据集已建：{DATASET_NAME}")

    for item in golden_items:
        api_call(
            "/api/public/dataset-items",
            (langfuse.create_dataset_item if LANGFUSE_READY else None),
            "dataset-item-local",
            dataset_name=DATASET_NAME,
            **item,
        )
        print(f"  ✓ 测试项已加入：{item['input']}  →  期望 {item['expected_output']}")

    if LANGFUSE_READY:
        return langfuse.get_dataset(DATASET_NAME).items
    return [SimpleNamespace(**item) for item in golden_items]


# ---------- 4. 真跑一次「数据集运行」 ----------
def run_dataset_experiment(items) -> None:
    """课案「数据」页那张对比图背后就是这一步：同一个数据集跑一遍，逐条记录输出与分数。"""
    print("\n" + "=" * 72)
    print("Dataset Run：拿数据集跑一遍系统，逐条记录「实际输出 + 分数」")
    print("=" * 72)

    run_items = []
    for item in items:
        answer = llm.invoke(item.input).content            # 被测系统（换成你的 Agent 流程）
        score = 1.0 if str(item.expected_output) in answer else 0.0
        run_items.append({
            "datasetItemId": f"<{item.input[:12]}…>",
            "runName": "baseline-v1",                      # 换个 run_name 再跑一次就能并排对比
            "traceId": "<本次运行的 trace id>",
            "output": answer[:60] + ("…" if len(answer) > 60 else ""),
            "scores": [{"name": "keyword_match", "value": score}],
        })
        # run_name 是并排对比的关键：换个名字再跑一次，两次 run 就能在 UI 上对照。
        print(f"  Q: {item.input}")
        print(f"     实际输出: {answer[:60]}…")
        print(f"     期望输出: {item.expected_output}")
        print(f"     分数: {score}")

    # 把这一轮 run 的每条记录打出来 —— 它就是 UI 上「数据集 → Runs」页看到的内容。
    print("\n  这次运行产生的 dataset run item（UI 上「数据集 → Runs」看到的就是它）：")
    print(json.dumps(run_items, ensure_ascii=False, indent=2))
    print("\n  换个 run_name（比如换个提示词或模型）再跑一遍，两次 run 就能并排对比 ——")
    print("  这就是「改完到底有没有变好」的客观依据，而不是凭感觉说「好像好一点」。")

    if LANGFUSE_READY:
        langfuse.flush()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # 入口：先讲清 Langfuse 怎么配，再走「打印报文 + 真跑一次」的降级流程。
    print("=" * 72)
    print("标注（Annotation Queue）与数据（Dataset）—— 人工评估与回归测试集")
    print("=" * 72)

    if not LANGFUSE_READY:
        print("\n【进入降级演示】Langfuse 密钥为空")
        print("  settings.langfuse_public_key = '' ，settings.langfuse_secret_key = ''")
        print("-" * 72)
        # 按课案「安装」一节的顺序给出四步，照着做就能看到真实的标注队列与数据集页面。
        print("要看到真实的标注队列与数据集页面，按课案「安装」一节准备环境：")
        print("  1) git clone https://github.com/langfuse/langfuse.git")
        print("     cd langfuse")
        print("     docker compose up -d          # 启动后访问 http://localhost:3000")
        print("  2) 首次注册的账号即为管理员；新建项目 → Settings → API Keys → 创建密钥")
        print("  3) 写入 F:\\ProGram\\Python_Base\\.env ：")
        print("        LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("        LANGFUSE_SECRET_KEY=sk-lf-...")
        print("        LANGFUSE_HOST=http://localhost:3000    # 本地 docker 部署用这个")
        print("  4) 重新运行本脚本，去 Human Annotation / Datasets 两个页面看结果")
        print("-" * 72)
        # 上面讲「怎么配」，下面讲「不配也能跑」—— 学员不会误以为必须先配好才能运行。
        print("下面不依赖 Langfuse 服务：把每一步的 REST 报文打印出来，")
        print("并真跑一次大模型产生「待标注的样本」和「数据集运行」记录。")
        print("=" * 72)
    else:
        print("Langfuse 已配置，标注与数据集将写入：", settings.langfuse_host)

    # 真实场景里，下面的 trace_ids 来自「自动打分偏低」的那批样本；
    # 这里用两条模拟 id 走通流程。
    configs = create_score_configs()
    queue_id = build_annotation_queue(configs, ["trace-low-001", "trace-low-002"])
    read_queue(queue_id)
    dataset_items = build_dataset_from_annotation()
    run_dataset_experiment(dataset_items)

    # 四步串起来就是闭环：自动指标找问题 → 人工给答案 → 数据集固化 → 下次改动回归。
    print("\n小结：自动指标负责「大规模找问题」，人工标注负责「给出标准答案」，")
    print("      数据集负责「把答案固化下来，下次改动拿它回归」——三者缺一不成闭环。")
