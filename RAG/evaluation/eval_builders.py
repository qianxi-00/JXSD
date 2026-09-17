"""评估集构造器:对照优化篇课案 `script/build_eval_set.py`。

课案的四类场景与硬性要求:
- 日期/人员筛选(单人单票事实型):只取 (person, ticket_type) 恰好一张的组,保证期望集无歧义;
- 多票据归因:两人同票种金额对比,期望同时召回两张票;
- 金额汇总:2~4 张的组,标准答案由 amount_fen 求和确定性计算;
- 缺失证据拒答:库里没有证据的问题,期望如实回答未找到(relevant_ids 为空)。
标准答案**全部由真实字段拼装**,不允许 LLM 生成;构造器是纯函数,幂等可重跑。

本模块只做"把票据行变成样本",取数(Milvus/PG)由 script/build_eval_set.py 负责,
这样构造逻辑可以完全离线测试。

为什么把构造逻辑单独拆成一个模块(而不是直接写在 build_eval_set.py 里):
取数要连 Milvus/PG(有外部依赖、跑一次慢),构造是纯函数(无依赖、可重复)。
拆开之后 `tests/test_eval_builders.py` 能喂几条假票据离线钉住「四类场景各生成什么」,
不必起任何服务。也正因构造是纯函数,样本可幂等重跑 —— 同样的票据输入永远得到
同样的评估集,评估结果才可跨天对比。

⚠ 精度提醒:标准答案**不允许** LLM 参与。LLM 生成的标准答案会引入「模型自己批改
自己」的循环,指标虚高且漂移。这里的金额来自 `amount_fen` 求和、日期来自 `date_int`
切片、票号/行程直接取字段 —— 全部可复算。
"""

from __future__ import annotations

# 票种英文枚举 → 中文问法。用 .get(ticket_type, ticket_type) 兜底:
# 出现没登记的票种时问题里直接用原枚举(难看但不会 KeyError 崩掉整批构造)
TYPE_ZH = {"flight": "机票", "train": "火车票", "invoice": "发票"}

# 各场景样本数量目标(与课案一致)
# 这三个常量是**目标上限**不是保证值:实现里遇到字段缺失会 continue 跳过,
# 所以实际条数可能少于目标(见 build_direct / build_aggregation 内的 continue)
N_DIRECT = 18
N_MULTI = 4
N_AGG = 6

# 通用问题:不该检索,应由大模型直接回答(课案 OUT_OF_SCOPE_QUESTIONS)
DIRECT_ANSWER_QUESTIONS = [
    "今天天气怎么样？",
    "推荐几本好看的小说。",
    "红烧肉怎么做才好吃？",
    "2026年世界杯冠军是谁？",
]

# 票据域内但知识库没有证据的问题:应检索后如实拒答
# 与上面 DIRECT_ANSWER_QUESTIONS 的区别是考点不同:那组考「路由对不对」(不该检索),
# 这组考「拒答诚不诚实」(检索了但没证据,要承认没找到)——用同一批问题测不出两种行为
REFUSAL_QUESTIONS = [
    "公司食堂装修花了多少钱？",
    "公司团建费用报销了吗？",
    "办公室绿植采购一共花了多少钱？",
    "去年的年会场地费是多少？",
]

# 拒答的标准答案。判分时是**字符串精确比对**,所以这句话既是样本答案也是判分基线,
# 改它等于改判分口径(答案事实准确率会跟着变),要同步看 stage_metrics 的判定方式
REFUSAL_ANSWER = "未找到相关票据信息。"


def fen_to_yuan(amount_fen: int) -> str:
    """分转元,保留两位小数。

    库里金额统一存「分」(int),样本答案统一写「元」的两位小数字符串。
    为什么不用 round(x/100, 2):浮点 round 会出现 0.1+0.2 类的表示误差,
    而 f-string 的 `:.2f` 直接按十进制格式化,结果稳定可复现 —— 评估集是要 diff 的文件,
    输出必须逐字稳定。
    纯整数运算的等价写法是 `f"{fen // 100}.{fen % 100:02d}"`,这里保留除法版以减少改动。
    """
    return f"{amount_fen / 100:.2f}"


def date_str(date_int: int) -> str:
    """date_int(如 20251014)转 ISO 日期字符串。

    库里日期是**整数** `YYYYMMDD`(不是 date 类型、不是字符串),所以这里按位切片:
    前 4 位年、中间 2 位月、后 2 位日。用切片而不是 datetime 解析,是为了不引入
    「非法日期抛异常」这条失败路径 —— 构造器批量跑 300 条票据时,一条脏数据
    不应中断整批。代价是 20251332 这种非法日期会被原样拼出来(见报告)。
    """
    s = str(date_int)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def group_by_person_type(tickets: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """按 (person, ticket_type) 分组,跳过 person 为空的记录;组内按 id 排序。

    这个分组是四类场景的**共同地基**,分组的粒度直接决定期望集有没有歧义:
    - 组大小 == 1 → 可以问「某某的机票票号是多少」,答案唯一(build_direct 用);
    - 组大小 2~4 → 可以问「一共花了多少钱」,答案由求和确定(build_aggregation 用);
    - 组大小 > 1 且不问汇总 → 同一个问题可能对应多张票,期望集天然有歧义,
      指标永远算不满,所以 build_direct 只取单元素组。

    跳过 person 为空:没人名就问不出「某某的...」这种自然问法,且拿空字符串当
    过滤条件会把多个人混成一组。

    组内按 id 排序 + 最后按 key 排序 = **确定性**:字典本身保持插入序(3.7+),
    但插入序取决于上游取数顺序(Milvus/PG 的返回顺序不保证稳定)。不显式排序的话,
    同一天跑两次可能得到不同的样本顺序甚至不同的取样(因为 limit 是「取前 N 个」),
    评估结果就无法对比了。
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for ticket in tickets:
        if not ticket.get("person"):
            continue
        key = (ticket["person"], ticket["ticket_type"])
        groups.setdefault(key, []).append(ticket)
    for items in groups.values():
        items.sort(key=lambda item: item["id"])
    return dict(sorted(groups.items()))


def make_sample(
    question: str,
    tickets: list[dict],
    ground_truth: str,
    task_type: str,
    expected_route: str = "search",
    expected_filter: dict | None = None,
) -> dict:
    """拼装一条评估样本(字段顺序固定,保证输出稳定)。

    两个字段是从 tickets 推导的,别手填:
    - `expected_ticket_ids` = 传进来的每张票的 id,顺序 = 传参顺序(组内已按 id 排序);
    - `need_citation` = `bool(tickets)`:有票据就必须带引用,没有票据(拒答/直答)
      自然不需要引用。把「要不要引用」和「有没有票据」绑定,避免出现
      「期望带引用但期望集为空」这种自相矛盾的样本。

    注意 expected_filter 默认 None 而不是 {}:这里的 `or {}` 让调用方可以显式传 None,
    也顺手挡掉传空 dict 的情况 —— 两种写法都表示「这条样本不约束筛选字段」。
    """
    return {
        "question": question,
        "task_type": task_type,
        "expected_route": expected_route,
        "expected_filter": expected_filter or {},
        "expected_ticket_ids": [ticket["id"] for ticket in tickets],
        "ground_truth": ground_truth,
        "need_citation": bool(tickets),
    }


def _filter_of(ticket: dict) -> dict:
    """用票据字段拼期望筛选条件(确定性)。

    为什么只能拼这两项:票据行里存在的、且 `pipeline/filters.py` 真会去抽的字段
    才有资格写进期望 —— 期望里多写一个字段(比如 route),而链路抽不出来,
    「筛选字段准确率」就会永远扣分,分不清是链路问题还是样本问题。

    已知口径问题(不在本次改动范围):route 字段在库里是**拼音/英文站名**
    (`Hefei-Wulumuqi`),而 filters 抽的是中文站名,两者语言不一致 ⇒ route 条件
    永远不会命中。所以这里刻意不把 route 放进期望筛选,否则这条指标一律为 0。

    ticket_type 与 person 都用 `if ticket.get(...)` 判空后再写:空字符串字段
    放进期望筛选会变成「必须等于空串」的硬条件,把本该命中的票排除掉。
    """
    filters: dict = {}
    if ticket.get("ticket_type"):
        filters["ticket_type"] = ticket["ticket_type"]
    if ticket.get("person"):
        filters["person"] = ticket["person"]
    return filters


def build_direct(groups: dict[tuple[str, str], list[dict]], limit: int = N_DIRECT) -> list[dict]:
    """日期/人员筛选:单人单票种的事实型问题(票号/金额/日期/路线四种问法轮换)。

    三个容易看漏的实现细节:

    1. **问法轮换用的是「组号」不是「样本号」**(`index % 4`):`index` 是 singles 的
       下标、在 continue 时也会前进。结果是——某类字段普遍缺失时,该类问法会整体
       被跳空,最终可能连续好几条都是「票号是多少」。这是**刻意的稀疏取样**,
       换来的是四类问法在 300 条票据里均匀铺开;不要误以为它是「样本计数器」。

    2. **四种问法各要求一个必填字段**(票号/金额/日期/路线),缺字段就 `continue`
       跳过该组。所以实际条数可能 < limit=N_DIRECT(18),而**任务类型仍记为
       "retrieval"** —— 这四条问法考的都是「能不能靠筛选+召回找回那一张票」。
       票据域里 flight 的日期/金额是 0 覆盖(登机牌上没有年份与金额),
       所以命中 kind==1/2 的机票组一定会被跳过。

    3. **task_type 用 "retrieval"** 与「直接回答」区分:本仓曾经把 4 条通用直答问题
       标成 task_type="faq"(见 build_direct_answer),那与这里的 retrieval 不是一回事。

    期望筛选条件用 `_filter_of(ticket)` 而不是从问题文本反推 —— 问题文本是给人看的,
    字段是给指标算的,后者才是唯一真相。
    """
    singles = [
        (key, items[0])
        for key, items in groups.items()
        if len(items) == 1 and items[0].get("ticket_no")
    ]

    samples: list[dict] = []
    for index, ((person, ticket_type), ticket) in enumerate(singles):
        if len(samples) >= limit:
            break
        zh = TYPE_ZH.get(ticket_type, ticket_type)
        kind = index % 4

        if kind == 0:
            # 票号问法:上面的 singles 已经保证 ticket_no 非空,这里不再判
            question = f"{person}的{zh}票号是多少？"
            answer = f"{person}的{zh}票号是{ticket['ticket_no']}。"
        elif kind == 1:
            # 金额问法:用 `not ticket.get("amount_fen")` 判,0 分也算缺失 ——
            # 0 元票据没有提问价值(且多半是 OCR 没抽到),跳过比生成「花了 0.00 元」更好
            if not ticket.get("amount_fen"):
                continue
            question = f"{person}的{zh}花了多少钱？"
            answer = f"{person}的{zh}金额是{fen_to_yuan(ticket['amount_fen'])}元。"
        elif kind == 2:
            if not ticket.get("date_int"):
                continue
            question = f"{person}的{zh}是哪天的？"
            answer = f"{person}的{zh}日期是{date_str(ticket['date_int'])}。"
        else:
            if not ticket.get("route"):
                continue
            question = f"{person}的{zh}是从哪到哪的？"
            answer = f"{person}的{zh}行程是{ticket['route']}。"

        samples.append(
            make_sample(question, [ticket], answer, "retrieval", expected_filter=_filter_of(ticket))
        )
    return samples


def build_multi(groups: dict[tuple[str, str], list[dict]], limit: int = N_MULTI) -> list[dict]:
    """多票据归因:两人同票种金额对比(机票/火车票各取,交替补足)。

    期望集是**两张票**(make_sample 传 [ticket_a, ticket_b])—— 这条场景考的是
    「一次召回能不能同时带回两个人的票」,只召回一张算失败。

    两个实现细节:
    - 只取 `len(items) == 1` 的单票组:从「一个人 2~4 张票」的组里取会撞上
      build_aggregation 的场景(汇总 vs 对比),而且「张三的机票」到底指哪一张说不清;
    - 遍历顺序是 flight 在前、train 在后,组内 `range(0, len-1, 2)` 两两配对 ⇒
      人数是奇数时最后一个人落单(不生成样本)。落单不会报错,只是样本少一条,
      且因 groups 已排序所以**每次都是同一个人落单**,可复现。

    `limit` 命中时直接 return(不是 break):两个票种的循环是嵌套的,
    break 只能退出内层,会继续拿火车票去凑数。
    """
    singles_by_type: dict[str, list[tuple[str, dict]]] = {"flight": [], "train": []}
    for (person, ticket_type), items in groups.items():
        if ticket_type in singles_by_type and len(items) == 1 and items[0].get("amount_fen"):
            singles_by_type[ticket_type].append((person, items[0]))

    samples: list[dict] = []
    for singles in (singles_by_type["flight"], singles_by_type["train"]):
        for index in range(0, len(singles) - 1, 2):
            if len(samples) >= limit:
                return samples
            (person_a, ticket_a), (person_b, ticket_b) = singles[index], singles[index + 1]
            zh = TYPE_ZH.get(ticket_a["ticket_type"], ticket_a["ticket_type"])
            question = f"{person_a}和{person_b}的{zh}分别花了多少钱？"
            answer = (
                f"{person_a}的{zh}{fen_to_yuan(ticket_a['amount_fen'])}元，"
                f"{person_b}的{zh}{fen_to_yuan(ticket_b['amount_fen'])}元。"
            )
            # task_type 用 "generation":虽然也要检索,但考点是「生成时有没有把两个人
            # 的金额对上号」—— 答反/漏一个都是错的,所以归到生成质量类
            samples.append(make_sample(question, [ticket_a, ticket_b], answer, "generation"))
    return samples


def build_aggregation(groups: dict[tuple[str, str], list[dict]], limit: int = N_AGG) -> list[dict]:
    """金额汇总:单人单票种 2~4 张的总金额与张数(标准答案由分数求和)。

    组大小卡在 2~4 张:`>= 2` 才是「汇总」(1 张归 build_direct 的事实型问题),
    `<= 4` 是为了让答案可人工核对 —— 一次汇总十几张票时,标准答案虽然算得出,
    但任何一处漏票都难定位,且超过 4 张的场景与「2~4 张」的考点不同。

    `any(not item.get("amount_fen") for item in items)` 是「整组要么全有金额、要么
    整组跳过」:只要有一张缺金额求和就偏小,而标准答案会把这个错值当成正确答案,
    以后每次评估都扣链路的分 —— 宁可不要这条样本。

    期望筛选条件取 `_filter_of(items[0])`:同组内 person/ticket_type 必然相同
    (它们就是分组的 key),取任意一张都一样;写成 items[0] 只是表达「这组共用一套筛选」。
    """
    samples: list[dict] = []
    for (person, ticket_type), items in groups.items():
        if len(samples) >= limit:
            break
        if not 2 <= len(items) <= 4:
            continue
        if any(not item.get("amount_fen") for item in items):
            continue

        zh = TYPE_ZH.get(ticket_type, ticket_type)
        total = fen_to_yuan(sum(item["amount_fen"] for item in items))
        question = f"{person}的{zh}一共报销了多少钱？"
        # 答案带张数(「共 N 张,合计 X 元」):只给金额的话,少召回一张票也可能
        # 恰好凑出碰巧一样的数字,把错误答案判成对;带上张数能多一重约束
        answer = f"{person}的{zh}共{len(items)}张，合计{total}元。"
        samples.append(
            make_sample(question, items, answer, "structured_aggregation", expected_filter=_filter_of(items[0]))
        )
    return samples


def build_refusal() -> list[dict]:
    """缺失证据拒答:票据域内但库里没有证据的问题。

    期望票据集为空 + 期望答案固定为 REFUSAL_ANSWER ⇒ 这条样本同时钉两件事:
    「别乱编」和「别把没找到说成找不到就别答了」。task_type 记 "generation"
    (考最终回答的措辞),路由仍是 search(必须先检索才知道没证据)。
    """
    return [
        make_sample(question, [], REFUSAL_ANSWER, "generation")
        for question in REFUSAL_QUESTIONS
    ]


def build_direct_answer() -> list[dict]:
    """通用问题:期望路由判定为直接回答(不检索)。

    expected_route="direct" 是这条场景的**唯一考点**:问「今天天气」不该去查票据库。

    ⚠ 口径注意:这里把通用直答问题标成了 task_type="faq",而它们并不在
    `data/preset_qa.json` 的 FAQ 层里(实测 eval_set ∩ preset_qa = ∅)。也就是说
    「faq」这个类型在当前实现里**既不覆盖 FAQ 快速返回层,也不能用来验收它**,
    只是被当成「不检索的一类」在用。要真正验收 FAQ 层,得另外构造走预设问法的样本
    (见报告「发现的问题」)。
    """
    return [
        make_sample(question, [], "", "faq", expected_route="direct")
        for question in DIRECT_ANSWER_QUESTIONS
    ]


def build_all(tickets: list[dict]) -> list[dict]:
    """按课案顺序汇总五类场景样本。

    顺序固定为 直接/多票/汇总/拒答/直答 —— 顺序不只是好看:`run_stage_eval --limit N`
    是「取前 N 条」,顺序稳定才能让不同人、不同天跑的 --limit 结果可比。
    ground_truth 为空字符串的两类(拒答/直答)也照样进样本:它们考的是路由与拒答行为,
    不该因为「没有标准答案」就被过滤掉。
    """
    groups = group_by_person_type(tickets)
    return (
        build_direct(groups)
        + build_multi(groups)
        + build_aggregation(groups)
        + build_refusal()
        + build_direct_answer()
    )


# 模块自检:用三条硬编码票据跑一遍全部构造器。断言只钉两条最关键的契约
# (汇总样本的期望集 = 两张票 / 存在 direct 路由样本),够用来发现「构造器整体跑不出来」
# 这类问题,又不会因为样本条数微调就红。
if __name__ == "__main__":
    demo = [
        {"id": "t1", "person": "张三", "ticket_type": "train", "ticket_no": "T1",
         "date_int": 20250305, "amount_fen": 43600, "route": "北京-上海"},
        {"id": "t2", "person": "张三", "ticket_type": "train", "ticket_no": "T2",
         "date_int": 20250321, "amount_fen": 40000, "route": "上海-北京"},
        {"id": "t3", "person": "李四", "ticket_type": "flight", "ticket_no": "F1",
         "date_int": 20250401, "amount_fen": 128000, "route": "A-B"},
    ]
    samples = build_all(demo)
    assert any(s["expected_ticket_ids"] == ["t1", "t2"] for s in samples), samples
    assert any(s["expected_route"] == "direct" for s in samples)
    print(f"eval_builders.py 自检通过,共 {len(samples)} 条样本")
