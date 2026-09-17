"""按基础篇课案「系统评估 · 样本生成」生成评估样本。

流程(课案原文):
    OCR 上下文 -> LLM 生成 {"事实型问题", "答案"}
    -> 两道质检:问题能否仅凭上下文回答 / 问题是否独立完整
    -> 两项都 >= 5 分才收进评估集,否则最多重试 3 次

课案强调:大模型可以辅助扩写问法,但标准答案必须由票据原始字段或人工确认结果给出。
因此这里只用票据自己的结构化字段做"确定性事实"(票号/金额/日期/人员/类型),
生成的"答案"仅在质检通过后作为 ground_truth 的候选,写入的样本仍保留 ticket_id,
方便人工复核时对回原始票据。

★ **本脚本与 `script/build_eval_set.py` 是两条互斥的样本生成路线,别混用**:
- `build_eval_set.py`:纯确定性(金额求和、日期切片),**零 LLM 成本**、可无限重复;
  但只能生成"字段能直接算出来"的问法;
- 本脚本:LLM 从 OCR 全文出题 + 两道质检,**每题约 3 次 LLM 调用**;
  能覆盖"字段算不出来"的问题(如行程目的、备注信息),但答案来自模型 ⇒
  **必须人工复核**才能当基准。
本项目实际使用的评估集(`data/eval_set.jsonl`)走的是第一条路线;
本脚本的产物是 `data/generated_qa.jsonl`,是**候选池**,不直接进评估。

⚠ 由此产生的口径问题:`expected_filter` 里的 `date_start/date_end` 用年度区间,
而 `pipeline/filters.py` 抽出来的日期字段口径未必一致(见报告),
用本脚本产出的样本做筛选准确率统计时要留意。

用法:
    uv run python RAG/script/generate_eval_samples.py --limit 20
    uv run python RAG/script/generate_eval_samples.py --limit 20 --out RAG/data/generated_qa.jsonl
"""

# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from config import settings
from core.database import get_milvus_client
from core.logger import logger
from core.prompts import (
    QA_GENERATION_PROMPT,
    QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT,
    QUESTION_STANDALONE_CRITIQUE_PROMPT,
)
from evaluation.eval_set import save_jsonl, validate_sample
from evaluation.sample_gen import parse_critique, passes_quality

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "generated_qa.jsonl"
# 最多重试 3 次(课案口径)。3 次是个折中:失败多半是**这次采样的运气**
# (LLM 偶尔生成一个上下文答不出的问题),重试通常能过;
# 但若提示词/模型本身有问题,3 次也足够暴露(每次都失败),不会无限烧钱
MAX_ATTEMPTS = 3

# 字段清单:比 build_eval_set 多了 `ocr_text`(**出题必须**),
# 少了 counterparty(出题用不上)。ocr_text 是上万字的大字段,
# 但本脚本本来就要把上下文喂给 LLM,不可避免
TICKET_FIELDS = ["id", "ticket_type", "ticket_no", "person", "date_int", "amount_fen", "route", "ocr_text"]


def _client() -> OpenAI:
    # 显式超时：SDK 默认 600 秒 × 重试 3 次，上游挂住时样本生成会卡死
    return OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
    )


def _chat(client: OpenAI, prompt: str, temperature: float = 0.3) -> str:
    """单轮对话的薄封装,统一返回正文(空则空串)。

    `temperature` 由调用方传:出题用 0.3(要一点多样性)、质检用 **0**(判分要可复现)。
    默认值 0.3 是为出题准备的 —— 调用质检时两处都显式传了 temperature=0。

    `resp.choices[0].message.content or ""`:思考型模型在只给 reasoning_content 不给
    正文时会返回 None,不兜住就会在下面的 `json.loads(None)` 处崩
    (正确做法是像 `llm/chat.py` 的路由那样显式关思考并给足预算,见报告)。
    """
    resp = client.chat.completions.create(
        model=settings.llm.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def load_tickets(limit: int | None) -> list[dict]:
    """从 Milvus 拉票据(只取有 ocr_text 的),limit 为 None 时取全部。

    `filter='ticket_type in [...]'` 用**白名单**而不是 `!= ""`:
    出题需要知道票种才能套对提示词,而且未知票种的 OCR 版面没验证过。

    两层过滤的顺序值得注意:
    1. **Milvus 侧**只按票种过滤(不按 ocr_text 是否为空 —— Milvus 对长文本的
       `!= ""` 判断在不同版本上行为不一致,放在 Python 侧更稳);
    2. **Python 侧** `(r.get("ocr_text") or "").strip()` 过滤空/纯空白 OCR:
       没有正文就无从出题。`or ""` 先兜住 None,再用 strip() 兜住纯空白 ——
       只写 `if r.get("ocr_text")` 会让 `"   \n"` 这种"看着有值"的记录漏过去。

    `rows[:limit] if limit else rows`:`limit` 是 None 或 0 时取全部。
    注意 Milvus 的 `limit=10000` 是硬上限,超过会**静默截断**。
    """
    client = get_milvus_client()
    rows = client.query(
        collection_name=settings.milvus.collection,
        filter='ticket_type in ["flight", "invoice", "train"]',
        output_fields=TICKET_FIELDS,
        limit=10000,
    )
    rows = [r for r in rows if (r.get("ocr_text") or "").strip()]
    return rows[:limit] if limit else rows


def expected_filter_of(ticket: dict) -> dict:
    """用票据自身的结构化字段拼出"期望筛选条件"(确定性,不依赖模型)。

    比 `eval_builders._filter_of` 多一项**日期区间**:
    - `date_int // 10000` 取年份,然后拼出 `YYYY0101` ~ `YYYY1231` 的整年区间。
      用整除而不是 str 切片,是因为 date_int 可能来自不同源(有的带前导零问题);
    - 这里只给"整年"粒度,是因为问法里通常只说"今年的票",没有更细的日期线索。

    ⚠ 这条 date_start/date_end 期望**依赖 filters 侧也抽同名字段**才算得对
    (见文件头与报告);若 filters 不抽日期,这两个字段会被统计成"没命中",
    拉低筛选准确率 —— 那是**样本口径问题**,不是链路问题。
    """
    filters: dict = {}
    if ticket.get("ticket_type"):
        filters["ticket_type"] = ticket["ticket_type"]
    if ticket.get("person"):
        filters["person"] = ticket["person"]
    if ticket.get("date_int"):
        year = int(ticket["date_int"]) // 10000
        filters["date_start"] = year * 10000 + 101
        filters["date_end"] = year * 10000 + 1231
    return filters


def generate_one(client: OpenAI, ticket: dict) -> dict | None:
    """为一条票据生成并质检一个样本;不通过返回 None。

    **每次尝试 = 3 次 LLM 调用**(出题 + 两道质检),所以 MAX_ATTEMPTS=3 时
    最坏一条票据 9 次调用 —— `--limit 20` 最坏 180 次。跑之前先想清楚成本。

    失败的**两类**原因被分开处理(这是这段控制流的关键):
    - **JSON 解析失败** ⇒ `continue`:重试**出题**。模型这次没按格式输出;
    - **质检不通过** ⇒ `continue`:重试**出题**。问题不达标,换一个问法;
    两类都走同一条 continue 路径,因为对策都是"重新出题",区别只在日志。

    `raw.strip().strip("`").removeprefix("json").strip()` 是三层清洗:
    剥首尾空白 → 剥 markdown 围栏反引号 → 去掉 `json` 语言标记 → 再剥一次空白。
    与 `sample_gen.parse_critique` 的 JSON 分支写法一致(那边解析返回值,这边解析正文)。

    两次质检**刻意用 temperature=0**:质检是判分行为,要可复现 ——
    同一个问题问两次必须得同一个结论,否则"重试"会变成"换个裁判碰运气"。

    `passes_quality(groundedness, standalone)` 两道都过才算通过(见 sample_gen)。

    最后 `validate_sample({...})` 而不是直接返回裸 dict:
    让生成物**在写盘前**就过评估集契约,避免"生成了一堆字段不全的样本,
    等 build/评估时才炸"。
    """
    context = ticket["ocr_text"]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = _chat(client, QA_GENERATION_PROMPT.format(context=context), temperature=0.3)
        try:
            payload = json.loads(raw.strip().strip("`").removeprefix("json").strip())
            question = str(payload["事实型问题"]).strip()
            answer = str(payload["答案"]).strip()
        except (ValueError, KeyError, TypeError) as exc:
            # 三种异常对应三种坏输出:JSON 坏(ValueError)、键名不对(KeyError)、
            # payload 不是 dict 而是 list/str(TypeError)。
            # 只截 120 字符进日志,避免把整段模型输出灌进日志
            logger.warning(f"[样本] 第 {attempt} 次生成无法解析 JSON({exc}): {raw[:120]}")
            continue

        groundedness = parse_critique(
            _chat(
                client,
                QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT.format(question=question, context=context),
                temperature=0,
            )
        )
        standalone = parse_critique(
            _chat(
                client,
                QUESTION_STANDALONE_CRITIQUE_PROMPT.format(question=question),
                temperature=0,
            )
        )
        if not passes_quality(groundedness, standalone):
            # 日志里把两份质检**原样**打出来(而只打"未通过"):不通过的原因
            # (是解析失败还是真给了低分)只有看了原文才能判断
            logger.info(
                f"[样本] 第 {attempt} 次质检未通过(groundedness={groundedness}, standalone={standalone}):{question}"
            )
            continue

        return validate_sample(
            {
                "question": question,
                "task_type": "generation",
                # 出题用的是 OCR 上下文,但这条样本在评估时走的是**真实检索**链路
                # (要与线上一致),所以期望路由是 search
                "expected_route": "search",
                "expected_filter": expected_filter_of(ticket),
                # 只标一张票(出题就是基于这一张)
                "expected_ticket_ids": [ticket["id"]],
                "ground_truth": answer,
                "need_citation": True,
            }
        )
    # 3 次都没过 ⇒ 返回 None,由 main 打印"跳过"。**不抛异常**:
    # 一条票据出不了题不该让整批挂掉
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="生成并质检票据问答评估样本")
    ap.add_argument("--limit", type=int, default=20, help="最多处理多少条票据")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 jsonl 路径")
    args = ap.parse_args()

    # `args.limit if args.limit > 0 else None`:把 0/负数统一翻译成 None(取全部),
    # 让 --limit 的语义与 build_eval_set/run_stage_eval 保持一致(0 = 全部)
    tickets = load_tickets(args.limit if args.limit > 0 else None)
    print(f"待处理票据 {len(tickets)} 条(Milvus collection={settings.milvus.collection})")

    client = _client()
    samples: list[dict] = []
    for index, ticket in enumerate(tickets, 1):
        sample = generate_one(client, ticket)
        if sample:
            samples.append(sample)
            print(f"[{index}/{len(tickets)}] 收录:{sample['question']}")
        else:
            print(f"[{index}/{len(tickets)}] 跳过:{ticket['id']}(质检未通过)")

    # 即使一条都没生成也照常写文件(空文件)再由下面 exit(1) 报错:
    # 留一个 0 字节产物比"什么都没有"更容易判断"脚本跑了但没出题"
    count = save_jsonl(samples, args.out)
    print(f"已写出 {count} 条样本 -> {args.out}")
    # 一条都没成功 ⇒ **非零退出码**:让 CI/批量脚本能直接发现,
    # 而不是"跑完了、文件是空的、没有任何报错"
    if not samples:
        sys.exit(1)


if __name__ == "__main__":
    main()
