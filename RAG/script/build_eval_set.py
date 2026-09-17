# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
# (顺序与理由同 run_stage_eval.py:先根后 RAG 根,insert 到最前以防同名模块串味)
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""评估集生成脚本(优化篇课案「RAG 评估 · 数据源生成与上传脚本」)。

⚠ 与 run_stage_eval.py 同款问题:docstring 在 import 之后 ⇒ **不是**模块 docstring
(`__doc__` 为 None),只是一段被丢弃的字符串。本次只加注释,不动位置(见报告)。

从 Milvus 真实票据字段**确定性**生成 data/eval_set.jsonl:
按 (person, ticket_type) 分组,覆盖 金额汇总 / 日期人员筛选 / 缺失证据拒答 /
多票据归因 四类场景;标准答案全部由真实字段拼装(金额求和、日期格式化),
不是模型生成;幂等可重跑。

**为什么"确定性 + 幂等"是硬要求**(而不是"用 LLM 造一批更真实的样本"):
LLM 造的样本每题都要人复核,而且每次重生成都会变 —— 样本一变,前后两次评估
就不可比,"这次优化到底有没有用"永远说不清。用真实字段拼装的样本没有这个问题:
同一天花板下,评估集的 diff 只有"票据数据变了"这一个原因。

**这是"离线可测"的分界线**:本脚本负责取数(Milvus),构造逻辑全部在
`evaluation/eval_builders.py`(纯函数)。所以只有本脚本需要 Milvus 才能跑,
`tests/test_eval_builders.py` 可以直接喂假票据离线验证四类样本的构造规则。

用法:
    uv run python RAG/script/build_eval_set.py
    uv run python RAG/script/build_eval_set.py --out RAG/data/eval_set.jsonl
"""

import argparse
from pathlib import Path

from config import settings
from core.database import get_milvus_client
from evaluation.eval_builders import build_all
from evaluation.eval_set import save_jsonl

# 默认输出到 RAG/data/ 下(相对脚本定位),与 CWD 无关
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "eval_set.jsonl"

# 只取构造样本需要的 8 个字段(而不是 SELECT *)。
# 为什么重要:Milvus 里还有 `semantic_text`(几千字)与 `ocr_text`(上万字)两个大字段,
# 全量拉 300 条会把几十 MB 文本读进内存,而构造器一个字都用不上。
# 这份字段清单同时也是"评估依赖的票据字段契约"—— 少一个字段对应的样本类型就会变少
# (如缺 amount_fen 就没有金额汇总样本)
FIELDS = ["id", "ticket_type", "ticket_no", "person", "date_int", "amount_fen", "route", "counterparty"]


def fetch_all_tickets() -> list[dict]:
    """拉取票据全量结构化字段,按主键排序保证确定性。

    两个"确定性"设计:
    - `filter='ticket_type != ""'`:排除没有票种的行(空票种无法分组、也问不出
      「某某的机票」这种问题)。注意 Milvus 的字符串过滤用 **`!= ""`**(双引号),
      这是 Milvus 表达式的语法要求,不是笔误;
    - `sorted(rows, key=lambda row: row["id"])`:Milvus 的 `query` **不保证返回顺序**
      (即使有 limit 也只保证"这些行",不保证次序)。不排序的话每次跑出的
      eval_set.jsonl 行序都可能不同,而 build_direct/build_aggregation 是"取前 N 个"
      ⇒ **样本集会变**,历史评估结果不可比。

    `limit=10000` 是硬上限而不是分页:当前库里约 300 条,10 倍余量足够;
    超过 10000 条后这里会**静默截断**(不会报错),届时评估集只覆盖一部分票据
    (见报告)。真要上万条需改为分页查询。
    """
    client = get_milvus_client()
    rows = client.query(
        collection_name=settings.milvus.collection,
        filter='ticket_type != ""',
        output_fields=FIELDS,
        limit=10000,
    )
    return sorted(rows, key=lambda row: row["id"])


def main() -> None:
    ap = argparse.ArgumentParser(description="从 Milvus 确定性生成 RAG 评估集")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 jsonl 路径")
    args = ap.parse_args()

    tickets = fetch_all_tickets()
    # 构造 → 落盘。`save_jsonl` 内部会对每条样本再过一次 validate_sample,
    # 所以"生成出来但字段不合法"的样本会在写盘前就报错,不会污染评估集
    samples = build_all(tickets)
    count = save_jsonl(samples, args.out)

    # 按 task_type 统计(用普通 dict 累加,不引入 Counter):
    # 这个分布是**判读报告时第一个该看的东西** —— 比如 faq 类为 0
    # 说明 FAQ 快速返回层根本没被样本覆盖,FAQ 相关结论无样本支撑
    by_type: dict[str, int] = {}
    for sample in samples:
        by_type[sample["task_type"]] = by_type.get(sample["task_type"], 0) + 1

    # 同时打印"票据条数"与"样本条数":两者差距过大(如 300 → 3)说明
    # 分组条件把大部分票据过滤掉了(如 person 为空),该去查数据而不是调构造器
    print(f"票据 {len(tickets)} 条 -> 样本 {count} 条")
    for task_type, number in sorted(by_type.items()):
        print(f"  {task_type}: {number}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
