"""阈值标定第一步:为预设问答生成同义问法(正例)。

课案(基础篇「评测集生成」)用 SimBERT / 往返翻译生成正例,本项目全部走外部 API,
因此改成 LLM 生成同义问法;生成结果只是"同义改写",标准答案仍来自 preset_qa.json,
不引入模型编造的内容。

**为什么正例要"生成"而不是人工写**:阈值标定需要每个 query 有若干条
"意思相同、说法不同"的正例来测算 recall 侧;人工写 20 个 query × 5 条太慢,
而 LLM 改写质量足够(这一步**只生成问法,不生成答案** —— 答案仍取预设的标准答案,
所以不存在"模型编造内容"的风险,这是与"用 LLM 造评估集"的关键区别)。

⚠ 这一步的产物只是**正例**;难负例在 `script/search_threshold.py` 里用 embedding
按相似度挖(见 evaluation/threshold.py 的说明)。两者合起来才是标定输入。

用法:
    uv run python RAG/script/build_threshold_dataset.py --limit 20
    uv run python RAG/script/build_threshold_dataset.py --limit 20 --out RAG/data/threshold_dataset.json
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
from pathlib import Path

from openai import OpenAI

from config import settings
from core.logger import logger
from evaluation.threshold import parse_paraphrases

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESET_QA_PATH = PROJECT_ROOT / "data" / "preset_qa.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "threshold_dataset.json"

# 提示词的三条要求都对应一个具体的失败模式,别删:
# 1.「信息完全一致,不要新增或删减条件」—— 否则模型可能生成
#    「张三的机票花了多少钱?」→「张三去年出差花了多少钱?」(丢了票种),
#    这种"正例"其实语义已变,会把阈值往低处拉;
# 2.「换动词/语序/称呼,不要照抄原句」—— 照抄原句的正例与 query 完全相同,
#    相似度恒为 1.0,对阈值搜索**没有信息量**(它永远最像,不提供区分度);
# 3.「每行一个,不要输出解释」—— 约束成可直接按行解析的格式,配合
#    evaluation/threshold.py::parse_paraphrases 的容错解析器使用。
#    注意提示词里说了"不要编号以外的任何内容",而解析器同时兼容
#    「1. xxx」「- xxx」「问法: xxx」等写法 —— 提示词给建议、解析器兜底,两道防线
PARAPHRASE_PROMPT = """下面是财务票据问答系统里的一条用户问题,请写出 {count} 个意思相同但说法不同的问法。

要求:
1. 保持询问的信息完全一致(同一个人、同一类票据、同一件事),不要新增或删减条件;
2. 用口语化的不同表达(换动词、换语序、换称呼),不要照抄原句;
3. 每行一个问法,不要编号以外的任何内容,不要输出解释。

原问题:{question}

同义问法:"""


def _client() -> OpenAI:
    # 显式超时：SDK 默认 600 秒 × 重试 3 次，上游挂住时标定会卡死
    return OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
    )


def generate_paraphrases(client: OpenAI, question: str, count: int = 5) -> list[str]:
    """让 LLM 生成同义问法,解析成列表(编号/项目符号都清掉)。

    ⚠ **不传 `enable_thinking`**:本项目默认 LLM 是思考模型(deepseek-flash),
    这里没显式关闭思考 ⇒ 模型的思考过程会占用输出,可能把同义问法挤掉或截断。
    本函数也**没有设 max_tokens**,所以被截断的风险取决于网关默认值。
    若发现"生成的正例偏少/为空",先查这里(见报告;短输出任务关思考是项目已确立的惯例)。

    `temperature=0.7` 与生成类任务一致(要多样性:5 条都一样就没意义了)。
    注意这与 llm_judge 的 `temperature=0`(要可复现)是相反取向,各有各的道理。

    解析交给 `parse_paraphrases`(纯函数、可离线测),本函数因此只剩"发请求"这一件事。
    """
    resp = client.chat.completions.create(
        model=settings.llm.model,
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(count=count, question=question)}],
        temperature=0.7,
    )
    return parse_paraphrases(resp.choices[0].message.content or "", limit=count)


def main() -> None:
    ap = argparse.ArgumentParser(description="为预设问答生成同义问法(阈值标定的正例)")
    # --limit 默认 20 是**成本控制**:每条问题一次 LLM 调用,65 条全跑要 65 次。
    # 标定只需要几十个 query 就足够找平台,不必全跑
    ap.add_argument("--limit", type=int, default=20, help="最多处理多少条预设问题(0 表示全部)")
    ap.add_argument("--count", type=int, default=5, help="每条问题生成几个同义问法")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    pairs = json.loads(PRESET_QA_PATH.read_text(encoding="utf-8"))
    # `if p.get("question")` 过滤掉缺问题的条目:否则下面会拿 None 去 format 提示词
    # (变成字面的 "None"),生成出一堆无意义的正例
    questions = [p["question"] for p in pairs if p.get("question")]
    if args.limit > 0:
        questions = questions[: args.limit]

    client = _client()
    dataset = []
    for index, question in enumerate(questions, 1):
        # 逐条调用并**立即打印进度**:本循环是纯外部 API 调用、最慢的一步,
        # 且被墙钟预算掐断时不会有 traceback(表现为进程悄悄退出)。
        # 逐条打印让"跑到哪了"能从输出里看出来
        paraphrases = generate_paraphrases(client, question, count=args.count)
        # 这里没做"某条生成失败就跳过"的容错:生成失败会拿到空列表,
        # 该 query 的 pos 就是 [] —— 它仍会进入数据集,但正例为 0 ⇒
        # 标定时该 query 的 tp 恒为 0、会拉低 recall。所以文件尾有一道总校验(见下)
        dataset.append({"query": question, "pos": paraphrases, "source": "preset_qa"})
        print(f"[{index}/{len(questions)}] {question} -> {len(paraphrases)} 条同义问法")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # indent=2 只为此文件可人工 review(它是标定的输入,出问题时第一眼看它)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    total_pos = sum(len(item["pos"]) for item in dataset)
    logger.info(f"[阈值标定] 写出 {len(dataset)} 条 query / {total_pos} 条正例 -> {out_path}")
    print(f"已写出 {len(dataset)} 条 query、{total_pos} 条正例 -> {out_path}")
    # **总校验**:一个正例都没生成出来时直接以非零码退出。
    # 这挡住的是最常见的静默故障 —— LLM 配置错/模型名不存在/提示词被网关拦,
    # 产出一份"有 query 但 pos 全空"的数据集,后续标定会得出毫无意义的平台
    # (全 0 的 F1 曲线也是"平台")。宁可这一步失败
    if not total_pos:
        raise SystemExit("没有生成任何正例,检查 LLM 配置或提示词")


if __name__ == "__main__":
    main()
