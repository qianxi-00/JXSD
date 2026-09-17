"""外部端点自检：换网关 / 换模型名 / 密钥过期时，别等主链路报错才发现。

课案有 `test_reranker_endpoint.py`（只探 reranker 一个端点），本脚本是它的等价物并扩面 ——
**四个外部 API 端点**各真探一次：

| 端点 | 探什么 | 为什么不能只看"HTTP 200" |
|---|---|---|
| Embedding | 向量维度是否等于配置（1024） | 换网关后维度变了，只在写库/检索时报"维度不匹配"，位置离根因很远 |
| Reranker | **语义是否成立**：同一句话的自相似分必须高于无关句 | 网关返回 200 但字段名变了（`relevance_score` 缺失）会让所有候选都变成 None 分 |
| 生成模型 | 一次极短问答是否真的出正文 | "端点通、模型名下架"常表现为 200 + 空正文（本项目真踩过：思考预算吃光正文为空） |
| 评判模型 | `score_answer()` 能否返回 1~5 分，并显示是否回退到生成模型 | 评判与生成分家这件事是配置出来的，光看能调通不知道用的是谁 |

Milvus / Redis / PostgreSQL / Neo4j 这四个**服务**不在这里探 —— 它们是 `GET /api/health`
的职责（那边每个探针有 3 秒上限、任一挂只标 degraded）。这里只管**出网的 API 端点**。

用法（在 Python_Base 目录下执行）：
    uv run python RAG/script/check_endpoints.py                 # 四个都探（4 次极小调用）
    uv run python RAG/script/check_endpoints.py --only embedding,rerank   # 只探免费的
    uv run python RAG/script/check_endpoints.py --json           # 机器可读
退出码：全部通过 = 0；有失败 = 1（可直接挂进 CI / 定时任务）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

# --- 路径引导：脚本直接跑时 sys.path[0] 是 RAG/script，需要手动挂上项目根与 RAG 根 ---
_BASE = Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
for _path in (str(_BASE), str(_BASE / "RAG")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import eval_llm_target, settings  # noqa: E402
from core.logger import logger  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - 老环境不支持就算了
    pass

ALL_CHECKS = ("embedding", "rerank", "llm", "judge")


def check_embedding() -> dict:
    """Embedding：真算一条向量，核对维度与配置一致。"""
    from retrieval.embedding import embed_query

    vector = embed_query("端点自检")
    expected = settings.embedding.embedding_size
    if len(vector) != expected:
        return {"ok": False, "detail": f"维度 {len(vector)} ≠ 配置 {expected}（换网关最常见的不一致）"}
    return {"ok": True, "detail": f"维度 {len(vector)}，模型 {settings.embedding.model}"}


def check_rerank() -> dict:
    """Reranker：不只看通不通，要看**语义是否成立**（相关的必须排在无关的前面）。"""
    from retrieval.rerank import rerank_scores

    documents = [{"semantic_text": "员工培训费用报销"}, {"semantic_text": "昨天下了一场大雨"}]
    rows = rerank_scores("员工培训费", documents)
    scores = [row.get("rerank_score") for row in rows]
    if any(score is None for score in scores):
        return {"ok": False, "detail": f"返回里缺 relevance_score（字段契约变了）：{scores}"}
    # 接口返回顺序不保证，所以按"相关句 > 无关句"来判，而不是按第一个元素
    relevant, irrelevant = scores[0], scores[1]
    if relevant <= irrelevant:
        return {"ok": False, "detail": f"语义异常：相关句 {relevant} 没有高于无关句 {irrelevant}"}
    return {
        "ok": True,
        "detail": f"相关 {relevant:.4f} > 无关 {irrelevant:.4f}（模型 {settings.rerank.model}）",
    }


def check_llm() -> dict:
    """生成模型：极短直答，必须返回**非空正文**（正文为空是"端点通但不可用"的典型）。"""
    from llm.chat import generate_direct_answer

    answer, reasoning = generate_direct_answer("只回复两个字：收到")
    answer = (answer or "").strip()
    if not answer:
        return {
            "ok": False,
            "detail": f"正文为空（reasoning {len(reasoning or '')} 字）—— 模型名/思考预算/路由需要查",
        }
    return {"ok": True, "detail": f"模型 {settings.llm.model}，正文 {len(answer)} 字：{answer[:20]}"}


def check_judge() -> dict:
    """评判模型：真打一次分，并如实报告有没有回退到生成模型。"""
    from evaluation.llm_judge import score_answer

    target = eval_llm_target()
    result = score_answer(
        question="张三的火车票多少钱？",
        answer="498.90 元",
        ground_truth="498.90 元",
    )
    who = target["model"] + ("（⚠ 回退到生成模型，未配 EVAL_LLM_MODEL）" if target["from_fallback"] else "")
    if result.get("score") is None:
        return {"ok": False, "detail": f"评分失败（模型 {who}）：{result.get('feedback', '')[:80]}"}
    return {"ok": True, "detail": f"模型 {who}，score={result['score']}"}


CHECKS = {
    "embedding": ("Embedding 端点", check_embedding),
    "rerank": ("Reranker 端点", check_rerank),
    "llm": ("生成模型端点", check_llm),
    "judge": ("评判模型端点", check_judge),
}


def run_checks(names) -> list[dict]:
    results = []
    for name in names:
        label, fn = CHECKS[name]
        started = time.perf_counter()
        try:
            outcome = fn()
        except Exception as exc:  # noqa: BLE001 - 自检脚本要如实记录任何异常，而不是自己崩掉
            outcome = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        results.append(
            {
                "name": name,
                "label": label,
                "ok": bool(outcome["ok"]),
                "detail": outcome["detail"],
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="外部端点自检（Embedding / Reranker / 生成 / 评判）")
    parser.add_argument("--only", default=",".join(ALL_CHECKS),
                        help=f"要探的端点，逗号分隔（可选 {', '.join(ALL_CHECKS)}；默认全都探）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    names = [item.strip() for item in args.only.split(",") if item.strip()]
    unknown = [item for item in names if item not in CHECKS]
    if unknown:
        parser.error(f"不认识的端点：{unknown}（可选 {', '.join(ALL_CHECKS)}）")

    results = run_checks(names)
    failed = [item["name"] for item in results if not item["ok"]]

    if args.json:
        # `--json` 时 stdout **只有 JSON**（人类可读结论走 stderr 日志、结论由退出码表达）：
        # 混着打的话调用方拿到的东西没法直接 json.loads，"机器可读"就名不副实。
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"[endpoints] 探 {len(results)} 个端点（每个一次极小调用）")
        for item in results:
            mark = "ok  " if item["ok"] else "FAIL"
            print(f"[endpoints]   [{mark}] {item['label']:16s} {item['seconds']:>6.3f}s  {item['detail']}")
        if failed:
            print(f"[endpoints] 失败：{', '.join(failed)} —— 主链路在这条路上会报错或静默降级")
        else:
            print("[endpoints] 全部通过")

    if failed:
        logger.error(f"[端点自检] 失败：{failed}")
        return 1
    logger.info(f"[端点自检] 全部通过：{[item['name'] for item in results]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
