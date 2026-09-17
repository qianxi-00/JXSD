"""大模型评分:对照基础篇课案「系统评估 · 第四步:使用大模型评估最终回答」。

课案明确:"EVALUATION_PROMPT 只用于评估最终回答质量,不能替代召回和重排指标"。
因此这里的分数是**补充信号**,分阶段指标仍以 evaluation/stage_metrics.py 为准。

比分阶段指标多一层解析:模型输出必须是
    Feedback: ... [RESULT] N
解析失败或分数越界时返回 None,由调用方决定是记 NaN 还是跳过。

与分阶段指标的关系(别把两者混在一起看):
- **分阶段指标是硬指标**:路由/筛选/召回/重排/关键事实覆盖,全部可复算、无模型参与;
- **本模块是软信号**:用 LLM 给最终回答打 1~5 分,能捕捉分阶段指标测不到的东西
  (语气是否得体、是否答非所问、是否啰嗦),但**会受评分模型自身波动影响**。
所以课案的原话是「不能替代召回和重排指标」—— 报告里两者并列展示,冲突时以硬指标为准。

为什么分数解析要单独一个函数(`parse_result_score`)而不是内联:
LLM 的评分输出是最不稳定的一环(可能带 []、【】、markdown 加粗、中文括号),
把它抽成纯函数后可以用一串字符串在离线单测里钉住所有边界,
不必花钱调 API 才能验证解析逻辑。`score_answer`(需要真实 API)则只在
`script/langfuse_evaluation.py` 的 RAGAS 流程里被间接使用。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# --- 路径引导：直接运行本文件时也能导入 RAG 内部包(与 data_process/app 下的约定一致) ---
# 本文件要 import config / core.* 这些仓库根下的包,直接跑自检时必须先补 sys.path
_BASE = Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
for _p in (str(_BASE), str(_BASE / "RAG")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import settings  # noqa: E402
from core.logger import logger  # noqa: E402
from core.prompts import EVALUATION_PROMPT  # noqa: E402

# 匹配「[RESULT] 5」/「【RESULT】5」/「[ RESULT ] **5**」这些写法:
# 方括号开闭各允许 [] 与 【】 混用(RESULT 本身用 re.I 兼容大小写),
# 后面允许 markdown 加粗星号与空白。`(\d+)` 这里是多位,越界由下面的范围检查兜
_RESULT_RE = re.compile(r"[\[【]\s*RESULT\s*[\]】]\s*\**\s*(\d+)", re.I)
# 评分域 1~5,与课案 EVALUATION_PROMPT 里的评分细则一一对应
MIN_SCORE = 1
MAX_SCORE = 5


def parse_result_score(text: str) -> int | None:
    """从评分输出里取出 [RESULT] 后的 1~5 整数,取不到或越界返回 None。

    为什么越界要返回 None 而不是"夹到边界"(如 9 → 5):
    越界说明模型**没按评分细则作答**(可能理解错了任务、或把别的数字当成了分数),
    这种输出不可信,记 5 分会把噪声伪装成满分。返回 None 让上游能把它记为
    「本次评分无效」,与「模型真给了 5 分」区分开 —— 这是整条评估链里唯一能发现
    "评分模型跑偏"的信号。

    正则用 `search` 而非 `match`/`fullmatch`:模型输出通常是
    「Feedback: 回答与标准答案一致 [RESULT] 5」这种自然语言 + 标记的混合,
    标记不在行首。所以只会取**第一个**出现的 RESULT 标记,后面的忽略。
    """
    if not text:
        return None
    match = _RESULT_RE.search(text)
    if not match:
        return None
    score = int(match.group(1))
    return score if MIN_SCORE <= score <= MAX_SCORE else None


def score_answer(question: str, answer: str, ground_truth: str, context: str = "") -> dict:
    """调用 LLM 按课案评分细则给最终回答打 1~5 分(需要真实 API)。

    返回值固定是 `{"score": int|None, "feedback": str}` 形状,即**评分失败不抛异常**,
    而是 score=None + feedback 里留着模型的原始输出。选择"失败也返回结构"的理由:
    评分是评估的**补充环节**,不该因为一条样本评不出分就让整轮评估崩掉;
    调用方按 score=None 决定是记 NaN 还是跳过这条。

    几个刻意的写法:
    - `from openai import OpenAI` 写在函数**内部**:本模块被 import 时(比如跑
      其它离线单测)不需要 openai 已安装、也不会触发任何网络相关的初始化;
    - 客户端**显式带 timeout**:OpenAI SDK 默认 600 秒 × 重试 3 次 ≈ 半小时,
      上游网关抖动一次就把整轮评估挂死(项目上真实踩过这类假死);
    - `temperature=0`:评分任务要的是**可复现**,同一份回答两次评分不该给出不同分数。
      注意:即使 temperature=0,LLM 评分仍可能有波动(与 batch/后端实现有关),
      所以本模块的分数只能当趋势看,不能当精确量;
    - `context or "(未提供检索上下文)"`:留给"没有检索上下文也能评"的调用方式,
      用占位串而不是空串,让模型知道这是**没给上下文**而不是**上下文是空的**;
    - 模型用 `settings.llm.model`(与生成用的是同一个模型),这与课案
      「评估模型与生成模型分开配置」有出入(见报告)。
    """
    from openai import OpenAI

    # 显式超时：SDK 默认 600 秒 × 重试 3 次，上游挂住时整轮评估会跟着卡死
    client = OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
    )
    prompt = EVALUATION_PROMPT.format(
        context=context or "(未提供检索上下文)",
        question=question,
        answer=answer,
        ground_truth=ground_truth,
    )
    resp = client.chat.completions.create(
        model=settings.llm.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content or ""
    score = parse_result_score(text)
    if score is None:
        # 解析失败必须留痕:否则"评分一直是 None"与"模型真的没给分"在报告里长得一样。
        # 只截 200 字符,避免把整段思考内容灌进日志
        logger.warning(f"[评估] 无法从模型输出解析评分: {text[:200]}")
    return {"score": score, "feedback": text.strip()}


# 自检只覆盖解析器的三条边界(命中 / 无标记 / 越界),**不调用真实 API** ——
# 自检必须能在没有 .env、没有网关的机器上跑通
if __name__ == "__main__":
    assert parse_result_score("Feedback: 一致 [RESULT] 5") == 5
    assert parse_result_score("没有标记") is None
    assert parse_result_score("[RESULT] 9") is None
    print("llm_judge.py 自检通过")
