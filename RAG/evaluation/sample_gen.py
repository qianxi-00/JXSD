"""评估样本生成的质量审核:对照基础篇课案「系统评估 · 样本生成」。

课案的样本生成流程:
    票据上下文 -> LLM 生成 {"事实型问题","答案"} -> 两道质检
    1) question_groundedness_critique_prompt:问题能否仅凭上下文无歧义回答;
    2) question_standalone_critique_prompt:问题是否独立、完整、指代清晰;
    两项都 >= 5 分才收进评估集,否则最多重试 3 次。

课案代码用 `json.loads(评价结果)` 再取 `["总评分"]`,但提示词要求的是
"回答::: 评价:... 总评分:..." 这种文本格式 —— 两者对不上,直接照抄会解析失败。
这里实现一个兼容两种输出的解析器(先按 JSON 解,失败再按正则提取),并在测试里锁住行为。

为什么质检这一步不能省(而是"两道题都过才收"):
LLM 生成的评估样本会带两类系统性缺陷 ——
- **问题无法从上下文无歧义回答**(groundedness):上下文里没有足够信息,模型却编出一个
  看似合理的问题 ⇒ 后期评估时这条样本永远答不对,扣的是**链路**的分,实际是样本的锅;
- **问题不独立/指代不清**(standalone):生成出「它的金额是多少?」这种带隐式指代的问题
  ⇒ 脱离生成时的上下文根本无法理解,作为评估集里的独立一条毫无意义。
两类缺陷都会**静默污染指标**(分母变大、分数变低,且看不出原因),所以宁可少收样本,
也要在入口把不合格的挡掉。这也是课案要求「最多重试 3 次」的原因 ——
失败多半是这次采样的运气,重试比直接放弃划算。

本模块只做**解析**(纯函数、无 API 调用),调用与重试逻辑在
`script/generate_eval_samples.py`。这样解析器的边界行为(裸文本 / JSON / 带 ``` 围栏 /
分数写成「5分」/ 解析不出来)都能在离线单测里钉住,不用花钱调 LLM 才能测。
"""

from __future__ import annotations

import json
import re
from typing import Any

MIN_SCORE = 5

# 总分数的提取:允许「总评分」后有全角/半角冒号、markdown 加粗星号(**5**)、任意空白。
# `(\d)` 只取**一位**数字 —— 本场景分数域是 1~5,写成 `(\d+)` 反而会把
# 「总评分:2025」这种噪声整段吃进来(见报告中对多位数的边界说明)
_SCORE_RE = re.compile(r"总评分\s*[:：]\s*\**\s*(\d)")
# 评价正文:抓到行尾即止(`(?:\n|$)`),`re.S` 让 `.` 能跨行但不改变上面的终止条件 ——
# 因为终止条件在 `.` 之外,所以多行输出只会取第一行,不会把后面的总评分也吞进来
_EVALUATION_RE = re.compile(r"评价\s*[:：]\s*(.+?)(?:\n|$)", re.S)


def parse_critique(text: str) -> dict[str, Any] | None:
    """解析质检输出,返回 {"总评分": int, "评价": str} 或 None(无法解析)。

    **双层解析**是刻意的容错设计(不是"没想清楚该用哪种"):
    同一份提示词下发,模型可能返回纯 JSON、也可能返回「回答::: 评价:... 总评分:...」
    的文本模板,还可能把 JSON 包在 ```json 围栏里。解析器先试 JSON,失败再走正则回退 ——
    任何一层走通都算成功,**两层都失败才返回 None**。
    返回 None 而不是默认分数:让调用方(质检流程)能区分「模型明确给了低分」与
    「模型输出看不懂」—— 前者可以重试生成,后者要重试的是**提示词/模型**,
    处理方式不同。上游 `passes_quality` 把 None 一律当不通过。

    几个细节:
    - 剥围栏用 `strip("`")` **不是**剥 ```json:它把首尾所有反引号都去掉,
      所以 ` ```json {..} ``` ` 变成 `json {..}`,再由 `removeprefix("json")` 去掉语言标记。
      这个写法对「围栏里的内容本身带反引号」不鲁棒,但对模型的实际输出够用;
    - 分数兼容 `{"score": 5}` 这种英文键(`data.get("总评分") or data.get("score")`),
      注意这里用 `or`:所以**中文键的 0 分会被英文键顶替**(0 是假值),
      对 1~5 的分数域无影响,但这是个值得知道的行为;
    - `str(score).strip().strip("分")` 兼容 `{"总评分": "5分"}` 这种把单位写进值里的输出。
      注意 `strip("分")` 会剥掉首尾**所有**的「分」字符,`"分5分"` 会变成 `"5"`;
    - `int(...)` 在 JSON 分支没有 try 包住:所以 `{"总评分": "满分"}` 会抛 ValueError
      而不是返回 None(见报告)。实际影响有限 —— 调用方本就在重试循环里。
    """
    if not text or not text.strip():
        return None
    raw = text.strip()

    # 1) 优先按 JSON 解析(课案的写法)
    candidate = raw
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.removeprefix("json").strip()
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            score = data.get("总评分") or data.get("score")
            if score is not None:
                return {"总评分": int(str(score).strip().strip("分")), "评价": str(data.get("评价", ""))}
    except (ValueError, TypeError):
        # JSON 解析失败(或 int() 转换失败)不报错,继续走下面的文本格式回退。
        # 这里只吞 ValueError/TypeError,别扩大成 except Exception ——
        # 那会把代码 bug 也一起吞掉,变成"永远解析不出来"的静默故障
        pass

    # 2) 回退:按提示词要求的文本格式提取
    score_match = _SCORE_RE.search(raw)
    if not score_match:
        return None
    evaluation_match = _EVALUATION_RE.search(raw)
    return {
        "总评分": int(score_match.group(1)),
        # 正则没匹配到「评价:」时不返回 None,而是给空字符串:
        # 有分数就算解析成功(分数才是判定依据),评价只是给人看的说明文本
        "评价": evaluation_match.group(1).strip() if evaluation_match else "",
    }


def passes_quality(*critiques: dict[str, Any] | None, min_score: int = MIN_SCORE) -> bool:
    """两道质检都达到 min_score 才算通过;任一解析失败视为不通过。

    用**可变参数**接收任意道检验(课案是两道:groundedness + standalone),
    这样将来加第三道质检不用改函数签名,只要多传一个实参。

    三道防线都要看得见:
    - `if not critiques`(一个都没传)也返回 False —— 否则 `all([])` 是 True,
      「忘了传质检」会被当成「质检通过」,静默放行所有样本;
    - `any(c is None ...)` 解析失败即拒:无法确认质量就不收,与
      sample_gen 的解析器约定一致;
    - `int(c.get("总评分", 0))` 缺失分数记 0(必然低于 min_score=5)⇒ 默认拒绝。
      注意是 `c.get(...)` 而不是 `c["总评分"]`:解析器保证有这个键,
      但直接传手写 dict 时用 get 更稳。
    """
    if not critiques or any(c is None for c in critiques):
        return False
    return all(int(c.get("总评分", 0)) >= min_score for c in critiques)


# 自检:同一份内容的**两种输出形态**(纯文本模板 / 纯 JSON)必须解析出同一个结果 ——
# 这是本模块存在的理由,也是唯一真正要钉的契约;另外钉住「解析不出 → None」
# 和「一道不合格 → 不通过」。
if __name__ == "__main__":
    # 文本形态带 markdown 加粗与全角冒号(真实模型输出常有的样子)
    text_form = "回答:::\n评价：资料齐全\n总评分：5"
    json_form = '{"评价": "资料齐全", "总评分": 5}'
    assert parse_critique(text_form) == {"总评分": 5, "评价": "资料齐全"}
    assert parse_critique(json_form) == {"总评分": 5, "评价": "资料齐全"}
    assert parse_critique("没有评分") is None
    assert passes_quality(parse_critique(text_form), parse_critique(json_form))
    assert not passes_quality(parse_critique("总评分：4"), parse_critique(text_form))
    print("sample_gen.py 自检通过")
