# -*- coding: utf-8 -*-
"""数据复盘工作流 —— 抖音作品数据采集 + LLM 多维诊断

课案出处：自媒体课案 → 数据复盘 → ``workflows/review.py``

课案的四个节点（本项目**结构原样保留**，prompt 也照抄）
    ① fetch    从抖音个人主页采集真实作品数据
    ② funnel   播放/互动量分析 + 转化率 + 流量漏斗 + 发布时间（temperature 0.3）
    ③ content  内容方向 / 标题技巧 / 节奏建议 / 竞争分析（temperature 0.4）
    ④ suggest  立即执行 / 短期策略 / 长期规划 / A/B 测试方案（temperature 0.4）
    串行连成一条链，全部走 ``workflows.llm_call``（绝不抛异常的统一 LLM 入口）。

与课案的落地差异
    | 项 | 课案 | 本项目 |
    |---|---|---|
    | 采集节点 | ``DOUYIN_PROJECT`` 硬编码绝对路径 + ``sys.path.insert`` + ``from backend.lib.douyin import Douyin`` | 全部删掉，改调 ``tools.douyin_client.fetch_user_videos()``（HTTP，走自托管服务） |
    | Cookie | 读本地 douyin 项目的 ``config/settings.json`` | ``settings.media.douyin_cookie``，由 ``tools/douyin_client`` 内部解析 |
    | ``/user/self`` | 节点内用 ``niquests`` 抓 HTML 抠 sec_uid | 下移到 ``tools/douyin_client``，本文件不再碰网络细节 |
    | 采集失败 | 返回 error_msg，但只此一条路 | 保留 error_msg，**另加** ``run_review_from_json()`` 降级入口（跳过采集节点） |
    | 返回值 | ``review_graph.invoke(...)`` 原样返回（只含被写过的键） | ``run_review()`` 补齐 6 个字段，契约稳定 |

为什么要有降级入口
    自托管采集服务（Docker）不一定在跑，而 agent 也起不动 Docker。
    没有它，「四节点链路」在离线环境里就完全没法验证。
    ``run_review_from_json()`` 走的是同一套 funnel → content → suggest 节点，
    只是换了个入口把 ``video_data`` 灌进去。

踩过的坑
    · 直接 ``python workflows/review.py`` 时 ``sys.path[0]`` 是 ``workflows/`` 目录，
      ``from workflows import llm_call`` 会 ModuleNotFoundError ——
      所以**路径引导必须在文件顶部、所有项目内 import 之前**（放在 ``__main__`` 里没用，
      模块级 import 早跑完了）。
    · LangGraph 的 ``invoke`` 只返回状态里**被写过的键**；前端直接取
      ``result["suggestions"]`` 会 KeyError，所以对外函数统一补齐字段。
"""

import sys
from pathlib import Path

# ---- 路径引导：必须在 import 项目内模块（workflows/*、tools/*）之前执行 ----
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json  # noqa: E402
from typing import TypedDict  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402

from workflows import llm_call  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 没数据时各节点的占位文案（课案原文如此，前端据此判断"没跑出东西"）
NO_DATA = "暂无数据"


class ReviewState(TypedDict):
    """复盘工作流的状态（字段与课案一致）。"""

    profile_url: str        # 抖音主页 URL
    video_data: str         # JSON: 作品列表 + 统计数据
    funnel_diagnosis: str
    content_assessment: str
    suggestions: str
    error_msg: str


# ============================================================
# Node 1: 采集真实数据
# ============================================================

def node_fetch_data(state: ReviewState) -> dict:
    """从抖音个人主页采集作品数据（课案走本地爬虫，本项目走自托管 REST 服务）。

    输入: 抖音用户主页 URL（如 https://www.douyin.com/user/xxx）
    输出: JSON 格式的作品列表，含点赞/评论/分享/收藏（本人主页另含播放）
    """
    url = (state.get("profile_url") or "").strip()
    if not url:
        return {"error_msg": "请输入抖音用户主页链接", "video_data": "[]"}

    from tools.douyin_client import fetch_user_videos

    print(f"[复盘] 正在采集: {url}")
    result = fetch_user_videos(url, limit=20)

    if result.get("error"):
        print(f"[复盘] 采集失败: {result['error'].splitlines()[0]}")
        return {"error_msg": result["error"], "video_data": "[]"}

    items = result.get("items") or []
    if not items:
        return {
            "error_msg": "未获取到作品数据，请检查链接是否正确或 Cookie 是否有效",
            "video_data": "[]",
        }

    print(f"[复盘] 采集完成: {len(items)} 条作品")
    return {"video_data": json.dumps(items, ensure_ascii=False, indent=2), "error_msg": ""}


# ============================================================
# Node 2~4: LLM 诊断（prompt 与课案逐字一致）
# ============================================================

def _data_of(state: ReviewState) -> str:
    """取作品数据；没数据时返回空串，让各节点短路成「暂无数据」。"""
    data = (state.get("video_data") or "").strip()
    return "" if data in ("", "[]", "{}") else data


def node_funnel(state: ReviewState) -> dict:
    """Node 2: LLM 流量漏斗诊断。"""
    data = _data_of(state)
    if not data:
        print("[复盘] 无作品数据，跳过漏斗诊断")
        return {"funnel_diagnosis": NO_DATA}

    prompt = f"""你是短视频运营诊断专家。以下是抖音个人主页的真实数据：

{data}

请诊断：
1. **播放/互动量分析** — 哪些作品播放/点赞/评论/分享/收藏高？高表现作品的共同特征是什么？
  （注：如果播放量为0或不含播放字段，说明是非本人主页，只基于互动数据做分析即可）
2. **转化分析** — 计算点赞→评论转化率、点赞→收藏转化率（如有播放量，加上播放→点赞转化率），找出各指标最优/最差作品
3. **流量漏斗** — 分析从曝光到互动的转化情况
4. **发布时间** — 哪个时间段发布效果最好？
每个结论用数据支撑。
"""
    print("[复盘] 漏斗诊断中 ...")
    return {"funnel_diagnosis": llm_call(prompt, temperature=0.3)}


def node_content(state: ReviewState) -> dict:
    """Node 3: LLM 内容质量评估。"""
    data = _data_of(state)
    if not data:
        print("[复盘] 无作品数据，跳过内容评估")
        return {"content_assessment": NO_DATA}

    prompt = f"""基于这些抖音作品真实数据，评估内容质量：

{data}

从以下维度评估：
1. **内容方向** — 哪些主题/选题表现最好？有没有明显的内容偏好？
2. **标题技巧** — 高播放作品的标题有什么共同特征？
3. **节奏建议** — 根据播放/互动数据，建议最佳视频时长
4. **竞争分析** — 对比同赛道平均水平，这组数据的优劣势在哪里
"""
    print("[复盘] 内容评估中 ...")
    return {"content_assessment": llm_call(prompt, temperature=0.4)}


def node_suggest(state: ReviewState) -> dict:
    """Node 4: LLM 生成可执行优化方案（吃 Node 2/3 的结论）。"""
    data = _data_of(state)
    if not data:
        print("[复盘] 无作品数据，跳过优化策略")
        return {"suggestions": NO_DATA}

    funnel = state.get("funnel_diagnosis", "")
    content = state.get("content_assessment", "")

    prompt = f"""基于抖音账号真实数据分析，生成可执行优化方案：

真实数据:
{data}

诊断结果:
{funnel}

内容评估:
{content}

请给出：
1. **立即执行**（3条 — 下一条视频就能用的改进）
2. **短期策略**（3条 — 1周内迭代方向）
3. **长期规划**（3条 — 1个月优化路径）
4. **A/B 测试方案** — 设计一个对比实验：改什么、怎么测、看什么指标
"""
    print("[复盘] 生成优化策略中 ...")
    return {"suggestions": llm_call(prompt, temperature=0.4)}


# ============================================================
# 图：四节点串行（与课案的节点名、边完全一致）
# ============================================================

builder = StateGraph(ReviewState)
builder.add_node("fetch", node_fetch_data)
builder.add_node("funnel", node_funnel)
builder.add_node("content", node_content)
builder.add_node("suggest", node_suggest)

builder.add_edge(START, "fetch")
builder.add_edge("fetch", "funnel")
builder.add_edge("funnel", "content")
builder.add_edge("content", "suggest")
builder.add_edge("suggest", END)

review_graph = builder.compile()

# 降级入口用的图：与上面同源，只是少了采集节点（video_data 由调用方灌进来）
_json_builder = StateGraph(ReviewState)
_json_builder.add_node("funnel", node_funnel)
_json_builder.add_node("content", node_content)
_json_builder.add_node("suggest", node_suggest)
_json_builder.add_edge(START, "funnel")
_json_builder.add_edge("funnel", "content")
_json_builder.add_edge("content", "suggest")
_json_builder.add_edge("suggest", END)

json_review_graph = _json_builder.compile()


# ============================================================
# 对外入口
# ============================================================

def _filled(state: dict, profile_url: str = "", error_msg: str = "") -> dict:
    """补齐 6 个字段 —— LangGraph 只返回被写过的键，前端不该为此兜底。"""
    return {
        "profile_url": state.get("profile_url") or profile_url,
        "video_data": state.get("video_data") or "[]",
        "funnel_diagnosis": state.get("funnel_diagnosis") or "",
        "content_assessment": state.get("content_assessment") or "",
        "suggestions": state.get("suggestions") or "",
        "error_msg": state.get("error_msg") or error_msg,
    }


def run_review(profile_url: str) -> dict:
    """运行数据复盘工作流（采集 → 漏斗诊断 → 内容评估 → 优化策略）。

    Args:
        profile_url: 抖音用户主页链接。

    Returns:
        ``{"profile_url", "video_data", "funnel_diagnosis",
           "content_assessment", "suggestions", "error_msg"}``
        **失败也只写 error_msg，不抛异常。**
    """
    profile_url = (profile_url or "").strip()
    if not profile_url:
        return _filled({}, profile_url, "请输入抖音用户主页链接")

    try:
        result = review_graph.invoke({"profile_url": profile_url})
    except Exception as exc:  # noqa: BLE001 —— 工作流出错不该把 Streamlit 页面带崩
        print(f"[复盘] 工作流异常: {exc}")
        return _filled({}, profile_url, f"复盘工作流异常: {exc}")
    return _filled(result, profile_url)


def run_review_from_json(raw_json: str) -> dict:
    """降级入口：跳过采集节点，用粘贴的作品数据跑 漏斗诊断 → 内容评估 → 优化策略。

    Args:
        raw_json: 用户粘贴的 JSON 数组 / 抖音原始响应 / CSV / TSV 文本。

    Returns:
        与 ``run_review()`` 同构的 dict；解析失败时只写 error_msg。
    """
    from tools.douyin_client import parse_manual_json

    parsed = parse_manual_json(raw_json)
    if parsed.get("error"):
        return _filled({}, "（手动粘贴数据）", parsed["error"])

    items = parsed.get("items") or []
    if not items:
        return _filled({}, "（手动粘贴数据）", "粘贴的数据里没有作品记录")

    data_json = json.dumps(items, ensure_ascii=False, indent=2)
    print(f"[复盘] 手动数据 {len(items)} 条作品，跳过采集节点直接诊断")
    try:
        result = json_review_graph.invoke(
            {"profile_url": "（手动粘贴数据）", "video_data": data_json}
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[复盘] 工作流异常: {exc}")
        return _filled({}, "（手动粘贴数据）", f"复盘工作流异常: {exc}")
    return _filled(result, "（手动粘贴数据）")


# ============================================================
# 离线自检（服务没起来也必须全绿；加 --llm 才真调模型）
# ============================================================

if __name__ == "__main__":
    print("=== 数据复盘工作流自检 ===")

    # 1) 图结构：节点与边必须与课案一致（这是「四节点串行」的硬约束）
    graph = review_graph.get_graph()
    nodes = set(graph.nodes)
    assert {"fetch", "funnel", "content", "suggest"} <= nodes, nodes
    edges = {(e.source, e.target) for e in graph.edges}
    expected = {
        ("__start__", "fetch"),
        ("fetch", "funnel"),
        ("funnel", "content"),
        ("content", "suggest"),
        ("suggest", "__end__"),
    }
    assert expected <= edges, f"缺少边: {expected - edges}\n实际: {sorted(edges)}"
    print(f"  review_graph 结构        OK  {len(nodes)} 节点 / {len(edges)} 条边，串行顺序正确")

    # 2) 降级图：只有三个诊断节点，没有 fetch
    json_nodes = set(json_review_graph.get_graph().nodes)
    assert {"funnel", "content", "suggest"} <= json_nodes, json_nodes
    assert "fetch" not in json_nodes, "降级入口不该有采集节点"
    print("  json_review_graph 结构   OK  无 fetch，仅三个诊断节点")

    # 3) 空输入 / 无效输入：返回结构化错误，不抛异常
    for bad in ("", "   ", "不是 JSON", "[]", "{}"):
        result = run_review_from_json(bad)
        assert set(result) == {
            "profile_url", "video_data", "funnel_diagnosis",
            "content_assessment", "suggestions", "error_msg",
        }, result.keys()
        assert result["error_msg"], f"{bad!r} 应该给出 error_msg"
        assert result["suggestions"] == "", result["suggestions"]
        assert result["video_data"] == "[]"
    print("  run_review_from_json 失败路径 OK  6 字段齐全 + error_msg 非空")

    result = run_review("")
    assert result["error_msg"] and result["video_data"] == "[]"
    print("  run_review 空链接         OK")

    # 4) 真实 LLM 链路（默认跳过，保证离线全绿）
    if "--llm" in sys.argv:
        from tools.douyin_client import SAMPLE_MANUAL_JSON

        print("  --llm：真调模型跑完整链路（漏斗诊断 → 内容评估 → 优化策略）...")
        result = run_review_from_json(SAMPLE_MANUAL_JSON)
        assert not result["error_msg"], result["error_msg"]
        assert result["funnel_diagnosis"] and result["content_assessment"]
        assert result["suggestions"]
        print(f"    漏斗诊断: {result['funnel_diagnosis'][:120]}...")
        print(f"    优化策略: {result['suggestions'][:120]}...")
        print("  LLM 链路                  OK")
    else:
        print("  LLM 链路                  跳过（加 --llm 才跑，会真实消耗额度）")

    print("\n自检完成")
