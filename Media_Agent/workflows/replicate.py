# -*- coding: utf-8 -*-
"""内容复刻工作流 —— 提取文案 → 拆解爆款 → 仿写 → 生成标题

课案出处：《3.自媒体Agent》→ 内容复刻 → workflows/replicate.py

本节要讲什么
    1. **带工具的 LangGraph 节点**。前两个模块的节点只调 LLM，
       这里的 ``node_extract`` 是「真去下载文件、真去跑语音识别」的 I/O 节点：
       链路长、外部依赖多、失败点是常态，因此它是本项目的「容错样板」。
    2. **一次丢链接、三步吃产出**：extract 写 ``original_text``，
       analyze 读它写 ``viral_analysis``，rewrite 读前两者写 ``rewritten``，
       titles 读 ``rewritten``。每个节点只碰自己的一前一后，串起来就是完整链路。
    3. **链路可以在中途体面地停下**。课案没判空：下载失败 → 文案是空串 →
       照样把空串喂给三个 LLM，模型会一本正经地分析不存在的文案（幻觉）。
       本实现加了短路：拿不到文案就写一句中文提示，后面三个节点逐级跳过，
       **一次 LLM 都不调**，前端能直接看出「链路断在第 1 步」。
    4. 视频链路之外还有**文章兜底**：课案的流程图写的是「下载视频/抓取文章」，
       但代码只实现了视频那一半。抖音/B站下载失败时改用 ``fetch_article()`` 抓正文，
       知乎/公众号这类图文链接也能走完整条链路。

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 提取节点 | `download_video` → `extract_audio_text`，不判空 | 先 `extract_url` 洗一遍链接，视频链路失败改走 `fetch_article` | 课案流程图本来就写了「下载视频/抓取文章」；下载失败时纯视频链路必然得到空串 |
    | 空文案 | 不判空，直接喂 LLM ×3 | 短路 + 中文提示，逐级跳过 | 否则模型会为「空文案」编出一篇爆款拆解，前端看起来像成功了 |
    | 节点容错 | 无 try | 四个节点全部 try/except | 与其余模块统一：失败写进 state 让链路走完 |
    | 温度 | 0.6 / 0.8 / 0.8 | 同左 | 拆解要稳、仿写要活，课案这个取值是对的 |
    | prompt | 见课案 | **逐字保留**（仅在前面加了短路判断） | 拆解维度是这个模块的核心资产 |
    | 自检 | 无 | 末尾 `__main__` 离线自检（打桩下载/ASR/抓取/LLM） | 下载真视频不可控，必须能在离线把 4 种分支都验一遍 |

踩过的坑
    · 本机没装 ``yt-dlp`` 或网络不通时，``download_video`` 返回的是**空字符串**而不是异常
      （工具层刻意不抛），所以节点里必须判 ``if not video_path``，
      否则会拿着 ``""`` 去调 ``extract_audio_text``。
    · ``extract_url()`` 要洗抖音/小红书的分享口令
      （``"7.65 复制打开抖音…https://v.douyin.com/xxx/…"`` 这种混合文本），
      课案把它讲在正文里、但代码里没用；不洗的话 yt-dlp 拿到整段文本会直接失败。
    · 直接 `python workflows/replicate.py` 时 `sys.path[0]` 是 `workflows/`，
      需要顶部那段 path 引导才能 `from workflows import llm_call`。
    · **断言 message 里不要放活引用**：`assert calls["x"] == [...], calls["x"]`
      在配了 `finally: 清空` 的自检里会骗人 —— 异常对象持有的是同一个 list，
      `finally` 先把它清空，traceback 打印出来就是 `AssertionError: []`，
      完全看不出真正调了什么。本文件第一版就这么踩了一次（真因是上一个用例的
      调用没清干净）。现在的做法：每个用例开头 `clear_calls()`，
      `finally` 只负责还原被替换的函数、不动调用记录。

运行方式
    离线自检（不联网、不下载视频）::

        Set-Location F:\\ProGram\\Python_Base\\Media_Agent
        & ..\\.venv\\Scripts\\python.exe workflows\\replicate.py

    真实联网跑一次（会下载视频 + 调 ASR + 调 LLM）::

        & ..\\.venv\\Scripts\\python.exe workflows\\replicate.py --live "https://www.bilibili.com/video/BVxxxx"
"""

import sys
from pathlib import Path
from typing import TypedDict

# Windows 控制台默认 GBK，本模块会打印中文日志
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 直接以脚本方式运行时，sys.path[0] 是 workflows/ 目录，项目根不在里面。
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph.graph import END, START, StateGraph  # noqa: E402

from tools.media_tools import (  # noqa: E402
    download_video,
    extract_audio_text,
    extract_url,
    fetch_article,
)
from workflows import llm_call  # noqa: E402

# 拆解要稳（判断题），仿写/起标题要活（创作题）
_TEMP_ANALYZE = 0.6
_TEMP_CREATIVE = 0.8

# 链路中断的标记：统一前缀，方便下游判断「这一步没有可分析的东西」
_SKIP_MARK = "⚠️"


def _is_usable(text: str) -> bool:
    """判断上游产出能不能继续往下用（空串 / 中断标记都不算）。"""
    value = (text or "").strip()
    return bool(value) and not value.startswith(_SKIP_MARK)


class ReplicateState(TypedDict):
    """内容复刻工作流的共享状态（契约已冻结，字段名不要改）。"""

    source_url: str      # 输入：视频/文章链接（允许带分享口令的混合文本）
    original_text: str   # 节点①产出：提取到的原文案
    viral_analysis: str  # 节点②产出：爆款要素拆解
    rewritten: str       # 节点③产出：仿写文案
    titles: str          # 节点④产出：5 个标题


# ==========================================================================
# 节点
# ==========================================================================
def node_extract(state: ReplicateState) -> dict:
    """节点①：链接 → 文案（视频下载+ASR 为主，文章抓取兜底）。

    课案这里只有 ``download_video`` + ``extract_audio_text`` 两行。
    本实现补了三件事：洗链接、判空、文章兜底 —— 都是「跑真实链接」才会遇到的问题。
    """
    try:
        raw = state.get("source_url", "") or ""
        # 抖音/小红书的分享口令是混合文本，先抠出纯 URL
        url = extract_url(raw)
        print(f"[内容复刻] ① 提取文案: {url[:100]}")

        text = ""
        # ---- 主链路：视频下载 → 语音识别 ----
        video_path = download_video(url)
        if video_path:
            text = extract_audio_text(video_path)
        else:
            print("[内容复刻] 视频下载失败（链接不是视频 / 未装 yt-dlp / 被反爬）")

        # ---- 兜底：当文章抓（图文链接，或视频链路没拿到文案）----
        if not text:
            print("[内容复刻] 改走文章抓取兜底...")
            text = fetch_article(url)

        if not text:
            print("[内容复刻] 视频与文章两条路都没拿到文案，链路将在此停下")
            return {"original_text": (
                f"{_SKIP_MARK} 未能获取到文案：视频下载/语音识别、文章抓取都失败了。\n"
                "可能原因：链接需要登录、被平台反爬、本机未安装 yt-dlp，"
                "或百炼语音识别未配置。"
            )}

        print(f"[内容复刻] ① 拿到文案 {len(text)} 字")
        return {"original_text": text}
    except Exception as exc:  # noqa: BLE001 —— 节点绝不抛
        print(f"[内容复刻] ① 提取文案失败: {exc}")
        return {"original_text": f"{_SKIP_MARK} 提取文案时出错: {exc}"}


def node_analyze(state: ReplicateState) -> dict:
    """节点②：拆解爆款要素（钩子/结构/情绪/金句/互动引导）。"""
    try:
        original_text = state.get("original_text", "") or ""
        if not _is_usable(original_text):
            print("[内容复刻] 没有可用文案，跳过爆款拆解")
            return {"viral_analysis": f"{_SKIP_MARK} 上游没拿到文案，无法拆解爆款结构。"}

        prompt = f"""拆解以下爆款文案的结构：

{original_text}

从以下维度分析：
1. **钩子类型**（前3句用了什么技巧：悬念/反差/痛点/利益）
2. **结构模板**（可复用的文案框架）
3. **情绪节奏**（每段的情绪高低点）
4. **金句亮点**（最容易引发共鸣的句子）
5. **互动引导**（点赞/评论/关注的引导方式）
"""
        print("[内容复刻] ② 拆解爆款要素中...")
        return {"viral_analysis": llm_call(prompt, temperature=_TEMP_ANALYZE)}
    except Exception as exc:  # noqa: BLE001
        print(f"[内容复刻] ② 爆款拆解失败: {exc}")
        return {"viral_analysis": f"{_SKIP_MARK} 爆款拆解出错: {exc}"}


def node_rewrite(state: ReplicateState) -> dict:
    """节点③：按同一结构模板仿写新文案。"""
    try:
        viral_analysis = state.get("viral_analysis", "") or ""
        original_text = state.get("original_text", "") or ""
        if not _is_usable(viral_analysis):
            print("[内容复刻] 没有爆款拆解结果，跳过仿写")
            return {"rewritten": f"{_SKIP_MARK} 缺少爆款拆解结果，无法仿写。"}

        prompt = f"""根据以下爆款分析，仿写一篇新文案：

【爆款要素分析】
{viral_analysis}

【参考原文】
{original_text}

要求：
- 保持相同结构模板和节奏
- 换成新主题（同类不同题）
- 保留钩子、转折、金句、互动引导
- 字数控制在原文的80%-120%
"""
        print("[内容复刻] ③ 仿写新文案中...")
        return {"rewritten": llm_call(prompt, temperature=_TEMP_CREATIVE)}
    except Exception as exc:  # noqa: BLE001
        print(f"[内容复刻] ③ 仿写失败: {exc}")
        return {"rewritten": f"{_SKIP_MARK} 仿写出错: {exc}"}


def node_titles(state: ReplicateState) -> dict:
    """节点④：为仿写文案生成 5 种类型的爆款标题。"""
    try:
        rewritten = state.get("rewritten", "") or ""
        if not _is_usable(rewritten):
            print("[内容复刻] 没有仿写结果，跳过标题生成")
            return {"titles": f"{_SKIP_MARK} 缺少仿写文案，无法生成标题。"}

        prompt = f"""为以下文案生成5个爆款标题：

{rewritten}

5种类型各一个：
1. 数字型（"3个方法..."）
2. 疑问型（"为什么..."）
3. 痛点型（"别再..."）
4. 悬念型（"原来..."）
5. 命令型（"一定要..."）
每个标题不超过25字。
"""
        print("[内容复刻] ④ 生成标题中...")
        return {"titles": llm_call(prompt, temperature=_TEMP_CREATIVE)}
    except Exception as exc:  # noqa: BLE001
        print(f"[内容复刻] ④ 标题生成失败: {exc}")
        return {"titles": f"{_SKIP_MARK} 标题生成出错: {exc}"}


# ==========================================================================
# 构建图：START → extract → analyze → rewrite → titles → END
# ==========================================================================
builder = StateGraph(ReplicateState)
builder.add_node("extract", node_extract)
builder.add_node("analyze", node_analyze)
builder.add_node("rewrite", node_rewrite)
builder.add_node("titles", node_titles)

builder.add_edge(START, "extract")
builder.add_edge("extract", "analyze")
builder.add_edge("analyze", "rewrite")
builder.add_edge("rewrite", "titles")
builder.add_edge("titles", END)

replicate_graph = builder.compile()


# ==========================================================================
# 对外接口
# ==========================================================================
def run_replicate(url: str) -> dict:
    """运行内容复刻工作流。

    Args:
        url: 视频链接或文章链接（允许是带分享口令的混合文本）。

    Returns:
        ``{"source_url","original_text","viral_analysis","rewritten","titles"}``。
        链路中断时对应字段以 ``⚠️`` 开头并说明卡在哪一步，不会抛异常。
    """
    try:
        result = replicate_graph.invoke({"source_url": url or ""})
    except Exception as exc:  # noqa: BLE001 —— 图本身出错也不往上抛
        print(f"[内容复刻] 工作流执行失败: {exc}")
        result = {"original_text": f"{_SKIP_MARK} 工作流执行失败: {exc}"}

    return {
        "source_url": result.get("source_url", url or ""),
        "original_text": result.get("original_text", ""),
        "viral_analysis": result.get("viral_analysis", ""),
        "rewritten": result.get("rewritten", ""),
        "titles": result.get("titles", ""),
    }


# ==========================================================================
# 离线自检
# ==========================================================================
def _live_check(url: str = "") -> None:
    """真实联网跑一次（下载视频 + ASR + LLM）—— 只在 ``--live`` 时执行。"""
    print("\n=== 真实联网跑一次（--live）===")
    if not url:
        url = input("粘贴一个视频/文章链接: ").strip()
    if not url:
        print("没有链接，跳过。")
        return

    result = run_replicate(url)
    print(f"\nsource_url : {result['source_url']}")
    print(f"original_text 长度: {len(result['original_text'])}")
    print(f"  前 200 字: {result['original_text'][:200]}")
    print(f"viral_analysis 长度: {len(result['viral_analysis'])}")
    print(f"  前 200 字: {result['viral_analysis'][:200]}")
    print(f"rewritten 长度: {len(result['rewritten'])}")
    print(f"titles 长度: {len(result['titles'])}")
    print(f"  前 200 字: {result['titles'][:200]}")


if __name__ == "__main__":
    print("=== 内容复刻工作流 自检（离线，不联网）===")

    # ---- 1) 图结构 ----
    graph = replicate_graph.get_graph()
    print(f"图节点: {sorted(graph.nodes)}")
    print(f"图边:   {graph.edges}")

    assert sorted(graph.nodes) == [
        "__end__", "__start__", "analyze", "extract", "rewrite", "titles",
    ], sorted(graph.nodes)

    edge_pairs = sorted((e.source, e.target) for e in graph.edges)
    assert edge_pairs == [
        ("__start__", "extract"),
        ("analyze", "rewrite"),
        ("extract", "analyze"),
        ("rewrite", "titles"),
        ("titles", "__end__"),
    ], edge_pairs
    print("  ✓ 4 节点串行链路接对了")

    # ---- 2) State 契约冻结 ----
    assert list(ReplicateState.__annotations__) == [
        "source_url", "original_text", "viral_analysis", "rewritten", "titles",
    ], ReplicateState.__annotations__
    print("  ✓ ReplicateState 字段与契约一致")

    # ---- 3) 打桩：下载 / ASR / 文章抓取 / LLM 全换成本地假数据 ----
    _real_download = download_video
    _real_asr = extract_audio_text
    _real_article = fetch_article
    _real_llm = llm_call

    calls: dict[str, list] = {"download": [], "asr": [], "article": [], "llm": []}

    def _stub_llm(prompt: str, temperature: float = 0.5, fallback: str = "") -> str:
        calls["llm"].append((prompt, temperature))
        return f"[STUB-{len(calls['llm'])}]"

    def wire(download_result: str, asr_result: str, article_result: str) -> None:
        """把四个外部依赖换成可编排的桩。"""
        def _stub_download(url: str, output_dir: str = None) -> str:
            calls["download"].append(url)
            return download_result

        def _stub_asr(path: str) -> str:
            calls["asr"].append(path)
            return asr_result

        def _stub_article(url: str) -> str:
            calls["article"].append(url)
            return article_result

        globals()["download_video"] = _stub_download
        globals()["extract_audio_text"] = _stub_asr
        globals()["fetch_article"] = _stub_article

    def clear_calls() -> None:
        for value in calls.values():
            value.clear()

    def reset_globals() -> None:
        """恢复真实依赖。**故意不清 calls** ——
        断言失败时 message 里那个列表是活引用，finally 里清空会让报错信息变成 ``[]``，
        完全看不出真正调了什么（本文件第一版就踩了这个坑，见文件头「踩过的坑」）。
        """
        globals()["download_video"] = _real_download
        globals()["extract_audio_text"] = _real_asr
        globals()["fetch_article"] = _real_article

    llm_call = _stub_llm  # noqa: F841

    try:
        # 3a) 视频链路：下载成功 + ASR 出文案 → 4 个节点全跑，共 3 次 LLM
        clear_calls()
        wire("C:/fake/demo.mp4", "大家好，今天讲三个提升效率的AI工具。", "")
        video_result = run_replicate("https://www.bilibili.com/video/BV1xx")

        assert calls["download"] == ["https://www.bilibili.com/video/BV1xx"], calls["download"]
        assert calls["asr"] == ["C:/fake/demo.mp4"], calls["asr"]
        assert calls["article"] == [], "视频链路拿到文案后不该再抓文章"
        assert video_result["original_text"].startswith("大家好"), video_result["original_text"]
        assert video_result["viral_analysis"] == "[STUB-1]"
        assert video_result["rewritten"] == "[STUB-2]"
        assert video_result["titles"] == "[STUB-3]"
        assert [t for _, t in calls["llm"]] == [0.6, 0.8, 0.8], calls["llm"]
        assert "大家好" in calls["llm"][0][0]
        assert "[STUB-1]" in calls["llm"][1][0]
        assert "[STUB-2]" in calls["llm"][2][0]
        print("  ✓ 视频链路：下载 → ASR → 拆解 → 仿写 → 标题（3 次 LLM）")

        # 3b) 分享口令：download_video 应该收到的是洗过的纯 URL
        clear_calls()
        wire("C:/fake/demo.mp4", "文案", "")
        run_replicate("7.65 复制打开抖音，看看【某某的作品】https://v.douyin.com/abcd/ 快来")
        assert calls["download"] == ["https://v.douyin.com/abcd/"], calls["download"]
        print("  ✓ 抖音分享口令里的纯 URL 被正确抠出来")

        # 3c) 文章兜底：下载失败 → 走 fetch_article，链路照样走完
        clear_calls()
        wire("", "", "这是一篇讲效率工具的公众号文章正文。")
        article_result = run_replicate("https://mp.weixin.qq.com/s/xxxx")
        assert calls["download"] == ["https://mp.weixin.qq.com/s/xxxx"], calls["download"]
        assert calls["asr"] == [], "下载失败时不该拿空路径去调 ASR"
        assert calls["article"] == ["https://mp.weixin.qq.com/s/xxxx"], calls["article"]
        assert article_result["original_text"].startswith("这是一篇"), article_result["original_text"]
        assert len(calls["llm"]) == 3, len(calls["llm"])
        print("  ✓ 视频下载失败 → 文章抓取兜底 → 链路走完")

        # 3d) 两条路都失败 → 短路，一次 LLM 都不调
        clear_calls()
        wire("", "", "")
        dead_result = run_replicate("https://example.com/nothing")
        assert dead_result["original_text"].startswith(_SKIP_MARK), dead_result["original_text"]
        assert calls["llm"] == [], f"拿不到文案时不该调 LLM，实际调了 {len(calls['llm'])} 次"
        assert dead_result["viral_analysis"].startswith(_SKIP_MARK)
        assert dead_result["rewritten"].startswith(_SKIP_MARK)
        assert dead_result["titles"].startswith(_SKIP_MARK)
        assert set(dead_result) == {
            "source_url", "original_text", "viral_analysis", "rewritten", "titles",
        }, set(dead_result)
        print("  ✓ 文案拿不到时短路：4 个字段都给中文提示，0 次 LLM")

        # 3e) 空链接
        clear_calls()
        wire("", "", "")
        empty_result = run_replicate("")
        assert empty_result["original_text"].startswith(_SKIP_MARK)
        assert calls["llm"] == []
        print("  ✓ 空链接不会崩（download_video/fetch_article 都白名单拒了空串）")
    finally:
        reset_globals()
        llm_call = _real_llm

    print("\n全部自检通过")

    if "--live" in sys.argv:
        argv = [a for a in sys.argv[1:] if a != "--live"]
        _live_check(argv[0] if argv else "")
    else:
        print("（跳过联网验证；加 --live \"<链接>\" 可真实下载一次）")
