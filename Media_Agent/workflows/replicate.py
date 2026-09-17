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
       短路判据同样要认「LLM 调用失败」—— ``llm_call`` 失败时返回的是
       ``[LLM调用失败: …]`` / ``[LLM未配置] …``，那也不是正文，
       不能拿去继续拆解/仿写（否则报错串会被当成文案，还白烧两次 LLM）。
    4. 视频链路之外还有**文章兜底**：课案的流程图写的是「下载视频/抓取文章」，
       但代码只实现了视频那一半。抖音/B站下载失败时改用 ``fetch_article()`` 抓正文，
       知乎/公众号这类图文链接也能走完整条链路。

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 提取节点 | `download_video` → `extract_audio_text`，不判空 | 先 `extract_url` 洗一遍链接，视频链路失败改走 `fetch_article` | 课案流程图本来就写了「下载视频/抓取文章」；下载失败时纯视频链路必然得到空串 |
    | 空文案 | 不判空，直接喂 LLM ×3 | 短路 + 中文提示，逐级跳过（判据见 `_is_usable`） | 否则模型会为「空文案」编出一篇爆款拆解，前端看起来像成功了 |
    | LLM 失败串 | 无此概念（课案直接 `ChatOpenAI(...).invoke`） | `[LLM调用失败]` / `[LLM未配置]` 同样算「不可用」，后续节点短路 | 本项目 `llm_call` 不抛异常、把异常收成提示文本，不认它就等于把报错当正文 |
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

    · 自检里那句 `llm_call = _stub_llm`（带 `# noqa: F841`）看着像一条写了没用的赋值，
      其实是**打桩能不能生效的关键**：`if __name__` 块在**模块作用域**执行，
      这一行重新绑定的是 `replicate` 自己的全局名 `llm_call`，
      而节点函数是运行时才按全局名去查的 —— 所以它确实把 LLM 换掉了。
      一旦把它挪进某个函数里变成普通局部变量，桩就再也打不上，
      自检会真去联网调模型（而且看起来还是「通过」的）。
    · 下载 / ASR / 抓取三个依赖同理，必须用 `globals()[...] = ...` 改写**本模块**的
      全局名：节点读的是 `replicate.download_video`，去改 `tools.media_tools` 里
      那几个原函数对节点没有任何影响。

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

# LLM 调用的失败串前缀 —— 见 ``workflows/__init__.py`` 的 ``llm_call``：
# 它**绝不抛异常**，失败时返回 ``[LLM调用失败: …]`` / ``[LLM未配置] …`` 两种提示文本。
# 这两种串不是正文：当成正文往下去，后面两个节点会拿着报错去「仿写」「起标题」，
# 各烧一次 LLM，最后把看起来像成功的产出渲染到页面上。
# （触发条件可达：ASR 与 LLM 是两把独立的 key，提取成功但 LLM 未配置/报错是常见组合。）
_LLM_FAIL_PREFIXES = ("[LLM调用失败", "[LLM未配置")


def _is_usable(text: str) -> bool:
    """判断上游产出能不能继续往下用（空串 / 中断标记 / LLM 失败串都不算）。

    这里是链路上**唯一**一道「上游产出能不能用」的闸门：四个节点都先过它，
    再决定要不要调 LLM。所以判据收在这里、而不是散在每个调用点的
    ``fallback=`` 上 —— 以后加新节点，只要沿用节点现有的写法，就自动继承
    「上游不可用就别往下传」；靠每个调用点自己记得传 ``fallback=_SKIP_MARK``
    更脆，新增一处忘了传就重新踩同一个坑。

    Args:
        text: 上游写进 state 的文本。允许空串，也容忍 ``None``
            （调用点已经先做过 ``state.get(...) or ""``，这里再兜一次，
            是为了让这个判据自己站得住 —— 传 ``None`` 进来不该 AttributeError）。

    Returns:
        ``True`` = 可以把它塞进 prompt 继续往下走；
        ``False`` = 必须短路，往下传 ``_SKIP_MARK`` 文案而不是拿它调 LLM。
    """
    # ``or ""`` 兜 None；``strip()`` 顺带把「只有换行/空格」的文案也判成不可用 ——
    # 那种文案喂给模型，它会一本正经地为空内容编一篇拆解出来。
    value = (text or "").strip()
    if not value or value.startswith(_SKIP_MARK):
        return False
    return not value.startswith(_LLM_FAIL_PREFIXES)


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

    Args:
        state: 只读 ``source_url``（可带分享口令）。

    Returns:
        差量 dict，且**只含** ``original_text``：

        * 成功 → 转写/抓取到的正文；
        * 两条路都失败 → ``⚠️`` 开头的中文提示，**不是空串** ——
          空串在下游看不出「断在第 1 步」，也没法告诉用户可能的原因。

    Raises:
        不抛异常。整段包在 ``try`` 里，任何一步炸了都转成 ``⚠️`` 文案写回 state；
        这是本模块的统一约定：失败也要让链路走完，页面才看得见断在哪。
    """
    try:
        raw = state.get("source_url", "") or ""
        # 抖音/小红书的分享口令是混合文本，先抠出纯 URL
        url = extract_url(raw)
        print(f"[内容复刻] ① 提取文案: {url[:100]}")

        text = ""
        # ---- 主链路：视频下载 → 语音识别 ----
        # `download_video` 的契约是**失败返回空串、绝不抛异常**（工具层刻意不抛，
        # 见 `tools/media_tools.py`）：链接不是视频、本机没装 yt-dlp、被平台反爬、
        # 解析结果为空 …… 全走这一条返回。所以这里判的是 `if video_path`，
        # 拿 `""` 去调 ASR 只会多打一行「路径为空」的日志。
        video_path = download_video(url)
        if video_path:
            # 这一步内部串了两跳：FFmpeg 抽单声道音频 → 百炼 qwen-audio-3.0-asr-flash
            # 转写；本机缺 ffmpeg 时它会退回「让识别服务直接吃视频文件」那条路。
            # 同样以空串表示失败，真因由 tools 层打印（未配 DASHSCOPE_API_KEY 等）。
            text = extract_audio_text(video_path)
            if not text:
                # 下载成功、只是没转出文字：真因在 ASR（没配 DASHSCOPE_API_KEY / 视频里
                # 没人声），此时**不抓正文** —— 同一个视频页 URL 交给 trafilatura 基本
                # 抓不到东西（B站/抖音页面没有多少可读正文），拿回空串照样短路，
                # 白花一次网络请求而已。
                print("[内容复刻] 视频已下载但转写为空，不再走文章兜底（视频页抓不出正文）")
        else:
            print("[内容复刻] 视频下载失败（链接不是视频 / 未装 yt-dlp / 被反爬）")

        # ---- 兜底：**视频拿不到**时才改按文章抓 ----
        # 课案流程图写的是「下载视频 / 抓取文章」二选一，但课案代码只实现了视频那半；
        # 知乎/公众号这类图文链接本来就没有视频可下，全靠它救。
        # ⚠️ 判据是「下载就失败」（`not video_path`），**不能**写成「`text` 为空」：
        #    后者把「下到了视频、只是没转出文字」也算进来，于是又拿视频页 URL 去抓
        #    一次注定抓不到的正文。课案的本意就是「视频拿不到才退而取正文」。
        if not video_path:
            print("[内容复刻] 改走文章抓取兜底...")
            text = fetch_article(url)

        # 走到这里还空 → 链路到此为止：返回的文案以 `_SKIP_MARK` 开头，
        # 下游三个节点过 `_is_usable` 时全部判 False、逐级往后传提示，
        # 最终**一次 LLM 都不调**（这就是本文件相对课案多出来的那条短路）。
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
    """节点②：拆解爆款要素（钩子/结构/情绪/金句/互动引导）。

    Args:
        state: 只读 ``original_text``。

    Returns:
        差量 dict，且**只含** ``viral_analysis``：

        * 正常 → LLM 输出的五维拆解文本；
        * 上游不可用（``_is_usable`` 为 False）→ ``⚠️`` 提示，**不调 LLM**；
        * LLM 自己失败 → 原样保留失败串 ``[LLM调用失败: …]`` / ``[LLM未配置] …``。

    Raises:
        不抛异常：失败转成 ``⚠️`` 文案写回 state。
    """
    try:
        original_text = state.get("original_text", "") or ""
        if not _is_usable(original_text):
            print("[内容复刻] 没有可用文案，跳过爆款拆解")
            return {"viral_analysis": f"{_SKIP_MARK} 上游没拿到文案，无法拆解爆款结构。"}

        # 【prompt 契约】输入＝上一步的原始文案全文（不截断）；输出＝五维拆解文本
        # （钩子类型 / 结构模板 / 情绪节奏 / 金句亮点 / 互动引导）。
        # 这段 prompt 与课案**逐字一致**：拆解维度是本章的核心资产，改它等于换课。
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
        # 「拆解要稳」（判断题）→ 温度 0.6，取值依据见文件头的 `_TEMP_ANALYZE`。
        # `llm_call` **绝不抛异常**：失败时它返回 `[LLM调用失败: …]` / `[LLM未配置] …`，
        # 这里刻意**原样写进 state**（不换成 `_SKIP_MARK`）—— 报错原文比一句
        # 「拆解出错」更能定位问题；而下游 `_is_usable` 认这个前缀，照样短路，
        # 所以「留下真因」和「不该继续往下跑」两件事不冲突。
        return {"viral_analysis": llm_call(prompt, temperature=_TEMP_ANALYZE)}
    except Exception as exc:  # noqa: BLE001
        print(f"[内容复刻] ② 爆款拆解失败: {exc}")
        return {"viral_analysis": f"{_SKIP_MARK} 爆款拆解出错: {exc}"}


def node_rewrite(state: ReplicateState) -> dict:
    """节点③：按同一结构模板仿写新文案。

    Args:
        state: 读两个字段 —— ``viral_analysis`` 用于判短路，``original_text``
            作为 prompt 里的「参考原文」。

    Returns:
        差量 dict，且**只含** ``rewritten``：仿写好的新文案；
        上游不可用时不调 LLM，直接给 ``⚠️`` 提示；LLM 失败则保留失败串原文。

    Raises:
        不抛异常：失败转成 ``⚠️`` 文案写回 state。
    """
    try:
        viral_analysis = state.get("viral_analysis", "") or ""
        original_text = state.get("original_text", "") or ""
        # 短路**只认** `viral_analysis`：它是上一步的产出，也是链路断点的唯一信号。
        # `original_text` 虽然也读，但它只是 prompt 里的参考材料 ——
        # 拿不到它的情况在节点②就已经短路了，轮不到这里判。
        if not _is_usable(viral_analysis):
            print("[内容复刻] 没有爆款拆解结果，跳过仿写")
            return {"rewritten": f"{_SKIP_MARK} 缺少爆款拆解结果，无法仿写。"}

        # 【prompt 契约】输入＝① 上一步的五维拆解（决定结构模板）
        #                      ② 原始文案（决定内容与篇幅基准）
        # 输出＝一篇「同类不同题」的新文案，字数被要求压在原文的 80%~120%。
        # 同样是课案原文，不改。
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
        # 「仿写要活」（创作题）→ 温度 0.8；失败串的处理同节点②（原样写回 state）。
        return {"rewritten": llm_call(prompt, temperature=_TEMP_CREATIVE)}
    except Exception as exc:  # noqa: BLE001
        print(f"[内容复刻] ③ 仿写失败: {exc}")
        return {"rewritten": f"{_SKIP_MARK} 仿写出错: {exc}"}


def node_titles(state: ReplicateState) -> dict:
    """节点④：为仿写文案生成 5 种类型的爆款标题。

    Args:
        state: 只读 ``rewritten``（**不读**原文与拆解 —— 标题要贴着新文案写）。

    Returns:
        差量 dict，且**只含** ``titles``：5 种类型各一行的标题文本；
        上游不可用时不调 LLM，直接给 ``⚠️`` 提示；LLM 失败则保留失败串原文。

    Raises:
        不抛异常：失败转成 ``⚠️`` 文案写回 state。
    """
    try:
        rewritten = state.get("rewritten", "") or ""
        if not _is_usable(rewritten):
            print("[内容复刻] 没有仿写结果，跳过标题生成")
            return {"titles": f"{_SKIP_MARK} 缺少仿写文案，无法生成标题。"}

        # 【prompt 契约】输入＝仿写后的文案；输出＝5 种类型各一条标题
        # （数字型 / 疑问型 / 痛点型 / 悬念型 / 命令型，每条 ≤25 字）。
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
        # 温度 0.8 同节点③；失败串的处理同节点②。
        return {"titles": llm_call(prompt, temperature=_TEMP_CREATIVE)}
    except Exception as exc:  # noqa: BLE001
        print(f"[内容复刻] ④ 标题生成失败: {exc}")
        return {"titles": f"{_SKIP_MARK} 标题生成出错: {exc}"}


# ==========================================================================
# 构建图：START → extract → analyze → rewrite → titles → END
# ==========================================================================
# 为什么是「一条直线的 4 节点串行」而不是带 `add_conditional_edges` 的分支：
# 短路靠**节点内部**返回 `⚠️` 文案 + 下游 `_is_usable` 判断，不靠图上的条件边。
# 好处是图的形状恒定（自检就把它钉成 4 节点 5 边），四个字段在任何路径下都存在，
# `run_replicate`/前端直接取就行，不会因为「某个键没被写过」而 KeyError
# （LangGraph 的 invoke 只返回写过的键）。
# 代价是链路断了也会照常把后面几个节点走一遍 —— 但那些节点一次 LLM 都不调，
# 真实开销只是几次空转。
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
        链路中断时对应字段以 ``⚠️`` 开头并说明卡在哪一步，不会抛异常
        （例外：中断的正是某一步的 LLM 调用时，该字段保留 ``[LLM调用失败: …]`` /
        ``[LLM未配置] …`` 原文 —— 报错原文比 ``⚠️`` 更能说明问题，
        而它后面的字段照旧以 ``⚠️`` 短路）。

    Raises:
        无。图本身出错也被 ``try`` 兜住、写成 ``⚠️`` 文案 —— 课案没有这一层，
        单点异常会直接抛给 Streamlit 页面。
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
    """真实联网跑一次（下载视频 + ASR + LLM）—— 只在 ``--live`` 时执行。

    Args:
        url: 视频/文章链接；空串则改为在控制台交互式询问（输入为空就跳过）。
    """
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
    # **为什么必须打桩**：这四件事没有一个是本地的 ——
    #   ① `download_video` 要连 B站/抖音/小红书（还会被反爬）；
    #   ② `extract_audio_text` 是百炼付费 ASR；
    #   ③ `fetch_article` 要真去请求网页；
    #   ④ `llm_call` 一次调用就是钱。
    # 而自检要把「视频链路 / 文章兜底 / 两条都失败 / LLM 失败串」四种分支都跑一遍，
    # 所以只能把它们换成「返回值可控」的假实现，让整条图真的跑起来
    # （而不是只断言节点函数存在）。跑完在 `finally` 里还原：
    # 自检不能把模块留在打桩状态，否则 `import` 它的页面会拿到假函数。
    _real_download = download_video
    _real_asr = extract_audio_text
    _real_article = fetch_article
    _real_llm = llm_call

    # 调用记录：断言「谁被调了、调了几次、参数是什么」全靠它
    calls: dict[str, list] = {"download": [], "asr": [], "article": [], "llm": []}

    def _stub_llm(prompt: str, temperature: float = 0.5, fallback: str = "") -> str:
        """顶掉 `llm_call` 的桩：记下调用参数，返回 ``[STUB-N]`` 这种可判定的回显。

        回显里带序号（而不是固定字符串）是为了让断言能验证「节点③拿到的 prompt
        里确实有节点②的产出」—— 见 3a) 的 ``"[STUB-1]" in calls["llm"][1][0]``。
        签名必须与真 `llm_call` 一致（含 `fallback`），节点是按关键字传的。
        """
        calls["llm"].append((prompt, temperature))
        return f"[STUB-{len(calls['llm'])}]"

    def wire(download_result: str, asr_result: str, article_result: str) -> None:
        """把三个 I/O 依赖换成「返回指定结果」的桩，用来编排分支。

        返回值由调用方给，所以同一套桩既能演「下载成功」，也能演「下载失败」：
        ``wire("C:/fake/demo.mp4", "文案", "")`` 是视频链路，
        ``wire("", "", "文章正文")`` 是文章兜底，``wire("", "", "")`` 是两条都断。
        """
        def _stub_download(url: str, output_dir: str = None) -> str:
            calls["download"].append(url)
            return download_result

        def _stub_asr(path: str) -> str:
            calls["asr"].append(path)
            return asr_result

        def _stub_article(url: str) -> str:
            calls["article"].append(url)
            return article_result

        # **为什么改的是 `globals()` 而不是 `tools.media_tools` 里的原函数**：
        # 本文件在模块顶部做过 `from tools.media_tools import download_video`，
        # 名字已经绑到**本模块**的全局命名空间，节点运行时按全局名去查；
        # 替换 `tools.media_tools` 下的实现对节点毫无影响。
        globals()["download_video"] = _stub_download
        globals()["extract_audio_text"] = _stub_asr
        globals()["fetch_article"] = _stub_article

    def clear_calls() -> None:
        """清空四条调用记录 —— 每个用例**开头**调一次（不是 `finally`，见文件头）。"""
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

    # 这一行**不是**无用赋值：`if __name__` 块在模块作用域执行，
    # 所以它重新绑定的是本模块的全局名 `llm_call` —— 节点运行时才按全局名去查，
    # 桩因此生效（`# noqa: F841` 只是让 linter 别再报「局部变量没用」）。
    # 一旦把它挪进某个函数里变成普通局部变量，桩就再也打不上，
    # 自检会真的联网调模型（见文件头「踩过的坑」）。
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

        # 3c2) 视频**下载成功、但转写为空** → 不该再抓正文（本轮收窄的判据）
        #      改前判据是「`text` 为空」，于是会拿同一个视频页 URL 去 `fetch_article`：
        #      视频页抓不出正文（拿回空串照样短路，无害），但白花一次网络请求。
        #      这里的文章桩**故意返回一段正文** —— 真被调到的话，链路会带着这段
        #      并不存在的"正文"继续跑完 3 次 LLM，下面两条断言都会红。
        clear_calls()
        wire("C:/fake/demo.mp4", "", "一篇不该被用到的正文（视频页根本没有正文）")
        asr_empty = run_replicate("https://www.bilibili.com/video/BV1xx")
        assert calls["asr"] == ["C:/fake/demo.mp4"], calls["asr"]
        assert calls["article"] == [], "下载成功但转写为空时不该再抓正文"
        assert asr_empty["original_text"].startswith(_SKIP_MARK), asr_empty["original_text"]
        assert calls["llm"] == [], f"没有文案时不该调 LLM，实际调了 {len(calls['llm'])} 次"
        print("  ✓ 下载成功但转写为空：不再抓正文，0 次 LLM")

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
        #     注意这一条**只验了工作流侧的行为**（空串进来不崩、短路）。
        #     下面那句 print 说的「download_video/fetch_article 都白名单拒了空串」
        #     是**工具层**的性质，此刻三个函数已被桩替换、桩并不做白名单判断，
        #     所以那句话在离线自检里其实没被验证 —— 真正的覆盖在
        #     `tools/media_tools.py` 自己的自检里（`download_video("") == ""`）。
        clear_calls()
        wire("", "", "")
        empty_result = run_replicate("")
        assert empty_result["original_text"].startswith(_SKIP_MARK)
        assert calls["llm"] == []
        print("  ✓ 空链接不会崩（download_video/fetch_article 都白名单拒了空串）")

        # 3f) LLM 失败串不是正文：extract 成功、但 analyze 那次 LLM 报错 / 没配 key 时，
        #     rewrite 与 titles 必须短路 —— 否则会拿着报错去编仿写，还白烧 2 次 LLM。
        #     （两把 key 相互独立，「ASR 配了、LLM 没配」是可达组合。）
        def _make_fail_llm(text: str, counter: list):
            """造一个「一调就返回失败串」的 llm_call 桩（两种失败串复用同一份）。

            与 `_stub_llm` 的区别：它把调用记在**自己的** counter 里（不是共享的
            `calls["llm"]`），因为这组用例的判据就是「总共被调了几次」，
            单独计数才不会被前一个用例的残留干扰。
            """
            def _stub(prompt: str, temperature: float = 0.5, fallback: str = "") -> str:
                counter.append((prompt, temperature))
                return text

            return _stub

        # 两种失败串各跑一遍：它们前缀不同（`[LLM调用失败` / `[LLM未配置`），
        # 而 `_LLM_FAIL_PREFIXES` 是一个元组 —— 只测其中一种会漏掉另一种没配上的情况。
        for label, fail_text in (
            ("调用失败", "[LLM调用失败: boom]"),
            ("未配置", "[LLM未配置] 根目录 .env 里没有 API_KEY。"),
        ):
            clear_calls()
            wire("C:/fake/demo.mp4", "大家好，今天讲三个提升效率的AI工具。", "")
            fail_calls: list = []

            # 同样是模块级重绑定（见上文 `llm_call = _stub_llm` 处的注释），
            # 这一轮把 LLM 换成「一调就返回失败串」的桩。
            llm_call = _make_fail_llm(fail_text, fail_calls)  # noqa: F841
            fail_result = run_replicate("https://www.bilibili.com/video/BV1xx")

            assert fail_result["original_text"].startswith("大家好"), fail_result["original_text"]
            assert fail_result["viral_analysis"] == fail_text, fail_result["viral_analysis"]
            assert len(fail_calls) == 1, (
                f"LLM 失败串被当成正文了：后续节点又调了 {len(fail_calls) - 1} 次 LLM"
            )
            assert fail_result["rewritten"].startswith(_SKIP_MARK), fail_result["rewritten"]
            assert fail_result["titles"].startswith(_SKIP_MARK), fail_result["titles"]
            print(f"  ✓ LLM {label}串不算正文：只调 1 次 LLM，后续 2 个节点逐级短路")
    finally:
        # 还原三个 I/O 依赖与 LLM 桩。**故意不清 `calls`** ——
        # 断言失败时 traceback 里的 message 持有的是同一个 list，清空等于把证据抹掉。
        reset_globals()
        llm_call = _real_llm

    print("\n全部自检通过")

    if "--live" in sys.argv:
        argv = [a for a in sys.argv[1:] if a != "--live"]
        _live_check(argv[0] if argv else "")
    else:
        print("（跳过联网验证；加 --live \"<链接>\" 可真实下载一次）")
