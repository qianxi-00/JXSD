# -*- coding: utf-8 -*-
"""口播视频工作流 —— 提词器 / 数字人出镜

课案出处：自媒体课案 → 口播视频 → workflows/video.py

双模式（与课案一致）
    📺 **提词器模式**：用户原文 → 提词器滚动显示，**不做任何 LLM 改写**
    🎭 **数字人模式**：用户原文 → TTS 配音 → 数字人出镜 → MP4

与课案的差异
    | 课案 | 本项目 |
    |---|---|
    | 数字人引擎：本地 HeyGem（Docker 三容器 / AutoDL） | 百炼爱诗 PixVerse 视频对口型 |
    | 声音克隆：AutoDL Fish-Speech | 百炼 CosyVoice 声音复刻（``tools/voice_clone.py``） |
    | 降级配音：Edge TTS | 不变（保留课案的降级链） |
    | 模特素材目录：HeyGem 的 Docker 挂载目录 | ``MEDIA_AVATAR_INPUT_DIR``（项目内 ``.cache/avatars``） |
    | 模式名 ``mode="heygem"`` | 改为 ``mode="avatar"``（不再用 HeyGem，沿用旧名会误导） |

**修掉课案的一个真 bug**
    课案 ``node_generate_video`` 里写的是 ``state['optimized']``，
    但 ``VideoState`` 里根本没有 ``optimized`` 这个字段（只有 ``raw_script``）——
    数字人模式一跑就 ``KeyError``。本项目统一用 ``state["raw_script"]``。

课案的设计意图（保留）
    台词**原样使用，不做 LLM 改写** —— 用户是拿它当提词器读稿的，
    改写等于换了稿子。所以这条链路上一次 LLM 都不调。
"""

import sys

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class VideoState(TypedDict, total=False):
    """口播视频工作流的状态。

    说明：课案把这里写成 ``TypedDict`` 且节点里读 ``state["optimized"]``，
    本项目改成 ``total=False`` 并要求所有键都显式声明 ——
    少一个键就是 KeyError，``total=False`` 至少让「没赋值」和「没声明」区分开。
    """

    raw_script: str          # 用户输入的台词（原样使用）
    mode: str                # "teleprompter" | "avatar"
    avatar_path: str         # 数字人模特视频路径（10~30 秒正面说话）
    use_cloned_voice: bool   # 是否尝试用克隆音色（失败自动降级 edge-tts）
    speaker_id: str          # 走 PixVerse 内置 TTS 时的音色 ID

    teleprompter: str        # 提词器内容 = 用户原文
    audio_path: str          # 生成的配音音频
    video_path: str          # 生成的数字人视频（提交后需轮询，这里通常为空）
    task_code: str           # 数字人任务 ID
    avatar_msg: str          # 给界面看的进度/错误说明


# ==========================================================================
# 节点 1：提词器（原样保留，不做任何改写）
# ==========================================================================
def node_teleprompter(state: VideoState) -> dict:
    """用户输入即提词器内容，不做任何 LLM 改写。"""
    return {"teleprompter": state.get("raw_script", "")}


# ==========================================================================
# 节点 2：生成视频（TTS 配音 → 数字人提交）
# ==========================================================================
def node_generate_video(state: VideoState) -> dict:
    """数字人模式：TTS 配音 → 提交对口型任务。

    提词器模式直接短路返回。

    **本节点不阻塞等待**：对口型任务是异步的（官方说 1~5 分钟），
    在 Streamlit 里阻塞会把界面卡死。这里只提交、留下 ``task_code``，
    由页面的「刷新进度」按钮去轮询。
    """
    mode = state.get("mode", "teleprompter")

    if mode == "teleprompter":
        return {"video_path": "", "audio_path": "", "task_code": "", "avatar_msg": ""}

    script = state.get("raw_script", "")      # ← 课案这里错写成 state["optimized"]
    avatar_video = state.get("avatar_path", "")
    if not script.strip():
        return {"avatar_msg": "台词为空", "task_code": "", "audio_path": "", "video_path": ""}

    # 先校验模特视频：缺了就直接返回提示，**不要白跑一次 TTS**。
    # （课案的顺序是先配音后检查，会白白消耗一次合成额度。）
    if not avatar_video:
        return {
            "audio_path": "",
            "video_path": "",
            "task_code": "",
            "avatar_msg": (
                "请先上传一段模特视频！\n\n"
                "数字人对口型需要一段 10~30 秒的正面说话视频（mp4/mov），"
                "用它的面部运动特征来驱动口型。\n"
                "**照片只能生成静态画面**，必须用视频。"
            ),
        }

    # ---------- Step 1: 产出配音音频 ----------
    audio_path = ""
    if state.get("use_cloned_voice", True):
        audio_path = _try_cloned_voice(script, avatar_video)
    if not audio_path:
        audio_path = _try_edge_tts(script)

    # ---------- Step 2: 提交数字人任务 ----------
    from tools.avatar_client import submit_lipsync

    if audio_path:
        # 有配音 → 音频驱动（音色就是我们克隆/合成的那个）
        submitted = submit_lipsync(video_path=avatar_video, audio_path=audio_path)
        driver = "音频驱动（用生成的配音）"
    else:
        # 没有配音（克隆与 edge-tts 都失败）→ 退回 PixVerse 内置 TTS，一步出片
        submitted = submit_lipsync(
            video_path=avatar_video,
            tts_text=script,
            speaker_id=state.get("speaker_id", "auto"),
        )
        driver = "TTS 文本驱动（PixVerse 内置音色）"

    if submitted["success"]:
        return {
            "audio_path": audio_path,
            "video_path": "",
            "task_code": submitted["task_id"],
            "avatar_msg": (
                f"任务已提交（{driver}）\n\n"
                f"任务编号：`{submitted['task_id']}`\n\n"
                "约 2~5 分钟后点下方「刷新进度」查看成品。"
            ),
        }
    return {
        "audio_path": audio_path,
        "video_path": "",
        "task_code": "",
        "avatar_msg": submitted["message"],
    }


def _try_cloned_voice(script: str, avatar_video: str) -> str:
    """尝试「克隆音色 → 合成」。任何一步失败都静默返回空串，让上游降级。"""
    try:
        from tools.voice_clone import clone_voice, tts_with_cloned_voice

        cloned = clone_voice(avatar_video)
        if not cloned["success"]:
            print(f"[口播视频] 声音克隆不可用（{cloned['message']}），降级 edge-tts")
            return ""

        path = tts_with_cloned_voice(script, cloned["voice_id"])
        if path:
            print(f"[口播视频] 克隆音色配音完成: {path}")
            return path
        print("[口播视频] 克隆音色合成失败，降级 edge-tts")
        return ""
    except Exception as exc:  # noqa: BLE001 —— 降级链上任何异常都不该中断
        print(f"[口播视频] 声音克隆异常（{exc}），降级 edge-tts")
        return ""


def _try_edge_tts(script: str) -> str:
    """edge-tts 兜底配音（免费、无需密钥）。"""
    try:
        from tools.media_tools import generate_tts

        return generate_tts(script)
    except Exception as exc:  # noqa: BLE001
        print(f"[口播视频] edge-tts 异常: {exc}")
        return ""


# ==========================================================================
# 构图
# ==========================================================================
builder = StateGraph(VideoState)
builder.add_node("teleprompter", node_teleprompter)
builder.add_node("generate", node_generate_video)

builder.add_edge(START, "teleprompter")
builder.add_edge("teleprompter", "generate")
builder.add_edge("generate", END)

video_graph = builder.compile()


# ==========================================================================
# 对外接口
# ==========================================================================
def run_video(
    raw_script: str,
    mode: str = "teleprompter",
    avatar_path: str = "",
    use_cloned_voice: bool = True,
    speaker_id: str = "auto",
) -> dict:
    """运行口播视频工作流。

    Args:
        raw_script: 用户原始台词（原样使用，不做改写）。
        mode: ``"teleprompter"``（只看稿）或 ``"avatar"``（数字人出镜）。
        avatar_path: 模特视频路径（数字人模式必填）。
        use_cloned_voice: 是否优先尝试克隆音色（失败自动降级 edge-tts）。
        speaker_id: 走 PixVerse 内置 TTS 时的音色 ID。

    Returns:
        ``{"teleprompter","audio_path","video_path","task_code","avatar_msg"}``
        —— 失败也在 ``avatar_msg`` 里给中文说明，不抛异常。
    """
    return video_graph.invoke({
        "raw_script": raw_script,
        "mode": mode,
        "avatar_path": avatar_path,
        "use_cloned_voice": use_cloned_voice,
        "speaker_id": speaker_id,
    })


def refresh_avatar_task(task_code: str) -> dict:
    """轮询数字人任务并（完成后）把成品下载到本地。

    给页面「刷新进度」按钮用。

    Returns:
        ``{"finished","success","video_path","message"}``
    """
    if not task_code:
        return {"finished": False, "success": False, "video_path": "",
                "message": "任务编号为空"}

    from tools.avatar_client import download_result, query_task

    q = query_task(task_code)
    if not q["finished"]:
        return {"finished": False, "success": False, "video_path": "",
                "message": q["message"]}

    if not q["success"]:
        return {"finished": True, "success": False, "video_path": "",
                "message": q["message"]}

    local = download_result(q["video_url"])
    if local:
        return {"finished": True, "success": True, "video_path": local,
                "message": "生成完成，已保存到本地"}
    return {"finished": True, "success": False, "video_path": "",
            "message": f"生成完成但下载失败，可手动访问: {q['video_url']}"}


if __name__ == "__main__":
    # 直接跑本文件时 sys.path[0] 是 workflows/，需要把 Media_Agent 加进来
    from pathlib import Path

    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    print("=== 口播视频工作流自检（离线）===")

    # 1) 图结构
    g = video_graph.get_graph()
    nodes = sorted(g.nodes)
    assert "teleprompter" in nodes and "generate" in nodes, nodes
    print(f"  图节点: {nodes}  OK")

    # 2) 提词器模式：原样返回，不调任何外部服务
    r = run_video("第一行台词\n第二行台词", mode="teleprompter")
    assert r["teleprompter"] == "第一行台词\n第二行台词", r
    assert r["audio_path"] == "" and r["task_code"] == "", r
    print("  提词器模式原样返回         OK")

    # 3) 数字人模式：没给模特视频 → 必须给出中文提示而不是抛异常
    r2 = run_video("你好", mode="avatar", avatar_path="")
    assert r2["task_code"] == "", r2
    assert "模特视频" in r2["avatar_msg"], r2["avatar_msg"]
    print("  数字人缺模特 → 中文提示    OK")

    # 4) 空台词
    r3 = run_video("", mode="avatar", avatar_path="x.mp4")
    assert "台词为空" in r3["avatar_msg"], r3
    print("  空台词处理                 OK")

    # 5) 课案的 KeyError 已修复：不再读 state['optimized']
    #    用字节码常量判断而不是搜源码 —— 源码里的**注释**也提到了这个词，
    #    搜字符串会误报（本条自检第一版就是这么挂的）。
    consts = node_generate_video.__code__.co_consts
    assert "optimized" not in consts, "仍然引用了不存在的 state['optimized']"
    print("  课案 KeyError 已修复       OK")

    # 6) 刷新接口的失败路径
    assert refresh_avatar_task("")["finished"] is False
    print("  refresh_avatar_task 空参   OK")

    print(f"\n  数字人模型: {settings.media.avatar_model}")
    print("全部自检通过")
