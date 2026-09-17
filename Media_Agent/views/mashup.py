# -*- coding: utf-8 -*-
"""视频剪辑页面 —— DeepAgents + video-use 技能 + HyperFrames

课案出处：自媒体课案 → 视频剪辑 → views/mashup.py

与课案的差异
    | 课案 | 本项目 |
    |---|---|
    | ``CACHE_DIR = ".cache/mashup"``（相对 CWD） | ``<项目目录>/.cache/mashup``（绝对路径，不随启动目录漂移） |
    | 课案 workflows 用 ``.cache/videos``、views 用 ``.cache/mashup``，**两边不一致**，兜底查找扫不到 | 两边统一取 ``settings.media.get_mashup_work_dir()`` |
    | ``DEFAULT_BGM`` 硬编码 ``C:\\Users\\13261\\Pictures\\风格\\...wav`` | 取 ``MEDIA_BGM_PATH``；没配就提示「本次不混音」 |
    | BGM 缓存文件 ``.cache/mashup/bgm.json`` | 同一个位置（但基准目录改成项目内绝对路径） |

保留课案的设计
    · 上传口播视频 + 多个穿插素材 + 自定义 BGM
    · BGM 路径可以手填也可以上传，选过的记在 bgm.json 里下次自动带出
    · 剪辑日志折叠展示，成品可预览可下载
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import settings  # noqa: E402

# 与 workflows/mashup.py 的沙箱目录保持一致（课案这里是分开的，导致兜底查找失效）
CACHE_DIR = Path(settings.media.get_mashup_work_dir())
MATERIALS_DIR = CACHE_DIR / "materials"
BGM_CACHE = CACHE_DIR / "bgm.json"

MEDIA_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mov", ".avi", ".webm")


def _load_bgm_path() -> str:
    """读上次用的 BGM 路径；没记录过就回落到配置里的 MEDIA_BGM_PATH。"""
    try:
        if BGM_CACHE.is_file():
            saved = json.loads(BGM_CACHE.read_text(encoding="utf-8")).get("bgm_path", "")
            if saved:
                return saved
    except Exception:  # noqa: BLE001
        pass
    return settings.media.bgm_path or ""


def _save_bgm_path(path: str) -> None:
    try:
        BGM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        BGM_CACHE.write_text(json.dumps({"bgm_path": path}, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[剪辑] BGM 路径保存失败（忽略）: {exc}")


def show_mashup() -> None:
    """视频剪辑 —— DeepAgents + video-use 技能 + HyperFrames 素材穿插。"""
    st.title("🎬 视频剪辑智能体")
    st.markdown(
        "上传口播视频 + 素材，AI 自动完成：生成动画素材 → 上下排布穿插 → "
        "字幕 → 背景音乐 → 渲染\n\n"
        "**技术栈**：`deepagents` + `video-use` 技能 + `moviepy`（+ `HyperFrames` 可选）"
    )

    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not settings.api_key:
        st.error("未配置 API_KEY，无法启动剪辑 agent（见根目录 .env）")
        return

    # ---------------- 输入视频 ----------------
    st.markdown("### 📁 输入口播视频")
    video_file = st.file_uploader(
        "上传口播视频", type=["mp4", "mov", "avi", "mkv", "webm"],
        help="主口播视频，AI 将在其中穿插素材",
    )

    video_path = ""
    if video_file:
        suffix = Path(video_file.name).suffix
        saved = CACHE_DIR / f"input_{abs(hash(video_file.name)) % 100000:05d}{suffix}"
        saved.write_bytes(video_file.getvalue())
        video_path = str(saved)
        st.video(video_file)
        st.success(f"已加载：{video_file.name}")

    path_input = st.text_input(
        "或输入视频路径", placeholder=r"例: F:\ProGram\Python_Base\Media_Agent\.cache\videos\xxx.mp4",
    )
    if path_input and os.path.exists(path_input):
        video_path = path_input
        st.video(path_input)

    # ---------------- 穿插素材 ----------------
    st.markdown("### 🖼️ 穿插素材（图片/视频）")
    st.caption("这些素材会由 AI 智能穿插到口播视频中，上下排布展示")

    material_files = st.file_uploader(
        "上传素材（可多选）",
        type=["png", "jpg", "jpeg", "gif", "webp", "mp4", "mov", "avi", "webm"],
        accept_multiple_files=True,
        help="上传多个图片或短视频，AI 会在口播的合适时机穿插展示",
    )

    material_paths = []
    if material_files:
        # 清掉旧素材，避免和这一次混在一起（课案原有行为）
        for old in MATERIALS_DIR.iterdir():
            try:
                if old.is_file():
                    old.unlink()
            except Exception:  # noqa: BLE001
                pass
        for mf in material_files:
            mp = MATERIALS_DIR / mf.name
            mp.write_bytes(mf.getvalue())
            material_paths.append(str(mp))
        st.success(f"已加载 {len(material_paths)} 个素材")

    existing = sorted(
        [f.name for f in MATERIALS_DIR.iterdir()
         if f.is_file() and f.suffix.lower() in MEDIA_EXTS],
        key=lambda x: (MATERIALS_DIR / x).stat().st_mtime, reverse=True,
    )
    if existing:
        st.caption(f"📂 已有素材 ({len(existing)} 个): {', '.join(existing[:10])}")

    # ---------------- 背景音乐 ----------------
    st.markdown("### 🎵 背景音乐")
    current_bgm = _load_bgm_path()

    col1, col2 = st.columns([3, 1])
    with col1:
        bgm_input = st.text_input(
            "背景音乐路径", value=current_bgm,
            placeholder="输入 WAV/MP3 文件路径（留空则不混音）",
            help="没配置就跳过混音，不影响其它剪辑步骤",
            key="bgm_path_input",
        )
    with col2:
        bgm_file = st.file_uploader(
            "📁 上传", type=["wav", "mp3", "m4a", "ogg"], help="上传新 BGM 会替换当前设置",
        )
        if bgm_file:
            target = CACHE_DIR / bgm_file.name
            target.write_bytes(bgm_file.getvalue())
            _save_bgm_path(str(target))
            st.success("已更新")
            st.rerun()

    if bgm_input != current_bgm:
        _save_bgm_path(bgm_input)
    bgm_path = bgm_input or current_bgm

    if bgm_path and os.path.exists(bgm_path):
        st.caption(f"✅ BGM: `{Path(bgm_path).name}`")
    elif bgm_path:
        st.warning(f"⚠️ BGM 文件不存在，本次将不混音: `{bgm_path}`")
    else:
        st.caption("未设置 BGM，本次不混音")

    # ---------------- 附加要求 ----------------
    st.markdown("### ✏️ 额外剪辑要求")
    requirements = st.text_area(
        "补充要求（可选）",
        placeholder="例：特定时间段剪掉、加特定文字、调色风格...",
        height=60,
    )

    # ---------------- 开始剪辑 ----------------
    if st.button("🎬 开始剪辑", type="primary", use_container_width=True):
        if not video_path:
            st.warning("请先上传口播视频或输入视频路径")
            return
        if not os.path.exists(video_path):
            st.error(f"视频文件不存在: {video_path}")
            return

        use_materials = material_paths or [
            str(MATERIALS_DIR / f)
            for f in existing
            if (MATERIALS_DIR / f).is_file()
        ]

        with st.spinner("🤖 deepagent 正在分析视频并剪辑（这一步可能要几分钟）..."):
            from workflows.mashup import mashup_graph

            result = mashup_graph.invoke({
                "input_video": video_path,
                "edit_style": "",
                "edit_requirements": json.dumps({
                    "materials": use_materials,
                    "bgm_path": bgm_path,
                    "extra_requirements": requirements,
                }, ensure_ascii=False),
            })

        st.session_state["mashup_result"] = {
            "output_video": result.get("output_video", ""),
            "editor_log": result.get("editor_log", ""),
            "steps": result.get("steps", ""),
        }
        st.session_state["mashup_input"] = video_path
        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M"),
            "action": "视频剪辑",
            "summary": Path(video_path).name,
        })

    # ---------------- 渲染结果 ----------------
    res = st.session_state.get("mashup_result")
    if not res:
        return

    editor_log = res.get("editor_log", "")
    output_video = res.get("output_video", "")

    if editor_log:
        with st.expander("📝 剪辑日志", expanded=not output_video):
            st.text(editor_log[:3000])

    with st.expander("🔧 工具调用轨迹", expanded=False):
        try:
            steps = json.loads(res.get("steps") or "[]")
        except Exception:  # noqa: BLE001
            steps = []
        if steps:
            for s in steps[:80]:
                st.text(str(s)[:300])
        else:
            st.caption("（本次没有记录到工具调用）")

    if output_video and os.path.exists(output_video):
        st.success("✅ 剪辑完成！")
        st.video(output_video)
        with open(output_video, "rb") as fh:
            st.download_button(
                "⬇️ 下载视频", fh.read(),
                file_name=Path(output_video).name, mime="video/mp4",
            )
        st.caption(f"输出文件：`{output_video}`")
    else:
        st.warning(
            f"没有找到成片。请检查上面的剪辑日志；\n"
            f"也可以手动去看沙箱目录：`{CACHE_DIR}`"
        )
