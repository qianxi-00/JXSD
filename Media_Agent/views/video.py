# -*- coding: utf-8 -*-
"""口播视频页面 —— 提词器 / 数字人出镜

课案出处：自媒体课案 → 口播视频 → views/video.py

与课案的差异
    | 课案 | 本项目 |
    |---|---|
    | 模特管理目录：HeyGem 的 Docker 挂载目录（``c:/duix_avatar_data/face2face``） | ``MEDIA_AVATAR_INPUT_DIR``（项目内 ``.cache/avatars``），不再依赖 Docker |
    | 生成方式：提交 HeyGem，拿 task_code 后点刷新 | 提交 PixVerse 对口型，同样拿 task_code 后点刷新（交互一致） |
    | 音色：只有「克隆声音」一条路 | 增加「PixVerse 内置音色」下拉 —— 克隆不可用时也能一步出片 |
    | 选中模特的记忆文件 ``.cache/heygem_state.json`` | ``.cache/avatar_state.json`` |

保留课案的设计
    · **提词器**是纯前端 HTML/JS 大字滚动（每次 3 行、1~15 秒/行可调），不调任何模型。
    · 结果先存 ``st.session_state`` 再渲染 —— 下载按钮会触发 rerun，不缓存页面会空白。
    · 模特用网格展示，支持预览 / 选择 / 二次确认删除。
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import settings  # noqa: E402

# 状态文件放在项目 .cache 下（课案放在 CWD 下的 .cache，会随启动目录漂移）
STATE_FILE = Path(_HERE) / ".cache" / "avatar_state.json"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

# 会话状态里需要初始化的键
_SESSION_KEYS = (
    "hg_has_result", "hg_video_path", "hg_audio_path",
    "hg_task_code", "hg_msg", "hg_script", "hg_mode",
)


# --------------------------------------------------------------------------
# 状态文件（记忆上次选中的模特）
# --------------------------------------------------------------------------
def _load_selected_avatar() -> str:
    try:
        if STATE_FILE.is_file():
            path = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("avatar_path", "")
            if path and os.path.exists(path):
                return path
    except Exception:  # noqa: BLE001 —— 状态文件坏了不该影响页面
        pass
    return ""


def _save_selected_avatar(path: str) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps({"avatar_path": path}, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[口播视频] 选中状态保存失败（忽略）: {exc}")


# --------------------------------------------------------------------------
# 模特列表
# --------------------------------------------------------------------------
def _list_avatars() -> list:
    """扫描模特目录，列出所有可用的视频/图片。

    跳过中间产物（``_muted`` 后缀、``tts_``/``voice_`` 前缀等），
    这些是流程自己生成的，不该出现在选择列表里。
    """
    avatar_dir = Path(settings.media.get_avatar_input_dir())
    avatars = []
    if not avatar_dir.is_dir():
        return avatars

    seen = set()
    try:
        entries = sorted(
            avatar_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
    except Exception:  # noqa: BLE001
        return avatars

    for full in entries:
        if not full.is_file():
            continue
        ext = full.suffix.lower()
        if ext not in VIDEO_EXTS and ext not in IMAGE_EXTS:
            continue
        name = full.stem
        if name.endswith("_muted"):
            continue
        low = str(full).lower()
        if any(p in low for p in ("voice_", "voiceover_", "voice_cloned", "tts_", "test_")):
            continue
        if name in seen:
            continue
        seen.add(name)
        avatars.append({
            "path": str(full),
            "name": full.name,
            "type": "video" if ext in VIDEO_EXTS else "image",
            "size_mb": full.stat().st_size / 1024 / 1024,
        })
    return avatars


# --------------------------------------------------------------------------
# 提词器（纯前端，课案原有的 HTML/JS 滚动实现）
# --------------------------------------------------------------------------
def _render_teleprompter(script: str) -> None:
    """大字提词器：每次 3 行，按设定秒数自动推进，到底自动停止。"""
    st.markdown("### 📺 提词器")

    speed = st.slider(
        "⏱ 换行速度（秒/行）", min_value=1, max_value=15, value=5,
        help="数值越小换行越快",
    )

    lines = [ln.strip() for ln in script.split("\n") if ln.strip()] or [script]
    total = len(lines)

    import streamlit.components.v1 as components

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
* {{ margin:0; padding:0; }}
body {{ background:#000; font-family:'Microsoft YaHei','PingFang SC',sans-serif; }}
.tp-line {{
    height: 106px; display: flex; align-items: center; justify-content: center;
    text-align: center; color: #fff; font-size: 50px; font-weight: bold;
    letter-spacing: 3px; padding: 0 12px;
}}
.tp-box {{ padding: 5px 0; height: 318px; overflow: hidden; }}
</style></head><body><div class="tp-box">
<div id="l1" class="tp-line"></div>
<div id="l2" class="tp-line"></div>
<div id="l3" class="tp-line"></div>
</div><script>
var lines = {json.dumps(lines, ensure_ascii=False)};
var total = lines.length;
var pos = 0;
function show() {{
    document.getElementById('l1').textContent = lines[pos] || '';
    document.getElementById('l2').textContent = lines[pos+1] || '';
    document.getElementById('l3').textContent = lines[pos+2] || '';
    pos++;
    if (pos < total - 2) setTimeout(show, {speed * 1000});
}}
show();
</script></body></html>"""
    components.html(html, height=330, scrolling=False)
    st.caption(f"💡 共 {total} 行，每次 3 行大字，{speed} 秒换一行 | 到底自动停止")


# --------------------------------------------------------------------------
# 页面
# --------------------------------------------------------------------------
def show_video() -> None:
    """口播视频 —— 提词器 / 数字人出镜。"""
    st.title("🎥 口播视频智能体")
    st.markdown("输入台词 → 提词器滚动 / 数字人出镜 → 视频下载")

    for key in _SESSION_KEYS:
        st.session_state.setdefault(key, "")

    mode = st.radio(
        "📌 生成模式",
        ["📺 提词器模式（只看稿）", "🎭 数字人生成（对口型 → MP4）"],
        horizontal=True,
    )
    is_avatar = "数字人" in mode

    raw = st.text_area(
        "✍️ 输入你的文案", height=180,
        placeholder="直接输入你要说的台词，原样使用不做改写...",
    )

    avatar_path = ""
    speaker_id = "auto"
    use_cloned_voice = True

    if is_avatar:
        st.markdown("#### 🎭 选择数字人模特")
        st.caption(
            "需要一段 **10~30 秒正面说话视频**（mp4/mov）。"
            "对口型靠的是视频里的面部运动特征，**照片只能生成静态画面**。"
        )

        avatars = _list_avatars()
        saved = _load_selected_avatar()
        selected_idx = next(
            (i for i, a in enumerate(avatars) if a["path"] == saved), -1
        )

        if avatars:
            cols_per_row = 4
            for row_start in range(0, len(avatars), cols_per_row):
                row = avatars[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for ci, a in enumerate(row):
                    idx = row_start + ci
                    with cols[ci]:
                        try:
                            data = Path(a["path"]).read_bytes()
                            if a["type"] == "image":
                                st.image(data, use_container_width=True)
                            else:
                                st.video(data)
                        except Exception:  # noqa: BLE001
                            st.warning("无法加载预览")

                        icon = "🎬" if a["type"] == "video" else "🖼️"
                        size = (
                            f"{a['size_mb']:.1f}MB" if a["size_mb"] < 1000
                            else f"{a['size_mb'] / 1024:.1f}GB"
                        )
                        selected = idx == selected_idx
                        st.caption(f"{'⭐ ' if selected else ''}{icon} {a['name'][:25]}\n{size}")

                        c_sel, c_del = st.columns([3, 1])
                        with c_sel:
                            if st.button(
                                "✅ 已选中" if selected else "🔘 选择此模特",
                                key=f"sel_avatar_{idx}",
                                use_container_width=True,
                                type="primary" if not selected else "secondary",
                                disabled=selected,
                            ):
                                _save_selected_avatar(a["path"])
                                st.rerun()
                        with c_del:
                            if st.button("🗑️", key=f"del_avatar_{idx}", help="删除此模特"):
                                st.session_state[f"confirm_del_{idx}"] = True

                        if st.session_state.get(f"confirm_del_{idx}"):
                            st.warning(f"确认删除 `{a['name']}`？")
                            cc1, cc2 = st.columns(2)
                            with cc1:
                                if st.button("✅ 确认", key=f"confirm_yes_{idx}", type="primary"):
                                    try:
                                        os.remove(a["path"])
                                        if selected:
                                            _save_selected_avatar("")
                                        st.success(f"已删除: {a['name']}")
                                        st.session_state.pop(f"confirm_del_{idx}", None)
                                        st.rerun()
                                    except Exception as exc:  # noqa: BLE001
                                        st.error(f"删除失败: {exc}")
                            with cc2:
                                if st.button("❌ 取消", key=f"confirm_no_{idx}"):
                                    st.session_state.pop(f"confirm_del_{idx}", None)
                                    st.rerun()
            st.markdown("---")
        else:
            st.info("还没有模特视频，请在下方上传一段。")

        new_file = st.file_uploader(
            "📤 上传新模特（10~30 秒正面说话视频）",
            type=["mp4", "mov", "avi", "png", "jpg", "jpeg"],
            key="new_avatar_upload",
        )
        if new_file:
            avatar_dir = Path(settings.media.get_avatar_input_dir())
            avatar_dir.mkdir(parents=True, exist_ok=True)
            save_path = avatar_dir / f"avatar_{abs(hash(new_file.name)) % 100000:05d}{Path(new_file.name).suffix}"
            save_path.write_bytes(new_file.getvalue())
            _save_selected_avatar(str(save_path))
            st.success(f"✅ 已上传: {save_path.name}")
            st.rerun()

        if saved and os.path.exists(saved):
            avatar_path = saved

        st.markdown("---")
        st.markdown("#### 🔊 配音方式")
        use_cloned_voice = st.checkbox(
            "优先使用克隆音色（CosyVoice 声音复刻）", value=True,
            help="从模特视频里克隆声音来配台词。失败会自动降级为 edge-tts 通用音色。",
        )
        if not use_cloned_voice:
            from tools.avatar_client import PIXVERSE_SPEAKERS

            labels = [f"{k} - {v}" for k, v in PIXVERSE_SPEAKERS.items()]
            pick = st.selectbox("PixVerse 内置音色", labels, index=0)
            speaker_id = pick.split(" - ")[0]
            st.caption("内置音色由 PixVerse 直接合成语音，一步出片，但无法使用你自己的声音。")

    # ---------------- 开始生成 ----------------
    if st.button("🎬 开始生成", type="primary", use_container_width=True):
        if not raw:
            st.warning("请输入台词")
            return

        mode_key = "avatar" if is_avatar else "teleprompter"
        with st.spinner("处理中..."):
            from workflows.video import run_video

            result = run_video(
                raw, mode=mode_key, avatar_path=avatar_path,
                use_cloned_voice=use_cloned_voice, speaker_id=speaker_id,
            )

        st.session_state["hg_script"] = raw
        st.session_state["hg_audio_path"] = result.get("audio_path", "")
        st.session_state["hg_video_path"] = result.get("video_path", "")
        st.session_state["hg_task_code"] = result.get("task_code", "")
        st.session_state["hg_msg"] = result.get("avatar_msg", "")
        st.session_state["hg_mode"] = mode_key
        st.session_state["hg_has_result"] = True
        st.rerun()

    # ---------------- 渲染结果 ----------------
    if not st.session_state.get("hg_has_result"):
        return

    mode_key = st.session_state.get("hg_mode", "teleprompter")
    script = st.session_state.get("hg_script", "")

    if mode_key == "teleprompter":
        _render_teleprompter(script)
    else:
        st.markdown("### 📝 台词（原样使用，未改写）")
        st.text_area("台词内容", script, height=150, key="script_display")
        st.download_button("📥 下载台词", script, file_name="台词.txt")

        st.markdown("### 🎥 视频")
        video_path = st.session_state.get("hg_video_path", "")
        audio_path = st.session_state.get("hg_audio_path", "")
        task_code = st.session_state.get("hg_task_code", "")
        msg = st.session_state.get("hg_msg", "")

        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path)

        if video_path and os.path.exists(video_path):
            st.success("✅ 数字人视频已生成")
            st.video(video_path)
            with open(video_path, "rb") as fh:
                st.download_button(
                    "⬇️ 下载视频", fh.read(),
                    file_name=os.path.basename(video_path), mime="video/mp4",
                )
        elif task_code:
            st.info(msg or "任务已提交，约 2~5 分钟完成")
            st.caption(f"任务编号: `{task_code}`")
            if st.button("🔄 刷新进度（生成完毕后点击查看）"):
                from workflows.video import refresh_avatar_task

                with st.spinner("查询中..."):
                    q = refresh_avatar_task(task_code)
                if q["success"]:
                    st.session_state["hg_video_path"] = q["video_path"]
                    st.session_state["hg_msg"] = q["message"]
                else:
                    st.session_state["hg_msg"] = q["message"]
                st.rerun()
        elif msg:
            st.warning(msg)

    if st.button("🗑️ 清除重新开始"):
        for key in _SESSION_KEYS:
            st.session_state.pop(key, None)
        st.rerun()
