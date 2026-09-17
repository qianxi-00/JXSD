# -*- coding: utf-8 -*-
"""口播视频页面 —— 提词器 / 数字人出镜

课案出处：自媒体课案 → 口播视频 → views/video.py

本节要讲什么
    1. **一个 radio 分出两条完全不同的链路**：提词器模式只渲染一段自包含的前端
       HTML/JS（不调模型、不花钱），数字人模式才走 ``run_video()``
       （克隆音色 → TTS → PixVerse 对口型）。两条链路的结果写进同一组 ``hg_*``
       session_state 键（``hg`` = 课案那套 HeyGem 的遗留前缀）。
    2. **提交与取结果是分开的两次交互**：PixVerse 出片要 2~5 分钟，提交后拿到的是
       ``task_code``；页面不阻塞轮询，而是给一个「🔄 刷新进度」按钮，
       用户想什么时候来取就什么时候点（见 ``workflows.video.refresh_avatar_task()``）。
    3. **模特就是目录里的真实文件**：列表直接扫 ``MEDIA_AVATAR_INPUT_DIR``，
       「上次选中谁」记在 ``.cache/avatar_state.json``；删除走「先点 🗑️ 再确认」两步，
       删的正好是当前选中项时会把记忆一起清掉。
    4. **结果先落 session_state 再渲染**：本页有六个按钮（选模特 / 删模特 / 确认删除 /
       下载台词 / 下载视频 / 刷新进度）都会触发整页 rerun，只存局部变量页面会空白。

与课案的差异
    | 课案 | 本项目 |
    |---|---|
    | 模特管理目录：HeyGem 的 Docker 挂载目录（``c:/duix_avatar_data/face2face``） | ``MEDIA_AVATAR_INPUT_DIR``（项目内 ``.cache/avatars``），不再依赖 Docker |
    | 生成方式：提交 HeyGem，拿 task_code 后点刷新 | 提交 PixVerse 对口型，同样拿 task_code 后点刷新（交互一致） |
    | 音色：只有「克隆声音」一条路 | 增加「PixVerse 内置音色」下拉，且**常驻**显示 —— 不勾克隆时它是主音色（一步出片），勾了则是克隆与 edge-tts 都失败后的兜底音色 |
    | 选中模特的记忆文件 ``.cache/heygem_state.json`` | ``.cache/avatar_state.json`` |
    | 模式名 ``mode="heygem"`` | 改为 ``mode="avatar"``（引擎换了，沿用旧名会误导） |
    | 上传框收「照片或视频」 | **只收视频** —— PixVerse 对口型要视频，照片只能出静态画面（页面文案原本就这么写了） |
    | ``_add_history`` 在按钮分支 | 同样放按钮分支（本项目其它 5 个页面统一如此；放渲染分支会每次 rerun 多记一条） |
    | 提词器用 ``st.components.v1.html`` | 改用 ``st.iframe``（前者 docstring 标明 2026-06-01 后移除，本机已在打弃用告警） |
    | ``use_container_width=True`` | ``width="stretch"``（streamlit 1.61 已弃用前者） |
    | 产物名用 ``abs(hash(...))`` | 改用 ``hashlib.md5(...)[:8]`` —— 字符串 hash 带进程级随机盐，跨进程不稳定、只堆积不复用 |

保留课案的设计
    · **提词器**是纯前端 HTML/JS 大字滚动（每次 3 行、1~15 秒/行可调），不调任何模型。
    · 结果先存 ``st.session_state`` 再渲染 —— 下载按钮会触发 rerun，不缓存页面会空白。
    · 模特用网格展示，支持预览 / 选择 / 二次确认删除。

踩过的坑
    · **上传的模特文件不能直接用原文件名**：模特目录是共享的，两台机器都传
      ``demo.mp4`` 会互相覆盖，所以落盘名取 ``md5(原始文件名)[:8]`` 当前缀。
      **不能**用内置 ``hash()`` —— 字符串 hash 带进程级随机盐，重启后同一个名字
      会算出另一个路径，目录里只堆积不复用。
    · **删除模特必须二次确认**：这个操作直接 ``os.remove`` 掉用户自己录的素材，
      没有回收站，按错一次就没了。确认框的中间态只能寄存在
      ``st.session_state[f"confirm_del_{idx}"]`` 里 —— ``st.button`` 的返回值
      只在点击那一轮为 True，承载不了跨 rerun 的状态。
    · **提词器没有暂停/继续**：拖动速度滑块会让那段 HTML 整体重挂一次，滚动从头开始。
      因为「播到第几行」是 iframe 内部 JS 的局部变量，Streamlit 侧看不到、
      也不需要知道（提词器不写 session_state）。

运行方式（由 main.py 侧边栏路由调用）::

    Set-Location F:\\ProGram\\Python_Base\\Media_Agent
    & ..\\.venv\\Scripts\\python.exe -m streamlit run main.py
"""

import hashlib
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
# 注：本项目不做图片模特 —— PixVerse 对口型要的是视频（照片只能出静态画面），
# 所以上传与列表都按 VIDEO_EXTS 过滤。原先那个 IMAGE_EXTS 常量已失去调用方，删掉。

# 上传框只收视频扩展名，且从 VIDEO_EXTS 派生 —— 手写两份会和模特列表的过滤走偏
AVATAR_UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in VIDEO_EXTS)

# 会话状态里需要初始化的键（hg = 课案那套 HeyGem 的前缀，引擎换了但键名沿用）
# 「🗑️ 清除重新开始」按钮 pop 的正好就是这一组
_SESSION_KEYS = (
    "hg_has_result", "hg_video_path", "hg_audio_path",
    "hg_task_code", "hg_msg", "hg_script", "hg_mode",
)


# --------------------------------------------------------------------------
# 状态文件（记忆上次选中的模特）
# --------------------------------------------------------------------------
def _load_selected_avatar() -> str:
    """读「上次选中哪个模特」（记忆文件 ``.cache/avatar_state.json``）。

    Returns:
        模特视频的绝对路径；没记录过、JSON 坏了、或记录的路径已经被删掉时一律返回
        ``""`` —— 调用方把空串当作「没选过」，页面照常渲染。
    """
    try:
        if STATE_FILE.is_file():
            path = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("avatar_path", "")
            # 记忆里存的是路径不是文件：用户手动清过 .cache 之后，
            # 这里必须再确认一次文件还在，否则会把一个不存在的路径当模特传下去
            if path and os.path.exists(path):
                return path
    except Exception:  # noqa: BLE001 —— 状态文件坏了不该影响页面
        pass
    return ""


def _save_selected_avatar(path: str) -> None:
    """记住当前选中的模特（传空串 = 清掉选择）。

    调用点：点击「选择此模特」、上传新模特、以及删除的正好是当前选中项时。
    写失败只在控制台打一行日志 —— 记住上次的选择是锦上添花，
    不能因为它不可写就让整个页面崩掉。
    """
    try:
        # 目录可能还不存在（首次运行、.cache 被清过），写之前先补上
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps({"avatar_path": path}, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[口播视频] 选中状态保存失败（忽略）: {exc}")


# --------------------------------------------------------------------------
# 操作历史（首页展示最近 10 条）
# --------------------------------------------------------------------------
def _add_history(action: str, summary: str) -> None:
    """往本次操作记录里追加一条（首页展示最近 10 条）。

    ``main.py`` 已经初始化了 ``st.session_state.history``；
    这里的 ``setdefault`` 只是让页面脱离 main.py 单独跑时也不炸。
    """
    from datetime import datetime

    history = st.session_state.setdefault("history", [])
    history.append({
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        # 只截 200 字符：首页那行只展示摘要，而且台词原文本来就存在 hg_script 里了
        "summary": summary[:200],
    })


# --------------------------------------------------------------------------
# 模特列表
# --------------------------------------------------------------------------
def _list_avatars() -> list:
    """扫描模特目录，列出所有可用的**视频**。

    图片（png/jpg…）选不了对口型 —— 只会生成静态画面，却会一路走到付费提交，
    所以照 ``VIDEO_EXTS`` 过滤掉。
    跳过中间产物（``_muted`` 后缀、``tts_``/``voice_`` 前缀等），
    这些是流程自己生成的，不该出现在选择列表里。

    Returns:
        ``[{"path","name","size_mb"}, ...]``，按文件修改时间**倒序** ——
        刚上传的排最前，不用滚到底去找。目录不存在或读不了时返回 ``[]``。
    """
    # 目录取自 config（项目内绝对路径 ``.cache/avatars``），不随启动目录漂移
    avatar_dir = Path(settings.media.get_avatar_input_dir())
    avatars = []
    if not avatar_dir.is_dir():
        return avatars

    # 按 stem 去重：同一段视频可能同时有 .mp4 与 .mov 两份，只留最新的那个
    seen = set()
    try:
        entries = sorted(
            avatar_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
    except Exception:  # noqa: BLE001
        # 目录读不了（权限 / 正被删）时当作「一个模特都没有」，
        # 让页面走「还没上传」的引导分支，而不是把异常抛到页面上
        return avatars

    for full in entries:
        if not full.is_file():
            continue
        ext = full.suffix.lower()
        if ext not in VIDEO_EXTS:
            continue
        name = full.stem
        # 中间产物一律不列：_muted 是流程自己去人声的中间文件、
        # tts_/voice_ 是配音产物 —— 出现在选择列表里只会让人选错，
        # 白白走一次付费的对口型提交
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
            # 换算成 MB 给缩略图下面那行说明用；1024 进制，与文件管理器的口径一致
            "size_mb": full.stat().st_size / 1024 / 1024,
        })
    return avatars


# --------------------------------------------------------------------------
# 提词器（纯前端，课案原有的 HTML/JS 滚动实现）
# --------------------------------------------------------------------------
def _render_teleprompter(script: str) -> None:
    """大字提词器：每次 3 行，按设定秒数自动推进，到底自动停止。

    纯前端实现：把一段自包含的 HTML/JS 用 ``st.iframe`` 挂上去，
    **不调任何模型、不读网络、也不写 ``st.session_state``** ——
    所以要花钱、要联网的只有数字人那条路。

    页面上的控件
        · ``st.slider``「⏱ 换行速度（秒/行）」1~15（默认 5）→ 只改这段 JS 里
          ``setTimeout`` 的间隔，值本身不落 session_state；
        · 底部 ``st.caption``：展示「共 N 行 / 每次 3 行 / M 秒换一行」。

    Args:
        script: 台词全文 —— 按换行切分并丢掉空行；全是空白时退化成一行原文。
    """
    st.markdown("### 📺 提词器")

    # 1~15 秒/行的依据：口播一行通常十几个字、语速 4~6 字/秒，5 秒是课案给的常见默认；
    # 下限 1 秒对应「快速过一遍」，上限 15 秒对应逐字精读
    speed = st.slider(
        "⏱ 换行速度（秒/行）", min_value=1, max_value=15, value=5,
        help="数值越小换行越快",
    )

    # 丢掉空行：口播稿里常有空行分段，留在提词器里就是「空一屏」；
    # `or [script]` 兜住全空输入，保证 JS 里至少有 1 行可显示
    lines = [ln.strip() for ln in script.split("\n") if ln.strip()] or [script]
    total = len(lines)

    # 这段 HTML 自包含（不引任何外部资源），由 `st.iframe` 以内联内容挂载。
    # `json.dumps(lines, ensure_ascii=False)` 负责把台词转成合法的 JS 字面量
    # （引号、反斜杠、换行都在这一步转义掉）；不开 ensure_ascii 是为了让中文原样
    # 出现在 iframe 的源码里便于对照阅读，而不是一排 \uXXXX。
    # JS 里的停止条件是 `pos < total - 2`：一屏显示 3 行，pos 推到 total-2 时
    # 屏幕上正好是最后三行 —— 停在这儿，再往下推就只剩一两行了。
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
    # height=330 与上面的 CSS 对齐：3 行 × 106px + 上下各 5px padding = 328，
    # 多出的几像素留给滚动条。改 CSS 里的行高必须同步改这里，否则会露出半行。
    # ⚠️ 从 `st.components.v1.html` 迁到 `st.iframe`：前者的 docstring 里写着
    #    「will be removed after 2026-06-01」（本机运行时也确实打这条弃用告警），
    #    而 streamlit 1.61.1 的 `st.iframe(src, ...)` 明确支持直接传 HTML 内容。
    #    唯一的能力差异：`st.iframe` 没有 `scrolling=` 参数（原值 False 即默认行为）。
    st.iframe(html, height=330)
    st.caption(f"💡 共 {total} 行，每次 3 行大字，{speed} 秒换一行 | 到底自动停止")


# --------------------------------------------------------------------------
# 页面
# --------------------------------------------------------------------------
def show_video() -> None:
    """口播视频页面（无参，供 main.py 路由调用）。

    页面上的控件
        · ``st.radio``「📌 生成模式」→ 提词器 / 数字人（决定后面走哪条链路）
        · ``st.text_area``「✍️ 输入你的文案」→ 台词原文（唯一的必填项）
        · 数字人模式下额外有：模特网格里的 ``st.button``「选择此模特」/「🗑️」/
          「✅ 确认」/「❌ 取消」、``st.file_uploader``「上传新模特」、
          ``st.checkbox``「优先使用克隆音色」、``st.selectbox``「PixVerse 内置音色」
        · ``st.button``「🎬 开始生成」→ 调 ``workflows.video.run_video()``
        · 结果区：``st.download_button``「下载台词」/「下载视频」、
          ``st.button``「🔄 刷新进度」→ ``workflows.video.refresh_avatar_task()``
        · ``st.button``「🗑️ 清除重新开始」→ pop 掉 ``_SESSION_KEYS`` 全部 7 个键

    数据流
        ``run_video()`` 返回的 ``audio_path`` / ``video_path`` / ``task_code`` /
        ``avatar_msg`` 逐个写进 ``st.session_state``（键名 ``hg_*``，见
        ``_SESSION_KEYS``），台词写 ``hg_script``、模式写 ``hg_mode``，
        最后置 ``hg_has_result=True`` 并 rerun；渲染分支只从 session_state 读。
        历史在按钮分支里追加一条。

    失败时页面显示什么
        · 台词为空 → 黄条「请输入台词」，直接 return，不调工作流；
        · 数字人模式没选到模特（记忆里的文件已不在）→ 由工作流回填中文说明，
          页面在「视频」区用黄条显示 ``hg_msg``；
        · 提交成功但还没出片 → 蓝条「任务已提交，约 2~5 分钟完成」+ 任务编号
          + 刷新按钮，刷新失败时黄条显示服务返回的原因；
        · 模特目录为空 → 灰条「还没有模特视频，请在下方上传一段」；
        · 某一个模特预览加载不出来 → 只在该格子里出一行黄字，不影响选择其它模特。
    """
    st.title("🎥 口播视频智能体")
    st.markdown("输入台词 → 提词器滚动 / 数字人出镜 → 视频下载")

    # 统一先初始化成空串：这样下面一律用 `st.session_state.get(...)` 时拿到的是确定的
    # `""` 而不是 None，两处 `if not xxx` 的语义也就一致了
    # （`hg_has_result` 同样走真值判断，它只在生成成功后置 True）
    for key in _SESSION_KEYS:
        st.session_state.setdefault(key, "")

    mode = st.radio(
        "📌 生成模式",
        ["📺 提词器模式（只看稿）", "🎭 数字人生成（对口型 → MP4）"],
        horizontal=True,
    )
    # 用选项文案里的「数字人」判断，而不是下标：文案是可读的，改文案时这行不用跟着改
    # （代价：文案里必须保留「数字人」三个字）
    is_avatar = "数字人" in mode

    raw = st.text_area(
        "✍️ 输入你的文案", height=180,
        placeholder="直接输入你要说的台词，原样使用不做改写...",
    )

    # 下面三个要传给 run_video() 的参数先给默认值：提词器模式下不会进后面的分支，
    # 但签名要求它们始终在场（默认值 = 无模特、克隆音色开关对提词器无意义）
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
        # -1 = 记忆里的模特不在列表里（文件被删 / 换了目录）：
        # 此时所有模特都显示成「🔘 选择此模特」，不会出现一个指向空气的「已选中」
        selected_idx = next(
            (i for i, a in enumerate(avatars) if a["path"] == saved), -1
        )

        if avatars:
            # 一行 4 个：再宽预览图就小到看不清脸了；超出的部分按 row_start 分片另起一行
            cols_per_row = 4
            for row_start in range(0, len(avatars), cols_per_row):
                row = avatars[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for ci, a in enumerate(row):
                    idx = row_start + ci
                    with cols[ci]:
                        try:
                            # 读成 bytes 再交给 st.video：显式给数据，
                            # 不用让 Streamlit 去猜入参是路径还是内容
                            data = Path(a["path"]).read_bytes()
                            # 模特列表已按 VIDEO_EXTS 过滤（见 _list_avatars），
                            # 这里必然都是视频 —— 原来那条 st.image 分支已不可达，删掉。
                            st.video(data)
                        except Exception:  # noqa: BLE001
                            # 单个预览读失败（文件被占用 / 刚被删）只提示这一格：
                            # 别的模特还要能选，不能让整页崩掉
                            st.warning("无法加载预览")

                        icon = "🎬"
                        size = (
                            f"{a['size_mb']:.1f}MB" if a["size_mb"] < 1000
                            else f"{a['size_mb'] / 1024:.1f}GB"
                        )
                        selected = idx == selected_idx
                        st.caption(f"{'⭐ ' if selected else ''}{icon} {a['name'][:25]}\n{size}")

                        # 3:1 的宽度比：选择是主操作，删除只是个窄按钮
                        c_sel, c_del = st.columns([3, 1])
                        with c_sel:
                            # 已选中的按钮直接 disabled + 换成次要样式：
                            # 重复点它只会白写一次状态文件、白触发一次 rerun
                            if st.button(
                                "✅ 已选中" if selected else "🔘 选择此模特",
                                key=f"sel_avatar_{idx}",
                                width="stretch",
                                type="primary" if not selected else "secondary",
                                disabled=selected,
                            ):
                                _save_selected_avatar(a["path"])
                                st.rerun()
                        with c_del:
                            # 点 🗑️ 不当场删，只置一个标记：`st.button` 的返回值只在
                            # 点击那一轮为 True，承载不了「确认框正开着」这种跨 rerun 的状态
                            if st.button("🗑️", key=f"del_avatar_{idx}", help="删除此模特"):
                                st.session_state[f"confirm_del_{idx}"] = True

                        # 二次确认：这一步会直接 os.remove 掉用户自己录的素材，
                        # 没有回收站，按错一次就没了
                        if st.session_state.get(f"confirm_del_{idx}"):
                            st.warning(f"确认删除 `{a['name']}`？")
                            cc1, cc2 = st.columns(2)
                            with cc1:
                                if st.button("✅ 确认", key=f"confirm_yes_{idx}", type="primary"):
                                    try:
                                        os.remove(a["path"])
                                        # 删掉的正好是当前选中项时，把记忆一起清掉 ——
                                        # 否则下次进来 _load_selected_avatar() 会把一个
                                        # 不存在的路径当模特传下去
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
            type=AVATAR_UPLOAD_TYPES,
            key="new_avatar_upload",
        )
        if new_file:
            # `file_uploader` 的值**跨 rerun 保留**，所以「有没有文件」不能当「要不要处理」：
            # 处理完那次 `st.rerun()` 回来它还在，于是又落盘又 rerun，页面卡在无限自转里
            # （同页 BGM 上传那处 AppTest 实测过：一次 `set_value` 后 25 秒超时）。
            # 指纹取「名字|字节数」，只处理**没见过的那一份**，后面的 rerun 自然收敛。
            # `.size` 是 `UploadedFile`（`BytesIO` 子类）在 `__init__` 里设的实例属性，
            # 不必把整个文件再 `getvalue()` 出来数一遍
            sig = f"{new_file.name}|{new_file.size}"
            if sig != st.session_state.get("_avatar_upload_sig"):
                st.session_state["_avatar_upload_sig"] = sig
                avatar_dir = Path(settings.media.get_avatar_input_dir())
                avatar_dir.mkdir(parents=True, exist_ok=True)
                # 落盘名带 md5(原始文件名)[:8]：模特目录是共享的，两台机器都传 demo.mp4
                # 会互相覆盖；也不能用内置 hash() —— 字符串 hash 带进程级随机盐，
                # 重启后同一个名字会算出另一个路径，目录里只堆积不复用
                save_path = avatar_dir / f"avatar_{hashlib.md5(new_file.name.encode('utf-8')).hexdigest()[:8]}{Path(new_file.name).suffix}"
                save_path.write_bytes(new_file.getvalue())
                # 上传即选中：省掉「上传完还得回上面点一次选择」
                _save_selected_avatar(str(save_path))
                st.success(f"✅ 已上传: {save_path.name}")
                # 立刻 rerun：让新文件出现在上方的模特网格里（网格是重新扫目录得到的）
                st.rerun()
        else:
            # 清空上传框就把指纹一起忘掉：否则「清空后再传同一个文件」会被当成
            # 已经处理过而跳过 —— 文件名与字节数都没变，指纹挡不住这种重传
            st.session_state.pop("_avatar_upload_sig", None)

        # 记忆里的路径确实还在，才拿它当模特；否则留空让 run_video() 回一句中文说明 ——
        # 工作流比这里更清楚各条链路分别需要什么
        if saved and os.path.exists(saved):
            avatar_path = saved

        st.markdown("---")
        st.markdown("#### 🔊 配音方式")
        # 默认勾选克隆音色：课案的主线就是「用你自己的声音」；
        # 取消勾选就走 PixVerse 内置音色这条更省事、但音色不属于你的路
        use_cloned_voice = st.checkbox(
            "优先使用克隆音色（CosyVoice 声音复刻）", value=True,
            help="从模特视频里克隆声音来配台词。失败会自动降级为 edge-tts 通用音色；"
                 "再失败才退回下面选的那个 PixVerse 内置音色。",
        )

        # 音色下拉**常驻**，不再只在取消勾选时才渲染：`workflows/video.py` 的降级链是
        # 「克隆 → edge-tts → PixVerse 内置 TTS」，勾着克隆时 `speaker_id` 照样会被用到
        # （前两级双双失败的那一刻）。早先只在未勾选时渲染，勾选状态下 `speaker_id` 恒为
        # "auto" —— 用户以为自己的声音在跑，实际拿到的是平台**随机分配**的嗓音，且没得选。
        # 推迟到这里才 import：提词器模式整段不进这个分支，用不上这个客户端模块；
        # 选项也直接来自它的 PIXVERSE_SPEAKERS，不在页面里手写一份副本
        from tools.avatar_client import PIXVERSE_SPEAKERS

        # 显示成「id - 中文名」方便挑，回传工作流的只取 id；
        # 分隔符必须与下面 split 里的 " - " 保持一致。默认仍是第 1 项 `auto - 随机`
        labels = [f"{k} - {v}" for k, v in PIXVERSE_SPEAKERS.items()]
        pick = st.selectbox("PixVerse 内置音色", labels, index=0)
        speaker_id = pick.split(" - ")[0]
        if use_cloned_voice:
            st.caption("上面这个只在克隆与 edge-tts 都失败时兜底；正常情况下用的是你自己的声音。")
        else:
            st.caption("内置音色由 PixVerse 直接合成语音，一步出片，但无法使用你自己的声音。")

    # ---------------- 开始生成 ----------------
    if st.button("🎬 开始生成", type="primary", width="stretch"):
        # 台词是唯一必填项：空台词时提前 return，不白跑一次 TTS
        # （走内置音色那条路还是一步计费调用）
        if not raw:
            st.warning("请输入台词")
            return

        # 工作流的 mode 只认这两个英文值。课案叫 "heygem"，本项目换了引擎才改名 ——
        # 沿用旧名会让人以为还在跑 HeyGem
        mode_key = "avatar" if is_avatar else "teleprompter"
        with st.spinner("处理中..."):
            # 延迟 import：这条链路会拉起克隆音色 / TTS / PixVerse 客户端，
            # 放在函数内，页面至少能先把输入区渲染出来
            from workflows.video import run_video

            result = run_video(
                raw, mode=mode_key, avatar_path=avatar_path,
                use_cloned_voice=use_cloned_voice, speaker_id=speaker_id,
            )

        # 结果字段逐个落 session_state：下面所有按钮（下载台词 / 下载视频 / 刷新进度）
        # 都会触发整页 rerun，只存局部变量的话点一次页面就空白
        st.session_state["hg_script"] = raw
        st.session_state["hg_audio_path"] = result.get("audio_path", "")
        st.session_state["hg_video_path"] = result.get("video_path", "")
        st.session_state["hg_task_code"] = result.get("task_code", "")
        st.session_state["hg_msg"] = result.get("avatar_msg", "")
        # 连「这次是哪种模式」一起存：渲染时按生成时的模式走，而不是按当前 radio
        st.session_state["hg_mode"] = mode_key
        st.session_state["hg_has_result"] = True
        # 历史只在这里记一次（放渲染分支的话，每次 rerun 都会重放一条）。
        # 摘要截 60 字符：首页一行放不下更长的台词，原文已经完整存在 hg_script 里
        _add_history("口播视频", raw[:60])
        # 主动 rerun 一次：让下面的渲染分支在干净的一轮里只按 session_state 重绘
        st.rerun()

    # ---------------- 渲染结果 ----------------
    # 没生成过：页面就停在输入区（提词器 / 模特网格 / 配音方式 / 开始生成）
    if not st.session_state.get("hg_has_result"):
        return

    # 用**生成时**的模式渲染，不用当前 radio 的值 ——
    # 否则生成完再切一次模式，页面会拿新模式的模板去渲染旧结果
    mode_key = st.session_state.get("hg_mode", "teleprompter")
    script = st.session_state.get("hg_script", "")

    if mode_key == "teleprompter":
        _render_teleprompter(script)
    else:
        st.markdown("### 📝 台词（原样使用，未改写）")
        # 固定 key 且不回写任何东西：这个框只用于查看/复制，
        # 在里面改字不会影响已经生成的结果（要改台词得回上面重新生成）
        st.text_area("台词内容", script, height=150, key="script_display")
        # data 用本轮从 session_state 恢复出来的 script，不用按钮那次的局部变量
        st.download_button("📥 下载台词", script, file_name="台词.txt")

        st.markdown("### 🎥 视频")
        video_path = st.session_state.get("hg_video_path", "")
        audio_path = st.session_state.get("hg_audio_path", "")
        task_code = st.session_state.get("hg_task_code", "")
        msg = st.session_state.get("hg_msg", "")

        # 先判存在再渲染：降级路径上工作流可能回填空串或已被清理的路径，
        # st.audio 拿到不存在的文件会直接报错
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path)

        # 数字人模式只有三种终态：① 有成品 → 预览 + 下载；② 只有任务号 → 提示 + 刷新按钮；
        # ③ 连任务号都没有但有 msg → 黄条显示失败原因
        if video_path and os.path.exists(video_path):
            st.success("✅ 数字人视频已生成")
            st.video(video_path)
            # download_button 的 data 不接受路径，只能把文件读成 bytes 传进去
            with open(video_path, "rb") as fh:
                st.download_button(
                    "⬇️ 下载视频", fh.read(),
                    file_name=os.path.basename(video_path), mime="video/mp4",
                )
        elif task_code:
            # ⚠️ `task_code` 存在**不等于**任务还活着：`query_task()` 对
            # FAILED / CANCELED / UNKNOWN 会把 `message` 写成「任务失败/不可查（…）」，
            # 而页面把 `hg_task_code` 一直留着（那是为了「下载失败时还能重试」）。
            # 不区分的话，一个**已经彻底失败**的任务会永远显示成蓝色的「处理中」+
            # 一个点了也没用的刷新按钮 —— 用户以为在跑，其实早就挂了。
            # 判据用 message 的失败前缀：这三串正是 `tools/avatar_client.py` 的
            # `query_task()` 在终态失败时回填的（「查询异常:」来自它自己的 except 分支）。
            _failed = msg.startswith(("任务失败/不可查", "查询异常"))
            if _failed:
                st.error(msg)
                st.caption(f"任务编号: `{task_code}`")
                st.caption("这个任务已经结束（失败或查不到）。可以清除后重新生成。")
            else:
                st.info(msg or "任务已提交，约 2~5 分钟完成")
                st.caption(f"任务编号: `{task_code}`")
                # 手动刷新而不是自动轮询：PixVerse 出片要 2~5 分钟，
                # 自动轮询会把页面一直卡在阻塞的 HTTP 请求上
                if st.button("🔄 刷新进度（生成完毕后点击查看）"):
                    from workflows.video import refresh_avatar_task

                    with st.spinner("查询中..."):
                        q = refresh_avatar_task(task_code)
                    # 只有成功才写 video_path；失败保留空串，
                    # 页面下一轮会走上面的 `_failed` 分支出红条
                    if q["success"]:
                        st.session_state["hg_video_path"] = q["video_path"]
                        st.session_state["hg_msg"] = q["message"]
                    else:
                        st.session_state["hg_msg"] = q["message"]
                    st.rerun()
        elif msg:
            st.warning(msg)

    # 逐个 pop 而不是 clear()：只清本页这 7 个键，
    # 别把 `history` 之类其它页面也在用的状态一起清掉
    if st.button("🗑️ 清除重新开始"):
        for key in _SESSION_KEYS:
            st.session_state.pop(key, None)
        st.rerun()
