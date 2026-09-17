# -*- coding: utf-8 -*-
"""视频剪辑页面 —— DeepAgents + video-use 技能 + HyperFrames

课案出处：自媒体课案 → 视频剪辑 → views/mashup.py

本节要讲什么
    1. **本页只收集输入，剪辑全在 `workflows/mashup.py` 里**（DeepAgents +
       `video-use` 技能 + moviepy，HyperFrames 可选）。页面把「素材列表 / BGM 路径 /
       附加要求」压成一个 JSON 字符串，塞进图 state 的 ``edit_requirements``。
    2. **素材是「本次上传优先、目录兜底」**：上传了新素材就先清空
       ``.cache/mashup/materials`` 再落盘，保证这次只跑这一批；没上传则回落到
       目录里已有的素材 —— 否则为了重跑一次还得把素材再传一遍。
    3. **BGM 路径是有记忆的**：选过的记在 ``.cache/mashup/bgm.json``，下次进页面
       自动带出（没记录过才用 ``MEDIA_BGM_PATH``）。文件不存在时页面**提前**说
       「本次将不混音」，而不是等剪辑跑完才发现没音乐。
    4. **日志与工具调用轨迹分两个折叠面板**：agent 的决定过程全在日志里，
       页面只做截断展示（前 3000 字符 / 前 80 条），方便出错时排查。

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

踩过的坑
    · **页面不能提前 import `workflows.mashup`**：那个模块在导入时就 monkey-patch 了
      ``subprocess.Popen`` / ``subprocess.run``（把 agent 起子进程的工作目录圈进沙箱）。
      所以本页把 ``from workflows.mashup import mashup_graph`` 放在按钮分支里，
      等真正要剪辑时才加载。
    · **清空 BGM 输入框要下一轮 rerun 才生效**：``current_bgm`` 是本轮开头从
      ``bgm.json`` 读到的旧值，而 ``bgm_path = bgm_input or current_bgm`` 会先回落到它；
      真正生效要等 ``bgm.json`` 被写空之后的下一次 rerun。
    · **上传 BGM 后必须 rerun**：输入框的 ``value`` 只在构建控件时生效，
      不 rerun 的话框里还显示旧路径，和实际用的文件对不上。

运行方式（由 main.py 侧边栏路由调用）::

    Set-Location F:\\ProGram\\Python_Base\\Media_Agent
    & ..\\.venv\\Scripts\\python.exe -m streamlit run main.py
"""

import hashlib
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
    """读上次用的 BGM 路径；没记录过就回落到配置里的 MEDIA_BGM_PATH。

    Returns:
        路径字符串；``bgm.json`` 不存在 / JSON 坏了 / 里存的是空串时，
        一律退化成 ``settings.media.bgm_path``（没配就是 ``""``，
        页面显示「未设置 BGM，本次不混音」）。
    """
    try:
        if BGM_CACHE.is_file():
            saved = json.loads(BGM_CACHE.read_text(encoding="utf-8")).get("bgm_path", "")
            # 空串按「没记录」处理：这样用户在页面上清空 BGM 之后，
            # 下一轮还能回落到 .env 里配的那个默认 BGM，而不是卡在空值
            if saved:
                return saved
    # 缓存文件坏了 / 读不了不该影响页面：直接当作「没记录过」，回落到 .env 里的配置
    except Exception:  # noqa: BLE001
        pass
    return settings.media.bgm_path or ""


def _save_bgm_path(path: str) -> None:
    """把当前 BGM 路径记到 ``.cache/mashup/bgm.json``（下次进页面自动带出）。

    写失败只在控制台打一行日志 —— 记不住上次的选择是小事，
    不能因为它让页面报错。
    """
    try:
        # 目录可能还不存在（首次运行、.cache 被清过），写之前先补上
        BGM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        BGM_CACHE.write_text(json.dumps({"bgm_path": path}, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[剪辑] BGM 路径保存失败（忽略）: {exc}")


def show_mashup() -> None:
    """视频剪辑页面（无参，供 main.py 路由调用）。

    页面上的控件
        · ``st.file_uploader``「上传口播视频」→ 落盘到 ``CACHE_DIR``（单个文件）
        · ``st.text_input``「或输入视频路径」→ 手填路径，非空且文件存在时会**覆盖**
          上传得到的路径
        · ``st.file_uploader``「上传素材（可多选）」→ 落盘到 ``MATERIALS_DIR``
        · ``st.text_input``「背景音乐路径」+ ``st.file_uploader``「📁 上传」→ BGM
        · ``st.text_area``「补充要求（可选）」→ 进 ``extra_requirements``
        · ``st.button``「🎬 开始剪辑」→ 调 ``workflows.mashup.mashup_graph.invoke()``
        · ``st.download_button``「⬇️ 下载视频」

    数据流
        三份输入压成一个 JSON 字符串交给图（``edit_requirements``）；返回的
        ``output_video`` / ``editor_log`` / ``steps`` 三个字段存进
        ``st.session_state["mashup_result"]``（只挑渲染要用的字段，不整份 state），
        输入视频路径存 ``["mashup_input"]``，并往 ``st.session_state["history"]``
        追加一条。日志与工具调用轨迹渲染成两个折叠面板，成品用 ``st.video``
        预览 + 下载。

    失败时页面显示什么
        · 没配 ``API_KEY`` → 红条「未配置 API_KEY，无法启动剪辑 agent」，整页早退；
        · 视频路径为空 / 文件不存在 → 黄条「请先上传口播视频或输入视频路径」、
          红条「视频文件不存在: …」，两条都不调 agent（省得白等几分钟）；
        · agent 跑完没找到成片 → 黄条「没有找到成片」，并给出沙箱目录
          ``CACHE_DIR`` 让用户自己去看中间产物（原因通常在剪辑日志里）。
    """
    st.title("🎬 视频剪辑智能体")
    st.markdown(
        "上传口播视频 + 素材，AI 自动完成：生成动画素材 → 上下排布穿插 → "
        "字幕 → 背景音乐 → 渲染\n\n"
        "**技术栈**：`deepagents` + `video-use` 技能 + `moviepy`（+ `HyperFrames` 可选）"
    )

    # 提前建目录：下面读 MATERIALS_DIR.iterdir()、写上传文件都假设它们已存在
    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not settings.api_key:
        # 用根 .env 的 API_KEY（不是百炼那个 DASHSCOPE_API_KEY）判断：
        # 剪辑走 DeepAgents，没配模型网关时挡在入口 —— 否则用户传完素材、
        # 点了按钮才失败，白等几分钟
        st.error("未配置 API_KEY，无法启动剪辑 agent（见根目录 .env）")
        return

    # ---------------- 输入视频 ----------------
    st.markdown("### 📁 输入口播视频")
    # 这里的类型白名单与 MEDIA_EXTS 分开写：主口播视频只收视频（图片做不了剪辑源），
    # 下面的素材框才允许图片
    video_file = st.file_uploader(
        "上传口播视频", type=["mp4", "mov", "avi", "mkv", "webm"],
        help="主口播视频，AI 将在其中穿插素材",
    )

    video_path = ""
    if video_file:
        suffix = Path(video_file.name).suffix
        # 与 workflows 共用一个沙箱目录（课案两边不一致，导致兜底查找扫不到）；
        # 文件名用 md5(原始名)[:8] 而不是 abs(hash(...)) —— 内置 hash 带进程级随机盐
        saved = CACHE_DIR / f"input_{hashlib.md5(video_file.name.encode('utf-8')).hexdigest()[:8]}{suffix}"
        saved.write_bytes(video_file.getvalue())
        video_path = str(saved)
        # 预览直接传 UploadedFile：它本身就是 file-like，不必再从刚落盘的路径读回来
        st.video(video_file)
        st.success(f"已加载：{video_file.name}")

    path_input = st.text_input(
        "或输入视频路径", placeholder=r"例: F:\ProGram\Python_Base\Media_Agent\.cache\videos\xxx.mp4",
    )
    # 手填路径写在后面，非空且文件存在时会覆盖上面上传得到的路径；
    # 不存在时**不报错** —— 用户可能刚敲了一半，每敲一个字符都会 rerun
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
        # 清掉旧素材，避免和这一次混在一起（课案原有行为）：
        # 「本次上传的才算数」，否则目录里会越堆越多、agent 拿到一堆没打算用的图
        for old in MATERIALS_DIR.iterdir():
            try:
                if old.is_file():
                    old.unlink()
            except Exception:  # noqa: BLE001
                # 文件可能正被上一次的渲染或别的进程占用：清不掉就算了，
                # 不能让一次 unlink 失败中断整批上传
                pass
        for mf in material_files:
            # 用原始文件名而不是哈希名：DeepAgents 在提示词里看到的就是文件名，
            # 可读的名字能帮它判断素材内容，`input_a1b2c3.mp4` 那种没有信息量
            mp = MATERIALS_DIR / mf.name
            mp.write_bytes(mf.getvalue())
            material_paths.append(str(mp))
        st.success(f"已加载 {len(material_paths)} 个素材")

    # 没上传素材时，这次剪辑用的就是目录里已有的（见下面的 use_materials 兜底），
    # 所以这里要按 mtime 倒序列出最近的一批，让用户确认自己将跑到哪些素材
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

    # 3:1 的宽度比：左边是要读/要改的长路径，右边只是个上传按钮
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
            # 必须 rerun：输入框的 value 只在构建控件时生效，
            # 不重跑一次框里还是旧路径，和实际要用的文件对不上
            st.rerun()

    # 没有 on_change 回调时，「输入框的值变了」只能在每次 rerun 时对比着判断；
    # 注意 current_bgm 是**本轮开头**读到的旧值，所以清空输入框要下一轮才真正生效
    if bgm_input != current_bgm:
        _save_bgm_path(bgm_input)
    bgm_path = bgm_input or current_bgm

    # 三种终态都要提前说清楚：文件在 → 确认用的是哪个；
    # 路径填了但文件没了 → 提前告知「本次将不混音」，而不是等剪辑跑完才发现没有音乐
    if bgm_path and os.path.exists(bgm_path):
        st.caption(f"✅ BGM: `{Path(bgm_path).name}`")
    elif bgm_path:
        st.warning(f"⚠️ BGM 文件不存在，本次将不混音: `{bgm_path}`")
    else:
        st.caption("未设置 BGM，本次不混音")

    # ---------------- 附加要求 ----------------
    st.markdown("### ✏️ 额外剪辑要求")
    # 这段自然语言会以 extra_requirements 塞进 edit_requirements JSON，
    # 由 DeepAgents 自己理解 —— 所以不需要任何结构化格式
    requirements = st.text_area(
        "补充要求（可选）",
        placeholder="例：特定时间段剪掉、加特定文字、调色风格...",
        height=60,
    )

    # ---------------- 开始剪辑 ----------------
    if st.button("🎬 开始剪辑", type="primary", width="stretch"):
        if not video_path:
            st.warning("请先上传口播视频或输入视频路径")
            return
        # 文件可能在填完路径之后被挪走/删掉：这里再确认一次，
        # 别把不存在的路径交给 agent，让它跑到一半才报「源视频不存在」
        if not os.path.exists(video_path):
            st.error(f"视频文件不存在: {video_path}")
            return

        # 本次上传优先；一个都没传才回落到目录里已有的素材 ——
        # 否则「只想重跑一次剪辑」也得把素材再传一遍
        use_materials = material_paths or [
            str(MATERIALS_DIR / f)
            for f in existing
            if (MATERIALS_DIR / f).is_file()
        ]

        with st.spinner("🤖 deepagent 正在分析视频并剪辑（这一步可能要几分钟）..."):
            # 必须在按钮分支里 import：这个模块导入时会 monkey-patch
            # subprocess.Popen / run（见模块 docstring 的「踩过的坑」），
            # 页面级提前导入会影响到页面上其它功能的子进程行为
            from workflows.mashup import mashup_graph

            # 直接 invoke 图而不是走 run_mashup()：两者等价（后者只是薄封装），
            # 这里保持课案的写法
            result = mashup_graph.invoke({
                "input_video": video_path,
                # 课案的这个字段没有对应控件，恒为空串；保留键是为了图 state 契约完整
                "edit_style": "",
                # state 里只能放字符串，所以三份输入压成 JSON 文本，由工作流侧解析
                "edit_requirements": json.dumps({
                    "materials": use_materials,
                    "bgm_path": bgm_path,
                    "extra_requirements": requirements,
                }, ensure_ascii=False),
            })

        # 只挑渲染要用的三个字段存进 session_state（不存整份 state，
        # 里面还有输入路径等大字段，没必要一直占着内存）；
        # 下载按钮触发的整页 rerun 会重新读这里，不缓存的话页面当场空白
        st.session_state["mashup_result"] = {
            "output_video": result.get("output_video", ""),
            "editor_log": result.get("editor_log", ""),
            "steps": result.get("steps", ""),
        }
        st.session_state["mashup_input"] = video_path
        # 本页是唯一没有 `_add_history()` 帮助函数、就地 append 的页面；
        # 字段名与其它页保持一致，首页的展示逻辑不用为它开分支。
        # 摘要直接用文件名（本来就短，不需要像其它页那样截断）
        st.session_state.setdefault("history", []).append({
            "time": datetime.now().strftime("%H:%M"),
            "action": "视频剪辑",
            "summary": Path(video_path).name,
        })

    # ---------------- 渲染结果 ----------------
    res = st.session_state.get("mashup_result")
    if not res:
        # 还没剪辑过：页面停在输入区（上传 / 素材 / BGM / 附加要求）
        return

    editor_log = res.get("editor_log", "")
    output_video = res.get("output_video", "")

    if editor_log:
        # 展开条件反着来：有成品时收起日志（用户先看视频），
        # 没成品时默认展开 —— 失败原因就在日志里
        with st.expander("📝 剪辑日志", expanded=not output_video):
            # 页面侧再截一道（前 3000 字符）：工作流正常收尾时已把 editor_log 截到 2000，
            # 但异常回填那条路径没截；这个上限保证任何情况下都不会把几万字塞进 DOM
            st.text(editor_log[:3000])

    with st.expander("🔧 工具调用轨迹", expanded=False):
        try:
            # steps 是工作流回填的 JSON 字符串；解析失败就退化成空列表，
            # 页面只显示「（本次没有记录到工具调用）」，不抛异常
            steps = json.loads(res.get("steps") or "[]")
        except Exception:  # noqa: BLE001
            steps = []
        if steps:
            # 上限 80 条、每条 300 字符：一次剪辑能产生上百条工具调用，
            # 全量渲染会把页面拖慢，也会把有用的信息冲出屏幕
            for s in steps[:80]:
                st.text(str(s)[:300])
        else:
            st.caption("（本次没有记录到工具调用）")

    # 存在性检查：工作流有时会把中间产物的路径回填进来，
    # 文件不在时不能直接丢给 st.video（它会报错）
    if output_video and os.path.exists(output_video):
        st.success("✅ 剪辑完成！")
        st.video(output_video)
        # download_button 的 data 不接受路径，只能把文件读成 bytes 传进去
        with open(output_video, "rb") as fh:
            st.download_button(
                "⬇️ 下载视频", fh.read(),
                file_name=Path(output_video).name, mime="video/mp4",
            )
        st.caption(f"输出文件：`{output_video}`")
    else:
        # 失败时的引导：把沙箱目录写出来，用户能自己去找中间产物 / 复现问题
        st.warning(
            f"没有找到成片。请检查上面的剪辑日志；\n"
            f"也可以手动去看沙箱目录：`{CACHE_DIR}`"
        )
