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
       页面只做截断展示（剪到与上游同一口径的 2000 字符 / 前 80 条），方便出错时排查。

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
      所以本页把 ``from workflows.mashup import run_mashup`` 放在按钮分支里，
      等真正要剪辑时才加载。
    · **BGM 以输入框为准，不回落到旧值**：``current_bgm`` 是本轮开头从 ``bgm.json``
      读到的旧值，早先 ``bgm_path = bgm_input or current_bgm`` 会在用户清空输入框时
      回落到它 —— 于是那一轮 ``bgm.json`` 已经写空，页面却还显示「✅ BGM: xxx.wav」
      并拿旧路径去剪辑。控件带 ``key``，``bgm_input`` 本身就是框里此刻的值，直接用即可。
    · **上传 BGM 后不能只 rerun**：带 ``key`` 的 ``st.text_input`` 在 rerun 之间由
      ``st.session_state`` 说了算，``value=`` 只在 key 首次出现时生效（1.61.1 实测），
      所以：① 首值改用 ``st.session_state.setdefault("bgm_path_input", current_bgm)``，
      控件不再传 ``value=``；② 上传那轮寄存 ``_bgm_pending``，下一轮构建控件**之前**
      回填 —— 控件实例化之后再改它的 session_state 会抛 ``StreamlitAPIException``。
      不这么做的话，上传完框里仍是旧路径，下一轮还会把旧路径写回 ``bgm.json``。
      ③ 上传那轮还要靠「名字|字节数」指纹把处理动作**收敛成一次**：``file_uploader``
      的值跨 rerun 保留，不设指纹时它每轮都为真、``st.rerun()`` 每轮都再来一次，
      ② 那条回填永远等不到「下一轮」—— 页面直接停在无限 rerun 里。
    · **同一个指纹约定用在三个上传框上**（BGM / 主口播视频 / 穿插素材）。
      后两个没有 ``st.rerun()``，所以不会自转，但同样每轮 rerun 都为真：
      主视频那处会白写一次几十 MB 的文件；素材那处更重 —— 先把整个
      ``materials/`` 清空再整批重写，用户在**别的框里**敲一个字符就触发一次，
      而且清空与重写之间有个窗口，并发读方可能看到半截目录。
      （上传框被清空时要把指纹 ``pop`` 掉，否则「清空后重传同一个文件」会被跳过。）

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

# 剪辑日志的字符上限：与 ``workflows/mashup.py`` 正常收尾那处 ``last_msg[:2000]`` 对齐
# （上游写的是内联字面量、没有可 import 的常量，所以只能对齐数值）。
# 页面这层只兜住上游**异常回填**那几条路径（源视频不存在 / 初始化失败 / 剪辑出错），
# 它们回的是很短的失败串 —— 正常收尾的日志在到达这里之前就已经被上游截过一次了。
_EDITOR_LOG_LIMIT = 2000


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
            # 空串按「没记录」处理：这样本函数仍会把 .env 里的默认 BGM 交出来。
            # ⚠️ 但页面层的输入框带 key，用户清空之后它一直是空串，所以那个默认值
            #    只对**新会话的首次渲染**生效 —— 本次会话里清空就是「不混音」。
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
        · ``st.button``「🎬 开始剪辑」→ 调 ``workflows.mashup.run_mashup()``
          （与其余 5 个页面一致，图 state 由工作流组装，页面不自己 ``invoke``）
        · ``st.button``「♻️ 重置剪辑 Agent」→ ``workflows.mashup.reset_editor_agent()``
          （改完 ``SKILL.md`` / 根 ``.env`` 后让缓存的 agent 失效；不联网，随时可点）
        · ``st.download_button``「⬇️ 下载视频」

    数据流
        三份输入压成一个 JSON 字符串交给 ``run_mashup()``（``edit_requirements``）；返回的
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
        # 同一个「名字|字节数」指纹约定（见下面 BGM 那处的注释）：这里没有 `st.rerun()`，
        # 所以不会自转，但 `file_uploader` 的值同样跨 rerun 保留 —— 不设指纹的话，
        # 用户在**别的任何输入框**里每敲一个字符都会白写一次几十 MB 的主视频。
        sig = f"{video_file.name}|{video_file.size}"
        if sig != st.session_state.get("_mashup_video_sig"):
            st.session_state["_mashup_video_sig"] = sig
            saved.write_bytes(video_file.getvalue())
        # 落在守卫外：跳过写入的那几轮也要把路径交出去（文件第一轮就写好了）
        video_path = str(saved)
        # 预览直接传 UploadedFile：它本身就是 file-like，不必再从刚落盘的路径读回来
        st.video(video_file)
        st.success(f"已加载：{video_file.name}")
    else:
        # 清空上传框就把指纹一起忘掉，否则「清空后再传同一个文件」会被当成已处理而跳过
        st.session_state.pop("_mashup_video_sig", None)

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
        # 同一根因、更重的后果：这里没有 `st.rerun()`（不会自转），但每轮 rerun 都会
        # **先把整个素材目录清空、再整批重写** —— 用户在别的框里敲一个字符就触发一次，
        # 而且清空与重写之间有个窗口，并发读方（或上一次渲染）可能看到半截目录。
        # 指纹取全部文件的「名字|字节数」拼起来：只有**换了素材**才真的清+写。
        sig = "|".join(f"{mf.name}|{mf.size}" for mf in material_files)
        if sig != st.session_state.get("_mashup_material_sig"):
            st.session_state["_mashup_material_sig"] = sig
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
        # 跳过写入那几轮也要报出「本次用的是哪些素材」：文件第一轮就落好了，
        # 按名字取回同一批路径即可（下游据此拼 `edit_requirements`）。
        # 放在守卫外还有个好处：`st.success` 的条数不会因为跳过而变成 0。
        material_paths = [str(MATERIALS_DIR / mf.name) for mf in material_files]
        st.success(f"已加载 {len(material_paths)} 个素材")
    else:
        st.session_state.pop("_mashup_material_sig", None)

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

    # 上传要把新路径**回填进输入框**：控件带 key 时 `value=` 只在 key 首次出现时生效，
    # 后面几轮一律由 `session_state` 说了算（streamlit 1.61.1 实测），所以「首值」改用
    # 官方那套 `setdefault` 写法，而上传那一轮只寄存一个待填值、在这里（构建控件**之前**）
    # 兑现 —— 控件实例化之后再改它的 session_state 会抛 StreamlitAPIException。
    # 顺带避开「同时传 value= 又写 session_state」时 Streamlit 打的那条告警。
    st.session_state.setdefault("bgm_path_input", current_bgm)
    pending_bgm = st.session_state.pop("_bgm_pending", "")
    if pending_bgm:
        st.session_state["bgm_path_input"] = pending_bgm

    # 3:1 的宽度比：左边是要读/要改的长路径，右边只是个上传按钮
    col1, col2 = st.columns([3, 1])
    with col1:
        bgm_input = st.text_input(
            "背景音乐路径",
            placeholder="输入 WAV/MP3 文件路径（留空则不混音）",
            help="没配置就跳过混音，不影响其它剪辑步骤",
            key="bgm_path_input",
        )
    with col2:
        bgm_file = st.file_uploader(
            "📁 上传", type=["wav", "mp3", "m4a", "ogg"], help="上传新 BGM 会替换当前设置",
        )
        if bgm_file:
            # `file_uploader` 的值**跨 rerun 保留**，所以「有没有文件」不能当「要不要处理」：
            # 处理完那次 `st.rerun()` 回来它还在，于是又写盘又 rerun，页面卡在无限自转里
            # （AppTest 实测：一次 `set_value` 后 25 秒超时）。指纹取「名字|字节数」，
            # 只处理**没见过的那一份**，这样后续 rerun 自然收敛。
            # `.size` 是 `UploadedFile`（`BytesIO` 子类）在 `__init__` 里设的实例属性，
            # 不必把整个文件再 `getvalue()` 出来数一遍
            sig = f"{bgm_file.name}|{bgm_file.size}"
            if sig != st.session_state.get("_bgm_upload_sig"):
                st.session_state["_bgm_upload_sig"] = sig
                target = CACHE_DIR / bgm_file.name
                target.write_bytes(bgm_file.getvalue())
                _save_bgm_path(str(target))
                # 寄存待回填值，让下一轮构建输入框时把框里的旧路径换成这个新路径
                # （只 rerun 是不够的：带 key 的控件在 rerun 之间由 session_state 说了算）
                st.session_state["_bgm_pending"] = str(target)
                st.success("已更新")
                st.rerun()
        else:
            # 清空上传框就把指纹一起忘掉：否则「清空后再传同一个文件」会被当成
            # 已经处理过而跳过 —— 文件名与字节数都没变，指纹挡不住这种重传
            st.session_state.pop("_bgm_upload_sig", None)

    # 没有 on_change 回调时，「输入框的值变了」只能在每次 rerun 时对比着判断；
    # current_bgm 是**本轮开头**从 bgm.json 读到的旧值，这里只拿它当「要不要落盘」的比对基准
    if bgm_input != current_bgm:
        _save_bgm_path(bgm_input)
    # 直接取输入框的值，**不再**回落到 current_bgm：控件带 key，`bgm_input` 永远是框里
    # 此刻显示的值，而 current_bgm 只是本轮开头读到的旧值 —— 回落会让「用户清空输入框」
    # 那一轮的显示与落盘错位一轮（bgm.json 当轮已写空，页面却还说「✅ BGM: xxx.wav」、
    # 并用旧路径去剪辑）。首轮那个 `setdefault` 保证框里就是 current_bgm，两者必然相等，
    # 所以不存在「刚进页面 BGM 就丢了」。
    bgm_path = bgm_input

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
            from workflows.mashup import run_mashup

            # 走 run_mashup() 而不是自己 mashup_graph.invoke(...)：另外 5 个模块的页面
            # 都走 run_xxx()，图 state 的组装归工作流管；页面少一处会漂移的契约副本
            # （原来这里还手填了一个课案遗留的「剪辑风格」字段，全仓 0 处读取，已随字段删掉）
            result = run_mashup(video_path, json.dumps({
                "materials": use_materials,
                "bgm_path": bgm_path,
                "extra_requirements": requirements,
            }, ensure_ascii=False))

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

    # ---------------- 维护：重置剪辑 Agent ----------------
    # 位置选在主按钮之后、结果区之前：结果区在没有结果时会提前 return，而这个按钮最该
    # 被按到的时刻恰恰是「刚改完 .env / SKILL.md，还没跑出过成片」—— 放结果区会被吞掉。
    # 视觉上只是一行说明 + 一个窄按钮，不抢「🎬 开始剪辑」的主位。
    col_tip, col_reset = st.columns([4, 1])
    with col_tip:
        st.caption(
            "改了 `.skills/video-use/SKILL.md` 或根目录 `.env`（密钥 / 模型名）之后点一下 → "
            "工作流会丢掉已初始化的剪辑 agent，下次剪辑按新内容重新初始化。"
        )
    with col_reset:
        if st.button("♻️ 重置剪辑 Agent", width="stretch"):
            # 同样必须在按钮分支里 import：workflows.mashup 导入时会 monkey-patch
            # subprocess.Popen / run（见模块 docstring 的「踩过的坑」）。
            # 这个函数只把模块级缓存置空，不联网、也不要求 agent 已经建过 —— 随时可点
            from workflows.mashup import reset_editor_agent

            reset_editor_agent()
            # 不 rerun：rerun 会把刚给出的提示一起冲掉，而用户要的正是「确认点成功了」
            st.success("已重置 —— 下次「开始剪辑」会用新的 SKILL.md / .env 重新初始化 agent。")

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
            # 上限与上游同口径（见 `_EDITOR_LOG_LIMIT`）：上游正常收尾时已把 editor_log
            # 截过一次，页面这层只兜住它异常回填那几条没截的路径 —— 早先写 3000 永远碰不到
            st.text(editor_log[:_EDITOR_LOG_LIMIT])

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
