# -*- coding: utf-8 -*-
"""视频剪辑工作流 —— DeepAgents + video-use 技能

课案出处：自媒体课案 → 视频剪辑 → workflows/mashup.py

课案的核心理念（保留）
    **不手写剪辑步骤**，而是把剪辑知识编码成 ``.skills/video-use/SKILL.md`` 操作手册，
    让 deepagent 自己按流程调用 moviepy / HyperFrames 完成：
    分析素材 → 转写生成字幕 → 生成动画叠加素材 → 上下排布穿插 → BGM 混音 → 渲染 MP4。

与课案的差异
    | 课案 | 本项目 |
    |---|---|
    | 转写：agent 自己写脚本调本地 funasr | 走项目自带的 ``tools.audio_transcriber``（百炼 Fun-ASR-Flash 云端） |
    | 默认 BGM：硬编码 ``C:\\Users\\13261\\Pictures\\风格\\...wav`` | ``MEDIA_BGM_PATH``；留空则不混音 |
    | 沙箱目录：``.cache/videos`` | ``MEDIA_MASHUP_WORK_DIR``（``.cache/mashup``）——课案两个文件的 CACHE_DIR 不一致，兜底查找扫不到，已统一 |
    | 无步数上限 | 加 ``recursion_limit=50`` + 兜住 ``GraphRecursionError`` |

课案原有的 Windows 适配**全部保留**（这些是真踩出来的，不要删）
    1. ``sys.stdout/stderr.reconfigure(encoding='utf-8')`` —— 否则日志中文变乱码
    2. ``subprocess.Popen`` monkey-patch：强制 UTF-8 + ``CREATE_NO_WINDOW`` +
       1MB ``bufsize``（ffmpeg 的 stderr 输出很猛，管道小了会把进程堵死）
    3. ``normalize_path()`` —— 修 Git Bash 与 Windows 混用产生的畸形路径
       （``C:\\c\\Users\\...`` / ``/c/Users/...`` / ``C:\\d\\Data\\...``）
    4. ``MSYS_NO_PATHCONV=1`` / ``MSYS2_ARG_CONV_EXCL=*`` 禁 Git Bash 路径自动转换
    5. ``LocalShellBackend._resolve_path`` monkey-patch —— virtual_mode 会把
       ``/c/Users/...`` 当虚拟路径拼到沙箱下面，必须在解析前先清洗
    6. 给子进程注入 ``CHROMIUM_PATH``（HyperFrames 渲染要浏览器）

本仓库实测补的两条（**课案没有，不加会翻车**）
    7. **把当前 venv 的 Scripts 目录顶到 PATH 最前面**：
       本机 PATH 里的 ``python`` 是 Windows Store 占位符，执行后**静默无输出**。
       不让 ``python`` 指向 ``.venv\\Scripts\\python.exe``，
       agent 跑 ``python script.py`` 就会「成功但什么也没发生」。
    8. **system_prompt 里必须说明 execute 跑的是 cmd.exe 不是 bash**：
       不写，模型会反复敲 ``ls`` / ``pwd`` / ``cat``，实测一路撞到
       ``GraphRecursionError``（Recursion limit of 50 reached）。
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# 路径引导：必须在 import 项目内模块（workflows/*）之前执行。
# 直接 `python workflows/mashup.py` 时 sys.path[0] 是 workflows/ 目录，
# 不加这一句 `from workflows import get_model` 会 ModuleNotFoundError。
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Windows 控制台强制 UTF-8（防止日志中文变乱码）—— 课案原有
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    except Exception:  # noqa: BLE001
        pass

# ---------------------------------------------------------------------------
# subprocess monkey-patch：UTF-8 + Windows 防卡 —— 课案原有
# ---------------------------------------------------------------------------
_orig_popen_init = subprocess.Popen.__init__


def _patched_popen_init(self, *args, **kwargs):
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "replace")
    if sys.platform == "win32":
        # CREATE_NO_WINDOW 防止弹控制台窗口把父进程阻塞
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        # 1MB 缓冲区：ffmpeg 的 stderr 很能刷，管道小了会死锁
        kwargs.setdefault("bufsize", 102400 * 1024)
    _orig_popen_init(self, *args, **kwargs)


subprocess.Popen.__init__ = _patched_popen_init  # type: ignore[method-assign]

from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from config import settings  # noqa: E402
from workflows import get_model  # noqa: E402


class MashupState(TypedDict, total=False):
    """视频剪辑工作流状态。"""

    input_video: str
    edit_style: str
    edit_requirements: str
    output_video: str
    editor_log: str
    steps: str
    edit_start: float


# ==========================================================================
# 路径清洗 —— 修 Git Bash / Windows 路径混用产生的畸形路径（课案原有）
# ==========================================================================
def normalize_path(p: str) -> str:
    """修复 Git Bash / Windows 路径混用导致的各种畸形路径。

    常见畸形及修复::

        C:\\c\\Users\\...  →  C:\\Users\\...
        C:/c/Users/...    →  C:\\Users\\...
        /c/Users/...      →  C:\\Users\\...
        /d/project/...    →  D:\\project\\...
        C:\\d\\Data\\...  →  D:\\Data\\...   （盘符字母不同时以第二个为准）
    """
    if not p or not isinstance(p, str):
        return p

    original = p

    # 1) 盘符叠加：C:\c\Users → C:\Users；C:\d\Data → D:\Data
    m = re.match(r"^([A-Za-z]):[\\/]([a-z])([\\/].+)$", p)
    if m:
        drive1, drive2, rest = m.group(1).lower(), m.group(2).lower(), m.group(3)
        p = f"{m.group(1)}:{rest}" if drive1 == drive2 else f"{m.group(2).upper()}:{rest}"

    # 2) 纯 Git Bash 路径：/c/Users/... → C:\Users\...
    if sys.platform == "win32":
        m2 = re.match(r"^/([a-zA-Z])/(.+)$", p)
        if m2:
            p = f"{m2.group(1).upper()}:\\{m2.group(2)}"

    # 3) 统一斜杠方向
    if sys.platform == "win32" and "\\" not in p and "/" in p:
        p = p.replace("/", "\\")
        # /videos、/videos/output.mp4 这种前导斜杠会被 Windows 误判成 C:\videos\...
        # 原始输入以 / 开头但不是 /c/ 格式 → 去掉前导反斜杠，恢复相对路径
        if original.startswith("/") and not re.match(r"^/[a-zA-Z]/", original):
            p = p.lstrip("\\")

    try:
        p = os.path.normpath(p)
    except Exception:  # noqa: BLE001
        pass

    if p != original:
        print(f"[PathFix] 路径已修复: {original!r} → {p!r}")
    return p


def safe_abspath(p: str) -> str:
    """先清洗路径再 abspath，防畸形路径。"""
    return os.path.abspath(normalize_path(p))


# ==========================================================================
# deepagent 懒加载 —— 只在真正要剪辑时才构造（构造一次成本不低）
# ==========================================================================
_editor_agent = None


def _find_chromium() -> str:
    """找本机浏览器可执行文件，给 HyperFrames 渲染用（课案原有）。"""
    for p in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ):
        if os.path.exists(p):
            return p
    return ""


def _build_editor_system_prompt(work_dir_env: str = "videos") -> str:
    """构造剪辑 agent 的系统提示词（课案原文 + 本仓库实测补充）。"""
    return (
        "你是视频剪辑专家。\n"
        "所有视频处理必须使用 moviepy 库！示例："
        "from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, "
        "concatenate_videoclips, TextClip, vfx, afx\n"
        "⚠️ SubtitlesClip 不在顶层，必须写："
        "from moviepy.video.tools.subtitles import SubtitlesClip\n"
        "禁止写 subprocess.run(['ffmpeg',...]) 或 ffprobe！禁止在脚本中直接调用 ffmpeg/ffprobe 子进程！\n"
        "moviepy 内部会自动调用 ffmpeg，你只需用它的 Python API：\n"
        "  - 加速: clip.with_effects([vfx.SpeedX(1.3)])\n"
        "  - 裁剪: clip.subclipped(start, end)\n"
        "  - 缩放: clip.resized(height=426)\n"
        "  - 音量: clip.with_effects([afx.VolumeX(3.0)])\n"
        "  - 拼接: concatenate_videoclips([clip1, clip2])\n"
        "  - 定位: clip.with_position(('center', y))\n"
        "  - TextClip 用 font= 和 font_size=（2.x 没有 fontsize=）\n"
        "  - 合成: CompositeVideoClip([top, bottom.with_position((0, top_h))], "
        "size=(w, top_h + bottom_h)) ← size 必须包含全部层高，"
        "例如原视频高 h、下方素材高 h//3 时就是 size=(w, h + h//3)；"
        "严禁写成 (w, h)，否则产生隐式蒙版裁掉底部\n"
        "  - 严禁: .with_mask() / .set_mask() / .to_mask() / clip.resized((w,h)) —— 都会产生隐式蒙版\n"
        "  - 字幕: 只能用 SubtitlesClip 加载 SRT 文件，严禁 for 循环 + TextClip（会导致字幕重复）\n"
        "    必须用 make_textclip 参数显式指定字幕位置，否则 SubtitlesClip 默认定位到合成帧最底部\n"
        "    （素材区域），字幕会被裁掉！正确写法：\n"
        "      def make_st(txt):\n"
        "          return TextClip(text=txt, font='Microsoft YaHei', font_size=52,\n"
        "              color='white', stroke_color='black', stroke_width=2,\n"
        "              size=(w, None), method='caption')\n"
        "              .with_position(('center', h * 0.75))  # ← 关键！定位在原视频区域内\n"
        "      subtitles = SubtitlesClip('subtitles.srt', encoding='utf-8', make_textclip=make_st)\n"
        "  - 混音: CompositeAudioClip([voice, bgm])\n"
        "  - 输出: clip.write_videofile('output.mp4', codec='libx264', audio_codec='aac')\n"
        "写 Python 脚本时 import moviepy 完成全部剪辑、合成、字幕、混音操作，"
        "然后 execute('python script.py')。\n"
        "\n"
        "=== 语音转文字（生成字幕）怎么做 ===\n"
        "**不要自己调 funasr**（本项目已不用本地模型）。直接用项目自带的工具：\n"
        "  from tools.audio_transcriber import transcribe_to_srt\n"
        "  srt = transcribe_to_srt(视频或音频路径, os.path.join(os.environ['WORK_DIR'], 'subtitles.srt'))\n"
        "它走百炼云端接口，接受视频路径（内部自己抽音频）。\n"
        "失败时返回空字符串 —— 此时**跳过字幕继续渲染**，不要因为没字幕就整个任务停下来。\n"
        "\n"
        "素材保持宽高比禁拉伸，输出分辨率与源视频完全一致。\n"
        "BGM: volume=0.1 人声: volume=3.0 混音: volume=3.0。\n"
        "BGM 文件不存在就跳过混音，不要报错停下。\n"
        "npx 加 --yes，禁 playwright install。\n"
        "HyperFrames 生成 HTML 时，每个文件 `<head>` 内必须加：\n"
        "  `<style>body{font-family:'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif}</style>`\n"
        "  否则中文全变方框！不写这条 style 的中文内容等于白做。\n"
        "如果 npx hyperframes 不可用（装不上/报错），**改用 moviepy 直接生成动画素材**：\n"
        "  ColorClip + TextClip + with_position 做位移与淡入淡出。不要因此中断任务。\n"
        "\n"
        "=== ⚠️ 运行环境（必须看，否则会卡死） ===\n"
        "execute 工具运行的是 **Windows 的 cmd.exe，不是 bash**：\n"
        "  列目录用 dir，不要用 ls；看当前路径用 cd，不要用 pwd；\n"
        "  看文件内容用 type，不要用 cat；删除命令一律不许用。\n"
        "  想列目录/读文件，优先用 ls / read_file 这类文件工具（它们跨平台），\n"
        "  只有真的要跑代码时才用 execute。\n"
        "运行 Python 脚本时直接写 `python 脚本名.py`（PATH 已指向虚拟环境，不要写全路径）。\n"
        "\n"
        "=== ⚠️ 沙箱路径规则（每次文件操作前必须遵守，违反者输出会被丢弃） ===\n"
        "你运行在一个文件沙箱中，当前工作目录就是沙箱根目录。\n"
        "所有文件操作（write、ls、read、execute）都只能访问沙箱内的文件。\n"
        "\n"
        "【输出目录】\n"
        f"  os.environ['WORK_DIR'] = '{work_dir_env}'（沙箱内的相对路径）\n"
        "  所有中间文件和最终输出都放在它下面。\n"
        "\n"
        "【Python 脚本内】\n"
        "  out = os.path.join(os.environ['WORK_DIR'], 'output.mp4')\n"
        "  写文件前创建目录：os.makedirs(os.path.dirname(out), exist_ok=True)\n"
        "\n"
        "【后端工具（write/ls/read）】\n"
        "  同样用 os.environ['WORK_DIR'] 拼接相对路径。\n"
        "  例如: ls videos/  或  write('videos/script.py', content)\n"
        "\n"
        "【外部文件（源视频、BGM）】\n"
        "  源视频和 BGM 的绝对路径会通过任务提示传入，直接读取即可，不需要复制到沙箱。\n"
        "\n"
        "【严禁】\n"
        "  禁止手写任何含盘符的路径（C:、D: 等）。\n"
        "  禁止使用 /c/Users 这种 Linux 风格绝对路径。\n"
        "  所有输出路径一律用 os.path.join(os.environ['WORK_DIR'], '文件名')。\n"
    )


def _get_editor_agent():
    """构造（并缓存）视频剪辑 deepagent。"""
    global _editor_agent
    if _editor_agent is not None:
        return _editor_agent

    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend

    model = get_model(temperature=0.5, use_deepagent_model=True)

    skills_dir = os.path.join(_PROJECT_ROOT, ".skills")
    cache_dir = settings.media.get_mashup_work_dir()
    os.makedirs(cache_dir, exist_ok=True)

    # ---- 子进程环境：课案原有 + 本仓库实测补充 ----
    env = os.environ.copy()

    chromium = _find_chromium()
    if chromium:
        env["CHROMIUM_PATH"] = chromium
        env["PUPPETEER_EXECUTABLE_PATH"] = chromium
        print(f"[deepagents] 浏览器: {chromium}")
    else:
        print("[deepagents] ⚠️ 未找到 Edge/Chrome，HyperFrames 渲染可能失败（会降级为 moviepy 动画）")

    # 禁 MSYS/Git Bash 路径自动转换（课案原有）
    env["MSYS_NO_PATHCONV"] = "1"
    env["MSYS2_ARG_CONV_EXCL"] = "*"
    env["GIT_BASH_PATH_CONV"] = "none"

    # ⚠️ 本仓库实测：PATH 里的 python 是 Windows Store 占位符，静默无输出。
    # 把当前 venv 的 Scripts 目录顶到最前面，让 agent 敲的 python 是真的解释器。
    venv_scripts = str(Path(sys.executable).resolve().parent)
    env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")

    # 注入项目路径，让 agent 引用而不是自己拼
    env["PROJECT_ROOT"] = _PROJECT_ROOT
    env["CACHE_DIR"] = cache_dir
    env["WORK_DIR"] = "videos"

    backend = LocalShellBackend(
        root_dir=cache_dir,
        virtual_mode=True,
        inherit_env=True,
        timeout=300,
        max_output_bytes=5_000_000,
        env=env,
    )

    # monkey-patch：清洗 Git Bash 路径 /c/Users/... → C:\Users\...
    # 否则 virtual_mode 会把它当成虚拟路径拼到沙箱下面（课案原有）
    _orig_resolve = backend._resolve_path

    def _patched_resolve(key: str):
        cleaned = normalize_path(key)
        if os.path.isabs(cleaned):
            try:
                rel = os.path.relpath(cleaned, cache_dir)
                if not rel.startswith(".."):
                    cleaned = rel
            except ValueError:
                pass  # 不同盘符，保持原样让后端正常拒绝
        return _orig_resolve(cleaned)

    backend._resolve_path = _patched_resolve  # type: ignore[method-assign]

    _editor_agent = create_deep_agent(
        model=model,
        skills=[skills_dir] if os.path.isdir(skills_dir) else [],
        backend=backend,
        system_prompt=_build_editor_system_prompt("videos"),
    )
    print("[deepagents] 视频剪辑 Agent 已初始化")
    print(f"[deepagents] 技能目录: {skills_dir}")
    print(f"[deepagents] 沙箱根目录: {cache_dir}")
    return _editor_agent


def reset_editor_agent() -> None:
    """丢弃缓存的 agent（改完 SKILL.md 或 .env 后想立刻生效时用）。"""
    global _editor_agent
    _editor_agent = None


# ==========================================================================
# 节点 1：deepagent 剪辑（全权负责）
# ==========================================================================
def node_edit_video(state: MashupState) -> dict:
    """把剪辑任务整包交给 deepagent。"""
    video = safe_abspath(state.get("input_video", ""))
    edit_start = time.time()

    if not video or not os.path.exists(video):
        return {"editor_log": f"源视频不存在: {video}", "output_video": "", "edit_start": edit_start}

    # ---- 解析剪辑要求 ----
    raw_req = state.get("edit_requirements", "")
    try:
        req = json.loads(raw_req) if raw_req else {}
    except Exception:  # noqa: BLE001 —— 传了非 JSON 就当纯文本要求
        req = {"extra_requirements": raw_req}

    materials = req.get("materials", []) or []
    bgm_path = normalize_path(req.get("bgm_path") or settings.media.bgm_path or "")
    extra_req = req.get("extra_requirements", "")

    materials_desc = ""
    if materials:
        materials_desc = "\n\n## 穿插素材\n"
        for i, mp in enumerate(materials, 1):
            mp = normalize_path(mp)
            kind = "视频" if os.path.splitext(mp)[1].lower() in (".mp4", ".mov", ".avi", ".webm") else "图片"
            materials_desc += f"{i}. [{kind}] `{os.path.abspath(mp)}`\n"

    if bgm_path and os.path.exists(bgm_path):
        bgm_info = f"\nBGM: `{bgm_path}`"
    else:
        bgm_info = "\nBGM: 无（本次不混音）"

    prompt = f"""## 剪辑任务

### 源视频
`{os.path.abspath(video)}`
{materials_desc}
{bgm_info}

### 用户要求
{extra_req if extra_req else "完整剪辑（始终保持和原视频像素一致）：转录 → 动画素材 → 排布穿插 → BGM → 字幕 → 渲染"}

先用 HyperFrames 生成至少 6~10 个动画素材（绝对不少于 6 个，越多越好），
注意要指明字体防止中文乱码。
开头素材是在最上的图层、与原视频像素一致；其他素材高度需要是 1/3，放在原视频下面。
根据口播内容节奏在不同时间点穿插素材，有的上下排布：上边用原来的口播视频，
下边用生成的素材（覆盖口播视频下部分，空素材的地方要全屏用原视频，不能黑屏）。
注意不要拉伸图像，不要上下都用原视频。

=== 字幕约束（必须严格遵守，否则字幕重复 + 下半截消失） ===
字幕只能生成一次！正确流程：调 tools.audio_transcriber.transcribe_to_srt() 生成一个 SRT 文件
→ 用 SubtitlesClip('subtitle.srt') 加载 SRT，仅此一次！
严禁 for 循环手动创建 TextClip 来逐行添加字幕！严禁同时使用 SubtitlesClip 和 TextClip！
关键：必须用 make_textclip 参数 + with_position 显式指定字幕位置，
否则 SubtitlesClip 默认把字幕定位到合成帧最底部（y ≈ h + h//3，是素材区域），
字幕下半截会被隐式蒙版裁掉！
正确写法：
  def make_st(txt):
      return TextClip(text=txt, font='Microsoft YaHei', font_size=52,
          color='white', stroke_color='black', stroke_width=2,
          size=(w, None), method='caption'
      ).with_position(('center', h * 0.75))  # ← 定位在原视频区域内，不是合成帧底部
  subtitles = SubtitlesClip('subtitles.srt', encoding='utf-8', make_textclip=make_st)

=== 蒙版约束（必须严格遵守，否则底部被隐式蒙版裁掉） ===
严禁在任何 clip 上调用 .with_mask() / .set_mask() / .to_mask()！
CompositeVideoClip 的 size 必须为 (w, h + h//3)，严禁写成 (w, h) ——
因为 (w, h) 会在 h 位置创建隐式裁剪蒙版，把素材层和字幕下半截全部裁掉！
video_clip 不要调用 .resized((w, h)) 改尺寸，保持原始分辨率，避免隐式蒙版！

=== 字幕位置约束 ===
字幕只能放在原视频区域内（y 坐标范围: h*0.02 ~ h*0.65），
计算 y = h*0.62 - clip_h//2，保证字幕底部离 h 边界至少留 20px 余量。
字幕 clip 排在 CompositeVideoClip 列表最后一项（最上层）。

=== 以上约束必须写到 Python 脚本的注释里 ===

之后加上背景音乐（如果上面给了 BGM 路径）：
如果背景音乐时长短于视频就重复使用；衔接时去掉开头音乐空白的地方（第一段不去）。
背景音乐需要降低至 10% 音量，总体音量调大至 300%。

- 最终输出路径用: os.path.join(os.environ['WORK_DIR'], 'mashup_final.mp4')
- ⚠️ 所有文件路径使用 os.path.join(os.environ['WORK_DIR'], 'xxx') 构造，不要用手写路径！
- 不要重复加字幕！只最后加一次！
完成后报告最终输出路径（绝对路径）。"""

    print("[剪辑] 启动 deepagent（流式输出）...")
    steps_log: list = []
    tool_count = 0
    last_chunk: dict = {}

    try:
        agent = _get_editor_agent()
    except Exception as exc:  # noqa: BLE001 —— agent 构造失败也要给出中文说明
        import traceback

        traceback.print_exc()
        return {
            "editor_log": f"deepagent 初始化失败: {exc}",
            "steps": json.dumps(steps_log, ensure_ascii=False),
            "output_video": "",
            "edit_start": edit_start,
        }

    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode="values",
            config={"recursion_limit": 50},   # 课案没放宽步数，长任务会撞上限
        ):
            last_chunk = chunk
            msgs = chunk.get("messages", [])
            if not msgs:
                continue
            last = msgs[-1]

            if hasattr(last, "content") and last.content and hasattr(last, "type"):
                c = str(last.content).strip()
                if c and c not in steps_log:
                    steps_log.append(c)
                    emoji = {"ai": "🤖", "tool": "🔧", "human": "👤"}.get(
                        getattr(last, "type", "?"), "📝"
                    )
                    print(f"  [{emoji}] {c[:300]}")

            for tc in (getattr(last, "tool_calls", None) or []):
                tool_count += 1
                name = tc.get("name", "?")
                args = tc.get("args", {}) or {}
                steps_log.append(f"[tool] {name}")
                cmd = args.get("command", args.get("cmd", ""))
                if cmd:
                    print(f"  [{tool_count}] ⚡ {name}: {str(cmd)[:200]}")
                elif name in ("write_file", "write"):
                    print(f"  [{tool_count}] 📄 {name}: {args.get('file_path', args.get('path', '?'))}")
                elif name in ("read_file", "read"):
                    print(f"  [{tool_count}] 📖 {name}: {args.get('file_path', args.get('path', '?'))}")
                else:
                    print(f"  [{tool_count}] 🔨 {name}")

            if getattr(last, "type", "") == "tool":
                result = str(getattr(last, "content", ""))[:200]
                if result:
                    print(f"  [↩] → {result}")

    except GraphRecursionError:
        # 深度智能体在「命令一直失败」时会不断换写法重试，最终撞上步数上限。
        # 兜住它并给中文提示，而不是甩一个 traceback。
        print(
            "[剪辑] 达到步数上限仍未收敛：模型大概率在反复尝试当前平台不存在的命令。\n"
            "       请检查 system_prompt 里是否说明了 execute 跑的是 cmd.exe（本文件已说明）。"
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return {
            "editor_log": f"剪辑出错: {exc}",
            "steps": json.dumps(steps_log, ensure_ascii=False),
            "output_video": "",
            "edit_start": edit_start,
        }

    # ---- 从最后一条消息里解析输出路径 ----
    output_video = ""
    last_msg = ""
    for m in reversed(last_chunk.get("messages", [])):
        c = str(getattr(m, "content", ""))
        if not c:
            continue
        last_msg = c
        for word in c.replace(",", " ").replace("\n", " ").split():
            word = normalize_path(word.strip("`\"'[]()"))
            if word.endswith(".mp4") and os.path.isfile(word):
                try:
                    if os.path.getmtime(word) > edit_start - 10 and \
                       os.path.getsize(word) != os.path.getsize(video):
                        output_video = word
                        break
                except OSError:
                    continue
        if output_video:
            break

    return {
        "editor_log": last_msg[:2000] if last_msg else "deepagent 完成",
        "steps": json.dumps(steps_log, ensure_ascii=False),
        "output_video": output_video,
        "edit_start": edit_start,
    }


# ==========================================================================
# 节点 2：兜底查找（agent 没报告路径时，按约定位置找）
# ==========================================================================
def node_find_output(state: MashupState) -> dict:
    """兜底查找输出文件。

    课案这里是按 ``.cache/videos/mashup_final.mp4`` 找的，
    但它两个文件的 CACHE_DIR 不一致导致扫不到。本项目统一用
    ``MEDIA_MASHUP_WORK_DIR``，所以这里能对上。
    """
    inp = safe_abspath(state.get("input_video", ""))

    out = normalize_path(state.get("output_video", ""))
    if out and os.path.isfile(out) and safe_abspath(out) != inp:
        return {"output_video": safe_abspath(out)}

    cache_dir = settings.media.get_mashup_work_dir()
    candidates = [
        os.path.join(cache_dir, "mashup_final.mp4"),
        os.path.join(cache_dir, "final_with_bgm.mp4"),
        os.path.join(cache_dir, "videos", "mashup_final.mp4"),
    ]
    for cand in candidates:
        cand = normalize_path(cand)
        if os.path.isfile(cand) and safe_abspath(cand) != inp:
            return {"output_video": safe_abspath(cand)}

    # 最后再按修改时间扫一遍目录
    newest = ""
    newest_mtime = 0.0
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            if not f.lower().endswith(".mp4"):
                continue
            fp = os.path.join(root, f)
            try:
                if os.path.getsize(fp) < 100 * 1024:
                    continue
                if safe_abspath(fp) == inp:
                    continue
                mt = os.path.getmtime(fp)
            except OSError:
                continue
            if mt > newest_mtime:
                newest, newest_mtime = fp, mt

    if newest:
        print(f"[剪辑] 兜底找到最新产出: {newest}")
        return {"output_video": safe_abspath(newest)}
    return {}


# ==========================================================================
# 构图
# ==========================================================================
builder = StateGraph(MashupState)
builder.add_node("edit", node_edit_video)
builder.add_node("find", node_find_output)

builder.add_edge(START, "edit")
builder.add_edge("edit", "find")
builder.add_edge("find", END)

mashup_graph = builder.compile()


def run_mashup(input_video: str, edit_requirements: str = "", **_kwargs) -> dict:
    """运行视频剪辑工作流。

    Args:
        input_video: 源口播视频路径。
        edit_requirements: JSON 字符串，形如
            ``{"materials": [...], "bgm_path": "...", "extra_requirements": "..."}``。

    Returns:
        ``{"input_video","output_video","editor_log","steps",...}``
    """
    return mashup_graph.invoke({
        "input_video": input_video,
        "edit_requirements": edit_requirements,
    })


if __name__ == "__main__":
    from pathlib import Path as _P

    _root = str(_P(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    print("=== 视频剪辑工作流自检（离线，不启动 agent）===")

    g = mashup_graph.get_graph()
    nodes = sorted(g.nodes)
    assert "edit" in nodes and "find" in nodes, nodes
    print(f"  图节点: {nodes}  OK")

    # 1) 路径清洗（课案的 normalize_path，这是最容易被改坏的一段）
    cases = {
        r"C:\c\Users\a.mp4": r"C:\Users\a.mp4",
        "C:/c/Users/a.mp4": r"C:\Users\a.mp4",
        r"C:\d\Data\a.mp4": r"D:\Data\a.mp4",
    }
    for raw, want in cases.items():
        got = normalize_path(raw)
        assert got == want, f"normalize_path({raw!r}) = {got!r}，期望 {want!r}"
        print(f"    {raw!r} → {got!r}")
    if sys.platform == "win32":
        got = normalize_path("/c/Users/a.mp4")
        assert got == r"C:\Users\a.mp4", got
        print(f"    '/c/Users/a.mp4' → {got!r}")
    print("  normalize_path            OK")

    # 2) 源视频不存在 → 中文提示，不抛异常
    r = run_mashup("不存在的视频.mp4")
    assert "源视频不存在" in r.get("editor_log", ""), r
    print("  源视频缺失处理             OK")

    # 3) 兜底查找：沙箱目录里没有 mp4 时应返回空而不是报错
    r2 = node_find_output({"input_video": "不存在.mp4", "output_video": ""})
    assert r2.get("output_video", "") == "", r2
    print("  兜底查找空目录             OK")

    # 4) SKILL.md 存在且格式正确（DeepAgents 靠 frontmatter 识别技能）
    skill = _P(_root) / ".skills" / "video-use" / "SKILL.md"
    assert skill.is_file(), f"技能文件缺失: {skill}"
    head = skill.read_text(encoding="utf-8")[:200]
    assert head.startswith("---"), "SKILL.md 必须以 YAML frontmatter 开头"
    assert "name: video-use" in head, head
    assert "description:" in head, head
    print(f"  SKILL.md 格式              OK  ({skill.stat().st_size} 字节)")

    # 5) system_prompt 覆盖了几条关键约束（这些丢了必然出问题）
    sp = _build_editor_system_prompt()
    for must in ("cmd.exe", "moviepy.video.tools.subtitles",
                 "h + h//3", "transcribe_to_srt", "Microsoft YaHei"):
        assert must in sp, f"system_prompt 缺少关键约束: {must}"
    print("  system_prompt 关键约束     OK")

    print(f"\n  沙箱目录: {settings.media.get_mashup_work_dir()}")
    print(f"  BGM: {settings.media.bgm_path or '（未配置，将不混音）'}")
    print(f"  编排模型: {settings.media_deepagent_model()}")
    print("全部自检通过")
