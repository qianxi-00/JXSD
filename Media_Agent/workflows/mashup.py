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
    | 转写：agent 自己写脚本调本地 funasr | 走项目自带的 ``tools.audio_transcriber``（百炼 ``qwen-audio-3.0-asr-flash``，Fun-ASR 家族的云托管版） |
    | 默认 BGM：硬编码 ``C:\\Users\\13261\\Pictures\\风格\\...wav`` | ``MEDIA_BGM_PATH``；留空则不混音 |
    | 沙箱目录：``.cache/videos`` | ``MEDIA_MASHUP_WORK_DIR``（``.cache/mashup``）——课案两个文件的 CACHE_DIR 不一致，兜底查找扫不到，已统一 |
    | 无步数上限 | 加 ``recursion_limit`` + 兜住 ``GraphRecursionError`` |
      （最初设 50，实测**不够**：一次超时命令的降级重试就吃掉十几个 super-step，
       29 步工具调用即撞上限。调到 150 后足以跑完「转写 → 生成素材 → 渲染 → 合成」。）

课案原有的 Windows 适配**全部保留**（这些是真踩出来的，不要删）
    1. ``sys.stdout/stderr.reconfigure(encoding='utf-8')`` —— 否则日志中文变乱码
    2. ``subprocess.Popen`` monkey-patch：强制 UTF-8 + ``CREATE_NO_WINDOW`` +
       ``bufsize=102400 * 1024``（≈100MB；ffmpeg 的 stderr 输出很猛，管道小了会把进程堵死）
    3. ``normalize_path()`` —— 修 Git Bash 与 Windows 混用产生的畸形路径
       （``C:\\c\\Users\\...`` / ``/c/Users/...`` / ``C:\\d\\Data\\...``）
    4. ``MSYS_NO_PATHCONV=1`` / ``MSYS2_ARG_CONV_EXCL=*`` 禁 Git Bash 路径自动转换
    5. ``LocalShellBackend._resolve_path`` monkey-patch —— virtual_mode 会把
       ``/c/Users/...`` 当虚拟路径拼到沙箱下面，必须在解析前先清洗
    6. 给子进程注入 ``CHROMIUM_PATH``（HyperFrames 渲染要浏览器）

本仓库实测补的四条（**课案没有，不加会翻车**）
    7. **把当前 venv 的 Scripts 目录顶到 PATH 最前面**：
       本机 PATH 里的 ``python`` 是 Windows Store 占位符，执行后**静默无输出**。
       不让 ``python`` 指向 ``.venv\\Scripts\\python.exe``，
       agent 跑 ``python script.py`` 就会「成功但什么也没发生」。
    8. **system_prompt 里必须说明 execute 跑的是 cmd.exe 不是 bash**：
       不写，模型会反复敲 ``ls`` / ``pwd`` / ``cat``，实测一路撞到
       ``GraphRecursionError``（Recursion limit of 50 reached）。
    9. **技能目录必须经 ``CompositeBackend`` 挂到虚拟路径**：
       课案把技能目录的**绝对路径**直接传给 ``create_deep_agent(skills=...)``，
       但 deepagents 规定 skills 路径**相对 backend 的 root_dir**
       （``backend.ls(source_path)`` 会走 ``_resolve_path`` 的越界检查）。
       实测第一次真跑剪辑即抛
       ``ValueError: Path ... outside root directory: <沙箱根>``，
       agent 一步都没执行。改为 ``routes={"/skills/": FilesystemBackend(...)}``。
    10. **Windows 上 `capture_output=True` 不能走管道**：
        管道句柄会被整棵子进程树继承，只要有进程长期存活，
        `communicate()` 就永远等不到 EOF —— 而且 `timeout=` 只负责"发现"超时，
        之后仍要回收管道，所以在 Windows 上**超时机制会整体失效**。
        实测：`execute(timeout=8)` 在 301.8 秒后才返回，`npx hyperframes render`
        直接把整轮剪辑卡死。补丁改成用临时文件收输出 + `Popen.wait(timeout)`
        判超时（见下方 `_patched_run`）。

本文件在系统里的位置
    「``views/`` → ``workflows/`` → ``tools/``」三层里的中间层。
    真实入口是 ``views/mashup.py``（它调本文件的包装函数 ``run_mashup()``），
    本文件对外暴露的是 ``run_mashup()`` 与编译好的 ``mashup_graph``
    （后者是内部契约，页面不该直接 ``invoke``）。

数据流（两个节点一条直线，没有条件边）
    ``node_edit_video``  把整包任务交给 deepagent，产出 ``output_video``；
    ``node_find_output`` 只在上一节点没报出可用路径时才按约定位置兜底找。
    两个节点都**不抛异常**：失败时把中文失败串回填到 ``editor_log``、
    ``output_video`` 留空，由视图层判断「到底出没出片」。

运行与自检
    自检（离线，不启动 agent、不调 LLM，约 4 秒）::

        cd Media_Agent
        $env:PYTHONUTF8=1
        & ..\\.venv\\Scripts\\python.exe workflows/mashup.py       # 期望末行「全部自检通过」

    自检覆盖的都是「改坏了会静默出错」的点：路径清洗、失败分支、
    ``/skills/`` 后端接线、以及下面两个 monkey-patch 的回归。
"""

import asyncio
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
    """``subprocess.Popen.__init__`` 的替身：补 UTF-8 与 Windows 防卡参数（课案原有）。

    只负责「调用方没指定时才给安全默认值」——全部用 ``setdefault``，
    显式传了参数的一律不动，所以对正常代码是透明的。

    Args:
        *args / **kwargs: 原样透传给真正的 ``Popen.__init__``。
    """
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        # 文本模式必须显式给编码：不指定就用系统区域默认编码（本机中文
        # Windows 是 GBK），而子进程（Git Bash / node / npx）吐的基本都是
        # UTF-8 —— 不指定就 UnicodeDecodeError；errors="replace" 保证零星
        # 脏字节只变 � 不炸。
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "replace")
    if sys.platform == "win32":
        # CREATE_NO_WINDOW 防止弹控制台窗口把父进程阻塞
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        # 缓冲区开成 ≈100MB（102400 * 1024）：ffmpeg 的 stderr 很能刷，
        # 管道小了会把子进程堵死在这个缓冲上（死锁，而不是变慢）。
        kwargs.setdefault("bufsize", 102400 * 1024)
    _orig_popen_init(self, *args, **kwargs)


subprocess.Popen.__init__ = _patched_popen_init  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# subprocess monkey-patch：Windows 上别用管道收输出（本仓库实测补的第 10 条）
#
# 症状：agent 跑 `npx hyperframes render ...` 时**永久卡死**，CPU 归零、
#   沙箱不再有任何产出。`faulthandler` 栈显示卡在
#       deepagents/backends/local_shell.py:304 -> subprocess.run -> communicate
#
# 根因（Windows 专有，两层，都实测过）：
#   ① `subprocess.run(capture_output=True)` 把 stdout/stderr 接到**管道**上，
#      管道句柄被**整棵子进程树**继承：
#          python -> cmd.exe(/c npx) -> node(npx) -> cmd.exe -> node(hyperframes)
#      只要树里存在一个长期存活的进程，`communicate()` 就永远等不到 EOF。
#   ② `timeout=` 只负责**发现**超时，之后仍要回收管道：
#        `except TimeoutExpired: process.kill(); process.communicate()`
#      —— Windows 上 `kill()` 只终结**直接子进程**，孙进程照旧攥着管道，
#      于是超时机制形同虚设。
#
#   独立探针实测（拿 `ping -n 300` 当"不死的孙进程"）：
#        subprocess.run(timeout=5)              -> 59.8s 才抛 TimeoutExpired
#        LocalShellBackend.execute(t=6)         -> 29.3s 才返回（= 孙进程自己的寿命）
#        execute(t=8) + `taskkill /F /T` 整树杀 -> 301.8s 才返回，**该方案无效**
#   最后一条是关键：`taskkill /T` 只能沿**活着的父进程**遍历，
#   而 `cmd /c start /b ...` 这类命令的直接子进程**早就退出了**，树根一没就无从下手。
#
# 修法：把 stdout/stderr 指向**临时文件**而不是管道 —— 没有管道就没有 EOF 可等。
#   · 用 `Popen.wait(timeout)` 判超时：Windows 上走 `WaitForSingleObject`，
#     与管道状态无关，超时**必定**按时触发；
#   · 超时后仍 `taskkill /F /T` 尽力清理进程树，再 `wait()` 收尾；
#   · 正常退出则从文件读回输出，`CompletedProcess` 语义与标准库一致。
#   即：把「无限卡死」降级成「一次有界的失败」——`LocalShellBackend.execute`
#   会捕获 `TimeoutExpired` 返回 exit_code=124 的中文提示，agent 据此走 moviepy 分支。
#
# 只接管 Windows + `capture_output=True` 这一种调用；其余一律走标准库原实现。
# ---------------------------------------------------------------------------
_orig_subprocess_run = subprocess.run


def _patched_run(*args, **kwargs):
    """``subprocess.run`` 的 Windows 专用替身：用临时文件收输出，避开管道死锁。

    只接管「Windows + ``capture_output=True`` + 没传 ``input``」这一种调用形态，
    其余一律透传给标准库（根因分析见上方那段块注释）。

    Args:
        *args: 与 ``subprocess.run`` 相同（命令串或参数列表）。
        **kwargs: 与 ``subprocess.run`` 相同。其中 ``capture_output`` /
            ``timeout`` / ``check`` 三个键由本函数自己消化，不下传给 ``Popen``。

    Returns:
        ``subprocess.CompletedProcess``。与标准库唯一的行为差别：
        stdout 与 stderr 合流后统一放在 ``stdout`` 字段，``stderr`` 恒为
        ``None``（调用方若按 ``(out, err)`` 解包，err 一定是 ``None``）。

    Raises:
        subprocess.TimeoutExpired: 超出 ``timeout`` 时抛出 —— **按时**抛出正是
            这个补丁存在的意义；抛出前会尽力清理进程树。
        subprocess.CalledProcessError: 调用方传了 ``check=True`` 且退出码非 0。
    """
    # 只接管这一种形态，别的走标准库原实现：
    #   · 非 Windows 没有「管道句柄被子进程树继承」这个问题，没必要陪跑；
    #   · 传了 input=... 时要往 stdin 写数据，临时文件方案不覆盖这块。
    if (sys.platform != "win32"
            or not kwargs.get("capture_output")
            or kwargs.get("input") is not None):
        return _orig_subprocess_run(*args, **kwargs)

    import tempfile

    timeout = kwargs.get("timeout")
    use_text = bool(kwargs.get("text") or kwargs.get("universal_newlines"))
    check = bool(kwargs.get("check"))

    # 三个键必须先从 kwargs 里摘掉：capture_output / timeout / check 都不是
    # Popen 的参数，原样传下去会 TypeError —— 后两个由本函数自己实现。
    kw = {k: v for k, v in kwargs.items()
          if k not in ("capture_output", "timeout", "check")}
    # 临时文件的模式要跟调用方的 text 设置对齐：文本模式下不显式给编码，
    # TemporaryFile 同样会退回 Windows 的 GBK 默认值（理由同 Popen 补丁）。
    fh_kwargs = (
        {"mode": "w+t",
         "encoding": kwargs.get("encoding") or "utf-8",
         "errors": kwargs.get("errors") or "replace"}
        if use_text else {"mode": "w+b"}
    )

    # TemporaryFile 是**匿名**临时文件（Windows 上带 O_TEMPORARY，关闭即删）：
    # 没有文件名要争，也不会在磁盘上留垃圾。
    with tempfile.TemporaryFile(**fh_kwargs) as fh:
        # stdout 与 stderr 指向**同一个**文件：两者不再可拆，但 agent 只要能
        # 「看到错误文本」就够了（与 capture_output 的合并语义一致）。
        kw["stdout"] = fh
        kw["stderr"] = fh
        proc = subprocess.Popen(*args, **kw)
        try:
            # 用 wait() 而不是 communicate()：Windows 上 wait() 走
            # WaitForSingleObject，只看进程本身；communicate() 要等管道 EOF，
            # 那正是死锁的来源。超时因此**必定按时触发**。
            retcode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                # 超时后尽力清理进程树。/T 只能沿**活着的父进程**往下遍历，
                # 对 `cmd /c start /b ...` 这种「树根早就退出」的形态无效 ——
                # 所以这步只是尽力而为，真正的保证来自「不再等管道」。
                # ⚠️ 这里的 subprocess.run 已经是打好补丁的版本（会再进一次本
                # 函数）：taskkill 没有孙进程，wait(timeout=None) 会正常返回，
                # 不会自己锁自己。
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, check=False,
                )
            except Exception:  # noqa: BLE001 —— 清理失败不能盖住原始超时
                pass
            # 不传 timeout：taskkill 之后进程一定会结束，这一步只是回收句柄。
            proc.wait()
            # 裸 raise 重抛原来的 TimeoutExpired —— 上面清理环节的任何失败
            # 都不该改变「这是一次超时」这个事实。
            raise
        # 必须在 with 块**内**读回：出了 with 文件就关了。
        fh.seek(0)
        out = fh.read()

    # 与标准库同构；第 4 个参数 stderr=None —— 两路合流后没法再拆开。
    result = subprocess.CompletedProcess(args, retcode, out, None)
    if check and retcode:
        # check=True 的语义要自己补：标准库是在 run() 内部检查的，
        # 而本函数绕开了它那段逻辑。
        raise subprocess.CalledProcessError(retcode, args, output=out)
    return result


subprocess.run = _patched_run  # type: ignore[assignment]

from langchain.agents.middleware import AgentMiddleware  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

# 下面这组 import 带 # noqa: E402（不是文件顶部的 import），是**必需的**：
# 上面的 sys.path 引导必须先生效，否则 python workflows/mashup.py 时
# sys.path[0] 是 workflows/ 目录，`from config import settings` 会 ModuleNotFoundError。
from config import settings  # noqa: E402
from workflows import get_model  # noqa: E402


class MashupState(TypedDict, total=False):
    """视频剪辑工作流状态。

    ``total=False``：节点可以只回填自己改动过的那几个键
    （见两个节点 return 的字典），不必凑齐全部字段。

    各字段的读写方：

    Attributes:
        input_video: 源口播视频路径。视图层写入，两个节点都读。
        edit_requirements: 剪辑要求（JSON 字符串或纯文本），``node_edit_video`` 读。
        output_video: 成品路径。``node_edit_video`` 写入（解析不到就是空串），
            ``node_find_output`` 既读又可能覆写它。
        editor_log: 给页面显示的一段日志（已截断到 2000 字符）。
        steps: 步骤流水，``json.dumps(..., ensure_ascii=False)`` 之后的字符串。
        edit_start: 本轮开始的时间戳，唯一用途是当「哪些 mp4 是本轮产物」的判据。
    """

    input_video: str
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

    为什么需要它：agent 敲的是 cmd.exe，但它写出来的路径常常是 Git Bash
    风格；两种风格混在一条命令里就会产生下面这些畸形串 —— 而文件工具把它
    当**虚拟路径**处理时会一路拼错（见 ``_get_editor_agent`` 里
    ``LocalShellBackend._resolve_path`` 的那段补丁）。

    常见畸形及修复::

        C:\\c\\Users\\...  →  C:\\Users\\...
        C:/c/Users/...    →  C:\\Users\\...
        /c/Users/...      →  C:\\Users\\...
        /d/project/...    →  D:\\project\\...
        C:\\d\\Data\\...  →  D:\\Data\\...   （盘符字母不同时以第二个为准）

    Args:
        p: 待清洗的路径。空串与任何非字符串原样返回 —— 调用点大多是
            ``state.get(...)``，拿到的可能就是 ``None``。

    Returns:
        清洗后的路径（已 ``normpath``）；``normpath`` 抛异常时返回规则 1~3
        处理过的中间结果，绝不因为脏输入返回空。

    Note:
        只要发生了改动就会 ``print`` 一行 ``[PathFix]``。**对整段散文逐词调用
        它会刷出几百行噪音**（它会把 URL 里的 ``/`` 也无差别换成 ``\\``）——
        调用点必须先按扩展名过滤再调，见 ``node_edit_video`` 的解析循环。
    """
    if not p or not isinstance(p, str):
        return p

    original = p

    # 1) 盘符叠加：C:\c\Users → C:\Users；C:\d\Data → D:\Data
    #    这是 Git Bash 把 `/c/...` 交给 cmd 时被补上盘符前缀的产物：
    #    两个字母相同 = 重复盘符（丢掉前一个），不同 = 第二个才是真盘符。
    m = re.match(r"^([A-Za-z]):[\\/]([a-z])([\\/].+)$", p)
    if m:
        drive1, drive2, rest = m.group(1).lower(), m.group(2).lower(), m.group(3)
        p = f"{m.group(1)}:{rest}" if drive1 == drive2 else f"{m.group(2).upper()}:{rest}"

    # 2) 纯 Git Bash 路径：/c/Users/... → C:\Users\...
    #    只在 Windows 上转：Linux 上 `/c/...` 很可能就是一条真实路径。
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

    # normpath 是纯字符串操作（只归整斜杠与 . / ..），包一层只为兜住
    # 含 NUL 之类的极端脏输入：清洗失败也好过让整条链路断在这里。
    try:
        p = os.path.normpath(p)
    except Exception:  # noqa: BLE001
        pass

    # 打印是留给排错的（能看到 agent 到底传了什么进来），但也是噪音源 ——
    # 见 docstring 里那条调用顺序约束。
    if p != original:
        print(f"[PathFix] 路径已修复: {original!r} → {p!r}")
    return p


def safe_abspath(p: str) -> str:
    """先清洗路径再 abspath，防畸形路径。

    本模块所有「要落盘、要跟输入比对」的路径都该过这一道：
    Windows 上 ``os.path.abspath("/c/Users/a.mp4")`` 会拼成
    ``<当前盘>:\\c\\Users\\a.mp4``，而本函数给出的是 ``C:\\Users\\a.mp4``。

    Args:
        p: 原始路径。

    Returns:
        绝对路径。⚠️ 传空串会得到**当前工作目录**（``abspath("")`` 的语义），
        所以调用方一律先判空再用。
    """
    return os.path.abspath(normalize_path(p))


# ==========================================================================
# deepagent 懒加载 —— 只在真正要剪辑时才构造（构造一次成本不低）
# ==========================================================================
_editor_agent = None


def _find_chromium() -> str:
    """找本机浏览器可执行文件，给 HyperFrames 渲染用（课案原有）。

    只探这四个默认安装位置，不去翻 PATH、注册表或 npx 缓存 —— 目的只是给
    HyperFrames 一个 ``CHROMIUM_PATH``，探不到也不改变主流程。

    Returns:
        找到的第一个浏览器可执行文件绝对路径；四个位置都没有时返回**空串**，
        调用方据此只打印一条告警（HyperFrames 本来就已经是降级链路）。
    """
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
    """构造剪辑 agent 的系统提示词（课案原文 + 本仓库实测补充）。

    Args:
        work_dir_env: 写进提示词的沙箱**相对**目录名，默认 ``"videos"``。
            必须与 ``_get_editor_agent()`` 注入的 ``env["WORK_DIR"]`` 是同一个
            字符串，否则模型按提示词拼出来的路径在沙箱里根本不存在。

    Returns:
        整段系统提示词（由一批字符串字面量拼成）。
    """
    # 段落顺序有讲究：先钉死「只用 moviepy」，再给具体 API 与踩坑点，
    # 最后才是运行环境与沙箱规则 —— 任务提示会接在这整段后面，越靠后的内容
    # 越贴近「本轮重点」的位置（反面教材见 _material_howto 的 docstring）。
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
        "  ⚠️ **中文字体必须给字体文件路径，不能写字族名**（本机实测）：\n"
        "      font='Microsoft YaHei'  →  ValueError: Invalid font ..., cannot open resource\n"
        "      font='C:/Windows/Fonts/msyh.ttc'  →  OK（微软雅黑）\n"
        "      （moviepy 2.x 把 font 直接交给 pillow 当文件加载，不做字体族解析。）\n"
        "  - 合成: CompositeVideoClip([top, bottom.with_position((0, top_h))], "
        "size=(w, top_h + bottom_h)) ← size 必须包含全部层高，"
        "例如原视频高 h、下方素材高 h//3 时就是 size=(w, h + h//3)；"
        "严禁写成 (w, h)，否则产生隐式蒙版裁掉底部\n"
        "  - 严禁: .with_mask() / .set_mask() / .to_mask() / clip.resized((w,h)) —— 都会产生隐式蒙版\n"
        "  - 字幕: 只能用 SubtitlesClip 加载 SRT 文件，严禁 for 循环 + TextClip（会导致字幕重复）\n"
        "    必须用 make_textclip 参数生成字幕，否则 SubtitlesClip 默认定位到合成帧最底部\n"
        "    （素材区域），字幕会被裁掉！正确写法：\n"
        "      FONT = 'C:/Windows/Fonts/msyh.ttc'   # 必须是字体文件\n"
        "      def make_st(txt):\n"
        "          return TextClip(text=txt, font=FONT, font_size=52,\n"
        "              color='white', stroke_color='black', stroke_width=2,\n"
        "              size=(w, None), method='caption')\n"
        "      # ⚠️ 位置要打在 SubtitlesClip 本身，写在 make_textclip 里的 position 不生效：\n"
        "      subtitles = (SubtitlesClip('subtitles.srt', encoding='utf-8',\n"
        "                                 make_textclip=make_st)\n"
        "                   .with_position(('center', h * 0.62)))  # ← 落在原视频区域内\n"
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
        # ⚠️ 下面这段与 _material_howto() 读的**是同一个开关**，必须一起改：
        #    只切一处时，模型会在同一次调用里同时收到「禁止碰 HyperFrames」与
        #    「先用 HyperFrames」两条相反指令，实测把首轮 E2E 拖到 301.8s 超时。
        + (
            # ---- 课案原方案：HyperFrames 生成 HTML/GSAP 动画素材 ----
            "HyperFrames 生成 HTML 时，每个文件 `<head>` 内必须加：\n"
            "  `<style>body{font-family:'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif}</style>`\n"
            "  否则中文全变方框！不写这条 style 的中文内容等于白做。\n"
            "\n"
            "=== ⚠️ HyperFrames 的时间预算（实测踩过，必须遵守） ===\n"
            "  最多尝试 3 次 `npx --yes hyperframes render`。任何一次失败、报错、超时或卡住，\n"
            "  **立刻放弃 HyperFrames**，改用 moviepy 生成动画素材，把剩下的步数用在真正的剪辑上。\n"
            "  实测反面教材：为了排查 hyperframes 去翻 npm 缓存目录、读 renderSetupWorker.js、\n"
            "  查浏览器路径、装 gsap、试 CDN —— 31 步全耗在这上面，最后一步剪辑都没做，\n"
            "  任务直接撞步数上限失败。**上述任何一件事都不要做。**\n"
            if settings.media.mashup_use_hyperframes
            # ---- 降级方案（默认）：直接用 moviepy 画动画素材 ----
            else
            "=== ⚠️ 动画素材一律用 moviepy 画，禁止碰 HyperFrames（本机已关掉） ===\n"
            "  本机 HyperFrames 的浏览器链路不可用：`npx hyperframes init/render` 实测会卡到\n"
            "  超时（exit_code=124），纯属浪费步数。**不要**执行任何 hyperframes 命令，\n"
            "  也不要去看它的文档、npm 缓存目录或内部源码。\n"
            "  直接用 moviepy 生成动画素材：ColorClip + TextClip + with_position\n"
            "  做位移与淡入淡出；中文字体写 font='C:/Windows/Fonts/msyh.ttc'（**不要写\n"
            "  'Microsoft YaHei'**，moviepy 2.x 会当文件加载并报 Invalid font）。\n"
            "  需要几个就生成几个，别在这件事上反复试错。\n"
        )
        + "\n"
        # ---- 运行环境与沙箱规则：这两段都是实测踩坑后补的，删了必然翻车 ----
        #    缺「cmd.exe 不是 bash」这一句，模型会反复敲 ls / pwd / cat，
        #    一路撞到 GraphRecursionError；缺「沙箱外路径只能走 execute」这一句，
        #    它会拿绝对路径去喂 read_file，抛 outside root directory 当场终结整轮。
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
        "【文件工具】ls / read_file / write_file / glob / grep 只能访问沙箱内的**虚拟路径**；\n"
        "【execute】 不受沙箱限制，能访问本机任意绝对路径（它跑的是真实的 cmd）。\n"
        "\n"
        # 【输出目录】
        "【输出目录】\n"
        # work_dir_env 参数唯一的落点就是下面这个 f-string：提示词里让模型
        # 用 os.environ['WORK_DIR'] 拼路径，而不是自己去猜沙箱根在哪。
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
        # 这一节讲的是「沙箱外路径」的**后果**：技能目录改挂到 /skills/ 虚拟路径
        # 之后，模型最容易犯的错就变成「拿绝对路径去喂文件工具」——实测抛
        # outside root directory 并把跑了几分钟的整轮任务当场打断。
        "【外部文件（源视频、BGM、项目源码）—— 这里最容易翻车，看清楚】\n"
        "  它们的绝对路径会通过任务提示传入。**只有 execute 能读沙箱外的路径**。\n"
        "  ⚠️ 把绝对路径交给 read_file / ls / glob / grep，一定会报\n"
        "     \"outside root directory\"，而且这个异常会**让整轮任务当场中断**\n"
        "     （deepagents 默认把工具异常直接抛出，不走「回给模型重试」）。\n"
        "  所以：要读沙箱外的文件，一律走 execute ——\n"
        "     读文本   execute('type F:\\\\...\\\\tools\\\\audio_transcriber.py')\n"
        "     列目录   execute('dir F:\\\\...\\\\tools /b')\n"
        "     跑脚本   execute('python F:\\\\...\\\\script.py')\n"
        "  在脚本内部用绝对路径完全合法（脚本是 execute 起的子进程，不受沙箱限制）。\n"
        "  不需要把源视频复制进沙箱。\n"
        "\n"
        # 「挂在虚拟路径下」这条必须在提示词里说清：磁盘上的真实位置
        # （<项目根>/.skills/）对 agent 不可见，它只能按 /skills/ 这个虚拟路径读。
        "【技能手册】\n"
        "  剪辑操作手册挂在虚拟路径 /skills/ 下（在沙箱之外，只读用途），用 read_file 读，\n"
        "  例如 /skills/video-use/SKILL.md。不要往 /skills/ 里写文件。\n"
        "\n"
        "【严禁】\n"
        "  禁止手写任何含盘符的路径（C:、D: 等）**交给文件工具**。\n"
        "  禁止使用 /c/Users 这种 Linux 风格绝对路径。\n"
        "  所有输出路径一律用 os.path.join(os.environ['WORK_DIR'], '文件名')。\n"
    )


def _material_howto() -> str:
    """「动画素材怎么生成」这段话 —— 任务提示与 system_prompt 必须**同源同开关**。

    为什么单拎出来：曾经只把 system_prompt 按 ``MEDIA_MASHUP_USE_HYPERFRAMES``
    切了分支，任务提示里却还留着课案原文「先用 HyperFrames 生成…」。
    同一次调用里模型同时收到「禁止碰 HyperFrames」和「先用 HyperFrames」两条
    相反指令，而任务提示更具体、位置更靠后 —— 实测 agent 仍去试 ``npx``，
    一路撞到 300 秒超时（见 VERIFY_REPORT.md 5.9②）。

    单拎成函数还有个好处：自检可以直接调用它断言，不必去拼整条 prompt。

    Args:
        无。

    Returns:
        一段可直接插进 prompt 的中文说明，内容由 ``settings.media
        .mashup_use_hyperframes`` 二选一决定。
    """
    # 开关只有一个，两个分支互斥：调用方（system_prompt 与任务提示）拿到的一定
    # 是同一份措辞 —— 这就是把这段单拎出来的全部意义。
    if settings.media.mashup_use_hyperframes:
        return (
            "先用 HyperFrames 生成至少 6~10 个动画素材（绝对不少于 6 个，越多越好），\n"
            "注意要指明字体防止中文乱码。"
        )
    return (
        "用 moviepy 直接生成至少 6~10 个动画素材（绝对不少于 6 个，越多越好）。\n"
        "**不要执行任何 hyperframes 命令、也不要去看它的文档或源码**"
        "（本机已关掉，理由见系统提示）。\n"
        "用 ColorClip + TextClip + with_position 画；中文字体给字体文件路径、"
        "不要写字族名。"
    )


class _ToolErrorToMessage(AgentMiddleware):
    """把工具异常转成「回给模型的错误消息」，而不是抛出去终结整轮任务。

    deepagents 默认走 langgraph 的 ``_default_handle_tool_errors``，它**直接 raise**
    （``tool_node.py:391``）。后果实测过：agent 把沙箱外的绝对路径传给了
    ``read_file``，抛 ``ValueError: ... outside root directory``，
    整轮 20+ 步、几分钟的工作**当场全部作废**，模型连纠正的机会都没有。

    挂上这个中间件后，模型会收到一条错误文本，可以自己换条路重试
    （例如改用 ``execute('type ...')`` 读沙箱外的文件）。

    注意只吞 ``Exception``：``KeyboardInterrupt`` / ``SystemExit`` 仍然透传。

    ⚠️ 同步与异步**两个钩子都要实现**：本模块现在走 ``agent.stream()``（同步），
    但基类 AgentMiddleware 的 ``awrap_tool_call`` 默认实现是抛 ``NotImplementedError``，
    只写同步版的话，哪天有人把这里改成 ``astream``，上面那个「工具异常终结整轮」的
    老 bug 会原样回来，而且报错点变成中间件里的 NotImplementedError，极难联想到。
    """

    # 提示语固定成一句话：只告诉模型「换条路」，不复述它哪里错了 ——
    # 具体错因异常文本里已经有了，多写反而容易把模型引回同一个写法。
    _HINT = "请换一种方式重试，不要重复同样的调用。"

    @classmethod
    def _as_message(cls, request, exc: Exception) -> ToolMessage:
        """把异常包装成一条 ``status="error"`` 的 ``ToolMessage``。

        同步/异步两条钩子共用它，保证两条路径给出的错误文本与 id 完全一致。

        Args:
            request: langchain 的工具调用请求，``tool_call`` 里带 id 与参数。
            exc: 工具里冒出来的异常。

        Returns:
            回给模型看的 ToolMessage。
        """
        return ToolMessage(
            # 保留异常类型名：模型看到 "ValueError" 比看到一整段自然语言更容易
            # 判断「是我参数给错了」而不是「工具坏了」。
            content=f"[工具执行出错] {type(exc).__name__}: {exc}\n{cls._HINT}",
            # tool_call_id 必须回填真实的调用 id：langgraph 靠它把结果对回那次
            # tool_call，写错会报「tool_call_id 不匹配」并把这条消息丢掉。
            tool_call_id=request.tool_call["id"],
            # status="error" 是给下游/UI 看的标记，模型照样能读 content。
            status="error",
        )

    def wrap_tool_call(self, request, handler):
        """同步钩子：工具抛异常时转成错误消息，正常返回则原样透传。

        Args:
            request: 工具调用请求。
            handler: 真正的工具执行函数。

        Returns:
            工具的返回值，或 ``_as_message()`` 造出来的错误消息。
        """
        try:
            return handler(request)
        except Exception as exc:  # noqa: BLE001 —— 故意的：工具失败不该终结整轮
            # 只吞 Exception：KeyboardInterrupt / SystemExit 继承自
            # BaseException，仍然透传 —— 用户按 Ctrl+C 必须能真的停下来。
            return self._as_message(request, exc)

    async def awrap_tool_call(self, request, handler):
        """异步版（与同步版同逻辑，只是 await）。见类 docstring 里为什么必须写。

        Args:
            request: 工具调用请求。
            handler: 真正的工具执行函数（协程）。

        Returns:
            工具的执行结果，或 ``_as_message()`` 造出来的错误消息。
        """
        try:
            return await handler(request)
        except Exception as exc:  # noqa: BLE001 —— 同同步版
            return self._as_message(request, exc)


def _get_editor_agent():
    """构造（并缓存）视频剪辑 deepagent。

    构造要建 LLM 客户端、扫技能目录、装后端，成本不低，所以整个进程只做一次
    （模块级 ``_editor_agent`` 缓存）。改完 SKILL.md 或 .env 要立刻生效，
    用 ``reset_editor_agent()`` 把缓存丢掉。

    Returns:
        一个可 ``.stream(...)`` 的 deepagent（LangGraph 编译产物）。

    Raises:
        Exception: 构造失败时原样抛出（deepagents 没装、技能目录坏掉、模型名
            不合法等）。``node_edit_video`` 会接住它并回填
            ``deepagent 初始化失败: ...``，不让异常冒到视图层。
    """
    global _editor_agent
    # 缓存命中直接返回：构造一次要几秒量级，不该每次点「开始剪辑」都付一遍。
    if _editor_agent is not None:
        return _editor_agent

    # 按需导入：deepagents 会连带拉起 langgraph/pydantic 一整套，只有真要起
    # agent 时才值得付这个代价 —— 模块其余部分（路径清洗、自检前半段）不必陪它初始化。
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend

    # 温度用默认的 0.5；关键在 use_deepagent_model=True —— 它把编排模型单独
    # 拎出来（走 MEDIA_DEEPAGENT_MODEL，见 workflows/__init__.py 的缓存键说明），
    # 剪辑链路很长，要的是稳，不是跟业务节点抢同一个模型。
    model = get_model(temperature=0.5, use_deepagent_model=True)

    skills_dir = os.path.join(_PROJECT_ROOT, ".skills")
    cache_dir = settings.media.get_mashup_work_dir()
    # 沙箱根要自己先建出来：LocalShellBackend 启动时不会替我们建 root_dir，
    # 首次跑剪辑会直接报目录不存在。
    os.makedirs(cache_dir, exist_ok=True)

    # ---- 子进程环境：课案原有 + 本仓库实测补充 ----
    # 必须 copy：下面是往字典里塞变量喂给子进程，直接改 os.environ 会污染
    # 整个 DSH 进程（连本进程后面起的别的子进程都跟着变）。
    env = os.environ.copy()

    chromium = _find_chromium()
    if chromium:
        # 两个变量名都设一遍：HyperFrames 自己读 CHROMIUM_PATH，底层 puppeteer
        # 认的是 PUPPETEER_EXECUTABLE_PATH，哪个版本读哪个不好确定，索性都给。
        env["CHROMIUM_PATH"] = chromium
        env["PUPPETEER_EXECUTABLE_PATH"] = chromium
        print(f"[deepagents] 浏览器: {chromium}")
    else:
        # 只告警不中断：HyperFrames 本来就是可选链路，缺浏览器时 agent 会走
        # moviepy 画动画素材的降级分支。
        print("[deepagents] ⚠️ 未找到 Edge/Chrome，HyperFrames 渲染可能失败（会降级为 moviepy 动画）")

    # 禁 MSYS/Git Bash 路径自动转换（课案原有）
    # 三条一起设：只要子进程链里有一层 Git Bash，它就会把 /c/Users 改写成
    # C:\c\Users —— 那正是 normalize_path 要收拾的畸形路径。从源头掐掉比事后修省事。
    env["MSYS_NO_PATHCONV"] = "1"
    env["MSYS2_ARG_CONV_EXCL"] = "*"
    env["GIT_BASH_PATH_CONV"] = "none"

    # ⚠️ 本仓库实测：PATH 里的 python 是 Windows Store 占位符，静默无输出。
    # 把当前 venv 的 Scripts 目录顶到最前面，让 agent 敲的 python 是真的解释器。
    # 用 sys.executable（当前跑起来的解释器）反推，比猜 ".venv" 目录可靠，
    # 换虚拟环境也自动跟着走。
    venv_scripts = str(Path(sys.executable).resolve().parent)
    env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")

    # 注入项目路径，让 agent 引用而不是自己拼
    env["PROJECT_ROOT"] = _PROJECT_ROOT
    env["CACHE_DIR"] = cache_dir
    # WORK_DIR 是**沙箱内的相对路径名**，不是绝对路径：提示词里让模型用它拼
    # 输出路径，实际落在 root_dir 下面。改这里必须同步改下面
    # _build_editor_system_prompt("videos") 的实参，两处是同一个字符串。
    env["WORK_DIR"] = "videos"

    # virtual_mode=True：把 root_dir 当虚拟根，绝对路径一律判越界 ——
    #   「文件工具读不了沙箱外」那条规则就是从这来的（execute 不受它约束）。
    # timeout=300：单条 shell 命令的上限。库默认只有 120 秒
    #   （`DEFAULT_EXECUTE_TIMEOUT`），而 `npx hyperframes render` 实测要卡到
    #   301.8 秒 —— 提到同量级，宁可让一条命令**有界失败**，也别让整轮任务挂住。
    # max_output_bytes=5MB：库默认 100KB，且超限是**保留头部、砍掉尾部**
    #   （`output[:max]`）—— 而 moviepy/ffmpeg 的错误信息基本都在最后几行，
    #   用默认值等于把模型最需要的东西先丢掉（给大 bufsize 也是同一个理由）。
    # inherit_env=True + env=env：先继承父进程环境，再叠加上面改造过的那份。
    sandbox = LocalShellBackend(
        root_dir=cache_dir,
        virtual_mode=True,
        inherit_env=True,
        timeout=300,
        max_output_bytes=5_000_000,
        env=env,
    )

    # monkey-patch：清洗 Git Bash 路径 /c/Users/... → C:\Users\...
    # 否则 virtual_mode 会把它当成虚拟路径拼到沙箱下面（课案原有）
    # 先抓住**绑定**过的原方法再替换：补丁里要调它，直接在补丁内部按名字引用
    # 会无限递归。
    _orig_resolve = sandbox._resolve_path

    def _patched_resolve(key: str):
        """``_resolve_path`` 的替身：交给后端解析之前先清洗畸形路径。"""
        cleaned = normalize_path(key)
        # 落在沙箱**内**的绝对路径要改写成相对沙箱根的相对路径：
        # virtual_mode 只认相对路径，绝对路径一律被判越界。
        if os.path.isabs(cleaned):
            try:
                rel = os.path.relpath(cleaned, cache_dir)
                # rel 以 ".." 开头 = 确实在沙箱外面：**故意不改写**，原样交给
                # 后端去拒绝。这里做的是「清洗畸形路径」，不是「绕过沙箱」。
                if not rel.startswith(".."):
                    cleaned = rel
            except ValueError:
                pass  # 不同盘符，保持原样让后端正常拒绝
        return _orig_resolve(cleaned)

    sandbox._resolve_path = _patched_resolve  # type: ignore[method-assign]

    # ⚠️ 本仓库实测补的第 9 条（课案没有，**不加剪辑链路必崩**）：
    # `skills=` 的路径**必须相对 backend 的 root_dir**（见 deepagents 的
    # `create_deep_agent` 文档："skills are loaded from disk relative to the
    # backend's root_dir"）。课案原样传技能目录的**绝对路径**，
    # SkillsMiddleware 加载时走 `backend.ls(绝对路径)`，被 virtual_mode 判定为
    # "outside root directory" 抛 ValueError，agent 还没开始干活就整个中断。
    # 解法：用 CompositeBackend 把真实技能目录挂到虚拟路径 `/skills/`，
    # 文件读写经路由落到真实磁盘，shell 执行仍走沙箱
    # （CompositeBackend.execute 只认 default 后端）。
    routes: dict = {}
    # 技能目录不存在就**不挂路由**：硬挂一条指向不存在目录的路由，
    # 中间件加载技能时会抛出另一种更难定位的错。
    if os.path.isdir(skills_dir):
        routes["/skills/"] = FilesystemBackend(root_dir=skills_dir, virtual_mode=True)

    # default=sandbox：除 /skills/ 之外的一切路径（含 execute）都落到沙箱；
    # /skills/ 只是只读手册，写不进去（execute 也不认路由）。
    backend = CompositeBackend(default=sandbox, routes=routes)

    _editor_agent = create_deep_agent(
        model=model,
        # 传的是**虚拟路径**（"/skills/"），不是磁盘路径 —— 见上面第 9 条。
        # 没有技能目录时给空列表，否则 create_deep_agent 会去找不存在的技能。
        skills=list(routes) or [],
        backend=backend,
        # 工具异常转成回给模型的消息，而不是抛出去终结整轮（见类 docstring）。
        middleware=[_ToolErrorToMessage()],
        # 实参必须与上面 env["WORK_DIR"] 写成同一个字符串。
        system_prompt=_build_editor_system_prompt("videos"),
    )
    print("[deepagents] 视频剪辑 Agent 已初始化")
    print(f"[deepagents] 技能目录: {skills_dir} → 虚拟路径 /skills/")
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
    """把剪辑任务整包交给 deepagent。

    本工作流的第一个节点，也是唯一会调 LLM / 起子进程的节点：它把源视频、
    穿插素材、BGM、用户要求和全部踩坑约束拼成一段任务提示，交给 deepagent
    按 ``.skills/video-use/SKILL.md`` 自己执行。

    本节点**不抛异常**：任何失败都表现为「``output_video`` 留空 +
    ``editor_log`` 里一段中文失败串」，视图层据此显示原因。

    Args:
        state: 至少要有 ``input_video``；``edit_requirements`` 可以是 JSON
            字符串，也可以是一段纯文本要求（见下方兜底分支）。

    Returns:
        部分状态字典，键为 ``editor_log`` / ``steps`` / ``output_video`` /
        ``edit_start``。``output_video`` 为空串即表示本轮没解析到成品。
    """
    # 源视频路径来自表单，同样可能是 /c/Users 这种形态，先清洗再 abspath。
    # ⚠️ **必须先判原始串再 abspath**：`safe_abspath("")` 返回的是**当前工作目录**
    #    （`os.path.abspath("")` 的语义），永远非空 —— 直接在 abspath 的结果上判空，
    #    空表单会被当成「源视频 = CWD」，`os.path.exists(CWD)` 为真，于是照常起 agent。
    raw_video = (state.get("input_video") or "").strip()
    video = safe_abspath(raw_video) if raw_video else ""
    # 记开始时刻不是为了计时，而是给下面解析输出时当「本轮新产物」的时间基准。
    edit_start = time.time()

    # 源视频不可用就直接返回失败串，**连 agent 都不起**：
    # 起一次 agent（建客户端、扫技能）的开销远高于一次 os.path.exists。
    # 「没给路径」与「路径失效」分开报 —— 前者是用户还没选视频（页面提示他选），
    # 后者带上是哪条路径（看得出是被删了还是被移走了），处置方式完全不同。
    if not video:
        return {"editor_log": "未指定源视频：请先上传或填写源口播视频路径。",
                "output_video": "", "edit_start": edit_start}
    if not os.path.exists(video):
        return {"editor_log": f"源视频不存在: {video}", "output_video": "", "edit_start": edit_start}

    # ---- 解析剪辑要求 ----
    raw_req = state.get("edit_requirements", "")
    try:
        req = json.loads(raw_req) if raw_req else {}
    except Exception:  # noqa: BLE001 —— 传了非 JSON 就当纯文本要求
        # 视图层有两种传法：结构化的 JSON，或用户直接敲的一整段文字。
        # 后者解析必然失败，这里把它当成「额外要求」而不是报错。
        req = {"extra_requirements": raw_req}

    # `or []` 兜住显式的 null：JSON 里写 "materials": null 时 .get 拿到 None，
    # 后面 for 会 TypeError。
    materials = req.get("materials", []) or []
    # 三级兜底：本次表单 > 全局配置 MEDIA_BGM_PATH > 空串（空 = 本次不混音）。
    bgm_path = normalize_path(req.get("bgm_path") or settings.media.bgm_path or "")
    extra_req = req.get("extra_requirements", "")

    materials_desc = ""
    if materials:
        materials_desc = "\n\n## 穿插素材\n"
        for i, mp in enumerate(materials, 1):
            mp = normalize_path(mp)
            # 只按扩展名分「视频/图片」，分错也不影响结果 —— 这段描述是给模型的
            # 提示，不是校验（真正的校验在 moviepy 那一步）。
            kind = "视频" if os.path.splitext(mp)[1].lower() in (".mp4", ".mov", ".avi", ".webm") else "图片"
            # 写**绝对路径**：素材都在沙箱外，只有 execute 够得着。
            # 这里用 os.path.abspath 而不是 safe_abspath，两者等价（上面已
            # normalize 过）；video 那处用 safe_abspath，语义相同、写法不同。
            materials_desc += f"{i}. [{kind}] `{os.path.abspath(mp)}`\n"

    if bgm_path and os.path.exists(bgm_path):
        bgm_info = f"\nBGM: `{bgm_path}`"
    else:
        # BGM 不存在**不报错**：只在提示里写明「本次不混音」，剪辑照常做完。
        bgm_info = "\nBGM: 无（本次不混音）"

    # 任务提示就是整轮剪辑的「施工图」，两个细节不能动：
    #   · `{_material_howto()}` 必须来自那个开关函数 —— 曾经这里的课案原文与
    #     system_prompt 的降级分支互相打架，把首轮 E2E 拖到 301.8s 超时；
    #   · 下面字幕/蒙版两大段是踩出来的硬约束，删掉必然出现「字幕重复 + 底部被裁」。
    prompt = f"""## 剪辑任务

### 源视频
`{os.path.abspath(video)}`
{materials_desc}
{bgm_info}

### 用户要求
{extra_req if extra_req else "完整剪辑（始终保持和原视频像素一致）：转录 → 动画素材 → 排布穿插 → BGM → 字幕 → 渲染"}

{_material_howto()}
开头素材是在最上的图层、与原视频像素一致；其他素材高度需要是 1/3，放在原视频下面。
根据口播内容节奏在不同时间点穿插素材，有的上下排布：上边用原来的口播视频，
下边用生成的素材（覆盖口播视频下部分，空素材的地方要全屏用原视频，不能黑屏）。
注意不要拉伸图像，不要上下都用原视频。

=== 字幕约束（必须严格遵守，否则字幕重复 + 下半截消失） ===
字幕只能生成一次！正确流程：调 tools.audio_transcriber.transcribe_to_srt() 生成一个 SRT 文件
→ 用 SubtitlesClip('subtitle.srt') 加载 SRT，仅此一次！
严禁 for 循环手动创建 TextClip 来逐行添加字幕！严禁同时使用 SubtitlesClip 和 TextClip！
关键：必须用 make_textclip 生成字幕，并且把位置打在 **SubtitlesClip 本身**
（写进 make_textclip 的 position 不生效，frame_function 只取图像数组），
否则 SubtitlesClip 默认把字幕定位到合成帧最底部（y ≈ h + h//3，是素材区域），
字幕下半截会被隐式蒙版裁掉！
正确写法：
  FONT = 'C:/Windows/Fonts/msyh.ttc'   # 中文字体必须是**字体文件路径**
  def make_st(txt):
      return TextClip(text=txt, font=FONT, font_size=52,
          color='white', stroke_color='black', stroke_width=2,
          size=(w, None), method='caption')
  subtitles = (SubtitlesClip('subtitles.srt', encoding='utf-8', make_textclip=make_st)
               .with_position(('center', h * 0.62)))  # ← 落在原视频区域内，不是合成帧底部

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
    # steps_log 是给页面看的「步骤流水」；tool_count 单独计数 —— 一条 ai 消息里
    # 可能挂多个 tool_call，按消息数记会严重低估步数。
    # last_chunk 留最后一份完整 state：values 模式下每步都是全量快照，
    # 循环结束后它正好保留终态。
    steps_log: list = []
    tool_count = 0
    last_chunk: dict = {}

    try:
        agent = _get_editor_agent()
    except Exception as exc:  # noqa: BLE001 —— agent 构造失败也要给出中文说明
        import traceback

        # 打全栈是给排错留证据：构造失败基本是环境问题（依赖缺失、技能目录坏掉），
        # 不会出现在 agent 自己的日志里。
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
            # values 模式：每步吐出**完整 state 快照**，所以下面只看最后一条消息
            # 就够；代价是同一批历史消息会被反复看到，靠 steps_log 去重。
            stream_mode="values",
            # recursion_limit=150：默认的 50 实测不够 —— 一次「命令超时 → 换个
            # 写法重试」的降级就要吃掉十几个 super-step，29 步工具调用即撞上限
            # 并抛 GraphRecursionError。提到 150 才够跑完
            # 「转写 → 生成素材 → 渲染 → 合成」。
            config={"recursion_limit": 150},
        ):
            last_chunk = chunk
            msgs = chunk.get("messages", [])
            if not msgs:
                continue
            last = msgs[-1]

            if hasattr(last, "content") and last.content and hasattr(last, "type"):
                c = str(last.content).strip()
                # 去重是**必须**的：values 模式会重放历史消息，不去重会把同一条
                # AI 回复打印十几遍，真正的失败信息反而被埋掉。
                if c and c not in steps_log:
                    steps_log.append(c)
                    # 只认三种已知类型，其余（自定义消息对象等）统一给 📝 ——
                    # 不为了日志好看去猜类型。
                    emoji = {"ai": "🤖", "tool": "🔧", "human": "👤"}.get(
                        getattr(last, "type", "?"), "📝"
                    )
                    # 截断到 300 字符：一段 AI 长回复可能上千行，全打出来会把
                    # 后面真正的工具调用记录冲走。
                    print(f"  [{emoji}] {c[:300]}")

            # tool_calls 单独遍历：一条 ai 消息里可能一次发起多个调用，
            # 这里每个都算一步，方便对照 recursion_limit 的消耗。
            for tc in (getattr(last, "tool_calls", None) or []):
                tool_count += 1
                name = tc.get("name", "?")
                args = tc.get("args", {}) or {}
                steps_log.append(f"[tool] {name}")
                # 命令参数名有两种：execute 用 command，部分工具用 cmd；
                # 都取一下，取不到再按工具名分派打印。
                cmd = args.get("command", args.get("cmd", ""))
                if cmd:
                    # 命令本身也截断：模型有时会把整段脚本塞进 command。
                    print(f"  [{tool_count}] ⚡ {name}: {str(cmd)[:200]}")
                elif name in ("write_file", "write"):
                    print(f"  [{tool_count}] 📄 {name}: {args.get('file_path', args.get('path', '?'))}")
                elif name in ("read_file", "read"):
                    print(f"  [{tool_count}] 📖 {name}: {args.get('file_path', args.get('path', '?'))}")
                else:
                    print(f"  [{tool_count}] 🔨 {name}")

            # 工具返回只回显前 200 字符：完整输出常常是几 MB（ffmpeg/moviepy 的
            # 日志），打到控制台既慢又没用，真要看得去翻沙箱里的文件。
            if getattr(last, "type", "") == "tool":
                result = str(getattr(last, "content", ""))[:200]
                if result:
                    print(f"  [↩] → {result}")

    except GraphRecursionError:
        # 深度智能体在「命令一直失败」时会不断换写法重试，最终撞上步数上限。
        # 兜住它并给中文提示，而不是甩一个 traceback。
        # 这里**故意不 re-raise**：撞上限时沙箱里往往已经有能用的中间产物，
        # 下面 node_find_output 还要去捞；抛出去就什么都没了。
        print(
            "[剪辑] 达到步数上限仍未收敛：模型大概率在反复尝试当前平台不存在的命令。\n"
            "       请检查 system_prompt 里是否说明了 execute 跑的是 cmd.exe（本文件已说明）。"
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        # 同理不往上抛：把异常文本回填成 editor_log，页面至少能显示「为什么没出片」，
        # 而不是让 Streamlit 弹一整页红色 traceback。
        traceback.print_exc()
        return {
            "editor_log": f"剪辑出错: {exc}",
            "steps": json.dumps(steps_log, ensure_ascii=False),
            "output_video": "",
            "edit_start": edit_start,
        }

    # ---- 收尾：把沙箱里散落的中间文件收进 _scratch/ ----
    # 放在「解析输出路径」之前：收纳只动 _scratch/ 与那批中间文件通配，
    # 不碰 *.mp4 成品，所以不会把下面要找的输出挪走。
    tidy_steps = _tidy_work_dir()
    _warn_stray_outside()
    steps_log.extend(f"[tidy] {n}" for n in tidy_steps)

    # ---- 从最后一条消息里解析输出路径 ----
    output_video = ""
    last_msg = ""
    # 从**最后一条**往前找：agent 通常把成品路径写在收尾那句回复里，倒序最快撞上。
    for m in reversed(last_chunk.get("messages", [])):
        c = str(getattr(m, "content", ""))
        if not c:
            continue
        last_msg = c
        # 按空白切词：路径可能被 markdown 反引号、逗号或换行包着，先粗切再逐个剥壳。
        for word in c.replace(",", " ").replace("\n", " ").split():
            # ⚠️ 顺序很关键：**先按扩展名过滤，再调 normalize_path**。
            #    normalize_path 只要改动了就打印 [PathFix]，而它会无差别地把
            #    文本里的 "/" 换成 "\"。对整段散文逐词调用，实测刷出几百行噪音
            #    （'https://cdn...' → 'https:\\cdn...'、'1/3' → '1\3'），
            #    把真正的日志全埋掉。
            word = word.strip("`\"'[]()")   # 剥掉 markdown / 标点外壳，剩下的才是候选路径
            if not word.lower().endswith(".mp4"):
                continue
            word = normalize_path(word)
            if os.path.isfile(word):
                try:
                    # 两道判据缺一不可：
                    #   · mtime > edit_start - 10：只认**本轮**产出的文件，减 10 秒
                    #     是给文件系统时间戳精度、以及「agent 在记时前后落盘」留的容差；
                    #   · 大小 != 源视频：挡掉 agent 把源视频复制/改名后当成成品的假阳性。
                    if os.path.getmtime(word) > edit_start - 10 and \
                       os.path.getsize(word) != os.path.getsize(video):
                        output_video = word
                        break
                except OSError:
                    # 文件在判定的这一瞬间被删/被占用都会抛 OSError，
                    # 跳过这个候选继续找，不因为一个坏候选放弃整次解析。
                    continue
        # 倒序遍历里拿到**第一个**可用路径就收工（也就是被最后提到的那个）。
        if output_video:
            break

    return {
        # 截断到 2000 字符：editor_log 会原样渲染进 Streamlit 页面，
        # 整段 agent 输出会把页面撑爆。
        "editor_log": last_msg[:2000] if last_msg else "deepagent 完成",
        "steps": json.dumps(steps_log, ensure_ascii=False),
        # 空串 = 没解析到成品，交给 node_find_output 兜底。
        "output_video": output_video,
        "edit_start": edit_start,
    }


# ==========================================================================
# 收尾：清理散落的中间文件（课案 3520-3532 做的是同一件事）
# ==========================================================================
# agent 的 execute 跑的是真实 cmd，能用任意绝对路径；它很容易在中转时把
# .wav / .txt / concat_list.txt / speedup_*.mp4 之类丢在沙箱根。
# 课案的做法是把它们搬进 CACHE_DIR —— 但课案的 CACHE_DIR 就是沙箱根，
# 等于原地搬，所以本项目按「搬进沙箱内的 _scratch/ 子目录」实现。
#
# ⚠️ 只动沙箱根（MEDIA_MASHUP_WORK_DIR）：它是本项目自己的运行时目录、
#    已被 .gitignore 覆盖，搬动零风险。
#    沙箱**外面**（仓库根、Media_Agent/）一律不碰 —— 那里的 `*.txt` 通配会误伤
#    `docs/课案全文提取.txt` 这类真文件。对外面只打印一行提醒，让人自己决定。
_STRAY_PATTERNS = ("*.wav", "*.txt", "concat_list*", "speedup*", "assembled*",
                   "final_with_bgm*", "temp_*", "tmp_*")
# 这些是流程产物，不是"散落文件"，不能收走
_STRAY_KEEP = {"mashup_final.mp4", "final_with_bgm.mp4"}
# 上面的通配都是沙箱根**一层**（glob 不递归），所以搬进 _scratch/ 之后
# 不会再被扫到 —— _tidy_work_dir() 可以重复调用而不自我搬运。


def _tidy_work_dir() -> list:
    """把沙箱根上散落的中间文件收进 ``<work_dir>/_scratch/``。

    只搬、**绝不删除** —— 万一是要紧东西，搬进工作目录里也还能找回来。

    Args:
        无。沙箱根取自 ``settings.media.get_mashup_work_dir()``，
        与 ``_get_editor_agent()`` 是同一个配置项。

    Returns:
        被搬走的文件名列表（供日志与自检使用）。
    """
    import glob as _glob
    import shutil

    cache_dir = settings.media.get_mashup_work_dir()
    scratch = os.path.join(cache_dir, "_scratch")
    moved = []
    for pattern in _STRAY_PATTERNS:
        for path in _glob.glob(os.path.join(cache_dir, pattern)):
            # isfile 挡掉同名目录；_STRAY_KEEP 是白名单 ——
            # 成品即使命中通配也一个都不动。
            if not os.path.isfile(path) or os.path.basename(path) in _STRAY_KEEP:
                continue
            try:
                # 建目录放在循环里而不是函数开头：一个散落文件都没有时，
                # 不该凭空造出一个空的 _scratch/ 目录。
                os.makedirs(scratch, exist_ok=True)
                dest = os.path.join(scratch, os.path.basename(path))
                # 重名就加时间戳后缀：既然「绝不删除」，同名旧文件只能靠改名共存。
                if os.path.exists(dest):
                    dest = f"{dest}.{int(time.time())}"
                shutil.move(path, dest)
                moved.append(os.path.basename(path))
            except OSError as exc:  # noqa: BLE001 —— 收尾失败不该影响交付
                # 这里一条 except 就能覆盖「搬动失败」的全部形态：
                # shutil.Error 继承自 OSError（磁盘满、文件被别的进程占用等）。
                # 只打印不重抛 —— 成品已经渲出来了，收尾失败不该让它算作失败。
                print(f"[剪辑] 收尾跳过 {os.path.basename(path)}: {exc}")
    if moved:
        print(f"[剪辑] 已收纳 {len(moved)} 个散落中间文件 → {scratch}")
    return moved


def _warn_stray_outside() -> list:
    """沙箱**外面**散落的中间文件只提醒、不动手。返回文件名列表。

    为什么要单独有这么一半：agent 的 execute 跑的是真实 cmd，能写任意绝对
    路径，偶尔会把中间文件丢在仓库根或 ``Media_Agent/`` 下。但那些地方的
    ``*.txt`` 通配会误伤 ``docs/课案全文提取.txt`` 这类真文件 ——
    **删用户文件是不可逆的**，所以这里只打印清单，让人自己决定。

    Args:
        无。

    Returns:
        沙箱外命中的疑似散落文件绝对路径列表（同样供日志与自检使用）；
        本函数只告警，不移动、不删除。
    """
    import glob as _glob

    found = []
    # 两个目录：仓库根（_PROJECT_ROOT 的父目录）与 Media_Agent/ 本身。
    # 用 dirname 现算而不是硬编码上层路径 —— 项目整体换位置也不会失配。
    for base in (os.path.dirname(_PROJECT_ROOT.rstrip("\\/")), _PROJECT_ROOT):
        # 这里的通配比 _STRAY_PATTERNS **窄**：去掉了 *.txt 与 temp_* / tmp_* /
        # final_with_bgm* —— 仓库这两层里这些名字太多（docs/ 下就有一堆真 .txt），
        # 列出来基本都是误报。宁可漏报也不刷屏。
        for pattern in ("*.wav", "concat_list*", "speedup*", "assembled*"):
            for path in _glob.glob(os.path.join(base, pattern)):
                if os.path.isfile(path):
                    found.append(path)
    if found:
        # 最多列 10 条：刷屏的告警等于没有告警。
        print(
            "[剪辑] ⚠️ 沙箱外发现疑似散落中间文件（**未自动清理**，请自行确认后删除）：\n"
            + "\n".join(f"        {p}" for p in found[:10])
        )
    return found


# ==========================================================================
# 节点 2：兜底查找（agent 没报告路径时，按约定位置找）
# ==========================================================================
def node_find_output(state: MashupState) -> dict:
    """兜底查找输出文件。

    课案这里是按 ``.cache/videos/mashup_final.mp4`` 找的，
    但它两个文件的 CACHE_DIR 不一致导致扫不到。本项目统一用
    ``MEDIA_MASHUP_WORK_DIR``，所以这里能对上。

    查找顺序：agent 报告过的路径 → 三个约定落点 → 按修改时间扫目录。

    ⚠️ 后两步是**猜**，猜出来的候选必须比本轮开始还新（``state["edit_start"]``，
    ``node_edit_video`` 在每条返回路径上都写了它）：源视频有效、agent 又没自报路径时，
    沙箱里可能正躺着**上一轮**的 ``mashup_final.mp4`` —— 只看「源视频此刻在不在」
    是拦不住它的（那判的是「这轮 agent 有没有机会跑」，不是「有没有产出」）。
    agent 自己报出来的路径是我们问到的明确答复、不是猜的，照旧只做 ``!= inp`` 判断。

    Args:
        state: ``output_video`` 为 ``node_edit_video`` 解析出的候选路径
            （可能是空串）。

    Returns:
        部分状态字典：找到时 ``{"output_video": <绝对路径>}``；
        一个都没找到时返回**空字典** —— LangGraph 里空字典表示「本节点不改动
        状态」，不是错误（视图层看到 ``output_video`` 仍为空，才知道这轮没出片）。
    """
    # 与 `node_edit_video` 同样**先取原始串再 abspath**：`safe_abspath("")` 会返回
    # 当前工作目录（永远非空），拿它当"源视频路径"去做下面那几处 `!= inp` 比对是错的 ——
    # 该字段没给时语义应是「不与任何候选相等」，而不是「碰巧等于 CWD 的候选被排除」。
    _raw_in = (state.get("input_video") or "").strip()
    inp = safe_abspath(_raw_in) if _raw_in else ""

    # 源视频没给、或给的路径已失效 ⇒ `node_edit_video` 提前退回了，agent 这轮**一步都没跑**。
    # 但图是直边（edit → find），find 照旧会执行；不在这里拦住的话，它按约定落点一捞
    # 就会把**上一轮**留在沙箱里的成片当成本轮产物交回去 ——
    # 用户会同时看到「源视频不存在」的失败日志和一个能下载的视频，分不清哪个是真的。
    # 注意这只挡「没有有效源视频」这一种情况；源视频有效但 agent 中途失败时，
    # 下面的约定落点兜底仍然照旧（那是防 agent 忘报路径的有意设计，不能一起关掉）。
    if not _raw_in or not os.path.isfile(inp):
        print("[剪辑] 源视频缺失或已失效，跳过产物查找（不拿旧成片顶替）")
        return {}

    out = normalize_path(state.get("output_video", ""))
    # agent 自己报的路径优先采信；`!= inp` 是防它把**源视频**当成品报回来
    # （路径与输入完全相同，必然不是产物）。
    # 这里**不做新鲜度过滤**：这是 agent 的明确答复，不是我们猜的 —— 它报什么就信什么
    # （内容对不对由页面上的人看，不由 mtime 判）。
    if out and os.path.isfile(out) and safe_abspath(out) != inp:
        return {"output_video": safe_abspath(out)}

    # 下面两步都是「猜」，猜出来的候选必须比**本轮开始时刻**新，否则它只可能来自上一轮。
    # 判据取自 `edit_start`：`node_edit_video` 在**每条**返回路径上都写了它（含两条失败
    # 路径），所以只要图跑过 editor，这个键就在。
    # 旧判据只看「源视频此刻在不在」（见上面的守卫）：源视频有效、agent 又没自报路径时
    # 它照样放行 —— 而那正是兜底查找存在的**唯一**理由，于是上一轮的成片就顶替了本轮产物。
    # 1 秒容差：照顾文件系统时间戳粒度，以及文件落盘与取时间之间的微小差。
    _edit_start = state.get("edit_start") or 0.0
    if not _edit_start:
        # 直接函数调用（不在图里跑）时没有「本轮」可言，跳过新鲜度过滤 ——
        # 否则会把「不传 edit_start 的自检用例」全部打翻。
        print("[剪辑] 未拿到 edit_start，跳过产物新鲜度过滤")

    def _is_stale(cand: str) -> bool:
        """候选是不是**非本轮产物**（早于 ``edit_start``，或读不到修改时间）。

        ``edit_start`` 缺失/为 0 时一律返回 ``False`` —— 那种场合没有「本轮」，
        不该过滤（见上）。读不到 mtime 也算不新鲜：宁可这轮报「没出片」，
        也不拿一个来路不明的文件顶替 —— 交错了用户在页面上看不出来。
        """
        if not _edit_start:
            return False
        try:
            _mt = os.path.getmtime(cand)
        except OSError:
            print(f"[剪辑] 跳过候选（读不到修改时间，无法确认是本轮产物）: {cand}")
            return True
        if _mt < _edit_start - 1.0:
            print(f"[剪辑] 跳过候选（修改时间比本轮开始早 {_edit_start - _mt:.0f}s，"
                  f"是旧产物）: {cand}")
            return True
        return False

    cache_dir = settings.media.get_mashup_work_dir()
    # 三个已知落点：直接落在沙箱根、经 BGM 步骤后的改名版本，以及模型把
    # WORK_DIR 拼成 videos/ 子目录的情况。
    candidates = [
        os.path.join(cache_dir, "mashup_final.mp4"),
        os.path.join(cache_dir, "final_with_bgm.mp4"),
        os.path.join(cache_dir, "videos", "mashup_final.mp4"),
    ]
    for cand in candidates:
        cand = normalize_path(cand)
        if os.path.isfile(cand) and safe_abspath(cand) != inp:
            if _is_stale(cand):
                continue      # 上一轮的成片，不能当本轮的产物交回去
            return {"output_video": safe_abspath(cand)}

    # 最后再按修改时间扫一遍目录。
    # ⚠️ 限深度 3：上面三条候选已覆盖已知落点，这里是**超额兜底**。
    #    无限制递归会把 .cache/mashup/** 里任意层级的 ≥100KB mp4 都当成品
    #    （包括 agent 自己放的中转文件、甚至被复制进沙箱的源视频），反而更容易选错。
    _MAX_DEPTH = 3
    # 深度按 os.sep 的个数**之差**算（不用 Path.parts）：基准目录自带的那部分
    # 前缀在相减时自然抵消，盘符与 UNC 前缀都不用特判。
    _base_depth = cache_dir.rstrip("\\/").count(os.sep)
    newest = ""
    newest_mtime = 0.0
    _stale_skipped = 0
    for root, dirs, files in os.walk(cache_dir):
        if root.rstrip("\\/").count(os.sep) - _base_depth >= _MAX_DEPTH:
            # ⚠️ 必须**原地**清空 dirs（写成 `dirs = []` 无效）：
            #    os.walk 靠读这个列表对象的内容决定下不下钻，重新绑定局部名字
            #    只是让本函数这边指向新列表，剪枝一点作用都没有。
            dirs[:] = []          # 到了深度上限就不再往下走
        for f in files:
            if not f.lower().endswith(".mp4"):
                continue
            fp = os.path.join(root, f)
            try:
                # 100KB 下限：成品 mp4 远大于它，用来滤掉缩略图、碎片文件。
                if os.path.getsize(fp) < 100 * 1024:
                    continue
                # 再挡一次源视频：它也可能被复制进沙箱。
                if safe_abspath(fp) == inp:
                    continue
                mt = os.path.getmtime(fp)
            except OSError:
                continue
            # 扫描出来的同样只是**候选**，一样要求是本轮产物（旧 mp4 不能因为
            # 「修改时间最大」就被选中 —— 上一轮的成片、随源视频复制进来的素材
            # 都可能比本轮的中间产物更新）。
            # 这里不调 `_is_stale`：它每个候选打一行日志，而这一段是**整目录扫描**，
            # 沙箱里躺着几十个旧 mp4 时日志会刷屏。改成计数，最后一句话汇总。
            if _edit_start and mt < _edit_start - 1.0:
                _stale_skipped += 1
                continue
            # 取修改时间最新的那个：最终成品总是最后落盘的。
            if mt > newest_mtime:
                newest, newest_mtime = fp, mt

    if newest:
        # 兜底命中说明 agent 没能报告路径，日志里留个痕方便定位。
        print(f"[剪辑] 兜底找到最新产出: {newest}")
        return {"output_video": safe_abspath(newest)}
    if _stale_skipped:
        # 「为什么没出片」的排查线索：扫描确实跑过，只是扫到的全是旧产物。
        print(f"[剪辑] 兜底扫描跳过 {_stale_skipped} 个旧 mp4（修改时间早于本轮开始）")
    return {}


# ==========================================================================
# 构图
# ==========================================================================
# 两个节点一条直线：edit 失败（output_video 为空）时也照样走 find ——
# 因为「agent 没报告路径」与「agent 根本没跑起来」在 find 眼里是同一件事，
# 都值得去约定位置捞一把。
builder = StateGraph(MashupState)
builder.add_node("edit", node_edit_video)
builder.add_node("find", node_find_output)

builder.add_edge(START, "edit")
builder.add_edge("edit", "find")
# find 之后直接结束（没有条件边）：找不到就返回 {}，整张图照样正常收尾。
builder.add_edge("find", END)

# 导入期就编译：编译只做静态检查，不建连接、不调模型，几毫秒的事；
# 好处是 views 层 import 本模块后拿到手就能 invoke。
mashup_graph = builder.compile()


def run_mashup(input_video: str, edit_requirements: str = "", **_kwargs) -> dict:
    """运行视频剪辑工作流。

    本文件对外的包装入口：``views/mashup.py`` 调的就是它（只传 ``input_video``
    与 ``edit_requirements`` 两个位置参数），页面不必知道 state 里有哪些键名叫什么，
    也不必自己拼初始 state —— 那是本模块的内部契约。

    Args:
        input_video: 源口播视频路径。
        edit_requirements: JSON 字符串，形如
            ``{"materials": [...], "bgm_path": "...", "extra_requirements": "..."}``。
        **_kwargs: 吞掉多余的具名参数。当前没有任何调用方用到它（页面只传上面两个
            位置参数）—— 留着是为了改签名时不必同步改调用方：**多传一个具名参数
            不会 ``TypeError``**，把「调用点崩在签名上」降级成「参数被忽略」。

    Returns:
        ``{"input_video","output_video","editor_log","steps",...}``；
        ``output_video`` 为空串表示本轮没出片（原因在 ``editor_log`` 里）。

    Note:
        同步阻塞调用：一轮剪辑要几分钟到十几分钟，调用方会一直等到它结束。
        两个节点都自己兜了异常并回填失败串，所以这里正常不会抛。
    """
    return mashup_graph.invoke({
        "input_video": input_video,
        "edit_requirements": edit_requirements,
    })


if __name__ == "__main__":
    # 自检必须**离线**（不启动 agent、不调 LLM），跑一次约 4 秒，可以随手跑。
    # 覆盖的每一项都是「改坏了会静默出错」的点：路径清洗、失败分支、
    # /skills/ 后端接线，以及两个 monkey-patch 的回归。
    # ⚠️ 这里**不再**重做 path 引导：模块顶部已用 `_PROJECT_ROOT` 插过 `sys.path`，
    #    而且本文件的项目内 import 早就在模块级执行完了 —— 再插一次是抄模板留下的冗余，
    #    删掉不影响任何分支（下面第 4 项仍用模块顶部的 `Path` 与 `_PROJECT_ROOT`）。
    print("=== 视频剪辑工作流自检（离线，不启动 agent）===")

    # 0) 图结构：两个节点都在，且确实编译过（编译失败的话 import 就炸了）。
    g = mashup_graph.get_graph()
    nodes = sorted(g.nodes)
    assert "edit" in nodes and "find" in nodes, nodes
    print(f"  图节点: {nodes}  OK")

    # 1) 路径清洗（课案的 normalize_path，这是最容易被改坏的一段）
    #    三个期望值都是课案里真实出现过的畸形形态；动 normalize_path 前先看这里红不红。
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
    #    这条直接决定页面显示成什么样：抛异常就是整页红色 traceback，
    #    回填中文串才是「源视频不存在: ...」这样一句能看懂的话。
    r = run_mashup("不存在的视频.mp4")
    assert "源视频不存在" in r.get("editor_log", ""), r
    print("  源视频缺失处理             OK")

    # 2b) 空路径**不许**被当成「源视频 = 当前工作目录」（本轮修的 bug）
    #     改前：`safe_abspath("")` 返回 CWD，`os.path.exists(CWD)` 为真 ⇒ 判据落空、
    #     照常起 agent 去剪一个目录。所以这条用例的核心不是"返回了提示"，
    #     而是**根本没去构造 agent**：把 `_get_editor_agent` 换成「一被调就抛」的桩
    #     （它是本节点唯一的重入口），再断言提示里不出现 CWD。
    #     ⚠️ 用**临时空目录**当沙箱：本图是 edit → find 一条直线，`node_find_output`
    #     照着设计会去约定位置捞成品 —— 上轮真跑出来的 `mashup_final.mp4` 会被它捞到，
    #     `output_video` 就跟本用例无关了。测试不能依赖共享目录的状态（同第 3 项）。
    #     state 只给 `input_video` 一个键也顺带验了：删掉 `edit_style` 之后，
    #     `mashup_graph.invoke` 仍能收下这个最小输入（`total=False` 的用处）。
    import tempfile

    _saved_work_dir = settings.media.mashup_work_dir
    _real_get_agent = _get_editor_agent
    _agent_built: list = []

    def _no_agent():
        """占住 `_get_editor_agent` 的位置：被调到就说明本节点不该往下走。"""
        _agent_built.append(1)
        raise AssertionError("空源视频路径时不该构造 agent")

    try:
        with tempfile.TemporaryDirectory(prefix="mashup_selfcheck_empty_") as _empty_dir:
            settings.media.mashup_work_dir = _empty_dir
            _get_editor_agent = _no_agent  # noqa: F841
            r_empty = mashup_graph.invoke({"input_video": ""})
    finally:
        # 无论断言过没过都要还原：改的是全局配置，留着会影响同一进程里的后续用例。
        settings.media.mashup_work_dir = _saved_work_dir
        _get_editor_agent = _real_get_agent

    assert _agent_built == [], "空路径时仍然去构造 agent 了（判据没拦住）"
    assert "未指定源视频" in r_empty.get("editor_log", ""), r_empty
    assert r_empty.get("output_video", "") == "", r_empty
    assert os.getcwd() not in r_empty.get("editor_log", ""), r_empty["editor_log"]
    print("  空源视频路径处理           OK（不构造 agent，也没拿 CWD 当视频）")

    # 2c) State 契约：`edit_style` 是课案遗留的**死字段**（全仓 0 处读），已删除。
    #     留一条断言钉住：谁再把它加回来（或视图层又按具名传参），这里先红。
    assert "edit_style" not in MashupState.__annotations__, MashupState.__annotations__
    print("  MashupState 无 edit_style  OK")

    # 3) 兜底查找：两个方向都要走通 —— 目录里没有成品时返回空；成品落在
    #    **约定落点之外**时，`os.walk` 那一段能把它扫出来。
    #    ⚠️ 源视频必须是**有效文件**：源视频缺失时上面的守卫会在更早处 `return {}`，
    #       这一段就成了恒真（改前正是如此：临时目录与配置改写全是死设置，
    #       把整段兜底代码换成 `return {}` 它照样打印 OK）。
    #    ⚠️ 还必须指向一个**临时目录**（每次新建，不带上一轮的残留）：这条用例直接跑在
    #       共享沙箱里会翻车 —— 真跑过一次剪辑后沙箱里就躺着 mashup_final.mp4，
    #       兜底查找会（正确地）把它找出来，断言就挂了。测试不应该依赖共享目录的状态。
    #    （`tempfile` 在 2b 那条用例里已经 import 过了。）
    _saved_work_dir = settings.media.mashup_work_dir
    try:
        with tempfile.TemporaryDirectory(prefix="mashup_selfcheck_") as _empty:
            _src_ok = os.path.join(_empty, "src.mp4")
            with open(_src_ok, "wb") as _f:
                _f.write(b"\x00" * 2048)   # 有效源视频，且小于 100KB 下限、不会被当成品
            settings.media.mashup_work_dir = _empty
            _t_scan = time.time()
            assert node_find_output(
                {"input_video": _src_ok, "output_video": "", "edit_start": _t_scan}
            ) == {}, "目录里没有任何成品时不该返回路径"
            # 非约定落点：只放在子目录里，三个候选名单都够不着，只有扫描那段找得到。
            _nested = os.path.join(_empty, "renders", "shot.mp4")
            os.makedirs(os.path.dirname(_nested), exist_ok=True)
            with open(_nested, "wb") as _f:
                _f.write(b"\x00" * (200 * 1024))   # 过 100KB 下限
            _scanned = node_find_output(
                {"input_video": _src_ok, "output_video": "", "edit_start": _t_scan}
            )
            assert safe_abspath(_scanned.get("output_video", "")) == safe_abspath(_nested), \
                f"约定落点之外的成品应当被扫描找到，实际 {_scanned}"
    finally:
        # 无论断言过没过都要还原配置，否则会影响同一进程里的后续用例。
        settings.media.mashup_work_dir = _saved_work_dir
    print("  兜底查找空目录/非约定落点  OK")

    # 3b) 源视频缺失时**不许**把沙箱里上一轮留下的成片当成本轮产物交回去。
    #     这条是修完 bug 补的：图是直边 `edit → find`，edit 提前退回了 find 照样跑，
    #     它会按约定落点捞到旧的 `mashup_final.mp4` —— 页面于是同时显示
    #     「源视频不存在」和一个能下载的视频，用户分不清哪个是真的。
    #     ⚠️ 必须**真的放一个旧成片**进去，否则这条与上面的 3) 没有区别、什么都测不出来。
    _saved_work_dir = settings.media.mashup_work_dir
    try:
        with tempfile.TemporaryDirectory(prefix="mashup_selfcheck_stale_") as _stale_dir:
            _stale_final = os.path.join(_stale_dir, "mashup_final.mp4")
            _src = os.path.join(_stale_dir, "src.mp4")
            for _p in (_stale_final, _src):
                with open(_p, "wb") as _f:
                    _f.write(b"\x00" * 2048)   # 内容无所谓，只要 `os.path.isfile` 为真
            settings.media.mashup_work_dir = _stale_dir

            # ① 源视频字段为空、② 源视频路径已失效 —— 两种都算「这轮 agent 一步都没跑」
            assert node_find_output({"input_video": "", "output_video": ""}) == {}, \
                "源视频为空时不该交回旧成片"
            assert node_find_output(
                {"input_video": os.path.join(_stale_dir, "nope.mp4")}
            ) == {}, "源视频路径失效时不该交回旧成片"

            # ③ 源视频**有效**、但成片是上一轮留下的（mtime 早于本轮开始）→ 仍然不许交回。
            #    判据是**新鲜度**（`edit_start`），不是「源视频此刻在不在」：
            #    后者在「源视频有效 + agent 没自报路径」时同样放行，而那种情况正是
            #    兜底查找存在的唯一理由 —— 上一版的守卫只堵了半扇门。
            _es = time.time()
            _old_mt = _es - 3600
            os.utime(_stale_final, (_old_mt, _old_mt))
            _got_old = node_find_output(
                {"input_video": _src, "output_video": "", "edit_start": _es})
            assert _got_old == {}, f"旧成片（mtime 早于 edit_start）不该被返回: {_got_old}"

            # ④ 反过来：**本轮新写的**成片必须照旧能被兜底找回来
            #    （那是防 agent 忘报路径的有意设计，不能连它一起关掉）。
            #    两条一起才说明新鲜度判据不是「一律拒绝」。
            _fresh_final = os.path.join(_stale_dir, "videos", "mashup_final.mp4")
            os.makedirs(os.path.dirname(_fresh_final), exist_ok=True)
            with open(_fresh_final, "wb") as _f:
                _f.write(b"\x00" * 2048)
            _got_new = node_find_output(
                {"input_video": _src, "output_video": "", "edit_start": _es})
            assert safe_abspath(_got_new.get("output_video", "")) == safe_abspath(_fresh_final), \
                f"本轮新写的成片必须被兜底找回，实际 {_got_new}"
    finally:
        settings.media.mashup_work_dir = _saved_work_dir
    print("  产物新鲜度两个方向          OK（旧成片不交回 / 本轮成片照旧找回）")

    # 4) SKILL.md 存在且格式正确（DeepAgents 靠 frontmatter 识别技能）
    #    frontmatter 写坏不会报错，只会让技能静默不生效 —— 所以必须断言。
    skill = Path(_PROJECT_ROOT) / ".skills" / "video-use" / "SKILL.md"
    assert skill.is_file(), f"技能文件缺失: {skill}"
    head = skill.read_text(encoding="utf-8")[:200]
    assert head.startswith("---"), "SKILL.md 必须以 YAML frontmatter 开头"
    assert "name: video-use" in head, head
    assert "description:" in head, head
    print(f"  SKILL.md 格式              OK  ({skill.stat().st_size} 字节)")

    # 5) system_prompt 覆盖了几条关键约束（这些丢了必然出问题）
    sp = _build_editor_system_prompt()
    for must in ("cmd.exe", "moviepy.video.tools.subtitles",
                 "h + h//3", "transcribe_to_srt", "msyh.ttc",
                 # 这两条是实测踩出来后补的：不写清楚，agent 会拿绝对路径去喂
                 # read_file，抛 outside root directory 并把整轮任务打断。
                 "outside root directory", "/skills/"):
        assert must in sp, f"system_prompt 缺少关键约束: {must}"
    # 字体：moviepy 2.x 把 font 当**文件**加载，写字体族名必崩。实测钉一下。
    _f = "C:/Windows/Fonts/msyh.ttc"
    if not os.path.isfile(_f):
        # 字体属于机器环境、不是代码问题，缺了就跳过而不是判失败。
        print(f"  ⚠️ 中文字体缺失，跳过字体检查: {_f}")
    else:
        from moviepy.video.VideoClip import TextClip as _TC
        # 真正建一个 TextClip：`method='caption'` + `size=(400, None)` 是
        # 提示词里教模型的写法，这里顺带证明那套写法在本机 moviepy 版本上成立。
        assert _TC(text="中文", font=_f, font_size=52, color="white",
                   size=(400, None), method="caption").size[0] == 400
        print(f"  中文字体（字体文件）        OK  ({_f})")
    # 素材生成方式必须跟着配置走，否则关掉 HyperFrames 也白关。
    # ⚠️ 两条路径都要断言：system_prompt **和任务提示**（后者见 node_edit_video）。
    #    曾经只切了 system_prompt，任务提示仍写「先用 HyperFrames…」，
    #    同一次调用里两条相反指令打架 —— 实测导致 agent 白试 npx 到 300s 超时。
    if settings.media.mashup_use_hyperframes:
        assert "HyperFrames 的时间预算" in sp, "开着 HyperFrames 却没给时间预算约束"
        assert "先用 HyperFrames" in _material_howto(), \
            "MEDIA_MASHUP_USE_HYPERFRAMES=true 但任务提示没让用 HyperFrames"
    else:
        assert "禁止碰 HyperFrames" in sp, \
            "MEDIA_MASHUP_USE_HYPERFRAMES=false 但 system_prompt 没关掉 HyperFrames"
        assert "先用 HyperFrames" not in _material_howto(), \
            "MEDIA_MASHUP_USE_HYPERFRAMES=false 但任务提示仍要求先用 HyperFrames（与 system_prompt 打架）"
        assert "moviepy" in _material_howto(), "关闭 HyperFrames 后任务提示没说改用 moviepy"
    print(f"  system_prompt 关键约束     OK  (HyperFrames={'on' if settings.media.mashup_use_hyperframes else 'off'})")

    # 6) 技能目录的后端接线 —— 这条是**真实踩过的崩溃**，必须有回归用例。
    #    症状：skills= 传绝对路径时 SkillsMiddleware 加载即抛
    #    "Path ... outside root directory: <沙箱根>"，agent 一步都没跑起来。
    #    这里用同一套 CompositeBackend 接线（不建 agent、不调 LLM）复现并验证。
    from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend

    _cache = settings.media.get_mashup_work_dir()
    os.makedirs(_cache, exist_ok=True)
    _skills = os.path.join(_PROJECT_ROOT, ".skills")
    _sandbox = LocalShellBackend(root_dir=_cache, virtual_mode=True)
    _composite = CompositeBackend(
        default=_sandbox,
        routes={"/skills/": FilesystemBackend(root_dir=_skills, virtual_mode=True)},
    )
    # 两个方向都要验：先能 ls 出技能名（路由挂上了），
    # 再能真的经虚拟路径读到 SKILL.md 的字节（内容也通了）。
    _ls = _composite.ls("/skills/")
    _names = [e["path"] for e in _ls.entries]
    assert any("video-use" in n for n in _names), f"/skills/ 路由没挂上: {_names}"
    _dl = _composite.download_files(["/skills/video-use/SKILL.md"])
    assert _dl and _dl[0].content, f"经虚拟路径读 SKILL.md 失败: {_dl}"
    print(f"  /skills/ 虚拟路由          OK  (读到 {len(_dl[0].content)} 字节)")

    # 7) Windows 管道死锁补丁 —— 防「execute 卡死且超时失效」的回归用例。
    #    构造「直接子进程立刻退出、孙进程继续攥着输出」的形态
    #    （`start /b` 就是典型），这正是 npx -> node -> 浏览器 那一串的形状。
    #    标准库实测要 59.8 秒才从 communicate() 出来；补丁走临时文件，秒回。
    assert subprocess.run is _patched_run, "subprocess.run 补丁被覆盖了"
    _t0 = time.time()
    try:
        # `ping -n 60` ≈ 59 秒寿命：刻意让它比标准库那 59.8 秒的实测值活得久，
        # 这样「超时没按时触发」才会暴露出来；换条短命令孙进程先死，测试就失去意义。
        _patched_run(
            "cmd /c start /b cmd /c ping -n 60 127.0.0.1",
            shell=True, capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        pass  # 也可接受：关键是**按时**出来，而不是被管道拖住
    _dt = time.time() - _t0
    # 阈值 20 秒留足余量：补丁正常是秒回，标准库在这台机器上约 60 秒，差一个数量级。
    assert _dt < 20, f"管道死锁没解决：耗时 {_dt:.1f}s（标准库在这台机器上要约 60s）"
    print(f"  管道死锁补丁               OK  ({_dt:.1f}s，标准库约 60s)")

    # 8) 工具异常必须转成「回给模型的消息」，不能抛出去终结整轮。
    #    实测：agent 把沙箱外的绝对路径喂给 read_file，抛 ValueError，
    #    整轮 20+ 步的工作当场作废。这个中间件就是防这个的。
    #    下面用假的 _Req / handler 直接调中间件，不经过 agent ——
    #    所以这条用例完全不依赖网络与模型。
    _mw = _ToolErrorToMessage()

    class _Req:
        tool_call = {"id": "call_probe", "name": "read_file", "args": {}}

    def _boom(_req):
        raise ValueError("Path:X outside root directory: Y")

    async def _aboom(_req):
        raise ValueError("Path:X outside root directory: Y")

    async def _aok(_req):
        return "ok"

    _msg = _mw.wrap_tool_call(_Req(), _boom)
    assert isinstance(_msg, ToolMessage), f"工具异常没有被转成 ToolMessage: {type(_msg)}"
    assert _msg.tool_call_id == "call_probe", _msg
    assert "outside root directory" in _msg.content, _msg.content
    # 正常返回不能被改动
    assert _mw.wrap_tool_call(_Req(), lambda _r: "ok") == "ok"
    # ⚠️ 异步钩子也必须在：基类的 awrap_tool_call 默认抛 NotImplementedError，
    #    只写同步版的话，改成 astream 就会让「工具异常终结整轮」原样回来。
    assert type(_mw).awrap_tool_call is not AgentMiddleware.awrap_tool_call, \
        "中间件没实现 awrap_tool_call —— 改成 astream 后工具异常又会终结整轮"
    _amsg = asyncio.run(_mw.awrap_tool_call(_Req(), _aboom))
    assert isinstance(_amsg, ToolMessage), f"异步路径没转成 ToolMessage: {type(_amsg)}"
    assert "outside root directory" in _amsg.content, _amsg.content
    assert asyncio.run(_mw.awrap_tool_call(_Req(), _aok)) == "ok"
    print("  工具异常不致命             OK（同步 + 异步两条钩子）")

    # 9) 收尾清理：散落中间文件要收进 _scratch/，而**成品不能被收走**。
    #    课案 3520-3532 有这段清理，本项目原先漏了。
    #    用例两个方向都断言：前者证明清理真的生效，后者证明没误伤成品。
    _work = settings.media.get_mashup_work_dir()
    os.makedirs(_work, exist_ok=True)
    # 文件名带 pid：多个自检/多个 agent 并发跑时不会互相踩。
    _stray = os.path.join(_work, f"selfcheck_stray_{os.getpid()}.txt")
    _keep = os.path.join(_work, "mashup_final.mp4")
    _keep_existed = os.path.exists(_keep)
    try:
        with open(_stray, "w", encoding="utf-8") as f:
            f.write("stray")
        if not _keep_existed:
            # 沙箱里本来没有成品时，先造一个假的占位（内容不重要，只看会不会被搬走）。
            with open(_keep, "wb") as f:
                f.write(b"\0" * 1024)          # 内容不重要，只看会不会被搬走
        _moved = _tidy_work_dir()
        assert os.path.basename(_stray) in _moved, f"散落文件没被收纳: {_moved}"
        assert not os.path.exists(_stray), "收纳后原位置不该还在"
        assert os.path.exists(os.path.join(_work, "_scratch", os.path.basename(_stray))), \
            "收纳后的文件没出现在 _scratch/"
        assert os.path.exists(_keep), "成品 mashup_final.mp4 被误收走了（它不在通配里，不该动）"
        print(f"  收尾清理                   OK  (收纳 {len(_moved)} 个)")
    finally:
        # 自检不留垃圾
        import shutil as _shutil

        # 只删自己造的东西：_keep 若是沙箱里原本就有的真成品，绝不能删。
        for p in (_stray,
                  os.path.join(_work, "_scratch", os.path.basename(_stray)),
                  *( [] if _keep_existed else [_keep] )):
            try:
                os.remove(p)
            except OSError:
                pass
        # 空目录也收掉：_tidy_work_dir 是「有东西才建」，别让自检留下一个空壳。
        _sc = os.path.join(_work, "_scratch")
        if os.path.isdir(_sc) and not os.listdir(_sc):
            _shutil.rmtree(_sc, ignore_errors=True)

    # 最后三行把「这次自检是在什么配置下跑的」摆出来，方便对照线上行为。
    print(f"\n  沙箱目录: {settings.media.get_mashup_work_dir()}")
    print(f"  BGM: {settings.media.bgm_path or '（未配置，将不混音）'}")
    print(f"  编排模型: {settings.media_deepagent_model()}")
    print("全部自检通过")
