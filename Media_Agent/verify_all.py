# -*- coding: utf-8 -*-
"""verify_all.py —— Media_Agent 一键自检
==========================================
对应课案章节：《3.自媒体Agent》全模块自检工具

**它做什么？**
    按「表驱动」的方式跑完 Media_Agent 的全部检查，最后统一汇总：

    第 1 层 · 模块自检（子进程）
        每个承担实际功能的模块（``tools/*.py`` 与 ``workflows/*.py``；
        ``tools/__init__.py``、``workflows/__init__.py`` 两个包入口除外）
        末尾都有一个 ``if __name__ == "__main__":`` 自检块（纯逻辑断言，不联网）。
        这里用子进程逐个跑，验证三件事：
            · 解释器能导入该模块的全部依赖；
            · 自检断言全过（退出码 0）；
            · 输出里没有 Traceback。
              （真正被断言的只有退出码：未捕获的 Traceback 会把退出码顶成非 0，
                而模块自己 catch 住又打印出来的 Traceback 不算失败 ——
                见 ``run_script()``，它只比 ``proc.returncode == 0``。）

    第 2 层 · 视图导入检查
        ``views/*.py`` 不能"直接运行"（要 Streamlit 运行时才成立），
        所以用 ``import`` 的方式验证它们语法正确、依赖齐全。

    第 3 层 · 环境契约检查
        配置项是否都能读到、DeepAgents 的 API 形状是否与代码假设一致
        （这一层是"升级依赖前先跑一下"的护栏）。
        内部按 ``_CONTRACT_SNIPPET`` 的编号分 7 组，前 6 组是判据、第 7 组只是打印：
            1. 配置项存在且类型正确 —— 分组字段 / 扁平字段 / 目录方法 / 模型名回退链；
            2. DeepAgents 的 API 形状 —— ``create_deep_agent`` / ``LocalShellBackend`` /
               ``CompositeBackend`` 的参数与方法名；
            3. 百炼 dashscope SDK 的形状 —— ASR / 声音复刻 / 视频合成三处用到的签名；
            4. 关键文件就位 —— ``.skills/video-use/SKILL.md`` 与它的 frontmatter；
            5. 缺密钥时必须优雅降级 —— 三个工具函数要返回结构，而不是抛异常；
            6. **文档键名契约** —— ``.env.example`` 里出现过的 ``MEDIA_*`` 键，
               必须真能被 ``MediaAgentSettings`` 读到（详见下文）；
            7. 语言提示 —— 打印模型名 / 端点 / 密钥状态，仅给人看，不参与判据。

**为什么用子进程而不是 import？**
    与 ``Back_End/verify_all.py`` 同一理由：每个模块都是"独立程序"，
    子进程运行能同时验证「路径计算不依赖当前工作目录」和「退出码为 0」。
    另外本项目的模块会在 import 时读 ``config``（进而读根 ``.env``），
    子进程隔离能避免相互污染。

**离线 vs 联网**
    默认**完全离线**，零密钥状态下也必须全绿 —— 这是本项目的验收底线。
    加 ``--live`` 会额外跑真实 API 调用（LLM / 热点 / 百炼），
    需要你在 .env 里配好密钥。

    区别分清楚，别把 ``--live`` 的失败当成回归：
        · 离线三层**不发起任何网络请求**，也不会因为「没配密钥」而失败 ——
          缺密钥走的是降级分支，而「降级分支能不能正确返回」本身就是被检查的契约；
        · ``--live`` 是**附加**一层：里面每一项失败只记录、不中断后续项，
          全部跑完再按失败列表决定退出码；
        · ``--live`` 里有两项其实只是「接线检查」——ASR 用**必然不存在的路径**验证
          参数校验分支，数字人只报到开通状态。看到它们 ✓ 不等于端到端通了：
          真实转写 / 真实提交需要素材与开通，得照 README 手动跑。

**怎么加一项检查（照着做）**
    A. 新模块登记进第 1 层
        1. 在模块末尾写 ``if __name__ == "__main__":`` 自检块：
           纯逻辑断言、**不联网**、失败就抛（``assert`` 或异常都行），跑完打印一行中文总结。
        2. 在 ``MODULE_SELF_CHECKS`` 里登记 ``("workflows/新模块.py", None)`` ——
           路径**相对 Media_Agent 目录**，不是相对仓库根。
        3. 第二个元素是「额外参数」：默认 ``None``；要给自检传开关时才写成列表。
        4. 包入口（``tools/__init__.py`` / ``workflows/__init__.py``）**不要登记**：
           它们没有可执行的 ``__main__`` 块，登记了只会得到一条
           「退出码 0 但什么都没跑」的假绿。
    B. 新页面登记进第 2 层
        把 ``"views.新页面"`` 加进 ``VIEW_MODULES`` 即可。
        页面文件不要写 ``__main__`` 自检 —— 它要在 Streamlit 运行时里才成立。
    C. 新契约写进第 3 层
        在 ``_CONTRACT_SNIPPET`` 里加一段，出问题就 ``problems.append("中文说明")``；
        只有 ``problems`` 为空时才会打印「环境契约检查全部通过」并以 0 退出。
        ★ 那段是 ``r`` 前缀的三引号原始字符串：``%HERE%`` / ``%ROOT%`` 由外层 ``.replace()``
          注入，**不要改成 f-string** —— 里面到处是 ``{`` ``}``，会被当成占位符。
    D. 加完自证一遍
        跑一次本脚本，看总数有没有 +1；并且新检查必须能在「修好之前」先红一次 ——
        一条永远不会失败的检查测不出任何东西。

    计数规则：``total`` 按检查项累加 —— 第 1 层每个登记的模块算 1 项，
    第 2、3 层各算 1 项，所以总数 = ``len(MODULE_SELF_CHECKS) + 2``
    （当前 15 + 2 = 17；加 ``--live`` 时 18）。

**与课案的落地差异**
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | 自检清单 | 正文只给「各模块自己带 ``__main__`` 自检」这套思路，没有可运行的一键脚本 | ``MODULE_SELF_CHECKS`` 表驱动 + 三层汇总 | 把「哪些模块要跑」变成一张可读的表，加模块只改一行 |
    | 场景覆盖 | 无 ``--live`` 概念 | 默认离线全绿 + ``--live`` 另跑真实调用 | 验收要在零密钥 / CI 环境里也能跑通，不能强制联网 |
    | 环境契约 | 无 | 第 3 层 6 组契约（含 ``.env.example`` 键名护栏） | 「升级依赖后静默失效」与「文档键名读不到」这两类问题**不报错、只是行为不对**，只能靠护栏提前钉住 |
    | 路径约定 | 依赖当前工作目录 | 全部按 ``__file__`` 推算 ``HERE`` / ``ROOT``，子进程再钉死 ``cwd`` | 从任意目录调用结果都一样 |
    | 文档键名契约 | 无 | 见第 3 层第 6 组 | 实测踩到过两次：字段名与文档键名对不上时，用户照文档改 ``.env`` **静默无效**，而默认值恰好又等于期望值，表面完全看不出问题 |
    | 绝对路径 | 无 | 无 | —— |

**踩过的坑**
    · **子进程要显式指定编码**：Windows 下 ``subprocess`` 默认按 GBK 解码，
      模块自检里的中文输出会 ``UnicodeDecodeError``。所以 ``run_script()`` 传了
      ``encoding="utf-8"`` + ``errors="replace"``，``_child_env()`` 再给子进程设
      ``PYTHONIOENCODING`` / ``PYTHONUTF8``（父子两边都要设，只设一边不够）。
    · **必须用 ``sys.executable`` 当解释器**：本机 PATH 里的 ``python`` 是
      Windows Store 占位符，执行后没有任何输出、也不报错 —— 子进程会「静默地什么都没跑」，
      看起来还像全绿。
    · **``_child_env()`` 要注入 ``PYTHONPATH``**：``workflows/*.py`` 直接跑时
      ``sys.path[0]`` 是 ``workflows/``，``from workflows import ...`` 会
      ModuleNotFoundError；注入之后「模块自带路径引导」和「外层注入」两条路才都成立。
    · **不要用 f-string 拼那三段内联脚本**：脚本里到处都是 ``{}``，
      f-string 会把它们当占位符吃掉；``%HERE%`` / ``%MODS%`` 占位 + ``.replace()``
      才是对的写法。
    · **文档键名护栏只比键名、不校默认值**：文档写 ``MEDIA_AVATAR_TIMEOUT=900``、
      代码默认值改成 600，它不会响 —— 这是已知边界，别当成它没工作。

运行方式::

    # 离线全量自检（默认）
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Media_Agent\\verify_all.py'

    # 额外跑真实 API
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Media_Agent\\verify_all.py' --live

本节知识点：
    1. subprocess.run 的 capture_output / text / encoding / timeout
    2. sys.executable 保证"用同一个解释器"跑子脚本（不碰运气找 python）
    3. PYTHONIOENCODING=utf-8 保证子进程输出是 UTF-8（Windows 默认 GBK）
    4. 表驱动扫描 + 统一汇总退出码
"""

# 打开 PEP 563：注解在运行期不求值，`list[str] | None` 这种写法在旧解释器上也能解析，
# 同时不改变任何运行期行为（本文件的注解只给自己人看）。
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

# ------------------------------------------------------------ 路径
# 全部按 __file__ 推算：不管从哪个工作目录调用本脚本，HERE / ROOT 都不变。
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                       # 仓库根（config.py 所在）
VENV_PYTHON = sys.executable             # 当前解释器就是 .venv 里的 python（不要用 PATH 里的 python）
# 单个模块自检的超时上限（秒）。实测最慢的两项（tools/douyin_client.py 与
# workflows/mashup.py）各约 4.5 秒，180 秒是给冷启动、慢磁盘与机器忙时留的余量 ——
# 设小了会在负载高的时候偶发假失败，那比漏检更糟。
PER_SCRIPT_TIMEOUT = 180

# 父子都转 UTF-8：子进程那边还会再通过 _child_env() 设一次环境变量（两边都要）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 用 `in sys.argv` 而不是 argparse：只有一个开关，还不限位置（与别的参数混排也不报错），
# 也不必为此引入依赖。
LIVE = "--live" in sys.argv


# ------------------------------------------------------------ 子进程环境
def _child_env() -> dict:
    """给子进程的环境：强制 UTF-8 + 把 Media_Agent 目录放进 sys.path。

    Returns:
        可直接传给 ``subprocess.run(env=...)`` 的环境字典 ——
        它是 ``os.environ`` 的一份拷贝，不会改动当前进程自己的环境。

    Args:
        无（全部取自本模块的 ``HERE`` 与当前进程环境）。
    """
    env = os.environ.copy()
    # 两个都设、互相兜底：PYTHONIOENCODING 管标准流的编解码，
    # PYTHONUTF8=1 打开解释器的 UTF-8 模式（文件系统编码等也一并跟着走）。
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # 让子脚本能 `from tools.xxx import ...` / `from workflows.xxx import ...`
    # 前置自己的目录：workflows/*.py 直接跑时 sys.path[0] 是 workflows/，
    # 光靠模块自带的路径引导还不够稳（有的模块没写）。
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_script(rel_path: str, extra_args: list[str] | None = None) -> dict:
    """用子进程跑一个模块的自检。

    Args:
        rel_path: 相对 ``Media_Agent`` 目录的脚本路径，如 ``"workflows/base.py"``。
        extra_args: 追加到命令行末尾的参数，默认 ``None``（等于不加参数）。

    Returns:
        统一形状的结果字典：``name`` / ``ok`` / ``code`` / ``out`` / ``err`` / ``seconds``。
        约定：``code = -1`` 表示脚本文件不存在，``-9`` 表示超时；两种情况都**不抛异常**，
        只是 ``ok=False``，好让汇总阶段能一次把所有失败项列全。
    """
    script = HERE / rel_path
    # 文件不存在也返回同形状的 dict：调用方按字段取值，缺字段会让汇总那段 KeyError。
    if not script.is_file():
        return {"name": rel_path, "ok": False, "code": -1,
                "out": "", "err": f"文件不存在: {script}", "seconds": 0.0}

    started = time.time()
    try:
        proc = subprocess.run(
            [VENV_PYTHON, str(script)] + (extra_args or []),
            # text=True 才拿得到 str；encoding 显式指定 UTF-8（Windows 默认 GBK 会解码失败）；
            # errors="replace" 保证子进程混进非 UTF-8 字节时不至于让整轮检查崩掉 ——
            # 少一个字符远比丢掉全部结果划算。
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            # cwd 钉在 HERE：让结果与「调用本脚本时人在哪个目录」无关。
            timeout=PER_SCRIPT_TIMEOUT, cwd=str(HERE), env=_child_env(),
        )
        return {
            "name": rel_path, "ok": proc.returncode == 0, "code": proc.returncode,
            "out": proc.stdout or "", "err": proc.stderr or "",
            "seconds": time.time() - started,
        }
    except subprocess.TimeoutExpired:
        # 超时单独标 -9（自定约定）：汇总里一眼能分清「跑挂了」和「跑太久」。
        return {"name": rel_path, "ok": False, "code": -9,
                "out": "", "err": f"超时（>{PER_SCRIPT_TIMEOUT}s）",
                "seconds": time.time() - started}


def tail(text: str, n: int = 6) -> list[str]:
    """截取文本末尾几行（失败清单里只贴关键输出，不淹没汇总区）。

    Args:
        text: 原始文本；``None`` 或空串都当空处理。
        n: 取最后几行，默认 6 —— 够看到 traceback 的最后一句真实异常，
            又不至于把整页输出铺满。

    Returns:
        过滤掉空行之后的末尾若干行；输入为空时返回空列表。
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-n:]


# ------------------------------------------------------------ 第 1 层：模块自检
# 清单元素是 (相对路径, 额外参数)，路径**相对 Media_Agent 目录**（不是仓库根）。
#
# 登记规则（照着做就不会踩坑）：
#   · 只登记带 `if __name__ == "__main__":` 自检块、且自检**纯离线**的模块；
#   · 包入口（tools/__init__.py、workflows/__init__.py）**不登记** —— 它们没有 __main__ 块，
#     登记了只会得到一条「退出码 0 但什么都没跑」的假绿；
#   · views/*.py 与 main.py 也不登记：前者要在 Streamlit 运行时里才成立（改走第 2 层），
#     后者一 import 就会 set_page_config 并真渲染页面；
#   · 第二个元素默认 None，要给自检传开关时才写成列表（如 ["--quick"]）。
# 当前 15 条 = tools 8 条 + workflows 7 条；三层合计 17 项（加 --live 时 18 项）。
MODULE_SELF_CHECKS = [
    # (相对路径, 额外参数)
    ("tools/dashscope_upload.py", None),
    ("tools/asset_host.py", None),
    ("tools/audio_transcriber.py", None),
    ("tools/media_tools.py", None),
    ("tools/trend_radar_client.py", None),
    ("tools/voice_clone.py", None),
    ("tools/avatar_client.py", None),
    ("tools/douyin_client.py", None),
    ("workflows/base.py", None),
    ("workflows/positioning.py", None),
    ("workflows/hot_topic.py", None),
    ("workflows/replicate.py", None),
    ("workflows/video.py", None),
    ("workflows/mashup.py", None),
    ("workflows/review.py", None),
]


# ------------------------------------------------------------ 第 2 层：视图导入
# 要导入的视图模块。不含 views/__init__.py（它没有页面函数，且 import 任何一个
# views.xxx 都会连带执行它），也不含 main.py（它一 import 就会渲染整个页面）。
VIEW_MODULES = [
    "views.home", "views.positioning", "views.hot_topic", "views.replicate",
    "views.video", "views.mashup", "views.review",
]

# 第 2 层的脚本内容。为什么用 `python -c` 而不是 `-m`：要在子进程里先把 HERE 插进
# sys.path 再 import，只有 `-c` 能把这段引导代码一起喂进去。
# 为什么用 %HERE% / %MODS% 占位再 .replace()：脚本里到处是 `{}`（字典、`%` 格式化），
# 换成 f-string 会把花括号当占位符吃掉。
# 脚本行为：逐个 __import__，把失败的收集起来一次性打印，最后 raise SystemExit(1)。
_VIEW_IMPORT_SNIPPET = '''
import sys
sys.path.insert(0, r"%HERE%")

mods = %MODS%

bad = []
for m in mods:
    try:
        __import__(m)
    except Exception as exc:
        bad.append("%s: %s: %s" % (m, type(exc).__name__, exc))

if bad:
    print("导入失败:")
    for b in bad:
        print("  " + b)
    raise SystemExit(1)

print("%d 个 views 模块全部导入成功" % len(mods))
'''.replace("%HERE%", str(HERE)).replace("%MODS%", repr(VIEW_MODULES))


# ------------------------------------------------------------ 第 3 层：环境契约
# 第 3 层的脚本内容，内部按编号分 7 组（每组前面有 `# ---- n. ----` 分隔）：
#   1. 配置项存在且类型正确（分组字段 / 扁平字段 / 目录方法 / 模型名回退链）
#   2. DeepAgents 的 API 形状（create_deep_agent / LocalShellBackend / CompositeBackend）
#   3. 百炼 dashscope SDK 的形状（ASR / 声音复刻 / 视频合成三处签名）
#   4. 关键文件就位（.skills/video-use/SKILL.md 及 frontmatter）
#   5. 缺密钥时必须优雅降级（三个工具函数返回结构而不是抛异常）
#   6. 文档键名契约（.env.example 里的 MEDIA_* 必须真能被 pydantic 读到）
#   7. 语言提示（打印模型名 / 端点 / 密钥状态，只是给人看，不参与判据）
# 判据：problems 为空才算过；非空则逐条打印后 raise SystemExit(1)。
# 占位符 %HERE% / %ROOT% 由下面的 .replace() 注入；用 r''' 是因为脚本里到处是反斜杠
# （正则、Windows 路径），用普通字符串会被当转义序列。
_CONTRACT_SNIPPET = r'''
"""环境契约检查：配置可读 + 依赖 API 形状符合代码假设 + 关键文件就位。"""
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, r"%HERE%")

from config import MediaAgentSettings, settings

problems = []

# ---- 1. 配置项存在且类型正确 ----
media = settings.media
for name in (
    "asr_model", "asr_language", "tts_model", "tts_fallback_voice",
    "avatar_model", "trendradar_api_url", "douyin_api_base",
    "llm_model", "deepagent_model",
):
    if not hasattr(media, name):
        problems.append(f"settings.media 缺少字段 {name}")

for name in ("dashscope_api_key", "dashscope_api_base", "dashscope_workspace_id",
             "api_key", "base_url", "model_name"):
    if not hasattr(settings, name):
        problems.append(f"settings 缺少字段 {name}")

if not settings.dashscope_api_endpoint.startswith("http"):
    problems.append("dashscope_api_endpoint 不是合法 URL")

# 目录方法必须返回绝对路径且已创建
for meth in ("get_video_output_dir", "get_image_output_dir",
             "get_avatar_input_dir", "get_mashup_work_dir"):
    p = getattr(media, meth)()
    if not p or not Path(p).is_absolute():
        problems.append(f"{meth}() 未返回绝对路径: {p!r}")
    elif not Path(p).is_dir():
        problems.append(f"{meth}() 返回的目录不存在: {p}")

# 模型名回退链：MEDIA_LLM_MODEL 留空时必须能回落到根 MODEL_NAME
if not settings.media_llm_model():
    problems.append("media_llm_model() 为空 —— 根 .env 的 MODEL_NAME 没读到")
if not settings.media_deepagent_model():
    problems.append("media_deepagent_model() 为空")

# ---- 2. DeepAgents API 形状（升级 deepagents 前先跑这一层）----
try:
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend

    sig = inspect.signature(LocalShellBackend.__init__)
    for param in ("root_dir", "virtual_mode", "inherit_env", "timeout",
                  "max_output_bytes", "env"):
        if param not in sig.parameters:
            problems.append(f"LocalShellBackend 构造参数缺少 {param}（deepagents 可能已升级）")
    if not hasattr(LocalShellBackend, "_resolve_path"):
        problems.append("LocalShellBackend 没有 _resolve_path —— mashup.py 的路径补丁会失效")

    # 技能目录靠 CompositeBackend 挂到虚拟路径 /skills/（mashup.py 的实测补丁 9）。
    # 这些形状一变，剪辑链路会以「agent 一步没跑就抛 outside root directory」的形式挂掉，
    # 报错点还在 deepagents 内部、跟 mashup.py 看着没关系，所以在这里提前钉住。
    for cls, params in (
        (FilesystemBackend, ("root_dir", "virtual_mode")),
        (CompositeBackend, ("default", "routes")),
    ):
        csig_cls = inspect.signature(cls.__init__)
        for param in params:
            if param not in csig_cls.parameters:
                problems.append(f"{cls.__name__} 构造参数缺少 {param}（deepagents 可能已升级）")
    for meth in ("ls", "download_files"):
        if not hasattr(CompositeBackend, meth):
            problems.append(f"CompositeBackend 缺少 {meth} —— SkillsMiddleware 加载技能会失败")

    csig = inspect.signature(create_deep_agent)
    for param in ("model", "backend", "system_prompt", "skills"):
        if param not in csig.parameters:
            problems.append(f"create_deep_agent 缺少参数 {param}")
except Exception as exc:
    problems.append(f"DeepAgents 导入/检查失败: {type(exc).__name__}: {exc}")

# ---- 3. 百炼 SDK 形状（ASR / 声音复刻 / 视频合成三处都依赖）----
try:
    from dashscope.audio.tts_v2 import SpeechSynthesizer, VoiceEnrollmentService
    from dashscope.utils.oss_utils import OssUtils

    if "target_model" not in inspect.signature(VoiceEnrollmentService.create_voice).parameters:
        problems.append("VoiceEnrollmentService.create_voice 签名变了")
    if "media" not in inspect.signature(
        __import__("dashscope").VideoSynthesis.async_call
    ).parameters:
        problems.append("VideoSynthesis.async_call 不再支持 media 参数（PixVerse 对口型要用）")
    if not hasattr(SpeechSynthesizer, "call"):
        problems.append("SpeechSynthesizer 没有 call 方法")
    if "model" not in inspect.signature(OssUtils.upload).parameters:
        problems.append("OssUtils.upload 签名变了")
except Exception as exc:
    problems.append(f"dashscope SDK 导入/检查失败: {type(exc).__name__}: {exc}")

# ---- 4. 关键文件就位 ----
skill = Path(r"%HERE%") / ".skills" / "video-use" / "SKILL.md"
if not skill.is_file():
    problems.append(f"技能文件缺失: {skill}")
else:
    head = skill.read_text(encoding="utf-8")[:200]
    if not head.startswith("---") or "name: video-use" not in head:
        problems.append("SKILL.md 的 frontmatter 不合法（DeepAgents 靠它识别技能）")

# ---- 5. 缺密钥时必须优雅降级（不是抛异常）----
try:
    from tools.audio_transcriber import transcribe
    from tools.avatar_client import submit_lipsync
    from tools.voice_clone import clone_voice

    if not isinstance(transcribe("", want_timestamps=False), dict):
        problems.append("transcribe 失败路径没返回 dict")
    if clone_voice("")["success"] is not False:
        problems.append("clone_voice 失败路径没返回 success=False")
    if submit_lipsync("")["success"] is not False:
        problems.append("submit_lipsync 失败路径没返回 success=False")
except Exception as exc:
    problems.append(f"降级路径检查异常: {type(exc).__name__}: {exc}")

# ---- 6. 文档键名契约：.env.example 里的 MEDIA_* 必须真能被 pydantic 读到 ----
# 键名 = "MEDIA_" + 字段名大写（由 MediaAgentSettings 的 env_prefix 决定）。
# 字段名写错时**不会报错**：那个键静默无效，只剩代码里的默认值在生效 ——
# 用户照文档改了 .env 却毫无反应。实测踩到过两次，其中一次字段就叫 use_hyperframes，
# 文档里写的 MEDIA_MASHUP_USE_HYPERFRAMES 根本读不到（默认值恰好是 false，表面看不出）。
env_example = Path(r"%ROOT%") / ".env.example"
if not env_example.is_file():
    problems.append(f"模板文件缺失: {env_example}")
else:
    # 注释掉的键（形如 `# MEDIA_VIDEO_OUTPUT_DIR=`）也算「文档声明过的键」，
    # 所以整篇全量扫描，不做行首过滤。
    documented = set(re.findall(r"\bMEDIA_[A-Z0-9_]+\b", env_example.read_text(encoding="utf-8")))
    readable = {"MEDIA_" + name.upper() for name in MediaAgentSettings.model_fields}

    for key in sorted(documented - readable):
        problems.append(
            f"文档里的 {key} 读不到：MediaAgentSettings 没有对应字段"
            f"（键名 = MEDIA_ + 字段名大写）。"
            f"要么把字段改名成 {key[6:].lower()}，要么把文档改成正确的键名。"
        )

    # 反向只做提示、不算失败：有几个内部调优项确有合理默认值，不必都写进模板。
    undocumented = sorted(readable - documented)
    if undocumented:
        print(f"  （提示）{len(undocumented)} 个字段未在 .env.example 出现: "
              + ", ".join(undocumented))

# ---- 7. 语言提示：不要误报 ----
print(f"文本模型: {settings.media_llm_model()}")
print(f"编排模型: {settings.media_deepagent_model()}")
print(f"百炼端点: {settings.dashscope_api_endpoint}")
print(f"百炼密钥: {'已配置' if settings.dashscope_api_key else '未配置（相关链路会降级）'}")

if problems:
    print("\n发现问题:")
    for p in problems:
        print("  ✗ " + p)
    raise SystemExit(1)
print("环境契约检查全部通过")
'''.replace("%HERE%", str(HERE)).replace("%ROOT%", str(ROOT))


# ------------------------------------------------------------ 联网检查（--live）
# --live 的脚本内容：真实 API 调用检查。
# 设计要点：每一项都用 check(name, fn) 包住 —— fn 抛异常只记成一条失败，不中断后面的项；
# 全部跑完再按失败列表 raise SystemExit(1)（全过则 0）。联网检查很贵（真实请求 + 计费），
# 一次跑完拿到全部结论，比「第一项失败就退出、然后反复重跑」划算得多。
# 其中 ASR / 数字人两项其实是**接线检查**：前者用不存在的音频路径验证参数校验分支，
# 后者只报到开通状态 —— 看到 ✓ 不等于端到端通了。
_LIVE_SNIPPET = r'''
"""真实 API 调用检查（--live）。每一项失败都只报告、不中断后续检查。"""
import json
import sys

sys.path.insert(0, r"%HERE%")

from config import settings

results = []


def check(name, fn):
    try:
        ok, detail = fn()
        results.append((name, bool(ok), detail))
    except Exception as exc:
        results.append((name, False, f"{type(exc).__name__}: {exc}"))


# ---- 1. LLM ----
def _llm():
    from workflows import llm_call
    out = llm_call("只回答两个字：收到", temperature=0.1)
    good = out and "LLM调用失败" not in out and "LLM未配置" not in out
    return good, (out or "")[:120].replace("\n", " ")


check("LLM 文本推理", _llm)


# ---- 2. 热点抓取（公共 API，无需密钥）----
def _hot():
    from tools.trend_radar_client import TrendRadarClient
    topics = TrendRadarClient().fetch_platform("weibo")
    return bool(topics), f"微博热榜 {len(topics)} 条" + (
        f"，第 1 条: {topics[0]['title'][:30]}" if topics else "")


check("热点抓取（NewsNow）", _hot)


# ---- 3. 百炼语音识别 ----
def _asr():
    if not settings.dashscope_api_key:
        return False, "跳过：未配置 DASHSCOPE_API_KEY"
    from tools.audio_transcriber import transcribe
    # 用一个必然不存在的路径验证「走到网络层之前就正确返回」，
    # 真正的音频转写请用你自己的一段音频单独跑。
    r = transcribe("__live_check_missing__.wav")
    if r["error"] and "不存在" in r["error"]:
        return True, "参数校验通过（真实转写需要一段音频，见 README 手动验证）"
    return False, r.get("error") or "返回不符合预期"


check("百炼 ASR 接线", _asr)


# ---- 4. 数字人模型是否已开通（不真提交任务，只报到）----
def _avatar():
    if not settings.dashscope_api_key:
        return False, "跳过：未配置 DASHSCOPE_API_KEY"
    return True, (
        f"模型 {settings.media.avatar_model}；"
        "真实提交需要一段人脸视频，且需先在百炼控制台开通 PixVerse"
    )


check("数字人接线", _avatar)


print()
for name, ok, detail in results:
    print(f"  {'✓' if ok else '✗'} {name}: {detail}")

failed = [n for n, ok, _ in results if not ok]
if failed:
    print(f"\n联网检查未全部通过：{failed}")
else:
    print("\n联网检查全部通过")
raise SystemExit(1 if failed else 0)
'''.replace("%HERE%", str(HERE))


# ------------------------------------------------------------ 主流程
def main() -> int:
    """跑完三层检查（加 ``--live`` 再跑一层），打印汇总并返回退出码。

    第 1 层遍历 ``MODULE_SELF_CHECKS``（每项一个子进程），第 2 层跑
    ``_VIEW_IMPORT_SNIPPET``，第 3 层跑 ``_CONTRACT_SNIPPET``，``--live`` 时再跑
    ``_LIVE_SNIPPET``。每层失败的项都收进 ``failed``，最后统一贴出来，而不是一失败就退出。

    Args:
        无（模式由模块级 ``LIVE`` 决定，它来自命令行里的 ``--live``）。

    Returns:
        ``0`` = 全部通过；``1`` = 有任意一项失败。
        调用方（文件末尾）用 ``sys.exit(main())`` 把它当进程退出码。
    """
    mode = "离线 + 联网" if LIVE else "仅离线（零密钥）"
    print("=" * 70)
    print(f"Media_Agent 一键自检 —— 模式：{mode}")
    print(f"项目目录: {HERE}")
    print(f"解释器:   {VENV_PYTHON}")
    print("=" * 70)

    failed: list[dict] = []
    # total 按「检查项」计数：第 1 层每登记一个模块算 1 项，第 2/3 层各 1 项
    #（--live 再加 1），所以它随 MODULE_SELF_CHECKS 的长度自动变。
    total = 0

    # ---- 第 1 层 ----
    print("\n【第 1 层】模块自检（子进程运行，各模块自己的 __main__ 断言）")
    print("-" * 70)
    for rel, args in MODULE_SELF_CHECKS:
        total += 1
        r = run_script(rel, args)
        # 每跑一项就打印一行（含退出码与耗时）—— 失败时靠这行定位是哪一步慢/挂。
        mark = "✓" if r["ok"] else "✗"
        print(f"  {mark} {rel:<34} 退出码={r['code']:<4} {r['seconds']:.1f}s")
        if not r["ok"]:
            failed.append(r)

    # ---- 第 2 层 ----
    # 第 2 层不像第 1 层那样有清单要遍历，它整层就是一次子进程调用，所以 total 只 +1。
    print("\n【第 2 层】视图导入检查（views/*.py 不能直接运行，改用 import）")
    print("-" * 70)
    total += 1
    view_result = run_script_via_code(_VIEW_IMPORT_SNIPPET, "views 导入")
    print(f"  {'✓' if view_result['ok'] else '✗'} {view_result['name']}  "
          f"退出码={view_result['code']}")
    if not view_result["ok"]:
        failed.append(view_result)

    # ---- 第 3 层 ----
    print("\n【第 3 层】环境契约检查（配置 + 依赖 API 形状 + 关键文件 + 降级路径）")
    print("-" * 70)
    total += 1
    contract = run_script_via_code(_CONTRACT_SNIPPET, "环境契约")
    # 第 3 层子进程的输出整段转发出来（编号 1~7 的分组结论与「提示」都在里面），
    # 只把它缩进一层区别于本脚本自己的输出。
    for line in (contract["out"] or "").splitlines():
        print(f"    {line}")
    print(f"  {'✓' if contract['ok'] else '✗'} 环境契约检查 退出码={contract['code']}")
    if not contract["ok"]:
        failed.append(contract)

    # ---- 联网（可选）----
    if LIVE:
        print("\n【附加】真实 API 调用检查（--live）")
        print("-" * 70)
        total += 1
        live = run_script_via_code(_LIVE_SNIPPET, "联网检查")
        for line in (live["out"] or "").splitlines():
            print(f"    {line}")
        print(f"  {'✓' if live['ok'] else '✗'} 联网检查 退出码={live['code']}")
        if not live["ok"]:
            failed.append(live)
    else:
        # 没加 --live 也要明确说一句「跳过了」：否则读者分不清
        #「联网检查没跑」和「跑了但输出被吞了」。
        print("\n【附加】真实 API 调用检查 —— 已跳过（加 --live 才会跑）")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    ok_count = total - len(failed)
    print(f"共 {total} 项检查，通过 {ok_count} 项，失败 {len(failed)} 项。")
    if failed:
        print("\n失败清单：")
        for r in failed:
            print(f"  ✗ {r['name']}（退出码 {r['code']}）")
            # err 优先（真正的报错在 stderr），没有才回退到 out；各取最后 6 行。
            for line in tail(r["err"] or r["out"], 6):
                print(f"      {line}")
        print("\n自检未全部通过 ✗")
        return 1

    print("\n全部自检通过 ✓")
    # 验收证据的落点写在这里，省得每次都要问「输出贴哪」。
    print("提示：把本文件的完整输出保存到 Media_Agent/VERIFY_REPORT.md 即可作为交付证据。")
    return 0


def run_script_via_code(code: str, name: str) -> dict:
    """用子进程 ``python -c`` 跑一段内联脚本（第 2 / 3 层与 ``--live`` 都走它）。

    定义在 ``main()`` **之后**是历史书写顺序，运行期没有影响 ——
    ``main()`` 只在被调用时才去解析这个名字，那时模块早已执行完。

    Args:
        code: 要执行的 Python 源码，本文件里由三段 ``*_SNIPPET`` 常量提供。
        name: 显示在汇总里的人类可读名字，如 ``"环境契约"``。

    Returns:
        与 ``run_script()`` 同形状的字典（``name`` / ``ok`` / ``code`` / ``out`` /
        ``err`` / ``seconds``）；子进程超时记 ``code = -9`` 且不抛异常。
    """
    started = time.time()
    try:
        proc = subprocess.run(
            # 与 run_script() 同一套参数：UTF-8 解码 + errors="replace" + 钉死 cwd +
            # 注入子进程环境。唯一区别是把「脚本路径」换成了源码字符串。
            [VENV_PYTHON, "-c", code],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=PER_SCRIPT_TIMEOUT, cwd=str(HERE), env=_child_env(),
        )
        return {"name": name, "ok": proc.returncode == 0, "code": proc.returncode,
                "out": proc.stdout or "", "err": proc.stderr or "",
                "seconds": time.time() - started}
    except subprocess.TimeoutExpired:
        return {"name": name, "ok": False, "code": -9, "out": "",
                "err": f"超时（>{PER_SCRIPT_TIMEOUT}s）", "seconds": time.time() - started}


# 用退出码把结论交给调用方（CI 与脚本链都靠它判断，不要改成 print 后正常结束）。
if __name__ == "__main__":
    sys.exit(main())
