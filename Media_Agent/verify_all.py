# -*- coding: utf-8 -*-
"""verify_all.py —— Media_Agent 一键自检
==========================================
对应课案章节：《3.自媒体Agent》全模块自检工具

**它做什么？**
    按「表驱动」的方式跑完 Media_Agent 的全部检查，最后统一汇总：

    第 1 层 · 模块自检（子进程）
        每个 ``tools/*.py`` 与 ``workflows/*.py`` 末尾都有一个
        ``if __name__ == "__main__":`` 自检块（纯逻辑断言，不联网）。
        这里用子进程逐个跑，验证三件事：
            · 解释器能导入该模块的全部依赖；
            · 自检断言全过（退出码 0）；
            · 输出里没有 Traceback。

    第 2 层 · 视图导入检查
        ``views/*.py`` 不能"直接运行"（要 Streamlit 运行时才成立），
        所以用 ``import`` 的方式验证它们语法正确、依赖齐全。

    第 3 层 · 环境契约检查
        配置项是否都能读到、DeepAgents 的 API 形状是否与代码假设一致
        （这一层是"升级依赖前先跑一下"的护栏）。

**为什么用子进程而不是 import？**
    与 ``Back_End/verify_all.py`` 同一理由：每个模块都是"独立程序"，
    子进程运行能同时验证「路径计算不依赖当前工作目录」和「退出码为 0」。
    另外本项目的模块会在 import 时读 ``config``（进而读根 ``.env``），
    子进程隔离能避免相互污染。

**离线 vs 联网**
    默认**完全离线**，零密钥状态下也必须全绿 —— 这是本项目的验收底线。
    加 ``--live`` 会额外跑真实 API 调用（LLM / 热点 / 百炼），
    需要你在 .env 里配好密钥。

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

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

# ------------------------------------------------------------ 路径
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                       # 仓库根（config.py 所在）
VENV_PYTHON = sys.executable             # 当前解释器就是 .venv 里的 python（不要用 PATH 里的 python）
PER_SCRIPT_TIMEOUT = 180

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LIVE = "--live" in sys.argv


# ------------------------------------------------------------ 子进程环境
def _child_env() -> dict:
    """给子进程的环境：强制 UTF-8 + 把 Media_Agent 目录放进 sys.path。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # 让子脚本能 `from tools.xxx import ...` / `from workflows.xxx import ...`
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_script(rel_path: str, extra_args: list[str] | None = None) -> dict:
    """用子进程跑一个模块的自检。"""
    script = HERE / rel_path
    if not script.is_file():
        return {"name": rel_path, "ok": False, "code": -1,
                "out": "", "err": f"文件不存在: {script}", "seconds": 0.0}

    started = time.time()
    try:
        proc = subprocess.run(
            [VENV_PYTHON, str(script)] + (extra_args or []),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=PER_SCRIPT_TIMEOUT, cwd=str(HERE), env=_child_env(),
        )
        return {
            "name": rel_path, "ok": proc.returncode == 0, "code": proc.returncode,
            "out": proc.stdout or "", "err": proc.stderr or "",
            "seconds": time.time() - started,
        }
    except subprocess.TimeoutExpired:
        return {"name": rel_path, "ok": False, "code": -9,
                "out": "", "err": f"超时（>{PER_SCRIPT_TIMEOUT}s）",
                "seconds": time.time() - started}


def tail(text: str, n: int = 6) -> list[str]:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-n:]


# ------------------------------------------------------------ 第 1 层：模块自检
MODULE_SELF_CHECKS = [
    # (相对路径, 额外参数)
    ("tools/dashscope_upload.py", None),
    ("tools/audio_transcriber.py", None),
    ("tools/media_tools.py", None),
    ("tools/trend_radar_client.py", None),
    ("tools/voice_clone.py", None),
    ("tools/avatar_client.py", None),
    ("tools/douyin_client.py", None),
    ("workflows/positioning.py", None),
    ("workflows/hot_topic.py", None),
    ("workflows/replicate.py", None),
    ("workflows/video.py", None),
    ("workflows/mashup.py", None),
    ("workflows/review.py", None),
]


# ------------------------------------------------------------ 第 2 层：视图导入
VIEW_MODULES = [
    "views.home", "views.positioning", "views.hot_topic", "views.replicate",
    "views.video", "views.mashup", "views.review",
]

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
_CONTRACT_SNIPPET = r'''
"""环境契约检查：配置可读 + 依赖 API 形状符合代码假设 + 关键文件就位。"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, r"%HERE%")

from config import settings

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
    from deepagents.backends import LocalShellBackend

    sig = inspect.signature(LocalShellBackend.__init__)
    for param in ("root_dir", "virtual_mode", "inherit_env", "timeout",
                  "max_output_bytes", "env"):
        if param not in sig.parameters:
            problems.append(f"LocalShellBackend 构造参数缺少 {param}（deepagents 可能已升级）")
    if not hasattr(LocalShellBackend, "_resolve_path"):
        problems.append("LocalShellBackend 没有 _resolve_path —— mashup.py 的路径补丁会失效")

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

# ---- 6. 语言提示：不要误报 ----
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
'''.replace("%HERE%", str(HERE))


# ------------------------------------------------------------ 联网检查（--live）
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
    mode = "离线 + 联网" if LIVE else "仅离线（零密钥）"
    print("=" * 70)
    print(f"Media_Agent 一键自检 —— 模式：{mode}")
    print(f"项目目录: {HERE}")
    print(f"解释器:   {VENV_PYTHON}")
    print("=" * 70)

    failed: list[dict] = []
    total = 0

    # ---- 第 1 层 ----
    print("\n【第 1 层】模块自检（子进程运行，各模块自己的 __main__ 断言）")
    print("-" * 70)
    for rel, args in MODULE_SELF_CHECKS:
        total += 1
        r = run_script(rel, args)
        mark = "✓" if r["ok"] else "✗"
        print(f"  {mark} {rel:<34} 退出码={r['code']:<4} {r['seconds']:.1f}s")
        if not r["ok"]:
            failed.append(r)

    # ---- 第 2 层 ----
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
        print("\n【附加】真实 API 调用检查 —— 已跳过（加 --live 才会跑）")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    ok_count = total - len(failed)
    print(f"共 {total} 项检查，通过 {ok_count} 项，失败 {len(failed)} 项。")
    if failed:
        print("\n失败清单：")
        for r in failed:
            print(f"  ✗ {r['name']}（退出码 {r['code']}）")
            for line in tail(r["err"] or r["out"], 6):
                print(f"      {line}")
        print("\n自检未全部通过 ✗")
        return 1

    print("\n全部自检通过 ✓")
    print("提示：把本文件的完整输出保存到 Media_Agent/VERIFY_REPORT.md 即可作为交付证据。")
    return 0


def run_script_via_code(code: str, name: str) -> dict:
    """用子进程 `python -c` 跑一段内联脚本。"""
    started = time.time()
    try:
        proc = subprocess.run(
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


if __name__ == "__main__":
    sys.exit(main())
