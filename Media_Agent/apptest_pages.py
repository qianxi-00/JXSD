# -*- coding: utf-8 -*-
"""七个页面无头渲染 + 本轮修复项的**交互级验收**（Streamlit AppTest）

用法：
    cd F:\\ProGram\\Python_Base\\Media_Agent
    $env:PYTHONUTF8=1
    & F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe apptest_pages.py

不联网、不调模型、不起服务：
  · 第一段：七个页面逐个切换，断言 `at.exception` 为空；
  · 第二段：断言本轮的 UI 修复**真的在页面上呈现**——
    这些是「渲染不报错」覆盖不到的：控件能渲染 ≠ 该有的控件真的在。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # Media_Agent/
_ROOT = _HERE                                    # 本文件就在 Media_Agent 下
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("PYTHONUTF8", "1")

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "🏠 首页",
    "🎯 账号定位",
    "🔥 热点监控",
    "📝 内容复刻",
    "🎥 口播视频",
    "🎬 视频剪辑",
    "📊 数据复盘",
]


def _labels(seq) -> list[str]:
    """把 AppTest 的元素列表拍成「label 字符串」列表（不同元素取 label 的属性名不同）。"""
    out = []
    for el in seq:
        for attr in ("label", "value"):
            v = getattr(el, attr, None)
            if isinstance(v, str) and v:
                out.append(v)
                break
    return out


def _open(page_label: str) -> AppTest:
    at = AppTest.from_file(str(_ROOT / "main.py"), default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(page_label).run()
    return at


def check_pages() -> list[str]:
    fails = []
    for label in PAGES:
        at = AppTest.from_file(str(_ROOT / "main.py"), default_timeout=180)
        at.run()
        if at.exception:
            fails.append(f"{label}: 首屏渲染抛异常 {at.exception[0].message}")
            print(f"  ✗ {label:<12} 首屏渲染抛异常")
            continue
        labels = [r.options for r in at.sidebar.radio]
        if not labels or label not in labels[0]:
            fails.append(f"{label}: 不在导航项里")
            print(f"  ✗ {label:<12} 不在导航项里")
            continue
        at.sidebar.radio[0].set_value(label).run()
        if at.exception:
            fails.append(f"{label}: {at.exception[0].message}")
            print(f"  ✗ {label:<12} 抛异常：{at.exception[0].message}")
        else:
            print(f"  ✓ {label:<12} 渲染 OK")
    return fails


def check_fixes() -> list[str]:
    """本轮修复项的交互级断言。每条都对应一个具体条目 ID。"""
    fails = []

    def want(cond: bool, cid: str, desc: str) -> None:
        if cond:
            print(f"  ✓ {cid:<6} {desc}")
        else:
            fails.append(f"{cid}: {desc}")
            print(f"  ✗ {cid:<6} {desc}")

    # ---- D3：首页「重置运行时缓存」按钮（clear_model_cache 的真入口）----
    at = _open("🏠 首页")
    btn = _labels(at.button)
    want(any("重置运行时缓存" in t for t in btn), "D3", f"首页有「重置运行时缓存」按钮（实际按钮：{btn}）")

    # ---- B3：口播视频页，勾着克隆音色时**也要**有内置音色下拉 ----
    at = _open("🎥 口播视频")
    at.radio[0].set_value("🎭 数字人生成（对口型 → MP4）").run()
    clones = [c for c in at.checkbox if "克隆音色" in (c.label or "")]
    selects = _labels(at.selectbox)
    want(bool(clones), "B3", "数字人模式有「优先使用克隆音色」勾选框")
    want(
        bool(clones) and clones[0].value is True,
        "B3",
        "该勾选框默认勾选（默认走克隆音色这条路）",
    )
    want(
        any("PixVerse 内置音色" in t for t in selects),
        "B3",
        f"**勾选状态下**内置音色下拉依然在（修复前它只在取消勾选时出现）。下拉：{selects}",
    )

    # ---- B1 / B2：热点页，失效平台在下拉里有标注，但**底层值仍是纯名** ----
    at = _open("🔥 热点监控")
    want(bool(at.selectbox), "B1", "热点页有平台下拉")
    if at.selectbox:
        sb = at.selectbox[0]
        # `.options` 拿到的是 `format_func` **格式化之后**的展示文案（这是 Streamlit
        # AppTest 的行为），所以「选项里有纯名『小红书』」这种断言本来就是错的 ——
        # 真正要保证的是：展示可以带标记，但**传给工作流的值**必须是纯平台名。
        opts = list(sb.options)
        decorated = next((o for o in opts if "小红书" in o), "")
        want(
            bool(decorated) and decorated != "小红书",
            "B1",
            f"失效平台「小红书」的展示文案带标注（选项：{opts}）",
        )
        # 选中它，断言底层 value 是纯名 —— 工作流是按平台名查表的
        try:
            sb.set_value(decorated).run()
            got = at.selectbox[0].value
            want(
                got == "小红书",
                "B1",
                f"选中带标注的那项后，底层 value 仍是纯名（实际 value={got!r}）",
            )
            # 真正要验的是「页面拦住了」：填上赛道再点按钮 →
            # 出黄条说明原因 + **不发起抓取**（否则就是白白等十几秒）。
            at.text_input[0].set_value("科技测评").run()
            at.button[0].click().run()
            warns = [(w.value or "") for w in at.warning]
            want(
                any("不可用" in w and "原因" in w for w in warns),
                "B1",
                f"点按钮后出黄条并写明原因（黄条：{warns}）",
            )
            want(not at.exception, "B1", "拦下时不抛异常")
        except Exception as exc:  # noqa: BLE001
            want(False, "B1", f"选中带标注项 / 点击按钮失败：{type(exc).__name__}: {exc}")

    # ---- D2：视频剪辑页「重置剪辑 Agent」按钮 ----
    at = _open("🎬 视频剪辑")
    btn = _labels(at.button)
    want(
        any("重置剪辑 Agent" in t for t in btn),
        "D2",
        f"剪辑页有「重置剪辑 Agent」按钮（reset_editor_agent 的真入口）。按钮：{btn}",
    )

    # ---- T1③：数字人任务**彻底失败**时要出红条，不能显示成蓝色的「处理中」 ----
    # 背景：`tools/avatar_client.query_task()` 对 FAILED/CANCELED/UNKNOWN 会把 message
    # 写成「任务失败/不可查（…）」；而页面为「下载失败可重试」把 `hg_task_code` 一直留着。
    # 两者叠在一起，若页面只判 `task_code` 存不存在，一个**早就挂了**的任务会永远
    # 显示蓝色「处理中」+ 一个点了也没用的刷新按钮 —— 用户以为在跑。
    # 这里直接把 session_state 摆成「失败终态」，断言渲染出的是 `st.error`。
    for _msg, _want_err, _tag in (
        ("任务失败/不可查（FAILED）: code=xxx", True, "FAILED"),
        ("查询异常: 打桩的网络错误", True, "查询异常"),
        ("处理中（RUNNING）", False, "进行中"),
    ):
        at = _open("🎥 口播视频")
        at.session_state["hg_has_result"] = True
        at.session_state["hg_task_code"] = "probe-task"
        at.session_state["hg_msg"] = _msg
        at.run()
        errs = [(e.value or "") for e in at.error]
        infos = [(e.value or "") for e in at.info]
        if _want_err:
            want(
                any(_msg[:12] in e for e in errs) and not errs == [],
                "T1③",
                f"{_tag} 的任务渲染成红条（error={errs} info={infos}）",
            )
        else:
            want(
                not errs and any("处理中" in i or "已提交" in i for i in infos),
                "T1③",
                f"{_tag} 的任务仍渲染成蓝色提示（error={errs} info={infos}）",
            )

    # ---- U1：三个上传框的「处理动作收敛成一次」 ----
    # 背景：`st.file_uploader` 的值**跨 rerun 保留**，所以 `if some_file:` 每轮都为真。
    # 后果分两档：
    #   · 带 `st.rerun()` 的那处（BGM）会**无限自转**，页面卡死；
    #   · 不带 `st.rerun()` 的两处（主口播视频、穿插素材）不自转，但每轮 rerun 都重写 ——
    #     主视频白写几十 MB；素材那处更重，先清空整个目录再整批重写。
    # 修法是「名字|字节数」指纹：只处理没见过的那一份。这里断言的就是**写盘会收敛**：
    # 记录落盘文件的 mtime，再触发几轮 rerun，mtime 不许变。
    #
    # 写盘要重定向到临时目录，别把探针文件丢进项目 `.cache/`（那是 agent 的沙箱根）。
    # 做法与修那个无限 rerun 时用的一致：直接改 `views.mashup` 的模块级常量
    # （脚本 import 的 `views.mashup` 与这里拿到的是同一个模块对象）。
    import tempfile

    try:
        import views.mashup as _vm

        with tempfile.TemporaryDirectory(prefix="apptest_mashup_") as _tmp:
            # ⚠️ 上下文变量是 **str**（不是 Path），必须先包一层 Path 才能用 `/`
            _cache = Path(_tmp) / "cache"
            _mats = Path(_tmp) / "materials"
            _cache.mkdir()
            _mats.mkdir()
            _saved = (_vm.CACHE_DIR, _vm.MATERIALS_DIR)
            _vm.CACHE_DIR = _cache
            _vm.MATERIALS_DIR = _mats
            try:
                at = AppTest.from_file(str(_ROOT / "main.py"), default_timeout=60)
                at.run()
                at.sidebar.radio[0].set_value("🎬 视频剪辑").run()

                # ⚠️ `FileUploader.set_value` 要的是 **3 元组** `(文件名, 内容, mime)`
                # （2 元组会被当成「多文件序列」再去逐项解包，报 too many values to unpack）
                # ① 主口播视频：上传后连跑两轮，落盘文件的时间戳不许变
                at.file_uploader[0].set_value(("u1_probe.mp4", b"x" * 4096, "video/mp4")).run()
                _written = list(_cache.glob("input_*.mp4"))
                if not _written:
                    want(False, "U1", "上传主视频后临时目录里没有落盘文件（重定向可能没生效）")
                else:
                    _m1 = _written[0].stat().st_mtime_ns
                    at.run()
                    at.run()
                    _m2 = _written[0].stat().st_mtime_ns
                    want(_m1 == _m2, "U1", f"重复 rerun 不再重写主视频（mtime {_m1} == {_m2}）")

                # ② 穿插素材：同样连跑两轮，目录里的文件时间戳不许变
                at.file_uploader[1].set_value(
                    [("u1_a.png", b"y" * 2048, "image/png"),
                     ("u1_b.png", b"z" * 2048, "image/png")]
                ).run()
                _mat = sorted(_mats.glob("*.png"))
                if len(_mat) != 2:
                    want(False, "U1", f"上传素材后临时目录里应当是 2 个文件，实际 {len(_mat)}")
                else:
                    _mm1 = [p.stat().st_mtime_ns for p in _mat]
                    at.run()
                    at.run()
                    _mm2 = [p.stat().st_mtime_ns for p in _mat]
                    want(_mm1 == _mm2, "U1", "重复 rerun 不再清空重写素材目录（时间戳未变）")
                    # 跳过写入那几轮也要把「本次用哪些素材」报出来，条数不能变 0
                    want(
                        any("已加载 2 个素材" in (s.value or "") for s in at.success),
                        "U1",
                        "跳过写入的轮次仍报出正确的素材条数（不会变成「已加载 0 个」）",
                    )
            finally:
                _vm.CACHE_DIR, _vm.MATERIALS_DIR = _saved
    except Exception as exc:  # noqa: BLE001
        want(False, "U1", f"上传收敛断言异常：{type(exc).__name__}: {exc}")

    return fails


def main() -> int:
    print("=== 第 1 段：七个页面渲染 ===")
    fails = check_pages()
    print("\n=== 第 2 段：本轮修复项的交互级验收 ===")
    fails += check_fixes()

    print()
    if fails:
        print(f"失败 {len(fails)} 项：")
        for f in fails:
            print("  -", f)
        return 1
    print("七页渲染 0 异常 + 本轮 UI 修复项全部到位 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
