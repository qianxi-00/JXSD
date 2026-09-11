"""conftest.py —— 07_测试框架 的公共 fixture
================================================================
对应课案章节：后端开发基础 → 测试框架（固定装置 Fixture）

pytest 会自动加载同目录的 conftest.py，里面的 fixture 不需要 import 就能用。

**fixture 是什么？** 它是"测试的准备工作"，核心价值有三个：
    1. 复用：多个测试共享同一套准备逻辑，不用每个用例写一遍；
    2. 清理：用 yield 实现 Setup / Teardown，保证测试之间互不污染；
    3. 可组合：fixture 可以使用别的 fixture（形成依赖链）。

**scope（作用域）决定 fixture 多久创建一次**：
    function（默认）每个测试函数一次 —— 最安全，隔离性最好
    class           每个测试类一次
    module          每个测试文件一次
    session         整个测试会话一次 —— 最快，但要注意状态污染
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import pytest

# ------------------------------------------------------------
# 路径引导：本文件位于 Back_End/07_测试框架/conftest.py
# 把本目录与 Back_End 加入 sys.path，保证可以 import calculator
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(BACK_END), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================
# 一、最基础的 fixture
# ============================================================
@pytest.fixture()
def calc():
    """提供一个全新的 Calculator 实例。

    为什么每个测试都要新实例？因为 Calculator 带 history 状态。
    如果所有测试共用一个实例，前一个测试的运算历史会污染后一个测试的断言 ——
    这就是"测试隔离"的重要性。

    scope 使用默认的 function：每个测试函数拿到的是不同对象。
    """
    from calculator import Calculator

    return Calculator()


@pytest.fixture()
def calc_pair():
    """一次提供两个实例，演示 fixture 返回多个值（元组解包）。"""
    from calculator import Calculator

    return Calculator(precision=2), Calculator(precision=8)


# ============================================================
# 二、带 Setup / Teardown 的 fixture（yield）
# ============================================================
@pytest.fixture()
def temp_file():
    """创建临时文件，测试结束后自动删除。

    yield 之前 = Setup（准备资源）
    yield 之后 = Teardown（清理资源，即使测试失败也会执行）

    课案里的写法是写死 "temp.txt"，这里用 tempfile 生成唯一路径，
    避免测试并行执行或上一次失败残留导致的互相干扰。
    """
    path = pathlib.Path(tempfile.gettempdir()) / "pytest_后端课案_临时文件.txt"
    path.write_text("test data", encoding="utf-8")     # Setup

    yield path                                          # 把路径交给测试

    # Teardown：无论测试通过还是失败都会执行
    if path.exists():
        path.unlink()


@pytest.fixture()
def sample_dir():
    """创建一个临时目录，里面放几个文件，测试结束后整个目录删掉。

    这是"文件相关测试"的推荐做法：绝不碰真实业务目录。
    """
    import shutil

    root = pathlib.Path(tempfile.mkdtemp(prefix="pytest_demo_"))
    (root / "a.txt").write_text("aaa", encoding="utf-8")
    (root / "b.txt").write_text("bbbbb", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "c.txt").write_text("cc", encoding="utf-8")

    yield root

    shutil.rmtree(root, ignore_errors=True)


# ============================================================
# 三、fixture 依赖 fixture（可组合）
# ============================================================
@pytest.fixture()
def prepared_calc(calc):
    """在 calc 的基础上再做一步准备：先算几个数，让 history 非空。

    fixture 之间可以互相依赖，pytest 会保证被依赖的先创建。
    这让"测试前置条件"可以像积木一样组合起来。
    """
    calc.add(1, 1)
    calc.add(2, 2)
    return calc


# ============================================================
# 四、参数化 fixture（让整个测试函数跑多遍）
# ============================================================
@pytest.fixture(params=[2, 4, 8], ids=["precision2", "precision4", "precision8"])
def any_precision(request):
    """参数化 fixture：使用它的测试会自动跑 3 遍（每档精度一遍）。

    与 @pytest.mark.parametrize 的区别：
        · parametrize 参数化的是"测试函数的入参"；
        · 参数化 fixture 参数化的是"测试环境/依赖对象"，更贴近"同一逻辑在多环境下验证"。
    """
    from calculator import Calculator

    return Calculator(precision=request.param)


# ============================================================
# 五、session 级 fixture 与 autouse
# ============================================================
@pytest.fixture(scope="session")
def suite_banner():
    """整个测试会话只创建一次的 fixture（适合"贵的"资源：数据库连接、模型加载）。"""
    print("\n[07_测试框架] 测试会话开始（session fixture 只执行一次）")
    return "后端开发基础 · 测试框架章节"


@pytest.fixture(autouse=True)
def _count_calls(request):
    """autouse=True：不需要在测试函数参数里写，自动对每个用例生效。

    这里用它打印每个用例的名称，方便观察测试执行顺序（教学用）。
    生产测试里常用 autouse fixture 做"重置数据库""清空缓存"等全局准备。
    """
    print(f"    → 开始执行：{request.node.name}")
    yield
