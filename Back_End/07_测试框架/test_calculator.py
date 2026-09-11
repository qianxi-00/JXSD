"""test_calculator.py —— pytest 完整用法演示
================================================================
对应课案章节：后端开发基础 → 测试框架（pytest 全部小节）

本文件覆盖了课案里提到的所有测试写法：
    1. 基本用法：test_ 开头的函数 + assert 断言
    2. 固定装置（fixture）：函数级 / yield 清理 / fixture 依赖 fixture / 参数化 fixture
    3. 参数化测试：@pytest.mark.parametrize（含"多参数组合"与"给用例起名"）
    4. 测试类：class TestXxx + 测试类里使用 fixture
    5. 异常断言：pytest.raises（并且检查异常消息）
    6. 跳过与预期失败：skip / skipif / xfail
    7. 近似断言：pytest.approx（浮点数不能用 == 比）
    8. 异步测试：@pytest.mark.asyncio（pytest-asyncio 已安装）
    9. 临时文件与目录：tmp_path fixture（pytest 内置）
   10. 标记（marker）与分组运行：@pytest.mark.slow + -m "not slow"

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m pytest 'F:\\ProGram\\Python_Base\\Back_End\\07_测试框架' -q
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m pytest 'F:\\ProGram\\Python_Base\\Back_End\\07_测试框架' -v
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m pytest 'F:\\ProGram\\Python_Base\\Back_End\\07_测试框架' -k "parametrize"
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m pytest 'F:\\ProGram\\Python_Base\\Back_End\\07_测试框架' -m "not slow"
"""

from __future__ import annotations

import asyncio
import math
import sys
import time

import pytest

from calculator import Calculator, fizzbuzz, is_prime


# ============================================================
# 一、基本用法：pytest 自动发现 test_ 开头的函数
# ============================================================
def test_addition():
    """最朴素的测试：断言式写法，比 unittest 的 assertEqual 更直观。"""
    assert 1 + 1 == 2


def test_string_upper():
    assert "hello".upper() == "HELLO"


def test_calculator_add(calc):
    """使用 fixture：函数参数名 calc 会自动匹配 conftest.py 里的同名 fixture。"""
    assert calc.add(1, 2) == 3
    assert calc.add(-1, 1) == 0
    assert calc.add(0.1, 0.2) == pytest.approx(0.3)      # 浮点数必须用 approx


def test_calculator_history(calc):
    """测试"副作用"：调用后 history 应当被记录。"""
    assert calc.history == [], "新实例的历史应当为空"
    calc.add(1, 2)
    calc.mul(3, 4)
    assert len(calc.history) == 2
    assert "1 + 2 = 3" in calc.history[0]

    calc.clear_history()
    assert calc.history == []


def test_fixture_isolation(calc):
    """验证 fixture 的隔离性：上一个测试对历史做的操作，不会影响这里。"""
    assert calc.history == [], "每个测试函数都应当拿到全新的 Calculator"


# ============================================================
# 二、参数化测试
# ============================================================
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),          # 第 1 组
    (0, 0, 0),          # 第 2 组
    (-1, 1, 0),         # 第 3 组
    (100, 200, 300),    # 第 4 组
])
def test_add_parametrized(calc, a, b, expected):
    """参数化：一次写好逻辑，多组数据各跑一遍，失败时能精确知道是哪组数据。"""
    assert calc.add(a, b) == expected


@pytest.mark.parametrize("a,b,expected", [
    (10, 2, 5),
    (7, 2, 3.5),
    (-9, 3, -3),
], ids=["整除", "小数结果", "负数"])
def test_div_parametrized(calc, a, b, expected):
    """给用例起中文名（ids），失败报告里一眼看懂是哪一档场景。"""
    assert calc.div(a, b) == expected


@pytest.mark.parametrize("value,expected", [
    (2, True), (3, True), (4, False), (97, True), (100, False), (1, False), (0, False),
])
def test_is_prime(value, expected):
    """模块级函数的参数化测试（不需要 fixture）。"""
    assert is_prime(value) is expected


@pytest.mark.parametrize("n,expected", [
    (1, "1"), (3, "Fizz"), (5, "Buzz"), (15, "FizzBuzz"), (30, "FizzBuzz"), (7, "7"),
])
def test_fizzbuzz(n, expected):
    assert fizzbuzz(n) == expected


@pytest.mark.parametrize("precision", [1, 2, 6])
def test_precision_effect(precision):
    """同一逻辑在不同配置下的表现（这里演示"参数即配置"）。"""
    calc = Calculator(precision=precision)
    result = calc.div(1, 3)
    assert round(result, precision) == result, f"结果应当只保留 {precision} 位小数"


# ============================================================
# 三、异常断言
# ============================================================
def test_div_by_zero(calc):
    """pytest.raises：断言"确实抛出了指定异常"。

    注意：不写 pytest.raises 而是直接 try/except，测试很容易写成"永远通过"的假测试。
    """
    with pytest.raises(ZeroDivisionError):
        calc.div(1, 0)


def test_div_by_zero_message(calc):
    """断言异常消息 —— 能顺便验证错误提示是否友好（对用户很重要）。"""
    with pytest.raises(ZeroDivisionError, match="除数不能为 0"):
        calc.div(10, 0)


def test_sqrt_negative(calc):
    with pytest.raises(ValueError) as exc_info:
        calc.sqrt(-1)
    # exc_info.value 就是异常对象，可以断言它的属性
    assert "不能对负数求平方根" in str(exc_info.value)
    print(f"\n    捕获到的异常类型：{type(exc_info.value).__name__}")


def test_average_empty_list(calc):
    with pytest.raises(ValueError, match="列表为空"):
        calc.average([])


def test_no_exception_for_valid_input(calc):
    """断言"不抛异常"。

    有时我们要确保某些输入是合法的（例如"边界值不该报错"），
    只要正常执行完就说明通过；如果抛异常，测试自然失败。
    """
    assert calc.sqrt(0) == 0
    assert calc.average([5]) == 5


# ============================================================
# 四、测试类
# ============================================================
class TestCalculatorBasics:
    """测试类：把相关测试组织在一起。

    规则：类名必须以 Test 开头，方法名必须以 test_ 开头，且**不能有 __init__**。
    """

    def test_subtraction(self, calc):
        assert calc.sub(5, 3) == 2
        assert calc.sub(3, 5) == -2

    def test_multiplication(self, calc):
        assert calc.mul(2, 3) == 6
        assert calc.mul(-2, 3) == -6
        assert calc.mul(0, 100) == 0

    def test_power(self, calc):
        assert calc.power(2, 10) == 1024
        assert calc.power(9, 0.5) == 3

    def test_average(self, calc):
        assert calc.average([1, 2, 3, 4]) == 2.5
        assert calc.average([10]) == 10


class TestStringOperations:
    """第二个测试类：演示"不同主题分开组织"。"""

    def test_upper(self):
        assert "hello".upper() == "HELLO"

    def test_lower(self):
        assert "WORLD".lower() == "world"

    def test_split(self):
        assert "a b c".split(" ") == ["a", "b", "c"]

    def test_strip(self):
        assert "  x  ".strip() == "x"


class TestFixtureInClass:
    """测试类里使用 fixture（含 fixture 依赖 fixture）。"""

    def test_prepared_calc_has_history(self, prepared_calc):
        """prepared_calc 依赖 calc，并预先做了两次加法。"""
        assert len(prepared_calc.history) == 2

    def test_parametrized_fixture(self, any_precision):
        """参数化 fixture 会让这个测试跑 3 遍（precision=2/4/8）。"""
        assert any_precision.precision in (2, 4, 8)
        assert any_precision.add(1, 1) == 2


class TestTempFile:
    """文件相关测试：用 conftest 里的临时文件 fixture。"""

    def test_read_temp_file(self, temp_file):
        content = temp_file.read_text(encoding="utf-8")
        assert content == "test data"
        assert temp_file.exists()

    def test_write_temp_file(self, temp_file):
        temp_file.write_text("新内容", encoding="utf-8")
        assert temp_file.read_text(encoding="utf-8") == "新内容"

    def test_sample_dir_structure(self, sample_dir):
        """临时目录里应当有 a.txt / b.txt / sub/c.txt。"""
        names = sorted(p.name for p in sample_dir.iterdir())
        assert names == ["a.txt", "b.txt", "sub"]
        assert (sample_dir / "sub" / "c.txt").read_text(encoding="utf-8") == "cc"


class TestBuiltinTmpPath:
    """pytest 内置的 tmp_path fixture：每个测试一个独立临时目录，自动清理。"""

    def test_tmp_path_is_isolated(self, tmp_path):
        f = tmp_path / "demo.txt"
        f.write_text("hello", encoding="utf-8")
        assert f.read_text(encoding="utf-8") == "hello"
        assert tmp_path.is_dir()
        print(f"\n    tmp_path = {tmp_path}")


# ============================================================
# 五、标记：跳过 / 预期失败 / 自定义 marker
# ============================================================
@pytest.mark.skip(reason="演示用：这个用例故意被跳过，不会计入失败")
def test_skipped_demo():
    raise AssertionError("永远不会执行到这里")


@pytest.mark.skipif(sys.version_info < (3, 10), reason="需要 Python 3.10+")
def test_skipif_demo():
    """条件跳过：环境不满足时自动跳过（这里环境是 3.12，所以会执行）。"""
    assert sys.version_info >= (3, 10)


@pytest.mark.xfail(reason="演示用：这是一个已知会失败的用例，xfail 表示'预期失败'")
def test_xfail_demo(calc):
    """xfail = 预期失败。如果它真的失败了 → 记为 xfail（不算失败）；
    如果它意外通过了 → 记为 xpass（默认不算失败，可配 strict 变成失败）。"""
    assert calc.add(1, 1) == 3, "故意写错的断言"


@pytest.mark.slow
def test_slow_demo():
    """自定义 marker：用 -m "not slow" 可以跳过它。

    必须在 pytest.ini 里注册 marker，否则 --strict-markers 会报错。
    注册位置：Back_End/pytest.ini 的 markers 配置。
    """
    time.sleep(0.01)
    assert True


# ============================================================
# 六、异步测试
# ============================================================
@pytest.mark.asyncio
async def test_async_add():
    """异步测试用例。

    pytest-asyncio 已安装；Back_End/pytest.ini 里设置了 asyncio_mode=auto，
    所以 async def 测试不写装饰器也能跑。这里显式写出来是为了说明"需要插件支持"。
    """
    await asyncio.sleep(0.01)
    assert Calculator().add(2, 3) == 5


@pytest.mark.asyncio
async def test_async_concurrency():
    """异步测试里也能做并发，例如同时校验多个"异步接口"。"""

    async def compute(x: int) -> int:
        await asyncio.sleep(0.005)
        return x * x

    results = await asyncio.gather(*(compute(i) for i in range(5)))
    assert results == [0, 1, 4, 9, 16]


# ============================================================
# 七、用 fixture 做"测试报告"式的收尾
# ============================================================
def test_session_fixture(suite_banner):
    """session 级 fixture：整个测试会话只创建一次。"""
    assert "测试框架" in suite_banner


def test_math_module_reference():
    """顺带说明：测试第三方库行为也可以，但通常没必要（别人已经测过了）。"""
    assert math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-9)


# ============================================================
# 八、把 TestClient 接口测试也放在这里（课案的"测试类 + Fixture 结合"）
# ============================================================
class TestFastAPIWithTestClient:
    """课案里演示的"测试类 + fixture 创建 TestClient"写法。

    这里测试的是 06_FastAPI 的一个示例应用（01_最小应用.py），
    示范"如何为别人的 FastAPI 应用写测试"。
    """

    @pytest.fixture()
    def client(self):
        """创建测试客户端 fixture：不启服务、不占端口。"""
        import importlib.util
        import pathlib

        from fastapi.testclient import TestClient

        script = (pathlib.Path(__file__).resolve().parent.parent
                  / "06_FastAPI" / "01_最小应用.py")
        spec = importlib.util.spec_from_file_location("pytest_fastapi_01", script)
        module = importlib.util.module_from_spec(spec)
        sys.modules["pytest_fastapi_01"] = module      # dataclass / pydantic 需要它
        spec.loader.exec_module(module)
        return TestClient(module.app)

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Hello World"

    def test_item_ok(self, client):
        resp = client.get("/items/42")
        assert resp.status_code == 200
        assert resp.json()["item_id"] == 42

    def test_item_type_error(self, client):
        resp = client.get("/items/abc")
        assert resp.status_code == 422

    def test_openapi(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert "/items/{item_id}" in resp.json()["paths"]
