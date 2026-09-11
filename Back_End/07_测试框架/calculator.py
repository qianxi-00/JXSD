"""calculator.py —— 被测模块（一个简单但完整的计算器）
================================================================
对应课案章节：后端开发基础 → 测试框架（被测对象）

为什么先写一个"平平无奇"的计算器？
    · 它足够小，能把注意力集中在**测试怎么写**上；
    · 它包含正常分支、边界条件、异常分支，正好覆盖测试的全部套路：
          正常用例（1+2=3）
          边界用例（除以 0、负数开方）
          异常用例（raise ValueError）
    · 它也有"故意的设计取舍"，例如除零是抛异常还是返回 None ——
      这决定了测试该怎么写断言。

本节知识点：
    1. 被测代码要"可测"：纯函数（无副作用）最好测
    2. 显式抛异常优于返回 None/静默失败 —— 更容易被测试发现，也更难被忽略
    3. 类型注解既让测试更好写，也让 IDE 能提前发现调用错误
"""

from __future__ import annotations


class Calculator:
    """四则运算计算器。

    用类来实现，是为了让测试也能演示"测试类 + fixture"的写法
    （见 test_calculator.py 里的 TestCalculator 与 calc fixture）。
    """

    def __init__(self, precision: int = 10) -> None:
        """precision：浮点结果保留的小数位数。

        引入这个参数是为了演示 fixture 的另一个常见用途：
        **给每个测试准备一个配置不同的对象**。
        """
        self.precision = precision
        self.history: list[str] = []      # 运算历史（带状态的对象，需要每个用例独立一份）

    # ---------- 基础运算 ----------
    def add(self, a: float, b: float) -> float:
        """加法。"""
        result = round(a + b, self.precision)
        self._record(f"{a} + {b} = {result}")
        return result

    def sub(self, a: float, b: float) -> float:
        """减法。"""
        result = round(a - b, self.precision)
        self._record(f"{a} - {b} = {result}")
        return result

    def mul(self, a: float, b: float) -> float:
        """乘法。"""
        result = round(a * b, self.precision)
        self._record(f"{a} * {b} = {result}")
        return result

    def div(self, a: float, b: float) -> float:
        """除法：除数为 0 时抛出 ZeroDivisionError。

        **为什么抛异常而不是返回 None？**
            返回 None 会悄悄传播到很远的地方才炸（甚至不炸，只是算错），
            抛异常则会在"出错的第一现场"就暴露，测试也只需要 assert 一行。
        """
        if b == 0:
            raise ZeroDivisionError("除数不能为 0")
        result = round(a / b, self.precision)
        self._record(f"{a} / {b} = {result}")
        return result

    # ---------- 进阶运算 ----------
    def power(self, base: float, exponent: int) -> float:
        """幂运算。"""
        result = round(base ** exponent, self.precision)
        self._record(f"{base} ** {exponent} = {result}")
        return result

    def sqrt(self, value: float) -> float:
        """平方根：负数会抛 ValueError。"""
        if value < 0:
            raise ValueError(f"不能对负数求平方根：{value}")
        result = round(value ** 0.5, self.precision)
        self._record(f"sqrt({value}) = {result}")
        return result

    def average(self, numbers: list[float]) -> float:
        """求平均值：空列表抛 ValueError。"""
        if not numbers:
            raise ValueError("列表为空，无法计算平均值")
        result = round(sum(numbers) / len(numbers), self.precision)
        self._record(f"avg({numbers}) = {result}")
        return result

    # ---------- 辅助 ----------
    def _record(self, text: str) -> None:
        """记录运算历史（内部方法，测试时用 history 断言副作用）。"""
        self.history.append(text)

    def clear_history(self) -> None:
        """清空历史。"""
        self.history.clear()


def is_prime(n: int) -> bool:
    """判断质数（模块级函数，演示"函数级测试 + 参数化"的写法）。"""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def fizzbuzz(n: int) -> str:
    """经典 FizzBuzz，用来演示参数化测试与"边界值"测试。"""
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)


if __name__ == "__main__":
    # 直接运行本文件时，做一次"手动冒烟测试"，方便快速确认代码没写坏
    print("=" * 72)
    print("calculator.py 冒烟测试（正式的自动化测试见 test_calculator.py）")
    print("=" * 72)
    calc = Calculator()
    print(f"  3 + 4      = {calc.add(3, 4)}")
    print(f"  10 - 3.5   = {calc.sub(10, 3.5)}")
    print(f"  2.5 * 4    = {calc.mul(2.5, 4)}")
    print(f"  10 / 4     = {calc.div(10, 4)}")
    print(f"  2 ** 10    = {calc.power(2, 10)}")
    print(f"  sqrt(144)  = {calc.sqrt(144)}")
    print(f"  avg([1,2,3]) = {calc.average([1, 2, 3])}")
    print(f"  is_prime(97) = {is_prime(97)}")
    print(f"  fizzbuzz(15) = {fizzbuzz(15)}")
    print("\n运算历史：")
    for line in calc.history:
        print("   ", line)
    print("\n冒烟测试完成。运行正式测试：")
    print("  & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m pytest "
          "'F:\\ProGram\\Python_Base\\Back_End\\07_测试框架' -q")
