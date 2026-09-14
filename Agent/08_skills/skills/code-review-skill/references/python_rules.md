# Python 代码审查规范（33 条）

> 本文件只在审查 Python 代码时才需要读入上下文 —— 这就是「渐进式披露」。

## 一、命名与风格（PEP 8）

1. 模块名 `snake_case`，且尽量短；不要用 `utils.py` 这种什么都往里塞的名字。
2. 类名 `PascalCase`；函数与变量 `snake_case`；常量 `UPPER_SNAKE_CASE`。
3. 私有成员用单下划线前缀 `_helper`；不要用双下划线做「伪私有」。
4. 单行不超过 100 字符（本项目约定）；续行用括号隐式拼接，不要用反斜杠。
5. import 分三组：标准库 / 第三方 / 本项目，组间空一行。
6. 不要 `from x import *`；它会让静态检查失效。
7. 函数之间空两行，类的方法之间空一行。
8. 注释写「为什么」，不复述「做了什么」。

## 二、类型注解

9. 公开函数必须有参数与返回值注解。
10. 用 `list[str]` / `dict[str, int]`（PEP 585 内置泛型），不要 `List` / `Dict`。
11. 允许 `None` 的用 `X | None`，不要用 `Optional[X]`。
12. 用 `Sequence` / `Iterable` 做参数类型，用 `list` 做返回值类型（里氏替换）。
13. 复杂结构用 `TypedDict` 或 `dataclass`，不要一路嵌套 `dict[str, Any]`。

## 三、异常处理

14. 禁止裸 `except:`；至少 `except Exception:`。
15. `except` 后必须做点什么：记录日志、重新抛出、或返回明确的兜底值。
16. 不要用异常做流程控制（比如用 `try/except KeyError` 代替 `if key in d`）。
17. 抛业务异常要带上下文：`raise ValueError(f"uid={uid} 不存在") from e`。
18. 资源获取一律用 `with`（文件、连接、锁）；不要手写 `try/finally` 关文件。

## 四、性能

19. 循环里不要做重复计算：把 `len(x)`、属性查找提到循环外。
20. 拼接大量字符串用 `"".join(parts)`，不要 `s += piece`。
21. 成员判断用 `set` / `dict`，不要用 `list` 做 `in`（O(n) → O(1)）。
22. 生成器优先：能用 `(x for x in ...)` 就不要先建一个 list。
23. 不要过早优化 —— 先测量（`timeit` / `cProfile`），再动手。

## 五、安全

24. 禁止 `eval()` / `exec()` 处理外部输入。
25. 拼接 SQL 一律用参数化查询，不要 f-string 拼表名、字段名。
26. `pickle.loads` 只能用于可信数据；外部数据用 `json`。
27. 口令、密钥、连接串不写进代码，统一走配置/环境变量。
28. 文件路径来自外部输入时，用 `Path.resolve()` 后校验是否越出根目录。

## 六、并发

29. 共享可变状态必须加锁（`threading.Lock`）；GIL 不能保证复合操作的原子性。
30. I/O 密集用 `asyncio` 或线程池，CPU 密集用 `ProcessPoolExecutor`。

## 七、常见陷阱

31. 禁止用可变对象（list / dict / set）做参数默认值 —— 默认值只在函数定义时求值
    一次，会跨调用累积。改成 `def f(bucket=None): bucket = bucket if bucket is not None else []`。
32. 不要在函数里原地修改传入的可变参数；确实需要就先 `copy()`，否则调用方的数据被悄悄改掉。
33. `is` 只用于 `None` / `True` / `False` / 单例；数值与字符串比较一律用 `==`。
