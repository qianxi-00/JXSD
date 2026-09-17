"""工程约定的静态回归用例（不需要连库、不需要模型）。

这两条都是"曾经真实发生过、修完之后必须拦住复发"的约定：

1. **模块说明必须真的是 docstring**：RAG 里有 16 个文件把模块说明写在了 `import` 语句
   **之后** —— 那不是 docstring，而是一个被丢弃的字符串表达式（`__doc__` 是 None，
   `help()`/pydoc 拿不到）。已全部归位，这里用 AST 静态拦住"再写回去"。
2. **用到的第三方顶层模块必须声明在依赖里**：`script/langfuse_evaluation.py` 一直在
   `from tqdm import tqdm`，而 `tqdm` 只是别的包的传递依赖、`pyproject.toml` 从没声明过 ——
   换个干净环境就会在跑评估时炸。
"""

import ast
import pathlib
import re

import pytest

RAG_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = RAG_ROOT.parent / "pyproject.toml"


def _python_files():
    return sorted(p for p in RAG_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_module_uses_a_fake_docstring():
    """文件里不能出现"看起来想当 docstring、其实不是首个语句"的模块级字符串。

    判据：文件里存在一个模块级（`tree.body` 直接子节点）的字符串表达式，但它**不是**
    第一个语句，且文件本身没有 docstring。这种写法 100% 是失误 —— 作者以为写了模块说明，
    实际 `__doc__` 是 None。
    """
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        if ast.get_docstring(tree) is not None:
            continue
        for index, node in enumerate(tree.body):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                offenders.append(f"{path.relative_to(RAG_ROOT)}:{node.lineno}（第 {index + 1} 个语句）")
                break
    assert offenders == [], "这些文件的模块说明不是 docstring（挪到文件首个语句即可）：" + ", ".join(offenders)


@pytest.mark.parametrize(
    ("module", "used_in"),
    [
        # 只钉"用到了但差点没声明"的那几个。不做通用推导：模块名 ≠ 发行包名
        # （bs4/beautifulsoup4、cv2/opencv…），通用规则会造出一堆假警报。
        ("tqdm", "script/langfuse_evaluation.py"),
    ],
)
def test_third_party_module_is_declared(module, used_in):
    """`used_in` 里 import 的顶层模块，必须出现在 `pyproject.toml` 的 dependencies 里。"""
    source = (RAG_ROOT / used_in).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert module in imported, f"{used_in} 里没找到 `{module}` 的 import —— 这条参数该删了"

    declared = re.findall(r'^\s*"([A-Za-z0-9_.\-]+)', PYPROJECT.read_text(encoding="utf-8"), flags=re.M)
    names = {item.split("[")[0].split(">")[0].split("=")[0].strip().lower() for item in declared}
    assert module.lower() in names, f"pyproject.toml 的 dependencies 里没有声明 {module}"
