# -*- coding: utf-8 -*-
"""
SKILL 实战：在磁盘上生成一个真能用的技能，并演示「渐进式披露」
================================================================
课案出处：Agent 课案 → skills → 原理（逐步加载机制）/ 目录结构 / SKILL.md

本节要讲什么：

    1. 【动手】真的在磁盘上造出一个可用的技能目录 —— 不是讲概念，而是让
       Agent/08_skills/skills/code-review-skill/ 这一个目录真实存在；
    2. 【渐进式披露实测】把课案的「选择 → 学习 → 使用」三步各跑一遍，
       每一步都打印【字符数 / 行数】，让「省 Token」从一句话变成一个看得见的数字：
           选择 = 只读 front-matter（name + description 几十个字符）
           学习 = 命中技能后才读 SKILL.md 正文
           使用 = 按正文里的路由规则，只读命中的那一个 references/ 文件；
    3. 【两个「为什么」】为什么 SKILL.md 正文要短（它只是路由器）、
       为什么 code-review-skill 要把三份规范拆到 references/ 里；
    4. 【实现细节】课案的 front-matter 是「极小的 YAML 子集」，
       本节用标准库 re 手写一个十几行的解析器，并逐行讲清它为什么这么写，
       以及它【不支持】什么（多行字符串 / 列表 / 锚点 / 流式语法）。

01 小节讲的是「是什么」，本节把它落到磁盘上，生成的技能目录长这样：

    Agent/08_skills/skills/code-review-skill/
    ├── SKILL.md                    ← 10 来行的「路由器」
    └── references/
        ├── python_rules.md         ← 只在审查 Python 时才读
        └── javascript_rules.md     ← 只在审查 JavaScript 时才读

为什么不用 PyYAML？
    规范第 1 节铁律要求不装新依赖。而且 front-matter 是个极小的 YAML 子集
    （只有顶层 `key: value` 和一层缩进的 map），用标准库 `re` 手写十几行就够了。
    本文件里的 parse_front_matter() 只支持本项目会写的形状，
    多行字符串 / 锚点 / 流式语法一律不支持 —— 够用即止，不假装是 YAML 解析器。

和相邻小节的关系：
    01 讲概念与字段表 → 本节造出一个技能 → 03 用 deepagents 加载它 →
    04 把它搬到 PostgreSQL 里 → 05 讲怎么装进 Claude Code。

运行前置条件：无（纯标准库，离线可跑）。
副作用：会新建/覆盖 Agent/08_skills/skills/ 下由本脚本生成的文件。

运行方式：
    uv run Agent/08_skills/02_SKILL示例_jxsd.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 技能父目录：deepagents 的 skills 参数要指向「技能的父目录」，就是这一层
SKILLS_ROOT = HERE / "skills"
SKILL_DIR = SKILLS_ROOT / "code-review-skill"
REFERENCES_DIR = SKILL_DIR / "references"


# ================================================================
# 0. 准备要写入磁盘的技能内容
# ================================================================
# SKILL.md：注意三点 ——
#   1. `---` 必须是第 1 行；
#   2. name 必须和所在目录名（code-review-skill）完全一致；
#   3. 正文要短，它就是课案说的「只有 10 行，充当路由器」——
#      正文里只负责告诉模型「什么语言去读哪个文件」，规范细节全放 references/。
SKILL_MD = """---
name: code-review-skill
description: 审查 Python / JavaScript 代码，检查规范性、性能与安全问题。当用户要求「代码审查 / review / 检查代码 / 优化这段代码」时使用。
trigger_keywords: 代码审查 review 检查代码 规范 性能 安全
license: MIT
compatibility: 需要能读取项目源码目录；Python 3.10+
metadata:
  author: jxsd
  version: "1.0.0"
allowed-tools: read_file write_file
---

# 代码审查技能

你是资深代码审查专家。收到审查请求后，严格按下面的流程执行。

## 第 1 步：判断语言

- 代码是 Python（`.py`，或含 `def` / `import` / 缩进块）→ 读 `references/python_rules.md`
- 代码是 JavaScript / TypeScript（`.js` `.ts`，或含 `const` / `=>` / `function`）→ 读 `references/javascript_rules.md`
- 混合项目 → 两种规范都读，分别给出结论

**只读命中的那一份规范**，不要为了「保险」把两份都读进来。

## 第 2 步：逐条比对

按规范的编号逐条检查，每条给出三样东西：

1. 结论：通过 / 不通过 / 存疑
2. 证据：`文件名:行号` + 原始代码片段
3. 修法：可直接替换的改法，不要写「建议优化」这种空话

## 第 3 步：汇总

输出一张表（严重 / 一般 / 建议三档），最后给一句总体结论。

不要重写整个文件，只给需要改的片段。
"""

# references/python_rules.md —— 课案里说这类规范「200 行」，这里写一份精简可用的版本。
PYTHON_RULES_MD = """# Python 代码审查规范（33 条）

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
"""

# references/javascript_rules.md —— 同理，只有审查 JS 时才读。
JAVASCRIPT_RULES_MD = """# JavaScript / TypeScript 代码审查规范（30 条）

> 本文件只在审查 JavaScript / TypeScript 代码时才需要读入上下文。

## 一、变量与作用域

1. 默认用 `const`；需要重新赋值才用 `let`；永远不要用 `var`。
2. 用 `===` / `!==`，不要用 `==` / `!=`（隐式类型转换是 bug 温床）。
3. 变量声明放在使用点附近，不要把所有声明堆在函数开头。
4. 避免在块级作用域外泄漏变量（尤其 `for` 循环里的闭包）。

## 二、函数与异步

5. 优先箭头函数；需要 `this` 动态绑定时才用 `function`。
6. 参数超过 3 个时改用「选项对象」解构。
7. `async` 函数里每个 `await` 都要能抛出可处理的错误。
8. 不要在 `forEach` 里 `await` —— 用 `for...of`，否则不会等待。
9. 并发请求用 `Promise.all` / `allSettled`，不要串行 `await` 拖慢。
10. 永远不要忘记 `catch`；`Promise` 链结尾必须有 `.catch()`。

## 三、错误处理

11. `throw` 只抛 `Error` 及其子类，不要抛字符串。
12. 自定义错误继承 `Error` 并设置 `this.name`，否则堆栈里认不出来。
13. 顶层用 try/catch 包住事件回调，防止一个异常打断整个流程。

## 四、性能

14. 循环内不要做 DOM 查询；先缓存到变量。
15. 频繁触发的 `scroll` / `resize` / `input` 要防抖（debounce）或节流（throttle）。
16. 大数组用 `map` / `filter` / `reduce`，但不要串成五层链式调用 —— 可读性优先。
17. 避免在渲染函数里创建新对象/新函数（会破坏 memo 优化）。

## 五、安全

18. 不要用 `innerHTML` 渲染外部数据；用 `textContent` 或框架的转义机制。
19. 不要 `eval` / `new Function`。
20. `postMessage` 必须校验 `event.origin`。
21. 敏感 token 不要放 `localStorage`，优先 httpOnly Cookie。

## 六、TypeScript 专项

22. 禁止 `any`；实在不确定用 `unknown` 再收窄。
23. 接口用 `interface`，联合/交叉类型用 `type`。
24. 函数返回值显式标注，尤其是导出函数。
25. 用可选链 `?.` 和空值合并 `??`，不要写 `a && a.b && a.b.c`。

## 七、工程化

26. 文件不超过 400 行；超了就按职责拆分。
27. 一个文件只做一件事；`index.js` 只做再导出。
28. 依赖锁文件（`package-lock.json` / `pnpm-lock.yaml`）必须提交。
29. 不要提交 `console.log`；用统一的 logger。
30. 涉及金额、时间、ID 的运算，注意浮点精度与字符串/数字混用。
"""


# ================================================================
# 1. 落盘：把上面三份内容真的写到磁盘上
# ================================================================
def stage_write_files() -> None:
    print("=" * 78)
    print("1. 生成技能文件（真实写盘）")
    print("=" * 78)

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    # 本技能要落盘的三个文件。写成 dict 而不是三次 write_text，
    # 是为了让「技能包由哪几个文件组成」一眼可见 —— dict 的键就是技能目录的结构。
    # 注意 references/ 里的两份规范【必须】分文件放：合成一份的话，
    # 「只读命中的那一份」就无从谈起了，渐进式披露也就没了。
    files = {
        SKILL_DIR / "SKILL.md": SKILL_MD,
        REFERENCES_DIR / "python_rules.md": PYTHON_RULES_MD,
        REFERENCES_DIR / "javascript_rules.md": JAVASCRIPT_RULES_MD,
    }
    for path, content in files.items():
        # newline="\n"：技能文件是给模型读的纯文本，固定 LF，避免 Windows 下 CRLF
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"  已写入 {path.relative_to(HERE.parent.parent)}  ({len(content)} 字符)")

    print()
    print("  生成后的目录树：")
    _print_tree(SKILLS_ROOT)


def _print_tree(root: Path) -> None:
    """打印目录树（只用于观察，逻辑刻意写得很朴素）。"""

    def walk(path: Path, prefix: str = "") -> None:
        children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        for index, child in enumerate(children):
            is_last = index == len(children) - 1
            branch = "└── " if is_last else "├── "
            if child.is_dir():
                print(f"    {prefix}{branch}{child.name}/")
                walk(child, prefix + ("    " if is_last else "│   "))
            else:
                text = child.read_text(encoding="utf-8")
                print(f"    {prefix}{branch}{child.name}    # {len(text.splitlines())} 行")

    print(f"    {root.name}/")
    walk(root)


# ================================================================
# 2. 最小 YAML front-matter 解析器（只依赖标准库 re）
# ================================================================
# 匹配文件开头的 `--- ... ---`，并把后面的正文一起捕获。
# 注意用 re.DOTALL，否则 `.` 不匹配换行，多行 front-matter 就抓不到。
#
# 逐段拆开看这条正则（这是本文件里唯一一处「值得盯着看」的正则）：
#     \A                从文件【开头】匹配，不是从第一行匹配 —— 前面有空行就解析失败，
#                       这正是课案说的「`---` 必须是第 1 行」的机器校验版本
#     ---[ \t]*\r?\n    第一行是三个减号；[ \t]* 允许行尾有空格，\r?\n 同时兼容
#                       LF 与 CRLF（文件可能是别人在 Windows 上编辑过的）
#     (.*?)             捕获组 1 = front-matter 正文；非贪婪，所以它只会吃到
#                       【第一个】单独的 `---` 为止（正文里再出现 `---` 不会被吃掉）
#     \r?\n---[ \t]*\r?\n?   闭合的 `---` 行；末尾 \r?\n? 用 ? 是因为
#                       文件可能恰好以 `---` 结尾、后面没有换行
#     (.*)\Z            捕获组 2 = Markdown 正文，一直吃到文件结尾（\Z）
_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict, str]:
    """把 SKILL.md 拆成 (元数据字典, 正文)。

    支持的 YAML 子集：
        key: value                     → 顶层标量
        key:                           → 顶层 map，缩进两格的子键
          sub: value
    不支持：多行字符串、列表 `- x`、锚点、流式语法。
    本项目自己生成的 SKILL.md 用的就是上面两种形状，够用。

    ⚠️ 这不是通用 YAML 解析器，是「够用即止」的窄解析器：
       front-matter 里写了列表（`allowed-tools:\\n  - read_file`）或
       块标量（`description: |`），这里会静默丢掉那一部分 —— 所以本项目的
       SKILL.md 一律用「空格分隔的标量」或「一层缩进 map」这两种形状。
    """
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        # 报错信息直接告诉读者「文件应该长什么样」，比抛一个 NoneType 崩溃有用得多
        raise ValueError("SKILL.md 必须以第 1 行的 `---` 开始，且以单独一行的 `---` 结束")

    raw_meta, body = match.group(1), match.group(2)   # 组 1 = 元数据，组 2 = 正文
    meta: dict = {}
    # current_key 记录「上一个没有值的顶层 key」。YAML 靠缩进表达父子关系，
    # 而逐行解析时每读到一行缩进内容，必须知道「我是谁的孩子」—— 就是它。
    current_key: str | None = None

    for line in raw_meta.splitlines():
        # 空行与 `#` 注释直接跳过
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        if line[0] in " \t":
            # 缩进行 → 属于上一个 key 的嵌套 map
            if current_key is None:
                # 开头就来缩进行（没有父 key）→ 非法 YAML，忽略而不是崩
                continue
            # 父 key 此刻必须是个 dict；若它之前被解析成标量，就地升级成 dict
            # （YAML 里同名 key 既当标量又当 map 是错的，但这里选择宽容处理）
            if not isinstance(meta.get(current_key), dict):
                meta[current_key] = {}
            # partition(":") 而不是 split(":")：只按【第一个】冒号切一刀，
            # 这样值里带冒号（比如 URL、时间 12:30）也不会被切坏
            sub_key, _, sub_value = line.strip().partition(":")
            # 顺手剥掉 YAML 里常见的引号：version: "1.0.0" → 1.0.0
            meta[current_key][sub_key.strip()] = sub_value.strip().strip('"').strip("'")
            continue

        key, sep, value = line.partition(":")
        if not sep:
            # 整行没有冒号 → 不是合法的 key: value，忽略（宽容策略，不抛异常）
            continue
        key = key.strip()
        value = value.strip()
        if value:
            # 顶层标量：顺手去掉 YAML 里常见的引号
            meta[key] = value.strip('"').strip("'")
            # 标量没有子键，后面若来缩进行就不该再挂到它下面
            current_key = None
        else:
            # `key:` 后面没有值 → 准备接收缩进的子键
            meta[key] = {}
            current_key = key

    return meta, body


def as_list(value: str | dict | None) -> list[str]:
    """把 `trigger_keywords: a b c` 这类空格/逗号分隔的标量转成列表。"""
    if not isinstance(value, str):
        return []
    return [item for item in re.split(r"[\s,，、]+", value) if item]


# ================================================================
# 3. 第一步「选择」：只读 front-matter，正文一个字都不进上下文
# ================================================================
def stage_select() -> list[dict]:
    print()
    print("=" * 78)
    print("2. 第一步「选择」：AI 只读取每个技能的名称和描述（只有几十字符）")
    print("=" * 78)

    skills: list[dict] = []
    for skill_md in sorted(SKILLS_ROOT.glob("*/SKILL.md")):
        full_text = skill_md.read_text(encoding="utf-8")
        meta, body = parse_front_matter(full_text)

        # 这是关键对比：常驻上下文的是 name+description，
        # 而 read_file 命中的才是全文（正文 + 后续 reference）。
        meta_chars = len(meta.get("name", "")) + len(meta.get("description", ""))
        full_chars = len(full_text)
        full_lines = len(full_text.splitlines())

        print(f"  发现技能：{meta.get('name')}")
        print(f"    description : {meta.get('description')}")
        print(f"    license     : {meta.get('license')}")
        print(f"    metadata    : {meta.get('metadata')}")
        print(f"    allowed-tools: {as_list(meta.get('allowed-tools'))}")
        print(
            f"    常驻上下文字符数（name+description）= {meta_chars} 字符；"
            f"SKILL.md 全文 = {full_lines} 行 / {full_chars} 字符"
        )
        print(f"    → 只读元数据可省下 {full_chars - meta_chars} 字符（{100 * (1 - meta_chars / full_chars):.1f}%）")
        print()

        skills.append({"path": skill_md, "meta": meta, "body": body, "full_text": full_text})

    if not skills:
        print("  （没有发现任何技能，请先确认上面的写盘步骤是否成功）")
    return skills


# ================================================================
# 4. 第二步「学习」：命中技能后，才把 SKILL.md 全文加载进来
# ================================================================
def stage_learn(skills: list[dict], user_query: str) -> dict | None:
    print("=" * 78)
    print("3. 第二步「学习」：匹配到技能，才把 SKILL.md 正文加载进来")
    print("=" * 78)
    print(f"  用户输入：{user_query}")

    hit = None
    for skill in skills:
        meta = skill["meta"]
        # 极简的关键词打分：description + trigger_keywords 里出现一个就算命中。
        # 真实的 deepagents 不跑这套算法 —— 它把 name/description 交给大模型，
        # 由模型自己决定用不用。这里用确定性规则是为了让演示结果可复现。
        keywords = as_list(meta.get("trigger_keywords")) + [
            token for token in re.split(r"[，。、；/\s]+", str(meta.get("description", ""))) if len(token) >= 2
        ]
        matched = [kw for kw in keywords if kw and kw in user_query]
        print(f"    候选技能 {meta.get('name')}：命中关键词 {matched or '（无）'}")
        if matched and hit is None:
            hit = skill

    if hit is None:
        print("  没有技能命中 → 一个字的正文都不会加载（这就是省 Token 的来源）")
        return None

    meta = hit["meta"]
    print()
    print(f"  ✅ 命中技能「{meta.get('name')}」，加载 {hit['path'].name} 全文：")
    print(f"     正文 {len(hit['body'].splitlines())} 行 / {len(hit['body'])} 字符"
          f"（对比：常驻的 name+description 只有 "
          f"{len(str(meta.get('name'))) + len(str(meta.get('description')))} 字符）")
    print("  " + "-" * 74)
    for line in hit["body"].splitlines():
        print("  | " + line)
    print("  " + "-" * 74)
    return hit


# ================================================================
# 5. 第三步「使用」：按 SKILL.md 的指示，只读命中的那个 reference
# ================================================================
def stage_use(skill: dict | None, user_query: str) -> None:
    print()
    print("=" * 78)
    print("4. 第三步「使用」：按 SKILL.md 的指示，只读命中的那个 reference 文件")
    print("=" * 78)

    if skill is None:
        print("  上一步没有命中技能，这里无事可做。")
        return

    # SKILL.md 正文里用相对路径点名了 references/xxx.md，
    # 这里把它们抓出来 —— 相当于模型 read_file 时的候选集。
    declared = re.findall(r"references/[\w./-]+\.md", skill["body"])
    print(f"  SKILL.md 正文里点名的参考文件：{sorted(set(declared))}")

    # 目标语言的判断：和 SKILL.md 里写的规则一致（Python → python_rules.md，以此类推）
    if "python" in user_query.lower() or "py" in user_query.lower():
        target_name = "python_rules.md"
    elif "javascript" in user_query.lower() or "js" in user_query.lower() or "ts" in user_query.lower():
        target_name = "javascript_rules.md"
    else:
        target_name = None

    all_refs = sorted(REFERENCES_DIR.glob("*.md"))
    total_all = sum(len(p.read_text(encoding="utf-8")) for p in all_refs)
    loaded_chars = 0

    for ref in all_refs:
        if target_name is not None and ref.name != target_name:
            print(f"  ✗ 跳过 {ref.name}（本次是 {target_name} 的任务，不加载）"
                  f"  —— 省下 {len(ref.read_text(encoding='utf-8'))} 字符")
            continue
        text = ref.read_text(encoding="utf-8")
        loaded_chars += len(text)
        preview = text.splitlines()[:6]
        print(f"  ✅ 加载 {ref.name}（{len(text.splitlines())} 行 / {len(text)} 字符），前几行：")
        for line in preview:
            print("      " + line)
        print("      ...")

    print()
    print("  【Token 账本】")
    print(f"    references/ 下全部参考文件合计 : {total_all} 字符")
    print(f"    本次实际加载                  : SKILL.md 正文 {len(skill['body'])} 字符"
          f" + {loaded_chars} 字符 = {len(skill['body']) + loaded_chars} 字符")
    print(f"    节省                          : {total_all - loaded_chars} 字符"
          f"（{100 * (1 - loaded_chars / total_all):.1f}% 的参考内容没有进上下文）")


# ================================================================
# 6. 小结 / 与 deepagents 的对应关系
# ================================================================
def section_summary() -> None:
    print()
    print("=" * 78)
    print("5. 小结：本节演示的三步，对应真实框架里的哪段代码")
    print("=" * 78)
    print(
        f"""
    本节做的事情（纯标准库、结果可复现）        真实 deepagents 里对应的实现
    ------------------------------------------------------------------------
    1. 选择：glob('*/SKILL.md') + 只解析 front-matter
                                             SkillsMiddleware.before_agent
                                             遍历 skills 源目录 → download_files
                                             取每个 SKILL.md → 只留元数据拼进系统提示词
    2. 学习：关键词命中后打印 SKILL.md 正文 模型自己调 read_file(SKILL.md, limit=1000)
    3. 使用：按正文里的 references/ 路径读模型自己决定再 read_file 哪个参考文件

    结论：技能作者要做的，就是把「什么时候读哪个文件」写清楚 ——
    框架只负责把 name/description 摆在模型面前，剩下的靠模型的判断力，
    以及你在 SKILL.md 里写的路由规则。

    生成的技能目录（下一节 03_deepagents技能_jxsd.py 会用它）：
        {SKILLS_ROOT}
    运行 03 之前请先跑本文件，否则技能的父目录是空的。
"""
    )


if __name__ == "__main__":
    stage_write_files()

    # 只解析元数据，不读正文 —— 这一步对应 agent 启动时的「技能选择」
    all_skills = stage_select()

    # 换一个查询就能看到「命中 JS 规范、不读 Python 规范」的对称效果：
    #   python 版： "帮我审查这段 Python 代码，看看有没有性能问题"
    #   js 版    ： "帮我 review 这段 javascript 代码"
    query = "帮我审查这段 Python 代码，看看规范性和性能问题"
    hit_skill = stage_learn(all_skills, query)
    stage_use(hit_skill, query)

    section_summary()
