# 代码格式化与质量工具（后端开发基础 · 代码格式化章节）

> 对应课案章节：`后端开发基础 → 代码格式化`（Git Hook / ruff / black / pre-commit）
> ⚠️ 本环境**未安装** ruff / black / isort / pre-commit，所以本章只提供**配置与命令说明**，
> 不提供可运行脚本。等你把工具装上（`uv add --dev ruff black pre-commit`）即可直接使用本目录的配置。

---

## 一、为什么需要自动格式化

代码风格问题（缩进几个空格、行太长、import 顺序、单双引号）**不应该占用人的时间**，
也不该出现在 Code Review 的评论里。正确做法是交给工具：

```
人写逻辑  →  工具统一风格  →  Code Review 只看逻辑
```

三个直接收益：

| 收益 | 说明 |
| --- | --- |
| 消灭无意义的 diff | 不会因为"我习惯两空格你习惯四空格"产生整文件改动 |
| 降低 Review 成本 | 讨论集中在业务逻辑，而不是引号 |
| 提前发现真 bug | Ruff 的规则集包含未使用变量、可变默认参数、`except:` 裸捕获等真实缺陷 |

**分工（重要）：**

| 工具 | 定位 | 会改代码吗 |
| --- | --- | --- |
| **Ruff** | 极快的 Linter + Formatter（Rust 实现），可替代 flake8 / isort / pyupgrade / 部分 black | `ruff check --fix` 会改；`ruff check` 不改 |
| **Black** | "不妥协的格式化器"，只负责排版，几乎不可配置 | `black .` 会改 |
| **isort** | 只排 import 顺序（Ruff 的 `I` 规则已覆盖） | `isort .` 会改 |
| **pre-commit** | 把上面这些挂到 Git 钩子上，提交前自动跑 | 由钩子决定 |

> 实践建议：**要么 Ruff 要么 Black，别同时用两个格式化器**（它们会互相改来改去）。
> 现在主流选择是 **Ruff 一把梭**（`ruff format` 已能替代 black 的绝大多数场景）。

---

## 二、安装

```bash
# 项目模式（推荐，会写入 pyproject.toml 的 dev 依赖组）
uv add --dev ruff black pre-commit

# pip 兼容模式
uv pip install ruff black pre-commit

# 运行（不激活虚拟环境）
uv run ruff --version
uv run black --version
```

---

## 三、常用命令

### Ruff

```bash
uv run ruff check .                 # 只检查，不改代码（CI 里用这个）
uv run ruff check --fix .           # 自动修复能修的问题
uv run ruff check --diff .          # 只显示会改什么，不真的改
uv run ruff check --select F,E9 .   # 只跑"真 bug"类规则
uv run ruff format .                # 格式化（作用类似 black）
uv run ruff format --check .        # 只检查格式，不改（CI 里用这个）
uv run ruff rule --all              # 列出所有规则
uv run ruff check --statistics .    # 按规则统计问题数量
```

### Black

```bash
uv run black .                      # 格式化当前目录
uv run black --check .              # 只检查（CI 用）
uv run black --diff .               # 显示差异
uv run black app.py tests/          # 指定文件/目录
```

### isort（如果单独使用）

```bash
uv run isort .                      # 排序 import
uv run isort --check-only --diff .  # 只检查
```

---

## 四、配置文件说明

本目录提供了两份配置，实际使用时把它们**放到项目根目录**（或 Back_End 目录）即可生效：

### `ruff.toml`

Ruff 会自动向上查找最近的 `ruff.toml` / `.ruff.toml` / `pyproject.toml` 里的 `[tool.ruff]`。
配置要点（详见文件内注释）：

- `line-length = 120`：国内团队常用 120（PEP 8 建议 79，太短了对中文注释不友好）；
- `target-version = "py312"`：按 Python 3.12 的语法特性做升级建议；
- `select`：开启规则集（`E/W` 风格、`F` 逻辑错误、`I` import 排序、`B` bugbear、`UP` 语法升级、`SIM` 简化建议……）；
- `ignore`：关掉与格式化器冲突或团队不接受少数规则（例如 `E501` 行长由 formatter 管）；
- `[lint.per-file-ignores]`：测试文件允许 `assert`、允许未使用变量（`_` 前缀）；
- `[format]`：`ruff format` 的风格参数（引号、缩进）。

### `.editorconfig`

EditorConfig 是**编辑器层面的统一约定**（VS Code / PyCharm / Vim 都支持）：

- `indent_style = space`、`indent_size = 4`：Python 用 4 空格；
- `end_of_line = lf`：强制 LF，避免 Windows 换行符混入 Git；
- `charset = utf-8`：中文注释不乱码；
- `insert_final_newline = true`：文件末尾留一个空行（POSIX 约定，很多工具要求）；
- `trim_trailing_whitespace = true`：删除行尾空格；
- 对 `*.md` 关闭行尾空格清理（Markdown 里两个空格表示换行）；对 `Makefile` 强制 Tab。

---

## 五、pre-commit：提交前自动检查（课案重点）

课案给出的是 `local` 钩子写法。把它保存为项目根目录的 `.pre-commit-config.yaml`：

```yaml
repos:                              # 定义要使用的仓库
  - repo: local                     # local 表示本项目定义的钩子（而不是从远程仓库引用）
    hooks:                          # 在这个仓库中定义的钩子列表
      - id: ruff-check              # 钩子的唯一标识符
        name: ruff check            # 在输出中显示的名称
        entry: uv run ruff check --fix   # 要执行的命令
        language: system            # language: system 表示直接运行系统命令，不使用虚拟环境
        types: [python]             # 只对 Python 文件生效

      - id: ruff-format
        name: ruff format
        entry: uv run ruff format    # 格式化代码
        language: system
        types: [python]

      - id: pytest
        name: pytest
        entry: uv run pytest
        language: system
        pass_filenames: false
        types: [python]
```

**字段含义逐条解释：**

| 字段 | 含义 |
| --- | --- |
| `repos` | 钩子来源列表。`repo: local` 表示钩子命令由本仓库自己定义 |
| `repo` | 远程仓库地址（如 `https://github.com/astral-sh/ruff-pre-commit`）或 `local` |
| `rev` | 远程仓库的版本（tag/commit），`local` 时不需要 |
| `id` | 钩子标识，`pre-commit` 用它来安装/跳过（`SKIP=ruff-check git commit`） |
| `name` | 输出里显示的名字 |
| `entry` | 真正执行的命令 |
| `language` | `system` 用系统命令；`python` 会为钩子创建独立虚拟环境 |
| `types` | 只对匹配的文件类型运行（`[python]` 只跑 .py） |
| `pass_filenames` | 是否把改动文件名追加到命令后面（`pytest` 一般设为 false） |
| `files` / `exclude` | 用正则限定作用范围 |

**常用命令：**

```bash
uv run pre-commit install                    # 安装钩子（此后每次 git commit 前自动运行）
uv run pre-commit run --all-files           # 手动运行所有钩子（第一次接入时跑一次全量）
uv run pre-commit run --files <文件>         # 只对指定文件运行
uv run pre-commit run ruff-check             # 只跑某一个钩子
uv run pre-commit autoupdate                 # 更新远程钩子的版本
uv run pre-commit uninstall                  # 卸载钩子
```

**工作流程：**

```
git commit → 触发钩子 → ruff check --fix → ruff format → pytest
         ├── 全部通过 → 提交成功
         └── 有文件被修改或测试失败 → 提交被拒绝，需要 git add 后重新提交
```

> 小技巧：紧急情况下想跳过钩子可以用 `git commit --no-verify`，
> 但**不要养成习惯** —— 那等于绕过了团队的质量门禁。

---

## 六、CI 里怎么用（GitHub Actions 示例）

```yaml
name: lint-and-test
on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5          # 安装 uv
      - run: uv sync --dev                    # 按 uv.lock 安装依赖
      - run: uv run ruff check .              # 检查但不修改
      - run: uv run ruff format --check .     # 检查格式但不修改
      - run: uv run pytest -q                 # 跑测试
```

CI 里**一定要用 `--check` 而不是 `--fix`**：CI 只负责判断"是否合规"，
自动修改是开发者本地的事。

---

## 七、本目录文件说明

| 文件 | 作用 | 使用方式 |
| --- | --- | --- |
| `README.md` | 本文档：工具讲解 + 命令速查 | 阅读 |
| `ruff.toml` | Ruff 的 lint + format 配置（带中文注释） | 复制到项目根目录，然后 `uv run ruff check .` |
| `.editorconfig` | 编辑器统一约定（缩进/换行/编码） | 复制到项目根目录，编辑器自动识别 |

> 注：本环境未安装 ruff/black，因此这两个配置文件**没有被实际执行过**，
> 它们的正确性依赖 Ruff 的配置规范（已按官方文档逐项注释）。
> 装上工具后运行 `uv run ruff check .` 即可验证。
