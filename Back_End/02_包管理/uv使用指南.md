# uv 使用指南（后端开发基础 · 包管理章节）

> 对应课案章节：`后端开发基础 → 包管理`（为什么需要包管理工具 / uv 介绍 / 管理 Python 版本 / 管理虚拟环境 / pip 兼容模式 / 项目管理 / uv run / 完整工作流）
> 本文是完整中文讲解 + 命令速查，覆盖课案全部小节。

---

## 一、为什么需要新的包管理工具

`pip` 是 Python 自带的包管理器，简单好用，但在真实项目里会暴露三个问题：

| 问题 | 具体表现 |
| --- | --- |
| **慢** | 依赖解析是纯 Python 实现，装几十个包要等好几分钟 |
| **依赖冲突难解决** | 解析器不强，经常装出一个"能装上但跑不起来"的环境 |
| **无法管理项目环境** | 只管"包"，不管"Python 版本"，也不锁版本，A 装完 B 装结果不一致 |

**uv 是更现代的解决方案。**

### uv 是什么

**uv 是用 Rust 编写的极速 Python 包管理器，比 pip 快 10-100 倍**，由 Astral（ruff 的团队）开发。它一个工具同时替代了：

| 被替代的工具 | uv 对应能力 |
| --- | --- |
| `pip` | `uv pip install`（兼容模式） |
| `pip-tools` | 锁文件 `uv.lock` |
| `virtualenv` / `venv` | `uv venv` |
| `pyenv` | `uv python install` |
| `poetry` / `pdm` | `uv add` / `uv sync` 项目模式 |
| `pipx` | `uv tool run`（`uvx`） |

**安装：**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows（网络受限时需要代理）
winget install uv
# 或用官方 PowerShell 脚本
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 验证安装
uv --version
```

---

## 二、管理 Python 版本

uv 内置 Python 版本管理（相当于把 pyenv 也替掉了）。它会从 astral 的镜像下载独立的 CPython 构建，不污染系统 Python。

### 查看可用的 Python 版本

```bash
uv python list
```

输出类似：

```
pypy-3.8.16-windows-x86_64-none                      <download available>
cpython-3.9.13-windows-x86_64-none                   E:\anaconda3\python.exe
cpython-3.12.12-windows-x86_64-none                  C:\Users\...\python.exe
```

- `<download available>`：可以联网安装，但本地还没有；
- 后面显示具体路径：**已经安装**并可在本机使用（uv 能找到系统里的 Python）。

### 安装 / 指定版本

```bash
# 安装特定版本
uv python install 3.12

# 安装多个版本
uv python install 3.11 3.12 3.13

# 设置全局默认版本
uv python default 3.12

# 为当前项目固定版本（推荐）
uv python pin 3.12
```

`uv python pin 3.12` 会在当前项目下生成 `.python-version` 文件，内容就是版本号：

```
3.12
```

**这个文件要提交到版本库**：其他开发者克隆项目后，uv 会自动使用相同的 Python 版本，避免"我这儿能跑你那儿不行"。

### 查询当前使用的 Python

```bash
uv python list --only-installed   # 只显示已安装的
uv python find                    # 显示当前项目实际使用的 Python 路径
uv python uninstall 3.10          # 卸载某个版本
```

---

## 三、管理虚拟环境

**为什么要虚拟环境：** 多个项目依赖同一库的不同版本（项目 A 要 fastapi 0.100，项目 B 要 0.141），装在全局必然打架。虚拟环境就是给每个项目一份独立的 `site-packages`。

```bash
# 在当前目录创建名为 .venv 的虚拟环境（使用 .python-version 指定的版本）
uv venv

# 使用指定 Python 版本创建
uv venv --python 3.12

# 指定虚拟环境名称/路径
uv venv myenv
```

**激活虚拟环境：**

```bash
# macOS / Linux
source .venv/bin/activate

# Windows（PowerShell）
.venv\Scripts\activate

# Windows（CMD）
.venv\Scripts\activate.bat
```

激活后命令行前面会出现 `(.venv)` 前缀，此时 `python` / `pip` 都指向虚拟环境内的。

**退出：**

```bash
deactivate
```

**删除：** 直接删掉 `.venv` 目录即可（它就是个文件夹，删掉不影响代码）。
`rm -rf .venv`（Linux）/ `Remove-Item -Recurse -Force .venv`（PowerShell）。

### 日常开发技巧：不激活也能跑

实际开发中可以用 `uv run` 直接运行脚本，**无需手动激活**：

```bash
uv run python main.py              # 自动使用项目 Python 版本和环境
uv run pytest                      # 自动使用项目的 pytest
uv run python -c "print('hello')"  # 快速测试
```

`uv run` 会先确保环境与 `uv.lock` 一致（必要时自动同步），然后执行命令。

> **PyCharm 注意**：使用 uv 环境时，解释器要选择项目里的 `.venv\Scripts\python.exe`（venv），**不要选择 uv 本身**，否则会出现无法引包的问题。

---

## 四、包管理（pip 兼容模式）

uv 提供了与 pip 完全兼容的命令接口（多一层 `uv pip`），可以直接替换已有工作流中的 pip 命令。

```bash
# ---------- 安装 ----------
uv pip install requests                     # 最新版本
uv pip install requests==2.31.0             # 指定版本
uv pip install "requests>=2.31.0,<3.0.0"    # 版本范围
uv pip install -r requirements.txt          # 从依赖文件批量安装
uv pip install fastapi uvicorn sqlalchemy   # 一次装多个

# ---------- 升级 ----------
uv pip install --upgrade requests
uv pip install --upgrade -r requirements.txt

# ---------- 卸载 ----------
uv pip uninstall requests
uv pip uninstall requests pandas numpy       # 卸载多个

# ---------- 查看 ----------
uv pip list                                 # 列出所有已安装的包
uv pip list --outdated                      # 列出有更新的包
uv pip show requests                        # 查看某个包的详情
uv pip check                                # 检查依赖冲突

# ---------- 导出依赖 ----------
uv pip freeze > requirements.txt
```

### 国内镜像加速

```bash
# 单次命令指定镜像（清华）
uv pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或设置环境变量（长期生效）
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple        # Linux/macOS
$env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"      # Windows PowerShell
```

> ⚠️ **重要区别**：`uv pip install` 只改环境，**不会**写进 `pyproject.toml`。
> 项目模式下应该用 `uv add`，否则别人 `uv sync` 时不会有这个依赖。

---

## 五、项目管理（推荐方式）

uv 支持以 `pyproject.toml` 为中心的现代项目管理方式，比 pip 模式更推荐，尤其适合团队协作和多环境部署。

### 初始化项目

```bash
uv init my_project
cd my_project
```

生成的基本结构：

```
my_project/
├── pyproject.toml    # 项目配置和依赖声明
├── .python-version   # 固定 Python 版本
├── README.md
└── main.py
```

`pyproject.toml` 关键片段：

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []          # 生产依赖，由 uv add 自动维护

[dependency-groups]
dev = []                   # 开发依赖（uv add --dev）

[tool.uv]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"   # 国内镜像
```

### 添加和移除依赖

```bash
uv add requests                     # 添加生产依赖
uv add "requests>=2.31.0"           # 指定版本约束
uv add fastapi uvicorn pydantic     # 一次加多个
uv add --dev pytest ruff black      # 加开发依赖（只在开发环境用）
uv remove requests                  # 移除依赖
```

`uv add` 会**同时**更新 `pyproject.toml` 和 `uv.lock`，并安装到 `.venv`。

### 安装项目全部依赖

```bash
uv sync
```

`uv sync` 说明：类似于 `pip install -r requirements.txt`，它会根据 `pyproject.toml` 和 `uv.lock` 安装所有依赖，确保环境与其他开发者完全一致（默认还会删掉多余包，让环境"精确等于"锁文件）。

### 生成 / 更新锁文件

```bash
uv lock                # 解析 pyproject.toml 并生成/更新 uv.lock
uv lock --upgrade      # 在允许范围内升级并重写锁文件
```

**锁文件说明：**

- `uv.lock` **应该提交到版本库**；
- 它记录了每个包的确切版本 + 哈希，确保团队所有成员使用完全相同的依赖版本；
- 当团队成员运行 `uv sync` 时，会严格安装 `uv.lock` 中指定的版本；
- **只有手动运行 `uv lock`（或 `uv add`/`uv sync --upgrade`）才会更新锁文件**，普通 `uv sync` 不会偷偷升级。

### 查看依赖树

```bash
uv tree                    # 树状显示依赖关系
uv tree --depth 1          # 只看第一层
```

### 项目模式 vs pip 模式对比

| 特性 | pip 模式 | uv 项目模式（推荐） |
| --- | --- | --- |
| 依赖声明 | requirements.txt | pyproject.toml |
| 锁文件 | 无 | uv.lock |
| 版本一致性 | 不保证 | 保证 |
| 团队协作 | 容易版本不一致 | 完全一致 |
| 添加依赖 | 手动编辑 | `uv add` 自动管理 |
| 移除依赖 | 手动编辑 | `uv remove` 自动管理 |
| 安装依赖 | `pip install -r` | `uv sync` |

---

## 六、运行脚本（uv run）

`uv run` 可以无需手动激活虚拟环境直接运行脚本或命令，uv 会自动找到并使用正确的环境：

```bash
uv run main.py                    # 直接运行 Python 脚本（自动使用项目环境）
uv run python main.py             # 显式写 python 也可以
uv run pytest                     # 运行项目中的测试
uv run uvicorn main:app --reload  # 运行第三方命令行工具
uv run --with rich python -c "import rich; print(rich.__version__)"   # 临时加一个依赖，不写入项目
```

**使用 `uv run` 相比手动激活环境的优势：**

- 不会误用错误的 Python 版本；
- 不会忘记激活环境（也就不存在"装到全局去了"）；
- 团队成员无需关心环境配置，开箱即用；
- CI/CD 里不用写"激活环境"的平台相关脚本（Windows 是 `.bat`，Linux 是 `source`）。

---

## 七、完整开发工作流示例

```bash
# 1. 创建新项目
uv init my_api
cd my_api

# 2. 设置 Python 版本
uv python pin 3.12

# 3. 添加依赖
uv add fastapi uvicorn sqlalchemy pydantic

# 4. 添加开发依赖
uv add --dev pytest ruff black

# 5. 安装所有依赖（clone 别人的项目也执行这一步）
uv sync

# 6. 运行开发服务器
uv run uvicorn main:app --reload

# 7. 运行测试
uv run pytest

# 8. 代码检查与格式化
uv run ruff check .
uv run black --check .
```

---

## 八、命令速查卡

| 分类 | 命令 | 作用 |
| --- | --- | --- |
| 版本 | `uv --version` / `uv self update` | 查看/升级 uv 自身 |
| Python | `uv python list` | 列出可用与已安装版本 |
| Python | `uv python install 3.12` | 安装指定版本 |
| Python | `uv python pin 3.12` | 项目固定版本（写 `.python-version`） |
| 环境 | `uv venv` / `uv venv --python 3.12` | 创建虚拟环境 |
| 环境 | `source .venv/bin/activate` | 激活（Linux/macOS） |
| 环境 | `.venv\Scripts\activate` | 激活（Windows） |
| 项目 | `uv init` | 初始化项目 |
| 项目 | `uv add pkg` / `uv add --dev pkg` | 加生产/开发依赖 |
| 项目 | `uv remove pkg` | 移除依赖 |
| 项目 | `uv sync` | 按锁文件同步环境 |
| 项目 | `uv lock` | 生成/更新锁文件 |
| 项目 | `uv tree` | 查看依赖树 |
| pip 兼容 | `uv pip install/list/freeze/uninstall` | 与 pip 一致的包操作 |
| 运行 | `uv run <cmd>` | 在项目环境中执行命令 |
| 工具 | `uv tool install ruff` / `uvx ruff check .` | 安装/临时运行全局 CLI 工具 |

---

## 九、本课案的执行约定（重要）

本仓库（`F:\ProGram\Python_Base`）的虚拟环境已经准备好：`F:\ProGram\Python_Base\.venv`（Python 3.12.12），所有示例脚本**统一直接用 venv 解释器运行**，不执行 `uv run` / `uv sync`：

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Back_End\06_FastAPI\01_最小应用.py'
```

原因：`uv run` 会先尝试与 `uv.lock` 同步，可能改动现有环境。学习阶段直接用 `.venv\Scripts\python.exe` 最稳妥。
上面的 `uv` 命令讲解依然有效——它们是**将来自己建项目时的标准做法**。
