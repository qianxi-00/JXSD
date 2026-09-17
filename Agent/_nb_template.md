# Agent 课程 Notebook 编写规范（所有 subagent 必须遵守）

> 本文件是本次改造的**唯一规范来源**。动手前完整读一遍，交付前用第 7 节的命令自查一遍。
> 样板文件：`Agent/01_langgraph/01_基础图与状态.ipynb`（先读它，照着它的密度写）。

---

## 1. 任务背景

`F:\ProGram\Python_Base\Agent\` 下原有 162 个课程 `.py`，现在要改造成 **32 个 Jupyter Notebook**：

- 原来的中文讲解注释 → **Markdown cell**（标题层级、表格、mermaid 图、官方链接）；
- 原来的代码 → **code cell**，按「一个 cell 讲一件事」切开；
- 原来散在各文件头的运行前置条件 → 收敛到开头的**「运行条件」区块**；
- 同一个 notebook 内**先起服务、后调用**，一条链路走到底，不再各文件互相依赖。

原 `.py` 已归档到 `Agent/_py_source/`（**只读参考，不要改**）。

---

## 2. 交付物与工作方式

**重要：不要手写 `.ipynb` 的 JSON。** 你写的是 **percent 格式的 `.py` 源**，由工具转成 notebook。

| 步骤 | 命令（在仓库根 `F:\ProGram\Python_Base` 下跑） |
|---|---|
| 1. 写源文件 | 路径：`F:\ProGramApp\DSH_Temporary\agent_nb\src\<章>\<notebook 名>.py` |
| 2. 转 notebook | `& .\.venv\Scripts\python.exe Agent\_tools\nbtool.py py2nb F:\ProGramApp\DSH_Temporary\agent_nb\src\<章>\<名>.py Agent\<章>\<名>.ipynb` |
| 3. 结构检查 | `& .\.venv\Scripts\python.exe Agent\_tools\nbtool.py check Agent\<章>\<名>.ipynb` |
| 4. 无头执行 | `& .\.venv\Scripts\python.exe Agent\_tools\run_notebooks.py Agent\<章>\<名>.ipynb` |
| 5. 反向还原（可选） | `& .\.venv\Scripts\python.exe Agent\_tools\nbtool.py nb2py Agent\<章>\<名>.ipynb <还原路径>.py` |

> ⚠️ **第 4 步只跑你自己那一个 notebook（传完整路径）。**
> 同一章会有多个 subagent 并发改造不同的 notebook，如果每人都跑
> `run_notebooks.py <章>`，就会把同章其它 notebook 也带着跑一遍 ——
> 既浪费模型额度，又会因为抢端口 / 抢 `tmp_nb_work` 目录而互相干扰。

percent 格式长这样：

```python
# %% [markdown]
# # 一级标题
# 正文。**每个 markdown cell 的每一行都必须是 `#` 注释。**
#
# | 列1 | 列2 |
# |---|---|
# | a | b |

# %%
print("这是一个 code cell，内容原样保留")
```

> `#` 后面**最多**只去掉一个空格 —— 所以 `#   - 子项` 会得到 `  - 子项`（保住缩进）。
> 不要写 `# %%` 之外的其他 cell 标记；不要在第一个标记之前写任何非空行（会直接报错）。

---

## 3. 固定模板（顺序不要变）

```
[md]  # <课时标题>                    ← 一句话定位 + 全节概念表 + 「由哪几个源文件合并」
[md]  ## 运行条件                      ← 必须有三档标记之一（见第 4 节）
[md]  ## 本节地图                      ← mermaid 图 + 等价表格 + 与上下节的衔接
[md]  ## 0. 环境引导                   ← 固定文案
[py]  bootstrap cell                  ← 逐字照抄第 5 节，不要改
[py]  前置条件自检 cell                ← 检查密钥/端口/包，缺了就打印中文提示并让后续跳过
[md]  ## 1. 课案原版：最短实现（NNN 行）
[py]  ...
[md]  ### 预期输出   ```text ... ```
[md]  ## 2. 完整版：<讲什么>
[md]  ### 2.1 <小节>
[py]  ...
[md]  ### 预期输出   ```text ... ```
... （按原 `_jxsd` 文件的小节继续）
[md]  ## 3. 官方文档补充（来自 `NN_XXX_官方补充.py`；没有就省略整节）
[md]  ## 小结
[md]  ## 常见坑
[md]  ## 官方链接
```

**硬性要求**

1. 每个 code cell 前面**必须**有一个 markdown cell 解释它要干什么；禁止「哑代码」。
2. 原 `.py` 里的中文讲解注释**全部提升为 markdown**；留在 code cell 里的注释只写「为什么」，
   不复述代码。
3. 每个有输出的 code cell 后面**必须**有 `### 预期输出` + ```` ```text ```` 块，
   内容是你**亲自跑出来的真实输出**（不是猜的）。见第 7 节。
4. 至少 3 个 markdown cell（`check` 会卡这一条）。
5. **源 `.py` 的代码一行都不能丢**（有 `coverage` 审计），只允许做第 6 节列出的那几种改写。

---

## 4. 「运行条件」区块的写法

必须用下面这张表 + **三档标记之一**：

```markdown
## 运行条件

| 项 | 说明 |
|---|---|
| 🟡 运行档位 | **需模型** —— 会真实调用 `.env` 里配置的大模型 |
| 依赖 | `langchain` / `langgraph`（venv 已装） |
| 密钥 | `settings.api_key`（已配置） |
| 前置服务 | 无 |
| 预计耗时 | 约 20 秒 |
```

| 标记 | 含义 | 例子 |
|---|---|---|
| 🟢 | **离线可跑**：不读 `.env`、不连模型、不起服务 | `01_基础图`、`04_function_call/tools` |
| 🟡 | **需模型**：要 `.env` 里的 `api_key` / `base_url` / `model_name` | 绝大多数 |
| 🔴 | **需外部服务**：要起 MCP 服务 / Docker / Langfuse / 数据库 | `05_mcp/*`、`06_langfuse/*` |

🔴 的 notebook **必须在 notebook 内自己把服务起起来**（见第 6 节第 5 条），
不要写成「请先在另一个窗口运行 XXX」——那就又回到「完全分离隔离」了。

---

## 5. bootstrap cell（逐字照抄，不要改）

```python
# ===== 环境引导（每个 notebook 的第一格，不要改）=====
import os
import sys
from pathlib import Path

NB_DIR = Path.cwd()                 # notebook 所在目录（chdir 之前先抓住）
ROOT = NB_DIR
while not (ROOT / "config.py").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("没找到 config.py：请在 Python_Base 仓库内运行本 notebook")
    ROOT = ROOT.parent

os.chdir(ROOT)                      # 让相对路径（data/、output.txt 等）都相对仓库根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKDIR = NB_DIR / "tmp_nb_work"    # 本 notebook 的临时工作目录（已被 .gitignore 覆盖）
WORKDIR.mkdir(exist_ok=True)

print("仓库根：", ROOT)
print("临时目录：", WORKDIR)
```

**为什么必须有它**：notebook 的 cwd 是它自己所在的目录，而 `config.py` 在仓库根 ——
少了这一格，后面每个 `from config import settings` 都会 `ModuleNotFoundError`。

> ⚠️ **`WORKDIR` 是同章的共享容器**。你写文件时必须再套一层**本 notebook 专属的子目录**：
> `WORKDIR / "backend_filesystem"`、`WORKDIR / "hitl_demo"` …… 名字取能唯一标识本课的英文短名。
> 否则同一章的两个 notebook 并发执行时会互相踩对方的文件。

---

## 6. 代码改写规则（只允许这 6 种）

1. **删掉文件头 docstring 与 `import sys; sys.stdout.reconfigure(...)`**
   —— 它们的内容已经进了 markdown，而 notebook 内核本来就是 UTF-8。**其余 import 一个都不能少。**
2. **删掉 `if __name__ == "__main__":` 这一层**
   —— notebook 里直接调用。原来缩进在里面的语句**顶格写出**（独立成 cell 更好）。
   ⚠️ 但作为「库」被别的 notebook 引用的模块（如 `04_function_call/tools.py`、`tool_desc.py`）
   要保留函数/常量定义，只把演示调用顶格。
3. **`Path(__file__).resolve().parent` → `NB_DIR`**
   —— notebook 里没有 `__file__`。凡是原来往脚本同级目录写文件的，一律改成
   `WORKDIR / "<本课专属子目录>"`（见第 5 节的提醒，务必套一层专属子目录）。
4. **`input()` → 预设答案函数**
   —— notebook 无头执行时 `input()` 必崩。照抄 `Agent/_py_source/02_langchain/09_人工审核_jxsd.py`
   里 `scripted_input(...)` 的做法：临时替换 `builtins.input`，按顺序吐预设答案，
   **用完必须兜底**（审核轮数由模型决定，写死长度会 IndexError），并在 markdown 里说明
   「真实交互请自己改这一格」。
5. **常驻服务 → 后台进程 + 末尾关停**
   —— `uvicorn.run` / `mcp.run` / `langgraph dev` 会把内核**永久阻塞**。必须改成：

   ```python
   import subprocess, sys, time, os

   env = {**os.environ, "PYTHONUTF8": "1", "NO_PROXY": "127.0.0.1,localhost"}
   server = subprocess.Popen(
       [sys.executable, str(WORKDIR / "server.py")],
       cwd=str(ROOT), env=env,
       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
   )
   # 轮询等端口就绪（不要把 sleep 写死太久）
   for _ in range(60):
       if port_ready(): break
       time.sleep(0.5)
   ```

   然后在**最后一个 cell** 里关掉它（Windows 上要连子进程树一起收）：

   ```python
   if server.poll() is None:
       subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
   ```

   ⚠️ 读 Windows 上刚被 taskkill 的进程持有的文件会有 `WinError 32`：
  删临时目录要**重试 + 容忍失败**（照抄 `Agent/_py_source/03_deepagents/19_异步子代理_官方补充.py`
  的 `remove_temp_dir`）。
6. **`ports` 统一用高位端口**
   —— 原来抢 8000/9000 的，notebook 里改成 8100/9100 一类，避免和别的 notebook 撞。

---

## 7. 交付前必须自测（缺一不可）

```powershell
cd F:\ProGram\Python_Base
# ① 转换
& .\.venv\Scripts\python.exe Agent\_tools\nbtool.py py2nb <你的源>.py Agent\<章>\<名>.ipynb
# ② 结构检查（必须「通过 N，有问题 0」）
& .\.venv\Scripts\python.exe Agent\_tools\nbtool.py check Agent\<章>\<名>.ipynb
# ③ 真跑一遍（必须 PASS 或预期的 PASS-降级）
& .\.venv\Scripts\python.exe Agent\_tools\run_notebooks.py <章>
```

**「预期输出」必须来自你自己跑出来的真实输出。** 想看到具体输出可以临时用：

```powershell
& .\.venv\Scripts\python.exe F:\ProGramApp\DSH_Temporary\agent_nb\src\_dump_outputs.py Agent\<章>\<名>.ipynb
```

**没跑过的不许写「预期输出」。** 如果某个 cell 因为缺前置条件没跑成，
就把那段写成「本机未就绪时会看到的中文提示」，并在 markdown 里说明。

**如果某格的输出里含不确定内容**（模型自己的措辞、时间戳、随机 UUID、本机路径、
对象地址……），必须在**同一个 markdown 格**里明确写出这件事，例如：

```markdown
> ⚠️ 模型措辞每次不同，只有「结构」与「工具入参」是一致的；上面的正文只是我这次跑出来的实测值。
```

为什么要专门写：项目里有 `nbtool.py verify` 会自动核对每段「预期输出」是否能在实跑输出里
找到。它认识上面这类**明确声明不确定**的措辞并跳过比对；不写，就会被当成错误报出来，
让人分不清「真错了」还是「本来就每次不同」。写法不限，讲清楚即可 ——
「由模型决定 / 每次不同 / 随机 / 时间戳 / 实测值 / 不稳定 / 别逐字比对」这些说法都能被识别。

⚠️ **两个实测踩出来的坑，务必避开：**

1. **讲「确定性」时不要出现声明词。** 有位 subagent 在确定性输出的那一格写了
   「所以它**不带『不确定』声明」—— 那三个字反而让整段被判成「已声明非确定」而跳过比对，
   **看起来通过、其实根本没核**。核对器现在会剔掉「不带/不是/并非/没有」这类否定用法，
   但你自己也别去踩这条线：想说「这段是确定的」，就直接说「本段输出确定、可逐字比对」。
2. **验证跑完就不要再跑 `py2nb`。** `nbformat` 每次生成的 **cell id 是随机的**，
   任何一次重新转换都会让「已验过的字节」和「最终交付的字节」不是同一份 —— 验证等于作废。
   正确顺序：`py2nb → check → run_notebooks → verify` **在同一次 shell 调用里按序跑完**。

> 想省一轮 verify 返工，可以先做**静态预演**：不跑 notebook，只看「哪几段会被核对、
> 哪几段会被跳过、命中的是哪个声明词」。`_verify_preview.py`（在 scratch 的 `src\` 下）
> 就是干这个的 —— 改完声明先跑它，比跑完 90 秒的 verify 再返工划算。

> **内存提示**：全量 verify 会串行起内核，每个约 200MB。如果同时还有别的进程在跑，
> 可能偶发 `DeadKernelError: Kernel died` / `MemoryError` —— 那是**环境争抢，不是内容缺陷**，
> 单独重跑一次即可；批量跑建议按章拆开传。

---

## 8. 常见错误（都踩过）

| 症状 | 原因 | 解法 |
|---|---|---|
| `ModuleNotFoundError: config` | 少了 bootstrap cell | 照抄第 5 节，放在第一格 |
| `NameError: __file__` | 源文件里用了 `__file__` | 改成 `NB_DIR` |
| 执行卡住不动 | 某个 cell 里 `uvicorn.run` / `mcp.run` 阻塞了内核 | 按第 6 节第 5 条改成后台进程 |
| `EOFError` / 卡在输入 | 有 `input()` | 按第 6 节第 4 条改成预设答案 |
| 中文乱码 | 没设 UTF-8 | 执行器已经设了 `PYTHONUTF8=1`；自己跑时也设一下 |
| `check` 报「缺少环境引导格」 | bootstrap 里没有 `sys.path.insert` + `config.py` 字样 | 逐字照抄第 5 节 |
| `check` 报「还带着执行输出」 | 提交前忘了 strip | `nbtool.py strip --all` |
| 端口被占 | 两个 notebook 抢同一个端口 | 按第 6 节第 6 条改用高位端口 |
| 连接被拒 / 502 | 本机 Clash 拦了 127.0.0.1 回环 | 执行器已设 `NO_PROXY`；自己跑时也设 |
| `check` 报「语法错误」 | code cell 里用了 IPython 魔法命令 | **不要用 `%magic` / `!shell`**（见下） |

> **不要用 IPython 魔法命令与 shell 转义**（`%time`、`%%capture`、`!pip install`）。
> 原因有两条：① `check` 会对每个 code cell 做 `ast.parse`，魔法命令不是合法 Python；
> ② 魔法命令只有 IPython 认，用 `nbclient` 执行或把代码复制出去单跑都会失败。
> 需要计时就用 `time.perf_counter()`，需要装包就写进「运行条件」让人自己装。

---

## 9. 交付报告格式

```
### 交付
- Agent/<章>/<名>.ipynb   ← 合并自 <源文件列表>
   markdown NN 格 / code NN 格

### 自测
- nbtool.py check  → 通过 1，有问题 0
- run_notebooks.py <章> → [PASS] XX.Xs  <章>/<名>.ipynb

### 与源文件的差异
- <哪段代码改写了、为什么；哪段降级了、为什么>
- <源 .py 里没进 notebook 的内容（如果有）>
```
