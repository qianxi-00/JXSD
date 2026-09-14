# -*- coding: utf-8 -*-
"""
DeepAgents 运行环境③：FilesystemBackend —— 对接真实磁盘
================================================================
前两个后端（State / Store）的文件都住在「数据库/内存」里，看不见摸不着。
FilesystemBackend 把它们搬到**真实磁盘**上：Agent 调 `write_file` 写出来的
文件，你用资源管理器就能打开。

本节要讲什么
    1. 它**解决什么问题**：让 Agent 直接改你机器上的真实项目文件，
       而不是在内存里自娱自乐 —— 从「虚拟文件系统」跨到「真文件系统」；
    2. 它的能力边界：**它可以「改代码」，但不能「运行代码」**
       （课案的三个反例：python app.py / pytest / npm run build）；
    3. `virtual_mode=True` 划的是什么边界（拦截 `../` 与绝对路径），
       以及它**管不到什么**（管不住 shell、管不住别的后端）；
    4. 什么时候选它：本地项目、CI/CD 改代码、处理真实数据文件；
       什么时候换人：任何需要「跑一下」的任务 → LocalShellBackend / Sandbox。

一、适合干什么（课案原话）
    - 修改本地代码项目
    - 创建配置文件
    - 编辑课程文档
    - CI/CD 自动修改代码
    - 处理真实数据文件

二、它**不能**干什么（本节的核心教学点）
    课案原文列了三条它跑不了的命令：

        python app.py
        pytest
        npm run build

    也就是说，FilesystemBackend 只提供文件操作（ls / read / write / edit /
    glob / grep / delete），**它可以「改代码」，但不能「运行代码」**。
    想跑代码就得换 LocalShellBackend（06）或 Sandbox（08）。

    补充一个容易踩的细节：即便用的是 FilesystemBackend，Agent 的工具清单里
    **依然会看到 `execute`**。原因是 deepagents 统一挂了 execute 工具，
    只在沙箱类后端上才真正生效；非沙箱后端调用它只会返回一条错误提示
    （源码注释原文：*For non-sandbox backends, the `execute` tool will return an error message.*）。
    所以判断「能不能执行」要看后端类型，别只看工具列表。

三、`virtual_mode=True` 是什么意思
    课案注释：`# 拦截 ../ 和绝对路径，防止逃逸出 root_dir`。
    打开后 root_dir 被当成一个「虚拟根目录 /」：
        Agent 请求 /hello.py      → 实际写到 {root_dir}/hello.py
        Agent 请求 ../../etc/xxx  → 直接拒绝
    这是给文件工具划边界；注意**它管不到 shell** —— 如果后端能执行命令
    （LocalShellBackend），命令照样能访问任意路径，virtual_mode 形同虚设。

四、和安全相关的取舍（本文件的临时目录改动）
    课案写的是 `FilesystemBackend(root_dir=".", virtual_mode=True)`，
    也就是把**运行时的当前目录**交给 Agent —— 从仓库根目录运行时就是整个仓库。
    本项目规范要求新文件不干扰既有代码，所以这里改成脚本同级的独立目录
    `Agent/03_deepagents/tmp_jxsd_deepagents_fs/`，其余参数与课案一致。

课案出处：Agent 课案 → deepAgents → 运行环境 → ③ FilesystemBackend

运行方式：
    uv run Agent/03_deepagents/05_后端_Filesystem_jxsd.py

前置条件：.env 里的 api_key / base_url / model_name 可调通（实测可用）。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 用 __file__ 定位，保证从任何工作目录运行都落在同一个地方（课案是 root_dir="."）
WORKDIR = Path(__file__).resolve().parent / "tmp_jxsd_deepagents_fs"

agent = None  # 延迟到 __main__ 里建，方便先建好目录


def build_agent():
    """建 Agent：backend 指向临时目录，virtual_mode=True 拦住路径逃逸。"""
    return create_deep_agent(
        model=llm,
        backend=FilesystemBackend(root_dir=str(WORKDIR), virtual_mode=True),
        system_prompt="你是文件助手，直接在根目录下操作文件，操作完汇报结果。",
    )


if __name__ == "__main__":
    WORKDIR.mkdir(parents=True, exist_ok=True)

    # 放一个初始文件，方便 Agent 有东西可看
    (WORKDIR / "hello.txt").write_text("你好，DeepAgents！\n", encoding="utf-8")

    print(f"Agent 的虚拟根目录：{WORKDIR}")
    print(f"初始磁盘内容：{sorted(p.name for p in WORKDIR.iterdir())}\n")

    agent = build_agent()

    # 课案原文任务：让它「创建 py 文件 → py 文件内容是创建 txt → 执行 py 文件」。
    # 这个任务故意设计成后半段做不到：FilesystemBackend 没有真正的 execute。
    print("===== 执行课案任务 =====")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "创建一个py文件，py文件里面的内容是创建一个txt文件，并执行py文件"}]},
        config={"recursion_limit": 50},
    )
    print(result["messages"][-1].content)

    # ---------- 事后核对磁盘：文件真的写进去了 ----------
    print("\n===== 磁盘实际结果 =====")
    for path in sorted(WORKDIR.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            print(f"  {path.relative_to(WORKDIR)}  ({size} 字节)")

    print(
        "\n结论：py 文件确实被创建到了真实磁盘上，但它是**没有被执行过**的 ——\n"
        "      FilesystemBackend 只有文件工具，没有可用的 execute。\n"
        "      想让它真的跑起来，看 06_后端_LocalShell_jxsd.py 和 08_后端_Sandbox_jxsd.py。"
    )
