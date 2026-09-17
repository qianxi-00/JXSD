# -*- coding: utf-8 -*-
r"""
DeepAgents 官方补充篇②：自定义 Backend 协议（非课案内容）
================================================================
来源与定位：
    本文件对照 DeepAgents **官方文档** backends.mdx 的「Custom backends」与
    「Add policy hooks」两节，补上课案七种后端课缺的最后一块：
    **课案讲了七种内置后端怎么用，没讲怎么造一个自己的后端。**
    官方出处：/oss/python/deepagents/backends.mdx

为什么不直接用内置的七个？
    内置后端覆盖了绝大多数场景（State/Store/Filesystem/LocalShell/ContextHub/
    Sandbox/Composite），但真实项目常遇到这些需求，只能自己写后端：
      - 文件根本不在磁盘上，而在**数据库 / 对象存储 / 内部配置中心**；
      - 每次读写都要**审计**（谁在什么时候改了哪个文件）或**内容校验**（禁止写入密钥）；
      - 需要**限流**（后端级 QPS 上限）或**只读模式**（给代理一个不能写的知识库）。

本地实测的两个关键事实（写自定义后端前必须知道）：
    A. `BackendProtocol` **没有任何强制抽象方法** —— 18 个方法
       （9 个同步 + 9 个对应的异步 a* 版本）全都带默认实现。
       也就是「想支持什么就重写什么」，没重写的能力会在使用时降级报错，
       而不是在实例化时就炸。
    B. 所有操作都返回**结构化结果对象**（dataclass），失败靠 `error` 字段表达，
       **不抛异常**：WriteResult(error, path) / ReadResult(error, file_data, ...) /
       LsResult(error, entries) / GrepResult(error, matches, truncated) …
       这样模型看到的是「工具返回了一句错误说明」，而不是整个运行崩掉。

缺口表对应：`Agent/官方文档缺口对照.md` 的 **DeepAgents 第 11 项**（自定义 Backend 协议）。

运行方式（项目根目录下，**全离线、0 次真实模型调用**）：
    uv run Agent/03_deepagents/15_自定义后端_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import time

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from deepagents import create_deep_agent
from deepagents.backends import BackendProtocol
from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    LsResult,
    ReadResult,
    WriteResult,
)
# 官方提供的两个小工具（自己手写分页逻辑极易和框架契约不一致，直接用它们最稳）：
#   create_file_data  → 把字符串包装成后端内部的 FileData 结构（dict）
#   slice_read_response → 按 offset/limit 切片并生成合法的 ReadResult（含 1 起算行号）
from deepagents.backends.utils import create_file_data, file_data_to_string, slice_read_response


# ================================================================
# 剧本模型（与 14_上下文治理_官方补充.py 同一手法）
# ================================================================
def ai_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    """构造一条「模型要调工具」的 AIMessage。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class ScriptedModel(ChatOpenAI):
    """按剧本依次吐消息的假模型：只覆写 _generate，其余框架方法沿用真实现。"""

    _script: list = PrivateAttr(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._script[self._cursor]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_scripted(script: list) -> ScriptedModel:
    model = ScriptedModel(model="scripted", api_key="offline", base_url="http://localhost:9")
    model._script = script
    return model


def show_tool_messages(result: dict, limit: int = 100) -> None:
    for message in result.get("messages", []):
        if message.type == "tool":
            body = str(message.content).replace("\n", " ")
            print(f"    [ToolMessage] {getattr(message, 'name', '?')}: {body[:limit]}")


# ================================================================
# Demo 1：最小自定义后端 —— 把「文件」放在内存字典里
# ================================================================
# 自定义后端的套路（官方 backends.mdx）：
#     1. 继承 BackendProtocol；
#     2. 重写你支持的操作，返回对应的 XxxResult 对象；
#     3. 失败时 **返回带 error 的结果**（别抛异常）；
#     4. create_deep_agent(backend=你的实例) 挂上去，文件工具立刻指向它。
# 下面这个后端把内容存在一个 dict 里（真实项目里换成数据库/对象存储的读写即可）。
class DictBackend(BackendProtocol):
    """把文件存在内存字典里的最小后端：只实现 write / read / ls 三个方法。

    这正是官方 backends.mdx 说的用法 ——「想支持什么就重写什么」：
    代理能写、能读、能列目录；glob / grep / edit 等没实现的走父类默认实现。
    （真实项目里把 dict 换成数据库或对象存储的读写即可，方法签名不用变。）
    """

    def __init__(self) -> None:
        # 结构：{"/notes/a.txt": FileData} —— 注意存的是框架的 FileData（dict），
        # 不是裸字符串：用官方 create_file_data() 包一下，读的时候才能交给
        # slice_read_response() 正确分页。
        self.files: dict[str, dict] = {}
        # 审计日志：自定义后端的附加价值之一（内置后端不会帮你记这个）
        self.audit: list[str] = []

    # ---------- 写 ----------
    def write(self, file_path: str, content: str) -> WriteResult:
        self.files[file_path] = create_file_data(content)
        self.audit.append(f"WRITE {file_path} ({len(content)} 字符)")
        return WriteResult(path=file_path, error=None)

    # ---------- 读（分页与行号契约交给官方 helper，别自己算）----------
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if file_path not in self.files:
            # 失败不抛异常，而是把原因塞进 error 字段
            return ReadResult(error=f"文件不存在：{file_path}")
        # slice_read_response 内部会做边界钳制、生成 1 起算的 start_line/end_line，
        # 并在越界时返回带 error 的结果（自己手写会踩 "1 <= start_line" 的校验）
        return slice_read_response(self.files[file_path], offset, limit)

    # ---------- 列目录（非递归，条目结构照官方 FileInfo：path / is_dir / size / modified_at）----------
    def ls(self, path: str) -> LsResult:
        normalized = path if path.endswith("/") else path + "/"
        entries: list[dict] = []
        subdirs: set[str] = set()
        for file_path, file_data in sorted(self.files.items()):
            if not file_path.startswith(normalized):
                continue
            relative = file_path[len(normalized):]
            if "/" in relative:
                # 属于更深层的文件 → 只把「直接子目录」汇总成一条，不递归展开
                subdirs.add(normalized + relative.split("/")[0] + "/")
                continue
            entries.append({
                "path": file_path,
                "is_dir": False,
                # 官方 FileInfo.size 是**字节数**，不是字符数（中文一字 3 字节）
                "size": len(file_data_to_string(file_data).encode("utf-8")),
                "modified_at": file_data.get("modified_at", "") if isinstance(file_data, dict) else "",
            })
        entries.extend({"path": d, "is_dir": True, "size": 0, "modified_at": ""} for d in sorted(subdirs))
        return LsResult(entries=entries)


def demo_1_minimal_backend() -> None:
    print("=" * 70)
    print("Demo 1：最小自定义后端 —— 只实现 write / read / ls")
    print("=" * 70)

    backend = DictBackend()
    # 先手工塞两个文件，方便后面 ls 有东西看
    backend.write("/notes/todo.txt", "买牛奶\n写周报")
    backend.write("/notes/idea.txt", "用自定义后端接内部知识库")

    # ---- Part A：只用我们实现了的三个能力 ----
    agent = create_deep_agent(
        model=make_scripted([
            ai_tool_call("write_file", {"file_path": "/notes/new.txt", "content": "代理写入的内容"}, "c1"),
            ai_tool_call("ls", {"path": "/notes"}, "c2"),
            ai_tool_call("read_file", {"file_path": "/notes/new.txt"}, "c3"),
            AIMessage(content="自定义后端读写都正常。"),
        ]),
        backend=backend,     # ← 关键一行：把文件工具指向我们自己的后端
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "写个文件然后看看目录"}]})
    show_tool_messages(result, limit=120)
    print(f"\n  后端里的文件：{sorted(backend.files)}")
    print("  后端自己记的审计日志：")
    for line in backend.audit:
        print(f"    {line}")
    print(
        "  ↑ 代理根本没碰磁盘 —— write_file / ls / read_file 全部落在我们自己的 dict 上。\n"
        "    真实项目把 dict 换成数据库或对象存储，就是一个「数据库后端」。\n"
        "    注意 audit 这份日志：内置后端不会给你记，这是自定义后端最容易加的价值。"
    )

    # ---- Part B：没实现的方法会怎样？（实测，很重要）----
    print("\n  --- Part B：调用一个我们**没实现**的能力（grep）---")
    probe_backend = DictBackend()
    probe_backend.write("/notes/a.txt", "里面有知识库三个字")
    probe_agent = create_deep_agent(
        model=make_scripted([
            ai_tool_call("grep", {"pattern": "知识库"}, "c1"),
            AIMessage(content="（不该走到这里）"),
        ]),
        backend=probe_backend,
    )
    try:
        probe_agent.invoke({"messages": [{"role": "user", "content": "搜一下知识库"}]})
    except Exception as exc:  # noqa: BLE001
        print(f"    抛错中断：{type(exc).__name__}: {str(exc)[:90]}")
        print(
            "    ↑ 结论（本机实测路径）：**没重写的方法走父类默认实现，调用时直接抛错、把运行打断** ——\n"
            "      因为 ToolNode 默认只把 ToolInvocationError 转成错误消息，其余异常一律 re-raise。\n"
            "      所以两种做法二选一：① 干脆实现它；② 显式重写成「返回带 error 的结果」，\n"
            "      让模型看到一句说明而不是让运行崩掉（Demo 2 的只读后端就是 ② 的写法）。"
        )
    else:
        print(
            "    ↑ 本次调用**成功**了 —— 说明这个版本的父类给该方法补了可用的默认实现。\n"
            "      这也提醒我们：**别按印象断言「某个方法一定没有默认实现」**，以运行结果为准；\n"
            "      但返回值是否可靠仍要看文档（Demo 2 的只读后端是更稳的写法：显式返回 error）。"
        )


# ================================================================
# Demo 2：只读后端 —— 给代理一个「只能看不能改」的知识库
# ================================================================
# 只读是最常见的自定义需求。做法有两种：
#     a) 重写 write/edit/delete，返回 error（工具还在，但一写就被拒）；
#     b) 用权限规则 FilesystemPermission(mode="deny")（14_上下文治理_官方补充.py Demo 2）——
#        那是「不改后端」的做法，适合权限按调用方/路径动态变化的场景。
# 本 Demo 用 a)，顺便看看「写被拒」时模型收到的是什么。
class ReadOnlyBackend(DictBackend):
    """在 DictBackend 基础上禁掉所有写操作：适合把内部文档库暴露给代理。"""

    _REJECT = "该知识库为只读模式，不允许修改"

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=self._REJECT, path=file_path)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return EditResult(error=self._REJECT, path=file_path)

    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(error=self._REJECT, path=file_path)


def demo_2_readonly_backend() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：只读后端 —— 写操作被拒（返回 error 而不是抛异常）")
    print("=" * 70)

    backend = ReadOnlyBackend()
    # 注意：只读后端不能靠 write() 塞数据（会被自己拒掉），要直接往存储里放 ——
    # 但**必须用 create_file_data 包装**，存裸字符串会让后面的读取报
    # TypeError: string indices must be integers（实测踩过）
    backend.files["/wiki/onboarding.md"] = create_file_data("新员工入职指南：第一天领电脑……")

    agent = create_deep_agent(
        model=make_scripted([
            ai_tool_call("read_file", {"file_path": "/wiki/onboarding.md"}, "c1"),
            ai_tool_call("write_file", {"file_path": "/wiki/hack.md", "content": "改一下指南"}, "c2"),
            AIMessage(content="文档读到了，写入被拒（只读库）。"),
        ]),
        backend=backend,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "看看入职指南，顺手改一下"}]})
    show_tool_messages(result, limit=110)
    print(f"\n  后端里的文件（应当没有 hack.md）：{sorted(backend.files)}")
    print(
        "  ↑ 写被拒时，模型收到的是**一句说明**（ToolMessage），运行没有崩 ——\n"
        "    这正是「结构化结果代替异常」的用处：拒绝是业务语义，不是程序错误。"
    )


# ================================================================
# Demo 3：把「审计 + 内容校验」做成策略钩子
# ================================================================
# 官方 backends.mdx 的「Add policy hooks」讲的是在后端层统一做限流/审计/校验 ——
# 相比在每个工具函数里各写一遍，放在后端层的好处是**所有入口都绕不过去**
# （模型的 write_file、edit_file、子代理的写入，最终都走后端的方法）。
# 本 Demo 演示两类策略：
#     1. 内容校验：拒绝写入疑似密钥（正则匹配 sk- 开头的串）；
#     2. 限流：每 N 秒最多 M 次写（超出直接拒绝）。
class PolicyBackend(DictBackend):
    """带内容校验与写入限流的后端。"""

    def __init__(self, max_writes_per_second: int = 2) -> None:
        super().__init__()
        self.max_writes_per_second = max_writes_per_second
        self._write_times: list[float] = []
        self.rejected: list[str] = []

    def write(self, file_path: str, content: str) -> WriteResult:
        # ---- 策略 1：内容校验（禁止把密钥写进文件）----
        import re

        if re.search(r"sk-[A-Za-z0-9]{8,}", content):
            self.rejected.append(f"内容校验拦截 {file_path}")
            return WriteResult(error="检测到疑似 API 密钥，已拒绝写入", path=file_path)

        # ---- 策略 2：写入限流（滑动窗口）----
        now = time.time()
        self._write_times = [t for t in self._write_times if now - t < 1.0]
        if len(self._write_times) >= self.max_writes_per_second:
            self.rejected.append(f"限流拦截 {file_path}")
            return WriteResult(error="写入过于频繁，请稍后再试", path=file_path)
        self._write_times.append(now)

        result = super().write(file_path, content)
        self.audit.append(f"AUDIT {file_path} 写入通过（内容校验 + 限流都过了）")
        return result


def demo_3_policy_hooks() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：策略钩子 —— 内容校验 + 写入限流（都在后端层统一做）")
    print("=" * 70)

    backend = PolicyBackend(max_writes_per_second=2)
    agent = create_deep_agent(
        model=make_scripted([
            ai_tool_call("write_file", {"file_path": "/work/a.txt", "content": "正常内容 A"}, "c1"),
            ai_tool_call("write_file", {"file_path": "/work/b.txt", "content": "正常内容 B"}, "c2"),
            ai_tool_call("write_file", {"file_path": "/work/c.txt", "content": "正常内容 C"}, "c3"),
            # 故意写一条"疑似密钥"的内容来触发策略拦截。
            # 这里用一眼可辨的假 key（真实场景里是用户不小心把真 key 写进文件）；
            # 用 FAKE 字样的另一个好处：不会被本仓库的密钥扫描脚本误报。
            ai_tool_call("write_file", {"file_path": "/work/secret.txt",
                                        "content": "sk-FAKEKEYFORDEMO1234567890"}, "c4"),
            AIMessage(content="有几次写入被策略拦下了。"),
        ]),
        backend=backend,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "写几个文件"}]})
    show_tool_messages(result, limit=80)
    print(f"\n  实际写入成功的文件：{sorted(backend.files)}")
    print(f"  被策略拒绝的记录：{backend.rejected}")
    # 断言兜住：三次写调用必须在 1 秒滑窗内完成，否则限流那条结论就不成立
    # （脚本模型下必然成立；万一机器慢到这个阈值失真，这里会直接报出来而不是印错结论）
    assert len(backend.rejected) == 2, f"预期两类策略各拦一次，实际：{backend.rejected}"
    print(f"  审计日志（通过策略的那几次）：{backend.audit}")
    print(
        "  ↑ 三次正常写入里，第 3 次被**限流**拦下（1 秒内最多 2 次）；\n"
        "    带密钥的那次被**内容校验**拦下。两类策略都写在后端层 ——\n"
        "    无论模型怎么绕（换工具、换子代理），只要落地到文件系统就得过这一关。"
    )


if __name__ == "__main__":
    demo_1_minimal_backend()
    demo_2_readonly_backend()
    demo_3_policy_hooks()
    print("\n全部 Demo 执行完毕（0 次真实模型调用，离线可复现）。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 实测结论（deepagents 0.7.13，本机）：
#    - BackendProtocol 的 __abstractmethods__ 为空 —— 18 个方法（9 同步 + 9 异步）
#      全有默认实现；自定义后端只需重写自己支持的那几个（Demo 1 只写了 write/read/ls）；
#    - **未重写的方法会在调用时抛 NotImplementedError 并打断整个运行**
#      （Demo 1 Part B 实测：调 grep → NotImplementedError: 空消息 → 运行中断）。
#      因为 ToolNode 只把 ToolInvocationError 转成消息，其余（含普通 ToolException）直接 re-raise；
#      要么实现它，要么显式重写成「返回带 error 的结果」（Demo 2 的写法）；
#    - 所有操作返回带 error 字段的 dataclass（WriteResult/ReadResult/LsResult/…），
#      字段**全部有默认值**，所以 `ReadResult(error="...")` 这种写法是合法的；
#    - Demo 2：读正常、写被拒时模型收到「该知识库为只读模式，不允许修改」这句话，
#      运行没有崩、文件也没写进去；
#    - Demo 3：前两次写入通过，第 3 次被限流拦下、带密钥那次被内容校验拦下，
#      两类策略都在后端层生效（审计日志同时记下了通过的那几次）。
# 2. 从零写后端的两个**契约坑**（都实测踩过，写之前先看）：
#    A. 存储里放的必须是框架的 **FileData（dict）**，不能是裸字符串 ——
#       用 `create_file_data(content)` 包一下；直接塞字符串会在读取时报
#       `TypeError: string indices must be integers`。
#    B. 分页/行号别自己算：`ReadResult` 的 `__post_init__` 校验
#       `1 <= start_line <= end_line`（行号 1 起算，而入参 offset 是 0 起算），
#       手写很容易踩；直接用官方 `slice_read_response(file_data, offset, limit)`
#       （在 `deepagents.backends.utils`）最稳。
# 3. 未收录（官方还有、本文件没做的）：
#    - **上传/下载**（upload_files / download_files 及 a* 版本）：用于二进制与多模态
#      文件（图片/PDF）进出后端，本文件未展开，需要时照 write/read 的模式实现；
#    - **SandboxBackendProtocol 的 execute**：比 BackendProtocol 多出「执行命令」能力
#      （对应 execute 工具；课案 06_后端_LocalShell / 08_后端_Sandbox 已讲内置实现），
#      自己造执行器涉及安全边界，本文件不演示；
#    - **MCPAdapter**（tools.mdx）：要起真实 MCP server，可用本仓库 05_mcp 章的服务；
#    - 完整后端建议**继承内置后端**（如 StateBackend）而不是从零写：glob/grep 这类方法
#      内部依赖共享 helper（grep_matches_from_files / _glob_search_files），
#      从零写等于要复刻整套语义 —— 本文件从零写是为了讲清契约，生产请继承。
# 4. 踩坑提示：
#    A. 自定义后端**不要抛异常**：抛了会让整个运行中断，而返回 error 结果只是让模型
#       看到一句拒绝说明。两者体验差别巨大（Demo 1 Part B 与 Demo 2 就是对照）。
#    B. 路径要自己规范：内置后端用 "/" 开头的虚拟路径；自定义后端若直接映射到磁盘，
#       记得防 `../` 越权（FilesystemBackend 的 virtual_mode 就是在做这件事）。
#    C. 失败了也要**如实填 error 字段**：填 None 会被当成成功，模型会以为写进去了。
#    D. 限流/校验这类横切逻辑放在后端层，比散落在工具函数里可靠 ——
#       所有写入入口（含子代理）最终都经过后端方法。
#    E. 只读后端有两种做法：重写方法返回 error（本文件 Demo 2），或用权限规则
#       FilesystemPermission(mode="deny")（14_上下文治理_官方补充.py Demo 2）；
#       前者适合「整个后端都只读」，后者适合「按路径/按调用方动态控制」。
