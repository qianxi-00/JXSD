# -*- coding: utf-8 -*-
"""
Langflow API 调用：用 requests 把可视化工作流当接口用
================================================================
课案出处：Agent 课案 → 工作流 → 可视化平台 → langflow → 使用 → API 调用

本节要讲什么：

    1. 【payload 四字段】Langflow 那个 /api/v1/run/<flow_id> 接口到底要传什么：
       output_type（要什么形态的返回）/ input_type（给的是什么形态的输入）/
       input_value（真正的输入内容）/ session_id（会话标识，决定记不记得上文）——
       这四个字段就是课案图解里那几个输入框，逐个搞清楚含义才会改；
    2. 【调用姿势】POST + {"x-api-key": ...} 请求头 + 流式/非流式两种取结果的差别；
    3. 【课案代码的三个坑】api_key 硬编码、flow_id 硬编码、服务没起时直接抛
       ConnectionError —— 以及本项目是怎么一个个补上的
       （参数外置 → 健康探测 → 失败降级为 dry-run）；
    4. 【工程化写法】命令行参数 > 环境变量 > 默认值的优先级，
       以及为什么密钥只能从环境变量读、且打印时只显示首尾几位；
    5. 【返回结构】Langflow 的响应是嵌套 JSON，真正的话术藏在
       outputs[0].outputs[0].outputs[0].results.message.text —— 本节顺手把它捞出来。

课案在 Langflow 里拖出一个 RAG 问答流程后，生成了访问密钥，然后给出这段调用代码：

    api_key = 'YOUR_API_KEY'
    url = "http://localhost:7860/api/v1/run/你的flow_id"
    payload = {"output_type": "chat", "input_type": "chat", "input_value": "橘醒时代"}
    payload["session_id"] = str(uuid.uuid4())
    headers = {"x-api-key": api_key}
    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()
    print(response.text)

三个字段的作用（课案图解里那三个输入框）：

    | 字段 | 含义 | 常见取值 |
    |---|---|---|
    | output_type | 你要什么形态的返回 | chat（对话）/ text / json / dataframe |
    | input_type  | 你给的是什么形态的输入 | chat（对话）/ text |
    | input_value | 真正的输入内容 | 用户那句话 |
    | session_id  | 会话标识，决定「记不记得上文」 | str(uuid.uuid4())，一个会话一个 |

本节相对课案的工程化改造（都是「不改就跑不稳」的地方）：

    1. api_key / flow_id 从【环境变量或命令行参数】取，绝不硬编码。
       课案里的 'YOUR_API_KEY' 是占位符，本文件也用占位符，不写任何真实口令。
    2. 发请求前先探测 GET {base_url}/health：
         不可达 → 打印中文提示 + 把完整 payload 打印出来（等价于 --dry-run），不抛异常；
         可达   → 才真的 POST。
       课案那段是直接 requests.request，本机没起 Langflow 时会扔
       ConnectionError 出来，教学脚本不能这样。
    3. 真正的密钥只放在环境变量 / .env 里，代码只读不写。

前置条件（任选其一）：
    A. 本机起了 Langflow：见本仓库 Agent/10_workflow_platform/02_平台对比与部署_jxsd.py
       里的安装/启动命令，默认监听 http://localhost:7860
    B. 只想看请求长什么样：直接跑本文件，探测失败会自动走 dry-run

用法：
    uv run Agent/10_workflow_platform/01_langflow_api_jxsd.py
    uv run Agent/10_workflow_platform/01_langflow_api_jxsd.py --input "帮我总结这篇论文"
    uv run Agent/10_workflow_platform/01_langflow_api_jxsd.py --api-key <KEY> --flow-id <ID>
    uv run Agent/10_workflow_platform/01_langflow_api_jxsd.py --dry-run

环境变量（优先级低于命令行参数）：
    LANGFLOW_BASE_URL   默认 http://localhost:7860
    LANGFLOW_API_KEY    访问密钥（Langflow 界面右上角 → Settings → API Keys 生成）
    LANGFLOW_FLOW_ID    流程 ID（流程 URL 里 /flow/<这一段>）
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import argparse
import json
import os
import uuid

# requests 不是标准库，但它是本节的必需品（课案那段也是用它）。
# 用 try/except 包住是为了「缺包时给一句人话」，而不是让 ImportError 糊在屏幕上 ——
# 教学脚本的失败提示本身也是教学内容。
try:
    import requests
except ImportError:  # pragma: no cover —— 本机已装（requests 2.34.2），这里只是保底
    print("缺少依赖 requests，请执行：uv add requests")
    sys.exit(0)


# 课案原文（api_key 处已用占位符；课案写的就是 'YOUR_API_KEY'）
COURSE_CODE = '''import requests
import os
import uuid


api_key = 'YOUR_API_KEY'
url = "http://localhost:7860/api/v1/run/你的flow_id"  # The complete API endpoint URL for this flow


# Request payload configuration
payload = {
    "output_type": "chat",
    "input_type": "chat",
    "input_value": "橘醒时代"
}
payload["session_id"] = str(uuid.uuid4())


headers = {"x-api-key": api_key}


try:
    # Send API request
    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes


    # Print response
    print(response.text)


except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")
'''

# 占位符常量：搜索这两个字符串就能确认「代码里没有真实密钥」
PLACEHOLDER_API_KEY = "YOUR_LANGFLOW_API_KEY"
PLACEHOLDER_FLOW_ID = "YOUR_FLOW_ID"


# ================================================================
# 1. 课案原文
# ================================================================
def section_course() -> None:
    print("=" * 78)
    print("1. 课案原文：Langflow 生成的 API 调用代码")
    print("=" * 78)
    print(COURSE_CODE)
    print("-" * 78)
    print(
        f"""
课案这段代码本身没毛病，但直接抄进项目会踩三个坑：

    坑 1：api_key 硬编码在源码里。真实项目里密钥必须走环境变量，
          否则一旦提交进 Git 就等于泄露（本文件用占位符 {PLACEHOLDER_API_KEY}）。
    坑 2：flow_id 硬编码。开发和线上用的是两个流程 ID，必须能配。
    坑 3：requests.request 会抛 ConnectionError。Langflow 没启动时，
          课案那段会直接崩，而不是给出「请先启动服务」的提示。

本文件把这三个坑都补上：参数外置 + 健康探测 + 失败降级。
"""
    )


# ================================================================
# 2. 参数解析：命令行 > 环境变量 > 默认值
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用 Langflow 的 /api/v1/run/<flow_id> 接口（api_key 与 flow_id 不硬编码）",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LANGFLOW_BASE_URL", "http://localhost:7860"),
        help="Langflow 服务地址，默认取环境变量 LANGFLOW_BASE_URL，再默认 http://localhost:7860",
    )
    parser.add_argument(
        "--flow-id",
        default=os.getenv("LANGFLOW_FLOW_ID", ""),
        help=f"流程 ID，默认取环境变量 LANGFLOW_FLOW_ID；为空则用占位符 {PLACEHOLDER_FLOW_ID}",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LANGFLOW_API_KEY", ""),
        help=f"访问密钥，默认取环境变量 LANGFLOW_API_KEY；为空则用占位符 {PLACEHOLDER_API_KEY}",
    )
    parser.add_argument(
        "--input",
        default="橘醒时代",  # 与课案保持一致的示例输入
        help="input_value，即真正发给流程的内容",
    )
    parser.add_argument(
        "--output-type",
        default="chat",
        choices=["chat", "text", "json", "dataframe"],
        help="output_type，想要什么形态的返回",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP 超时秒数",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要发送的请求，不发出去",
    )
    return parser.parse_args()


def build_payload(input_value: str, output_type: str) -> dict:
    """构造课案那段 payload —— 注意这里就是课案四个字段的照抄。"""
    payload = {
        "output_type": output_type,   # 要什么形态的返回
        "input_type": "chat",         # 给的是对话形态的输入
        "input_value": input_value,   # 真正的输入内容
    }
    # session_id 每次调用都新生成一个 uuid4：
    # 同一个 session_id 的多次调用会被 Langflow 当成同一轮会话（带上文记忆），
    # 想连续对话就把这个值固定下来复用 —— 工程上一般由业务层的会话 ID 决定。
    payload["session_id"] = str(uuid.uuid4())
    return payload


# ================================================================
# 3. 健康探测：先问 /health，再决定发不发请求
# ================================================================
def probe_health(base_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """探测 Langflow 是否活着。

    返回 (是否可达, 说明文字)。
    /health 是 Langflow 自带的健康检查端点，未鉴权、开销极小，适合做前置探测。
    """
    url = f"{base_url.rstrip('/')}/health"
    try:
        response = requests.get(url, timeout=timeout)
        # 能拿到响应就算「服务活着」，哪怕状态码不是 200 ——
        # 有了响应说明进程在监听，后面真实的 POST 失败会给出更具体的错误码。
        return True, f"GET {url} → HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        # 最常见的失败：Langflow 没启动，TCP 连接直接被拒
        return False, f"GET {url} → 连接被拒绝（服务没起来）"
    except requests.exceptions.Timeout:
        # 服务在，但卡住了（比如正在初始化数据库）—— 和「没起来」要分开报
        return False, f"GET {url} → 超时（{timeout} 秒内没有响应）"
    except requests.exceptions.RequestException as exc:
        # RequestException 是上面两个的【父类】，必须放在最后兜底：
        # 一旦写在前面，ConnectionError / Timeout 就永远轮不到，报错会变模糊。
        return False, f"GET {url} → {type(exc).__name__}: {exc}"


# ================================================================
# 4. 打印「将要发送 / 已发送」的请求（dry-run 与真实调用共用）
# ================================================================
def print_request_preview(method: str, url: str, headers: dict, payload: dict) -> None:
    print("  ── 请求预览 ──")
    print(f"    {method} {url}")
    for key, value in headers.items():
        print(f"    Header: {key}: {value}")
    print("    Body（JSON，中文不转义，方便肉眼核对）：")
    body = json.dumps(payload, ensure_ascii=False, indent=6)
    for line in body.splitlines():
        print("      " + line)


# ================================================================
# 5. 主流程
# ================================================================
def main() -> int:
    # 参数优先级顺序就写在这里：命令行 > 环境变量 > 硬编码默认值。
    # argparse 的 default 直接读 os.getenv，所以「没写 --api-key 就用环境变量」是自动的。
    args = parse_args()
    base_url = args.base_url.rstrip("/")   # 去掉末尾斜杠：否则拼出 //health、//api/v1/run
    # 空值一律退回占位符：这样 dry-run 打印出来的请求长什么样，一眼就能看出哪里还没配
    flow_id = args.flow_id or PLACEHOLDER_FLOW_ID
    api_key = args.api_key or PLACEHOLDER_API_KEY

    print("=" * 78)
    print("2. 参数来源与本次调用的目标")
    print("=" * 78)
    print(f"    base_url    : {base_url}      （--base-url / LANGFLOW_BASE_URL）")
    print(f"    flow_id     : {flow_id}      （--flow-id / LANGFLOW_FLOW_ID）")
    # 密钥一律只显示前后几位，避免整串口令出现在终端/日志里
    shown_key = api_key if api_key.startswith("YOUR_") else f"{api_key[:4]}...{api_key[-4:]}"
    print(f"    api_key     : {shown_key}      （--api-key / LANGFLOW_API_KEY，只显示首尾）")
    print(f"    input_value : {args.input}")
    print(f"    output_type : {args.output_type}")
    print(f"    dry_run     : {args.dry_run}")

    url = f"{base_url}/api/v1/run/{flow_id}"
    payload = build_payload(args.input, args.output_type)
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    print()
    print("=" * 78)
    print("3. 健康探测：先确认服务在不在")
    print("=" * 78)
    alive, detail = probe_health(base_url)
    print(f"    {detail}")
    print(f"    → {'服务可达，继续真实调用' if alive else '服务不可达，降级为 dry-run'}")

    # ---------- 分支 A：不可达 / 用户显式 dry-run ----------
    if args.dry_run or not alive:
        print()
        print("=" * 78)
        print("4. DRY-RUN：只打印将要发送的请求，不真的发出去")
        print("=" * 78)
        print_request_preview("POST", url, headers, payload)
        print()
        if not alive and not args.dry_run:
            print("=" * 78)
            print("5. 中文排障提示：Langflow 还没起来")
            print("=" * 78)
            print(
                f"""
    探测结果：{detail}

    这不是代码错误 —— 是 Langflow 服务没启动。按下面顺序处理：

    1) 装（未装过才需要，conda 环境方式）：
           conda create -n langflow python=3.11 -y
           conda activate langflow
           python -m pip install --upgrade uv
           uv pip install --python "$env:CONDA_PREFIX\\python.exe" langflow
           langflow --version

    2) 起（每次要用之前）：
           conda activate langflow
           $env:LANGFLOW_SSRF_PROTECTION_ENABLED = "false"
           $env:LANGFLOW_SSRF_ALLOWED_HOSTS = "localhost,127.0.0.1"
           langflow run --host 127.0.0.1 --port 7860

    3) 在浏览器打开 {base_url} ，拖一个流程出来，然后：
           · 流程页 URL 里的 /flow/<这一串> 就是 flow_id
           · 右上角 Settings → API Keys 生成密钥，填进环境变量：
                 $env:LANGFLOW_FLOW_ID = "<你的 flow_id>"
                 $env:LANGFLOW_API_KEY = "{PLACEHOLDER_API_KEY}"
           或者直接用命令行参数：--flow-id <ID> --api-key <KEY>

    4) 服务起来后重跑本文件，就会走真实调用分支。

    详细部署命令（含生产环境 docker compose 方式）见同目录的
    02_平台对比与部署_jxsd.py。
"""
            )
        else:
            print("    （--dry-run 指定，跳过真实请求。要去掉这个参数并确保 Langflow 已启动。）")
        return 0

    # ---------- 分支 B：服务可达，真的发请求 ----------
    print()
    print("=" * 78)
    print("4. 真实调用：POST " + url)
    print("=" * 78)
    print_request_preview("POST", url, headers, payload)
    print()

    try:
        response = requests.request("POST", url, json=payload, headers=headers, timeout=args.timeout)
        response.raise_for_status()  # 4xx/5xx 直接抛，交给下面统一处理
    except requests.exceptions.HTTPError as exc:
        print(f"  ！HTTP 错误：{exc}")
        print(f"    HTTP {response.status_code}，响应体：")
        print("    " + response.text[:800])
        if response.status_code in (401, 403):
            print(f"    → 密钥不对或没传。检查 --api-key / LANGFLOW_API_KEY（占位符是 {PLACEHOLDER_API_KEY}）。")
        elif response.status_code == 404:
            print(f"    → flow_id 不存在。检查 --flow-id / LANGFLOW_FLOW_ID（占位符是 {PLACEHOLDER_FLOW_ID}）。")
        return 1
    except requests.exceptions.RequestException as exc:
        print(f"  ！请求失败：{type(exc).__name__}: {exc}")
        return 1

    print(f"  ✅ HTTP {response.status_code}")
    print("  ── 原始响应体（课案就是 print(response.text)）──")
    print("  " + "-" * 74)
    for line in response.text[:2000].splitlines():
        print("  | " + line)
    print("  " + "-" * 74)

    # Langflow 的返回是嵌套 JSON，真正的话术藏在 outputs[0].outputs[0].outputs[0].results.message.text
    # 这里顺手把它捞出来，免得每次都要肉眼在 JSON 里翻。
    print("  ── 试着直接取出回答文本 ──")
    try:
        data = response.json()
        text = (
            data.get("outputs", [{}])[0]
            .get("outputs", [{}])[0]
            .get("outputs", [{}])[0]
            .get("results", {})
            .get("message", {})
            .get("text")
        )
        print("  " + (text if text else "（这个流程的返回结构不是标准 chat 输出，请看上面的原始响应体）"))
    except ValueError:
        print("  （响应不是 JSON，跳过解析）")
    return 0


if __name__ == "__main__":
    section_course()
    sys.exit(main())
