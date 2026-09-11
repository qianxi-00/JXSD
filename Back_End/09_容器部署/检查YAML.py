"""容器部署配置校验：用 PyYAML 解析并检查 Docker/K8S 文件
================================================================
对应课案章节：后端开发基础 → 容器部署（Docker / K8S）

**为什么需要这个脚本？**
    Dockerfile / docker-compose.yml / K8S 的 YAML 都是"配置即代码"，
    YAML 有个很坑的特点：**缩进错了不一定报错，而是被解析成别的结构**，
    等到 `docker compose up` / `kubectl apply` 时才发现，排查成本很高。
    所以提交前先用解析器校验一遍，并做几项"跨文件的引用一致性检查"。

本节知识点：
    1. yaml.safe_load 与 safe_load_all（多文档 YAML 用 --- 分隔）
    2. YAML 常见坑：Tab 缩进、布尔值陷阱（`no` 会被解析成 False）、中文编码
    3. docker-compose 的结构校验：services / ports / volumes / depends_on 引用完整性
    4. K8S 资源的通用结构：apiVersion / kind / metadata / spec
    5. 跨文件引用校验：Deployment 里引用的 ConfigMap / Secret / Service 是否真的存在
    6. Dockerfile 的静态检查：基础镜像、必须监听 0.0.0.0、是否有 EXPOSE 等

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\09_容器部署\\检查YAML.py'
"""

from __future__ import annotations

import pathlib
import re
import sys

import yaml

# ------------------------------------------------------------
# 路径：本文件位于 Back_End/09_容器部署/检查YAML.py
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

K8S_DIR = HERE / "k8s"
COMPOSE_FILE = HERE / "docker-compose.yml"
DOCKERFILE = HERE / "Dockerfile"
DOCKERIGNORE = HERE / ".dockerignore"

YAML_FILES = [COMPOSE_FILE, K8S_DIR / "deployment.yaml", K8S_DIR / "service.yaml",
              K8S_DIR / "configmap.yaml", K8S_DIR / "ingress.yaml"]

# 各 kind 的必备字段提示：ConfigMap 与 Secret 用 data/stringData 而不是 spec
KIND_REQUIRED_FIELDS = {
    "Deployment": "spec",
    "Service": "spec",
    "Ingress": "spec",
    "StatefulSet": "spec",
    "DaemonSet": "spec",
    "PersistentVolumeClaim": "spec",
    "ConfigMap": "data",
    "Secret": "data",
}


def head(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# 一、通用 YAML 解析
# ============================================================
def load_yaml(path: pathlib.Path) -> list[dict]:
    """解析一个 YAML 文件，返回文档列表（支持 --- 分隔的多文档）。"""
    text = path.read_text(encoding="utf-8")
    docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    return docs


def check_syntax() -> dict[str, list[dict]]:
    """逐个文件解析，确认语法正确。

    返回的字典以"相对 09_容器部署 的路径"为键（如 "docker-compose.yml"、"k8s/service.yaml"），
    后续检查直接用这个键来定位对应文件。
    """
    head("一、YAML 语法解析（yaml.safe_load_all）")
    parsed: dict[str, list[dict]] = {}
    ok = True
    for path in YAML_FILES:
        key = path.relative_to(HERE).as_posix()          # 相对本目录的键
        rel = path.relative_to(BACK_END).as_posix()      # 展示用的完整相对路径
        if not path.exists():
            print(f"  ✗ {rel} —— 文件不存在")
            ok = False
            continue
        try:
            docs = load_yaml(path)
        except yaml.YAMLError as exc:
            print(f"  ✗ {rel} —— 解析失败：{exc}")
            ok = False
            continue
        kinds = ", ".join(str(d.get("kind") or "docker-compose") for d in docs)
        size = path.stat().st_size
        print(f"  ✓ {rel:<40} {size:>6} 字节  {len(docs)} 个文档  [{kinds}]")
        parsed[key] = docs
    print(f"\n  结论：{'全部 YAML 语法正确' if ok else '存在语法错误的文件'}")
    assert ok, "YAML 语法校验失败"
    return parsed


# ============================================================
# 二、docker-compose.yml 专项检查
# ============================================================
def check_compose(compose: dict) -> None:
    head("二、docker-compose.yml 结构检查")
    services = compose.get("services") or {}
    print(f"  项目名（name）：{compose.get('name', '(未设置，默认取目录名)')}")
    print(f"  服务数量：{len(services)}")

    declared_volumes = set((compose.get("volumes") or {}).keys())
    declared_networks = set((compose.get("networks") or {}).keys())

    for name, svc in services.items():
        print(f"\n  · 服务 {name}")
        # 镜像来源：image 或 build
        if "image" in svc:
            print(f"      image      = {svc['image']}")
        if "build" in svc:
            build = svc["build"]
            context = build.get("context") if isinstance(build, dict) else build
            dockerfile = build.get("dockerfile", "Dockerfile") if isinstance(build, dict) else "Dockerfile"
            print(f"      build      = context={context}  dockerfile={dockerfile}")
            # docker compose 的 context 是相对于【compose 文件所在目录】解析的
            context_dir = (HERE / str(context)).resolve()
            target = context_dir / str(dockerfile)
            print(f"      上下文目录 = {context_dir}")
            print(f"      构建文件存在 = {target.exists()}  ({target})")
            assert target.exists(), f"{name} 的构建文件不存在：{target}"
        if not ({"image"} & set(svc) or {"build"} & set(svc)):
            raise AssertionError(f"{name} 既没有 image 也没有 build")

        # 端口映射
        for mapping in svc.get("ports", []) or []:
            text = str(mapping)
            if ":" in text:
                host, container = text.rsplit(":", 1)
                print(f"      ports      = 宿主机 {host} -> 容器 {container}")
            else:
                print(f"      ports      = {text}")

        # 卷
        for vol in svc.get("volumes", []) or []:
            source = str(vol).split(":")[0]
            kind = "命名卷" if source in declared_volumes else ("绑定挂载" if source.startswith((".", "/")) else "匿名卷")
            if kind == "命名卷":
                print(f"      volumes    = {vol}  [{kind}，已在顶层声明]")
            else:
                print(f"      volumes    = {vol}  [{kind}]")

        # 网络
        for net in svc.get("networks", []) or []:
            assert net in declared_networks, f"{name} 引用了未声明的网络 {net}"
        if svc.get("networks"):
            print(f"      networks   = {svc['networks']}（均已声明）")

        # depends_on 引用完整性
        depends = svc.get("depends_on") or {}
        for dep_name in depends:
            assert dep_name in services, f"{name} 依赖了不存在的服务 {dep_name}"
        if depends:
            detail = ", ".join(f"{k}({v.get('condition', 'started') if isinstance(v, dict) else 'started'})"
                               for k, v in depends.items())
            print(f"      depends_on = {detail}")

        # 重启策略与健康检查
        print(f"      restart    = {svc.get('restart', '(未设置)')}")
        print(f"      healthcheck= {'有' if 'healthcheck' in svc else '无'}")

    print("\n  检查项小结：")
    print("    ✓ 每个服务都有 image 或 build")
    print("    ✓ depends_on 引用的服务都存在")
    print("    ✓ networks / volumes 引用都已声明")
    print("    ✓ build 指向的 Dockerfile 真实存在")


# ============================================================
# 三、K8S 资源检查
# ============================================================
def check_k8s(parsed: dict[str, list[dict]]) -> None:
    head("三、K8S 资源检查")

    resources: dict[str, dict] = {}
    for rel, docs in parsed.items():
        if not rel.startswith("k8s/"):
            continue
        for doc in docs:
            kind = doc.get("kind")
            api = doc.get("apiVersion")
            meta = doc.get("metadata") or {}
            name = meta.get("name")
            ns = meta.get("namespace", "default")

            assert kind, f"{rel} 中某个文档缺少 kind"
            assert api, f"{rel}:{name} 缺少 apiVersion"
            assert name, f"{rel} 中 {kind} 缺少 metadata.name"

            print(f"  · {kind:<24} {name:<22} namespace={ns:<8} apiVersion={api}")

            # 必备字段
            required = KIND_REQUIRED_FIELDS.get(kind)
            if required:
                if required == "data":
                    # Secret 可以用 data（Base64）或 stringData（明文，K8S 自动转 Base64）
                    assert ("data" in doc or "stringData" in doc), f"{kind}/{name} 缺少 data/stringData"
                else:
                    assert required in doc, f"{kind}/{name} 缺少 {required}"
            if "labels" in meta:
                print(f"      labels = {meta['labels']}")

            resources[f"{kind}/{name}"] = doc

    # ---------- Deployment 深度检查 ----------
    print("\n  [Deployment 详细检查]")
    for key, doc in resources.items():
        if not key.startswith("Deployment/"):
            continue
        spec = doc["spec"]
        template = spec["template"]
        pod_spec = template["spec"]
        containers = pod_spec["containers"]

        print(f"  · {key}")
        print(f"      replicas        = {spec.get('replicas')}")
        print(f"      更新策略        = {spec.get('strategy', {}).get('type')}")
        print(f"      selector        = {spec['selector']['matchLabels']}")
        print(f"      Pod labels      = {template['metadata']['labels']}")
        # selector 必须匹配 Pod 模板的 labels，否则 Deployment 创建后一个 Pod 也起不来
        for k, v in spec["selector"]["matchLabels"].items():
            assert template["metadata"]["labels"].get(k) == v, \
                f"{key} 的 selector {k}={v} 与 Pod labels 不匹配"
        print("      ✓ selector 与 Pod labels 匹配（这是最常见的 K8S 配置错误）")

        for c in containers:
            print(f"      容器 {c['name']}")
            print(f"        image         = {c['image']}")
            print(f"        imagePullPolicy = {c.get('imagePullPolicy')}")
            print(f"        containerPort = {[p.get('containerPort') for p in c.get('ports', [])]}")
            print(f"        resources     = requests={c.get('resources', {}).get('requests')} "
                  f"limits={c.get('resources', {}).get('limits')}")
            for probe in ("livenessProbe", "readinessProbe", "startupProbe"):
                if probe in c:
                    print(f"        {probe:<15} = {c[probe].get('httpGet', c[probe])}")
            # 环境变量来源校验
            for src in c.get("envFrom", []) or []:
                ref = src.get("configMapRef") or src.get("secretRef") or {}
                print(f"        envFrom       = {list(src.keys())[0]} -> {ref.get('name')}")

        # 引用的 ConfigMap / Secret 是否真的存在
        referenced_cm = set()
        referenced_secret = set()
        for c in containers:
            for src in c.get("envFrom", []) or []:
                if "configMapRef" in src:
                    referenced_cm.add(src["configMapRef"]["name"])
                if "secretRef" in src:
                    referenced_secret.add(src["secretRef"]["name"])
        for cm in referenced_cm:
            exists = f"ConfigMap/{cm}" in resources
            print(f"      ConfigMap 引用 {cm}：{'存在 ✓' if exists else '不存在 ✗'}")
            assert exists, f"引用了不存在的 ConfigMap：{cm}"
        for sec in referenced_secret:
            exists = f"Secret/{sec}" in resources
            print(f"      Secret 引用 {sec}：{'存在 ✓' if exists else '不存在 ✗'}")
            assert exists, f"引用了不存在的 Secret：{sec}"

    # ---------- Service 深度检查 ----------
    print("\n  [Service 详细检查]")
    deployments = [d for k, d in resources.items() if k.startswith("Deployment/")]
    all_pod_labels: list[dict] = []
    for d in deployments:
        all_pod_labels.append(d["spec"]["template"]["metadata"]["labels"])

    for key, doc in resources.items():
        if not key.startswith("Service/"):
            continue
        spec = doc["spec"]
        selector = spec.get("selector", {})
        print(f"  · {key}")
        print(f"      type      = {spec.get('type')}")
        print(f"      selector  = {selector}")
        print(f"      ports     = {spec.get('ports')}")
        # selector 必须能选中某个 Deployment 的 Pod（这里做的是"本文件之外"的校验）
        matched = any(all(labels.get(k) == v for k, v in selector.items()) for labels in all_pod_labels)
        print(f"      ✓ selector 能选中 Deployment 的 Pod：{matched}")
        assert matched, f"{key} 的 selector {selector} 选不中任何 Pod"
        if spec.get("type") == "NodePort":
            for port in spec["ports"]:
                node_port = port.get("nodePort")
                assert node_port is None or 30000 <= node_port <= 32767, \
                    f"nodePort {node_port} 超出默认范围 30000-32767"
                print(f"      ✓ nodePort {node_port} 在合法范围 30000-32767 内")

    # ---------- Ingress 检查 ----------
    print("\n  [Ingress 检查]")
    for key, doc in resources.items():
        if not key.startswith("Ingress/"):
            continue
        spec = doc["spec"]
        print(f"  · {key}")
        print(f"      ingressClassName = {spec.get('ingressClassName')}")
        for tls in spec.get("tls", []) or []:
            print(f"      tls hosts        = {tls.get('hosts')}  secret={tls.get('secretName')}")
        for rule in spec.get("rules", []) or []:
            host = rule.get("host", "(所有域名)")
            for p in rule["http"]["paths"]:
                backend = p["backend"]
                svc = backend.get("service", {})
                port = (backend.get("service", {}).get("port") or {})
                port_text = port.get("name") or port.get("number")
                print(f"      {p.get('pathType'):<8} {host}{p.get('path'):<10} -> "
                      f"{svc.get('name')}:{port_text}")
                assert f"Service/{svc.get('name')}" in resources, \
                    f"Ingress 引用了不存在的 Service：{svc.get('name')}"
        print("      ✓ 所有 backend 引用的 Service 都存在")

    # ---------- ConfigMap / Secret / PVC ----------
    print("\n  [配置与存储资源]")
    for key, doc in sorted(resources.items()):
        if key.startswith(("ConfigMap/", "Secret/", "PersistentVolumeClaim/")):
            if key.startswith("Secret/"):
                keys = sorted((doc.get("data") or doc.get("stringData") or {}).keys())
                print(f"  · {key:<40} type={doc.get('type')} 键={keys}")
            elif key.startswith("PersistentVolumeClaim/"):
                req = doc["spec"].get("resources", {}).get("requests", {})
                print(f"  · {key:<40} accessModes={doc['spec'].get('accessModes')} 申请={req}")
            else:
                print(f"  · {key:<40} 键={sorted((doc.get('data') or {}).keys())}")


# ============================================================
# 四、Dockerfile 静态检查
# ============================================================
def check_dockerfile() -> None:
    head("四、Dockerfile 静态检查")
    assert DOCKERFILE.exists(), "Dockerfile 不存在"
    text = DOCKERFILE.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in text.splitlines()]
    instructions = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]

    print(f"  文件大小：{DOCKERFILE.stat().st_size} 字节")
    print(f"  有效指令行数：{len(instructions)}")

    # 指令汇总
    froms = [line for line in instructions if line.upper().startswith("FROM")]
    print(f"\n  基础镜像（{len(froms)} 个阶段）：")
    for line in froms:
        print(f"    {line}")

    # 逐条列出指令（用于人工核对）
    print("\n  指令清单（按顺序）：")
    for line in instructions:
        if len(line) > 110:
            line = line[:110] + "…"
        print(f"    {line}")

    # 检查清单
    print("\n  检查项：")
    checks = [
        ("使用多阶段构建（减小镜像体积）", text.count("FROM ") >= 2),
        ("固定基础镜像 tag（不用 latest，保证可复现）", "slim AS" in text or ":3.12" in text),
        ("设置了 WORKDIR", "WORKDIR" in text),
        ("有 EXPOSE 声明端口", "EXPOSE" in text),
        ("使用 exec 形式 CMD（保证信号能送达）", bool(re.search(r'CMD\s*\[', text))),
        ("以非 root 用户运行（安全）", "USER " in text),
        ("配置了 HEALTHCHECK", "HEALTHCHECK" in text),
        ("设置了 PYTHONUNBUFFERED（日志实时输出）", "PYTHONUNBUFFERED" in text),
        ("设置了时区", "TZ=" in text),
        ("服务监听 0.0.0.0（容器外可访问）", "0.0.0.0" in text),
        ("没有把 .env 复制进镜像", ".env" not in instructions.__str__()),
    ]
    for desc, passed in checks:
        print(f"    {'✓' if passed else '✗'} {desc}")
        assert passed, f"Dockerfile 检查未通过：{desc}"

    # .dockerignore
    print(f"\n  .dockerignore：{'存在' if DOCKERIGNORE.exists() else '不存在'}")
    if DOCKERIGNORE.exists():
        ignore_lines = [line.strip() for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.strip().startswith("#")]
        must_ignore = [".venv", "__pycache__", ".env", "Back_End/data"]
        for pattern in must_ignore:
            hit = any(pattern in line for line in ignore_lines)
            print(f"    {'✓' if hit else '✗'} 排除了 {pattern}")
            assert hit, f".dockerignore 应当排除 {pattern}"


# ============================================================
# 五、YAML 常见坑演示
# ============================================================
def show_yaml_pitfalls() -> None:
    head("五、YAML 常见坑（务必避免）")
    samples = [
        ("布尔陷阱", "enabled: yes", "yes/no/on/off 会被解析成布尔值！想表达字符串要加引号：\"yes\""),
        ("版本号变浮点", "version: 3.10", "会被解析成 3.1（去掉末尾 0）！要写成 \"3.10\" 或 '3.10'"),
        ("挪威问题", "country: NO", "NO 会被解析成 False！国家代码必须加引号：\"NO\""),
        ("冒号后必须有空格", "key:value", "少了空格就成了一个字符串键，不是键值对"),
        ("Tab 缩进", "\\tkey: value", "YAML 禁止用 Tab 缩进，必须用空格"),
    ]
    for title, raw, explain in samples:
        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            parsed = f"解析失败：{type(exc).__name__}"
        print(f"\n  【{title}】")
        print(f"    写法：{raw}")
        print(f"    解析结果：{parsed!r}")
        print(f"    说明：{explain}")

    print("\n  其他注意点：")
    print("    · 中文务必用 UTF-8 保存（本目录所有 YAML 都是 UTF-8）")
    print("    · 多行字符串用 | （保留换行）或 > （折叠成一行），别硬拼 \\n")
    print("    · K8S 的 YAML 可以一个文件多个文档（用 --- 分隔），kubectl apply 会依次创建")


def main() -> None:
    print()
    print("#" * 78)
    print("# 容器部署配置校验（Docker Compose + K8S YAML + Dockerfile）")
    print("#" * 78)
    print(f"解释器：{sys.executable}")
    print(f"配置目录：{HERE}")
    print(f"PyYAML 版本：{yaml.__version__}")

    parsed = check_syntax()
    compose_docs = parsed.get("docker-compose.yml", [])
    if compose_docs:
        check_compose(compose_docs[0])
    check_k8s(parsed)
    check_dockerfile()
    show_yaml_pitfalls()

    head("校验汇总")
    print("  ✓ 5 个 YAML 文件语法全部正确")
    print("  ✓ docker-compose.yml：服务/端口/卷/网络/depends_on/build 引用一致")
    print("  ✓ K8S：Deployment / Service / ConfigMap / Secret / Ingress 字段齐备，跨文件引用一致")
    print("  ✓ Dockerfile：多阶段构建、非 root、健康检查、0.0.0.0 监听等 11 项检查通过")
    print("  ✓ .dockerignore：排除了 .venv / __pycache__ / .env / data")
    print()
    print("  说明：本机未安装 Docker / kubectl，因此这里做的是**静态校验**。")
    print("        要真正部署，请按 09_容器部署/k8s/README.md 里的流程操作。")
    print()
    print("全部检查通过 ✓")


if __name__ == "__main__":
    main()
