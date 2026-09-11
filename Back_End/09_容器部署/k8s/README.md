# ============================================================
# k8s/README.md —— K8S 核心概念与部署实操（中文讲解）
# ============================================================
# 对应课案章节：后端开发基础 → 容器部署 → K8S
# （核心概念 / 工作负载类型 / Namespace / Minikube / kuboard / 应用部署）
# 本目录的 4 个 YAML 都可用 `kubectl apply -f` 直接部署，并已通过 检查YAML.py 语法校验。
# ============================================================

## 一、K8S 是什么，为什么需要它

**Kubernetes（K8s）是一个容器编排平台**，用于自动化部署、扩缩容和管理容器化应用。

对比一下就清楚了：

| | Docker | Kubernetes |
| --- | --- | --- |
| 管什么 | 一台机器上的单个/多个容器 | 一整个集群（几十上百台机器） |
| 挂了怎么办 | 手动 `docker start` | 自动重建 Pod（自愈） |
| 流量变大 | 手动改配置再起容器 | 一条命令/自动扩到 N 个副本 |
| 发布新版本 | 停旧容器起新容器（有停机） | 滚动更新，零停机 |
| 服务地址 | 容器 IP（重建就变） | Service 提供固定入口 + 负载均衡 |

**为什么要用 K8s？** 想象你要部署一个购物系统：架构复杂、规模庞大，
需要根据访问量自动分配服务器资源，某个容器宕机后要自动灾难恢复、故障转移。
这时 K8s 就能大显身手。

---

## 二、核心概念（对应课案表格）

| 概念 | 说明 | 在本目录 YAML 中的体现 |
| --- | --- | --- |
| **Cluster（集群）** | K8s 管理的全部主机节点连同控制平面的集合 | minikube 起的那个单节点集群 |
| **Node（节点）** | 集群中的单台服务器/主机，运行着若干 Pod | minikube 的 docker 容器就是"一个节点" |
| **Control Plane（控制平面）** | 集群的中心管理计算机：API Server + Scheduler + Controller Manager + etcd。实时监测节点与 Pod 状态，Pod 挂了立刻启用备用 Pod 替换 | `kube-system` 命名空间里的组件 |
| **Pod** | K8s 中**最小调度单元**，是一个或多个容器的集合，共享网络与存储 | `deployment.yaml` 的 `spec.template` |
| **ReplicaSet（副本集）** | 保证"始终有 N 个 Pod 在跑"的控制器，Pod 挂了自动补一个 | Deployment 会自动创建并管理 RS |
| **Service** | 为 Pod 提供**稳定的访问入口**并负载均衡。类型有 ClusterIP / NodePort / LoadBalancer | `service.yaml` |

**工作流程：**

1. 整个集群由 Control Plane 集中管理，它通过 API 与各 Node 通信；
2. Control Plane 实时监测各节点网络状态与 Pod 健康状况；
3. 某个 Pod 宕机 → ReplicaSet 立即启动备用 Pod 替换（故障恢复）；
4. 访问量增加 → K8s 自动创建更多 Pod 分担负载（负载均衡/扩缩容）；
5. Service 负责把外部请求路由到正确的 Pod，提供稳定访问入口。

---

## 三、工作负载类型

| 类型 | 说明 | 适用场景 | 本目录是否使用 |
| --- | --- | --- | --- |
| **Deployment** | 无状态应用，支持滚动更新、回滚、扩缩容，Pod 之间相互独立 | Web 服务、API 服务、后端应用 | ✅ `deployment.yaml` |
| **StatefulSet** | 有状态应用，保证 Pod 的顺序与唯一性，每个 Pod 有稳定网络标识与持久存储 | 数据库（MySQL/PostgreSQL）、消息队列（Kafka） | 生产上 MySQL 应使用它 |
| **DaemonSet** | 确保**每个 Node 上都有一个** Pod 副本，新节点加入时自动部署 | 日志收集（Fluentd）、监控代理（Node Exporter）、网络插件 | 集群组件常用 |

**StatefulSet 示例（生产部署 MySQL 时用，本目录未包含以保持精简）：**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql-headless        # 必须配合 Headless Service 提供稳定 DNS
  replicas: 3
  selector:
    matchLabels: { app: mysql }
  template:
    metadata:
      labels: { app: mysql }
    spec:
      containers:
        - name: mysql
          image: mysql:8.0
          volumeMounts:
            - name: data
              mountPath: /var/lib/mysql
  volumeClaimTemplates:               # 每个 Pod 一份独立存储（这是 StatefulSet 的关键能力）
    - metadata: { name: data }
      spec:
        accessModes: ["ReadWriteOnce"]
        resources: { requests: { storage: 10Gi } }
```

---

## 四、Namespace（命名空间）

命名空间用于**隔离和管理**集群资源，特别适合多用户/多项目环境：

| 特性 | 说明 |
| --- | --- |
| 资源隔离 | 不同命名空间里的资源名可以重复，互不影响 |
| 资源配额 | 可对每个命名空间设置 CPU/内存配额（ResourceQuota）与上限（LimitRange） |
| 便捷管理 | 便于对大量资源分类、筛选与授权（RBAC 按 Namespace 授权） |

常见内置命名空间：

| 命名空间 | 说明 |
| --- | --- |
| `default` | 默认命名空间，未指定时资源就放这里 |
| `kube-system` | K8s 系统组件（DNS、调度器、控制平面组件等） |
| `kube-public` | 公共资源，全局可读 |
| `kube-node-lease` | 节点心跳检测，用于判断节点是否存活 |

本目录所有 YAML 都放在 `demo` 命名空间：

```bash
kubectl create namespace demo
```

---

## 五、部署方式对比

| 方式 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 亲自搭建 | 完全可控 | 步骤繁琐，需长期维护 | 生产环境（有运维团队） |
| 云服务（ACK/TKE/EKS） | 托管省心，控制面免维护 | 需付费 | 生产环境 |
| **Minikube** | 免费、本地完整集群、支持几乎所有功能 | 单节点，非生产级 | **学习测试（本课案）** |

---

## 六、Minikube 实操

### 1. 安装

1. 下载安装：https://minikube.sigs.k8s.io/docs/start/ ，点 latest release 下载 `minikube-installer.exe`，
   安装后把 `C:\Program Files\Kubernetes\Minikube` 加入环境变量 PATH；
2. 安装 `kubectl`（K8s 命令行客户端）；
3. 需要先装好 Docker（Minikube 用 Docker 驱动跑集群）。

### 2. 启动集群

```powershell
minikube start --driver=docker --image-mirror-country=cn --cpus=4 --memory=8192 --disk-size=40g --apiserver-ips=<你的机器IP>
# --apiserver-ips 通过 ipconfig 获得的宿主机 IP，便于 kuboard 等工具从外部访问
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `--driver=docker` | 用 Docker 作为虚拟化驱动（Windows 上最省事） |
| `--image-mirror-country=cn` | 使用国内镜像源拉取 K8s 组件镜像 |
| `--cpus` / `--memory` / `--disk-size` | 分配给集群的资源（低于 2 核 2G 会起不来） |
| `--apiserver-ips` | 额外暴露的 API Server 地址，供 kuboard 等外部工具连接 |

### 3. 验证

```bash
kubectl get nodes
minikube status
```

### 4. 常用命令

```bash
minikube stop                 # 停止集群
minikube delete               # 删除集群（释放资源）
minikube dashboard            # 打开 K8S 官方 Web 管理界面
minikube service myapi        # 在浏览器中打开某个 NodePort/LoadBalancer 服务
minikube ssh                  # 进入 Minikube 虚拟机
minikube image ls             # 查看 Minikube 内部的镜像
minikube image load myapi:v1.0 --alsologtostderr=true   # 把本地镜像加载进集群

kubectl get pods              # 查看默认命名空间的 Pod
kubectl get pods -n kube-system   # 查看指定命名空间
kubectl get po -A             # 查看所有命名空间的 Pod
kubectl get services          # 查看所有服务
kubectl get nodes             # 查看节点信息
kubectl get all -n demo       # 查看某命名空间下所有资源
kubectl describe pod <pod名> -n demo    # 排障神器：看事件（Events）
kubectl logs -f <pod名> -n demo         # 跟踪日志
kubectl exec -it <pod名> -n demo -- sh  # 进入容器
```

---

## 七、用本目录的 YAML 部署（完整流程）

```bash
# 1. 构建镜像（在仓库根目录执行，因为 Dockerfile 的构建上下文是根目录）
docker build -f Back_End/09_容器部署/Dockerfile -t backend-demo:v1.0 .

# 2. 把本地镜像加载进 Minikube（否则会去 Docker Hub 找，然后 ImagePullBackOff）
minikube image load backend-demo:v1.0 --alsologtostderr=true

# 3. 创建命名空间并部署
kubectl create namespace demo
kubectl apply -f Back_End/09_容器部署/k8s/configmap.yaml
kubectl apply -f Back_End/09_容器部署/k8s/deployment.yaml
kubectl apply -f Back_End/09_容器部署/k8s/service.yaml
kubectl apply -f Back_End/09_容器部署/k8s/ingress.yaml

# 4. 观察滚动更新过程
kubectl get pods -n demo -w

# 5. 访问服务
minikube service book-api-nodeport -n demo     # 自动打开浏览器
kubectl port-forward -n demo svc/book-api-svc 8000:8000   # 或者用端口转发本地访问

# 6. 扩缩容
kubectl scale deployment book-api --replicas=4 -n demo
kubectl get pods -n demo

# 7. 回滚
kubectl rollout history deployment/book-api -n demo
kubectl rollout undo deployment/book-api -n demo

# 8. 清理
kubectl delete -f Back_End/09_容器部署/k8s/ingress.yaml
kubectl delete namespace demo
```

**课案里的命令面对应关系：**

```bash
# 课案（命令式创建）
kubectl create deployment myapi --image=myapi:v1.0 --port=80
kubectl expose deployment myapi --port=8888 --target-port=80 --type=LoadBalancer

# 等价的声明式写法（本目录 YAML，推荐 —— 可以进 Git、可评审、可回滚）
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

底层流程：`Deployment → 创建 ReplicaSet → ReplicaSet 启动 Pod → Pod 运行 myapi 容器`。

**端口概念别搞混：**

| 端口 | 含义 |
| --- | --- |
| `--target-port=80` | **容器里程序真正监听的端口**（代码里 `--port 80` / `EXPOSE 80`） |
| `--port=8888` | **Service 在集群内部暴露的端口**（集群内访问 `svc:8888`） |
| NodePort（如 31234） | **节点上随机/指定端口**，集群外通过 `<节点IP>:31234` 访问 |
| `minikube service myapi` | 自动把 NodePort 映射到本机并打开浏览器 |

---

## 八、kuboard：可视化集群管理

**kuboard 是 K8s 第三方更友好的可视化管理与监控工具**，提供 Web 界面查看集群状态、
管理资源、查看日志等。

| 功能 | 说明 |
| --- | --- |
| 资源管理 | 可视化管理 Deployment、Service、Pod、ConfigMap、Secret 等 |
| 日志查看 | 实时查看容器日志，方便调试 |
| 终端 | 在浏览器中直接进入容器终端 |
| 监控 | 查看资源使用情况（CPU、内存） |
| 编辑 | 直接在 Web 界面编辑 YAML |
| 多集群 | 支持管理多个 K8s 集群 |

**与 Minikube Dashboard 对比：**

| 特性 | Minikube Dashboard | kuboard |
| --- | --- | --- |
| 安装 | 内置，一键启动 | 需单独安装 |
| 功能 | 基础资源管理 | 更丰富（终端、日志编辑、监控） |
| 多集群 | 不支持 | 支持 |
| 适用场景 | 快速查看 | 生产级管理 |

**安装与连接 Minikube：**

```bash
# 1. 启动 kuboard（把 <你的机器IP> 换成你机器的实际 IP，用 ipconfig / ifconfig 查看）
docker run -d --restart=unless-stopped --name=kuboard \
  -p 8080:80/tcp -p 10081:10081/tcp \
  -e KUBOARD_ENDPOINT="http://<你的机器IP>:8080" \
  -e KUBOARD_AGENT_SERVER_TCP_PORT="10081" \
  -v /root/kuboard-data:/data \
  eipwork/kuboard:v3

# 2. 登录：http://<你的机器IP>:8080/kuboard/cluster
#    注意：不能用 localhost/127.0.0.1 访问
#    管理员用户名 admin，默认密码 Kuboard123（登录后立刻改掉！）

# 3. 在界面里获取 Agent 安装配置（复制页面给出的指令）
curl -k 'http://localhost:8080/kuboard-api/cluster/minikube/kind/KubernetesCluster/minikube/resource/installAgentToKubernetes?token=<页面上的token>' > kuboard-agent.yaml

# 4. 把 Agent 部署到集群里，Kuboard 就能管理这个集群了
kubectl apply -f ./kuboard-agent.yaml
```

---

## 九、排障速查

| 现象 | 原因与处理 |
| --- | --- |
| Pod 状态 `ImagePullBackOff` | 镜像拉不到。本地镜像要先 `minikube image load`；私有仓库要配 `imagePullSecrets` |
| Pod 状态 `CrashLoopBackOff` | 容器启动就崩。`kubectl logs <pod> --previous` 看上一次的日志 |
| Pod 状态 `Pending` | 没有节点能容纳它（资源不够 / PVC 没绑定）。`kubectl describe pod` 看 Events |
| Pod 状态 `OOMKilled` | 内存超过 limits。调大 `resources.limits.memory` 或优化内存使用 |
| Service 访问不通 | 检查 `selector` 是否与 Pod 的 labels 完全一致；`kubectl get endpoints -n demo` |
| 就绪探针一直失败 | `/health` 路径写错，或 `initialDelaySeconds` 太短、应用还没起来 |
| Ingress 404 | `ingressClassName` 不对，或 Ingress Controller 没安装（`minikube addons enable ingress`） |
| 更新后没生效 | `kubectl rollout status deployment/xxx`；镜像 tag 用了 `latest` 且没改，导致没触发滚动 |
