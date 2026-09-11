# Docker 常用命令速查（后端开发基础 · 容器部署前置）

> 对应课案章节：`Linux → 最小化镜像`、`API 开发部署 → 安装 Docker`、`容器部署 → Docker`
> 本文讲解 Docker 的三个核心概念与全生命周期命令，并给出后端部署最常用的实战组合。

---

## 一、为什么后端要学 Docker

**核心问题：** 代码在你电脑上跑得好好的，到了服务器就报"缺依赖 / 版本不对 / 系统库不一致"。

**Docker 的解法：** 把 **应用代码 + 运行时 + 系统依赖 + 配置** 一起打包成一个**镜像（Image）**，镜像在任何装了 Docker 的机器上都能以**完全一致**的方式运行。

| 概念 | 类比 | 说明 |
| --- | --- | --- |
| 镜像 Image | 类 / 安装光盘 | 只读模板，分层存储 |
| 容器 Container | 实例 / 运行中的程序 | 镜像的一次运行，可读写、可随时销毁 |
| 仓库 Registry | 应用商店 | 存放镜像的地方，如 Docker Hub、阿里云 ACR |

再记住两句关键：
- **容器不是虚拟机**：它和宿主机共享内核，靠 namespace（隔离视图）+ cgroup（限制资源）实现"看起来像独立机器"，所以启动只要毫秒级。
- **容器是无状态的**：容器删了数据就没了。需要持久化的东西（数据库文件、日志、上传文件）必须挂载 **Volume** 或 **bind mount**。

---

## 二、安装 Docker（Linux 服务器）

```bash
# 方式一：官方一键脚本（教学/快速部署）
bash <(wget -qO- https://xuanyuan.cloud/docker.sh)   # 第三方脚本，生产环境请先查看脚本内容

# 方式二：官方仓库（推荐生产使用，以 Ubuntu 为例）
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker

# 安装独立版 docker-compose 二进制（老版本命令 docker-compose）
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

**安装后自检：**

```bash
sudo systemctl status docker     # 服务是否 active (running)
sudo docker run hello-world      # 拉镜像 + 跑容器，输出 Hello from Docker! 即成功
sudo docker compose version      # Compose V2 插件版（命令是 docker compose，中间有空格）
```

> 小技巧：把当前用户加入 docker 组，就不用每条命令都敲 `sudo`
> `sudo usermod -aG docker $USER && newgrp docker`

> 国内加速：修改 `/etc/docker/daemon.json` 配置 `registry-mirrors`，然后 `sudo systemctl restart docker`。

---

## 三、镜像相关命令

```bash
docker pull nginx:1.25            # 拉取镜像（不写 tag 默认 latest，生产必须写死 tag）
docker images                     # 列出本地镜像（等价 docker image ls）
docker inspect nginx:1.25         # 查看镜像详细信息（JSON）
docker history nginx:1.25         # 查看镜像分层历史（排查体积来源）
docker rmi nginx:1.25             # 删除镜像（等价 docker image rm）
docker rmi $(docker images -q -f dangling=true)   # 清理 <none> 悬空镜像
docker image prune -a             # 一键清理未被使用的镜像（-a 更彻底）
docker save -o nginx.tar nginx:1.25   # 导出镜像为 tar（离线搬运）
docker load -i nginx.tar              # 从 tar 导入镜像
docker tag myapi:v1.0 registry.cn-hangzhou.aliyuncs.com/ns/myapi:v1.0  # 打标签
docker push registry.cn-hangzhou.aliyuncs.com/ns/myapi:v1.0            # 推送到仓库
```

**镜像命名规则：** `[仓库地址/命名空间/]名称[:标签]`
例：`registry.cn-hangzhou.aliyuncs.com/study/myapi:v1.0`
不写仓库地址时默认走 Docker Hub（`docker.io/library/`）。

---

## 四、容器相关命令（全生命周期）

```bash
# ---------- 运行 ----------
docker run nginx:1.25                      # 前台运行
docker run -d nginx:1.25                   # -d 后台运行
docker run -it --rm busybox                # -i 交互 + -t 终端 + --rm 退出即删
docker run -d --name myapi -p 8000:80 myapi:v1.0
#            ↑名字        ↑宿主端口:容器端口
docker run -d -v /opt/data:/app/data myapi:v1.0      # 挂载目录（持久化）
docker run -d -v myvol:/app/data myapi:v1.0          # 挂载命名卷
docker run -d -e APP_ENV=prod -e MYSQL_HOST=db myapi:v1.0   # 注入环境变量
docker run -d --env-file .env myapi:v1.0             # 从文件读环境变量
docker run -d --network mynet --name api myapi:v1.0  # 加入自定义网络
docker run -d --restart=unless-stopped myapi:v1.0    # 崩溃/重启后自动拉起
docker run -d --memory=512m --cpus=1.5 myapi:v1.0    # 限制资源
docker run --rm myapi:v1.0 python -c "print('一次性任务')"   # 覆盖默认命令

# ---------- 查看 ----------
docker ps                    # 只看运行中的容器
docker ps -a                 # 包括已退出的容器（排障必看！）
docker logs -f --tail 100 myapi    # 实时看最后 100 行日志
docker stats                 # 实时资源占用（CPU/内存/网络）
docker inspect myapi         # 容器详细信息（IP、挂载、环境变量）
docker top myapi             # 容器内进程
docker port myapi            # 端口映射关系

# ---------- 进入容器 ----------
docker exec -it myapi /bin/bash    # 开一个新终端（推荐，退出不影响主进程）
docker exec -it myapi sh           # alpine/slim 镜像可能没有 bash
docker attach myapi                # 附着到主进程（Ctrl+C 会把容器也停掉，少用）

# ---------- 启停/删除 ----------
docker stop myapi        # 优雅停止（先发 SIGTERM，10 秒后才 SIGKILL）
docker start myapi       # 启动已存在的容器
docker restart myapi     # 重启
docker kill myapi        # 立即强杀
docker rm myapi          # 删除已停止的容器
docker rm -f myapi       # 强制删除运行中的容器
docker container prune   # 清理所有已停止容器

# ---------- 拷贝文件 ----------
docker cp myapi:/app/logs/app.log ./app.log    # 容器 → 宿主机
docker cp ./config.yaml myapi:/app/config.yaml # 宿主机 → 容器

# ---------- 构建 ----------
docker build -t myapi:v1.0 .            # 用当前目录的 Dockerfile 构建
docker build -f docker/Dockerfile -t myapi:v1.0 .   # 指定 Dockerfile 路径
docker build --no-cache -t myapi:v1.0 . # 不用缓存（依赖装错时救命）
```

### `docker run` 高频参数速记

| 参数 | 含义 | 备注 |
| --- | --- | --- |
| `-d` | 后台运行 | detached |
| `-it` | 交互式终端 | 进容器调试必备 |
| `--rm` | 退出后自动删除容器 | 一次性任务 |
| `--name` | 容器名 | 不写会随机生成 |
| `-p 宿主:容器` | 端口映射 | 少了它外网访问不了 |
| `-v 宿主:容器` | 目录挂载 | 持久化/改配置免重建 |
| `-e KEY=VAL` | 环境变量 | 覆盖配置 |
| `--network` | 加入网络 | 容器间用容器名互相访问 |
| `--restart` | 重启策略 | `no`/`on-failure`/`always`/`unless-stopped` |
| `--memory` `--cpus` | 资源限制 | 防止单个容器拖垮整机 |

---

## 五、网络与数据卷

```bash
# ---------- 网络 ----------
docker network ls                        # 列出网络（默认有 bridge/host/none）
docker network create mynet              # 创建自定义 bridge 网络
docker network inspect mynet             # 查看网段与已连接容器
docker network connect mynet myapi       # 把运行中的容器接入网络
docker network disconnect mynet myapi

# ---------- 数据卷 ----------
docker volume ls                         # 列出数据卷
docker volume create myvol
docker volume inspect myvol              # 看它在宿主机的真实路径
docker volume rm myvol
docker volume prune                      # 清理未使用的卷（会删数据！）
```

**三种网络模式：**
- `bridge`（默认）：容器有独立 IP，通过 `-p` 映射端口对外。
- `host`：容器直接使用宿主机网络，性能最好但端口会冲突。
- `none`：无网络，用于纯计算任务。

**为什么要自定义网络：** 同一个自定义网络里的容器，可以**直接用容器名当域名**互相访问，例如 FastAPI 容器里连数据库写 `mysql:3306` 即可，不用管 IP。

---

## 六、Docker Compose（多容器编排）

单容器用 `docker run`，一整套服务（Web + MySQL + Redis）用 Compose 描述成 YAML，一条命令起停。

```bash
docker compose up -d            # 后台启动全部服务（自动建网络）
docker compose up -d --build    # 先重新构建镜像再启动
docker compose ps               # 查看服务状态
docker compose logs -f api      # 跟踪某个服务日志
docker compose exec api bash    # 进入某个服务容器
docker compose restart api      # 重启服务
docker compose down             # 停止并删除容器/网络
docker compose down -v          # 连数据卷一起删（清库）
docker compose config           # 校验并打印最终配置（排查变量替换问题）
```

> Compose V1 命令是 `docker-compose`（带横杠），V2 是 `docker compose`（带空格子命令），不要混用。

---

## 七、清理与排障

```bash
# ---------- 磁盘清理 ----------
docker system df                 # 查看镜像/容器/卷占用
docker system prune              # 清理停止的容器 + 悬空镜像 + 无用网络
docker system prune -a --volumes # 彻底清理（会删数据卷，危险）

# ---------- 排障三板斧 ----------
docker ps -a                     # 1. 容器是不是 Exited 了？
docker logs --tail 200 myapi     # 2. 日志里报什么错？
docker exec -it myapi sh         # 3. 进容器里手动试（能 curl 通吗？文件在吗？）

# ---------- 常见问题 ----------
# Q: 容器启动就退出 (Exited 0)
# A: 容器必须有一个前台进程。检查 CMD 是否是后台命令，或手动加 `sleep infinity` 测试。
#
# Q: 端口被占用  bind: address already in use
# A: 宿主机端口冲突。ss -tulnp | grep 8000 找到占用者，或换成 -p 8001:80。
#
# Q: 容器里连不上 localhost 的数据库
# A: 容器内的 localhost 指容器自己！要用宿主机 IP、host.docker.internal，
#    或者用 Compose 服务名（推荐）。
#
# Q: 改了代码但容器里没变
# A: 镜像里是构建时的快照。要么重新 build，要么开发时挂载源码目录 -v $(pwd):/app。
#
# Q: Alpine 镜像里 pip install 报缺少 gcc
# A: musl 环境没有预编译 wheel，需 apk add build-base，或换 python:3.12-slim。
```

---

## 八、后端部署最短路径（本课案使用的流程）

```bash
# 1. 本地构建镜像
docker build -t myapi:v1.0 .

# 2. 本地跑起来验证
docker run --rm -p 8000:80 myapi:v1.0
curl http://127.0.0.1:8000/docs

# 3. 推送到镜像仓库（云服务器能拉到）
docker tag myapi:v1.0 registry.cn-hangzhou.aliyuncs.com/ns/myapi:v1.0
docker push registry.cn-hangzhou.aliyuncs.com/ns/myapi:v1.0

# 4. 服务器上拉取并启动（配合安全组 + 防火墙放行端口）
docker pull registry.cn-hangzhou.aliyuncs.com/ns/myapi:v1.0
docker run -d --name myapi --restart=unless-stopped -p 8099:80 myapi:v1.0

# 5. 验证
curl http://127.0.0.1:8099/docs        # 服务器本机
curl http://公网IP:8099/docs           # 自己的电脑
```
