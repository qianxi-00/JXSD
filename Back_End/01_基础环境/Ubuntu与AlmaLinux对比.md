# Ubuntu 与 AlmaLinux 对比（后端开发基础 · 发行版章节）

> 对应课案章节：`Linux → Ubuntu`、`Linux → almalinux`、`Linux → Ubuntu vs almalinux`

---

## 一、先理解"它们是什么"

### Ubuntu

**Ubuntu 是基于 Debian 的 Linux 发行版，由 Canonical 公司维护**，每 6 个月发一个版本（如 24.04），每 2 年发一个 LTS 长期支持版（5 年安全更新）。

特点：
- 软件包最新、生态最好，遇到问题搜索命中率最高；
- 文档/教程最多，AI 工具生成的部署脚本默认基本都按 Ubuntu 写；
- 云厂商镜像齐全，Docker 官方镜像 `ubuntu:24.04` 也很常用；
- 默认带 `ufw` 防火墙、`apt` 包管理器。

### AlmaLinux

**AlmaLinux OS 是一个开源、社区驱动的 Linux 系统，用于填补 CentOS Linux 稳定版停服留下的空白。**

背景故事：CentOS Linux 曾经是"免费的 RHEL 复刻版"，2021 年 Red Hat 宣布 CentOS Linux 停止更新（转向滚动发布的 CentOS Stream），于是社区发起了 AlmaLinux / Rocky Linux 来接班。

特点：
- 与 RHEL **二进制兼容**，企业里跑的商业软件（如 Oracle、部分中间件）只认 RHEL 系；
- 版本更"保守"，一个主版本支持 10 年，稳定性优先；
- 包管理器是 `yum`/`dnf`，包格式 `.rpm`；
- 默认防火墙是 `firewalld`（基于 nftables/iptables）；
- 国内阿里云、腾讯云的"Alibaba Cloud Linux / TencentOS"命令与其高度一致。

---

## 二、在 Docker 里安装并体验两个系统

```shell
# ============ Ubuntu ============
docker pull ubuntu:24.04
docker run -d --name myubuntu ubuntu:24.04 sleep infinity
docker exec -it myubuntu /bin/bash

# 容器内：
apt update                 # 更新包索引（必须先做，否则找不到包）
apt install -y curl vim    # 安装软件，-y 自动确认

# ============ AlmaLinux ============
docker pull almalinux:8
docker run -d --name myalma almalinux:8 sleep infinity
docker exec -it myalma /bin/bash

# 容器内：
yum update -y
yum install -y curl vim
```

> 容器里没有 systemd，所以 `systemctl` 不可用属正常现象；要练习服务管理需要真实虚拟机或云服务器。

---

## 三、Ubuntu 包管理（APT）详解

`apt` = Advanced Package Tool，软件源配置在 `/etc/apt/sources.list`（及 `sources.list.d/`）。

```bash
# ---------- 索引与升级 ----------
sudo apt update              # 1. 从软件源拉取"有哪些包、什么版本"的索引（不安装东西）
sudo apt upgrade             # 2. 升级所有已安装的包到最新版
sudo apt full-upgrade        # 允许为了解决依赖而卸载/安装包

# ---------- 查询 ----------
apt list --installed             # 列出所有已安装软件
apt list --installed | less      # 分页查看（Space 翻页、q 退出、/ 搜索）
apt list --upgradable            # 有哪些可升级
apt search keyword               # 按关键字搜索包
apt show package_name            # 显示包详情（版本、依赖、体积、来源）
dpkg -L package_name             # 这个包到底装了哪些文件（找配置文件路径神器）
dpkg -S /usr/bin/curl            # 反过来：这个文件属于哪个包

# ---------- 安装 ----------
sudo apt install package_name
sudo apt install -y nginx curl   # -y 不交互确认（脚本里必加）

# ---------- 卸载 ----------
sudo apt remove  package_name    # 卸载，保留配置文件
sudo apt purge   package_name    # 连配置文件一起删
sudo apt autoremove              # 清理"没人依赖的"自动安装包

# ---------- 清理缓存 ----------
sudo apt autoclean               # 删除已失效的 .deb 缓存
sudo apt clean                   # 删除全部下载缓存
```

**软链接小知识：** `/etc/nginx/sites-enabled/` 里放的是 `sites-available/` 的软链接，这就是"启用/禁用站点"的实现方式。

---

## 四、AlmaLinux 包管理（YUM / DNF）详解

`dnf` 是 `yum` 的下一代实现，两者命令几乎完全兼容，`yum` 通常只是 `dnf` 的软链。

```bash
# ---------- 更新 ----------
yum update                    # 更新所有包
yum check-update              # 只检查有哪些可更新（对应 apt update + 列表）
yum makecache                 # 生成本地元数据缓存

# ---------- 查询 ----------
yum list installed            # 列出已安装包
yum search keyword            # 搜索包
yum info package_name         # 显示包信息
yum provides /usr/bin/curl    # 这个文件/命令来自哪个包（对应 dpkg -S）
rpm -ql package_name          # 包安装了哪些文件（对应 dpkg -L）

# ---------- 安装 / 卸载 ----------
yum install -y package_name
yum remove  package_name
yum clean all                 # 清理缓存
```

**额外工具：** AlmaLinux 8/9 上安装 `epel-release` 后可用 `dnf --enablerepo=epel install xxx` 获取更多社区软件包。

---

## 五、两者对比总表

### 1. 基础差异

| 特性 | Ubuntu | AlmaLinux |
| --- | --- | --- |
| 发行版家族 | Debian | Red Hat |
| 上游 | Debian | RHEL（二进制兼容） |
| 维护方 | Canonical（商业公司） | AlmaLinux OS Foundation（社区） |
| 包管理器 | APT | YUM / DNF |
| 包格式 | `.deb` | `.rpm` |
| 软件源配置 | `/etc/apt/sources.list` | `/etc/yum.repos.d/*.repo` |
| 默认防火墙 | `ufw` | `firewalld` |
| 默认服务管理 | systemd | systemd |
| 发布节奏 | 6 个月一版，LTS 2 年一版 | 跟随 RHEL 大版本，支持约 10 年 |
| 软件新旧 | 较新 | 较保守、稳定 |
| 适用场景 | Web 服务、Docker、AI/新框架部署 | 企业内网、传统中间件、长周期稳定服务 |

### 2. 命令对比（做同一件事）

| 操作 | Ubuntu | AlmaLinux / CentOS |
| --- | --- | --- |
| 更新包索引 | `apt update` | `yum check-update` |
| 升级全部包 | `apt upgrade` | `yum update` |
| 安装包 | `apt install pkg` | `yum install pkg` |
| 卸载包 | `apt remove pkg` | `yum remove pkg` |
| 搜索包 | `apt search pkg` | `yum search pkg` |
| 查看包信息 | `apt show pkg` | `yum info pkg` |
| 列出已安装 | `apt list --installed` | `yum list installed` |
| 清理缓存 | `apt autoremove` / `apt clean` | `yum clean all` |
| 查文件属于哪个包 | `dpkg -S <file>` | `rpm -qf <file>` |
| 查包安装了哪些文件 | `dpkg -L pkg` | `rpm -ql pkg` |
| 防火墙放行端口 | `ufw allow 8099/tcp` | `firewall-cmd --permanent --add-port=8099/tcp` |
| 防火墙生效 | `ufw reload` | `firewall-cmd --reload` |
| 查看监听端口 | `ss -tulnp` | `ss -tulnp` |

### 3. 防火墙命令对比（部署时最容易踩坑）

```bash
# ---------- Ubuntu: ufw ----------
sudo ufw status                  # 查看状态
sudo ufw allow 22/tcp            # 放行 SSH（做之前先放行，否则会把自己关在门外）
sudo ufw allow 8099/tcp
sudo ufw enable                  # 启用（会警告可能断开 ssh）
sudo ufw delete allow 8099/tcp   # 撤销规则
sudo ufw reload                  # 重载规则

# ---------- AlmaLinux: firewalld ----------
sudo firewall-cmd --state                                   # 是否运行
sudo firewall-cmd --permanent --add-port=8099/tcp            # 永久添加（必须 --permanent 才能持久）
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload                                   # 重载使永久规则生效
sudo firewall-cmd --list-ports                               # 查看已放行端口
sudo firewall-cmd --list-all                                 # 查看完整区域配置
```

> 记住：**`firewall-cmd` 不加 `--permanent` 的重启就丢**；加了 `--permanent` 不 `--reload` 就不生效。这两步都要做。

---

## 六、选型建议（后端工程师视角）

| 你的目标 | 推荐 |
| --- | --- |
| 学习、跑通 Docker/FastAPI/新框架 | **Ubuntu LTS**（教程最多，`apt` 装东西最省心） |
| 公司要求与 RHEL 一致、跑商业软件 | **AlmaLinux** |
| 阿里云/腾讯云买机器图省事 | 选 Ubuntu 镜像；要 RHEL 系就选 Alibaba Cloud Linux |
| 容器基础镜像 | `python:3.12-slim`（Debian 系）优先，体积与兼容性平衡最好 |

**跨发行版通用技能**（换系统也不会白学）：
`ls/cd/cp/mv/rm/find/grep/tail`、权限与用户、`ss`/`curl`/`ssh`、`systemctl`、Docker 命令、环境变量与 PATH、日志文件位置。
