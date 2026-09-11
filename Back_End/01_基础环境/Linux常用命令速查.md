# Linux 常用命令速查（后端开发基础 · Linux 章节）

> 对应课案章节：`后端开发基础 → Linux`
> 本文是**讲解型速查表**：先讲原理，再给命令。命令不需要"运行"，但需要你能在真实服务器/Docker 容器里照着敲。

---

## 一、什么是 Linux

Linux 严格来说只是一个**操作系统内核（Kernel）**：它负责管理 CPU 调度、内存分配、磁盘 IO、网络协议栈、设备驱动。
我们平时说的"Linux 系统"其实是 **内核 + GNU 工具集 + 包管理器 + 桌面/服务程序** 的组合，这个组合叫做**发行版（Distribution）**。

**为什么后端必须学 Linux？**

| 优势 | 说明 |
| --- | --- |
| 开源免费 | 内核与大部分工具代码公开，服务器不用买授权 |
| 稳定可靠 | 可以连续运行几年不重启，适合长期在线的服务 |
| 安全性高 | 多用户 + 严格权限模型，服务以最小权限运行 |
| 社区支持 | 遇到问题几乎都能搜到答案 |
| 后端主流 | 绝大多数服务器、容器镜像、云主机都是 Linux |

**一句话理解后端部署链路：**
`你的代码 → 打进 Docker 镜像（Linux 文件系统）→ 跑在云服务器（Linux）上 → 通过端口对外提供服务`

---

## 二、最小化 Linux 镜像（理解"内核 + 工具集"的最佳方式）

Docker 镜像本质上就是一个**精简的 Linux 根文件系统**。越小越好，因为它决定了拉取速度和被攻击面。

### 1. busybox（约 1 MB）

静态编译的 BusyBox，一个可执行文件里塞进了 `ls`、`pwd`、`cat`、`echo`、`vi`、`sh` 等所有常用基础命令，是最接近"只用内核"的可运行系统。

```bash
docker pull busybox
docker run -it --rm busybox     # -it 交互式终端，--rm 退出后自动删除容器
```

进入后你只能使用 BusyBox 自带的那批命令，**没有 apt / yum，也没有 glibc 的完整版**，所以不适合跑 Python 之类依赖动态库的程序。

### 2. alpine（约 5 MB）

比 BusyBox 多了 `apk` 包管理器和 musl libc，可以现场装软件，是生产环境最常用的最小化基础镜像。

```bash
docker pull alpine
docker run -it --rm alpine

# 容器内执行：
apk update            # 1. 更新本地包索引（相当于 apt update）
apk add curl vim      # 2. 安装单个/多个包
```

> 注意：alpine 用的是 musl libc 而不是 glibc，某些预编译轮子（wheel）装不上，需要重新编译。
> 后端推荐基础镜像排序：`python:3.12-slim`（glibc，兼容最好）> `alpine`（最小但坑多）。

---

## 三、发行版家族

Linux 发行版可以粗略分成两大"家族"，家族决定了你用哪个**包管理器**：

```
Red Hat 家族                          Debian 家族
├── RHEL（Red Hat 商业旗舰）          ├── Debian（上游）
├── CentOS Linux（2021 年停更）       └── Ubuntu（最成功的衍生版）
├── CentOS Stream
└── AlmaLinux（CentOS 停更后的主流替代）
```

| 家族 | 代表发行版 | 包管理器 | 包格式 |
| --- | --- | --- | --- |
| Red Hat | RHEL / AlmaLinux / Rocky | `yum` / `dnf` | `.rpm` |
| Debian | Debian / Ubuntu | `apt` | `.deb` |

国内云厂商的"Alibaba Cloud Linux"、"TencentOS"属于 Red Hat 系兼容分支，命令与 AlmaLinux 基本一致。

---

## 四、在 Docker 里"随时开一台 Linux"

学习阶段不需要买服务器，用 Docker 起一个 Ubuntu / AlmaLinux 容器即可，用完删掉。

```shell
# 1. 拉取并后台启动 Ubuntu 容器
docker pull ubuntu:24.04
docker run -d --name myubuntu ubuntu:24.04 sleep infinity   # 让容器不退出
docker exec -it myubuntu /bin/bash                          # 进入容器的 shell

# 2. AlmaLinux 同理
docker pull almalinux:8
docker run -d --name myalma almalinux:8 sleep infinity
docker exec -it myalma /bin/bash
```

- `-d`：后台运行（detached）
- `--name`：给容器起名字，后面可以直接用名字操作
- `sleep infinity`：容器必须有"前台进程"才不会退出，这里用死循环占位
- `docker exec -it`：进入一个**正在运行**的容器并开交互式终端

---

## 五、文件与目录操作

Linux 一切皆文件：目录、磁盘、网卡、进程信息都能以文件形式访问（如 `/proc`、`/dev`）。

```bash
# ---------- 查看当前位置 ----------
pwd                     # print working directory，查看当前目录

# ---------- 列出文件 ----------
ls                      # 列出当前目录文件
ls -a                   # 列出所有文件（包括以 . 开头的隐藏文件）
ls -l                   # 长格式：权限、链接数、所有者、大小、修改时间
ls -la                  # 两者组合（最常用）
ls -lh                  # 配合 -l 使用，把大小显示成 K/M/G 更易读

# ---------- 切换目录 ----------
cd /                    # 切换到根目录
cd ~                    # 切换到当前用户主目录（root 的 ~ 是 /root）
cd -                    # 回到上一次所在目录
cd ..                   # 上一级目录
cd /home                # 绝对路径
cd ./conf               # 相对路径

# ---------- 创建目录 ----------
mkdir mydir             # 创建单个目录
mkdir -p dir1/dir2/dir3 # -p：递归创建多级目录（父目录不存在也不报错）

# ---------- 删除 ----------
rmdir mydir             # 只能删空目录
rm -r mydir             # 递归删除目录及其内容（会逐个询问）
rm -rf mydir            # 强制递归删除，不询问（危险！根目录下慎用）
rm -f file.txt          # 强制删除文件

# ---------- 复制 / 移动 ----------
cp file1 file2          # 复制文件
cp -r dir1 dir2         # 递归复制目录
cp -a dir1 dir2         # 归档复制，保留权限/时间戳（备份常用）
mv file1 file2          # 移动；目标在同一目录时就是"重命名"
mv dir1 /opt/           # 移动目录

# ---------- 查看 / 搜索文件内容 ----------
cat file.txt            # 一次全部打印
head -n 10 file.txt     # 前 10 行
tail -n 10 file.txt     # 后 10 行
tail -f app.log         # 实时跟踪追加内容（看日志必备）
wc -l file.txt          # 统计行数
find / -name "*.log"    # 按名字查找文件
grep -rn "ERROR" /var/log   # 在目录里递归搜索文本
```

### Vim 极简常用命令

服务器上默认只有 vim/vi，必须会一点：

| 分类 | 按键 | 作用 |
| --- | --- | --- |
| 模式 | `i` | 进入插入模式 |
| 模式 | `ESC` | 返回普通模式 |
| 模式 | `:` | 进入底行命令模式 |
| 移动 | 方向键 / `hjkl` | 左 下 上 右 |
| 移动 | `gg` / `G` | 跳到首行 / 末行 |
| 编辑 | `dd` | 删除（剪切）当前行 |
| 编辑 | `yy` / `p` | 复制当前行 / 粘贴 |
| 编辑 | `u` | 撤销 |
| 保存退出 | `:w` / `:q` / `:wq` / `:q!` | 保存 / 退出 / 保存退出 / 强制退出不保存 |
| 查找替换 | `/内容`、`n`、`N` | 查找、下一个、上一个 |
| 查找替换 | `:%s/旧/新/g` | 全文替换 |

---

## 六、软链接与硬链接

**原理先行：** Linux 文件由两部分组成——**目录项（文件名）** 和 **inode（真正记录数据块位置的结构）**。
- **硬链接**：新建一个目录项，直接指向**同一个 inode**。等价于给同一份数据起了第二个名字，删除任意一个名字数据都还在（inode 的链接计数减 1，减到 0 才释放数据）。
- **软链接（符号链接）**：新建一个**独立的文件**，内容就是"目标路径"这个字符串。类似 Windows 快捷方式，目标被删掉它就变成"悬空链接"。

```bash
# 软链接（符号链接）
ln -s /path/to/target linkname      # -s = symbolic
# 最经典的后端用法：Nginx 站点配置"启用"其实就是建软链接
ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/myapp

# 硬链接
ln /path/to/target linkname

# 查看链接信息：软链接会显示  linkname -> /path/to/target
ls -l linkname

# 删除链接（软链接只是删掉那个"快捷方式"，不影响原文件）
rm linkname
```

| 对比项 | 软链接 | 硬链接 |
| --- | --- | --- |
| 原理 | 存的是目标**路径字符串** | 与目标共享**同一份数据块** |
| 跨磁盘/分区 | ✅ 支持 | ❌ 不支持 |
| 指向目录 | ✅ 支持 | ❌ 不支持（普通用户） |
| 原文件删除后 | 链接失效（悬空链接） | 数据仍可访问 |
| inode | 有自己的 inode | 与目标同一个 inode |
| 常见用途 | 配置文件快捷方式、版本切换 | 文件备份、节省空间 |

> 生产小技巧：发布新版本时用软链接做原子切换
> `ln -sfn /opt/app/releases/v2 /opt/app/current`，回滚只需再指回 v1。

---

## 七、文件权限管理

**原理：** Linux 是**多用户**系统，每个文件都记录 **所有者（owner）**、**所属组（group）**、**其他人（others）** 三类身份，每类身份各有 `r/w/x` 三个权限位。

```
-rwxr-xr-x  1 root root  1234 May 1 10:00 script.sh
│└┬┘└┬┘└┬┘
│ │  │  └── 其他人：r-x
│ │  └───── 所属组：r-x
│ └──────── 所有者：rwx
└────────── 文件类型：- 普通文件、d 目录、l 软链接
```

| 权限 | 对文件 | 对目录 |
| --- | --- | --- |
| `r` (4) | 读取内容 | 可以 `ls` 列出条目 |
| `w` (2) | 修改内容 | 可以在目录内创建/删除文件 |
| `x` (1) | 作为程序执行 | 可以 `cd` 进入目录 |

**八进制换算：** `rwx` 三位二进制，从左到右每位权重 `4 / 2 / 1`：

| 权限位 | 二进制 | 八进制 |
| --- | --- | --- |
| `---` | 000 | 0 |
| `--x` | 001 | 1 |
| `-w-` | 010 | 2 |
| `-wx` | 011 | 3 |
| `r--` | 100 | 4 |
| `r-x` | 101 | 5 |
| `rw-` | 110 | 6 |
| `rwx` | 111 | 7 |

例：`rwxr-xr-x` = `7` + `5` + `5` = **755**；`rw-r--r--` = `6` + `4` + `4` = **644**。
**记忆技巧：`r=4`、`w=2`、`x=1`，三类身份相加。**

| 八进制 | 权限 | 用途 |
| --- | --- | --- |
| 755 | rwxr-xr-x | 可执行文件、目录 |
| 644 | rw-r--r-- | 普通文件 |
| 600 | rw------- | 敏感文件（密钥、`.env`） |
| 777 | rwxrwxrwx | 所有人可读写执行（**不安全，禁止在生产使用**） |

```bash
# 修改权限
chmod 755 file.txt      # 直接给八进制（最常用）
chmod +x script.sh      # 给所有身份加执行权限
chmod -x file.txt       # 移除执行权限
chmod -R 755 dir        # -R 递归处理目录

# 修改所有者和所属组
chown user file.txt         # 只改所有者
chown user:group file.txt   # 同时改所有者和组
chown -R user:group dir     # 递归修改
chgrp group file.txt        # 只改组
```

> 安全实践：`.env`、私钥文件用 `chmod 600`；Web 目录用 `755`；绝对不要 `chmod -R 777`。

---

## 八、进程管理

**原理：** 每个运行中的程序都是一个**进程**，操作系统给它分配一个唯一编号 **PID**。父进程创建子进程，所有进程最终挂在 1 号进程（systemd）下。

```bash
# ---------- 查看进程 ----------
ps              # 只看当前终端的进程
ps -ef          # 看全部进程（System V 风格，含 PPID 父进程号）
ps aux          # 看全部进程的详细信息（BSD 风格，最常用）
ps aux | grep python    # 组合 grep 找特定进程

# ---------- 实时查看 ----------
top             # 动态刷新的进程面板（按 q 退出、P 按 CPU 排序、M 按内存排序）
htop            # 更友好的 top（需要 apt install htop）

# ---------- 结束进程 ----------
kill 1234       # 发送 SIGTERM(15)，请求进程优雅退出
kill -9 1234    # 发送 SIGKILL(9)，强制杀死（无法被捕获，可能丢数据）
pkill -f uvicorn  # 按命令行关键字批量结束

# ---------- 后台运行 ----------
python main.py &        # 放到后台运行（关掉终端就死）
nohup python main.py > app.log 2>&1 &   # 忽略挂断信号 + 重定向日志，关终端也不死

# ---------- 作业控制 ----------
jobs            # 查看当前 shell 的后台任务
fg %1           # 把 1 号后台任务调到前台
bg %1           # 把暂停的任务继续放后台
```

### `ps aux` 输出字段详解

| 字段 | 全称 / 含义 | 核心作用 |
| --- | --- | --- |
| `USER` | 进程所有者用户名 | 权限排查关键：这是谁启动的 |
| `PID` | Process ID | 进程唯一标识，`kill`/`top` 的操作依据 |
| `%CPU` | CPU 占用率 | 多核系统可超过 100%，代表占用多个核心 |
| `%MEM` | 内存占用率 | 物理内存百分比，反映内存负载 |
| `VSZ` | Virtual Size | 虚拟内存总量（KB），含代码、共享库、交换空间 |
| `RSS` | Resident Set Size | **实际**占用的物理内存（KB），排查内存泄漏看它 |
| `TTY` | 关联终端 | `?` 表示与终端无关（后台服务/守护进程） |
| `STAT` | 进程状态 | `R` 运行、`S` 可中断休眠、`D` 不可中断、`Z` 僵尸、`T` 停止 |
| `TIME` | CPU 累计时间 | 自启动以来消耗的 CPU 时间，**不是**运行时长 |
| `COMMAND` | 命令行信息 | 启动进程的完整命令，用于定位进程来源 |

> 后端排障顺序：`ps aux | grep 服务名` → 看 PID → `kill PID` → 再用 `systemctl`/`docker` 重启。

---

## 九、网络管理

**原理：** 服务器程序要在某个 **IP:端口** 上"监听（listen）"，客户端才能连上。后端 80% 的"部署成功但访问不了"都是端口/网络问题。

```bash
# ---------- 安装网络工具（不同发行版命令不同） ----------
apt install -y iproute2 net-tools iputils-ping curl lsof          # Ubuntu/Debian
yum install -y iproute net-tools iputils curl lsof procps-ng      # AlmaLinux/CentOS

# ---------- 查看网络接口 ----------
ifconfig            # 旧版工具（net-tools）
ip addr             # 新版（iproute2），推荐
ip route            # 查看路由表/默认网关

# ---------- 查看监听端口 ----------
netstat -tuln       # t=TCP u=UDP l=listen n=数字显示端口（旧）
ss -tulnp           # 更现代更快，-p 显示所属进程（推荐）

# ---------- 测试连通性 ----------
ping www.baidu.com          # ICMP 测试，看丢包与延迟
curl -I www.baidu.com       # 只看 HTTP 响应头，验证 HTTP 链路
curl -v http://127.0.0.1:8000/health   # -v 显示完整请求/响应过程

# ---------- 端口占用排查 ----------
lsof -i :8080               # 谁占用了 8080
netstat -tulnp | grep 8080  # 同上，另一种写法
fuser -k 8080/tcp           # 强杀占用者（慎用）

# ---------- 下载 ----------
wget https://example.com/file.tar.gz
curl -O https://example.com/file.tar.gz
```

### SSH 远程连接

```bash
ssh user@192.168.1.100          # 默认 22 端口
ssh -p 2222 user@host           # 指定端口
ssh -i ~/.ssh/id_rsa user@host  # 指定私钥
scp file.tar.gz user@host:/opt/ # 上传文件
```

> Windows 上也可以用图形化工具 **FinalShell**（内置 SFTP 文件浏览器 + 终端），操作方式和 SSH 完全一致：填公网 IP、端口 22、用户名 root、密码。

---

## 十、系统管理

```bash
# ---------- 系统信息 ----------
uname -a            # 内核版本、主机名、架构
hostname            # 主机名
df -h               # 磁盘使用情况（-h 人类可读）
du -sh /var/log     # 统计目录占用大小
free -h             # 内存使用情况（yum install -y procps-ng）
nproc               # CPU 核心数
uptime              # 运行时长与负载
cat /etc/os-release # 查看发行版版本

# ---------- 日志 ----------
cat /var/log/dpkg.log       # 一次性查看系统日志
tail -n 100 /var/log/syslog # 看最后 100 行
tail -f /var/log/nginx/error.log  # 实时跟踪（-f = follow）

# ---------- 服务管理（systemd，容器内一般不可用，需要真实/虚拟机系统） ----------
systemctl list-units --type=service --all  # 列出所有服务
systemctl start   service   # 启动
systemctl stop    service   # 停止
systemctl restart service   # 重启
systemctl status  service   # 查看状态
systemctl enable  service   # 开机自启
systemctl disable service   # 取消自启
journalctl -u service -f    # 实时看某个服务的日志
```

> 容器里没有 systemd（除非特权容器 + 特殊镜像），所以 `systemctl` 在 Docker 里通常报 `System has not been booted with systemd`，这属于正常现象，不是环境坏了。

---

## 十一、后端常见 Linux 场景速查

| 场景 | 命令 |
| --- | --- |
| 看服务是否在监听 | `ss -tulnp \| grep 8000` |
| 实时看应用日志 | `tail -f /opt/app/logs/app.log` |
| 临时开个 HTTP 服务传文件 | `python3 -m http.server 8000` |
| 找哪个进程占端口 | `lsof -i :8000` |
| 查看磁盘是否满了导致写日志失败 | `df -h` |
| 查看内存是否被 OOM | `dmesg \| grep -i "killed process"` |
| 统计某接口被访问次数 | `grep "GET /api" access.log \| wc -l` |
| 批量结束同名进程 | `pkill -f "uvicorn main:app"` |
