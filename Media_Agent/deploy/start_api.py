# -*- coding: utf-8 -*-
"""抖音采集服务的启动器 —— 放在镜像里替换默认的 start.sh。

课案出处：**抖音采集这一段没有对应课案**。课案的做法是 ``sys.path.insert`` + ``import``
本地 erma0/douyin 项目（已因合规审查清空），本项目改成自托管
``Evil0ctal/Douyin_TikTok_Download_API``，本文件就是那次替换的落地产物。

⚠️ 别混淆：课案**确实**有 Docker 内容，但那是**数字人（HeyGem）**模块的
（``docker pull guiji2025/heygem.ai`` + ``docker compose up -d`` + AutoDL 实例），
与这里的抖音采集服务是两回事 —— 本模块在课案里没有前身，
所以也不存在「与课案的差异表」。

为什么不能直接用镜像默认的 start.sh（两处都是实测踩出来的）：

1. 它跑 ``uvicorn.run(..., reload=True)``。
   reload 要起监督进程 + 子进程（多进程 + /dev/shm，容器里默认只有 64MB），
   在本机 Docker Desktop 上**服务起不来**：容器状态是 running，
   但容器内 80 端口始终没有监听，``docker logs`` 里只有 DNS 报错，
   前台跑 60 秒 stdout 一个字都不输出。关掉 reload 后 45 秒内正常启动。

2. 镜像内的 ``crawlers/douyin/web/config.yaml`` **自带一份早就过期的 Cookie**，
   抖音对采集接口直接返回 **403**。
   而且客户端在 HTTP 请求头里传的 Cookie **不会**被服务端转发给抖音 ——
   实测「带 Cookie」与「不带 Cookie」返回的是完全相同的 400，
   说明那个头根本没参与上游请求。**必须把有效 Cookie 写进容器内这份配置。**

Cookie 的唯一真源是根 ``.env`` 的 ``MEDIA_DOUYIN_COOKIE``，
由 compose 的 ``env_file`` 传进容器；这里只做注入，不留副本。

配合关系（谁调它、它又依赖谁）
    ``deploy/douyin-api.compose.yml`` 把它挂成 ``/app/start_api.py``（只读），
    并用 ``command: ["python3", "-u", "/app/start_api.py"]`` 覆盖镜像默认的启动命令
    （镜像默认跑的是那个带 ``reload=True`` 的 ``start.sh``）；
    容器内监听 80，映射到宿主机 8080，与 ``config.py`` 的
    ``douyin_api_base = "http://127.0.0.1:8080"`` 对齐。

⚠️ 本模块**没有自检**：``if __name__ == "__main__"`` 里是真正的启动逻辑，
在镜像里跑就会常驻监听，不能当自检用。**在本机直接跑它则必然失败**（实测 exit 1）::

    [启动器] ⚠️ 没收到 MEDIA_DOUYIN_COOKIE，沿用镜像自带 Cookie（…）
    ModuleNotFoundError: No module named 'app'      ← ``app.main:app`` 只存在于镜像内

好消息是它**不会误开服务**：``reload=False`` 时 uvicorn 先 ``config.load_app()``
导入应用字符串、再建 Server 绑端口（``uvicorn/main.py`` 的 ``run()``），
所以在导入这一步就倒了，宿主机上不会留下监听的端口。
本文件因此也不在 ``verify_all.py`` 的模块自检清单里。

**它只能在容器里跑**：``app.main:app`` 与 ``/app/crawlers/douyin/web/config.yaml``
都是镜像内部的布局，宿主机上没有对应物，所以本机**不存在**可以跑起来的入口。

那「想在本机验证它」该验证什么？——**验证它拉起来的那个服务，而不是它本身**。
它只做两件事：把 Cookie 写进容器内的爬虫配置、再以 ``reload=False`` 拉起 uvicorn；
两件事都能在容器起来之后从外面观察（第 1、3 两条在 ``VERIFY_REPORT.md`` 第 8 节
第 4 条里有实测记录；第 2 条的日志文本就取自本文件 ``inject_cookie()`` 里那行 ``print``）：

1. 服务有没有真起来 —— ``GET http://127.0.0.1:8080/docs`` 返回 **200**；
2. Cookie 有没有注入成功 —— ``docker logs`` 里出现
   ``[启动器] ✅ 已注入 Cookie（N 字符）→ /app/crawlers/douyin/web/config.yaml``
   （没注入时会走上面那个 ⚠️ 分支，日志里同样看得见）；
3. 注入有没有真生效 —— ``/api/douyin/web/handler_user_profile`` 能返回 200 带真实用户数据，
   而不是因 Cookie 被拒。

复现命令与容器名见 ``deploy/douyin-api.compose.yml`` 与 ``README.md``「数据复盘」一节。
"""
import os
import re

# Cookie 从容器环境变量读（compose 的 env_file 把根 .env 传进来）。
# 在 import 时就取一次：进程活着的期间不需要重读，重启容器即可换 Cookie。
COOKIE = (os.environ.get("MEDIA_DOUYIN_COOKIE") or "").strip()
# 镜像内爬虫配置的绝对路径 —— 这是镜像里的既定布局（工作目录是 /app），
# 不是本项目可配的项；换镜像版本时这里要跟着改。
CFG = "/app/crawlers/douyin/web/config.yaml"


def inject_cookie() -> None:
    """把 MEDIA_DOUYIN_COOKIE 写进容器的爬虫配置（只改 Cookie 那一行）。

    每个失败分支都**只打印中文提示并返回**，不抛异常、也不 ``sys.exit``：
    容器一旦退出，连 ``/docs`` 和 ``handler_user_profile`` 这些还能用的端点都没了；
    而带着过期 Cookie 起来，至少能让客户端拿到可读的 403，比整个服务不起来强。

    Returns:
        None: 无返回值。以下情况都算「正常返回」——
        环境变量没传 Cookie、配置文件读不到、文件里没有 Cookie 行、写回失败。
    """
    if not COOKIE:
        print(
            "[启动器] ⚠️ 没收到 MEDIA_DOUYIN_COOKIE，沿用镜像自带 Cookie"
            "（那份早已过期，抖音会回 403）。请在根 .env 里填好再重启容器。",
            flush=True,
        )
        return

    try:
        with open(CFG, encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        # 镜像布局变了（路径不存在 / 无权限）也会走到这里，同样不阻断启动
        print(f"[启动器] ⚠️ 读不到 {CFG}: {exc}", flush=True)
        return

    # 只替换第一处 `Cookie:` 的值，其余内容（含注释与 msToken/ttwid 段）保持原样。
    # 值用 YAML 单引号标量（内部单引号翻倍）：Cookie 里有分号/等号/空格，
    # 不加引号多数情况也能解析，但显式加引号更稳。
    # ``(?m)`` 让 ``^`` 匹配每一行的行首；``count=1`` 是关键 —— 配置文件后面还有
    # 别的地方出现 "Cookie" 字样（如注释或其它站点的段），全替会改坏别的站点配置。
    quoted = "'" + COOKIE.replace("'", "''") + "'"
    new_text, n = re.subn(
        r"(?m)^(\s*)Cookie:.*$",
        lambda m: f"{m.group(1)}Cookie: {quoted}",
        text,
        count=1,
    )
    if n == 0:
        print(f"[启动器] ⚠️ {CFG} 里没找到 Cookie 行，未注入", flush=True)
        return

    # 容器可写层 / 挂载卷上原地覆盖：只写回本进程刚读的那份，不影响宿主机文件
    with open(CFG, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"[启动器] ✅ 已注入 Cookie（{len(COOKIE)} 字符）→ {CFG}", flush=True)


if __name__ == "__main__":
    # ⚠️ 这里**不是自检**，别照着别的模块的写法当成自检来跑：
    #    在容器里跑它 = 常驻前台服务；在宿主机跑它 = ModuleNotFoundError（见文件头）。
    #    想验证它，请按文件头那三条去查容器里的服务，而不是跑这个文件。
    # 执行顺序不能换：**先注入 Cookie，再起 uvicorn**。
    # 反过来的话服务已经开始接请求（爬虫模块在启动时读配置），注入就晚了。
    inject_cookie()
    import uvicorn

    # ⚠️ 本文件存在的唯一理由就在这个 ``reload=False``：镜像默认的 start.sh 用的是
    #    ``uvicorn.run(..., reload=True)``（开发特性，要起监督进程 + 子进程，
    #    容器里 /dev/shm 默认只有 64MB），实测在本机 Docker Desktop 上**服务起不来**
    #    —— 容器状态是 running、容器内 80 端口始终没监听、``docker logs`` 只有 DNS 报错、
    #    前台跑 60 秒 stdout 一个字都不输出。关掉 reload 后 45 秒内启动、``GET /docs`` 返回 200。
    #    常驻服务本来就该关掉 reload，**别把这一行改回 True**。
    # ``log_level="info"`` 配合 compose 里的 ``command: ["python3", "-u", ...]``：
    #    保证日志实时刷到 ``docker logs``，否则排查下面那个「什么都不输出」时会误判成进程挂了。
    # ``port=80`` 是镜像内 ``config.yaml`` 的 API.Host_Port，由 compose 映射到宿主 8080。
    uvicorn.run("app.main:app", host="0.0.0.0", port=80, reload=False, log_level="info")
