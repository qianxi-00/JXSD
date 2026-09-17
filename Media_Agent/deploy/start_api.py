# -*- coding: utf-8 -*-
"""抖音采集服务的启动器 —— 放在镜像里替换默认的 start.sh。

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
"""
import os
import re

COOKIE = (os.environ.get("MEDIA_DOUYIN_COOKIE") or "").strip()
CFG = "/app/crawlers/douyin/web/config.yaml"


def inject_cookie() -> None:
    """把 MEDIA_DOUYIN_COOKIE 写进容器的爬虫配置（只改 Cookie 那一行）。"""
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
        print(f"[启动器] ⚠️ 读不到 {CFG}: {exc}", flush=True)
        return

    # 只替换第一处 `Cookie:` 的值，其余内容（含注释与 msToken/ttwid 段）保持原样。
    # 值用 YAML 单引号标量（内部单引号翻倍）：Cookie 里有分号/等号/空格，
    # 不加引号多数情况也能解析，但显式加引号更稳。
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

    with open(CFG, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"[启动器] ✅ 已注入 Cookie（{len(COOKIE)} 字符）→ {CFG}", flush=True)


if __name__ == "__main__":
    inject_cookie()
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=80, reload=False, log_level="info")
