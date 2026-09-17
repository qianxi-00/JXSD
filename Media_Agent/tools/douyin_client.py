# -*- coding: utf-8 -*-
"""抖音作品数据采集客户端 —— 自托管 Douyin_TikTok_Download_API

课案出处：自媒体课案 → 数据复盘 → 「从抖音个人主页采集作品数据」那一段

课案原文怎么做的
    课案用 ``sys.path.insert`` 把本地项目 **erma0/douyin** 挂进来，再
    ``from backend.lib.douyin import Douyin``，直接用它的爬虫抓主页作品；
    Cookie 从那个项目的 ``config/settings.json`` 读。

为什么必须换方案
    ① erma0/douyin 已因合规审查于 2026-06 被清空（仓库只剩 README），
       ``backend.lib.douyin`` 这个模块根本不存在了 —— 课案的接法是 ModuleNotFoundError；
    ② 课案还把绝对路径 ``C:/Users/13261/Documents/project/douyin`` 写死在代码里。

本项目方案：自托管 Evil0ctal/Douyin_TikTok_Download_API（REST 接口，仍在维护）::

    # 用项目里的 compose 定义（推荐，已把 reload 关掉）：
    cd Media_Agent/deploy
    docker compose -f douyin-api.compose.yml up -d

    # 等价的手写命令：
    docker run -d --name dtk-v4 --restart unless-stopped -p 8080:80 \
      evil0ctal/douyin_tiktok_download_api:V4.1.2 \
      python3 -u -c "import uvicorn; uvicorn.run('app.main:app', \
        host='0.0.0.0', port=80, reload=False, log_level='info')"

⚠️ **不要直接用镜像默认的 `start.sh`**（也就是不写上面最后那行 `python3 -u -c ...`）：
它跑的是 ``uvicorn.run(..., reload=True)``，实测在本机 Docker Desktop 上
**服务起不来** —— 容器状态是 running、但容器内 80 端口始终没有监听，
``docker logs`` 里只有 DNS 报错，前台跑 60 秒 stdout 一个字都不输出。
reload 要起监督进程 + 子进程（多进程 + /dev/shm，容器里默认仅 64MB），
而 reload 本来就是开发特性，常驻服务应当关掉。关掉后 45 秒内启动、``GET /docs`` 返回 200。

与课案的落地差异
    | 项 | 课案 | 本项目 |
    |---|---|---|
    | 数据源 | 本地 erma0/douyin 爬虫（已失效） | 自托管 Douyin_TikTok_Download_API 的 REST 接口 |
    | 引入方式 | ``sys.path.insert`` + ``import`` | HTTP GET，不再动 ``sys.path`` |
    | 服务地址 | 硬编码绝对路径 | ``settings.media.douyin_api_base``（默认 http://127.0.0.1:8080） |
    | Cookie | 读对方项目的 settings.json | ``settings.media.douyin_cookie``（根 .env），页面可运行时覆写 |
    | 字段映射 | 对方 parser 已把 statistics 摊平到顶层 | 先查嵌套 ``statistics`` 再查顶层，两种写法都认 |
    | 服务不可达 | 抛异常 | 返回 ``{"items": [], "error": "中文说明"}``，另有 ``parse_manual_json()`` 降级入口 |

端点是怎么定的（**核对过 V4.1.2 源码，不是猜的**）
    ::

        app/main.py                      app.include_router(api_router, prefix="/api")
        app/api/router.py                router.include_router(douyin_web.router, prefix="/douyin/web")
        app/api/endpoints/douyin_web.py  @router.get("/fetch_user_post_videos", ...)
                                         sec_user_id / max_cursor / count
        app/api/models/APIResponseModel.py  → {"code": 200, "router": "...", "data": <作品数据>}

    ⇒ ``GET /api/douyin/web/fetch_user_post_videos?sec_user_id=..&max_cursor=0&count=20``

    ⚠️ **仍未核实的一点**：只知道外层信封，**不知道 ``data`` 内部是
    ``{"aweme_list": [...]}`` 还是别的层级**（那需要真跑起服务才能看到）。
    所以 ``_extract_aweme_list()`` 会把常见层级都试一遍，而不是只认一个键。
    ⚠️ 上游 v5（现 main 分支）改成了 ``/api/v1/...`` 且要 API Key，本文件按 v4 写；
    换版本 / 换挂载前缀时**只改下面的 ``ENDPOINT_USER_POST_VIDEOS`` 常量**即可。

踩过的坑
    · 本机很可能**没有跑这个 Docker 服务**，且 agent 无权启动它 ——
      所以所有对外函数都按「服务不可达」这条路径设计，绝不抛异常。
    · 抖音原始数据里 ``video.duration`` 是**毫秒**（课案也是 /1000）；
      但手动粘贴的数据里常直接写 ``45`` 或 ``"45秒"``，所以 ``_to_duration_text()`` 两种都认。
    · 只有本人主页才有真实播放量，别人的主页 ``play_count`` 恒为 0 ——
      课案据此决定要不要输出「播放」字段，这里保留同一判断。
"""

import csv
import io
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List

import requests

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# 一、端点与字段配置（换接口版本只改这一节）
# ============================================================

# 抖音「获取用户主页作品数据」接口路径（含 /api 与 /douyin/web 两段前缀）。
ENDPOINT_USER_POST_VIDEOS = "/api/douyin/web/fetch_user_post_videos"

# 页面运行时 Cookie 覆写用的环境变量名。
# 为什么要有它：settings 是 import 时构造成型的（lru_cache），页面里粘的 Cookie
# 没法回写进 settings；而契约把 fetch_user_videos() 的签名冻死了（不收 cookie 参数）。
# 所以留这一个「调用时读取」的覆写点，别的模块不用感知。
COOKIE_ENV_KEY = "MEDIA_DOUYIN_COOKIE"

# 每个计数指标的候选来源键：中文键（手动粘贴） → 抖音原始英文键 → 驼峰写法
STAT_KEYS: Dict[str, tuple] = {
    "点赞": ("点赞", "digg_count", "diggCount"),
    "评论": ("评论", "comment_count", "commentCount"),
    "分享": ("分享", "share_count", "shareCount"),
    "收藏": ("收藏", "collect_count", "collectCount"),
    "播放": ("播放", "play_count", "playCount"),
}

# 作品列表可能出现的键名（不同版本 / 不同信封层级）
_LIST_KEYS = ("aweme_list", "awemeList", "items", "list")
# 需要再往里钻一层的信封键
_ENVELOPE_KEYS = ("data", "aweme_detail", "result")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# sec_user_id 固定以 MS4wLjABAAAA 开头（抖音自己生成的编码）
_SEC_UID_RE = re.compile(r"(MS4wLjABAAAA[A-Za-z0-9_\-]+)")
_USER_PATH_RE = re.compile(r"/user/([A-Za-z0-9_\-]+)")

# 给页面文本框用的示例数据（自检也用它，避免两处样例漂移）
SAMPLE_MANUAL_JSON = json.dumps(
    [
        {"标题": "AI工具提效实测", "发布时间": "2026-09-01", "时长": "45秒",
         "点赞": 1200, "评论": 35, "分享": 12, "收藏": 180},
        {"标题": "三个Python技巧", "发布时间": "2026-08-25", "时长": "60秒",
         "点赞": 320, "评论": 8, "分享": 3, "收藏": 45},
    ],
    ensure_ascii=False,
    indent=2,
)


# ============================================================
# 二、内部小工具
# ============================================================

def _fail(message: str) -> dict:
    """统一的失败返回（所有对外函数失败都走这里，绝不抛异常）。"""
    return {"items": [], "error": message}


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _to_int(value: Any) -> int:
    """尽力转 int：``"1,200"`` / ``1200.0`` / ``None`` 都不炸。"""
    if isinstance(value, str):
        value = value.strip().replace(",", "")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _first(record: dict, keys: tuple) -> Any:
    """按候选键顺序取第一个非空值。"""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _stat(record: dict, keys: tuple) -> int:
    """取一个计数指标：先查嵌套的 ``statistics``，再查顶层（两种数据源写法都认）。"""
    for container in (_as_dict(record.get("statistics")), record):
        for key in keys:
            if key in container:
                return _to_int(container[key])
    return 0


def _to_time_text(value: Any) -> str:
    """``create_time`` 是秒级时间戳 → 本地日期；已经是字符串就原样返回。"""
    if value in (None, ""):
        return "未知"
    ts = _to_int(value)
    if ts > 0:
        if ts > 10_000_000_000:   # 兼容毫秒时间戳
            ts //= 1000
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except (OSError, OverflowError, ValueError):
            pass
    return str(value)


def _to_duration_text(record: dict) -> str:
    """时长 → ``"45秒"``。

    抖音原始数据里 ``video.duration`` 是毫秒；手动粘贴的数据常直接写秒数
    （``45`` 或 ``"45秒"``）。这里三种都认：带非数字字符的原样返回，
    ≥1000 当毫秒，<1000 当秒。
    """
    raw = None
    for container in (_as_dict(record.get("video")), record):
        raw = _first(container, ("时长", "duration", "duration_ms", "video_duration"))
        if raw is not None:
            break
    if raw is None:
        return "未知"
    if isinstance(raw, str) and not raw.strip().isdigit():
        return raw.strip()
    value = _to_int(raw)
    if value <= 0:
        return "未知"
    seconds = value // 1000 if value >= 1000 else value
    return f"{seconds}秒"


def _normalize_item(record: dict) -> Dict[str, Any]:
    """把一条原始作品记录整成课案要的中文键字段。"""
    return {
        "标题": str(_first(record, ("标题", "desc", "title", "caption")) or "（无标题）")[:80],
        "发布时间": _to_time_text(_first(record, ("发布时间", "create_time", "time", "publish_time"))),
        "时长": _to_duration_text(record),
        "点赞": _stat(record, STAT_KEYS["点赞"]),
        "评论": _stat(record, STAT_KEYS["评论"]),
        "分享": _stat(record, STAT_KEYS["分享"]),
        "收藏": _stat(record, STAT_KEYS["收藏"]),
    }


def _build_items(records: List[dict], limit: int = 0) -> List[Dict[str, Any]]:
    """把原始记录批量整成中文键作品列表。

    保留课案的 ``has_play_count`` 判断：**只有本人主页才有真实播放量**，
    别人主页的 play_count 恒为 0 —— 为 0 时不输出「播放」字段，
    免得把 0 当成"这个视频没人看"去做诊断。
    """
    records = [r for r in records if isinstance(r, dict)]
    if limit and limit > 0:
        records = records[:limit]
    has_play = any(_stat(r, STAT_KEYS["播放"]) > 0 for r in records[:5])
    items = []
    for record in records:
        item = _normalize_item(record)
        if has_play:
            item["播放"] = _stat(record, STAT_KEYS["播放"])
        items.append(item)
    return items


def _extract_aweme_list(payload: Any) -> List[dict]:
    """从各种可能的信封里找出作品列表（不知道 data 内部层级，所以逐个试）。"""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in _LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    for key in _ENVELOPE_KEYS:
        if key in payload:
            found = _extract_aweme_list(payload[key])
            if found:
                return found
    return []


def _parse_delimited(text: str) -> List[dict]:
    """解析从表格 / Excel 复制来的 CSV、TSV（课案没有这个降级入口，是本项目补的）。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    delimiter = "\t" if lines[0].count("\t") > lines[0].count(",") else ","
    try:
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except csv.Error as exc:
        print(f"[复盘] CSV 解析失败: {exc}")
        return []
    header = [h.strip().strip('"') for h in rows[0]]
    if len(header) < 2:
        return []
    records = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        records.append({k: v.strip() for k, v in zip(header, row)})
    return records


# ============================================================
# 三、配置读取
# ============================================================

def api_base() -> str:
    """自托管服务根地址（去掉尾部斜杠）。"""
    return (settings.media.douyin_api_base or "").strip().rstrip("/")


def resolve_cookie() -> str:
    """取抖音 Cookie：**页面运行时覆写** 优先，其次根 .env 的 ``MEDIA_DOUYIN_COOKIE``。"""
    return os.environ.get(COOKIE_ENV_KEY, "").strip() or (
        settings.media.douyin_cookie or ""
    ).strip()


def cookie_hint() -> str:
    """给页面显示的一行 Cookie 状态提示。"""
    cookie = resolve_cookie()
    if cookie:
        return f"✅ 已配置抖音 Cookie（{len(cookie)} 字符）"
    return (
        "⚠️ 未配置抖音 Cookie（根目录 .env 的 MEDIA_DOUYIN_COOKIE 为空）。"
        "未登录时接口只能拿到非本人主页的公开数据，且常被风控挡下 —— "
        "点下面的按钮从 Edge 读取，或手动粘贴。"
    )


def is_available(timeout: float = 3.0) -> bool:
    """自托管服务是否可达（探活）。

    只要**收到任何 HTTP 响应**就算可达 —— 不校验状态码，因为不同版本根路径
    给出的东西不一样（v4 是 PyWebIO 页面，v5 是控制台）。连不上/超时才返回 False。
    """
    base = api_base()
    if not base:
        return False
    try:
        requests.get(base, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 —— 探活失败就是「不可达」，不该往外抛
        return False


def extract_sec_user_id(text: str) -> str:
    """从主页链接里取 ``sec_user_id``（纯函数，不发网络请求）。

    支持三种写法：完整主页链接、``.../user/<sec_uid>``、直接粘 sec_uid 本身。
    ``/user/self`` 取不到（那不叫 id），返回空串 —— 由 ``fetch_user_videos()``
    走 ``_resolve_self_sec_uid()`` 再解析一次。
    """
    if not text:
        return ""
    text = text.strip()
    match = _SEC_UID_RE.search(text)
    if match:
        return match.group(1)
    if text.startswith("MS4w"):
        return text
    match = _USER_PATH_RE.search(text)
    if match and match.group(1) != "self":
        return match.group(1)
    return ""


def _resolve_self_sec_uid() -> str:
    """把 ``.../user/self`` 解析成真实 sec_user_id（课案同款：抓主页 HTML 正则提取）。

    需要有效的登录 Cookie；没有 Cookie 时抖音返回的是登录页，正则必然取不到，
    于是返回空串让调用方给出可读的错误提示。
    """
    cookie = resolve_cookie()
    headers = {"User-Agent": _UA}
    if cookie:
        headers["Cookie"] = cookie
    try:
        resp = requests.get("https://www.douyin.com/user/self", headers=headers, timeout=15)
        sec_uid = extract_sec_user_id(resp.text)
        if sec_uid:
            print(f"[复盘] 解析到自己的主页: {sec_uid}")
        return sec_uid
    except Exception as exc:  # noqa: BLE001
        print(f"[复盘] 解析 /user/self 失败: {exc}")
        return ""


# ============================================================
# 四、对外接口
# ============================================================

def fetch_user_videos(profile_url: str, limit: int = 20) -> dict:
    """采集抖音个人主页的作品数据。

    Args:
        profile_url: 主页链接（``https://www.douyin.com/user/xxx``，也接受 ``/user/self``
            或直接粘 sec_user_id）。
        limit: 最多返回多少条。

    Returns:
        ``{"items": [...], "error": str}``。items 每条是中文键::

            {"标题", "发布时间", "时长", "点赞", "评论", "分享", "收藏"}   # 有播放量时加 "播放"

        **任何失败都返回 ``items=[]`` + 中文 error，不抛异常。**
    """
    base = api_base()
    if not profile_url or not profile_url.strip():
        return _fail("请输入抖音用户主页链接")

    profile_url = profile_url.strip()
    # /user/self 是个特殊写法，得先换成真实 sec_user_id
    if profile_url.rstrip("/").endswith("/user/self") or profile_url.strip() == "self":
        sec_user_id = _resolve_self_sec_uid()
        if not sec_user_id:
            return _fail(
                "无法把 /user/self 解析成 sec_user_id（多半是 Cookie 无效或已过期）。\n"
                "两个办法：① 在上方 Cookie 配置里重新获取；"
                "② 直接在浏览器打开自己主页，把地址栏形如 "
                "https://www.douyin.com/user/MS4wLjABAAAA... 的链接粘过来。"
            )
    else:
        sec_user_id = extract_sec_user_id(profile_url)
        if not sec_user_id:
            return _fail(
                "无法从链接里识别 sec_user_id。请使用主页链接形式：\n"
                "https://www.douyin.com/user/MS4wLjABAAAA...（可在浏览器地址栏直接复制）"
            )

    if not base:
        return _fail(
            "未配置抖音采集服务地址（settings.media.douyin_api_base 为空）。\n"
            "请在根目录 .env 里设置 MEDIA_DOUYIN_API_BASE=http://127.0.0.1:8080"
        )

    if not is_available():
        return _fail(
            f"抖音采集服务不可达（{base}）。\n"
            "该服务需要自己起（用项目里的 compose，见 Media_Agent/deploy/）：\n"
            "    cd Media_Agent/deploy\n"
            "    docker compose -f douyin-api.compose.yml up -d\n"
            "⚠️ 别用镜像默认启动命令：它的 uvicorn 开着 reload，本机实测起不来服务。\n"
            "服务没起来时，请改用页面下方的「📋 手动粘贴作品数据」降级入口 —— "
            "诊断链路完全一样，只是跳过采集节点。"
        )

    cookie = resolve_cookie()
    headers = {"User-Agent": _UA, "Accept": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    params = {"sec_user_id": sec_user_id, "max_cursor": 0, "count": max(1, int(limit))}

    print(f"[复盘] 正在采集: {profile_url} → {base}{ENDPOINT_USER_POST_VIDEOS}")
    try:
        resp = requests.get(
            f"{base}{ENDPOINT_USER_POST_VIDEOS}", params=params, headers=headers, timeout=25
        )
    except Exception as exc:  # noqa: BLE001 —— 网络失败是常态，不能往外抛
        print(f"[复盘] 采集请求失败: {exc}")
        return _fail(f"采集请求失败: {exc}")

    if resp.status_code >= 400:
        # v4 出错时 HTTP 400 + {"detail": {...}}；这里把能读到的信息都带上
        detail = resp.text[:300]
        try:
            body = resp.json()
            detail = str(body.get("detail") or body)[:300]
        except ValueError:
            pass
        print(f"[复盘] 接口返回 {resp.status_code}: {detail}")
        return _fail(
            f"采集接口返回 {resp.status_code}: {detail}\n"
            "\n"
            "⚠️ 2026-09 实测：这个错误**多半不是你的问题**。自托管 v4 镜像"
            "（`evil0ctal/douyin_tiktok_download_api:V4.1.2`，2025-03 构建）\n"
            "调抖音 `aweme/v1/web/aweme/post/` 会固定被回 **403**，"
            "因为该端点的请求签名 `a_bogus` 算法已过期。判据：\n"
            "  · 同一个容器、同一份 Cookie，`/handler_user_profile` 能正常返回 200\n"
            "    （说明 Cookie 与网络都没问题）；\n"
            "  · 换成**任意公开大号**（不是本人主页）请求同一端点，同样 400/403\n"
            "    （说明不是账号被风控）。\n"
            "官方 README 也写明 **v4 没有自维护的身份池**，v5 才有。\n"
            "\n"
            "可行的出路：\n"
            "  1. 短期：用页面下方的「📋 手动粘贴作品数据」降级入口 ——\n"
            "     后面的漏斗诊断 / 内容评估 / 优化策略**完全一样**，只是数据要手工取；\n"
            "  2. 长期：迁移到 v5（`/api/v1/...` + API Key + 控制台），\n"
            "     但它要 4 个容器、且其中 2 个需从源码构建，是一次独立改造。"
        )

    try:
        payload = resp.json()
    except ValueError:
        return _fail(f"采集接口返回的不是 JSON（前 200 字符）: {resp.text[:200]}")

    if isinstance(payload, dict):
        code = payload.get("code")
        if code is not None and _to_int(code) != 200:
            return _fail(
                f"采集接口业务错误 code={code}: {str(payload.get('message') or payload)[:200]}"
            )

    records = _extract_aweme_list(payload)
    if not records:
        return _fail(
            "采集接口调用成功，但没解析出作品列表。\n"
            "可能原因：该主页没有公开作品、Cookie 无权限，或服务版本与代码里的端点不匹配"
            f"（当前用 {ENDPOINT_USER_POST_VIDEOS}）。"
        )

    items = _build_items(records, limit=limit)
    print(f"[复盘] 采集完成: {len(items)} 条作品")
    return {"items": items, "error": ""}


def parse_manual_json(raw: str) -> dict:
    """降级入口：把手动粘贴的 JSON / CSV / TSV 文本转成同样的 ``{"items", "error"}``。

    支持三种形态：
      · JSON 数组，字段用中文键（标题/点赞/…）或抖音原始英文键（desc/digg_count/…）；
      · 抖音接口的完整响应（``{"code":200,"data":{"aweme_list":[...]}}``），自动扒出列表；
      · 从表格 / Excel 复制出来的 CSV 或 TSV（首行是表头）。
    """
    if not raw or not raw.strip():
        return _fail("请粘贴作品数据（JSON 数组，或从表格复制的 CSV/TSV）")

    text = raw.strip()
    parsed_ok = True
    try:
        payload = json.loads(text)
    except ValueError:
        parsed_ok = False

    records = _extract_aweme_list(payload) if parsed_ok else _parse_delimited(text)
    if not records:
        return _fail(
            "没能从粘贴内容里解析出作品数据。\n"
            "支持：JSON 数组、抖音接口原始响应、或从表格复制的 CSV/TSV（首行表头）。"
        )

    items = _build_items(records)
    if not items:
        return _fail("解析结果为空，请检查每行是否至少包含「标题」或「desc」字段")
    print(f"[复盘] 手动数据解析完成: {len(items)} 条作品")
    return {"items": items, "error": ""}


# ============================================================
# 五、离线自检（服务没起来也必须全绿）
# ============================================================

if __name__ == "__main__":
    print("=== 抖音采集客户端自检 ===")

    # 1) sec_user_id 提取（纯函数）
    uid = "MS4wLjABAAAANXSltcLCzDGmdNFI2Q_QixVTr67NiYzjKOIP5s03CAE"
    assert extract_sec_user_id(f"https://www.douyin.com/user/{uid}") == uid
    assert extract_sec_user_id(f"https://www.douyin.com/user/{uid}?tab=post") == uid
    assert extract_sec_user_id(uid) == uid
    assert extract_sec_user_id("https://www.douyin.com/user/self") == ""
    assert extract_sec_user_id("") == ""
    assert extract_sec_user_id("随便一段文字") == ""
    print("  extract_sec_user_id      OK  链接 / 裸 id / self / 垃圾输入")

    # 2) 手动 JSON（页面降级入口的主路径）
    r = parse_manual_json(SAMPLE_MANUAL_JSON)
    assert r["error"] == "", r["error"]
    assert len(r["items"]) == 2, r["items"]
    first = r["items"][0]
    assert first["标题"] == "AI工具提效实测"
    assert first["点赞"] == 1200 and first["评论"] == 35
    assert first["分享"] == 12 and first["收藏"] == 180
    assert first["时长"] == "45秒" and first["发布时间"] == "2026-09-01"
    assert "播放" not in first, "样例没有播放量，不该凭空补一个"
    print(f"  parse_manual_json(JSON)  OK  {len(r['items'])} 条，首条: {first['标题']}")

    # 3) 抖音原始英文键 + 嵌套 statistics + 毫秒时长 + 播放量
    raw_payload = {
        "code": 200,
        "router": ENDPOINT_USER_POST_VIDEOS,
        "data": {"aweme_list": [
            {"desc": "英文键作品", "create_time": 1788000000, "video": {"duration": 45000},
             "statistics": {"digg_count": 1200, "comment_count": 35, "share_count": 12,
                            "collect_count": 180, "play_count": 8600}},
        ]},
    }
    r = parse_manual_json(json.dumps(raw_payload, ensure_ascii=False))
    assert r["error"] == "", r["error"]
    item = r["items"][0]
    assert item["标题"] == "英文键作品"
    assert (item["点赞"], item["评论"], item["分享"], item["收藏"]) == (1200, 35, 12, 180)
    assert item["时长"] == "45秒", item["时长"]
    assert item["播放"] == 8600, item
    assert item["发布时间"].startswith("20"), item["发布时间"]
    print(f"  parse_manual_json(原始响应) OK  含播放={item['播放']}，时间={item['发布时间']}")

    # 4) CSV / TSV（从表格复制）
    csv_text = (
        "标题,发布时间,时长,点赞,评论,分享,收藏\n"
        "AI工具提效实测,2026-09-01,45秒,1200,35,12,180\n"
        "三个Python技巧,2026-08-25,60秒,320,8,3,45\n"
    )
    r = parse_manual_json(csv_text)
    assert r["error"] == "", r["error"]
    assert len(r["items"]) == 2 and r["items"][1]["点赞"] == 320
    print(f"  parse_manual_json(CSV)   OK  {len(r['items'])} 条")

    # 5) 失败路径全部返回结构化错误，绝不抛异常
    for bad in ("", "   ", "这不是 JSON", "[]", "{}", "null", "标题,点赞\n"):
        r = parse_manual_json(bad)
        assert isinstance(r, dict) and r["items"] == [] and r["error"], bad
    print("  失败路径                 OK  空串 / 垃圾文本 / 空数组 都返回 error")

    r = fetch_user_videos("")
    assert r["items"] == [] and "主页链接" in r["error"]
    r = fetch_user_videos("https://example.com/没有任何 id")
    assert r["items"] == [] and r["error"], r
    print("  fetch_user_videos 参数校验 OK")

    # 6) 服务探活 + 真实调用：无论服务在不在，都必须不抛异常且返回约定结构
    avail = is_available()
    assert isinstance(avail, bool)
    print(f"  is_available()           OK  {avail}"
          f"{'' if avail else '（服务没起来，属预期 —— 走降级入口）'}")

    result = fetch_user_videos(
        "https://www.douyin.com/user/MS4wLjABAAAANXSltcLCzDGmdNFI2Q_QixVTr67NiYzjKOIP5s03CAE"
    )
    assert isinstance(result, dict) and set(result) == {"items", "error"}
    assert isinstance(result["items"], list) and isinstance(result["error"], str)
    if result["error"]:
        print(f"  采集实调用               返回可读错误: {result['error'].splitlines()[0]}")
    else:
        print(f"  采集实调用               OK  {len(result['items'])} 条作品")

    # 7) Cookie 提示文案
    hint = cookie_hint()
    assert isinstance(hint, str) and hint
    print(f"  cookie_hint()            OK  {hint.splitlines()[0][:40]}")

    print("\n自检完成")
