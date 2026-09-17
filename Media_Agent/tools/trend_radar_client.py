# -*- coding: utf-8 -*-
"""多平台热点抓取客户端 —— TrendRadar / NewsNow 聚合 API

课案出处：自媒体课案 → 热点监控 → TrendRadar 热点抓取客户端

课案原文参考：
    TrendRadar  GitHub: https://github.com/sansan0/TrendRadar
    NewsNow API: https://newsnow.busiyi.world/api/s

课案直接照抄可用的部分：平台 ID 映射、请求头、重试与抖动间隔、
工作流要的字段格式（``source/title/heat/url``）。

⚠️ 课案还有一个 ``fetch_for_workflow()``（把中文名翻成 id、逐平台抓、合并后按热度
降序）。**本项目没有实现它**：生产路径是 ``fetch_platform_hot()`` → ``fetch_platform()``
（由 ``workflows/hot_topic.py`` 的并行节点逐平台调用），而「合并 + 按热度降序 + 去重」
落在那个工作流的 ``node_filter`` 里。两者是同一件事的两份实现，留着迟早各改一半 ——
所以只保留生产在走的那份（同 ``main.py`` 里删掉 ``add_to_history()`` 的理由）。

与课案的落地差异
    | 项 | 课案 | 本项目 |
    |---|---|---|
    | API 地址 | 类里硬编码 ``DEFAULT_API_URL = "https://newsnow.busiyi.world/api/s"`` | ``settings.media.trendradar_api_url``（可指向自部署 newsnow） |
    | 平台表 | ``PLATFORM_IDS`` 24 项，含 ``neteasenews`` / ``github-trending`` | 拆成两张：``PLATFORM_IDS`` **只含当前可用的 17 项**（把课案那两个键订正为 ``netease-news`` / ``github-trending-today``），失效的 7 项移到 ``RETIRED_PLATFORM_IDS`` |
    | 中文名→ID 表 | 写在 ``fetch_for_workflow()`` 里的局部变量 ``name_to_ids`` | 提到模块级常量 ``NAME_TO_IDS``（**只被工作流用来判「这个平台的 id 是不是空的」**，页面与工作流各有一份自己的平台清单，见 ``NAME_TO_IDS`` 上方注释） |
    | 适配工作流 | ``fetch_for_workflow()``（抓 + 合并 + 排序） | **不实现**——见上面那段理由；合并排序在 ``workflows/hot_topic.py`` 的 ``node_filter`` |
    | 日志前缀 | ``[TrendRadar]`` | ``[热点]``（与其它 tools 模块的 ``[模块名]`` 风格统一） |
    | 失败处理 | ``fetch_platform()`` 重试后返回空列表 | 同课案，未改；但**下面「踩过的坑」写明页面路径实际不经过带间隔的批抓** |
    | 默认客户端 | ``_default_client = None`` + ``_get_client()`` 懒加载单例 | 同课案，未改 |
    | 热度字段 | ``_estimate_heat()`` 估算并直接叫 ``heat`` | 沿用同一算法，但字段与页面标注都写明是**估算值** |

    换成自部署的 newsnow：改 ``settings.media.trendradar_api_url``（或直接
    ``TrendRadarClient(api_url=...)``），例如 ``docker run -d -p 3000:3000 ourongxing/newsnow``。

⚠️ 关于「热度」字段
    NewsNow 公共 API 只返回标题与链接，**不返回真实热度值**。
    课案用 ``_estimate_heat(rank, total)`` 按排名反推一个 0~1000000 的估算值 ——
    这个数**不是真实热度**，只用来给前端排序和展示。本项目沿用课案做法，
    但在字段名与注释里写清楚它是估算值，避免被当成真实数据用进分析结论。

踩过的坑
    · **平台 id 的真实来源是 NewsNow，不是 TrendRadar**（课案原注释写错了，
      详见下面 ``PLATFORM_IDS`` 上方的台账）：上游 ``shared/sources.json`` 才是权威，
      TrendRadar 的 ``config.yaml`` 只共享了 15 个 id、与课案那 24 项只交集 10 个。
    · **上游 id 会随时间失效**：2026-09-17 实测有 7 项已经死了（重试 2 次后静默返回
      ``[]``，白等 7.6s / 12.0s）。处置分三层：
      ① ``RETIRED_PLATFORM_IDS`` 收着这 7 项（**保留而不是删**，删了就看不出
      「这个平台曾经接过」，也是日后回访的清单）；② ``PLATFORM_IDS`` 从此**只含
      可用的**，遍历它的新代码不会再把死平台捎进去；③ ``NAME_TO_IDS`` 里
      「小红书」的 id 列表置空，工作流那个抓取节点据此**一个请求都不发**就跳过，
      原因登记在 ``UNAVAILABLE_PLATFORMS``（页面据此告诉用户「为什么是 0 条」）——
      置空是因为选中它会必然白等约 12 秒再显示 0 条，那不是网络抖动。
      ⚠️ 但 ``NAME_TO_IDS`` **既不是页面下拉的来源，也不是工作流取数的来源**：页面用
      ``views/hot_topic.py`` 自己手写的 ``PLATFORM_OPTIONS``，工作流用
      ``workflows/hot_topic.py`` 自己那份 ``FETCH_SOURCES``。生产路径上读本表的只有
      ``workflows/hot_topic.py`` 一处，判据是「这个平台名的 id 列表是不是空的」。
    · **核上游 id 必须带浏览器 UA**：裸 ``GET .../api/s?id=<id>&latest`` 会回 403，
      那是 Cloudflare 拦默认 UA，**不是平台下线**；带上 ``DEFAULT_HEADERS`` 就正常。
    · **带请求间隔的批抓方法在页面上根本没被调用**：页面走的是
      ``fetch_platform_hot()`` → ``fetch_platform()``，中间没有 ``sleep``，
      所以「固定间隔 + 抖动防限流」只对 ``TrendRadarClient.fetch_hot_topics_by_platform()``
      这个批量方法有效（它当前全仓无调用方）。平台一多仍可能撞限流。
    · 请求失败一律**返回空列表不抛异常**：调用点在 LangGraph 节点里，
      抛出去整条热点监控图就断了；代价是界面只显示 0 条，
      看不出是「真没热搜」还是「接口挂了」—— 要区分得看控制台日志：
      成功打「获取成功（最新/缓存，N 条）」，失败打「最终失败: <异常>」。
"""

import contextlib
import io
import random
import sys
import time
from typing import Dict, List, Optional

import requests

from config import settings

# Windows 控制台默认按 GBK 输出，而本模块日志里有全角括号与中文平台名；
# 不改流编码会在个别终端下 UnicodeEncodeError 把自检打挂（tools/ 下统一这么做）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 平台 ID 映射 —— 真实来源是 **NewsNow 的 source 列表**（`newsnext/newsnow` 的
# `shared/sources.json` 里的键），**不是** TrendRadar 的 `config.yaml`。
#
# ⚠️ 原注释「来自 TrendRadar 的 config.yaml」实测不成立（那是**课案**里的注释，
#    课案:287 原话如此）：
#    TrendRadar `config/config.yaml` 只有 15 个 id，与本表（24 项）交集仅 10 个，
#    表里 14 项它根本没有。本模块打的 API 就是 NewsNow 的 `/api/s`。
#    另：课案里这两项拼法与本仓库不同，本项目已经订正过 ——
#        `neteasenews` → `netease-news`、`github-trending` → `github-trending-today`。
#
# ⚠️ 上游 id 会变，本表只是 **2026-09-17 的一次快照**。核对某个 id 现在还有效吗：
#    · 首选 ``fetch_platform_hot("<id>")`` 看返回条数（0 条基本就是死了）；
#    · 或直接 GET ``https://newsnow.busiyi.world/api/s?id=<id>&latest``
#      —— 注意**必须带 DEFAULT_HEADERS 里的浏览器 UA**：不带 UA 时 Cloudflare 对
#      python-requests 的默认 UA 一律回 403，看起来像「所有平台全挂了」。
#
# ⚠️ 已知在上游失效的键（**已拆到下面的 `RETIRED_PLATFORM_IDS`**，不是删掉了 ——
#    删了就再也看不出「这个平台曾经接过」）。2026-09-17 实测这 7 项**全部 HTTP 500**，
#    且都不在 `shared/sources.json`（66 项）里。（`netease-news` 在课案里还写作
#    `neteasenews`，两种拼法在上游都死了；本表只留前者，别名记在这条注释里。）
#    失效表现：`fetch_platform(pid)` 会重试 2 次（每次失败后 sleep 2~5s），
#    实测白等 7.6s（网易新闻）/ 12.0s（小红书）后静默返回 `[]`，界面显示 0 条。
#    所以看到「某个平台总是 0 条」，先按上面的方法核 id，别去怀疑网络或限流。
#
# ⚠️ 拆开的意义：`PLATFORM_IDS` 从此**只含当前可用的**，任何「遍历它」的新代码
#    （填下拉、做探活、生成报表）默认就不会把死平台捎进去。死键的连带影响上一轮
#    已经收口过一次：`NAME_TO_IDS` 里「小红书」的 id 列表已置空、`"全部"` 也已去掉
#    `xiaohongshu`，所以那两条路不会再为死键白等约 12 秒。
PLATFORM_IDS = {
    # 主流平台
    "douyin": "抖音",
    "weibo": "微博",
    "zhihu": "知乎",
    "bilibili-hot-search": "B站热搜",
    "toutiao": "今日头条",
    "baidu": "百度热搜",
    # 更多平台
    "kuaishou": "快手",
    "wallstreetcn-hot": "华尔街见闻",
    "cls-hot": "财联社",
    "thepaper": "澎湃新闻",
    "ithome": "IT之家",
    "36kr": "36氪",
    "sspai": "少数派",
    "hupu": "虎扑",
    "tieba": "百度贴吧",
    "juejin": "掘金",
    "github-trending-today": "GitHub趋势",
}

# 曾经接过、但 2026-09-17 实测已在上游失效的 id。
# **保留而不是删除**：这是「我们接过这个平台」的历史登记，也是回访清单 ——
# 哪天想确认某个源是否复活，直接 `fetch_platform_hot("<id>")` 看返回条数即可。
# ⚠️ 不要拿这个表去填任何面向用户的可选列表。
RETIRED_PLATFORM_IDS = {
    "xiaohongshu": "小红书",
    "tencent-news": "腾讯新闻",
    "netease-news": "网易新闻",
    "sogou": "搜狗热搜",
    "guancha": "观察者网",
    "acfun": "AcFun",
    "csdn": "CSDN",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 中文平台名 → 要抓的平台 ID。
# 课案把它写成 `fetch_for_workflow()` 里的局部变量 `name_to_ids`，这里提到模块级。
# ⚠️ **生产路径上只有一处读它**：`workflows/hot_topic.py` 用来判「这个平台名的 id
#    列表是不是空的」（空 = 当前抓不到，节点直接跳过、一个请求都不发）。
#    页面下拉用的是 `views/hot_topic.py` 自己手写的 `PLATFORM_OPTIONS`，工作流取数
#    用的是 `workflows/hot_topic.py` 自己那份 `FETCH_SOURCES` —— 这两层清单的一致性
#    由 `verify_all.py` 第 3 层的跨层契约钉住（页面 `PLATFORM_OPTIONS` 去掉「全部」
#    必须与工作流 `PLATFORM_SENDS` 键集相等）。
# ⚠️ 键集必须与 `UNAVAILABLE_PLATFORMS` 对得上：每个可选项要么有 id（能抓），
#    要么有不可用原因（抓不了，但用户能看到为什么）。自检里**双向**都钉住。
# ⚠️ 「小红书」的 id 列表**故意留空**（上游 `xiaohongshu` 已失效并已移入
#    `RETIRED_PLATFORM_IDS`）：空列表 = 工作流那条路一个请求都不发，不再白等约 12 秒；
#    原因由 `UNAVAILABLE_PLATFORMS` 提供（页面据此告诉用户「为什么是 0 条」）。
NAME_TO_IDS = {
    "抖音": ["douyin"],
    "小红书": [],
    "微博": ["weibo"],
    "知乎": ["zhihu"],
    "B站": ["bilibili-hot-search"],
    "今日头条": ["toutiao"],
    "百度": ["baidu"],
    "全部": [
        "douyin", "weibo", "zhihu", "bilibili-hot-search",
        "toutiao", "baidu",
    ],
}

# 有平台名、但当前拿不到数据的平台：中文名 → 中文不可用原因。
#
# 为什么单独一张表、而不是写在 `PLATFORM_IDS` 或 `NAME_TO_IDS` 里：
#   `PLATFORM_IDS` 是 **id 登记表**（含死键，保留是为了能回溯「曾经接过哪些源」），
#   `NAME_TO_IDS` 只管「抓哪些 id」，两者都装不下「给用户看的原因」。
# 页面（`views/hot_topic.py`）直接 import 它：下拉框里给这些平台加说明，
# 用户选中时把原因显示出来 —— 「0 条」必须能解释成「上游这个源死了」，
# 而不是让用户怀疑「今天没热搜」或「我的网络有问题」。
UNAVAILABLE_PLATFORMS = {
    "小红书": (
        "上游聚合源 `xiaohongshu` 已失效（HTTP 500，且已不在 NewsNow 的 "
        "shared/sources.json 里），当前拿不到该平台数据；"
        "本项保留只为告知原因，不会再发起请求。"
    ),
}


class TrendRadarClient:
    """热点抓取客户端（对应课案里基于 TrendRadar DataFetcher 的那套）。

    用法::

        client = TrendRadarClient()
        topics = client.fetch_platform("douyin")
        # [{"title": "...", "heat": 900000, "url": "...", "rank": 1, ...}, ...]
        #
        # 想「多平台合并 + 按热度降序」的话，那是 workflows/hot_topic.py 的
        # `node_filter` 干的活（并行抓取 → reducer 合并 → 排序去重），
        # 本类里没有对应的合并方法 —— 课案那个 fetch_for_workflow() 已被有意去掉。

    这里的方法都**不抛异常**：抓不到就是空列表 / 空 dict，
    由调用方（LangGraph 节点）按「本轮没热点」处理。
    """

    def __init__(self, api_url: str = None, proxy_url: str = None):
        """构造客户端。

        Args:
            api_url: NewsNow API 地址；不传时取 ``settings.media.trendradar_api_url``
                （默认 ``https://newsnow.busiyi.world/api/s``，可换成自部署的 newsnow）。
            proxy_url: 可选 HTTP(S) 代理地址；传了才会出现在请求里，不传就是直连。
        """
        self.api_url = api_url or settings.media.trendradar_api_url
        self.proxy_url = proxy_url

    # ---------------- 单平台 ----------------
    def fetch_platform(self, platform_id: str, max_retries: int = 2) -> List[Dict]:
        """抓单个平台的热榜。

        Args:
            platform_id: NewsNow 的平台 id（如 ``"douyin"``）；认不出时日志里直接原样打印。
                为空串时**直接返回空列表、一个请求都不发**。
            max_retries: 失败后的重试次数，默认 2 —— 即最多请求 3 次，
                每次失败等 2~5 秒随机时长（见下面的 ``random.uniform``）。

        Returns:
            ``[{"title","url","mobile_url","platform","platform_id","rank","heat"}, ...]``
            失败返回空列表（**不抛异常**，也不区分「平台死了」与「网络挂了」）。
        """
        # 空 id 守卫：`NAME_TO_IDS` 里失效平台的 id 列表是空的，遍历它压根不会进这个
        # 函数；但显式传空串（页面 / 工作流拿到空值）也得走同一条路 —— 直接返回，
        # 别去发一次 `?id=&latest` 的无效请求（那会白等 3 次重试 ≈ 12 秒才拿到 0 条）。
        if not platform_id:
            print("[热点] 平台 id 为空，跳过抓取")
            return []

        url = f"{self.api_url}?id={platform_id}&latest"
        proxies = (
            {"http": self.proxy_url, "https": self.proxy_url} if self.proxy_url else None
        )
        platform_name = PLATFORM_IDS.get(platform_id, platform_id)

        for attempt in range(max_retries + 1):
            try:
                # 10 秒超时：公共 API 正常在 1 秒内返回，超过 10 秒基本就是它挂了；
                # 再长会让「批量抓多平台」把页面卡到超时
                resp = requests.get(
                    url, proxies=proxies, headers=DEFAULT_HEADERS, timeout=10
                )
                resp.raise_for_status()
                data = resp.json()

                status = data.get("status", "")
                # NewsNow 正常返回 success（最新）或 cache（缓存命中）
                if status not in ("success", "cache"):
                    # 故意用 raise 而不是 return []：这样能落进下面的统一 except，
                    # 一样享受「重试 + 抖动等待」，不用为状态异常单独写一套退避
                    raise ValueError(f"API 状态异常: {status}")

                items = data.get("items", []) or []
                topics = []
                # rank 从 1 开始且**在过滤之后**连续编号（展示口径）：否则空标题被
                # `continue` 跳过后名次会缺口（1,2,4…），页面上看着像丢了数据。
                rank = 0
                for item in items:
                    title = str(item.get("title") or "").strip()
                    if not title:
                        continue
                    rank += 1
                    topics.append({
                        "title": title,
                        "url": item.get("url", ""),
                        "mobile_url": item.get("mobileUrl", ""),
                        "platform": platform_name,
                        "platform_id": platform_id,
                        "rank": rank,
                    })

                # heat 的 `total` 必须与 rank **同源**，所以这里传的是**过滤后**的
                # `len(topics)`，不是上游给回来的 `len(items)` —— rank 已经按过滤后连续
                # 编号，total 若仍按过滤前算，存活条目的 heat 会被整体抬高。
                # 这不只是展示口径：heat 是**跨平台合并排序键**（`workflows/hot_topic.py`
                # 的 `node_filter` 按它降序排），所以某个平台一旦出现空标题条目，
                # 它的存活条目就会在合并排序里往前跳。
                # 代价如实写明：有空标题条目的平台，其存活条目热度会**略升**；
                # 这是让 rank 与 total 同源的必要代价（不是「口径自洽」）。
                for t in topics:
                    t["heat"] = self._estimate_heat(t["rank"], len(topics))

                status_info = "最新" if status == "success" else "缓存"
                print(f"[热点] {platform_name} 获取成功（{status_info}，{len(topics)} 条）")
                return topics

            except Exception as exc:  # noqa: BLE001 —— 热榜接口偶发失败是常态
                if attempt < max_retries:
                    # 2~5 秒随机等待：固定间隔容易和别的客户端撞在同一拍上被限流
                    wait = random.uniform(2, 5)
                    print(f"[热点] {platform_name} 失败，{wait:.1f}s 后重试: {exc}")
                    time.sleep(wait)
                else:
                    print(f"[热点] {platform_name} 最终失败: {exc}")
                    return []
        return []

    # ---------------- 多平台 ----------------
    def fetch_hot_topics_by_platform(
        self, platform_ids: List[str] = None, request_interval: float = 0.3
    ) -> Dict[str, List[Dict]]:
        """批量抓多平台，返回 ``{平台ID: [热点], ...}``。

        名字里的 ``by_platform`` 指的是**返回值按平台 id 分桶**：早先它和模块级那个
        便捷函数同名（后者已删）却一个返回 dict、一个返回一维 list，极易误用，
        所以给类方法改成了这个能看出返回形状的名字。

        「固定间隔 + 随机抖动」**只存在于本方法**（见下面的 ``time.sleep``）。
        ⚠️ **本方法当前全仓没有调用方**：页面 / 工作流走的是 ``fetch_platform_hot()``
        → ``fetch_platform()``，那条路中间没有 sleep。别以为「防限流已经做好了」。

        Args:
            platform_ids: 要抓的平台 id 列表；不传时抓主流 6 个
                （抖音/微博/知乎/B站/头条/百度）。
            request_interval: 基础请求间隔（秒），实际等待是
                ``request_interval + uniform(-0.1, 0.2)`` 且下限 0.1 秒。

        Returns:
            Dict[str, List[Dict]]: ``{平台ID: 热点列表}``；**抓不到的平台不会出现在
            结果里**（而不是给个空列表），所以调用方要靠 ``len(results)`` 判断成功几个。
        """
        if platform_ids is None:
            platform_ids = ["douyin", "weibo", "zhihu",
                            "bilibili-hot-search", "toutiao", "baidu"]

        results: Dict[str, List[Dict]] = {}
        for i, pid in enumerate(platform_ids):
            topics = self.fetch_platform(pid)
            if topics:
                results[pid] = topics
            # 最后一个平台后面不用等（省掉一次无意义的 sleep）；
            # max(0.1, ...) 是防抖下限：抖动是 ±0.1/±0.2，不减到 0 或负数
            if i < len(platform_ids) - 1:
                time.sleep(max(0.1, request_interval + random.uniform(-0.1, 0.2)))

        total = sum(len(v) for v in results.values())
        print(f"[热点] 批量抓取完成: {len(results)}/{len(platform_ids)} 平台，共 {total} 条")
        return results

    # ---------------- 辅助 ----------------
    @staticmethod
    def _estimate_heat(rank: int, total: int) -> int:
        """按排名**估算**热度（不是真实热度值，NewsNow 不提供）。

        第 1 名 ≈ 1000000，末位 ≈ 1000000/total。仅用于前端排序与展示。

        Args:
            rank: 名次，从 1 开始。
            total: 该平台返回的总条数。

        Returns:
            int: 名次越高越大，``rank=1`` 恒为 ``1000000``。
            ⚠️ 末位不是整 ``1000000/total``：``ratio`` 是浮点减出来的，
            自检里 ``_estimate_heat(10, 10)`` 实际得 ``99999`` 而非 ``100000`` ——
            只影响展示末位精度，不影响排序（同一 total 下仍单调递减）。
            ``total<=0`` 时按 1 处理（``max(total, 1)`` 防除零）。
        """
        base = 1_000_000
        ratio = 1.0 - (rank - 1) / max(total, 1)
        return int(base * ratio)

    @staticmethod
    def get_available_platforms() -> Dict[str, str]:
        """返回**当前可用**平台的 ``{ID: 中文名}``。

        早先这里写「页面里就靠它填平台下拉」，是假的 —— 页面用的是
        ``views/hot_topic.py`` 自己手写的 ``PLATFORM_OPTIONS``（那份只有 5 个平台，
        因为工作流的 ``FETCH_SOURCES`` 只注册了 5 个抓取节点；两者的一致性由
        ``verify_all.py`` 第 3 层的跨层契约钉住）。本方法目前**没有生产调用方**，
        留着是因为它是「可用平台」的单一出口。

        已失效的 7 个 id 拆到了 ``RETIRED_PLATFORM_IDS``，所以这里返回的**都是活的** ——
        拿它填下拉不会再出现「选中一个必然 0 条的选项」那种情况。

        Returns:
            Dict[str, str]: ``PLATFORM_IDS`` 的**浅拷贝** —— 调用方改了不会污染模块级常量。
        """
        return dict(PLATFORM_IDS)


# ---------------- 便捷函数（对应课案的同名函数）----------------
# 懒加载单例：第一次真的要用时才建客户端。这样 import 本模块不会因为读 settings
# 或建 requests 会话而产生副作用（页面 import 它时还在很前面）。
_default_client: Optional[TrendRadarClient] = None


def _get_client() -> TrendRadarClient:
    """取模块级默认客户端（懒加载单例，第一次调用时才实例化）。

    Returns:
        TrendRadarClient: 全模块共用的那一个实例；重复调用返回同一对象
        （所以自定义 ``api_url`` 请自己 ``TrendRadarClient(...)``，别指望改这个）。
    """
    global _default_client
    if _default_client is None:
        _default_client = TrendRadarClient()
    return _default_client


def fetch_platform_hot(platform_id: str) -> List[Dict]:
    """便捷函数：抓单个平台。

    这是页面 / 工作流**实际走的那条路**（``workflows/hot_topic.py`` 里 import 它，
    按平台并行抓取后交给 LLM 筛选）。

    Args:
        platform_id: NewsNow 平台 id（如 ``"douyin"``），不是中文名。

    Returns:
        List[Dict]: 该平台热点列表，字段见 ``TrendRadarClient.fetch_platform()``；
        平台失效或网络失败时返回**空列表**（重试 2 次后放弃，不抛异常）；
        ``platform_id`` 为空串时**直接返回空列表、一个请求都不发**。
    """
    return _get_client().fetch_platform(platform_id)


if __name__ == "__main__":
    # 自检挂在 __main__ 下：verify_all.py 的「第 1 层：模块自检」会逐个
    # ``python <模块>.py`` 跑这一段。默认**不联网**（联网那段要显式加 --net），
    # 保证没网 / 上游挂了时自检依然 exit 0。
    print("=== 热点抓取客户端自检 ===")

    # 1) 纯逻辑：热度估算单调递减
    #    断言「不抛异常 + 严格递减 + 封顶 1000000」，是这个模块唯一不碰网络的可测部分
    heats = [TrendRadarClient._estimate_heat(i, 10) for i in range(1, 11)]
    assert heats == sorted(heats, reverse=True), heats
    assert heats[0] == 1_000_000, heats[0]
    assert all(0 < h <= 1_000_000 for h in heats)
    print(f"  _estimate_heat            OK  {heats[:3]} ... {heats[-1]}")

    # 2) 平台映射表可用；可用/失效两张表必须**互不重叠、且合起来是完整台账**
    assert PLATFORM_IDS["douyin"] == "抖音"
    assert NAME_TO_IDS["全部"][0] == "douyin"
    assert TrendRadarClient.get_available_platforms()
    _overlap = set(PLATFORM_IDS) & set(RETIRED_PLATFORM_IDS)
    assert not _overlap, f"同一个 id 不该同时出现在可用与失效两张表里: {sorted(_overlap)}"
    # 失效表里那 7 项就是台账写的那 7 个 —— 少一个说明有人把死键偷偷放回可用表，
    # 多一个说明加了新的失效项却没更新台账注释
    assert set(RETIRED_PLATFORM_IDS) == {
        "xiaohongshu", "tencent-news", "netease-news",
        "sogou", "guancha", "acfun", "csdn",
    }, sorted(RETIRED_PLATFORM_IDS)
    # 「可用」表里不允许再混进已知失效的 id（上一轮拆表就是为了让这个恒成立）
    assert "xiaohongshu" not in PLATFORM_IDS, "失效 id 不该留在可用表里"
    assert TrendRadarClient.get_available_platforms() == PLATFORM_IDS
    print(f"  平台映射表                OK  可用 {len(PLATFORM_IDS)} 项 / "
          f"已退休 {len(RETIRED_PLATFORM_IDS)} 项（两表不重叠）")

    # 3) 失效平台的处置（B1）：小红书已从 NAME_TO_IDS 摘掉、原因在新常量里。
    #    ⚠️ 「选中它一个请求都不发」这条断言**不在本模块**了：那条路
    #    （课案的 `fetch_for_workflow()`）已删，改由工作流那个抓取节点的守卫负责，
    #    在 `workflows/hot_topic.py` 的自检 3g 里钉住。下面只剩**数据层**的不变量。
    assert NAME_TO_IDS["小红书"] == [], NAME_TO_IDS["小红书"]
    assert "xiaohongshu" not in NAME_TO_IDS["全部"], NAME_TO_IDS["全部"]
    # 键集对齐，**双向**都要钉（页面下拉与工作流判据都依赖它）：
    #   ① 每个不可用原因都必须挂在某个可选项上，否则页面永远显示不出这句话；
    #   ② 反过来，「有平台名、但一个 id 都没有」的项也必须登记了原因 —— 这正是上一轮
    #      要消灭的「静默 0 条」形态：往 `NAME_TO_IDS` 插一行 `"快手": []` 而不登记原因，
    #      用户选中它只会拿到 0 条，且没有任何解释。
    assert set(UNAVAILABLE_PLATFORMS) <= set(NAME_TO_IDS), UNAVAILABLE_PLATFORMS
    _no_ids = {n for n, ids in NAME_TO_IDS.items() if not ids}
    assert _no_ids <= set(UNAVAILABLE_PLATFORMS), (
        "这些平台没有可抓的 id，却没在 UNAVAILABLE_PLATFORMS 里登记原因: "
        f"{sorted(_no_ids - set(UNAVAILABLE_PLATFORMS))}"
    )
    print(f"  失效平台已登记+摘键        OK  {list(UNAVAILABLE_PLATFORMS)}")

    # 3b) rank 连续性（D11）：上游 items 里夹一个空标题时，名次必须按**过滤后**的
    #     输出顺序重排（1,2,3 而不是 1,2,4），且 heat 仍随 rank 单调递减。
    #     用打桩顶掉 `requests.get`，本段依然离线可跑；finally 里务必还原，
    #     否则后面的 --net 分支会拿着桩跑（看起来通过、其实什么都没调）。
    _real_get = requests.get
    try:
        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"status": "success", "items": [
                    {"title": "第一条", "url": "u1"},
                    {"title": "   ", "url": "u2"},      # 空标题：会被 continue 跳掉
                    {"title": "第三条", "url": "u3"},
                    {"title": "第四条", "url": "u4"},
                ]}

        requests.get = lambda *a, **k: _FakeResp()
        stubbed = TrendRadarClient().fetch_platform("douyin")
    finally:
        requests.get = _real_get
    ranks = [t["rank"] for t in stubbed]
    assert ranks == [1, 2, 3], f"名次应在过滤后连续，实际 {ranks}"
    assert all(a["heat"] > b["heat"] for a, b in zip(stubbed, stubbed[1:])), stubbed
    # heat 的 total 必须与 rank **同源**（都用过滤后的 3 条，而不是上游给回来的 4 条）：
    # 传 `len(items)` 时第 2 名的 heat 会是 750000 而不是 666666 —— 而 heat 是跨平台
    # 合并排序键，某个平台一有空标题条目，它的存活条目就会在合并排序里往前跳。
    assert stubbed[1]["heat"] == TrendRadarClient._estimate_heat(2, 3), stubbed
    print(f"  rank 过滤后连续           OK  {ranks}，heat 按过滤后 3 条算")

    # 3c) 空 id 这条路必须**一个请求都不发**，并且要把原因写进日志。
    #     用一个「访问即失败」的 `requests` 替身真的钉住：这条路上只要有人碰
    #     `requests`（哪怕只是读 `.get`），替身就把访问记进 `_touched` 并抛异常 ——
    #     断言 `_touched == []` 即为「零请求」。
    #     为什么记下来再断言、而不是只靠抛异常：`fetch_platform()` 里有
    #     `except Exception`（网络失败要重试），AssertionError 同样会被它吞掉并重试，
    #     光靠抛异常这条断言会「静默通过」。
    #     （「按中文平台名选到失效平台也不发请求」这条不在本模块测了：承载它的
    #      课案函数 `fetch_for_workflow()` 已删，改由工作流抓取节点的守卫负责，
    #       见 `workflows/hot_topic.py` 自检 3g。）
    class _ExplodingRequests:
        def __getattr__(self, name):
            _touched.append(name)
            raise AssertionError(f"这条路径不该访问 requests.{name}")

    _real_requests = requests
    _touched: List[str] = []
    _log = io.StringIO()
    try:
        requests = _ExplodingRequests()
        with contextlib.redirect_stdout(_log):
            _blank = fetch_platform_hot("")         # 命中的是 fetch_platform 的空 id 守卫
    finally:
        requests = _real_requests

    assert _blank == [], _blank
    assert _touched == [], f"这条路径不该访问 requests，实际访问了 {_touched}"
    # 顺带钉住「守卫要留下原因」：空 id 守卫存在的意义就是让日志能解释「为什么 0 条」。
    # 删掉守卫、只靠 for 循环「自然不执行」时，上面两条仍会通过，这条会红。
    assert "跳过抓取" in _log.getvalue(), _log.getvalue()
    print("  空 id 零请求               OK  「访问即失败」替身未被碰，且日志留下原因")

    # 4) 真实网络抓取（可用 --net 触发；默认跳过，保证离线也能全绿）
    #    ⚠️ 不要用这个自检去核「某个平台 id 还有效吗」—— 微信/微博这类活着的平台
    #    也可能因为偶发限流返回 0 条。核 id 的正确姿势见 PLATFORM_IDS 上方的台账。
    if "--net" in sys.argv:
        client = TrendRadarClient()
        topics = client.fetch_platform("weibo")
        if topics:
            print(f"  联网抓取微博热榜          OK  {len(topics)} 条，"
                  f"第 1 条: {topics[0]['title'][:30]}")
        else:
            print("  联网抓取微博热榜          失败（网络或 API 不可达）")
    else:
        print("  联网抓取                  跳过（加 --net 才跑）")

    print("\n自检完成")
