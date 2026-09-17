# -*- coding: utf-8 -*-
"""多平台热点抓取客户端 —— TrendRadar / NewsNow 聚合 API

课案出处：自媒体课案 → 热点监控 → TrendRadar 热点抓取客户端

课案原文参考：
    TrendRadar  GitHub: https://github.com/sansan0/TrendRadar
    NewsNow API: https://newsnow.busiyi.world/api/s

课案直接照抄可用的部分：平台 ID 映射、请求头、重试与抖动间隔、
``fetch_for_workflow()`` 的输出格式（工作流依赖 ``source/title/heat/url`` 四个字段）。

与课案的落地差异
    | 项 | 课案 | 本项目 |
    |---|---|---|
    | API 地址 | 类里硬编码 ``DEFAULT_API_URL = "https://newsnow.busiyi.world/api/s"`` | ``settings.media.trendradar_api_url``（可指向自部署 newsnow） |
    | 平台表 | ``PLATFORM_IDS`` 24 项，含 ``neteasenews`` / ``github-trending`` | 同名表，但把这两个键订正为 ``netease-news`` / ``github-trending-today``，并补了失效键台账 |
    | 中文名→ID 表 | 写在 ``fetch_for_workflow()`` 里的局部变量 ``name_to_ids`` | 提到模块级常量 ``NAME_TO_IDS``，页面与工作流也能直接用 |
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
      TrendRadar 的 ``config.yaml`` 只共享了 15 个 id、与本表 24 项只交集 10 个。
    · **上游 id 会随时间失效**：2026-09-17 实测有 7 项已经死了（重试 2 次后静默返回
      ``[]``，白等 7.6s / 12.0s）。失效键**不删**，只登记在注释台账里 ——
      删键会牵动 ``NAME_TO_IDS`` 与页面的平台下拉，属行为变更。
    · **核上游 id 必须带浏览器 UA**：裸 ``GET .../api/s?id=<id>&latest`` 会回 403，
      那是 Cloudflare 拦默认 UA，**不是平台下线**；带上 ``DEFAULT_HEADERS`` 就正常。
    · **带请求间隔的批抓方法在页面上根本没被调用**：页面走的是
      ``fetch_platform_hot()`` → ``fetch_platform()``，中间没有 ``sleep``，
      所以「固定间隔 + 抖动防限流」只对 ``TrendRadarClient.fetch_hot_topics()``
      这个批量方法有效（它当前全仓无调用方）。平台一多仍可能撞限流。
    · 请求失败一律**返回空列表不抛异常**：调用点在 LangGraph 节点里，
      抛出去整条热点监控图就断了；代价是界面只显示 0 条，
      看不出是「真没热搜」还是「接口挂了」—— 要区分得看控制台日志：
      成功打「获取成功（最新/缓存，N 条）」，失败打「最终失败: <异常>」。
"""

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
# ⚠️ 已知在上游失效的键（**别删** —— 删了会连带影响 NAME_TO_IDS 与页面的平台下拉，
#    属行为变更）。2026-09-17 实测，下面这 7 项（8 个名字，netease-news 与
#    neteasenews 是同一平台的两种拼法 —— 后者是课案里的写法、本表里没有它，
#    列在这里是为了提醒「两种拼法在上游都死了」）**全部 HTTP 500**，且都不在
#    `shared/sources.json`（66 项）里：
#        netease-news / neteasenews / xiaohongshu / tencent-news / sogou /
#        guancha / acfun / csdn
#    失效表现：`fetch_platform(pid)` 会重试 2 次（每次失败后 sleep 2~5s），
#    实测白等 7.6s（网易新闻）/ 12.0s（小红书）后静默返回 `[]`，界面显示 0 条。
#    所以看到「某个平台总是 0 条」，先按上面的方法核 id，别去怀疑网络或限流。
#
# ⚠️ 连带影响：`NAME_TO_IDS` 里的「小红书」和「全部」都直接用了失效键
#    （`xiaohongshu`），所以**页面选「小红书」必然先白等约 12 秒再显示 0 条**；
#    选「全部」也会为它多等一次。这是已登记的行为，不是网络抖动。
PLATFORM_IDS = {
    # 主流平台
    "douyin": "抖音",
    "weibo": "微博",
    "zhihu": "知乎",
    "bilibili-hot-search": "B站热搜",
    "toutiao": "今日头条",
    "baidu": "百度热搜",
    # 更多平台
    "xiaohongshu": "小红书",
    "kuaishou": "快手",
    "wallstreetcn-hot": "华尔街见闻",
    "cls-hot": "财联社",
    "thepaper": "澎湃新闻",
    "tencent-news": "腾讯新闻",
    "netease-news": "网易新闻",
    "sogou": "搜狗热搜",
    "guancha": "观察者网",
    "ithome": "IT之家",
    "36kr": "36氪",
    "sspai": "少数派",
    "hupu": "虎扑",
    "tieba": "百度贴吧",
    "acfun": "AcFun",
    "juejin": "掘金",
    "csdn": "CSDN",
    "github-trending-today": "GitHub趋势",
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

# 中文平台名 → 平台 ID（Streamlit 页面直接传中文名）
# 课案把它写成 `fetch_for_workflow()` 里的局部变量 `name_to_ids`，这里提到模块级，
# 便于页面 / 工作流复用同一份映射（避免两处各写一份后漂移）。
# ⚠️ 「小红书」「全部」里含已失效的 `xiaohongshu`，选中就会白等约 12 秒后显示 0 条
#    （见上面 PLATFORM_IDS 的台账）；保留它是刻意行为，不是遗漏。
NAME_TO_IDS = {
    "抖音": ["douyin"],
    "小红书": ["xiaohongshu"],
    "微博": ["weibo"],
    "知乎": ["zhihu"],
    "B站": ["bilibili-hot-search"],
    "今日头条": ["toutiao"],
    "百度": ["baidu"],
    "全部": [
        "douyin", "weibo", "zhihu", "bilibili-hot-search",
        "toutiao", "baidu", "xiaohongshu",
    ],
}


class TrendRadarClient:
    """热点抓取客户端（对应课案里基于 TrendRadar DataFetcher 的那套）。

    用法::

        client = TrendRadarClient()
        topics = client.fetch_for_workflow("抖音")
        # [{"source": "抖音", "title": "...", "heat": 900000, "url": "..."}, ...]

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
            max_retries: 失败后的重试次数，默认 2 —— 即最多请求 3 次，
                每次失败等 2~5 秒随机时长（见下面的 ``random.uniform``）。

        Returns:
            ``[{"title","url","mobile_url","platform","platform_id","rank","heat"}, ...]``
            失败返回空列表（**不抛异常**，也不区分「平台死了」与「网络挂了」）。
        """
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
                # rank 从 1 开始（展示口径）；``len(items)`` 传的是**过滤前**的总数，
                # 与 rank 同口径，估算热度不会因为跳过个别空标题而整体偏高
                for i, item in enumerate(items, 1):
                    title = str(item.get("title") or "").strip()
                    if not title:
                        continue
                    topics.append({
                        "title": title,
                        "url": item.get("url", ""),
                        "mobile_url": item.get("mobileUrl", ""),
                        "platform": platform_name,
                        "platform_id": platform_id,
                        "rank": i,
                        "heat": self._estimate_heat(i, len(items)),
                    })

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
    def fetch_hot_topics(
        self, platform_ids: List[str] = None, request_interval: float = 0.3
    ) -> Dict[str, List[Dict]]:
        """批量抓多平台，返回 ``{平台ID: [热点], ...}``。

        带请求间隔 + 随机抖动，降低触发公共 API 限流的概率。

        ⚠️ **本方法当前全仓没有调用方**：页面走的是 ``fetch_platform_hot()`` →
        ``fetch_platform()``，那条路没有间隔。别以为「防限流已经做好了」。

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

    # ---------------- 适配工作流 ----------------
    def fetch_for_workflow(
        self, platform: str = "全部", platform_ids: List[str] = None
    ) -> List[Dict]:
        """抓取并转成工作流要的格式，按估算热度降序。

        Args:
            platform: 中文平台名（``NAME_TO_IDS`` 的键）；认不出时退化成
                抖音 + 微博 + 知乎三个平台。
            platform_ids: 直接指定平台 id 列表；给了就不看 ``platform``。

        Returns:
            ``[{"source","title","heat","url","platform_id","rank"}, ...]``
            —— 这是**多个平台混在一起的一维列表**，已按估算热度降序。
            抓不到任何数据时返回空列表（不抛异常）。
        """
        if platform_ids is None:
            platform_ids = NAME_TO_IDS.get(platform, ["douyin", "weibo", "zhihu"])

        # ⚠️ 这里对每个平台是**连续请求、没有 sleep**（课案也如此）。间隔只存在于
        #    上面那个批量方法里，而页面不经过它 —— 平台一多确有撞限流的风险。
        all_topics: List[Dict] = []
        for pid in platform_ids:
            for t in self.fetch_platform(pid):
                all_topics.append({
                    "source": t["platform"],
                    "title": t["title"],
                    "heat": t["heat"],
                    "url": t["url"],
                    "platform_id": t["platform_id"],
                    "rank": t["rank"],
                })

        # 排序放在这一层（而不是 fetch_platform）：合并多平台后才谈得上「总热度榜」。
        # 注意 heat 是估算值，所以这其实是「按名次加权后的大致排序」。
        all_topics.sort(key=lambda x: x["heat"], reverse=True)
        return all_topics

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
        """返回全部可用平台的 ``{ID: 中文名}``。

        Returns:
            Dict[str, str]: ``PLATFORM_IDS`` 的**浅拷贝** —— 调用方改了不会污染模块级常量
            （页面里就靠它填平台下拉）。注意里面含已失效的 id，见 ``PLATFORM_IDS`` 台账。
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


def fetch_hot_topics(platform: str = "全部") -> List[Dict]:
    """便捷函数：抓热点并返回工作流格式。

    ⚠️ 与**同名的方法** ``TrendRadarClient.fetch_hot_topics()`` 不是一回事：
    那个返回 ``{平台ID: [热点]}`` 的**字典**（多平台批量、带请求间隔），
    这个返回按热度降序的**一维列表**（走 ``fetch_for_workflow()``、无间隔）。
    看名字容易混，按返回类型区分。

    Args:
        platform: 中文平台名（``NAME_TO_IDS`` 的键，如 ``"抖音"`` / ``"全部"``）。

    Returns:
        List[Dict]: ``[{"source","title","heat","url","platform_id","rank"}, ...]``；
        抓不到就是空列表（不抛异常）。
    """
    return _get_client().fetch_for_workflow(platform=platform)


def fetch_platform_hot(platform_id: str) -> List[Dict]:
    """便捷函数：抓单个平台。

    这是页面 / 工作流**实际走的那条路**（``workflows/hot_topic.py`` 里 import 它，
    按平台并行抓取后交给 LLM 筛选）。

    Args:
        platform_id: NewsNow 平台 id（如 ``"douyin"``），不是中文名。

    Returns:
        List[Dict]: 该平台热点列表，字段见 ``TrendRadarClient.fetch_platform()``；
        平台失效或网络失败时返回**空列表**（重试 2 次后放弃，不抛异常）。
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

    # 2) 平台映射表可用
    assert PLATFORM_IDS["douyin"] == "抖音"
    assert NAME_TO_IDS["全部"][0] == "douyin"
    assert TrendRadarClient.get_available_platforms()
    print(f"  平台映射表                OK  {len(PLATFORM_IDS)} 个平台")

    # 3) 真实网络抓取（可用 --net 触发；默认跳过，保证离线也能全绿）
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
