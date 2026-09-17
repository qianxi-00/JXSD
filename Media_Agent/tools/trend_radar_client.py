# -*- coding: utf-8 -*-
"""多平台热点抓取客户端 —— TrendRadar / NewsNow 聚合 API

课案出处：自媒体课案 → 热点监控 → TrendRadar 热点抓取客户端

课案原文参考：
    TrendRadar  GitHub: https://github.com/sansan0/TrendRadar
    NewsNow API: https://newsnow.busiyi.world/api/s

课案直接照抄可用的部分：平台 ID 映射、请求头、重试与抖动间隔、
``fetch_for_workflow()`` 的输出格式（工作流依赖 ``source/title/heat/url`` 四个字段）。

本项目的改动
    1. API 地址改从 ``settings.media.trendradar_api_url`` 读（课案硬编码在类里），
       方便换成自部署的 newsnow：``docker run -d -p 3000:3000 ourongxing/newsnow``
    2. 请求间隔用固定+随机抖动，避免被公共 API 限流（课案已有，保留）
    3. 失败一律返回空列表，不抛异常

⚠️ 关于「热度」字段
    NewsNow 公共 API 只返回标题与链接，**不返回真实热度值**。
    课案用 ``_estimate_heat(rank, total)`` 按排名反推一个 0~1000000 的估算值 ——
    这个数**不是真实热度**，只用来给前端排序和展示。本项目沿用课案做法，
    但在字段名与注释里写清楚它是估算值，避免被当成真实数据用进分析结论。
"""

import random
import sys
import time
from typing import Dict, List, Optional

import requests

from config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 平台 ID 映射 —— 来自 TrendRadar 的 config.yaml
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
    """

    def __init__(self, api_url: str = None, proxy_url: str = None):
        self.api_url = api_url or settings.media.trendradar_api_url
        self.proxy_url = proxy_url

    # ---------------- 单平台 ----------------
    def fetch_platform(self, platform_id: str, max_retries: int = 2) -> List[Dict]:
        """抓单个平台的热榜。

        Returns:
            ``[{"title","url","mobile_url","platform","platform_id","rank","heat"}, ...]``
            失败返回空列表。
        """
        url = f"{self.api_url}?id={platform_id}&latest"
        proxies = (
            {"http": self.proxy_url, "https": self.proxy_url} if self.proxy_url else None
        )
        platform_name = PLATFORM_IDS.get(platform_id, platform_id)

        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(
                    url, proxies=proxies, headers=DEFAULT_HEADERS, timeout=10
                )
                resp.raise_for_status()
                data = resp.json()

                status = data.get("status", "")
                # NewsNow 正常返回 success（最新）或 cache（缓存命中）
                if status not in ("success", "cache"):
                    raise ValueError(f"API 状态异常: {status}")

                items = data.get("items", []) or []
                topics = []
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
        """
        if platform_ids is None:
            platform_ids = ["douyin", "weibo", "zhihu",
                            "bilibili-hot-search", "toutiao", "baidu"]

        results: Dict[str, List[Dict]] = {}
        for i, pid in enumerate(platform_ids):
            topics = self.fetch_platform(pid)
            if topics:
                results[pid] = topics
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

        Returns:
            ``[{"source","title","heat","url","platform_id","rank"}, ...]``
        """
        if platform_ids is None:
            platform_ids = NAME_TO_IDS.get(platform, ["douyin", "weibo", "zhihu"])

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

        all_topics.sort(key=lambda x: x["heat"], reverse=True)
        return all_topics

    # ---------------- 辅助 ----------------
    @staticmethod
    def _estimate_heat(rank: int, total: int) -> int:
        """按排名**估算**热度（不是真实热度值，NewsNow 不提供）。

        第 1 名 ≈ 1000000，末位 ≈ 1000000/total。仅用于前端排序与展示。
        """
        base = 1_000_000
        ratio = 1.0 - (rank - 1) / max(total, 1)
        return int(base * ratio)

    @staticmethod
    def get_available_platforms() -> Dict[str, str]:
        """返回全部可用平台的 ``{ID: 中文名}``。"""
        return dict(PLATFORM_IDS)


# ---------------- 便捷函数（对应课案的同名函数）----------------
_default_client: Optional[TrendRadarClient] = None


def _get_client() -> TrendRadarClient:
    global _default_client
    if _default_client is None:
        _default_client = TrendRadarClient()
    return _default_client


def fetch_hot_topics(platform: str = "全部") -> List[Dict]:
    """便捷函数：抓热点并返回工作流格式。"""
    return _get_client().fetch_for_workflow(platform=platform)


def fetch_platform_hot(platform_id: str) -> List[Dict]:
    """便捷函数：抓单个平台。"""
    return _get_client().fetch_platform(platform_id)


if __name__ == "__main__":
    print("=== 热点抓取客户端自检 ===")

    # 1) 纯逻辑：热度估算单调递减
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
