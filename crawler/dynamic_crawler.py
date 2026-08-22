"""动态采集示例 —— 抓取亚马逊"继续选购"个性化推荐页的商品。

对应课案「动态采集」部分。课案用 Playwright 直接驱动浏览器；这里换用
Scrapling 的 StealthyFetcher（同样基于真实浏览器渲染，但带指纹伪装）来抓页面，
亚马逊这类做了自动化检测的真实站点用纯 requests 拿不到、用裸 Playwright
也容易被识破，所以用仓库里专门做隐身抓取的取件器。

提取字段：标题 / 价格 / 评分 / 评论数 / 图片 URL。

实现细节：这个页面不是搜索结果页，没有 `data-component-type="s-search-result"`
这类稳定容器类名（容器 class 里夹着 `_cDEzb_*` 这种会随会话变化的混淆后缀）。
所以这里不靠容器类名定位，而是用「评分锚点」法：每个商品评分图标
`span.a-icon-alt` 作为锚点，向上找到同时包含 `/dp/` 标题链接和售价 `a-price`
的最近共同祖先作为一张商品卡，再从中取标题/价格/评分/评论四项字段——
这样对结构变化最不敏感。图片单独用「标题链接」作锚：商品图在标题链接的
兄弟位置，从标题链接向上找最近的 `img[src*='/images/I/']` 即可（离得近，
通常 up=1~2），比用评分块作锚找图稳定得多。
解析用 BeautifulSoup（行为确定可测）；抓取仍由 Scrapling 负责隐身渲染。
"""
from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup
from scrapling.fetchers import StealthyFetcher

from conf import settings


OUTPUT = "amazon_kitchen_products.json"

# 评分块文本形如 "4.3 颗星，最多 5 颗星" / "4.6 out of 5 stars"
RATING_RE = re.compile(r"([\d.]+)\s*(?:out of 5\s*)?(?:颗星|星级|stars)")
# 评价/购买提示等噪声词：标题链接若含这些说明是个把整块都包起来的大包裹链接
NOISE_RE = re.compile(r"颗星|JPY|市场价|过去一个月|顾客购买|已查看|个选项")


def _reviews_count(text: str | None) -> int | None:
    """从 "11,706 customer reviews" / "688 customer reviews" 解析评论数。"""
    m = re.search(r"[\d,]+", text or "")
    if not m:
        return None
    try:
        return int(m.group().replace(",", ""))
    except ValueError:
        return None


def _image_of(a) -> str | None:
    """从标题链接 a 向上找最近的亚马逊商品图（src 含 /images/I/）。"""
    node = a
    for _ in range(14):
        node = node.parent
        if node is None:
            return None
        for im in node.select("img"):
            s = im.get("src", "") or ""
            if "/images/I/" in s:
                # 协议相对 URL（以 // 开头）补上 https:
                return ("https:" + s) if s.startswith("//") else s
    return None


def parse_items(html: str) -> list[dict]:
    """从整页 HTML 解析出所有商品。返回 list[{asin, title, price, rating, reviews, image}]。"""
    soup = BeautifulSoup(html, "lxml")
    items: list[dict] = []
    seen: set[str] = set()
    for ic in soup.select("span.a-icon-alt"):
        m = RATING_RE.match(ic.get_text(strip=True))
        if not m:
            continue
        node = ic
        for _ in range(8):  # 向上最多 8 层找齐标题 + 售价的卡片块
            node = node.parent
            if node is None:
                break
            dp_links = node.select("a[href*='/dp/']")
            price_els = node.select("span.a-price:not(.a-text-price) .a-offscreen")
            if not (dp_links and price_els):
                continue
            titles = []
            for a in dp_links:
                href = a.get("href", "")
                mm = re.search(r"/dp/([A-Z0-9]{10})", href)
                if not mm:
                    continue
                t = a.get_text(strip=True)
                if t and not NOISE_RE.search(t):  # 排除把整块打包的污染大链接
                    titles.append((mm.group(1), t, a))
            if not titles:
                continue
            asin, title, title_a = max(titles, key=lambda x: len(x[1]))
            reviews = ""
            for s in node.select("a[aria-label], span[aria-label]"):
                lbl = s.get("aria-label", "") or ""
                if "customer reviews" in lbl:
                    reviews = lbl
                    break
            try:
                rating = float(m.group(1))
            except ValueError:
                rating = None
            if asin not in seen:
                seen.add(asin)
                items.append({
                    "asin": asin,
                    "title": title,
                    "price": price_els[0].get_text(strip=True) or None,
                    "rating": rating,
                    "reviews": _reviews_count(reviews),
                    "image": _image_of(title_a),
                })
            break
    return items


class AmazonDynamicCrawler:
    """用 Scrapling StealthyFetcher 抓取亚马逊商品列表页。"""

    def __init__(self, start_url: str | None = None):
        self.start_url = start_url or (settings.start_url or settings.search_url_for(settings.keyword))
        self.items: list[dict] = []

    def crawl(self) -> list[dict]:
        print(f"抓取: {self.start_url}")
        page = StealthyFetcher.fetch(
            self.start_url, headless=settings.headless, network_idle=True, solve_cloudflare=False
        )
        self.items = parse_items(page.html_content)
        print(f"解析出 {len(self.items)} 条商品")
        return self.items

    def save(self, path: str = OUTPUT):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)
        print(f"已保存到 {path}")


def main():
    crawler = AmazonDynamicCrawler()
    crawler.crawl()
    crawler.save()
    for it in crawler.items[:5]:
        print(it)


if __name__ == "__main__":
    main()
