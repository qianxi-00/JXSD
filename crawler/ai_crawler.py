"""智能采集示例 —— 用 LLM 从亚马逊"继续选购"页抽取商品。

对应课案「智能采集」部分，流程一致：
    抓取页面 HTML -> 转成 Markdown -> LLM 理解内容 -> 返回结构化 JSON

两点差异（都为了适配真实的亚马逊页）：
1. 抓取用 Scrapling 的 StealthyFetcher（指纹伪装），而非裸 Playwright。
2. 不把整页约 850KB 的 HTML 干喂给 LLM，而是先把每个商品卡片的 HTML 单独
   转成 Markdown 再拼接送进去 —— 这正是 Scrapling 官方推荐的「先抽取
   目标内容再交给 AI，以降低 token 用量」的做法（见仓库 README 的
   MCP / Agent Skill 章节）。

卡片切分沿用 dynamic_crawler 的「评分锚点」法（见该文件注释）：
以每个 `span.a-icon-alt`（商品评分）为锚向上找同时含 `/dp/` 标题链接和
售价 `a-price` 的最近共同祖先，作为一张卡片。卡片 HTML 转成 Markdown 后，
再在该卡 markdown 末尾补一行商品图片（用 `dynamic_crawler._image_of` 从
标题链接定位，因为图片在标题链接的兄弟位置、不在评分锚点找到的紧凑块内），
让 LLM 也能拿到图片 URL。此法不依赖页面里会变的 `_cDEzb_*` 混淆类名。
"""
from __future__ import annotations

import json
import re

import html2text
import openai
from bs4 import BeautifulSoup
from scrapling.fetchers import StealthyFetcher

from conf import settings
from dynamic_crawler import _image_of


SYSTEM_PROMPT = """你是一个专业的数据抽取助手，擅长从网页内容中提取结构化信息。
请从下面给定的 Markdown 内容中抽取所有商品的信息。每个商品包含以下字段：
- title: 商品标题（完整文本，字符串）
- price: 价格（货币符号 + 金额，原样保留，例如 "JPY 30,131"）
- rating: 评分（数字，例如 4.6）
- reviews: 评论数（整数，例如 11706）
- image: 商品图片 URL（取 Markdown 里 `![商品图片](...)` 形式的图片地址，完整 URL 字符串）

只返回一个 JSON 对象，不要包含任何额外的说明文字或 markdown 代码块标记。
返回格式示例：
{
  "items": [
    {"title": "...", "price": "...", "rating": 4.6, "reviews": 11706, "image": "https://m.media-amazon.com/images/I/xxx.jpg"}
  ]
}
"""

OUTPUT = "amazon_kitchen_products_ai.json"

RATING_RE = re.compile(r"[\d.]+\s*(?:out of 5\s*)?(?:颗星|星级|stars)")
NOISE_RE = re.compile(r"颗星|JPY|市场价|过去一个月|顾客购买|已查看|个选项")

_h = html2text.HTML2Text()
_h.ignore_links = True
_h.ignore_images = False  # 保留卡片里的图片（如有）作为 markdown
_h.body_width = 0


def _parse_json(content: str) -> dict:
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return json.loads(s)


def cards_to_markdown(page) -> tuple[str, int]:
    """把每个商品卡片转成 Markdown 再拼接，并在每张卡末尾补一行商品图片。"""
    soup = BeautifulSoup(page.html_content, "lxml")
    parts = []
    seen: set[str] = set()
    for ic in soup.select("span.a-icon-alt"):
        if not RATING_RE.match(ic.get_text(strip=True)):
            continue
        node = ic
        for _ in range(8):
            node = node.parent
            if node is None:
                break
            dp_links = node.select("a[href*='/dp/']")
            price_els = node.select("span.a-price:not(.a-text-price) .a-offscreen")
            if not (dp_links and price_els):
                continue
            title_a = None
            for a in dp_links:
                if NOISE_RE.search(a.get_text(strip=True)):
                    continue
                mm = re.search(r"/dp/([A-Z0-9]{10})", a.get("href", ""))
                if mm and mm.group(1) not in seen:
                    seen.add(mm.group(1))
                    title_a = a
                    break
            if title_a is None:
                continue
            md = _h.handle(str(node)).strip()
            img = _image_of(title_a)
            if img:
                md = md + "\n\n" + f"![商品图片]({img})"
            if md:
                parts.append(md)
            break
    return "\n\n---\n\n".join(parts), len(parts)


class AICrawler:
    """AI 辅助的智能爬虫。"""

    def __init__(self, start_url: str | None = None):
        self.start_url = start_url or (settings.start_url or settings.search_url_for(settings.keyword))
        self.items: list[dict] = []

    def extract_with_llm(self, markdown_content: str) -> list[dict]:
        client = openai.OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        resp = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Markdown 内容：\n{markdown_content}"},
            ],
            temperature=settings.temperature,
            response_format={"type": "json_object"},
        )
        data = _parse_json(resp.choices[0].message.content)
        if isinstance(data, dict):
            return data.get("items", [data])
        return data

    def run(self) -> list[dict]:
        print(f"抓取: {self.start_url}")
        page = StealthyFetcher.fetch(
            self.start_url, headless=settings.headless, network_idle=True, solve_cloudflare=False
        )
        markdown, n = cards_to_markdown(page)
        print(f"已把 {n} 个卡片转成 Markdown（{len(markdown)} 字符）")
        if not n:
            print("未命中任何商品卡片，可能被拦截或页面结构变化，请检查 headless/代理。")
            return []
        self.items = self.extract_with_llm(markdown)
        print(f"AI 提取到 {len(self.items)} 条")
        return self.items

    def save(self, path: str = OUTPUT):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)
        print(f"已保存到 {path}")


def main():
    crawler = AICrawler()
    crawler.run()
    crawler.save()
    for it in crawler.items[:5]:
        print(it)


if __name__ == "__main__":
    main()
