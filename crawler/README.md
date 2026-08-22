# crawler —— 亚马逊商品采集示例

两个脚本对应课案「数据采集」的两种采集方式，都换成用
[Scrapling](https://github.com/D4Vinci/Scrapling) 的 `StealthyFetcher` 做取页面这一步：
亚马逊做了自动化检测，纯 `requests` 静态请求拿不到数据，裸 Playwright 也容易被识破，
所以用仓库里专门做隐身抓取的取件器（指纹伪装，基于真实浏览器渲染）。

| 文件 | 对应课案部分 | 做法 |
| --- | --- | --- |
| `dynamic_crawler.py` | 动态采集 | StealthyFetcher 渲染页面 + BeautifulSoup 提取 |
| `ai_crawler.py` | 智能采集 | StealthyFetcher 抓取 + 卡片转 Markdown + LLM 抽 JSON |
| `conf.py` | 配置 | 目标 URL、LLM 配置 |

采集目标：亚马逊移动端「继续选购」个性化推荐页（`/hz/mobile/mission/`）
**所有商品**的 标题 / 价格 / 评分 / 评论数 / 图片 URL。该页是无分页的单页推荐流。

## 安装

```bash
pip install "scrapling[fetchers]"
scrapling install          # 下载隐身浏览器内核(Camoufox 等)，必需
pip install pydantic-settings beautifulsoup4 lxml
pip install html2text openai   # 仅智能采集需要
```

> `pip install scrapling` 默认只装解析引擎，不含取件器；必须用
> `[fetchers]` 再加 `scrapling install` 才能用到 `StealthyFetcher`。

## 运行

```bash
# 动态采集（CSS 锚点提取）-> amazon_kitchen_products.json
python dynamic_crawler.py

# 智能采集（LLM 抽取）  -> amazon_kitchen_products_ai.json
# 先在 conf.py 里填好 api_key/model_name/base_url
python ai_crawler.py
```

目标 URL 已写在 `conf.py` 的 `start_url`（亚马逊「继续选购」个性化推荐页）；
想爬别的亚马逊商品列表页，把它换成完整 URL 即可。

## 提取策略：为什么不用固定选择器

最早的版本按搜索结果页结构写死选择器（`div[data-component-type="s-search-result"]`、
`h2.a-text-normal`、`a.s-underline-text` 等）。这次目标换成了亚马逊移动端
「继续选购」个性化推荐页后，这些选择器全部 0 命中：

- 没有稳定的容器类名，容器 class 里夹着 `_cDEzb_*` 这种会随会话变化的混淆后缀；
- 商品成交三种布局（单商品、两商品并列、首屏特推）混在一页；
- 商品标题文本甚至不在固定的标题元素里，有的标题链接干脆把整块
  （评分、价格、评论）都包进一个 `<a>` 里，直接取 `text` 会把整段都拼进来。

所以这一版改用「评分锚点」相对定位法，不依赖容器类名：

1. 每个商品评分块 `span.a-icon-alt`（文本如 `4.3 颗星，最多 5 颗星`）作锚点；
2. 从它往上最多 8 层，找同时包含「指向 `/dp/<ASIN>` 的标题链接」和
   「售价 `span.a-price:not(.a-text-price) .a-offscreen`」的最近共同祖先，
   把这块当作一张商品卡；
3. 在该块内取标题/价格/评分/评论四项；图片单独用标题链接作锚
   （从标题链接向上找最近的 `img[src*='/images/I/']`）。

各字段定位（相对像均在该卡片内）：

| 字段 | 取法 | 说明 |
| --- | --- | --- |
| ASIN | 标题链接 `href` 里的 `/dp/<ASIN>` 正则 | 商品唯一标识 |
| 标题 | `a[href*='/dp/']` 文本，过滤噪声后取最长 | 排除把整块打包的污染大链接 |
| 价格 | `span.a-price:not(.a-text-price) .a-offscreen` | 只取实际售价，排除划线市场价 |
| 评分 | `span.a-icon-alt` 文本正则 `([\d.]+)\s*颗星` | 自动适配中文「颗星」/英文「out of 5 stars」 |
| 评论数 | `[aria-label]` 里含 `customer reviews` 的 `aria-label` | `11,706 customer reviews` -> `11706` |
| 图片 | 从标题链接 `a[href*='/dp/']` 向上找最近的 `img[src*='/images/I/']` 的 `src` | 商品图在标题链接的兄弟位置、离得近，比用评分块作锚稳定 |

这些就是课案讲的 `.class` / 属性选择器 / 后代选择器 / `::text` / `::attr()`
那套 CSS 选择器的延伸，只是这里因为容器不稳定，改用「内容语义锚点 +
向上寻公共祖先」的写法，比写死容器更不易碎。

> 这些选择器是在用户贴的本页真实 HTML 上离线验证过的：38 个唯一商品、
> 四项字段加图片均零缺失。但线上实抓时，隐身浏览器渲染出的 DOM 可能与贴来的
> 快照略有出入（或被风控拦截），届时按上方策略调整锚点即可。

## 智能采集为什么只喂卡片而不喂整页

亚马逊页 HTML 约 850KB。整页转 Markdown 喂给 LLM 既贵又可能超上下文。
所以先按上面的「评分锚点」把每个商品的 HTML 切出来，单独转 Markdown 再拼接，
并在每张卡 markdown 末尾补一行 `![商品图片](url)`，
只把"目标内容"交给 LLM —— 这是 Scrapling 官方在 README 的 MCP / Agent Skill
章节推荐的"先抽取后交给 AI、降低 token"的做法。

## 备注

抓取仍由 Scrapling 的 `StealthyFetcher` 负责（隐身渲染 + 指纹伪装）；
解析改用 BeautifulSoup，是因为它在「评分锚点向上寻共同祖先」这种需要
精确父级游历的场景上行为确定可测，便于用真实快照离线验证选择器。

## 注意事项

- **反爬与稳定性**：亚马逊风控较重，同一出口 IP 频繁抓取可能触发验证码。
  稳定采集建议配合代理池（Scrapling 自带 `ProxyRotator`，或外接住宅代理），
  并控制频率。
- **API key**：智能采集的 `api_key` 在 `conf.py`，也可用环境变量 `API_KEY` 覆盖。
- **合规**：仅用于学习研究，遵守目标站点的 robots.txt 与服务条款，控制请求频率。
