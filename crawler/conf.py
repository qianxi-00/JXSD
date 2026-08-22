"""采集配置。

字段说明：
- 目标站点：start_url 直接指向亚马逊移动端"继续选购"个性化推荐页
  (mobile mission，非搜索结果页，无分页)；url/keyword/max_pages
  仅在未设 start_url 时兜底使用。
- LLM：只有智能采集 (ai_crawler.py) 用到，可换成任意 OpenAI 兼容接口
  (DeepSeek、OpenAI 官方、本地 vLLM 等)。

pydantic-settings 会按字段名自动读取同名环境变量(大小写不敏感)，
所以 `api_key` 字段可用环境变量 `API_KEY` 覆盖，无需改代码。
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- 目标站点 ---
    # 亚马逊"继续选购"个性化推荐页(mobile mission)，单页、无分页
    start_url: str = (
        "https://www.amazon.com/-/zh/hz/mobile/mission/?_encoding=UTF8"
        "&p=3soxvJly1SBAJgfQkuUG1sOSePZ14JHN5%2Bkmpd2CT%2BPLEW229JUSnH96HZrpxeLuIQcVQEAOpVOmbkz0RI"
        "WPyry7IUdlFIy62AY%2BM4WtlToCSXQv0yVU9U86VDdb%2F%2Fsv5P8JsIqCsTIpH9LaoJqrdKOBnqH%2FCL7i9q%2Fu"
        "dVfaMMDEdw6di6tZhw2F4dY7J5bmCwuZVOBM%2FIXjxV%2F6p%2FVzs%2Fl21w01Gco4ZoDE5wDVa0q3N%2BeRpH1G"
        "TWC3%2BGHQO1JDxkwXEZFIsNPQoyirHNAlo6beq6UxyONQK0Rq%2FlN7uSvX4UdVh2zfhJnlZOjSxen579DoVlqxWd"
        "oZv8ejjiK0Q32kDPkOdIK2GuXzLpVmjh9TSbDr%2B1i2Iv%2Bsgw7Ztw2YsInDtvldwj8loK4viwYJTPzxjmqt%2FMX"
        "e2zzCHV3HMnLISJPF573QfSlPQOgoDzdJ8jEyfR3n4LF2Iw8xkNZnzu6D%2FygXx6tE"
        "&pd_rd_w=VdFle"
        "&content-id=amzn1.sym.567e5c5f-48c8-48b3-aff4-1448a2e1facc%3Aamzn1.symc.050ea944-f1cf-4610-b462-3b604f2f4082"
        "&pf_rd_p=567e5c5f-48c8-48b3-aff4-1448a2e1facc"
        "&pf_rd_r=WZHV2CVN66ME0FR69S4D"
        "&pd_rd_wg=5EUFg"
        "&pd_rd_r=26330927-59b9-4cb5-a63d-e7297c838909"
        "&ref_=pd_hp_d_btf_ci_mcx_mr_ca_id_hp_d"
    )
    keyword: str = "kitchen products"  # 仅在未设 start_url 时用
    base_search_url: str = "https://www.amazon.com/s?k={query}&language=zh&_encoding=UTF8"
    max_pages: int = 1  # 此页无分页，始终为 1
    headless: bool = True

    # --- LLM（智能采集用）---
    api_key: str = "sk-5RpsSDPqKMNSvK3FfMzFXVOGorcrKh5ChvSyziQ8QxRh2Cos"
    model_name: str = "grok-4.6"
    base_url: str = "https://qianxi7988.me/v1"
    temperature: float = 0.0

    def search_url_for(self, keyword: str) -> str:
        """根据关键词拼接搜索结果 URL（未设 start_url 时的兜底）。"""
        return self.base_search_url.format(query=keyword.replace(" ", "+"))


settings = Settings()
