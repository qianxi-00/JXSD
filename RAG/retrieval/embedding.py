"""向量模型:基于 OpenAI 兼容接口的文本向量化"""

import time

from openai import OpenAI

from config import settings

_client: OpenAI | None = None


def get_embedding_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.embedding.api_key, base_url=settings.embedding.base_url)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """将文本批量向量化,带简单重试,返回与输入顺序一致的向量列表"""
    if not texts:
        return []
    client = get_embedding_client()
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.embeddings.create(
                model=settings.embedding.model,
                input=texts,
                dimensions=settings.embedding.embedding_size,
            )
            data = sorted(resp.data, key=lambda d: d.index)
            return [d.embedding for d in data]
        except Exception as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"embedding 请求失败: {last_exc}")


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
