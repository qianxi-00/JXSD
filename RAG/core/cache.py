"""RAG 缓存层:精确问答缓存(短 TTL) + 预设问答相似度缓存,接入 RAG 流程最前端"""

import hashlib
import json
from pathlib import Path

import numpy as np

from config import settings
from core.logger import logger
from core.redis_client import get_redis_client

PRESET_QA_PATH = Path(__file__).resolve().parent.parent / "data" / "preset_qa.json"
EXACT_KEY_PREFIX = "rag:qa:exact:"
PRESET_PAIRS_KEY = "rag:preset:pairs"
PRESET_VECTORS_KEY = "rag:preset:vectors"


class AnswerCache:
    """两层缓存:
    1. 精确缓存:10 分钟内相同 query 直接返回之前的回答;
    2. 预设缓存:query 与 60 条预设问题做 embedding 余弦相似度,>= 阈值直接返回预设答案。
    """

    def __init__(self) -> None:
        self.redis = get_redis_client()
        self._preset_questions: list[str] | None = None
        self._preset_answers: list[str] | None = None
        self._preset_matrix = None

    def lookup(self, query: str) -> dict | None:
        """依次尝试两层缓存,Redis 异常时降级返回 None(继续走 RAG)"""
        try:
            exact = self._lookup_exact(query)
            if exact is not None:
                return {**exact, "cache_hit": "exact"}
            return self._lookup_preset(query)
        except Exception as exc:
            logger.warning(f"[缓存] lookup 失败,降级走 RAG: {exc}")
            return None

    def store(self, query: str, answer: str, sources: list[dict]) -> None:
        """把完整问答结果写入精确缓存,10 分钟过期"""
        try:
            payload = json.dumps(
                {"question": query, "answer": answer, "sources": sources},
                ensure_ascii=False,
            )
            self.redis.set(self._exact_key(query), payload, ex=settings.redis.exact_ttl)
        except Exception as exc:
            logger.warning(f"[缓存] store 失败: {exc}")

    def seed_preset(self, force: bool = False) -> int:
        """将预设问答对及其问题向量写入 Redis,返回写入条数"""
        if not force and self.redis.exists(PRESET_PAIRS_KEY, PRESET_VECTORS_KEY) == 2:
            return 0
        from retrieval.embedding import embed_texts

        pairs = json.loads(PRESET_QA_PATH.read_text(encoding="utf-8"))
        vectors = embed_texts([p["question"] for p in pairs])
        pipe = self.redis.pipeline()
        pipe.set(PRESET_PAIRS_KEY, json.dumps(pairs, ensure_ascii=False))
        pipe.set(PRESET_VECTORS_KEY, json.dumps(vectors))
        pipe.execute()
        logger.info(f"[缓存] 预设问答写入 Redis 完成,共 {len(pairs)} 条")
        return len(pairs)

    @staticmethod
    def _exact_key(query: str) -> str:
        return EXACT_KEY_PREFIX + hashlib.md5(query.encode("utf-8")).hexdigest()

    def _lookup_exact(self, query: str) -> dict | None:
        raw = self.redis.get(self._exact_key(query))
        if raw is None:
            return None
        return json.loads(raw)

    def _ensure_preset_loaded(self) -> None:
        if self._preset_matrix is not None:
            return
        self.seed_preset()
        pairs = json.loads(self.redis.get(PRESET_PAIRS_KEY))
        vectors = np.array(json.loads(self.redis.get(PRESET_VECTORS_KEY)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._preset_questions = [p["question"] for p in pairs]
        self._preset_answers = [p["answer"] for p in pairs]
        self._preset_matrix = vectors / np.maximum(norms, 1e-10)

    def _lookup_preset(self, query: str) -> dict | None:
        self._ensure_preset_loaded()
        from retrieval.embedding import embed_query

        vec = np.array(embed_query(query), dtype=np.float32)
        vec = vec / max(float(np.linalg.norm(vec)), 1e-10)
        sims = self._preset_matrix @ vec
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < settings.redis.sim_threshold:
            return None
        return {
            "question": self._preset_questions[best],
            "answer": self._preset_answers[best],
            "sources": [],
            "cache_hit": "preset",
            "similarity": round(score, 4),
        }
