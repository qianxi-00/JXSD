"""Redis 连接管理"""

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from config import settings

_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            password=settings.redis.password or None,
            db=settings.redis.db,
            decode_responses=True,
            socket_timeout=1,
            socket_connect_timeout=1,
            retry=Retry(NoBackoff(), 0),
        )
    return _client
