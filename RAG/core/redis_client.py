"""Redis 连接管理

为什么是「1 秒超时 + 零重试」这两个看起来很激进的参数(而不是默认值):
Redis 在本项目里只承担**缓存**职责 —— 命中就省一次 LLM/检索,不命中就走完整链路,
功能上永远是"可降级"的。而默认配置(无超时、失败重试)在 Redis 慢或挂掉时的行为是:
请求**卡在缓存层**、重试几轮后才抛错,单次问答平白多等数秒到数十秒。
用户感知到的就是"整个问答变慢了",而真正的原因在缓存。

所以这里的选择是把**故障时间预算压到最小**:连不上/读不到就直接失败,
由 `core/cache.py::AnswerCache.lookup` 的 try/except 捕获并降级去走 RAG。
「失去缓存」的代价远小于「被缓存拖慢」。

三个参数的细节:
- `socket_timeout=1`:单条命令(读/写)的等待上限。设为 1 秒的依据是本地/同机 docker
  部署的 Redis 正常响应在毫秒级,1 秒已经是很宽的余量;真慢到这个程度说明它在故障;
- `socket_connect_timeout=1`:建连上限。Redis 没起时不再等 TCP 默认超时;
- `retry=Retry(NoBackoff(), 0)`:显式声明**重试 0 次**。`NoBackoff()` 表示失败之间
  不等待,次数 0 表示不重试 —— 两条合起来才是"立刻失败"。**写这个是为了显式**:
  redis-py 不同版本的默认重试策略不同,不写就等于把行为交给版本决定。

⚠ 本模块是**进程级单例**(`_client` 全局变量 + 惰性创建)。这意味着:
1. 第一个调用者决定连接参数,后面所有调用复用同一个连接池;
2. 测试里想换指向(比如指向假 Redis)必须重置 `_client`,没有提供 setter ——
   这让「同一进程内换 Redis 实例」变得不方便(见报告);
3. 惰性创建的副作用:配置错误(主机名写错)不会在 import 时暴露,而是第一次
   访问缓存时才报错并降级 —— 表现是"缓存一直不命中",很容易被误判成缓存逻辑坏。
"""

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from config import settings

# 进程级单例。用 None 做"未初始化"的哨兵 —— 注意不能写成 `if not _client` 之外的
# 其它判空方式,redis.Redis 实例的真值永远是 True,这里用 `is None` 是必须的
_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """返回进程级共享的 Redis 客户端(首次调用时创建)。

    `decode_responses=True` 是必须的:cache.py 里到处用 `json.loads(redis.get(...))`,
    不开启这个开关拿到的就是 bytes,json.loads 虽然能接受 bytes,但
    `self.redis.exists(...) == 2`、字符串比较之类的操作会因类型不一致出诡异结果。
    开启后所有返回值都是 str,业务层不需要到处 .decode()。

    `password or None`:配置里密码为空串时传 None 而不是 "" —— 空字符串会被
    redis-py 当成"要发 AUTH 命令、密码是空",对没设密码的 Redis 直接报认证错误。
    """
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
