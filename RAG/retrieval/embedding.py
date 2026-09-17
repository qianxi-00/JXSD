"""向量模型:基于 OpenAI 兼容接口的文本向量化

在 RAG 链路里的位置
==================
这是**入口级依赖**:向量路召回与 FAQ 缓存(相似度层)都要先经过它。调用方共用同一个客户端:
  - `retrieval/vector_retrieval.py::vector_search` —— 把用户问题向量化后去 Milvus 检索;
  - `core/cache.py` —— 启动时批量算标准问法向量(相似度层只认显式打了 `layer: faq` 的 5 条,
    避免跨人串答案),请求时算 query 向量再做余弦比对;
  - `graph_rag/builder.py` / `graph_rag/retriever.py` —— 算图谱实体与社区摘要的向量;
  - `data_process/embed_tickets.py` —— 给入库票据批量算向量(空 `semantic_text` 的行会跳过)。

当前模型栈与换栈代价
====================
`BAAI/bge-m3` @ SiliconFlow,原生 **1024 维**,与 `EMBEDDING_SIZE` 一致 —— Milvus 集合的
`vec` 字段就是按这个维度建的。**换 embedding 模型必须重算库里所有存量向量**:不同模型的向量
空间不通用,两套向量混在同一个集合里,余弦相似度就没有意义了,症状是"召回一批完全无关的票"
而且全程不报错(旧栈 `Qwen/Qwen3-Embedding-4B` 是原生 2560 维、靠 MRL 截断成 1024,与本栈不通用)。

本文件守住的两个坑
==================
1. `dimensions` **只有显式开启才下发**:`BAAI/bge-m3` 不接受这个参数,传了直接
   `400 20015 The parameter is invalid`。偏偏它和旧栈都能凑出 1024 维 ⇒
   「维度对得上」不等于「参数也能传」,这是换模型时最容易踩的一步。
2. 返回维度**必须校验**:维度不符写进 Milvus 会变成"不报错但搜不准"的静默坏数据,
   比直接 400 难查得多,所以这里宁可抛异常把链路打断。

对应课案:`RAG 基础篇` 数据入库(向量化)一节 + `RAG 优化篇` 换模型/重算向量一节。
"""

import time

from openai import OpenAI

from config import settings

# 进程级单例。OpenAI 客户端内部有连接池,每次调用都新建一个会白白重建连接、拖慢向量化。
# 这里只缓存**客户端**,不缓存向量结果 —— 向量缓存(精确层 / FAQ 相似度层)是 `core/cache.py`
# 的职责,两层缓存的边界不要混。
_client: OpenAI | None = None


def get_embedding_client() -> OpenAI:
    """返回进程内共享的 embedding 客户端(懒加载:首次调用时才构造)。

    为什么手写单例而不用 `functools.lru_cache`:本函数无参,手写更直白;
    而且测试里可以直接 monkeypatch 模块级 `_client` 或本函数(`tests/test_embedding.py`)。
    """
    global _client
    if _client is None:
        # 显式超时：SDK 默认 600 秒 × 重试 3 次，上游挂住时向量化会长时间不返回
        _client = OpenAI(
            api_key=settings.embedding.api_key,
            base_url=settings.embedding.base_url,
            timeout=settings.embedding.timeout,
        )
    return _client


def _request_kwargs() -> dict:
    """构造请求参数。

    `dimensions` **只有显式开启才下发**:它不是所有模型都支持的参数 ——
    实测 SiliconFlow 上 `BAAI/bge-m3` 传了直接 `400 20015 The parameter is invalid`,
    而不传时正常返回 1024 维;需要 MRL 截断的 Qwen3-Embedding 系才用得上它。

    开关是配置项 `EMBEDDING_SEND_DIMENSIONS`(`settings.embedding.send_dimensions`),默认 False。
    换模型时按目标模型的能力切:接受 `dimensions` 才打开,否则整个向量化环节全 400
    (且报错点在第三方网关、信息量只有一行 `20015`,很容易误判成代码 bug)。

    返回的是「展开进 `client.embeddings.create(**kwargs)` 的字典」,所以不支持时返回空字典,
    而不是返回 `{"dimensions": None}` —— `None` 也会被当成"传了该参数"。
    """
    if settings.embedding.send_dimensions:
        return {"dimensions": settings.embedding.embedding_size}
    return {}


def _assert_dimensions(vectors: list[list[float]]) -> None:
    """校验返回维度与配置一致。

    Milvus 集合是按 `EMBEDDING_SIZE` 建的:换模型后维度对不上会变成"能写进去但搜不准"
    的静默坏数据,所以这里宁可报错。

    实现细节(读的时候值得留意):
    - `{len(v) for v in vectors}` 是**集合**:先把本批次的维度去重,所以
      `sizes == [expected]` 等价于「本批每条向量都是 expected 维」;
      只要有任意一条不符(或同一批里出现两种维度)都会失败,不会只校验第一条;
    - 返回的是 `sorted` 后的列表,错误信息里能直接看到实际维度,便于判断是"换错模型"
      还是"网关偷偷截断";
    - 这里抛 `RuntimeError` 而不是 `ValueError`:它与「请求失败」用同一个异常类型,
      调用方(`pipeline/rag_pipeline.py`、`data_process/embed_tickets.py`)无需区分处理。
    """
    expected = settings.embedding.embedding_size
    sizes = sorted({len(vector) for vector in vectors})
    if sizes != [expected]:
        raise RuntimeError(
            f"embedding 维度不符:模型 {settings.embedding.model} 返回 {sizes} 维,"
            f"而配置 EMBEDDING_SIZE={expected}。Milvus 集合按 {expected} 维建立,"
            f"维度不一致会写入搜不准的坏向量 —— 请换回 {expected} 维的模型,"
            f"或改 EMBEDDING_SIZE 并重建集合。"
        )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """将文本批量向量化,带简单重试,返回与输入顺序一致的向量列表

    约束与语义(调用方依赖这些):
    - **空输入直接返回空列表,不发请求**:建库时整批票据可能都是空 `semantic_text`
      (`data_process/embed_tickets.py` 会先过滤),不该为它白跑一次网络;
    - **返回值与 `texts` 一一对应且同序**:调用方是靠下标把向量与文本配对的
      (如 `core/cache.py` 把问法与向量分别序列化存进 Redis),顺序错了就是"问 A 命中 B"。
      接口对 `resp.data` 的顺序没有强保证,所以按每条自带的 `index` 重新排序;
    - **失败重试 3 次**,退避 2s / 4s / 6s,全部失败抛 `RuntimeError` 并把最后一次的原始
      异常拼进消息(否则现场只剩一个语焉不详的失败);
    - **维度校验不参与重试**:`_assert_dimensions` 刻意写在 `try` 之外 ——
      维度不符是配置问题,重试一万次也一样,直接抛出去
      (`tests/test_embedding.py::test_dimension_mismatch_is_not_retried` 钉住了"只请求一次")。

    已知的小瑕疵(不改,只记):最后一次 attempt 失败后仍会 `sleep(6)` 才抛异常;
    另外**只校验维度、不校验条数** —— 网关若少返回几条,调用方的 zip/下标配对会静默错位
    (见 README §8.3)。
    """
    if not texts:
        return []
    client = get_embedding_client()
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.embeddings.create(
                model=settings.embedding.model,
                input=texts,
                **_request_kwargs(),
            )
        except Exception as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
            continue
        vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        _assert_dimensions(vectors)  # 维度不符是配置问题,重试没有意义
        return vectors
    raise RuntimeError(f"embedding 请求失败: {last_exc}")


def embed_query(text: str) -> list[float]:
    """单条查询向量化:把问题/实体名转成向量,供向量召回与 FAQ 相似度比对使用。

    返回裸列表而不是嵌套列表,是因为调用方要的都是一条向量
    (`vector_search` 塞给 Milvus 的 `data=[vector]`,`core/cache.py` 直接 `np.array(...)`)。
    注意:**不在这里校验空串**。空串的向量是一个无意义的"语义中心点",会让相似度比对
    退化成"跟谁都差不多";需要防的话由调用方负责
    (`data_process/embed_tickets.py` 就是先过滤空 `semantic_text` 再调的)。
    """
    return embed_texts([text])[0]
