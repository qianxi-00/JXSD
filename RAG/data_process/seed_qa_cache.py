# =============================================================================
# 预问答缓存灌库脚本（RAG 基础篇「FAQ 缓存 / 预设问答」）
#
# 职责：把 data/preset_qa.json 里的预问答对写进 Redis 的两层缓存。
#   第 1 层 相似度层（rag:preset:pairs + rag:preset:vectors）：layer=faq 的标准问法，
#           连同它们的问题向量一起写入，查的时候做余弦相似度匹配；
#   第 2 层 精确层（rag:qa:exact:*）：layer=detail 的明细问答，只按归一化问题精确命中。
#
# 为什么必须分两层（这是本脚本存在的唯一理由）：
#   明细问答问的是"某个人的某一张票"，答案里带具体姓名/金额/座位号。这类问题彼此
#   embedding 相似度极高——实测「赵凡的登机牌座位号是多少？」与「赵飞的登机牌座位号
#   是多少？」相似度 0.9008，高于相似度阈值 0.79。一旦明细问答进了相似度层，问赵凡
#   的票就会命中赵飞的答案，把**别人的票**答出去；而且预设命中时 sources 为空，
#   界面上完全看不出是答错了还是真没数据。所以明细只做精确命中（问题一字不差才命中）。
#
# 本脚本是**幂等**的：不加 --force 时两层各自检查自己的"已灌库标记"（preset 层看
# rag:preset:pairs / rag:preset:vectors 是否都已存在，detail 层看 rag:detail:seeded），
# 已存在就跳过并返回 0——所以输出里的 "already exist" 不是失败，是正常的重复执行。
# =============================================================================

# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
# 这段必须在任何业务 import **之前**执行：本脚本既能被 `python -m data_process.seed_qa_cache`
# 调用，也能被直接 `python RAG/data_process/seed_qa_cache.py` 调用。后一种情况下
# sys.path[0] 是脚本所在目录（RAG/data_process），既看不到仓库根的 config，也看不到
# RAG 下的 core 包，import 会直接 ModuleNotFoundError。
import sys as _sys
from pathlib import Path as _Path

# 从本文件出发向上找名为 "Python_Base" 的祖先目录；两个终止条件缺一不可：
#   _BASE.parent != _BASE —— 已经到盘符根（如 F:\），再往上没有意义；
#   _BASE.name != "Python_Base" —— 找到了目标目录。
# 若整个路径里没有 Python_Base（例如仓库被改名/被复制到别处），循环会停在盘符根，
# 下面插进 sys.path 的路径就是错的——但不会抛异常，表现为后面 import 失败，
# 排查时先确认这一层。用 while 而不是写死 ../.. 是为了容忍目录被整体搬移。
_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
# 注意插入顺序：insert(0) 是头插，所以后执行的这一行反而排在 sys.path 更前面，
# 最终顺序是 [RAG, Python_Base, ...]。RAG 在前是为了让 RAG 内的包名优先命中；
# 因为仓库根没有与 RAG 内同名的模块，这里不会产生遮蔽问题。
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""将预设问答对及其问题向量写入 Redis 缓存

用法: uv run python -m data_process.seed_qa_cache [--force]
"""

import sys
from pathlib import Path

# 第二道导入兜底（第一道见文件顶部的 sys.path 引导）。
# 顶部引导已经把 Python_Base 根和 RAG 根插进 sys.path，正常路径下这里不会走到 except；
# 保留 try/except 是为了容忍"顶部引导没找到 Python_Base 目录"的异常场景：
# 那时按本文件位置再补一次 RAG 根（core 包所在目录），尽量把 import 救回来。
try:
    from core.cache import AnswerCache
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.cache import AnswerCache


def main() -> None:
    # 用裸 sys.argv 而非 argparse：本脚本只有一个布尔开关，为此引入解析层不划算。
    # 代价是匹配很宽松：`--force-anything` 只要包含 `--force` 也会触发强制刷新。
    force = "--force" in sys.argv
    # AnswerCache() 构造时就会连 Redis（core.redis_client）。Redis 不可用时这里会抛异常，
    # 脚本直接失败——这是**有意**的：灌库失败必须让人看见，不能静默当成功。
    cache = AnswerCache()
    # 两层分别写:标准问法进相似度层,明细问答只进精确层(避免"答成别人的票")
    # 两个调用是独立幂等的，各自检查各自的标记位，不要在这里做"任一为 0 就跳过另一个"的短路。
    preset_count = cache.seed_preset(force=force)
    detail_count = cache.seed_details(force=force)
    # 返回 0 有两种含义：本次真的没写（已存在），或者 preset_qa.json 里该层本来就没有条目。
    # 脚本不区分这两者，只打印"已存在"——排查时以 preset_qa.json 的 layer 字段为准。
    if preset_count:
        print(f"seeded {preset_count} preset(faq) QA pairs into redis")
    else:
        print("preset(faq) QA pairs already exist in redis (use --force to refresh)")
    if detail_count:
        print(f"seeded {detail_count} detail QA pairs into redis (exact match only)")
    else:
        print("detail QA pairs already exist in redis (use --force to refresh)")


if __name__ == "__main__":
    main()
