# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""Langfuse 评估数据集上传脚本(优化篇课案「数据源生成与上传脚本」)。

字段映射与课案一致:
    question         -> input.question
    ground_truth     -> expected_output
    expected_ticket_ids -> metadata.relevant_ids

**为什么要把评估集推到 Langfuse**(而不是本地跑算了):
本地评估只能给出一个总分;Langfuse 的价值在于**条目级可下钻** ——
哪个问题没召回、哪条回答被判低分,可以在 Web 界面上逐条看输入/输出。
`metadata.relevant_ids` 就是条目级 Recall@K 的判分依据(见 langfuse_evaluation.py)。

幂等语义:数据集已存在时直接复用、跳过上传;`--rebuild` 先删掉同名数据集的
全部数据项再重新上传(课案备注:Langfuse 公共 API 没有 DELETE /datasets/{name},
只能逐条删 dataset_items)。

⚠ 这里的幂等是"**整个数据集粒度**"的:判断依据是"名字存在"而不是"内容一致"。
所以改了 `data/eval_set.jsonl` 之后**必须** `--rebuild`,否则 Langfuse 里还是老样本,
而本地跑出来的报告用的是新样本 —— 两边对不上时会以为"Langfuse 的指标算错了"(见报告)。

用法:
    uv run python RAG/script/upload_langfuse_dataset.py
    uv run python RAG/script/upload_langfuse_dataset.py --rebuild
"""

import argparse
from pathlib import Path
from typing import Any

from config import settings
from evaluation.eval_set import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval_set.jsonl"
# 上传前必须齐全的三个字段 —— 它们分别对应 Langfuse 侧的
# input.question / expected_output / metadata.relevant_ids,缺任何一个都会
# 让条目级评估**无从判分**(而不是报错)。所以宁可在这里硬失败
REQUIRED_FIELDS = ("question", "expected_ticket_ids", "ground_truth")


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict[str, Any]]:
    """读取并校验评估集(必填字段齐全)。

    `load_jsonl` 已经保证了字段**类型**合法(evaluation/eval_set.py 的 validate_sample
    会给每个字段补默认值),所以这里的"缺字段"检查实际上永远为假 ——
    `validate_sample` 返回的字典**必定**含这三个键。
    保留它的理由:这是一道**防御性断言**,万一将来 validate_sample 的字段清单改了
    (或被绕过直接用 json.loads 读),这里能立刻发现"上传的数据不完整"。
    属冗余而非缺陷(见报告),本次不动。

    `FileNotFoundError` 里带上"先跑 build_eval_set.py"的提示:这是最常见的失败原因
    (全新克隆的仓库里 data/eval_set.jsonl 不存在),直接给出下一步命令比让使用者猜要好。
    """
    if not path.exists():
        raise FileNotFoundError(f"评估集不存在: {path}(先跑 script/build_eval_set.py)")
    rows = load_jsonl(path)
    for index, row in enumerate(rows, 1):
        missing = [name for name in REQUIRED_FIELDS if name not in row]
        if missing:
            raise ValueError(f"eval_set 第 {index} 行缺少字段: {missing}")
    return rows


def delete_dataset_items(langfuse: Any, dataset_name: str) -> int:
    """删除同名数据集的全部数据项(数据集不存在时视为无需删除)。

    ⚠ 两个刻意的设计(踩过才知道的):
    1. **只删 items,不删数据集本身**:Langfuse 公共 API 没有
       `DELETE /datasets/{name}`,数据集壳子删不掉。所以 --rebuild 的语义是
       "清空内容、保留数据集",而不是"删了重建";
    2. **逐条删除**(而不是批量接口):`dataset.items` 是**一次性取回全部**数据项,
       条目多时这里会把整个列表拉进内存;Langfuse 也没有提供批量删除。
       当前评估集只有几十条,可以接受;上百条要考虑分批 + 限流(见报告)。

    `except Exception: return 0` 把"数据集不存在"和"**鉴权失败/网络不通**"混在一起了:
    后者也会返回 0(显示"清空 0 条"),然后后面的 create_dataset_item 才开始报错。
    行为上最终仍会失败,只是**错误信息指向的位置不对**。属可改进项(见报告)。

    返回删除条数只为打印,调用方不使用该值做判断。
    """
    try:
        dataset = langfuse.get_dataset(dataset_name)
    except Exception:  # noqa: BLE001 不存在即无需删除
        return 0
    for item in dataset.items:
        langfuse.api.dataset_items.delete(id=item.id)
    return len(dataset.items)


def upload_dataset(samples: list[dict[str, Any]], dataset_name: str, rebuild: bool = False) -> None:
    """把样本上传为 Langfuse 数据集。

    **凭证检查放在最前面、且抛明确异常**:缺凭证时 Langfuse SDK 的报错信息
    与"网络不通"很像,容易往错的方向查。这里直接点名要配哪两个变量。

    `from langfuse import Langfuse` 是函数内导入:本模块的 `load_eval_set` 是可离线
    测试的纯逻辑,把 SDK 导入放函数里可以让测试不依赖 langfuse 包已安装/版本正确。

    控制流要连起来读(三段存在性判断):
    1. `rebuild=True` 时先清空 items(数据集壳子留着);
    2. 再探一次数据集是否存在 —— 因为删 items 不会删掉数据集,**所以 rebuild 之后
       `exists` 通常仍然是 True**,`elif not rebuild` 这个分支就是为此设的:
       清空过的数据集要**继续往下走去重传**,而不是"已存在,跳过";
    3. 只有"不存在"才 `create_dataset`。

    这段控制流的正确性依赖"重建时数据集一定已存在"这个前提。若 rebuild=True
    但数据集**本来就不存在**,会走到 `not exists` → 创建 → 继续上传,结果也正确。

    `flush()` 是**必须**的:Langfuse SDK 默认**异步批量**上报,不 flush 的话
    脚本可能在数据真正发出去之前就退出,表现为"显示上传了 36 条,但界面上只有 3 条"。
    打印的条数来自 `len(samples)`,是**打算**上传的条数,不等于**成功落库**的条数。
    """
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        raise ValueError(
            "缺少 Langfuse 凭证:请在 .env 配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"
            "(可选 LANGFUSE_HOST)"
        )

    from langfuse import Langfuse

    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

    if rebuild:
        deleted = delete_dataset_items(langfuse, dataset_name)
        print(f"已清空数据集 {dataset_name} 的 {deleted} 条旧数据项,准备重建")

    # 用 try/except 探测存在性(SDK 没有 exists 接口),靠"取不到就抛异常"当判据
    exists = True
    try:
        langfuse.get_dataset(dataset_name)
    except Exception:  # noqa: BLE001 不存在则创建
        exists = False

    if not exists:
        langfuse.create_dataset(name=dataset_name)
    elif not rebuild:
        print(f"数据集 {dataset_name} 已存在,跳过上传(需要覆盖请加 --rebuild)")
        return

    for sample in samples:
        # metadata 里除了判分必需的 relevant_ids,还带了 task_type / expected_route /
        # expected_filter —— 这些**不参与** Langfuse 侧的指标计算,但让界面里
        # 能按场景筛选、并核对"这条样本本来期望什么"。上传时多带元数据是划算的:
        # 之后再想补就得整个 --rebuild
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": sample["question"]},
            expected_output=sample["ground_truth"],
            metadata={
                "relevant_ids": sample["expected_ticket_ids"],
                "task_type": sample["task_type"],
                "expected_route": sample["expected_route"],
                "expected_filter": sample["expected_filter"],
            },
        )
    langfuse.flush()
    print(f"数据集 {dataset_name} 已上传 {len(samples)} 条样本")


def main() -> None:
    ap = argparse.ArgumentParser(description="Langfuse 评估数据集上传脚本")
    # 默认值是 False ⇒ 默认**不覆盖**(保护已有数据集:重复上传会让条目翻倍)
    ap.add_argument("--rebuild", action="store_true", help="删除同名数据集后重新上传")
    # 数据集名的默认值来自 .env(LANGFUSE_DATASET_NAME),与 langfuse_evaluation.py 读同一个配置 ——
    # 两边必须一致,否则会出现"上传到了 A、评估读的是 B"⇒ 评估报"数据集为空"
    ap.add_argument("--dataset-name", default=settings.langfuse_dataset_name)
    ap.add_argument("--eval-set", default=str(EVAL_SET_PATH))
    args = ap.parse_args()

    samples = load_eval_set(Path(args.eval_set))
    print(f"评估集 {args.eval_set}:{len(samples)} 条")
    upload_dataset(samples, args.dataset_name, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
