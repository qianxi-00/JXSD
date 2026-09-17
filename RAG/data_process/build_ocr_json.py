# =============================================================================
# OCR 第二步：Markdown → 入库 JSON（基础篇「OCR 清洗 / 结果处理」）
#
# 位置：data/output/<分类>/<stem>.md ──(本脚本)──> data/ocr_results{,_clean}.json
#       ──(tick_extract.py)──> Milvus
#
# 输出两个文件，职责不同，别混用：
#   ocr_results.json       —— 原样（未清洗）的识别文本，留档用；字段结构与课案一致。
#   ocr_results_clean.json —— 清洗后的文本 + cleaning 统计块，**下游 tick_extract.py 读的是这个**。
#   （准确说 tick_extract 只读 ocr_results_clean.json，见其 OCR_JSON_PATH 常量。）
#
# 清洗口径三条（沿用课案）：
#   1. 去 Markdown 版面标记（图片 div、标题符）——它们是版面元素，不是票面文字；
#   2. 压缩行内空白与连续空行——OCR 常在字段之间吐一堆空格/空行；
#   3. 去重默认 **consecutive**（只删连续重复行）。
#
# 为什么默认不是 all（这是本文件最容易踩错的一处）：
#   课案文档明确写了 `all`（全局去重）会误删发票表格里重复的商品/税率/金额行——
#   一张发票上"税率 6%"、"金额 100.00" 这类行本来就会重复出现，它们是**合法数据**，
#   全局去重会把第二行起全部删掉，抽出来的金额/明细就残缺了。
#   `all` 因此只作为可选项保留（`--dedupe all`），默认走 consecutive。
# =============================================================================

# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
# 目前本模块没有用到 config/RAG 包，但保留这段是为了与同目录其他脚本的调用方式一致：
# 既能 `python -m data_process.build_ocr_json`，也能直接按路径运行。
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""把 PaddleOCR 产出的 Markdown 汇总成入库用的 JSON(对照基础篇课案「OCR 清洗 / 结果处理」)。

补的是一条**断掉的链**:
    课案:  ocr_results.json → clean_ocr_json.py --dedupe consecutive → ocr_results_clean.json
    本项目:data/output/<分类>/<stem>.md  →(本脚本)→  data/ocr_results.json
                                             + data/ocr_results_clean.json →(tick_extract.py)→ Milvus

在此之前 `paddle_ocr.py` 只写 Markdown、`tick_extract.py` 只读 JSON,中间那一步没有任何脚本,
入库数据只能来自课案服务器上的另一次 DeepSeek-OCR-2 运行 —— 换机就复现不出来。

清洗口径沿用课案:
- 去掉 Markdown 图片/标题标记(版面元素,不是票面文字);
- 压缩空行与行内空白;
- `--dedupe` 默认 **consecutive**(只删连续重复行)。课案文档明确说 `all` 会误删发票表格里
  重复的商品/税率/金额行 —— 那正是要保留的字段。

用法:
    uv run python RAG/data_process/build_ocr_json.py
    uv run python RAG/data_process/build_ocr_json.py --dedupe all --categories invoice
"""

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "output"
# 两个输出路径都可以用命令行覆盖（--raw / --clean），默认落在 data/ 下。
# 名称沿用课案的 ocr_results.json / ocr_results_clean.json，方便与课案文档逐字段对照。
DEFAULT_RAW_PATH = DATA_DIR / "ocr_results.json"
DEFAULT_CLEAN_PATH = DATA_DIR / "ocr_results_clean.json"
# 与 paddle_ocr.DEFAULT_CATEGORIES 保持一致（也决定入库后的 ticket_type 取值）。
DEFAULT_CATEGORIES = ("flight", "invoice", "train")

# PaddleOCR-VL 的 Markdown 会把版面图片单独切成 <div ...><img ...></div>
# 三条正则的**执行顺序有讲究**，见 strip_markdown 的说明：
#   _IMG_DIV_RE 先整块吃掉 <div><img/></div>（带属性的 div、img 自闭合与不自闭合都覆盖）；
#   _IMG_TAG_RE 再兜住零散的 <img>（没被 div 包住的那些）；
#   _HEADING_RE 去掉行首 1~6 个 # 与后面的空白，只删标记、保留标题文字。
_IMG_DIV_RE = re.compile(r"<div[^>]*>\s*<img[^>]*/?>\s*</div>", re.I)
_IMG_TAG_RE = re.compile(r"<img[^>]*/?>", re.I)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)


def strip_markdown(text: str) -> str:
    """去掉 Markdown 版面标记,只留票面文字。

    - 整块的 `<div><img></div>` 与零散 `<img>` 直接删掉(它们是切出来的图片引用);
    - `## 发票` 这类标题符去掉,保留标题文字(票面上确实印着"发票"二字);
    - 压缩行内空白、把连续空行压成单个空行。
    """
    # 空文本短路：后面全是正则与按行处理，空串走完整流程只是浪费。
    if not text:
        return ""
    # 顺序必须是"先去 div 块、再去裸 img"：反过来先删 <img> 的话，
    # `<div><img/></div>` 会退化成 `<div></div>`，而 _IMG_DIV_RE 要求 div 里必须有 img，
    # 于是再也匹配不上，正文里就留下一串空的 <div></div>（后面 tick_extract 的
    # clean_semantic 虽会把标签清掉，但那是入库前的最后一道，这里的目标是尽早别产生垃圾）。
    out = _IMG_DIV_RE.sub("", text)
    out = _IMG_TAG_RE.sub("", out)
    # 标题符只删 `## ` 这几个字符，**不删标题文字**：票面上印着"发票""行程单"这类字样，
    # 它们是抽字段时的重要锚点（例如 extract_invoice 靠"发票编号""开票日期"定位）。
    out = _HEADING_RE.sub("", out)
    lines = []
    # 统一换行符：Windows 上 OCR 结果可能是 \r\n（写文件时又可能被再转换一次），
    # 先归一成 \n 再按行处理，否则 split("\n") 会留下行尾的 \r，
    # 表现为"看起来一样的行却不相等"，进而让去重与字段匹配静默失效。
    for line in out.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        # [ \t\u3000]+ → 单个半角空格：\u3000 是全角空格，中文票据的 OCR 结果里极常见，
        # 不处理的话"金额：100"这种行会因为夹着全角空格而匹配不上正则。
        line = re.sub(r"[ \t\u3000]+", " ", line).strip()
        lines.append(line)
    out = "\n".join(lines)
    # 3 个以上连续换行压成 2 个（即最多保留一个空行）。留一个空行而不是全删，
    # 是因为下游 tick_extract 的 extract_flight 用 next_non_empty 跳过空行取姓名，
    # 空行是它要处理的正常输入；这里只需要别让 OCR 吐出成片的空白。
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def dedupe_consecutive(text: str) -> tuple[str, int]:
    """删除**连续重复**的行,返回 (新文本, 删除行数)。

    只处理相邻重复:发票表格里隔几行又出现同样的"税率 6%"是合法数据,不能删。
    """
    if not text:
        return "", 0
    kept: list[str] = []
    removed = 0
    # previous 只在"行被保留"时才更新，所以判断的是**紧邻上一行**是否相同：
    # A A A 会删掉第 2、3 行；A B A 一行都不删（B 隔开了它们），这正是我们要的语义。
    previous: str | None = None
    for line in text.split("\n"):
        # `line and` 这个条件很关键：空行不进比较，也不会把 previous 更新成空串。
        # 否则"空行 + 空行"会被算成重复并被删掉，连续空行就无法保留，
        # 而 extract_flight 的空行跳过逻辑依赖空行的真实存在。
        if line and line == previous:
            removed += 1
            continue
        kept.append(line)
        previous = line
    return "\n".join(kept), removed


def dedupe_all(text: str) -> tuple[str, int]:
    """删除**全局重复**的行(课案的 --dedupe all,会误删表格合法重复行,仅作可选项保留)。"""
    if not text:
        return "", 0
    # seen 记录"出现过的非空行"。与 consecutive 的关键差别：删掉的判定不看相邻性，
    # 所以发票表格里第 2 次出现的同一行"金额 100.00"也会被删——那行往往属于**另一件商品**。
    # 只有确认整份文档是"纯重复噪声"（比如同一段文字被 OCR 吐了两遍）时才该用这个口径。
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for line in text.split("\n"):
        if line and line in seen:
            removed += 1
            continue
        if line:
            seen.add(line)
        # 注意看行不入 seen（空行不参与去重），所以空行会原样保留。
        kept.append(line)
    return "\n".join(kept), removed


def clean_text(text: str, dedupe: str = "consecutive") -> tuple[str, int]:
    """按 `dedupe` 口径清洗文本,返回 (清洗后文本, 删除的行数)。"""
    stripped = strip_markdown(text)
    # "none" 只是跳过去重，不跳过 strip_markdown —— 版面标记永远要清。
    if dedupe == "none":
        return stripped, 0
    if dedupe == "all":
        return dedupe_all(stripped)
    # 兜底走 consecutive（含传了未知值的情况）：默认口径是"最保守的那一个"，
    # 万一未来新增了 dedupe 选项而这里忘了加分支，行为退化到"只删连续重复"，
    # 也就是最不容易误删数据的一档，而不是静默退回 none 或 all。
    return dedupe_consecutive(stripped)


def build_records(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
) -> list[dict]:
    """把 data/output/<分类>/*.md 汇总成入库记录(字段与课案 ocr_results.json 对齐)。"""
    records: list[dict] = []
    for category in categories:
        category_dir = output_dir / category
        # 目录不存在就跳过整个分类（而不是报错）：允许只跑过部分分类，
        # 也允许 --categories 里带上还没识别的分类。
        if not category_dir.is_dir():
            continue
        # sorted 让记录顺序稳定可复现；glob("*.md") 不含子目录。
        for markdown_path in sorted(category_dir.glob("*.md")):
            # errors="replace"：OCR 出来的文件偶尔带非法字节（尤其是 Windows 上被别的
            # 工具重存过），直接 read_text 会 UnicodeDecodeError 让整批失败；
            # 替换成 U+FFFD 只坏一条记录，不影响其他票据。
            text = markdown_path.read_text(encoding="utf-8", errors="replace")
            stem = markdown_path.stem
            empty = not text.strip()
            records.append(
                {
                    # 坑：这两处路径是**按 stem 拼出来的**，且后缀写死 .png。
                    #   * 若原图是 .jpg/.jpeg/.bmp（paddle_ocr.IMAGE_EXTS 是允许的），
                    #     这里拼出的路径指向一个不存在的文件；
                    #   * 若这张图被 PaddleOCR 切成了多页，paddle_ocr 存的是 <stem>_1.md、
                    #     <stem>_2.md，于是 stem 变成 "<名字>_1"，
                    #     拼出的 data/<分类>/<名字>_1.png 同样不存在。
                    # 现在数据里是单页 + .png，所以没暴露；换图源时要一起改这里。
                    "image_path": f"data/{category}/{stem}.png",
                    "source_file": f"data/{category}/{stem}.png",
                    "source_dir": category,
                    "image_name": f"{stem}.png",
                    # ticket_type 直接取目录名——所以 data/ 下的子目录名就是业务取值，
                    # 改目录名等于改库里的 ticket_type（tick_extract 沿用这个字段选抽取器）。
                    "ticket_type": category,
                    # 这里的 ticket_no 只是个占位（用图片名），并不代表票据号：
                    # 真正的票据号由 tick_extract 的抽取器从正文里抽，抽不到就是 None，
                    # 而且 build_record 用的是 `**extract` 展开，会**覆盖**掉这个键。
                    "ticket_no": stem,
                    "ocr_text": text,
                    # 空文本标 failed：入库前能一眼看出哪些图没认出来。
                    # 这个标记不会阻断入库，只是给人和报表看的。
                    "ocr_status": "failed" if empty else "success",
                    # 引擎名写死、模型名留空：见下面 main() 里的说明（这是本文件的已知缺口）。
                    "ocr_engine": "paddleocr-cloud",
                    "ocr_model": "",
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="PaddleOCR Markdown → 入库 JSON")
    # 四个路径/口径全都能覆盖，方便在实验目录上跑而不用动 data/ 下的正式产物。
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--raw", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--clean", default=str(DEFAULT_CLEAN_PATH))
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--dedupe",
        choices=["consecutive", "all", "none"],
        default="consecutive",
        help="consecutive=只删连续重复行(课案默认,推荐);all=删全局重复行(会误删表格合法重复)",
    )
    args = parser.parse_args()

    records = build_records(Path(args.output_dir), tuple(args.categories))
    # 没有任何 .md 说明前置步骤没跑（或 --output-dir 指错了）。
    # 这里用 SystemExit 直接终止并给出可操作的提示，而不是写出一个空 JSON ——
    # 空 JSON 会让下游 tick_extract 报"提取到 0 条"，掩盖真正的原因。
    if not records:
        raise SystemExit(f"{args.output_dir} 下没有找到任何 <分类>/*.md —— 先跑 paddle_ocr.py")

    changed = 0
    removed_total = 0
    # 注意这里是**原地改写** records：清洗结果直接写回 record["ocr_text"]。
    # 这正是后面要重新调一次 build_records() 拿"原始文本"的原因（见下方的说明）。
    for record in records:
        cleaned, removed = clean_text(record["ocr_text"], args.dedupe)
        removed_total += removed
        # 与 strip() 后的原文比较：只有真正的空白差异不算"改动"，
        # 这样 changed 才代表"清洗真的删掉了东西"，而不是"末尾多了个换行"。
        if cleaned != record["ocr_text"].strip():
            changed += 1
        record["ocr_text"] = cleaned

    success = sum(1 for r in records if r["ocr_status"] == "success")
    # cleaning 统计块只写在 clean 文件里，用来回答"这次清洗到底动了多少"——
    # 换 dedupe 口径、调 strip 规则后，对比这个块就能看出影响面。
    cleaning = {
        "script": Path(__file__).name,
        "dedupe": args.dedupe,
        "records": len(records),
        "success": success,
        "failed": len(records) - success,
        "changed": changed,
        "removed_lines": removed_total,
    }
    # 取第一条的引擎名当整份文件的引擎名。要是将来混用了多个引擎，
    # 这个字段就不再成立——现在的数据是单一来源，所以可以这么写。
    engine = records[0]["ocr_engine"]

    for path, payload in (
        # 关键点：raw 这一份**重新调用了一次 build_records()**。
        # 不能复用上面的 records —— 它已经被清洗过（原地改写），
        # 直接拿来写 raw 就会把"原始识别文本"也写成清洗后的版本，raw 文件失去留档意义。
        # 代价是重新读一遍所有 .md（IO 翻倍），换来 raw/clean 两份语义清晰。
        # 也正因如此，raw 反映的是"本次运行时刻磁盘上的 Markdown"，
        # 而不是"第一次识别时的输出"——Markdown 若被改过，raw 也会跟着变。
        (Path(args.raw), {"ocr_engine": engine, "results": build_records(Path(args.output_dir), tuple(args.categories))}),
        # clean 这一份带 cleaning 统计块；tick_extract 读的就是这个文件（只取 results 字段）。
        (Path(args.clean), {"ocr_engine": engine, "results": records, "cleaning": cleaning}),
    ):
        # parents=True：--raw/--clean 可能指向还不存在的目录。
        path.parent.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=False：保留中文原文（票据字段全是中文，转义成 \uXXXX 后没法用肉眼核对）。
        # indent=2：这两个文件是给人看/给下游读的，格式化后便于 diff。
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # 打印 len(records) 而不是重算后的条数：两份的条数必然相同，
        # 复用已有变量避免多读一次（raw 那次重算虽然也读了一遍文件，但条数不会变）。
        print(f"写出 {len(records)} 条 -> {path}")

    print(
        f"清洗({args.dedupe}): 处理 {len(records)} 条,改动 {changed} 条,"
        f"删除重复行 {removed_total} 行,成功 {success} / 失败 {len(records) - success}"
    )


if __name__ == "__main__":
    main()
