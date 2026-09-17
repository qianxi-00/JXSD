"""从 data/ocr_results_clean.json 提取三类票据字段,并可批量写入 Milvus tick 集合

用法:
  uv run python -m data_process.tick_extract                 # 干跑:统计 + 样本报告(data/tick_extract_report.json)
  uv run python -m data_process.tick_extract --type invoice --limit 5
  uv run python -m data_process.tick_extract --insert        # 建 tick 集合并全量入库
  uv run python -m data_process.tick_extract --insert --recreate
  uv run python -m data_process.tick_extract --save data/tick_extracted.jsonl
"""

# =============================================================================
# OCR 第三步：入库 JSON → 票据字段 → Milvus（基础篇「字段抽取 / 入库」）
#
# 位置：data/ocr_results_clean.json ──(本脚本)──> 字段抽取结果
#                                              ──(--insert 时)──> Milvus 的 tick 集合
#       （不写库时只产出报告与可选的 jsonl，见下面的"默认干跑"）
# 约定：**票据字段抽取的唯一实现就是本文件**。别处（脚本、测试、评估）要用字段，
#       一律 import 这里的 extract_* / build_record，不要再抄一份正则——
#       抄一份的代价是"某个字段的抽法改好了，另一处还是老逻辑"，而两边都能跑，
#       只会表现为"同一张票在不同报表里金额不同"这种极难定位的不一致。
#
# 三条贯穿全文件的取值约定（下面每个抽取器都遵守）：
#   1. 抽不到就是 **None**，绝不落成 0 / ""。0 分会和"这张票金额真是 0"混淆，
#      而且在 Milvus 上会让 `amount_fen >= 100` 之类的过滤条件误命中"没抽到金额"的票。
#      （两个刻意的例外：extract_invoice 把 route 设成 ""、extract_train 的 counterparty
#        保持 None —— 前者表示"发票这类票据没有行程概念"，见对应函数说明。）
#   2. 金额统一存**分**（整数），日期统一存 **date_int**（如 20250305 的整数）。
#      这样 Milvus 的范围过滤与 SQL 的数值比较才能直接用，不需要在查询侧做格式转换。
#   3. 主键 id 由 source_file 派生（见 ticket_id），不用票据号——票据号常常抽不到，
#      而 source_file 必然存在。
#
# 默认是**干跑**（只统计 + 出报告），加 --insert 才写 Milvus：
# 字段抽取质量要先用报告（coverage 覆盖率 + samples 样本）确认过再入库。
# 命令行参数的完整用法见紧随其后的模块说明字符串。
#
# 幂等性（实测确认，别再退回 insert）：写库用的是 **upsert**，同一张票（主键 id 由
# source_file 派生）重跑只会覆盖、不会多出一行。为什么强调这一点——旧实现用 insert，
# 在一次性集合上实测"同主键插两次 = 两行"，重跑一次语料就翻倍（检索里同一张票出现两遍、
# 评估分母也跟着错）。改用 upsert 的同时还要**先读回旧向量**，因为 upsert 是整实体覆盖，
# 而本脚本产出的 vec 是 None（向量由 embed_tickets 单独回填）—— 见 `_carry_over_vectors`。
# 注意 `--insert` 这个参数名保留不变（文档与脚本都在用），它的语义现在是"写入/覆盖"。
# =============================================================================

# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
# 必须在 from config import settings 之前：直接按脚本路径运行时
# sys.path[0] 是 RAG/data_process，看不到仓库根的 config。
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）


import argparse
import hashlib
import json
import re
import sys
import time
from decimal import Decimal
from pathlib import Path

# 第二道导入兜底：本脚本会被以两种方式调用（-m 模块 / 直接跑文件），
# 而且 __init__.py 为空，包语义很弱。这里再按"本文件位置"补一次 RAG 根，
# 让 `from config import settings` 在两种调用方式下都能成立。
try:
    from config import settings
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 只读 clean 文件：未清洗的 ocr_results.json 里带着 Markdown 图片标记与重复空行，
# 那些噪声会干扰正则（例如标题行把"姓名"顶到行首之外）。
OCR_JSON_PATH = PROJECT_ROOT / "data" / "ocr_results_clean.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "tick_extract_report.json"

# 六个抽取字段的清单，被三处共用：
#   * 每个 extract_* 用它初始化"全 None 的结果字典"（dict.fromkeys(FIELDS)）；
#   * build_report 按它算每个字段的覆盖率；
#   * 与其他文件的字段名保持一致（filters.py / rerank / text_to_sql.Ticket 同名字段）。
# 注意 FIELDS 用的是 **counterparty**（对方单位，发票=卖方、火车票=null），
# 与 Milvus 建表时的字段名一致；改名要同时改 milvus_init / embed_tickets / 建表 schema。
FIELDS = ("ticket_no", "person", "date_int", "amount_fen", "route", "counterparty")
# 中文人名：2~4 个汉字。这里只描述**形状**，不承担语义排除——
# "机场""车站"这类词同样符合这个形状，排除逻辑统一放在 is_person_name 一处，
# 这样"哪些词不算人名"只有一个可改、可测的地方（测试里直接断言 is_person_name）。
CN_NAME = r"[\u4e00-\u9fa5]{2,4}"
# 机场名在 OCR 结果里常被识别成拼音大写 + JICHANG（"XX机场"的拼音），
# 允许中间有空格是因为中英混排时会出现 "BEIJING JICHANG" 这种带空格的写法。
# 末尾不带 \b，是因为后面接的常常是换行或中文，加 \b 反而会漏。
AIRPORT_RE = r"[A-Z][A-Z ]*JICHANG"
# 火车票日期：写成"2025 年 3 月 5 日"，年月日之间允许任意空白（OCR 会插空格）。
TRAIN_DATE_RE = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
# 火车票金额：必须是"￥/¥ + 数字 + 元"这个完整形态。
# 要求 元 结尾是为了避开票面上的其他数字（车次号、座位号、证件号后四位等）。
TRAIN_MONEY_RE = r"[￥¥]\s*([0-9,]+(?:\.[0-9]+)?)\s*元"
# 发票金额：靠"总金额"这个**文字标签**定位，且要求两位小数（发票金额固定两位）。
# 用文字标签而不是"找最大的数"，是因为发票上还有税额、单价等一堆数字，
# 靠数值大小猜会取错（价税合计往往才是要的那个数，但它不一定最大）。
INVOICE_AMOUNT_RE = r"总金额\s*([0-9,]+\.[0-9]{2})"

# 票面上的中文航司名（"中国国际航空(公司)" / "海南航空" …）。允许"公司"后缀可选。
# 只描述形状，"这句话到底属于哪家"交给 _canonical_airline 的别名表，与 CN_NAME 同一分工。
CN_AIRLINE_RE = re.compile(r"([\u4e00-\u9fa5]{2,8}航空)(?:公司)?")
# 航班号：两个字母/数字 + 3~4 位数字（`ZH9146` / `CA 1234`）。OCR 会在中间插空格，
# 所以两段之间允许一个空白；末尾要 \b，否则会把长数字串的前 4 位当成航班号。
FLIGHT_NO_RE = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{2})\s?(\d{3,4})(?![0-9])")

# IATA 两字码 → 规范中文名（不带"公司"后缀，全表口径一致）。
# 只收**实测在票面上出现过**的码 + 常见航司；未收录的码一律保持 None，不猜。
# 实测出现过的：CZ 15 / CA 15 / HU 7 / SC 7 / MF 6 / MU 5 / JD 5 / ZH 1 / G5 2 / 8L 2
AIRLINE_BY_IATA = {
    "CA": "中国国际航空",
    "CZ": "中国南方航空",
    "MU": "中国东方航空",
    "HU": "海南航空",
    "SC": "山东航空",
    "MF": "厦门航空",
    "JD": "首都航空",
    "ZH": "深圳航空",
    "G5": "华夏航空",
    "8L": "云南祥鹏航空",
    "3U": "四川航空",
    "GS": "天津航空",
    "9C": "春秋航空",
    "HO": "吉祥航空",
    "KN": "中国联合航空",
    "PN": "西部航空",
    "FM": "上海航空",
    "EU": "成都航空",
    "NS": "河北航空",
    "TV": "西藏航空",
}
# 别名 → 规范名。中文全称/简称、英文名都要能落到同一个值上，
# 否则按 counterparty 分组统计会把同一家航司拆成几条。
AIRLINE_ALIASES = {
    "中国国际航空": "中国国际航空",
    "中国国际航空公司": "中国国际航空",
    "国航": "中国国际航空",
    "AIR CHINA": "中国国际航空",
    "中国南方航空": "中国南方航空",
    "中国南方航空公司": "中国南方航空",
    "南航": "中国南方航空",
    "CHINA SOUTHERN": "中国南方航空",
    "中国东方航空": "中国东方航空",
    "中国东方航空公司": "中国东方航空",
    "东航": "中国东方航空",
    "CHINA EASTERN": "中国东方航空",
    "海南航空": "海南航空",
    "海南航空公司": "海南航空",
    "海航": "海南航空",
    "Hainan Airlines": "海南航空",
}


def ticket_id(source_file: str) -> str:
    """根据 source_file 的 SHA-256 前 24 位生成稳定 ID"""
    # 为什么要哈希而不是直接用路径当主键：Milvus 的 VARCHAR 主键有长度上限（建表是 64），
    # 长路径/中文路径容易超；哈希后定长、纯十六进制，也顺带把中文与非 ASCII 归一掉了。
    # 24 个十六进制字符 = 96 bit，碰撞概率可以忽略。
    # 坑：id 完全由 source_file 的**字符串**决定，所以改路径前缀、把 .png 换成 .jpg、
    # 或者换一台机器导致相对路径不同，都会算出一个新 id ——重新入库就变成**重复行**
    # （旧 id 还在库里），而不是更新。要改命名口径就得重建集合（--recreate）。
    return "ticket_" + hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:24]


def yuan_to_fen(value: str) -> int:
    # 用 Decimal 而不是 float：金额是十进制数，float 是二进制浮点，乘 100 会出现
    # 表示误差——实测 float("1.005") * 100 == 100.49999999999999、float("8.165") * 100
    # == 816.4999999999999，取整后就少 1 分。Decimal 是精确十进制，
    # 配合先剔掉千分位逗号（"1,234.00"）就能算准。
    # 注意 to_integral_value() 用的是 decimal 默认的 **ROUND_HALF_EVEN**（银行家舍入）：
    # 恰好落在半分位时 .125 会舍成 .12 而不是 .13。票据金额都是两位小数，
    # 乘 100 后没有小数部分，正常数据不会触发；真遇到三位小数的脏数据时要显式指定舍入方式。
    return int((Decimal(value.replace(",", "")) * 100).to_integral_value())


def clean_semantic(text: str) -> str:
    """整理检索文本:去掉 HTML 标签、换行、多余空白等影响理解的符号"""
    # semantic_text 是**唯一**用于向量化与重排的文本（embed_tickets 嵌入它、
    # rerank._request 把它发给接口），所以它的整洁程度直接决定召回质量：
    #   * 留着 <div>/<img> 之类的标签 → 这些标记会占掉 embedding 的语义权重；
    #   * 留着换行 → 同一张票的关键词被切到两行，向量会略微跑偏（影响不大但没必要）；
    #   * 连续空格 → 白白撑长文本长度，还可能触发接口的长度截断。
    # 这里不去标点、不做分词：中文标点对 bge-m3 是有用的语义信号。
    t = re.sub(r"<[^>]+>", " ", text)
    # \u3000 是全角空格，中文 OCR 里大量出现；\t/\r/\n 统一压成半角空格。
    t = re.sub(r"[\r\n\t\u3000]+", " ", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


def is_person_name(s: str) -> bool:
    # 全匹配 2~4 个汉字，再排除以"机场""站"结尾的词：
    # 它们本身也符合"2~4 个汉字"，是最容易被人名规则误吞的一类（"虹桥机场""北京南站"）。
    # 注意这里只排除"结尾"，不能排除"包含"——"张家界站"这类地名与人名同形，只能靠上下文。
    return bool(re.fullmatch(CN_NAME, s)) and not s.endswith(("机场", "站"))


def _canonical_airline(name: str) -> str:
    """把票面上的航司写法归一到规范名（`中国国际航空公司` / `国航` → `中国国际航空`）。

    查不到别名表的（冷门航司、OCR 生造写法）**原样返回但去掉"公司"后缀** ——
    不硬塞进某个规范名，也不能返回空：它确实是票面上写着的中文航司名。
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return ""
    if cleaned in AIRLINE_ALIASES:
        return AIRLINE_ALIASES[cleaned]
    return re.sub(r"公司$", "", cleaned)


def _airline_from_text(text: str) -> str | None:
    """票面上**写着**航司名时的抽法，两级：

    ① 别名表里找（含 `国航`/`南航`/`AIR CHINA` 这类不含"航空"两字的简称与英文名）——
       按别名长度从长到短扫，避免 `国航` 先命中而把 `中国国际航空公司` 切一半；
       比较时两边都转大写，OCR 的大小写不可靠；
    ② 通用中文航司名形状 `XX航空(公司)`：别名表里没有的冷门航司靠这一级兜住，
       如实返回（去掉"公司"后缀），不硬塞进某个规范名。
    """
    upper = (text or "").upper()
    for name in sorted(AIRLINE_ALIASES, key=len, reverse=True):
        if name.upper() in upper:
            return AIRLINE_ALIASES[name]
    match = CN_AIRLINE_RE.search(text or "")
    return _canonical_airline(match.group(1)) if match else None


def _airline_from_flight_no(text: str) -> str | None:
    """票面没写航司名时，从航班号的 IATA 两字码反查；查不到返回 None。

    OCR 会把字母认错（实测遇到 `H0`，真码是 `HO`），所以在原码不命中时做一次
    **数字→字母**的等价替换再查（只换 `0→O`、`1→I`；不反向替换，因为 IATA 码里
    数字是有意义的，见 `3U`/`9C`/`8L`/`G5`）。查不到就返回 None —— 宁可不填，
    也不要凭长相猜一家航司（错填的承运方会静默进 SQL 过滤与分组统计）。
    """
    for code, _digits in FLIGHT_NO_RE.findall(text or ""):
        if code in AIRLINE_BY_IATA:
            return AIRLINE_BY_IATA[code]
        swapped = code.replace("0", "O").replace("1", "I")
        if swapped in AIRLINE_BY_IATA:
            return AIRLINE_BY_IATA[swapped]
    return None


def extract_flight(text: str) -> dict:
    """登机牌:只有 Jan01 无年份与金额,date/amount 恒为 null

    为什么 date/amount 是"恒为 null"而不是"没抽到就不填"：登机牌版面上只有
    "Jan01" 这样的**月日**（没有年份），也没有票价——它们本来就不存在于这份凭证上。
    所以这里刻意不去猜年份、更不去凑金额：**猜出来的日期比没有日期更危险**，
    因为它会静默进入 date_int 过滤（"2025 年的机票"就会把猜错的票算进来）。
    这三类票据里只有登机牌是这种"字段天生缺失"的情况。
    """
    # dict.fromkeys(FIELDS) 生成 {字段: None} 的骨架——这就是"抽不到就是 None"的落点。
    # 千万别改成 dict.fromkeys(FIELDS, 0) 之类：0 分会与真实的 0 元票混淆。
    out = dict.fromkeys(FIELDS)
    if not text:
        return out

    # 电子客票号：ETKT 后面的 6 位以上数字。这里的 \s* 允许"ETKT 123456"这种带空格的写法
    # （OCR 有时把标签与号码分开）。{6,} 只做长度下限、不做位数上限，
    # 因为不同航司的票号长度并不一致。
    m = re.search(r"ETKT\s*(\d{6,})", text)
    if m:
        out["ticket_no"] = m.group(1)

    # 先按行切开并 strip：下面所有定位（姓名、机场）都建立在"行"这个粒度上，
    # 且 strip 掉行首尾空白能让 fullmatch / ^...$ 之类的判断不被空格干扰。
    lines = [ln.strip() for ln in text.split("\n")]

    def next_non_empty(start: int) -> str | None:
        """往后找第一个非空行。

        不同 OCR 引擎的版面差异很大：同一张登机牌，DeepSeek-OCR-2 把中文名紧跟在下一行，
        PaddleOCR-VL 则在英文名与中文名之间插空行。以前只按固定偏移取行，
        空行一插就把姓名漏成 null（实测 7 张登机牌全中招）。

        所以这个函数是"按**内容**跳行"而不是"按固定偏移取行"：偏移法假设版面固定，
        而版面恰恰是换 OCR 引擎/换图源时最先变的东西。凡是"标签在上、值在下面某一行"的字段，
        都该用这种跳空行的取法。返回 None 表示后面全是空行（已经到底）。
        """
        for index in range(start, len(lines)):
            if lines[index]:
                return lines[index]
        return None

    # 姓名是登机牌上最难抽的字段：同一张票在不同引擎下会出现三种版面，
    # 下面三个分支正好对应它们。循环里遇到第一个能确定的就 break——
    # 一张登机牌只有一个"姓名"标签，不做"多行投票"。
    for i, line in enumerate(lines):
        if "姓名" not in line:
            continue
        # 冒号可能没有；split(":", 1) 只切第一个冒号，
        # 防止值里本身带冒号（"姓名: ZHANG/SAN" 这种写法）时被切碎。
        rest = line.split(":", 1)[1].strip() if ":" in line else ""
        cand = None
        if rest:
            # 版面一：`姓名: ZHANG SAN` + 下一行中文名（中文名常被 OCR 挤到下一行）。
            # [A-Z][A-Z ]{1,15} 不含点与斜杠，所以 "ZHANG/SAN" 不会命中这个分支，
            # 会落到下面的 elif 被整体当值处理。
            if re.fullmatch(r"[A-Z][A-Z ]{1,15}", rest):
                following = next_non_empty(i + 1)
                if following and is_person_name(following):
                    cand = following
            elif is_person_name(rest):
                # 版面二：中文名就在同一行（`姓名: 张三`）——最省事的情况。
                cand = rest
        else:
            # 版面三：标签独占一行，值在下面（`姓名` / `ZHANG SAN` / `张三`）。
            # 这里用 next_non_empty(i+1) 与 next_non_empty(i+2)——注意 i+2 是
            # "再往后一个非空行"，不是"第 i+2 行"，所以中间插了几个空行都还能对上。
            # 前提是英文名只占**一行**；若英文名被 OCR 拆成两行，i+2 会取到英文的第二段，
            # is_person_name 判断失败 → cand 保持 None → 交给下面的兜底正则。
            english = next_non_empty(i + 1)
            chinese = next_non_empty(i + 2) if english else None
            if (
                english
                and chinese
                and re.fullmatch(r"[A-Z][A-Z ]{1,15}", english)
                and is_person_name(chinese)
            ):
                cand = chinese
        if cand:
            out["person"] = cand
            break
    if out["person"] is None:
        # 兜底：拼音名与中文名之间可能有空行
        # 这一条不依赖"姓名"标签，直接在全文中找"全大写拼音行 + 若干换行 + 中文名"的形状。
        # re.M 让 ^/$ 按行生效；\n+ 同时覆盖"紧邻换行"与"中间夹空行"两种版面
        # （空行在文本里也是 \n，所以 \n+ 天然吃得下）。
        # 只在上面按标签抽取失败时才跑——它比标签法更容易误抽（例如同行两位乘客）。
        for m in re.finditer(rf"^([A-Z][A-Z ]{{1,15}})\n+({CN_NAME})$", text, re.M):
            if is_person_name(m.group(2)):
                out["person"] = m.group(2)
                break

    # 行程：首选按"自 From / 至 To"两个标签配对取值，这样出发地/到达地的顺序是确定的。
    # AIRPORT_RE = [A-Z][A-Z ]*JICHANG，即拼音大写 + JICHANG（"XX机场"被 OCR 成拼音）。
    # [^\n]*? 是懒惰匹配：允许标签与机场名之间有其他文字（航班号、航站楼），
    # 但不会跨行去别的字段里找。
    from_m = re.search(rf"自\s*From[^\n]*?({AIRPORT_RE})", text)
    to_m = re.search(rf"至\s*To[^\n]*?({AIRPORT_RE})", text)
    if from_m and to_m:
        # 去掉空格拼成 "出发-到达"，**顺序必须保持 from 在前**：
        # 下游 filters.build_milvus_filter 用的是 `route like "出发%到达%"`，
        # 顺序反了会让"从北京到上海"一条票都查不出来。
        out["route"] = f"{from_m.group(1).replace(' ', '')}-{to_m.group(1).replace(' ', '')}"
    else:
        # 标签不全时的兜底：把全文所有机场名按出现顺序收集，用 dict.fromkeys 去重
        # （保留首次出现的顺序，比 set 更可控），再取**第一个和最后一个**当出发/到达。
        # 这依赖"票面上出发地先出现、到达地后出现"的版面习惯——是个启发式，
        # 只在标签法失败时才用。
        airports = list(dict.fromkeys(a.replace(" ", "") for a in re.findall(AIRPORT_RE, text)))
        if len(airports) >= 2:
            out["route"] = f"{airports[0]}-{airports[-1]}"

    # 承运方：先中文名、再"航班号的 IATA 两字码"。
    #
    # 为什么必须加 IATA 这一路（真机数据说的）：100 张机票里只有 17 张能在票面上找到
    # "中国国际航空公司"这种中文全称，其余 83 张**一个中文航司名都没有** —— 票面只写航班号
    # （如 `ZH9146 Jan01 G`）。实测那 65 张非空 OCR 的缺口票里，航司信号**只有**两字码：
    # CZ 15 / CA 15 / HU 7 / SC 7 / MF 6 / MU 5 / JD 5 / ZH 1。所以按码查表是这里唯一可行的做法。
    #
    # 归一化：命中后一律写表里的**规范名**（不带"公司"后缀），不写票面原文 ——
    # 同一家航司在不同票上会写成"中国国际航空/中国国际航空公司/国航"，不归一的话
    # 按 counterparty 分组统计会分裂成好几条。
    # 顺序：票面写明航司名 > 航班号两字码（后者只是"票面没写"时的替代信号）。
    out["counterparty"] = _airline_from_text(text) or _airline_from_flight_no(text)
    return out


def _company_after(lines: list[str], label: str):
    # 发票上的"卖方/客户"都是以**独占一行的标签**开头，单位名在下一行。
    # 这里要求 line == label（全等而不是包含）：发票正文里也可能出现"卖方"字样
    # （比如备注栏），用包含匹配会取到错的那一行。
    for i, line in enumerate(lines):
        if line == label:
            # label 出现在最后一行时没有"下一行"，直接返回 None，避免 IndexError。
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                # 三个条件都在排除"下一行其实不是单位名"的情况：
                #   * 必须含汉字——单位名是中文，纯 ASCII 的多半是编号/税号；
                #   * 不含 "<"——下一行可能是被 OCR 拆出来的 HTML/版面片段（如 <div）；
                #   * 不能是"纯数字与空白"——那是税号、电话、账号。
                # 任一不满足就返回 None（而不是继续往后找）：发票版面上标签后面
                # 紧跟的就是单位名，继续往后找会跨到无关段落里去。
                if re.search(r"[\u4e00-\u9fa5]", nxt) and "<" not in nxt and not re.fullmatch(r"[0-9\s]+", nxt):
                    return nxt
            return None
    return None


def extract_invoice(text: str) -> dict:
    """发票:route 固定为空字符串,date 取开票日期,amount 取总金额

    route 用 "" 而不是 None 是刻意的：发票这类票据**没有行程概念**，
    "" 表达的是"该字段对此类票据无意义"，与 None 的"有这个概念但没抽到"区分开。
    下游 filters.build_milvus_filter 只在 route_from/route_to 同时存在时才加 route 条件，
    所以 "" 不会被当成过滤条件；但按 route 做分组统计时要注意这里多了一类空值。
    """
    out = dict.fromkeys(FIELDS)
    out["route"] = ""
    if not text:
        return out

    # 发票编号：标签后允许中英文冒号（：/:），取值限字母数字与连字符——
    # 这样遇到"发票编号：12345 开票日期：..."挤在一行的版面时不会把日期一起吞进来。
    #
    # 标签认两种写法：票面上"发票编号"与"发票号码"都常见（真实电子发票多用"发票号码"）。
    # ⚠ 诚实说明覆盖情况：本仓这 100 张发票**全部**写的是"发票编号"（实测 100/100 抽到票号），
    # 所以"发票号码"这一支**没有真实样本覆盖**，目前只有单测覆盖（见
    # `test_invoice_number_label_variants`）。加它是为了换一批真发票时不至于整列抽空 ——
    # 那属于"静默少一批字段"，比抽错更难发现。
    m = re.search(r"发票(?:编号|号码)[：:]\s*([A-Za-z0-9\-]+)", text)
    if m:
        out["ticket_no"] = m.group(1)

    # 日期标签有两种写法（开票日期 / 开具日期），两种都认。
    # 格式是 ISO 风格的 2025-03-05，与火车票的"2025 年 3 月 5 日"不同，所以两者不能共用一个正则。
    m = re.search(r"(?:开票日期|开具日期)[：:]\s*(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        y, mo, d = m.groups()
        # 统一算成 YYYYMMDD 整数（20250305）而不是存日期字符串：
        # 整数在 Milvus 的范围过滤与 PostgreSQL 的数值比较里都能直接用，且按数值序排列。
        out["date_int"] = int(y) * 10000 + int(mo) * 100 + int(d)

    # 金额走"总金额"文字标签 + 两位小数（见 INVOICE_AMOUNT_RE 的说明）。
    m = re.search(INVOICE_AMOUNT_RE, text)
    if m:
        out["amount_fen"] = yuan_to_fen(m.group(1))

    lines = [ln.strip() for ln in text.split("\n")]
    # 发票的"对方单位"是卖方，"人"是客户：字段名是从票据语义映射过来的
    # ——person 在火车票/机票里是乘客，在发票里就是买方（客户）。
    # 三种票据共用同一套字段，跨票据的按人统计才能成立。
    out["counterparty"] = _company_after(lines, "卖方")
    out["person"] = _company_after(lines, "客户")
    return out


def extract_train(text: str) -> dict:
    """火车票:counterparty 固定为 null,person 取证件号后的姓名

    counterparty 保持 None（不是 ""）：火车票是铁路系统的凭证，票面上**没有交易对方**，
    None 表达"此类票据不存在该字段"。这与发票把 route 设成 "" 是同一种"语义空值"处理，
    只是方向相反——具体哪个字段用哪种表示，看的是"这类票据有没有这个概念"。
    """
    out = dict.fromkeys(FIELDS)
    if not text:
        return out

    # 票号：字母 T 开头的 12~16 位数字。\b 两侧限界，避免把长数字串的中间一段当成票号。
    # 捕获组只抓数字、返回时再补回 "T" 前缀——因为 \b 与 T 在原文里可能被 OCR 插空格
    # （"T 1234..."），补前缀比在正则里容纳空格更可控。
    m = re.search(r"\bT(\d{12,16})\b", text)
    if m:
        out["ticket_no"] = "T" + m.group(1)

    m = re.search(TRAIN_DATE_RE, text)
    if m:
        y, mo, d = m.groups()
        # 与发票相同的 YYYYMMDD 整数口径（见 extract_invoice 的说明）。
        out["date_int"] = int(y) * 10000 + int(mo) * 100 + int(d)

    m = re.search(TRAIN_MONEY_RE, text)
    if m:
        out["amount_fen"] = yuan_to_fen(m.group(1))

    # 姓名：靠**身份证号**定位——"6~10 位数字 + **** + 3~4 位数字 + 校验位(0-9 或 X)"，
    # 掩码后的证件号后面紧跟的就是姓名。用证件号当锚点而不是找"姓名"标签，
    # 是因为火车票版面上"证件号与姓名"的相对位置比文字标签稳定得多
    # （标签常被 OCR 打散或与相邻字粘连）。
    m = re.search(r"[0-9]{6,10}\*{4}[0-9]{3,4}[0-9X]\s*(" + CN_NAME + r")", text)
    if m:
        out["person"] = m.group(1)

    # 车站：中文站名在 OCR 结果里常被认成拼音（"北京南" → "Beijingnan"），
    # 所以按 [A-Z][a-z]{2,} 这种"首字母大写 + 小写"的形状去认，两条路径：
    #   1. 整行就是一个拼音站名；
    #   2. 行尾是"站 <拼音>"（"北京南站 Beijingnan" 这种中英并列的版面）。
    # 两条都收集进 stations，最后去重取首尾——与登机牌路线同一套启发式（先出现=出发）。
    stations = []
    for line in text.split("\n"):
        line = line.strip()
        if re.fullmatch(r"[A-Z][a-z]{2,}", line):
            stations.append(line)
            continue
        m2 = re.search(r"站\s+([A-Z][a-z]{2,})\s*$", line)
        if m2:
            stations.append(m2.group(1))
    stations = list(dict.fromkeys(stations))
    # 至少两个站才能拼出"出发-到达"；只有一个时不填（而不是留下"XX-"这种半成品）。
    if len(stations) >= 2:
        out["route"] = f"{stations[0]}-{stations[-1]}"
    return out


# 票据类型 → 抽取器的分派表。key 就是 data/ 下的目录名（也就是 ticket_type）。
# 新增票据类型时：这里加一条 + 上面写一个 extract_xxx + 确认字段抽法齐备。
EXTRACTORS = {"flight": extract_flight, "invoice": extract_invoice, "train": extract_train}


def build_record(item: dict) -> dict:
    # 一条 ocr_results_clean.json 记录 → 一条 Milvus 实体。字段分三层：
    #   1. 派生字段：id（由 source_file 哈希而来，见 ticket_id）；
    #   2. 抽取字段：`**extract` 把六个字段平铺进来（ticket_no/person/date_int/amount_fen/
    #      route/counterparty），值可能是 None——**故意保留 None 而不是补默认值**，
    #      建表时这些列都是 nullable=True，正是为了让"没抽到"如实落库；
    #   3. 原文与检索文本：ocr_text（给人看、给 read_file 读）与 semantic_text（给向量化）。
    source_file = item.get("source_file", "")
    ticket_type = item.get("ticket_type", "")
    # `or ""` 而不是 `or item.get("ticket_no")`：空文本时不要去回退到文件里那个占位票号
    # （build_ocr_json 写的 ticket_no 其实就是图片名），否则会把"图片名"当成真实票据号入库。
    text = item.get("ocr_text") or ""
    # 用分派表取抽取器；遇到未知 ticket_type 时给一个"全 None"的兜底函数，
    # 保证本条仍然能入库（id/正文/向量都在，只是没有结构化字段），
    # 而不是整批因为一条未知类型就崩掉。
    extract = EXTRACTORS.get(ticket_type, lambda _t: dict.fromkeys(FIELDS))(text)
    return {
        "id": ticket_id(source_file),
        "ticket_type": ticket_type,
        **extract,
        # semantic_text 是清洗后的同一段文本（见 clean_semantic），embed_tickets 会嵌入它。
        "semantic_text": clean_semantic(text),
        "ocr_text": text,
        "source_file": source_file,
        # vec 先留 None：向量由 embed_tickets.py 单独回填。
        # 分两步是为了"换 embedding 模型时只重跑向量、不动字段"，见该脚本的说明。
        "vec": None,
    }


def _has_vector(value) -> bool:
    """判断一个向量的取值是不是"真有向量"。

    不用 `if value`：Milvus 返回的向量可能是 numpy 数组，而 numpy 对多元素数组做真值
    判断会**直接抛 ValueError**（"truth value of an array is ambiguous"）。用长度判断更稳。
    """
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:  # 不是序列（理论上不会出现）时按"有值"处理
        return True


def _merge_existing_vectors(batch: list[dict], existing: dict) -> int:
    """把库里已有的向量填回 batch（纯函数，便于单测）。返回填了几条。

    只填 `vec` 为空的行：已有非空向量的行本来就该被这次写覆盖（那是 embed_tickets
    刚算好的结果，不该被旧值顶掉）。
    """
    filled = 0
    for row in batch:
        if not _has_vector(row.get("vec")):
            old = existing.get(str(row.get("id")))
            if _has_vector(old):
                row["vec"] = old
                filled += 1
    return filled


def _carry_over_vectors(client, name: str, batch: list[dict]) -> int:
    """upsert 前把同一批 id 的**已有向量**读回来填进 batch。

    为什么必须做：Milvus 的 upsert 是**整实体覆盖**（先标记删除、再插新版本），而
    `build_record` 产出的 `vec` 是 None（向量由 embed_tickets 单独回填）⇒ 直接 upsert
    会把已经算好的向量抹成 NULL。后果很隐蔽：结构化查询仍查得到这些票，
    语义检索却再也搜不到 —— 实测探针（`.dsh_tmp/probe_upsert_semantics.py`）确认：
    upsert 时带 `vec=None` 或**干脆不带** vec 字段，两种写法都会把向量抹掉；
    而 `query(output_fields=["id", "vec"])` 能把向量读回来。
    """
    ids = [str(row["id"]) for row in batch if row.get("id")]
    if not ids:
        return 0
    expr = "id in [" + ", ".join(f'"{i}"' for i in ids) + "]"
    try:
        rows = client.query(collection_name=name, filter=expr, output_fields=["id", "vec"])
    except Exception as exc:  # noqa: BLE001 读不回来不该让整轮导入失败
        # 降级后果要说清楚：本批向量会被清空（与修复前一样），但必须重跑 embed_tickets 补齐。
        print(f"WARN 读取已有向量失败，本批的向量将被清空（之后需重跑 embed_tickets）: {exc}")
        return 0
    existing = {str(r.get("id")): r.get("vec") for r in rows if _has_vector(r.get("vec"))}
    filled = _merge_existing_vectors(batch, existing)
    if filled:
        print(f"  保留已有向量 {filled}/{len(batch)} 条（upsert 是整实体覆盖，不读回来就会抹掉）")
    return filled


def ensure_collection(client, name: str, recreate: bool = False) -> None:
    # 延迟 import pymilvus：本模块的字段抽取部分（也是测试覆盖的部分）不应该强依赖
    # Milvus 客户端能装上/能连上。只有真要建集合时才需要它。
    from pymilvus import DataType

    if client.has_collection(name):
        # 已存在且没要求重建 → 直接返回（幂等，配合 --insert 可以反复跑）。
        if not recreate:
            return
        # 注意 recreate 是**先删后建**：drop_collection 会把已有数据与向量一起清掉，
        # 之后必须重跑 embed_tickets.py 回填向量。跑之前确认没有别的东西在依赖这个集合。
        client.drop_collection(name)

    # auto_id=False：主键由我们自己给（ticket_id 的哈希值），要让"同一张票永远同一个 id"。
    # enable_dynamic_field=False：拒绝 schema 之外的字段，
    # 这样"代码里多加了一个字段但没改建表逻辑"会在插入时直接报错，而不是静默存进去。
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    # max_length 必须给足：中文在 Milvus 里按 UTF-8 字节算长度，
    # 一个汉字 3 字节。semantic_text/ocr_text 给到 8192/16384 是为了容下整张票据的正文，
    # 调小会在插入时报 "string exceeds max length"，而且报错只提到超长、不提到是哪条记录。
    schema.add_field("ticket_type", DataType.VARCHAR, max_length=16)
    # 六个抽取字段全部 nullable=True：这是"抽不到就是 None"能落地的前提。
    # 若某列设成 not null，插入 None 会被拒，就只能补 0/"" 上去——
    # 那正是我们要避免的（0 分与真的 0 元无法区分）。
    schema.add_field("ticket_no", DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field("person", DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field("date_int", DataType.INT64, nullable=True)
    schema.add_field("amount_fen", DataType.INT64, nullable=True)
    schema.add_field("route", DataType.VARCHAR, max_length=256, nullable=True)
    schema.add_field("counterparty", DataType.VARCHAR, max_length=128, nullable=True)
    schema.add_field("semantic_text", DataType.VARCHAR, max_length=8192)
    schema.add_field("ocr_text", DataType.VARCHAR, max_length=16384)
    schema.add_field("source_file", DataType.VARCHAR, max_length=256)
    # 维度写死 1024（BAAI/bge-m3 的维度），⚠️ 与 settings.embedding.embedding_size 是**两处**。
    # 换 embedding 模型且维度变化时，这两处要一起改并重建集合——
    # 只改配置不改这里，回填向量会在 upsert 时报维度不匹配；
    # 只改这里不改配置，embed_texts 返回的维度才是实际决定因素。
    # 另外 Milvus 的维度和索引是绑死的：改维度必须重建集合，不能原地改。
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=1024, nullable=True)

    index_params = client.prepare_index_params()
    # AUTOINDEX 让 Milvus 按数据量自选索引类型（小数据量 HNSW、大数据量更省内存的方案），
    # 教学项目不必手工调参。
    # metric_type=COSINE 与检索侧保持一致：cosine 只看方向不看模长，
    # bge-m3 的向量本就是按余弦相似度用的；换成 L2 会让"相似度阈值"这个语义整个失效
    # （L2 是距离，越小越像，而 filters/rerank 里的阈值是按相似度调的）。
    index_params.add_index(field_name="vec", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)


def build_report(records: list[dict]) -> dict:
    # 先按票据类型分组：三类票据的字段天然缺失情况不同（机票没有日期金额、发票没有行程），
    # 混在一起算覆盖率会看不出"到底是哪类抽坏了"。
    by_type: dict[str, list[dict]] = {}
    for r in records:
        by_type.setdefault(r["ticket_type"], []).append(r)
    # coverage：每个类型下每个字段"非空"的条数。
    # 判空用 `not in (None, "")` 同时排除 None 与空串——因为 extract_invoice 的 route
    # 刻意是 ""（语义空值），不该被算成"抽到了行程"。
    # 看报告时的经验值：机票的 date_int/amount_fen 覆盖率永远是 0（这两列在登机牌上
    # 天生不存在，见 extract_flight，不是 bug）；其余字段的覆盖率如果明显低于同类其他字段，
    # 才说明抽取器出了问题。
    coverage = {
        t: {f: sum(1 for r in rs if r.get(f) not in (None, "")) for f in FIELDS}
        for t, rs in by_type.items()
    }
    # samples：每类抽前 3 条，且把两段长文本截到 120 字符——
    # 报告是要用肉眼看/直接 diff 的，全文会把它撑成几 MB 而失去"快速核对"的作用。
    # 截断只作用于报告副本（{**r, ...} 生成新字典），不影响入库用的 records。
    samples = {}
    for t, rs in by_type.items():
        samples[t] = [{**r, "ocr_text": r["ocr_text"][:120], "semantic_text": r["semantic_text"][:120]} for r in rs[:3]]
    return {
        "total": len(records),
        "by_type": {t: len(rs) for t, rs in by_type.items()},
        "coverage": coverage,
        "samples": samples,
    }


def load_items(args) -> list[dict]:
    # data 是 {"ocr_engine": ..., "results": [...]} 的包装（见 build_ocr_json），
    # 用 .get("results", []) 是为了容忍文件存在但 results 缺失的情况。
    data = json.loads(OCR_JSON_PATH.read_text(encoding="utf-8"))
    items = data.get("results", [])
    # --type 过滤放在 --limit 之前，顺序不能反：反了就是"先截前 N 条再筛类型"，
    # 只会得到恰好落在前 N 条里的那几类，看起来像"这类票据一条都没有"。
    if args.type:
        items = [it for it in items if it.get("ticket_type") == args.type]
    # 注意这里用的是真值判断（不是 `is not None`）：--limit 没传时是 None → 跳过；
    # 传 0 时也会被跳过（0 是 falsy），即 --limit 0 等价于不限制，
    # 与 paddle_ocr.collect_images 的 `is not None` 口径**不一致**，调试时别混淆。
    if args.limit:
        items = items[: args.limit]
    return items


def blank_semantic_text_count(client, name: str) -> int:
    """数 `semantic_text` 为空的行数（这些行永远召不回，却占着 `count(*)` 的分母）。

    单独成函数只为一件事：**能被测**。它原先嵌在 `main()` 里，而 `main()` 要连真 Milvus、
    真写库 —— 于是"空文本行到底有没有被如实报出来"这件事（B 表那条欠账的核心）
    只有"连着真库跑一次"这一条验证路径。抽出来后可以用假 client 钉住。
    """
    rows = client.query(collection_name=name, filter='semantic_text == ""', output_fields=["count(*)"])
    return int(rows[0].get("count(*)", 0)) if rows else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="提取票据字段并写入 Milvus tick 集合")
    # --type 的 choices 与 EXTRACTORS 的 key 是同源的三个值，新增票据类型要一起改。
    ap.add_argument("--type", choices=["flight", "invoice", "train"], help="只处理某类票据")
    ap.add_argument("--limit", type=int, help="最多处理多少条,调试用")
    ap.add_argument("--insert", action="store_true", help="写入 Milvus")
    ap.add_argument("--recreate", action="store_true", help="删除并重建 tick 集合")
    ap.add_argument("--save", metavar="PATH", help="把提取结果另存为 jsonl")
    ap.add_argument("--report", metavar="PATH", default=str(DEFAULT_REPORT_PATH))
    args = ap.parse_args()

    records = [build_record(it) for it in load_items(args)]

    # 报告永远先出（不管后面插不插库）：字段抽取是正则启发式，必须先用
    # coverage（覆盖率）+ samples（样本）确认抽得对，再决定要不要 --insert。
    # 这也是"默认干跑"的实现方式——不加 --insert 时函数到这里就结束了。
    report = build_report(records)
    print(f"extracted {report['total']} records: {report['by_type']}")
    # 逐类型打印各字段的非空条数：某个字段突然整列掉 0，一眼就能看出来。
    for t, cov in report["coverage"].items():
        print(f"  [{t}] " + " ".join(f"{f}={n}" for f, n in cov.items()))
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {args.report}")

    if args.save:
        # jsonl（一行一条）而不是一个 JSON 数组：便于逐行 diff 与按行过滤，
        # 也是评估脚本能流式读的格式。ensure_ascii=False 保留中文原文。
        Path(args.save).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
        )
        print(f"records -> {args.save}")

    if args.insert:
        # 延迟到真正要入库时才 import：Milvus 客户端不在时，"抽取 + 出报告"这条路径
        # （以及单元测试）依然可用。
        from core.database import get_milvus_client

        client = get_milvus_client()
        name = settings.milvus.collection
        ensure_collection(client, name, recreate=args.recreate)
        print(f"collection '{name}' ready")

        # 每 100 条一批：Milvus 的一次请求有数据量上限（且单条 JSON 过大时请求体也会被拒），
        # 分批是必需的；100 是"批次数不至于太多、单批也不至于过大"的经验值。
        #
        # ⚠ 用 **upsert** 而不是 insert（本条曾记为欠账"导入不幂等"，实测确认过）：
        # 一次性集合上的探针（`.dsh_tmp/probe_upsert_semantics.py`）实测：
        #   · 同主键 insert 两次 → **row_count=2，真的多出一行**（重跑一次语料翻倍、
        #     检索里同一张票出现两遍、评估的分母也跟着错）；
        #   · 同主键 upsert 两次 → 逻辑上仍是一行（`count(*)` 稳定），但 `row_count`
        #     **会涨**——upsert 是"标记删除 + 插入新版本"，旧版本在 compaction 前仍计入
        #     row_count。所以**判断幂等要数 `count(*)`，不能看 row_count**；
        #   · upsert 是**整实体覆盖**：带 `vec=None` 或干脆不带 vec 字段，都会把已算好的
        #     向量抹成 NULL（实测两种写法都抹）—— 结构化检索还查得到、语义检索却查不到，
        #     这种不一致最难发现。因此下面先读回旧向量再写（见 `_carry_over_vectors`）。
        # 字段完整性：`build_record` 产出的键与集合 schema 一一对应（多一个少一个都会被
        # Milvus 拒绝或写成空），所以这里不需要再补白名单。
        upserted = 0
        for i in range(0, len(records), 100):
            batch = records[i : i + 100]
            _carry_over_vectors(client, name, batch)
            res = client.upsert(collection_name=name, data=batch)
            # res.get("upsert_count", len(batch))：服务端不回计数时按批次大小计，
            # 保证最后打印的数至少是个合理值（真实条数以 count(*) 为准）。
            upserted += res.get("upsert_count", res.get("insert_count", len(batch)))
        print(f"upserted {upserted} rows")

        # flush 是让写入落盘/可查，某些 Milvus 版本/模式下不支持就抛异常——
        # 这里故意吞掉（pass）：flush 失败不等于写入失败，下面的计数轮询才是判据。
        try:
            client.flush(collection_name=name)
        except Exception:
            pass
        # 等数据可见：写入是异步可见的，立刻查会出现"写了 300 条但查不到"的假象。
        # 判据用 **count(*)**（逻辑条数，幂等时稳定）而不是 row_count（含软删旧版本，
        # upsert 之后必然虚高）。最多等 5×2=10 秒，等不到就打印当前值（不报错）——
        # 这是诊断信息，不是断言。
        live = 0
        for _ in range(5):
            try:
                got = client.query(collection_name=name, filter='id != ""', output_fields=["count(*)"])
                live = int(got[0].get("count(*)", 0)) if got else 0
            except Exception:
                live = 0
            if live >= len(records):
                break
            time.sleep(2)
        stats = client.get_collection_stats(collection_name=name)
        print(f"count(*): {live}（本轮抽出 {len(records)} 条）")
        print(f"row_count: {stats.get('row_count')}（含 upsert 标记删除的旧版本，不作为幂等判据）")

        # 空文本行的存在必须被**点出来**，不能让它悄悄混在 count(*) 里：
        # 这些行的 vec 恒为 NULL、BM25 也切不出词 ⇒ **永远召不回**，但它们照样占着分母，
        # 于是"300 条票据"与"实际能召回的 282 条"是两个数。两个数都对，混着用才错，
        # 所以这里把两个数一起打出来（真机实测 300 行里 18 行为空，全是 OCR 失败的 flight）。
        # 刻意**不删这些行**：删了就等于把"有 18 张票的 OCR 是空的"这件事从库里抹掉，
        # 而那正是该被修的数据缺口（失败明细见 data/ocr_results_clean.json 的 ocr_status=failed）。
        blank = 0
        try:
            blank = blank_semantic_text_count(client, name)
        except Exception as exc:  # noqa: BLE001 - 统计失败不该让导入失败
            print(f"WARN 统计空文本行失败（不影响导入结果）: {exc}")
        if blank:
            print(
                f"空 semantic_text: {blank} 条 ⇒ **可召回 {live - blank} 条**；"
                "这些行的 vec 恒为 NULL、关键词也切不出词，语义/关键词两路都搜不到它们"
                "（不是 bug，是 OCR 没出文本；修数据前别把它们算进'能查到的票据数'）"
            )

        # 检索前必须 load：Milvus 的集合要先加载进内存才能 query/search，
        # 否则报 "collection not loaded"。注意此时 vec 还没回填（那是 embed_tickets.py 的事），
        # 所以下面只做标量 query 抽样，不做向量搜索。
        client.load_collection(collection_name=name)
        sample = client.query(
            collection_name=name,
            filter='ticket_type == "train"',
            limit=2,
            # 只取标量字段（不含 vec/正文）：抽样是为了肉眼确认字段抽对了，
            # 把 ocr_text 拉出来会把终端刷满。
            output_fields=["id", "ticket_no", "person", "date_int", "amount_fen", "route", "counterparty"],
        )
        print(f"query sample: {sample}")
        # 显式关闭连接。注意没有 try/finally：中途异常时连接不关，
        # 一次性脚本靠进程退出回收即可。
        client.close()


if __name__ == "__main__":
    main()
