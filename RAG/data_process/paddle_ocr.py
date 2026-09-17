"""PaddleOCR 图片识别:调用 AI Studio 的 PaddleOCR API,将 data 目录下的图片批量转换为 Markdown"""

# =============================================================================
# OCR 第一步：图片 → Markdown（PaddleOCR 云 API，基础篇「OCR 识别」）
#
# 在本项目的数据链上的位置：
#   data/<分类>/*.png ──(本脚本，PaddleOCR 云)──> data/output/<分类>/<stem>.md
#                     ──(build_ocr_json.py)──> data/ocr_results{,_clean}.json
#                     ──(tick_extract.py)──> Milvus
# 也就是说本脚本只负责"把图认成文字"，清洗与入库是后面两个脚本的事。
#
# 为什么走云 API 而不是本地 PaddleOCR：本地要自己拉模型、装 paddlepaddle（Windows 上
# 装 GPU 版尤其折腾），而云 API 只依赖一个 PADDLEOCR_TOKEN。代价是整条链依赖外网与
# 令牌额度，识别失败会体现为本脚本退出码 1（见 main）。
#
# 云 API 是**异步 job** 模型，不是一问一答，本脚本的三个函数正好对应三个阶段：
#   submit_job  → POST 提交，拿到 jobId
#   poll_job    → GET 轮询状态，done 后拿到 jsonl 结果文件的下载地址
#   save_markdown → 下载 jsonl，逐行解析，按原图片名写成 Markdown
# 这里刻意不做并发：一批只有几十张图，串行 + 打印进度更容易观察，也不会打爆令牌配额。
# =============================================================================

# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
# 必须位于业务 import 之前：直接以脚本路径运行时 sys.path[0] 是 RAG/data_process。
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）

import argparse
import json
import sys
import time
from pathlib import Path

import requests

try:
    from config import settings
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# 输出目录与输入目录分开（data/output/<分类>/ 而不是就地写回 data/<分类>/）：
# 否则下一次 collect_images 会把生成的 .md 和原图混在一起，也很难确认哪些图还没跑过。
OUTPUT_DIR = DATA_DIR / "output"
# 三个分类与 data/ 下的子目录同名，同时就是数据里的 ticket_type 取值
# （见 tick_extract.build_record：ticket_type 直接沿用目录名，所以目录改名会连带改变字段值）。
DEFAULT_CATEGORIES = ("flight", "invoice", "train")

# 只认这四种后缀。注意 .webp / .tif 不在其中——它们会被静默忽略（不是报错），
# 表现为"图片明明在目录里，脚本却说待处理 0 张"。
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
# 三个可选项全部关掉的取舍：文档矫正、去扭曲、图表识别对票据这种"已经很正的排版图"
# 收益很低，却会显著增加单张耗时与失败面（云 API 上每开一项都是一个额外的模型阶段）。
# 保持全 False 是"最小可用识别"的默认值；真遇到拍歪的票据再逐项打开。
OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}
# 轮询间隔 5 秒：太短会白白消耗云 API 的 QPS 配额，太长又让批量任务等得不耐烦；
# 单张票据通常十几秒内 done。
POLL_INTERVAL_SECONDS = 5
# 轮询总超时 10 分钟：队列拥堵或大图时会超过 1 分钟，但**没有任何任务值得等 10 分钟以上**，
# 与其挂死不如报超时让 run() 记入失败清单，回头单独重试（脚本支持跳过已完成的图）。
POLL_TIMEOUT_SECONDS = 600


def submit_job(image_path: Path) -> str:
    """提交 OCR 任务,返回 jobId"""
    # 鉴权头是固定的 "bearer <token>"（**小写 bearer**，云 API 文档给的就是这个大小写，
    # 照抄为准；鉴权头大小写对不上时服务端通常直接 401）。令牌来自 .env 的
    # PADDLEOCR_TOKEN，经 config.py 汇总，本文件不碰 os.environ。
    headers = {"Authorization": f"bearer {settings.paddleocr.token}"}
    # 同一个函数要支持两种入参：远端 URL（str 以 http 开头）和本地文件。
    # 走 `str(image_path).startswith("http")` 判断而不是 isinstance，是为了兼容
    # 调用方传进来的是 Path("https://...") 这种合法但少见的形式。
    if str(image_path).startswith("http"):
        # 远端 URL 分支：JSON 请求体，optionalPayload 是**嵌套对象**。
        headers["Content-Type"] = "application/json"
        payload = {
            "fileUrl": str(image_path),
            "model": settings.paddleocr.model,
            "optionalPayload": OPTIONAL_PAYLOAD,
        }
        # 60 秒：只是把 URL 交给服务端排队，不等识别结果，所以不需要长超时。
        resp = requests.post(settings.paddleocr.job_url, json=payload, headers=headers, timeout=60)
    else:
        # 本地文件分支：multipart/form-data 上传。两个坑：
        #   1. optionalPayload 必须 json.dumps 成**字符串**再放进 form 字段——
        #      这个分支由 requests 编成表单，嵌套 dict 会被编坏（服务端收到 [object Object] 之类）；
        #   2. 不要手动设 Content-Type，requests 要自己带 boundary，手写会破坏 multipart 解析。
        data = {
            "model": settings.paddleocr.model,
            "optionalPayload": json.dumps(OPTIONAL_PAYLOAD),
        }
        # 120 秒：这一条要真正把图片字节传上去，大图/慢网下 60 秒会不够。
        with open(image_path, "rb") as f:
            resp = requests.post(
                settings.paddleocr.job_url,
                headers=headers,
                data=data,
                files={"file": f},
                timeout=120,
            )
    # 非 200 一律抛 RuntimeError（而不是返回 None 或 False）：
    # run() 用 try/except 收集失败清单，抛出才能把 HTTP 状态码和响应正文带进日志。
    # resp.text[:300] 截断是为了防止把整页 HTML 错误页刷进终端。
    if resp.status_code != 200:
        raise RuntimeError(f"提交任务失败 HTTP {resp.status_code}: {resp.text[:300]}")
    # 成功响应的外壳是 {"data": {"jobId": ...}}；这里直接下标取值——字段缺失会抛
    # KeyError 被 run() 记为失败。宁可炸得响一点，也不要静默返回空 jobId 去轮询。
    return resp.json()["data"]["jobId"]


def poll_job(job_id: str) -> str:
    """轮询任务状态,完成后返回结果 jsonl 的下载地址"""
    headers = {"Authorization": f"bearer {settings.paddleocr.token}"}
    # 详情接口就是在提交接口后面拼 jobId（job_url 形如 .../api/v2/ocr/jobs）。
    job_url = f"{settings.paddleocr.job_url}/{job_id}"
    # 用 time.monotonic() 而不是 time.time() 算截止时间：monotonic 不受系统时钟调整
    # （NTP 校准、手动改时间）影响，后者在跨秒跳变的机器上可能让超时永远不触发或立刻触发。
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while True:
        # 截止检查放在 while 顶部而不是 sleep 之后：这样"提交即超时"的极端情况也会先被拦下。
        if time.monotonic() > deadline:
            raise TimeoutError(f"任务 {job_id} 轮询超时({POLL_TIMEOUT_SECONDS} 秒)")
        resp = requests.get(job_url, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"查询任务失败 HTTP {resp.status_code}: {resp.text[:300]}")
        info = resp.json()["data"]
        state = info["state"]
        # 状态机一共四个值。注意下面**只有 done / failed 会跳出循环**：
        # 若云 API 将来新增状态（或返回 "queued" 这类未列出的值），这里会一路空转到超时，
        # 报出的是 TimeoutError 而不是"遇到未知状态"——排查时先看日志里有没有打印过进度。
        if state == "pending":
            print(f"  [{job_id}] 排队中...")
        elif state == "running":
            # extractProgress 可能缺失（进度还没上报），所以用 `.get(...) or {}` 兜住 None，
            # 再对每项取 '?' 默认值——纯粹为了让进度打印不至于把整个轮询打断。
            progress = info.get("extractProgress") or {}
            print(f"  [{job_id}] 识别中 {progress.get('extractedPages', '?')}/{progress.get('totalPages', '?')} 页")
        elif state == "done":
            print(f"  [{job_id}] 识别完成")
            # 返回的是 jsonl 结果的下载地址（resultUrl.jsonUrl）。这是**带签名的临时链接**，
            # 有有效期，必须马上下载（save_markdown 紧接着就 GET 它），不能存起来晚点用。
            return info["resultUrl"]["jsonUrl"]
        elif state == "failed":
            raise RuntimeError(f"任务失败:{info.get('errorMsg')}")
        time.sleep(POLL_INTERVAL_SECONDS)


def save_markdown(jsonl_url: str, image_path: Path, output_dir: Path) -> None:
    """下载 jsonl 结果,将 Markdown 文本按原图片文件名保存"""
    # raise_for_status()：失败要抛异常而不是继续解析错误页（错误页不是 jsonl，
    # 直接 json.loads 会报一个与真正原因无关的 JSONDecodeError）。
    resp = requests.get(jsonl_url, timeout=120)
    resp.raise_for_status()
    texts = []
    # jsonl = 一行一个独立 JSON（不是一个大 JSON 数组），所以必须逐行 loads。
    # 先 strip 再判空行：末尾换行会产生一个空串，直接 json.loads("") 会报 JSONDecodeError，
    # 报错内容与真实原因完全无关，非常难查。
    for line in resp.text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # 结构：{"result": {"layoutParsingResults": [{"markdown": {"text": ...}}, ...]}}
        # 每张图一行、每个 layoutParsingResults 是一页。这里把多页/多段的 Markdown
        # **拆成多个 text** 而不是拼成一个——所以下面才会出现 _1/_2 多文件命名。
        result = json.loads(line)["result"]
        for item in result["layoutParsingResults"]:
            texts.append(item["markdown"]["text"])

    # 识别成功但一个字都没认出来（空白票据、纯图无文字）：抛异常让 run() 记失败，
    # 而不是写一个空 .md 出去——否则 build_ocr_json 会把它当成 ocr_status=success 的空文本，
    # 一路入库成一个什么都搜不到的"幽灵票据"。
    if not texts:
        raise RuntimeError("识别结果为空")
    # parents=True：data/output/<分类>/ 往往还不存在；
    # exist_ok=True：断点续跑时目录已存在不该报错。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 单页直接用图片名（下游 build_ocr_json 靠 *.md 的 stem 反推图片名与票据号，
    # 所以绝大多数文件都走这一支、名字必须干净）；多页才加 _1.._n 后缀。
    if len(texts) == 1:
        paths = [output_dir / f"{image_path.stem}.md"]
    else:
        paths = [output_dir / f"{image_path.stem}_{i + 1}.md" for i in range(len(texts))]
    # 逐页写文件，编码固定 utf-8（票据里有中文与全角符号，用系统默认编码在 Windows 上会乱码）。
    for text, path in zip(texts, paths):
        path.write_text(text, encoding="utf-8")
        print(f"  已保存:{path}")


def ocr_image(image_path: Path, output_dir: Path) -> None:
    """识别单张图片并保存 Markdown"""
    # 三阶段的串行编排：没有重试（重试由 run() 那一层"重跑脚本"承担，
    # 因为跳过逻辑是按图片名判断的，重跑天然只补失败的图）。
    print(f"处理:{image_path.name}")
    job_id = submit_job(image_path)
    jsonl_url = poll_job(job_id)
    save_markdown(jsonl_url, image_path, output_dir)


def collect_images(category: str, limit: int | None) -> list[Path]:
    """收集某个分类下需要识别的图片"""
    category_dir = DATA_DIR / category
    # 目录不存在时抛 FileNotFoundError 而不是返回空列表：拼错分类名会静默跳过整个分类，
    # 那种"跑完了但什么都没生成"的问题很难查。
    if not category_dir.is_dir():
        raise FileNotFoundError(f"分类目录不存在:{category_dir}")
    # sorted 保证顺序稳定（按文件名），配合 --limit 才能得到可复现的"先跑前 N 张"。
    # suffix.lower() 是为了兼容 .PNG / .JPG 这类大写后缀。
    images = sorted(p for p in category_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    # limit 只截断不筛选。注意判断用的是 `is not None`，所以 `--limit 0` 会得到空列表
    # （不是"不限制"的含义）；裸跑（无 --limit）时 default=None 才是不限制。
    if limit is not None:
        images = images[:limit]
    return images


def run(categories: list[str], limit: int | None, overwrite: bool) -> list[Path]:
    """按分类批量识别,返回失败的图片列表"""
    failed: list[Path] = []
    for category in categories:
        output_dir = OUTPUT_DIR / category
        images = collect_images(category, limit)
        print(f"\n===== 分类[{category}] 待处理 {len(images)} 张 =====")
        for image_path in images:
            # 跳过判断用 `{stem}*.md` 前缀通配而不是 `{stem}.md`：
            # 因为多页结果会存成 <stem>_1.md、<stem>_2.md，精确匹配会把它们当成"没跑过"
            # 而重复识别。代价是前缀也会误判（t1.png 的产物会让 t10.png 被跳过），
            # 命名时避免出现互为前缀的文件名。
            if not overwrite and any(output_dir.glob(f"{image_path.stem}*.md")):
                print(f"跳过(已有结果):{image_path.name}")
                continue
            try:
                ocr_image(image_path, output_dir)
            except Exception as exc:
                # 单张失败不中断整批：网络抖动、单张图损坏都可能发生，
                # 收集起来最后统一汇报并让退出码非 0（见 main）。
                print(f"  识别失败:{image_path.name} -> {exc}")
                failed.append(image_path)
            # 每张之间歇 1 秒，给云 API 留出配额余量。
            time.sleep(1)
    return failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PaddleOCR 图片批量识别,Markdown 输出到 data/output/<分类>/"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        # choices 限制取值=DEFAULT_CATEGORIES，所以新增分类必须同时改常量和 data/ 下的目录，
        # 不能靠命令行塞一个新名字（那会被 argparse 直接拒绝）。
        choices=DEFAULT_CATEGORIES,
        help="要处理的分类,默认全部",
    )
    parser.add_argument("--limit", type=int, default=None, help="每个分类最多处理多少张图片,调试用")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 Markdown 结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed = run(args.categories, args.limit, args.overwrite)
    # 有失败就以 1 退出：批处理脚本要让调用方（人、CI、上游脚本）能从退出码看出"这批没跑干净"，
    # 否则"部分成功"会被当成成功。
    if failed:
        print(f"\n完成,共 {len(failed)} 张失败:")
        for path in failed:
            print(f"  - {path}")
        sys.exit(1)
    print("\n全部处理成功")


if __name__ == "__main__":
    main()
