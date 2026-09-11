# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""PaddleOCR 图片识别:调用 AI Studio 的 PaddleOCR API,将 data 目录下的图片批量转换为 Markdown"""

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
OUTPUT_DIR = DATA_DIR / "output"
DEFAULT_CATEGORIES = ("flight", "invoice", "train")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 600


def submit_job(image_path: Path) -> str:
    """提交 OCR 任务,返回 jobId"""
    headers = {"Authorization": f"bearer {settings.paddleocr.token}"}
    if str(image_path).startswith("http"):
        headers["Content-Type"] = "application/json"
        payload = {
            "fileUrl": str(image_path),
            "model": settings.paddleocr.model,
            "optionalPayload": OPTIONAL_PAYLOAD,
        }
        resp = requests.post(settings.paddleocr.job_url, json=payload, headers=headers, timeout=60)
    else:
        data = {
            "model": settings.paddleocr.model,
            "optionalPayload": json.dumps(OPTIONAL_PAYLOAD),
        }
        with open(image_path, "rb") as f:
            resp = requests.post(
                settings.paddleocr.job_url,
                headers=headers,
                data=data,
                files={"file": f},
                timeout=120,
            )
    if resp.status_code != 200:
        raise RuntimeError(f"提交任务失败 HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()["data"]["jobId"]


def poll_job(job_id: str) -> str:
    """轮询任务状态,完成后返回结果 jsonl 的下载地址"""
    headers = {"Authorization": f"bearer {settings.paddleocr.token}"}
    job_url = f"{settings.paddleocr.job_url}/{job_id}"
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"任务 {job_id} 轮询超时({POLL_TIMEOUT_SECONDS} 秒)")
        resp = requests.get(job_url, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"查询任务失败 HTTP {resp.status_code}: {resp.text[:300]}")
        info = resp.json()["data"]
        state = info["state"]
        if state == "pending":
            print(f"  [{job_id}] 排队中...")
        elif state == "running":
            progress = info.get("extractProgress") or {}
            print(f"  [{job_id}] 识别中 {progress.get('extractedPages', '?')}/{progress.get('totalPages', '?')} 页")
        elif state == "done":
            print(f"  [{job_id}] 识别完成")
            return info["resultUrl"]["jsonUrl"]
        elif state == "failed":
            raise RuntimeError(f"任务失败:{info.get('errorMsg')}")
        time.sleep(POLL_INTERVAL_SECONDS)


def save_markdown(jsonl_url: str, image_path: Path, output_dir: Path) -> None:
    """下载 jsonl 结果,将 Markdown 文本按原图片文件名保存"""
    resp = requests.get(jsonl_url, timeout=120)
    resp.raise_for_status()
    texts = []
    for line in resp.text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        result = json.loads(line)["result"]
        for item in result["layoutParsingResults"]:
            texts.append(item["markdown"]["text"])

    if not texts:
        raise RuntimeError("识别结果为空")
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(texts) == 1:
        paths = [output_dir / f"{image_path.stem}.md"]
    else:
        paths = [output_dir / f"{image_path.stem}_{i + 1}.md" for i in range(len(texts))]
    for text, path in zip(texts, paths):
        path.write_text(text, encoding="utf-8")
        print(f"  已保存:{path}")


def ocr_image(image_path: Path, output_dir: Path) -> None:
    """识别单张图片并保存 Markdown"""
    print(f"处理:{image_path.name}")
    job_id = submit_job(image_path)
    jsonl_url = poll_job(job_id)
    save_markdown(jsonl_url, image_path, output_dir)


def collect_images(category: str, limit: int | None) -> list[Path]:
    """收集某个分类下需要识别的图片"""
    category_dir = DATA_DIR / category
    if not category_dir.is_dir():
        raise FileNotFoundError(f"分类目录不存在:{category_dir}")
    images = sorted(p for p in category_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
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
            if not overwrite and any(output_dir.glob(f"{image_path.stem}*.md")):
                print(f"跳过(已有结果):{image_path.name}")
                continue
            try:
                ocr_image(image_path, output_dir)
            except Exception as exc:
                print(f"  识别失败:{image_path.name} -> {exc}")
                failed.append(image_path)
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
        choices=DEFAULT_CATEGORIES,
        help="要处理的分类,默认全部",
    )
    parser.add_argument("--limit", type=int, default=None, help="每个分类最多处理多少张图片,调试用")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 Markdown 结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed = run(args.categories, args.limit, args.overwrite)
    if failed:
        print(f"\n完成,共 {len(failed)} 张失败:")
        for path in failed:
            print(f"  - {path}")
        sys.exit(1)
    print("\n全部处理成功")


if __name__ == "__main__":
    main()
