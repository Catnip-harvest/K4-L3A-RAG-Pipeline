"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Gộp hai nguồn của Task 1 (PDF quy chế) và Task 2 (JSON bài viết) về cùng một
định dạng Markdown, mỗi file mở đầu bằng khối metadata giống nhau. Task 4 chỉ
cần đọc thư mục data/standardized/ mà không phải biết dữ liệu gốc là PDF hay
JSON.

Hai điểm đáng lưu ý:
    * MarkItDown đọc PDF tiếng Việt đôi khi ra chữ vỡ (mojibake) hoặc mất dấu
      cách. Vì vậy sau khi convert có một bước kiểm tra nhanh: file phải chứa
      được vài từ tiếng Việt thông dụng. Thà báo lỗi to còn hơn lặng lẽ đẩy rác
      vào vector store rồi tuần sau mới phát hiện.
    * Chuẩn hoá U+00A0/U+FEFF cho cả nhánh PDF, không riêng nhánh tin tức —
      PDF cũng có ký tự này và nó phá phép so khớp chuỗi y hệt.
"""

from __future__ import annotations

import json
import sys
import re
from pathlib import Path

from markitdown import MarkItDown


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
MANIFEST_PATH = LANDING_DIR / "legal" / "MANIFEST.json"

# Tên file không dấu -> tiêu đề tiếng Việt đầy đủ. Tên file phải không dấu để
# an toàn trên mọi hệ thống, nhưng tiêu đề hiển thị thì nên đọc được, vì nó đi
# thẳng vào phần trích dẫn nguồn mà chatbot trả cho sinh viên.
LEGAL_TITLES: dict[str, str] = {
    "qc-dao-tao-thac-si-2021": (
        "Quy chế tuyển sinh và đào tạo trình độ thạc sĩ (2021)"
    ),
    "so-tay-sinh-vien-k60": "Sổ tay sinh viên K60",
    "so-tay-ho-tro-co-van-hoc-tap": (
        "Sổ tay hỏi đáp hỗ trợ cố vấn học tập và sinh viên"
    ),
    "de-an-tuyen-sinh-2025": "Đề án tuyển sinh đại học 2025",
}

# Nếu không thấy từ nào trong số này thì gần như chắc chắn bản convert bị hỏng
# font hoặc PDF là ảnh scan không có lớp text.
VIETNAMESE_MARKERS = ("sinh viên", "học", "trường", "đào tạo")

# Dưới ngưỡng này thì coi như convert thất bại (test acceptance đòi >= 200).
MIN_OUTPUT_CHARS = 200


def normalize_unicode(text: str) -> str:
    """Bỏ U+00A0 (&nbsp;) và U+FEFF — xem chú thích ở Task 2."""
    cleaned = text.replace(" ", " ").replace("﻿", "")
    return redact_personal_numbers(cleaned)


# So di dong di kem ten mot nguoi cu the, vi du:
#   "hotline: 0941.740.673 (Mr. Hoang)/ 091.600.3638 (Mr. Huy)"
# Truong cong bo cac so nay de thi sinh goi, nhung chung van la so di dong gan
# voi mot ca nhan. Ban PDF goc trong data/landing/ giu nguyen nhu truong da phat
# hanh; cai repo khong nen tao ra la mot chi muc tim kiem duoc cua so ca nhan.
# Tong dai co dinh (028.38962819) khong khop mau nay nen duoc giu lai - do la so
# cua don vi va co the la dap an cua mot cau hoi benchmark.
_PERSONAL_MOBILE = re.compile(
    r"\b0[\d.\s]{7,13}\d(?=\s*\(\s*(?:Mr|Ms|Mrs|Ong|Ba|Co|Thay|Anh|Chi)\b)",
    re.IGNORECASE,
)


def redact_personal_numbers(text: str) -> str:
    """Thay so di dong ca nhan bang nhan, giu nguyen so tong dai cua don vi."""
    return _PERSONAL_MOBILE.sub("[so dien thoai da luoc bo]", text)


def load_legal_sources() -> dict[str, str]:
    """Đọc MANIFEST.json để lấy URL gốc cho từng file PDF.

    Không có manifest thì vẫn convert được, chỉ là mục Source ghi "unknown" —
    mất điểm truy xuất nguồn gốc chứ không làm hỏng pipeline.
    """
    if not MANIFEST_PATH.exists():
        print("Warning: MANIFEST.json not found — source URLs will be 'unknown'")
        return {}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        Path(entry["filename"]).stem: entry["url"]
        for entry in manifest.get("documents", [])
    }


def looks_vietnamese(text: str) -> bool:
    """Kiểm tra nhanh bản convert có đúng là tiếng Việt đọc được không."""
    lowered = text.lower()
    return any(marker in lowered for marker in VIETNAMESE_MARKERS)


def convert_legal_docs() -> list[tuple[str, int]]:
    """Convert PDF/DOCX trong landing/legal sang standardized/legal.

    Trả về danh sách (tên file, số ký tự) để hàm gọi in báo cáo.
    """
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_legal_sources()
    converter = MarkItDown()
    results: list[tuple[str, int]] = []

    documents = sorted(
        path for path in legal_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".doc", ".docx"}
    )

    for path in documents:
        stem = path.stem
        try:
            converted = converter.convert(str(path))
        except Exception as error:
            print(f"  FAILED to convert {path.name}: {type(error).__name__}: {error}")
            continue

        body = normalize_unicode(converted.text_content or "").strip()
        title = LEGAL_TITLES.get(stem, stem.replace("-", " ").capitalize())
        url = sources.get(stem, "unknown")

        if len(body) < MIN_OUTPUT_CHARS:
            # Không ghi file rỗng ra đĩa: test acceptance sẽ đỏ với thông báo
            # khó hiểu, trong khi lỗi thật nằm ở khâu convert.
            print(f"  FAILED {path.name}: only {len(body)} chars extracted — skipped")
            continue
        if not looks_vietnamese(body):
            print(
                f"  WARNING {path.name}: no recognisable Vietnamese words found — "
                "likely mojibake or a scanned PDF without a text layer"
            )

        header = (
            f"# {title}\n\n"
            f"**Source:** {url}\n\n"
            f"**Type:** legal\n\n"
            f"---\n\n"
        )
        # write_text ghi đè, nên chạy lại là cập nhật tại chỗ, không nhân bản.
        destination = output_dir / f"{stem}.md"
        destination.write_text(header + body + "\n", encoding="utf-8")
        results.append((destination.name, len(header) + len(body) + 1))

    return results


def convert_news_articles() -> list[tuple[str, int]]:
    """Convert JSON trong landing/news sang standardized/news.

    Đặt tên file theo ``slug`` thay vì ``article_NN`` để tên file nói lên nội
    dung; nếu JSON cũ chưa có slug thì lùi về tên file gốc.
    """
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, int]] = []

    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        body = normalize_unicode(data.get("content_markdown", "")).strip()
        if len(body) < MIN_OUTPUT_CHARS:
            print(f"  FAILED {path.name}: only {len(body)} chars of content — skipped")
            continue

        header = (
            f"# {normalize_unicode(data['title']).strip()}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n"
            f"**Type:** news\n\n"
            f"---\n\n"
        )
        destination = output_dir / f"{data.get('slug') or path.stem}.md"
        destination.write_text(header + body + "\n", encoding="utf-8")
        results.append((destination.name, len(header) + len(body) + 1))

    return results


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing và in báo cáo số ký tự từng file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Legal documents:")
    legal = convert_legal_docs()
    for name, length in legal:
        print(f"  {name:<40} {length:>8,} chars")

    print("\nNews articles:")
    news = convert_news_articles()
    for name, length in news:
        print(f"  {name:<40} {length:>8,} chars")

    print(
        f"\nStandardized {len(legal)} legal + {len(news)} news "
        f"= {len(legal) + len(news)} Markdown files"
    )
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    convert_all()
