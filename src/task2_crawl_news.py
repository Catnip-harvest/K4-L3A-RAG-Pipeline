"""
Task 2 — Crawl bài viết/thông báo trên cổng thông tin UTC.

Mười trang dưới đây là các trang giới thiệu đơn vị phục vụ sinh viên của Trường
Đại học Giao thông Vận tải (phòng đào tạo, công tác sinh viên, ký túc xá, thư
viện, trạm y tế...). Đây chính là loại thông tin sinh viên hay hỏi chatbot nhất,
nên chúng bổ sung tốt cho bốn văn bản quy chế ở Task 1.

Hai đường crawl:
    1. Crawl4AI (ưu tiên) — cần Playwright Chromium.
    2. requests + bộ trích xuất HTML bằng thư viện chuẩn (dự phòng).
Khi Crawl4AI hỏng (thiếu browser, import lỗi...), script tự chuyển sang đường 2
và in rõ đã dùng đường nào. Chọn cách này vì bài lab chấm chất lượng dữ liệu,
không chấm thư viện; để cả lần chạy chết vì thiếu browser là mất trắng.

Bốn bẫy dữ liệu đã gặp khi làm thủ công, xử lý ở phần "làm sạch" bên dưới:
    * Mega-menu ~200 dòng lặp trên mọi trang.
    * Khối "Bài viết xem nhiều" / "Tin tức nổi bật" ở cuối trang.
    * Bảng danh sách cán bộ kèm SỐ ĐIỆN THOẠI CÁ NHÂN — tuyệt đối không đưa vào
      repo.
    * Ký tự U+00A0 (&nbsp;) rải khắp HTML, phá vỡ mọi phép so khớp chuỗi và cắt
      chunk theo dấu phân cách ở Task 4.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import TypedDict

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://www.utc.edu.vn/gioi-thieu/phong-bao-ve",
    "https://www.utc.edu.vn/gioi-thieu/phong-cong-tac-chinh-tri-va-sinh-vien",
    "https://www.utc.edu.vn/gioi-thieu/phong-dao-tao-dai-hoc",
    "https://www.utc.edu.vn/gioi-thieu/trung-tam-dao-tao-truc-tuyen-utc",
    "https://www.utc.edu.vn/gioi-thieu/phong-ke-hoach-tai-chinh",
    "https://www.utc.edu.vn/gioi-thieu/ban-quan-ly-ky-tuc-xa",
    "https://www.utc.edu.vn/gioi-thieu/phong-phap-che-va-kiem-soat-noi-bo",
    "https://www.utc.edu.vn/gioi-thieu/phong-quan-ly-chat-luong",
    "https://www.utc.edu.vn/gioi-thieu/trung-tam-thong-tin-thu-vien",
    "https://www.utc.edu.vn/gioi-thieu/tram-y-te",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT_SECONDS = 60

PATH_CRAWL4AI = "crawl4ai"
PATH_REQUESTS = "requests+stdlib"

# Ngưỡng báo động: dưới mức này gần như chắc chắn trang chỉ còn lại rác.
MIN_CONTENT_CHARS = 400

# Một dòng xuất hiện trên >= 60% số trang thì bị coi là khung giao diện.
BOILERPLATE_PAGE_RATIO = 0.60
# ...nhưng chỉ xoá khi nó nằm trong một chuỗi >= 3 dòng chung liên tiếp. Đây là
# mấu chốt: "THÔNG TIN CHUNG" hay "Chức năng nhiệm vụ" cũng lặp trên nhiều
# trang nhưng đứng lẻ giữa nội dung thật, còn menu thì đi thành mảng dài.
BOILERPLATE_MIN_RUN = 3

# Cắt bỏ từ dòng đầu tiên khớp một trong các mốc này tới hết trang.
TAIL_MARKERS = ("Đăng bởi:", "Bài viết xem nhiều", "Tin tức nổi bật")

# Cắt bỏ từ dòng đầu tiên khớp mẫu này tới hết trang — đây là bảng nhân sự kèm
# họ tên và số di động cá nhân. Lưu ý một trang viết sai chính tả tiêu đề thành
# "ĐỘI NGŨ CÁN B=Ộ CHUYÊN VIÊN" (lẫn dấu "="), nên phải neo thêm vào dòng tiêu
# đề cột "Họ tên" của bảng thì mới chắc ăn.
PERSONAL_SECTION_PATTERN = re.compile(
    r"^(ĐỘI NGŨ|DANH SÁCH CÁN BỘ|DANH SÁCH NHÂN SỰ|Họ tên$|Họ và tên$)"
)

# Số điện thoại trên dòng có "ĐT:" — giữ lại nhãn, bỏ con số.
PHONE_PATTERN = re.compile(r"\b0\d[\d\s.\-]{7,12}\d\b")
PHONE_PLACEHOLDER = "[số điện thoại đã lược bỏ]"

# Hộp thư của đơn vị thì giữ, hộp thư cá nhân thì bỏ cả dòng. Danh sách trắng
# liệt kê tường minh vì đoán theo hình dạng tên rất dễ sai với tên tiếng Việt
# viết liền (vd "vanphongdaotao" trông không khác gì "nguyenvanan").
INSTITUTIONAL_EMAILS = {
    # Hộp thư chung của trường, có ở chân mọi trang.
    "dhgtvt@utc.edu.vn",
    "info@utc.edu.vn",
    "daotao@utc.edu.vn",
    "ctsv@utc.edu.vn",
    "thuvien@utc.edu.vn",
    "baove@utc.edu.vn",
    "ktx@utc.edu.vn",
    "ytte@utc.edu.vn",
    "yte@utc.edu.vn",
    "tramyte@utc.edu.vn",
    "khtc@utc.edu.vn",
    "qlcl@utc.edu.vn",
    "phapche@utc.edu.vn",
    "elearning@utc.edu.vn",
    "utconline@utc.edu.vn",
    "tuyensinh@utc.edu.vn",
    "vanthu@utc.edu.vn",
    "hanhchinh@utc.edu.vn",
}
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]*utc\.edu\.vn", re.IGNORECASE)

# Thẻ không chứa nội dung đọc được.
SKIPPED_TAGS = {"script", "style", "noscript", "svg", "head", "iframe", "form"}
# Thẻ kết thúc một dòng văn bản.
BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "td", "th", "section", "article",
    "header", "footer", "nav", "ul", "ol", "table", "blockquote",
    "figcaption", "h1", "h2", "h3", "h4", "h5", "h6",
}

# Đặt sẵn cờ ở cấp module: Crawl4AI hỏng một lần thì hỏng cho cả mười trang,
# thử lại chỉ tốn thêm chín lần chờ browser timeout.
_crawl4ai_unavailable = False
_paths_used: set[str] = set()


class Article(TypedDict):
    """Bản ghi một bài đã crawl, đúng schema mà tests/test_acceptance.py đòi."""

    url: str
    slug: str
    title: str
    date_crawled: str
    content_markdown: str


def normalize_unicode(text: str) -> str:
    """Bỏ U+00A0 và U+FEFF.

    Không phải chuyện thẩm mỹ: U+00A0 trông hệt dấu cách nhưng khác mã, nên
    `"Phòng Đào tạo" in content` trả về False một cách khó hiểu, và bộ cắt chunk
    theo dấu phân cách ở Task 4 không tách được đoạn.
    """
    return text.replace(" ", " ").replace("﻿", "")


def slugify(url: str) -> str:
    """Lấy đuôi URL làm id ngắn không dấu, vd '.../phong-bao-ve' -> 'phong-bao-ve'."""
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    folded = unicodedata.normalize("NFKD", tail)
    ascii_only = "".join(ch for ch in folded if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return cleaned or "article"


class TextExtractor(HTMLParser):
    """Đổi HTML thành danh sách dòng văn bản, chỉ dùng thư viện chuẩn.

    Không dùng BeautifulSoup vì nó chỉ có mặt gián tiếp qua dependency của
    Crawl4AI; đường dự phòng mà lại phụ thuộc vào thứ mình đang dự phòng cho thì
    vô nghĩa. Mỗi ô bảng (<td>/<th>) thành một dòng riêng — nhờ vậy tiêu đề cột
    "Họ tên" đứng một mình và bộ lọc dữ liệu cá nhân bên dưới neo được vào nó.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.title: str = ""
        self.first_heading: str = ""
        self._buffer: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._heading_level = 0

    def _flush(self) -> None:
        text = normalize_unicode("".join(self._buffer))
        self._buffer.clear()
        collapsed = re.sub(r"\s+", " ", text).strip()
        if not collapsed:
            return
        if self._heading_level:
            if not self.first_heading:
                self.first_heading = collapsed
            collapsed = f"{'#' * self._heading_level} {collapsed}"
        self.lines.append(collapsed)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIPPED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in BLOCK_TAGS:
            self._flush()
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                self._heading_level = int(tag[1])

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
            self.title = normalize_unicode("".join(self._buffer)).strip()
            self._buffer.clear()
            return
        if tag in BLOCK_TAGS:
            self._flush()
            self._heading_level = 0

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._buffer.append(data)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._flush()


def clean_title(raw_title: str, fallback: str) -> str:
    """Bỏ phần tên trường lặp ở đuôi thẻ <title>, nếu có.

    Chỉ cắt sau dấu "|" và chỉ khi phần đuôi đúng là tên trường. Bản đầu tiên
    cắt theo cả " - " và làm hỏng "Phòng Kế hoạch - Tài chính" thành "Phòng Kế
    hoạch": tên đơn vị tiếng Việt rất hay có dấu gạch nối ở giữa, nên cắt theo
    dấu gạch là sai nhiều hơn đúng.
    """
    title = normalize_unicode(raw_title).strip()
    head, separator, tail = title.partition("|")
    if separator and len(head.strip()) >= 5:
        looks_like_site_name = any(
            hint in tail.lower()
            for hint in ("utc", "giao thông vận tải", "gtvt", "đại học")
        )
        if looks_like_site_name:
            title = head.strip()
    return title or fallback


def strip_personal_data(lines: list[str]) -> list[str]:
    """Cắt bảng nhân sự, che số điện thoại, bỏ dòng có email cá nhân.

    Ưu tiên cắt cả khối thay vì lọc từng dòng: bảng nhân sự còn có chức danh,
    đơn vị, ảnh — ghép lại vẫn đủ nhận diện một người cụ thể.
    """
    kept: list[str] = []
    for line in lines:
        if PERSONAL_SECTION_PATTERN.match(line.lstrip("# ").strip()):
            break

        if "ĐT:" in line or "ĐT :" in line:
            line = PHONE_PATTERN.sub(PHONE_PLACEHOLDER, line)

        emails = EMAIL_PATTERN.findall(line)
        if emails and any(
            email.lower() not in INSTITUTIONAL_EMAILS for email in emails
        ):
            continue

        kept.append(line)
    return kept


def strip_tail_sections(lines: list[str]) -> list[str]:
    """Cắt từ mốc cuối trang ("Đăng bởi:", "Bài viết xem nhiều"...) trở đi."""
    for index, line in enumerate(lines):
        bare = line.lstrip("# ").strip()
        if any(bare.startswith(marker) for marker in TAIL_MARKERS):
            return lines[:index]
    return lines


def collapse_blank_lines(lines: list[str]) -> str:
    """Ghép các dòng lại, không để quá một dòng trống liên tiếp."""
    output: list[str] = []
    for line in lines:
        if not line.strip():
            if output and not output[-1].strip():
                continue
            output.append("")
        else:
            output.append(line.rstrip())
    return "\n".join(output).strip() + "\n"


def remove_shared_boilerplate(pages: dict[str, list[str]]) -> dict[str, list[str]]:
    """Xoá mega-menu bằng cách đối chiếu chéo giữa các trang.

    Cách làm: đếm số trang chứa mỗi dòng; dòng nào có mặt ở >= 60% số trang là
    "dòng chung". Nhưng chỉ xoá khi nó nằm trong chuỗi >= 3 dòng chung liên
    tiếp. Điều kiện chuỗi giữ lại các tiêu đề mục thật như "THÔNG TIN CHUNG" —
    chúng lặp lại nhưng bị kẹp giữa nội dung riêng của từng trang, còn menu thì
    luôn đi thành mảng dài liền nhau.
    """
    page_count = len(pages)
    if page_count < 2:
        return pages

    appearances: Counter[str] = Counter()
    for lines in pages.values():
        appearances.update(set(lines))

    threshold = max(2, round(page_count * BOILERPLATE_PAGE_RATIO))
    common = {line for line, count in appearances.items() if count >= threshold}

    cleaned: dict[str, list[str]] = {}
    for slug, lines in pages.items():
        flags = [line in common for line in lines]
        drop = [False] * len(lines)
        start = 0
        while start < len(lines):
            if not flags[start]:
                start += 1
                continue
            end = start
            while end < len(lines) and flags[end]:
                end += 1
            if end - start >= BOILERPLATE_MIN_RUN:
                for index in range(start, end):
                    drop[index] = True
            start = end
        cleaned[slug] = [
            line for index, line in enumerate(lines) if not drop[index]
        ]
    return cleaned


async def fetch_with_crawl4ai(url: str) -> tuple[str, list[str]]:
    """Đường ưu tiên: Crawl4AI. Ném exception để hàm gọi chuyển sang dự phòng."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)
        if not getattr(result, "success", True):
            raise RuntimeError(getattr(result, "error_message", "crawl failed"))
        markdown = str(getattr(result, "markdown", "") or "")
        if not markdown.strip():
            raise RuntimeError("crawl4ai returned empty markdown")
        metadata = getattr(result, "metadata", None) or {}
        title = str(metadata.get("title") or "")
        lines = [
            normalize_unicode(line).rstrip()
            for line in markdown.splitlines()
        ]
        return title, [line for line in lines if line.strip()]


def fetch_with_requests(url: str) -> tuple[str, list[str]]:
    """Đường dự phòng: tải HTML thô rồi tự trích xuất bằng thư viện chuẩn."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    # Máy chủ UTC khai báo charset không nhất quán; ép UTF-8 theo nội dung thật.
    response.encoding = response.apparent_encoding or "utf-8"

    extractor = TextExtractor()
    extractor.feed(response.text)
    extractor.close()
    title = extractor.title or extractor.first_heading
    return title, extractor.lines


async def crawl_article(url: str) -> Article:
    """Crawl một trang và làm sạch ở mức từng trang.

    Việc xoá mega-menu cần đối chiếu nhiều trang với nhau nên không làm được ở
    đây; ``crawl_all`` làm bước đó sau khi đã gom đủ mười trang.
    """
    global _crawl4ai_unavailable

    title_raw = ""
    lines: list[str] = []

    if not _crawl4ai_unavailable:
        try:
            title_raw, lines = await fetch_with_crawl4ai(url)
            _paths_used.add(PATH_CRAWL4AI)
        except Exception as error:
            _crawl4ai_unavailable = True
            print(
                f"  Crawl4AI unavailable ({type(error).__name__}: "
                f"{str(error).splitlines()[0][:120]}) — using {PATH_REQUESTS}"
            )

    if not lines:
        title_raw, lines = fetch_with_requests(url)
        _paths_used.add(PATH_REQUESTS)

    slug = slugify(url)
    lines = strip_tail_sections(lines)
    lines = strip_personal_data(lines)

    return Article(
        url=url,
        slug=slug,
        title=clean_title(title_raw, slug.replace("-", " ").title()),
        date_crawled=datetime.now().astimezone().isoformat(),
        content_markdown=collapse_blank_lines(lines),
    )


async def crawl_all() -> None:
    """Crawl toàn bộ ARTICLE_URLS, làm sạch chéo rồi lưu từng bài một JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    articles: dict[int, Article] = {}
    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            print(f"[{index:02d}/{len(ARTICLE_URLS)}] {url}")
            articles[index] = await crawl_article(url)
        except Exception as error:
            # Một trang hỏng không được kéo theo chín trang còn lại.
            print(f"  Failed: {url} — {type(error).__name__}: {error}")

    # Bước đối chiếu chéo: chỉ chạy được khi đã có toàn bộ corpus trong tay.
    page_lines = {
        index: article["content_markdown"].splitlines()
        for index, article in articles.items()
    }
    cleaned = remove_shared_boilerplate(page_lines)

    print()
    for index, article in sorted(articles.items()):
        article["content_markdown"] = collapse_blank_lines(cleaned[index])
        output = DATA_DIR / f"article_{index:02d}.json"
        output.write_text(
            json.dumps(article, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        length = len(article["content_markdown"])
        warning = "  <-- SUSPICIOUSLY SHORT" if length < MIN_CONTENT_CHARS else ""
        print(f"Saved {output.name}  {length:>6,} chars  {article['slug']}{warning}")

    print(f"\nCrawled {len(articles)}/{len(ARTICLE_URLS)} pages")
    print(f"Crawl path used: {', '.join(sorted(_paths_used)) or 'none'}")


if __name__ == "__main__":
    # Console Windows mặc định là cp1252 và chết ngay khi in tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(crawl_all())
