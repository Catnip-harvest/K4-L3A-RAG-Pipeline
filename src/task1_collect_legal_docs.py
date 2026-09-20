"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Chủ đề của nhóm: "Dịch vụ sinh viên và quy chế đào tạo — Trường Đại học Giao
thông Vận tải (UTC)".

Bốn tài liệu dưới đây đều là PDF công khai trên các tên miền con của utc.edu.vn
và đều có lớp text nhúng thật (không phải ảnh scan), nên MarkItDown ở Task 3 đọc
được mà không cần OCR.

Script được viết theo hướng *idempotent*: chạy lại nhiều lần không tải lại file
đã có. Lý do là corpus này được build đi build lại trong lúc tinh chỉnh pipeline;
tải lại 3 MB mỗi lần vừa chậm vừa dễ bị máy chủ trường chặn vì spam request.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"

# Một số cổng thông tin của trường trả 403 cho User-Agent mặc định của requests
# ("python-requests/x.y"). Khai báo UA trình duyệt desktop để tải được file.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT_SECONDS = 60

# Bảng nguồn: tên file không dấu -> URL gốc. Tên file phản ánh nội dung tài liệu
# để người đọc repo biết ngay file nào là gì mà không phải mở ra.
SOURCES: dict[str, str] = {
    "qc-dao-tao-thac-si-2021.pdf": (
        "https://www.utc.edu.vn/Upload/FilePost/2022/06/09/qc-thac-si-2021.pdf"
    ),
    "so-tay-sinh-vien-k60.pdf": (
        "https://fme.utc.edu.vn/sites/fme.utc.edu.vn/files/"
        "SO%20TAY%20SINH%20VIEN%20K60.pdf"
    ),
    "so-tay-ho-tro-co-van-hoc-tap.pdf": (
        "https://dee.utc.edu.vn/sites/dee.utc.edu.vn/files/"
        "Sotay%20Hotro%20CVHT-SV%20DDT%20V1.0.pdf"
    ),
    "de-an-tuyen-sinh-2025.pdf": (
        "https://tuyensinh.utc.edu.vn/sites/ts.utc.edu.vn/files/"
        "29.5.25_%C4%90%E1%BB%81%20%C3%A1n%20tuy%E1%BB%83n%20sinh%202025.pdf"
    ),
}

# Ngưỡng tối thiểu để coi một file là tải thành công. Khi máy chủ trả về trang
# lỗi HTML thay vì PDF, nội dung thường chỉ vài KB — chặn sớm tốt hơn là để
# MarkItDown ở Task 3 báo lỗi khó hiểu.
MIN_VALID_BYTES = 1024


class ManifestEntry(TypedDict):
    """Một dòng trong MANIFEST.json, mô tả xuất xứ của một file PDF."""

    filename: str
    url: str
    sha256: str
    bytes: int
    downloaded_at: str


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def compute_sha256(path: Path) -> str:
    """Tính SHA-256 của file, đọc theo khối để không nạp cả file vào RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, ManifestEntry]:
    """Đọc manifest cũ (nếu có), khoá theo tên file."""
    if not MANIFEST_PATH.exists():
        return {}
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Manifest hỏng thì bỏ qua và dựng lại, không làm hỏng cả lần chạy.
        print("Warning: MANIFEST.json is not valid JSON — rebuilding it")
        return {}
    return {entry["filename"]: entry for entry in raw.get("documents", [])}


def write_manifest(entries: list[ManifestEntry]) -> None:
    """Ghi manifest theo thứ tự tên file để diff giữa hai lần chạy sạch sẽ."""
    payload = {
        "topic": (
            "Dịch vụ sinh viên và quy chế đào tạo — "
            "Trường Đại học Giao thông Vận tải (UTC)"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": sorted(entries, key=lambda entry: entry["filename"]),
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def download_one(filename: str, url: str) -> None:
    """Tải một tài liệu về DATA_DIR, ghi đè nếu đã tồn tại."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    if len(response.content) < MIN_VALID_BYTES:
        raise ValueError(
            f"{filename}: server returned only {len(response.content)} bytes — "
            "likely an error page, not the PDF"
        )
    (DATA_DIR / filename).write_bytes(response.content)


def download_documents() -> None:
    """Tải các PDF còn thiếu và cập nhật MANIFEST.json.

    Quy tắc bỏ qua: file đã tồn tại, kích thước khớp manifest cũ và lớn hơn
    ngưỡng tối thiểu thì giữ nguyên. Chỉ so kích thước chứ không so hash vì
    hash phải đọc lại toàn bộ file; kích thước đủ để phát hiện tải dở.
    """
    # mkdir trực tiếp thay vì gọi setup_directory() để hàm này gọi độc lập được
    # mà không in trùng dòng "Ready:" khi entrypoint đã gọi setup_directory().
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    previous = load_manifest()
    entries: list[ManifestEntry] = []

    for filename, url in SOURCES.items():
        path = DATA_DIR / filename
        known = previous.get(filename)
        size_matches_manifest = (
            path.exists()
            and known is not None
            and path.stat().st_size == known["bytes"]
        )
        already_valid = path.exists() and path.stat().st_size >= MIN_VALID_BYTES

        if size_matches_manifest:
            print(f"Skip (unchanged): {filename}")
            entries.append(known)
            continue

        if already_valid:
            # File có sẵn nhưng chưa có trong manifest (ví dụ tải tay lần đầu).
            # Không tải lại, chỉ lập hồ sơ xuất xứ cho nó.
            print(f"Adopt (already on disk): {filename}")
        else:
            print(f"Downloading: {filename}")
            try:
                download_one(filename, url)
            except Exception as error:
                # Một file hỏng không nên giết cả lần chạy — ba file còn lại
                # vẫn đủ vượt ngưỡng tối thiểu của bài lab.
                print(f"Failed: {filename} — {error}")
                continue

        entries.append(
            ManifestEntry(
                filename=filename,
                url=url,
                sha256=compute_sha256(path),
                bytes=path.stat().st_size,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    write_manifest(entries)
    total_bytes = sum(entry["bytes"] for entry in entries)
    print(f"\nLegal documents ready: {len(entries)} files, {total_bytes:,} bytes")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    setup_directory()
    download_documents()
