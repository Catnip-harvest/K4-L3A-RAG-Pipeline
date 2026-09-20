"""
Task 8 — PageIndex vectorless fallback.

Fallback tồn tại để xử lý đúng một tình huống: dense search trả về điểm thấp
trên mọi ứng viên, nghĩa là câu hỏi không nằm gần bất kỳ vùng nào trong không
gian vector. Lúc đó lấy thêm chunk từ cùng một index cũng vô ích — cần một cách
truy xuất **khác về bản chất**.

Ý tưởng của PageIndex là bỏ vector hoàn toàn: dựng cây mục lục của tài liệu rồi
đi xuống theo cấu trúc, giống cách con người tra một cuốn quy chế — mở mục lục,
chọn chương, đọc mục.

Module này có hai đường:

* ``_remote_search`` — gọi dịch vụ PageIndex khi có ``PAGEINDEX_API_KEY``.
* ``_local_tree_search`` — bản chạy offline, dựng cây heading từ Markdown đã
  chuẩn hoá, chấm điểm theo **độ phủ từ khoá của câu hỏi trên từng mục**, rồi
  trả về mục thắng.

Bản local không phải BM25 nói cách khác: BM25 chấm điểm từng chunk 500 ký tự
rời rạc và thưởng cho tần suất từ; cây này chấm điểm ở mức *mục tài liệu*, ưu
tiên mục phủ được NHIỀU LOẠI từ khoá khác nhau của câu hỏi, nên nó kéo về được
cả những mục dài mà chunking đã xé vụn.

Vì sao vẫn gắn nhãn ``retrieval_method="pageindex"`` cho đường local: contract
định nghĩa "pageindex" là *tuyến vectorless*, không phải tên nhà cung cấp.
``provider_name()`` cho biết đường nào thực sự chạy, và UI hiển thị nó.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
DOC_ID_CACHE = Path(__file__).parent.parent / "pageindex_doc_ids.json"

REQUEST_TIMEOUT = 20

# Một heading Markdown, hoặc một dòng in hoa kiểu văn bản hành chính Việt Nam
# ("CHƯƠNG II", "ĐIỀU 5", "THÔNG TIN CHUNG") mà MarkItDown không nhận ra.
_HEADING = re.compile(r"^(#{1,6}\s+.*|[^a-zà-ỹ\n]{6,80})$", re.MULTILINE)
_TOKEN = re.compile(r"\w+", re.UNICODE)

# Từ chức năng tiếng Việt: xuất hiện ở mọi mục nên không phân biệt được gì.
_STOPWORDS = {
    "là", "của", "và", "có", "cho", "các", "được", "trong", "với", "những",
    "một", "này", "thì", "ở", "về", "khi", "nào", "bao", "nhiêu", "gì", "ai",
    "tại", "đến", "từ", "theo", "hay", "hoặc", "sẽ", "đã", "phải", "không",
}

_tree_cache: list[dict] | None = None


def provider_name() -> str:
    """Tên đường fallback đang hoạt động — UI và báo cáo hiển thị giá trị này."""
    return "pageindex-api" if PAGEINDEX_API_KEY else "local-tree"


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS}


def _build_tree() -> list[dict]:
    """Dựng cây mục từ Markdown đã chuẩn hoá. Cache trong bộ nhớ."""
    global _tree_cache
    if _tree_cache is not None:
        return _tree_cache

    sections: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        _tree_cache = sections
        return sections

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8").replace(" ", " ")
        doc_type = "legal" if "legal" in path.parts else "news"
        doc_title = path.stem
        url = ""
        for line in text.split("\n", 20)[:20]:
            if line.startswith("# "):
                doc_title = line[2:].strip() or doc_title
            elif line.startswith("**Source:**"):
                url = line.replace("**Source:**", "").strip()

        # Cắt tài liệu tại từng heading; mỗi mảnh là một "mục" của cây.
        boundaries = [match.start() for match in _HEADING.finditer(text)] or [0]
        if boundaries[0] != 0:
            boundaries.insert(0, 0)
        boundaries.append(len(text))

        for index in range(len(boundaries) - 1):
            body = text[boundaries[index] : boundaries[index + 1]].strip()
            if len(body) < 80:
                continue
            heading = body.split("\n", 1)[0].lstrip("# ").strip()
            sections.append(
                {
                    "id": f"{path.relative_to(STANDARDIZED_DIR).as_posix()}::section-{index}",
                    "content": body[:2000],
                    "heading": heading,
                    "tokens": _tokens(body),
                    "metadata": {
                        "source": path.name,
                        "title": doc_title,
                        "doc_type": doc_type,
                        "url": url,
                        "chunk_index": index,
                    },
                }
            )

    _tree_cache = sections
    return sections


def _local_tree_search(query: str, top_k: int) -> list[dict]:
    """Đi xuống cây mục theo độ phủ từ khoá, không dùng vector."""
    sections = _build_tree()
    wanted = _tokens(query)
    if not sections or not wanted:
        return []

    scored: list[tuple[float, dict]] = []
    for section in sections:
        overlap = wanted & section["tokens"]
        if not overlap:
            continue
        # Độ phủ = tỉ lệ loại từ khoá của câu hỏi xuất hiện trong mục. Thưởng
        # nhẹ cho heading khớp, vì trong văn bản quy chế heading mang nhiều tín
        # hiệu hơn thân bài.
        coverage = len(overlap) / len(wanted)
        heading_bonus = 0.15 if wanted & _tokens(section["heading"]) else 0.0
        scored.append((min(1.0, coverage + heading_bonus), section))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    results: list[dict] = []
    for score, section in scored[: max(top_k, 0)]:
        results.append(
            {
                "id": section["id"],
                "content": section["content"],
                "score": float(score),
                "metadata": dict(section["metadata"]),
                "retrieval_method": "pageindex",
            }
        )
    return results


def upload_documents() -> None:
    """Upload tài liệu lên PageIndex và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY trống — dùng cây mục local, không cần upload.")
        return

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = json.loads(DOC_ID_CACHE.read_text(encoding="utf-8")) if DOC_ID_CACHE.exists() else {}

    for path in sorted((STANDARDIZED_DIR).rglob("*.md")):
        key = path.relative_to(STANDARDIZED_DIR).as_posix()
        if key in cache:
            continue
        try:
            response = client.submit_document(str(path))
            cache[key] = getattr(response, "doc_id", None) or response["doc_id"]
            print(f"Uploaded: {key} -> {cache[key]}")
        except Exception as error:
            print(f"Failed: {key} — {type(error).__name__}: {error}")

    DOC_ID_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _remote_search(query: str, top_k: int) -> list[dict]:
    """Query PageIndex API và parse retrieved nodes."""
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = json.loads(DOC_ID_CACHE.read_text(encoding="utf-8")) if DOC_ID_CACHE.exists() else {}
    if not cache:
        return []

    results: list[dict] = []
    for source, doc_id in cache.items():
        response = client.retrieve(doc_id=doc_id, query=query, timeout=REQUEST_TIMEOUT)
        for rank, node in enumerate(getattr(response, "nodes", []) or [], start=1):
            results.append(
                {
                    "id": f"{source}::node-{getattr(node, 'node_id', rank)}",
                    "content": getattr(node, "text", "") or "",
                    # API không luôn trả score — dùng thứ hạng làm score giảm dần.
                    "score": float(getattr(node, "score", 0.0) or 1.0 / rank),
                    "metadata": {
                        "source": Path(source).name,
                        "title": Path(source).stem,
                        "doc_type": "legal" if source.startswith("legal") else "news",
                        "url": "",
                        "chunk_index": rank - 1,
                    },
                    "retrieval_method": "pageindex",
                }
            )

    results = [item for item in results if item["content"].strip()]
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[: max(top_k, 0)]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult.

    Dịch vụ ngoài lỗi thì rơi về cây local thay vì ném lên trên — Task 9 vẫn có
    ``try/except`` riêng, nhưng mất mạng không nên làm mất luôn cả fallback.
    """
    if not query or not query.strip() or top_k <= 0:
        return []

    if PAGEINDEX_API_KEY:
        try:
            remote = _remote_search(query, top_k)
            if remote:
                return remote
        except Exception as error:
            print(f"PageIndex API lỗi ({type(error).__name__}) — chuyển sang cây local.")

    return _local_tree_search(query, top_k)


if __name__ == "__main__":
    upload_documents()
    for result in pageindex_search("ký túc xá", top_k=3):
        print(f"{result['score']:.3f}  {result['id']}")
