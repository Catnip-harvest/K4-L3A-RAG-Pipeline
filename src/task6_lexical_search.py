"""
Task 6 — Lexical search bằng BM25.

BM25 bù đúng chỗ dense search yếu: mã văn bản ("QĐ 2109/QĐ-ĐHGTVT"), con số
("214 phòng"), tên riêng và từ viết tắt. Embedding gom những thứ đó vào vùng
"ngữ nghĩa gần nhau" và đánh mất tính chính xác từng ký tự; BM25 thì khớp đúng
token.

Corpus phải là **cùng tập chunk** mà Task 4 đã index. Nếu hai retriever chunk
riêng, RRF sẽ gộp hai bảng xếp hạng trỏ tới hai tập ID khác nhau và không có ID
nào trùng để cộng điểm. Vì vậy ``CORPUS`` đọc từ ``data/index/chunks.json`` do
Task 4 ghi ra.

Tokenizer cố tình đơn giản: tiếng Việt viết rời từng âm tiết nên tách theo
khoảng trắng đã đủ tốt, và nó giữ được các token số/mã mà word-segmenter hay
cắt nhầm.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CHUNK_CACHE = Path(__file__).parent.parent / "data" / "index" / "chunks.json"

CORPUS: list[dict] = []

# Giữ chữ, số và dấu tiếng Việt; bỏ dấu câu. Không bỏ dấu thanh — "học" và
# "hoc" là hai token khác nhau và tài liệu luôn viết có dấu.
_TOKEN = re.compile(r"\w+", re.UNICODE)

_index_cache: dict[tuple[int, int], Any] = {}


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _load_corpus_from_disk() -> list[dict]:
    if not CHUNK_CACHE.exists():
        return []
    return json.loads(CHUNK_CACHE.read_text(encoding="utf-8"))


def _active_corpus() -> list[dict]:
    """Trả về corpus đang dùng.

    Đọc module-global ``CORPUS`` ở thời điểm gọi chứ không bắt giữ lúc import,
    để test có thể thay corpus bằng monkeypatch.
    """
    if CORPUS:
        return CORPUS
    loaded = _load_corpus_from_disk()
    if loaded:
        CORPUS.extend(loaded)
    return CORPUS


def corpus_size() -> int:
    """Số chunk BM25 đang phục vụ — UI hiển thị con số này ở thanh trạng thái."""
    return len(_active_corpus())


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([_tokenize(item["content"]) for item in corpus])


def _cached_index(corpus: list[dict]):
    """Dựng index một lần cho mỗi corpus — tokenize lại mỗi query rất tốn."""
    key = (id(corpus), len(corpus))
    if key not in _index_cache:
        _index_cache[key] = build_bm25_index(corpus)
    return _index_cache[key]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    corpus = _active_corpus()
    tokens = _tokenize(query)
    if not corpus or not tokens or top_k <= 0:
        return []

    scores = _cached_index(corpus).get_scores(tokens)
    order = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)

    wanted = set(tokens)
    results: list[dict] = []
    for position in order[:top_k]:
        score = float(scores[position])
        item = corpus[position]
        # Điều kiện loại bỏ là "không khớp token nào", KHÔNG phải "score <= 0".
        #
        # BM25Okapi tính idf = log(N - n + 0.5) - log(n + 0.5). Trên corpus rất
        # nhỏ, một token xuất hiện ở đúng một nửa số tài liệu cho idf = 0, nên
        # MỌI score đều bằng 0 kể cả khi tài liệu khớp hoàn hảo. Lọc theo score
        # sẽ trả về rỗng. Lọc theo độ khớp token vẫn đúng ý định (bỏ chunk
        # không liên quan để khỏi làm loãng RRF) mà không phụ thuộc kích thước
        # corpus.
        if score <= 0.0 and not (wanted & set(_tokenize(item["content"]))):
            continue
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": score,
                "metadata": dict(item["metadata"]),
                "retrieval_method": "bm25",
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("ký túc xá 214 phòng", top_k=3):
        print(f"{result['score']:.3f}  {result['id']}")
