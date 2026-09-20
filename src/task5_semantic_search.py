"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output theo SearchResult, sort giảm dần và không quá top_k.

``embed_texts`` và ``get_collection`` được import vào namespace của module này
(không gọi qua ``task4.``) để test có thể monkeypatch chúng mà không cần chạm
tới ChromaDB thật.
"""

from __future__ import annotations

from .task4_chunking_indexing import as_query, embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    return _search(query, top_k, where=None)


def semantic_search_filtered(query: str, top_k: int = 10, where: dict | None = None) -> list[dict]:
    """Như ``semantic_search`` nhưng giới hạn trong một tập metadata.

    Tách thành hàm riêng vì contract khoá chữ ký của ``semantic_search`` ở đúng
    hai tham số ``(query, top_k)``. Dùng cho chế độ cân bằng theo doc_type ở
    Task 9: corpus của nhóm lệch 90.6% legal / 9.4% news, nên một truy vấn về
    dịch vụ sinh viên gần như luôn bị các chunk quy chế lấn át.
    """
    return _search(query, top_k, where=where)


def _search(query: str, top_k: int, where: dict | None) -> list[dict]:
    if not query or not query.strip() or top_k <= 0:
        return []

    query_vector = embed_texts([as_query(query)])[0]
    kwargs = {
        "query_embeddings": [query_vector],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    response = get_collection().query(**kwargs)

    results: list[dict] = []
    seen: set[str] = set()
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        # Chroma có thể trả cùng một ID hai lần nếu index bị ghi trùng; contract
        # cấm ID lặp trong kết quả nên lọc ngay tại đây.
        if item_id in seen:
            continue
        seen.add(item_id)
        results.append(
            {
                "id": item_id,
                "content": content,
                # Collection dùng cosine distance, nên similarity = 1 - distance.
                # Kẹp về 0 để không bao giờ trả score âm cho tầng fallback.
                "score": max(0.0, 1.0 - float(distance)),
                "metadata": dict(metadata),
                "retrieval_method": "dense",
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("Ký túc xá có bao nhiêu phòng?", top_k=3):
        print(f"{result['score']:.3f}  {result['id']}")
