"""
Task 7 — Reciprocal Rank Fusion.

Vì sao không cộng thẳng score: cosine similarity nằm trong [0, 1] còn BM25 là
số dương không chặn trên (ở corpus này thường 2–20). Cộng trực tiếp thì BM25
nuốt chửng dense. Chuẩn hoá min-max cũng không cứu được, vì nó phụ thuộc vào
chính tập kết quả của query đó nên thang đo đổi theo từng câu hỏi.

RRF bỏ hẳn giá trị score và chỉ dùng **thứ hạng**:

    RRF(d) = sum over lists of 1 / (k + rank(d)),  rank bắt đầu từ 1

Hằng số k = 60 (mặc định trong bài báo gốc của Cormack 2009) làm phẳng phần
đầu bảng: chênh lệch giữa hạng 1 và hạng 2 nhỏ, nên một tài liệu được **cả
hai** retriever xếp hạng trung bình sẽ thắng một tài liệu chỉ một bên xếp hạng
nhất. Đó chính là hành vi mong muốn của hybrid search.

Hệ quả quan trọng: RRF score KHÔNG phải độ tương đồng. Nó chỉ so sánh được
trong phạm vi một lần fuse, và không bao giờ được đem so với threshold fallback
(xem Task 9).
"""

from __future__ import annotations


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item["id"]
            # Một ID chỉ được tính điểm một lần cho mỗi bảng xếp hạng.
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            items.setdefault(item_id, item)

    ranked_ids = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)

    results: list[dict] = []
    for item_id in ranked_ids[: max(top_k, 0)]:
        result = dict(items[item_id])
        result["metadata"] = dict(items[item_id]["metadata"])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


def rrf_rank_table(ranked_lists: dict[str, list[dict]], k: int = 60) -> dict[str, dict]:
    """Bảng thứ hạng từng retriever cho một ID — dùng để vẽ biểu đồ giải thích.

    Không tham gia pipeline; tách riêng để ``rerank_rrf`` giữ đúng chữ ký mà
    contract yêu cầu.
    """
    table: dict[str, dict] = {}
    for name, ranked_list in ranked_lists.items():
        for rank, item in enumerate(ranked_list, start=1):
            entry = table.setdefault(
                item["id"],
                {"id": item["id"], "metadata": item["metadata"], "rrf_score": 0.0},
            )
            if f"{name}_rank" in entry:
                continue
            entry[f"{name}_rank"] = rank
            entry[f"{name}_score"] = float(item["score"])
            entry["rrf_score"] += 1.0 / (k + rank)
    return table


if __name__ == "__main__":
    print("Implement rerank_rrf, then run contract tests.")
