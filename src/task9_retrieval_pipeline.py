"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng: dense + BM25 chạy song song về mặt logic, RRF gộp đúng một lần, rồi một
cổng ngưỡng quyết định có cần fallback vectorless hay không.

**Quy tắc quan trọng nhất của module này:** cổng ngưỡng so sánh với *cosine
score gốc của dense*, không phải RRF score. Hai đại lượng này không cùng thang.
Cosine nằm trong [0, 1] và có nghĩa tuyệt đối — 0.2 nghĩa là "không có gì trong
kho gần câu hỏi này". RRF score là tổng các 1/(60+rank), luôn rơi quanh
0.016–0.033 dù kết quả tốt hay tệ, vì nó chỉ mã hoá thứ hạng. Đem RRF score so
với 0.3 thì fallback sẽ kích hoạt ở *mọi* câu hỏi, kể cả câu trả lời hoàn hảo.

``retrieve_with_trace`` trả thêm một bản ghi các bước để UI vẽ lại quyết định.
``retrieve`` giữ đúng chữ ký mà contract yêu cầu.
"""

from __future__ import annotations

import os
import time

from .task4_chunking_indexing import EMBEDDING_MODEL
from .task5_semantic_search import semantic_search, semantic_search_filtered
from .task6_lexical_search import corpus_size, lexical_search
from .task7_reranking import rerank_rrf, rrf_rank_table
from .task8_pageindex_vectorless import pageindex_search, provider_name


SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or 0.3)
DEFAULT_TOP_K = 5
RRF_K = 60

# Lấy rộng hơn top_k trước khi fuse: RRF chỉ đổi được thứ hạng của những ứng
# viên nó nhìn thấy. Nếu mỗi nhánh chỉ đưa top_k thì chunk mà dense xếp hạng 7
# và BM25 xếp hạng 1 sẽ không bao giờ có cơ hội.
CANDIDATE_MULTIPLIER = 2


def _milliseconds(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _title_of(item: dict) -> str:
    metadata = item.get("metadata") or {}
    return metadata.get("title") or metadata.get("source") or item.get("id", "")


DOC_TYPES = ("legal", "news")


def _execute(
    query: str,
    top_k: int,
    score_threshold: float,
    use_reranking: bool,
    balanced: bool = False,
) -> tuple[list[dict], dict]:
    """Chạy pipeline và ghi lại từng bước.

    ``balanced`` bật chế độ cân bằng theo ``doc_type``. Lý do: corpus của nhóm
    lệch 90.6% legal / 9.4% news, và riêng Sổ tay sinh viên K60 đã chiếm 34.6%
    toàn bộ index. Một câu hỏi về dịch vụ sinh viên phải cạnh tranh với 914
    chunk quy chế, nên trang đúng (7 chunk) gần như không bao giờ lọt top-5.

    Cách sửa khai thác đúng bản chất của RRF: thay vì một bảng xếp hạng dense
    toàn cục, chạy dense RIÊNG trong từng doc_type rồi đưa cả hai bảng vào RRF.
    Một chunk news hạng 1 trong nhóm news nhận 1/(60+1) dù xét toàn cục nó chỉ
    đứng hạng 200. RRF chỉ quan tâm thứ hạng, nên nó gộp được hai bảng có kích
    thước rất chênh lệch mà không cần chuẩn hoá score.
    """
    overall = time.perf_counter()
    candidates = max(top_k * CANDIDATE_MULTIPLIER, top_k)

    started = time.perf_counter()
    dense = semantic_search(query, top_k=candidates)
    dense_by_type: list[list[dict]] = []
    if balanced:
        dense_by_type = [
            semantic_search_filtered(query, top_k=candidates, where={"doc_type": doc_type})
            for doc_type in DOC_TYPES
        ]
    dense_ms = _milliseconds(started)

    started = time.perf_counter()
    sparse = lexical_search(query, top_k=candidates)
    sparse_ms = _milliseconds(started)

    started = time.perf_counter()
    if use_reranking:
        lists = [*dense_by_type, sparse] if balanced else [dense, sparse]
        hybrid = rerank_rrf(lists, top_k=top_k)
    else:
        hybrid = dense[:top_k]
    fusion_ms = _milliseconds(started)

    best_dense_score = float(dense[0]["score"]) if dense else 0.0

    fallback_stage = {
        "name": "fallback",
        "label": "Vectorless fallback",
        "triggered": False,
        "reason": "",
        "provider": provider_name(),
        "error": None,
        "count": 0,
        "elapsed_ms": 0.0,
        "results": [],
    }

    chosen = hybrid[:top_k]
    decision = "hybrid"
    decision_reason = (
        f"best dense cosine {best_dense_score:.3f} >= ngưỡng {score_threshold:.2f} "
        f"→ dùng kết quả hybrid"
    )

    if best_dense_score < score_threshold:
        fallback_stage["triggered"] = True
        fallback_stage["reason"] = (
            f"best dense cosine {best_dense_score:.3f} < ngưỡng {score_threshold:.2f}"
        )
        started = time.perf_counter()
        try:
            recovered = pageindex_search(query, top_k=top_k)
            fallback_stage["elapsed_ms"] = _milliseconds(started)
            fallback_stage["results"] = recovered
            fallback_stage["count"] = len(recovered)
            if recovered:
                chosen = recovered[:top_k]
                decision = "pageindex"
                decision_reason = (
                    f"{fallback_stage['reason']} → chuyển sang tuyến vectorless "
                    f"({fallback_stage['provider']})"
                )
            else:
                decision_reason = (
                    f"{fallback_stage['reason']} → fallback không tìm được gì, "
                    f"giữ kết quả hybrid"
                )
        except Exception as error:
            # Dịch vụ ngoài chết không được phép làm chatbot chết theo.
            fallback_stage["elapsed_ms"] = _milliseconds(started)
            fallback_stage["error"] = f"{type(error).__name__}: {error}"
            decision_reason = (
                f"{fallback_stage['reason']} → fallback lỗi "
                f"({type(error).__name__}), giữ kết quả hybrid"
            )

    table = rrf_rank_table({"dense": dense, "bm25": sparse}, k=RRF_K)
    final_rank = {item["id"]: rank for rank, item in enumerate(chosen, start=1)}
    rank_movement = [
        {
            "id": entry["id"],
            "title": (entry.get("metadata") or {}).get("title", entry["id"]),
            "source": (entry.get("metadata") or {}).get("source", ""),
            "dense_rank": entry.get("dense_rank"),
            "bm25_rank": entry.get("bm25_rank"),
            "final_rank": final_rank.get(entry["id"]),
            "rrf_score": round(entry["rrf_score"], 6),
            "dense_score": entry.get("dense_score"),
            "bm25_score": entry.get("bm25_score"),
        }
        for entry in table.values()
    ]
    rank_movement.sort(key=lambda row: (row["final_rank"] is None, row["final_rank"] or 99, -row["rrf_score"]))

    trace = {
        "query": query,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "use_reranking": use_reranking,
        "stages": [
            {
                "name": "dense",
                "label": "Dense (vector)",
                "count": len(dense),
                "elapsed_ms": dense_ms,
                "results": dense,
            },
            {
                "name": "bm25",
                "label": "BM25 (keyword)",
                "count": len(sparse),
                "elapsed_ms": sparse_ms,
                "results": sparse,
            },
            {
                "name": "rrf",
                "label": "RRF fusion" if use_reranking else "Dense only (RRF tắt)",
                "count": len(hybrid),
                "elapsed_ms": fusion_ms,
                "results": hybrid,
                "k": RRF_K,
            },
            fallback_stage,
        ],
        "best_dense_score": best_dense_score,
        "decision": decision,
        "decision_reason": decision_reason,
        "rank_movement": rank_movement,
        # Danh sách ID thực sự được trả về. Cần thiết vì khi fallback chạy thì
        # mọi final_rank trong rank_movement đều None — ID của pageindex không
        # nằm trong bảng RRF. Không có key này thì một trace đã lưu không tự mô
        # tả được kết quả cuối, phải đọc thêm GenerationResult.sources.
        "final_ids": [item["id"] for item in chosen],
        "index": {
            "embedding_model": EMBEDDING_MODEL,
            "chunks": corpus_size(),
            "balanced": balanced,
        },
        "elapsed_ms": _milliseconds(overall),
    }
    return chosen, trace


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    results, _ = _execute(query, top_k, score_threshold, use_reranking)
    return results


def retrieve_with_trace(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    balanced: bool = False,
) -> tuple[list[dict], dict]:
    """Như ``retrieve`` nhưng kèm bản ghi từng bước để UI vẽ lại quyết định."""
    return _execute(query, top_k, score_threshold, use_reranking, balanced)


if __name__ == "__main__":
    found, record = retrieve_with_trace("Ai quản lý ký túc xá của trường?", top_k=3)
    print(record["decision_reason"])
    for item in found:
        print(f"  {item['score']:.4f}  {item['retrieval_method']:<9} {item['id']}")
