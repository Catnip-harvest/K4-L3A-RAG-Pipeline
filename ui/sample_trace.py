"""Hard-coded, realistic pipeline traces used by DEMO_MODE.

WHY this file exists: the UI is built against a backend contract, and a live
demo cannot depend on an API key, a warm Chroma index or the network. These
fixtures are shaped exactly like the real ``retrieve_with_trace`` /
``generate_with_trace`` output, so every graphic in the app is exercised by them
and the demo keeps a guaranteed-working fallback.

The content mirrors the real corpus in ``data/standardized`` — student services
and training regulations of Trường Đại học Giao thông Vận tải (UTC).

Two scenarios are provided because the rubric grades both halves of the story:

``grounded``  - best dense score clears the threshold, RRF fusion wins, the
                answer carries verifiable citations.
``refusal``   - best dense score falls short, the vectorless fallback is tried
                and fails, and the pipeline declines to answer.

The RRF scores are not invented: they are the real ``sum(1 / (k + rank))`` with
k = 60 for the ranks listed, so the slope chart and the score chart agree with
each other and with anything an attentive student recomputes by hand.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

Scenario = Literal["grounded", "refusal"]

RRF_K = 60


def _rrf(*ranks: int | None) -> float:
    """Reciprocal-rank-fusion score for a document's ranks across input lists."""
    return round(sum(1.0 / (RRF_K + r) for r in ranks if r is not None), 6)


# --- Corpus fixtures ---------------------------------------------------------
# ids follow task4's convention: "<folder>/<file>.md::chunk-<n>".
_CHUNKS: dict[str, dict[str, Any]] = {
    "news/ban-quan-ly-ky-tuc-xa.md::chunk-1": {
        "title": "Ban Quản lý ký túc xá",
        "source": "ban-quan-ly-ky-tuc-xa.md",
        "doc_type": "news",
        "url": "https://www.utc.edu.vn/gioi-thieu/ban-quan-ly-ky-tuc-xa",
        "chunk_index": 1,
        "content": (
            "Ban quản lý Ký túc xá có chức năng giúp Hiệu trưởng quản lý toàn diện "
            "khu ký túc xá của trường; tổ chức phục vụ học tập, sinh hoạt của sinh "
            "viên được bố trí ở nội trú và đảm bảo trật tự an toàn, vệ sinh, cảnh "
            "quan. Địa chỉ: số 99 đường Nguyễn Chí Thanh, Hà Nội. Điện thoại: "
            "024.37741734. Ký túc xá có 03 khối nhà từ 4–5 tầng với tổng số 214 "
            "phòng khép kín, sức chứa 1500 sinh viên."
        ),
    },
    "legal/so-tay-sinh-vien-k60.md::chunk-18": {
        "title": "Sổ tay sinh viên K60 — Quy định về sinh viên nội trú",
        "source": "so-tay-sinh-vien-k60.md",
        "doc_type": "legal",
        "url": "https://fme.utc.edu.vn/sites/fme.utc.edu.vn/files/SO TAY SINH VIEN K60.pdf",
        "chunk_index": 18,
        "content": (
            "Sinh viên có nhu cầu ở nội trú nộp đơn đăng ký ký túc xá theo mẫu của "
            "nhà trường trong thời gian thông báo đầu mỗi năm học. Thứ tự ưu tiên xét "
            "duyệt: sinh viên thuộc diện chính sách, con thương binh liệt sĩ, sinh "
            "viên hộ nghèo, sinh viên ở xa và sinh viên năm thứ nhất. Sinh viên nội "
            "trú phải ký cam kết thực hiện nội quy ký túc xá, đóng tiền phòng theo "
            "học kỳ và đăng ký tạm trú với công an phường sở tại."
        ),
    },
    "news/phong-cong-tac-chinh-tri-va-sinh-vien.md::chunk-0": {
        "title": "Phòng Công tác chính trị và Sinh viên",
        "source": "phong-cong-tac-chinh-tri-va-sinh-vien.md",
        "doc_type": "news",
        "url": "https://www.utc.edu.vn/gioi-thieu/phong-cong-tac-chinh-tri-va-sinh-vien",
        "chunk_index": 0,
        "content": (
            "Phòng Công tác chính trị và Sinh viên là đầu mối tiếp nhận hồ sơ đăng ký "
            "nội trú, xét duyệt đối tượng ưu tiên và tham mưu cho Hiệu trưởng ban hành "
            "quyết định bố trí chỗ ở. Phòng cũng giải quyết các chế độ chính sách, học "
            "bổng, bảo hiểm y tế và xác nhận giấy tờ cho sinh viên."
        ),
    },
    "news/phong-ke-hoach-tai-chinh.md::chunk-2": {
        "title": "Phòng Kế hoạch – Tài chính",
        "source": "phong-ke-hoach-tai-chinh.md",
        "doc_type": "news",
        "url": "https://www.utc.edu.vn/gioi-thieu/phong-ke-hoach-tai-chinh",
        "chunk_index": 2,
        "content": (
            "Phòng Kế hoạch – Tài chính tổ chức thu học phí, lệ phí ký túc xá và các "
            "khoản thu khác theo thông báo từng học kỳ. Sinh viên nộp tiền phòng ký "
            "túc xá qua tài khoản ngân hàng của trường theo mã sinh viên; biên lai "
            "điện tử được tra cứu trên cổng thông tin sinh viên. Ngoài tiền phòng, "
            "sinh viên nội trú thanh toán tiền điện, nước theo chỉ số thực tế hằng tháng."
        ),
    },
    "news/phong-bao-ve.md::chunk-1": {
        "title": "Phòng Bảo vệ",
        "source": "phong-bao-ve.md",
        "doc_type": "news",
        "url": "https://www.utc.edu.vn/gioi-thieu/phong-bao-ve",
        "chunk_index": 1,
        "content": (
            "Phòng Bảo vệ bảo đảm an ninh trật tự trong khuôn viên trường và khu ký "
            "túc xá, kiểm soát người và phương tiện ra vào, phối hợp với Ban quản lý "
            "ký túc xá xử lý các vụ việc mất an ninh trong giờ giới nghiêm."
        ),
    },
    "legal/de-an-tuyen-sinh-2025.md::chunk-9": {
        "title": "Đề án tuyển sinh 2025 — Điều kiện ăn ở và chính sách hỗ trợ",
        "source": "de-an-tuyen-sinh-2025.md",
        "doc_type": "legal",
        "url": "https://tuyensinh.utc.edu.vn/de-an-tuyen-sinh-2025",
        "chunk_index": 9,
        "content": (
            "Nhà trường bố trí ký túc xá cho tân sinh viên có nhu cầu, ưu tiên thí "
            "sinh ở xa và thuộc diện chính sách. Mức thu ký túc xá được công bố công "
            "khai cùng thông báo nhập học; nhà trường cam kết đáp ứng khoảng 1500 chỗ "
            "ở trong năm học 2025–2026."
        ),
    },
    "legal/so-tay-sinh-vien-k60.md::chunk-31": {
        "title": "Sổ tay sinh viên K60 — Quy định về kỷ luật sinh viên",
        "source": "so-tay-sinh-vien-k60.md",
        "doc_type": "legal",
        "url": "https://fme.utc.edu.vn/sites/fme.utc.edu.vn/files/SO TAY SINH VIEN K60.pdf",
        "chunk_index": 31,
        "content": (
            "Các hình thức kỷ luật gồm khiển trách, cảnh cáo, đình chỉ học tập có thời "
            "hạn và buộc thôi học. Sinh viên nội trú vi phạm nội quy ký túc xá từ hai "
            "lần trở lên có thể bị đình chỉ quyền ở nội trú trong học kỳ tiếp theo."
        ),
    },
    "news/tram-y-te.md::chunk-0": {
        "title": "Trạm Y tế",
        "source": "tram-y-te.md",
        "doc_type": "news",
        "url": "https://www.utc.edu.vn/gioi-thieu/tram-y-te",
        "chunk_index": 0,
        "content": (
            "Trạm Y tế đặt trong khu ký túc xá, thực hiện sơ cấp cứu, khám chữa bệnh "
            "ban đầu cho sinh viên và cán bộ, quản lý hồ sơ bảo hiểm y tế sinh viên và "
            "tổ chức khám sức khoẻ đầu khoá."
        ),
    },
}


def _result(chunk_id: str, score: float, method: str) -> dict[str, Any]:
    """Build a SearchResult from the fixture corpus."""
    chunk = _CHUNKS[chunk_id]
    return {
        "id": chunk_id,
        "content": chunk["content"],
        "score": score,
        "metadata": {
            "source": chunk["source"],
            "title": chunk["title"],
            "doc_type": chunk["doc_type"],
            "url": chunk["url"],
            "chunk_index": chunk["chunk_index"],
        },
        "retrieval_method": method,
    }


# --- Scenario: grounded ------------------------------------------------------
# Dense and BM25 each return 6 candidates and 4 of them overlap. The final order
# below is the arithmetic consequence of these ranks, not a hand-picked list:
# the BM25-only "Đề án tuyển sinh" chunk survives into the top-5 while the
# dense-only "Phòng Bảo vệ" chunk just misses. That contrast is the whole point
# of the slope chart.
# Cosine values sit in 0.86-0.92 on purpose: intfloat/multilingual-e5-small
# packs every embedding into a narrow cone, which is exactly why the team
# calibrated SCORE_THRESHOLD to 0.853 instead of the starter's 0.30.
_GROUNDED_DENSE_RANKS: list[tuple[str, float]] = [
    ("news/ban-quan-ly-ky-tuc-xa.md::chunk-1", 0.9118),
    ("news/phong-cong-tac-chinh-tri-va-sinh-vien.md::chunk-0", 0.8974),
    ("legal/so-tay-sinh-vien-k60.md::chunk-18", 0.8903),
    ("news/phong-bao-ve.md::chunk-1", 0.8836),
    ("news/phong-ke-hoach-tai-chinh.md::chunk-2", 0.8752),
    ("legal/so-tay-sinh-vien-k60.md::chunk-31", 0.8661),
]
_GROUNDED_BM25_RANKS: list[tuple[str, float]] = [
    ("legal/so-tay-sinh-vien-k60.md::chunk-18", 9.41),
    ("news/ban-quan-ly-ky-tuc-xa.md::chunk-1", 8.15),
    ("legal/de-an-tuyen-sinh-2025.md::chunk-9", 7.02),
    ("news/phong-ke-hoach-tai-chinh.md::chunk-2", 6.44),
    ("news/phong-cong-tac-chinh-tri-va-sinh-vien.md::chunk-0", 5.03),
    ("news/tram-y-te.md::chunk-0", 4.10),
]

_GROUNDED_QUERY = "Sinh viên muốn ở ký túc xá thì đăng ký ở đâu và phải đóng những khoản gì?"

_GROUNDED_ANSWER = (
    "Tóm tắt quy trình đăng ký ở ký túc xá và các khoản phải nộp:\n\n"
    "**1. Nộp hồ sơ ở đâu.** Sinh viên nộp đơn đăng ký ký túc xá theo mẫu của nhà "
    "trường trong thời gian thông báo đầu mỗi năm học [2]. Đầu mối tiếp nhận hồ sơ "
    "và xét duyệt đối tượng ưu tiên là **Phòng Công tác chính trị và Sinh viên** [3]; "
    "đơn vị quản lý trực tiếp khu nội trú là **Ban Quản lý ký túc xá**, số 99 Nguyễn "
    "Chí Thanh, điện thoại 024.37741734 [1].\n\n"
    "**2. Thứ tự ưu tiên.** Sinh viên diện chính sách, con thương binh liệt sĩ, hộ "
    "nghèo, sinh viên ở xa và sinh viên năm thứ nhất được xét trước [2]. Tân sinh "
    "viên có nhu cầu cũng được bố trí chỗ ở, quy mô khoảng 1500 chỗ [5].\n\n"
    "**3. Các khoản phải nộp.** Tiền phòng đóng theo học kỳ [2], nộp qua tài khoản "
    "ngân hàng của trường theo mã sinh viên [4]. Ngoài tiền phòng còn có tiền điện "
    "và nước tính theo chỉ số thực tế hằng tháng [4].\n\n"
    "**4. Nghĩa vụ kèm theo.** Sinh viên nội trú phải ký cam kết thực hiện nội quy ký "
    "túc xá và đăng ký tạm trú với công an phường sở tại [2]."
)


def _movement(
    dense_ranks: list[tuple[str, float]],
    bm25_ranks: list[tuple[str, float]],
) -> list[dict[str, Any]]:
    """Fuse two ranked lists into the rank_movement table the UI draws."""
    dense_rank = {cid: i + 1 for i, (cid, _) in enumerate(dense_ranks)}
    bm25_rank = {cid: i + 1 for i, (cid, _) in enumerate(bm25_ranks)}
    dense_score = dict(dense_ranks)
    bm25_score = dict(bm25_ranks)

    fused = sorted(
        set(dense_rank) | set(bm25_rank),
        key=lambda cid: (
            -_rrf(dense_rank.get(cid), bm25_rank.get(cid)),
            -dense_score.get(cid, 0.0),
        ),
    )
    return [
        {
            "id": cid,
            "title": _CHUNKS[cid]["title"],
            "source": _CHUNKS[cid]["source"],
            "dense_rank": dense_rank.get(cid),
            "bm25_rank": bm25_rank.get(cid),
            "final_rank": i + 1,
            "rrf_score": _rrf(dense_rank.get(cid), bm25_rank.get(cid)),
            "dense_score": dense_score.get(cid),
            "bm25_score": bm25_score.get(cid),
        }
        for i, cid in enumerate(fused)
    ]


def _grounded_trace() -> dict[str, Any]:
    dense_results = [_result(c, s, "dense") for c, s in _GROUNDED_DENSE_RANKS]
    bm25_results = [_result(c, s, "bm25") for c, s in _GROUNDED_BM25_RANKS]
    movement = _movement(_GROUNDED_DENSE_RANKS, _GROUNDED_BM25_RANKS)
    rrf_results = [_result(m["id"], m["rrf_score"], "hybrid") for m in movement]

    return {
        "query": _GROUNDED_QUERY,
        "top_k": 5,
        "score_threshold": 0.30,
        "use_reranking": True,
        "stages": [
            {"name": "dense", "label": "Dense (vector)", "count": len(dense_results),
             "elapsed_ms": 128.4, "results": dense_results},
            {"name": "bm25", "label": "BM25 (keyword)", "count": len(bm25_results),
             "elapsed_ms": 6.2, "results": bm25_results},
            {"name": "rrf", "label": "RRF fusion", "count": len(rrf_results),
             "elapsed_ms": 1.7, "results": rrf_results, "k": RRF_K},
            {"name": "fallback", "label": "Vectorless fallback", "triggered": False,
             "reason": "best_dense_score ≥ score_threshold", "provider": "pageindex",
             "error": None, "count": 0, "elapsed_ms": 0.0, "results": []},
        ],
        "best_dense_score": _GROUNDED_DENSE_RANKS[0][1],
        "decision": "hybrid",
        "decision_reason": "Điểm dense cao nhất vượt ngưỡng nên giữ kết quả hybrid RRF.",
        "rank_movement": movement,
        "elapsed_ms": 136.3,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "llm_elapsed_ms": 1843.0,
        "context_chars": 2614,
        "prompt_chars": 2981,
        "refused": False,
        "reorder_map": [],
        "citations": [],
    }


# --- Scenario: refusal -------------------------------------------------------
_REFUSAL_QUERY = "Giá vé máy bay Hà Nội – Đà Nẵng cho sinh viên là bao nhiêu?"

# Out-of-domain queries still score 0.82-0.85 with this model — the whole
# point of the calibration finding. Below 0.853, so the gate fires.
_REFUSAL_DENSE_RANKS: list[tuple[str, float]] = [
    ("news/phong-ke-hoach-tai-chinh.md::chunk-2", 0.8412),
    ("legal/de-an-tuyen-sinh-2025.md::chunk-9", 0.8357),
    ("news/phong-cong-tac-chinh-tri-va-sinh-vien.md::chunk-0", 0.8291),
    ("news/tram-y-te.md::chunk-0", 0.8240),
]
_REFUSAL_BM25_RANKS: list[tuple[str, float]] = [
    ("legal/de-an-tuyen-sinh-2025.md::chunk-9", 2.41),
    ("news/phong-ke-hoach-tai-chinh.md::chunk-2", 1.88),
    ("news/tram-y-te.md::chunk-0", 1.02),
]

_REFUSAL_ANSWER = (
    "Tôi không thể xác minh thông tin này từ nguồn hiện có. Bộ tài liệu của hệ thống "
    "chỉ gồm quy chế đào tạo và thông tin dịch vụ sinh viên của Trường Đại học Giao "
    "thông Vận tải, không chứa dữ liệu về giá vé máy bay. Để tránh bịa thông tin, "
    "pipeline chủ động từ chối thay vì suy đoán."
)


def _refusal_trace() -> dict[str, Any]:
    dense_results = [_result(c, s, "dense") for c, s in _REFUSAL_DENSE_RANKS]
    bm25_results = [_result(c, s, "bm25") for c, s in _REFUSAL_BM25_RANKS]
    movement = _movement(_REFUSAL_DENSE_RANKS, _REFUSAL_BM25_RANKS)
    rrf_results = [_result(m["id"], m["rrf_score"], "hybrid") for m in movement]

    return {
        "query": _REFUSAL_QUERY,
        "top_k": 5,
        "score_threshold": 0.30,
        "use_reranking": True,
        "stages": [
            {"name": "dense", "label": "Dense (vector)", "count": len(dense_results),
             "elapsed_ms": 119.8, "results": dense_results},
            {"name": "bm25", "label": "BM25 (keyword)", "count": len(bm25_results),
             "elapsed_ms": 5.4, "results": bm25_results},
            {"name": "rrf", "label": "RRF fusion", "count": len(rrf_results),
             "elapsed_ms": 1.4, "results": rrf_results, "k": RRF_K},
            {"name": "fallback", "label": "Vectorless fallback", "triggered": True,
             "reason": "best_dense_score < score_threshold", "provider": "pageindex",
             "error": "PAGEINDEX_API_KEY chưa được cấu hình — bỏ qua fallback.",
             "count": 0, "elapsed_ms": 42.1, "results": []},
        ],
        "best_dense_score": _REFUSAL_DENSE_RANKS[0][1],
        "decision": "pageindex",
        "decision_reason": (
            "Điểm dense cao nhất dưới ngưỡng nên pipeline chuyển sang PageIndex; "
            "fallback không trả về kết quả nên hệ thống từ chối trả lời."
        ),
        "rank_movement": movement,
        "elapsed_ms": 168.7,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "llm_elapsed_ms": 0.0,
        "context_chars": 0,
        "prompt_chars": 0,
        "refused": True,
        "reorder_map": [],
        "citations": [],
    }


def _reorder_map(final_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror task10's ``reorder_for_llm``: even ranks first, odd ranks reversed."""
    positions = {r["id"]: i + 1 for i, r in enumerate(final_results)}
    reordered = final_results[::2] + final_results[1::2][::-1]
    return [
        {"id": r["id"], "from_rank": positions[r["id"]], "to_rank": i + 1}
        for i, r in enumerate(reordered)
    ]


def _apply_settings(
    trace: dict[str, Any],
    top_k: int,
    score_threshold: float,
    use_reranking: bool,
) -> dict[str, Any]:
    """Re-derive the demo trace under the sidebar settings.

    WHY: a demo whose graphics ignore the sliders teaches nothing. Moving the
    threshold must be able to flip the gate, and switching reranking off must
    visibly collapse the final ranking back onto the dense ranking — that is the
    clearest possible demonstration of what RRF contributes.
    """
    trace = copy.deepcopy(trace)
    trace["top_k"] = top_k
    trace["score_threshold"] = score_threshold
    trace["use_reranking"] = use_reranking

    stages = {s["name"]: s for s in trace["stages"]}
    dense_results: list[dict[str, Any]] = stages.get("dense", {}).get("results", [])

    if use_reranking:
        for i, row in enumerate(trace["rank_movement"]):
            row["final_rank"] = i + 1 if i < top_k else None
        final_results = stages.get("rrf", {}).get("results", [])[:top_k]
    else:
        # Dense-only: the final order IS the dense order and RRF never runs.
        dense_order = {r["id"]: i + 1 for i, r in enumerate(dense_results)}
        for row in trace["rank_movement"]:
            rank = dense_order.get(row["id"])
            row["final_rank"] = rank if rank is not None and rank <= top_k else None
        trace["rank_movement"].sort(
            key=lambda r: (r["final_rank"] is None, r["final_rank"] or 99, -r["rrf_score"])
        )
        rrf = stages.get("rrf")
        if rrf is not None:
            rrf["label"] = "Dense only (RRF tắt)"
            rrf["elapsed_ms"] = 0.0
            rrf["count"] = 0
            rrf["results"] = []
        final_results = [dict(r) for r in dense_results[:top_k]]

    best_dense = float(trace["best_dense_score"])
    fallback = stages.get("fallback")
    if best_dense >= score_threshold:
        trace["decision"] = "hybrid"
        trace["decision_reason"] = (
            f"Điểm dense cao nhất {best_dense:.4f} ≥ ngưỡng {score_threshold:.3f} → giữ "
            f"nhánh {'hybrid RRF' if use_reranking else 'dense'}."
        )
        if fallback is not None:
            fallback["triggered"] = False
            fallback["elapsed_ms"] = 0.0
            fallback["reason"] = "best_dense_score ≥ score_threshold"
    else:
        trace["decision"] = "pageindex"
        trace["decision_reason"] = (
            f"Điểm dense cao nhất {best_dense:.4f} < ngưỡng {score_threshold:.3f} → "
            "chuyển sang PageIndex (vectorless)."
        )
        if fallback is not None:
            fallback["triggered"] = True
            fallback["reason"] = "best_dense_score < score_threshold"
            fallback["elapsed_ms"] = fallback.get("elapsed_ms") or 42.1
            if fallback.get("error") is None and not fallback.get("results"):
                fallback["error"] = (
                    "PAGEINDEX_API_KEY chưa được cấu hình — bỏ qua fallback."
                )
            final_results = list(fallback.get("results") or [])[:top_k]

    trace["_final_results"] = final_results
    trace["elapsed_ms"] = round(
        sum(float(s.get("elapsed_ms") or 0.0) for s in trace["stages"]), 1
    )
    return trace


def sample_generation(
    query: str,
    *,
    top_k: int = 5,
    score_threshold: float = 0.30,
    use_reranking: bool = True,
    scenario: Scenario | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a ``(GenerationResult, trace)`` pair shaped like the real backend.

    ``scenario=None`` picks automatically from the query text so a presenter can
    trigger the refusal state live just by asking something out of scope.
    """
    if scenario is None:
        scenario = "refusal" if is_out_of_scope(query) else "grounded"

    base = _refusal_trace() if scenario == "refusal" else _grounded_trace()
    base["query"] = query or base["query"]
    trace = _apply_settings(base, top_k, score_threshold, use_reranking)

    final_results: list[dict[str, Any]] = trace.pop("_final_results", [])
    if scenario == "refusal" or not final_results:
        trace["refused"] = True
        trace["citations"] = []
        trace["reorder_map"] = []
        trace["llm_elapsed_ms"] = 0.0
        trace["context_chars"] = 0
        trace["prompt_chars"] = 0
        return (
            {"answer": _REFUSAL_ANSWER, "sources": [], "retrieval_source": "none"},
            trace,
        )

    # Citation numbers follow rank order, exactly as task10.citation_numbers does.
    answer = _GROUNDED_ANSWER
    kept = len(final_results)
    for number in range(kept + 1, 6):
        # Drop citation markers whose card is no longer on screen at this top_k,
        # so a marker can never point at a source the audience cannot see.
        answer = answer.replace(f" [{number}]", "")
    citations = [
        {
            "marker": f"[{i + 1}]",
            "source_id": r["id"],
            "title": r["metadata"]["title"],
            "matched": True,
        }
        for i, r in enumerate(final_results)
        if f"[{i + 1}]" in answer
    ]

    trace["refused"] = False
    trace["citations"] = citations
    trace["reorder_map"] = _reorder_map(final_results)
    trace["context_chars"] = sum(len(r["content"]) for r in final_results)
    trace["prompt_chars"] = trace["context_chars"] + 612
    trace["llm_elapsed_ms"] = 1843.0

    return (
        {
            "answer": answer,
            "sources": final_results,
            "retrieval_source": "hybrid",
        },
        trace,
    )


def sample_retrieval(
    query: str,
    *,
    top_k: int = 5,
    score_threshold: float = 0.30,
    use_reranking: bool = True,
    scenario: Scenario | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieval-only counterpart of :func:`sample_generation`."""
    result, trace = sample_generation(
        query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
        scenario=scenario,
    )
    return result["sources"], trace


# Keyword check used only to auto-select the demo scenario. Deliberately narrow:
# "vay vốn" and "học bổng" are IN the corpus, so they must not appear here.
_OUT_OF_SCOPE_HINTS = (
    "vé máy bay", "bitcoin", "thời tiết", "chứng khoán", "bóng đá", "giá vàng",
    "tỷ giá", "crypto", "harvard", "thú cưng", "nấu ăn", "điện thoại iphone",
)


def is_out_of_scope(query: str) -> bool:
    """Cheap keyword check used only to auto-select the demo scenario."""
    lowered = (query or "").lower()
    return any(hint in lowered for hint in _OUT_OF_SCOPE_HINTS)


SAMPLE_QUESTIONS: tuple[str, ...] = (
    _GROUNDED_QUERY,
    "Ai quản lý ký túc xá của trường và liên hệ ở đâu?",
    "Sinh viên nội trú vi phạm nội quy thì bị xử lý thế nào?",
    _REFUSAL_QUERY,
)

CORPUS_STATS: dict[str, Any] = {
    "documents": 14,
    "legal": 4,
    "news": 10,
    "chunks": 1009,
    "embedding_model": "intfloat/multilingual-e5-small",
    "vector_store": "ChromaDB",
    "threshold": 0.853,
}
