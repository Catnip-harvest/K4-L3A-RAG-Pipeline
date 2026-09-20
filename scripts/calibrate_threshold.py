"""Hiệu chỉnh SCORE_THRESHOLD bằng dữ liệu, không bằng phỏng đoán.

Repo mặc định 0.3. Con số đó đến từ trực giác "cosine dưới 0.3 là không liên
quan", vốn đúng với các model có embedding trải rộng. Model của nhóm là
``multilingual-e5-small`` và nó KHÔNG như vậy: toàn bộ embedding nằm trong một
hình nón hẹp, nên mọi cosine đều rơi vào dải 0.78–0.90 kể cả với câu hỏi hoàn
toàn ngoài domain. Để threshold 0.3 thì fallback không bao giờ kích hoạt.

Script này chạy mọi câu hỏi trong golden dataset, tách hai nhóm in-domain và
out-of-domain, rồi báo cáo khoảng cách giữa hai phân bố và ngưỡng tách tốt nhất.

Chạy:
    python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task5_semantic_search import semantic_search  # noqa: E402


ROOT = Path(__file__).parent.parent
DATASET = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
OUTPUT = ROOT / "group_project" / "evaluation" / "threshold_calibration.json"

TOP_K = 10


def best_dense_score(question: str) -> float:
    results = semantic_search(question, top_k=TOP_K)
    return float(results[0]["score"]) if results else 0.0


def main() -> int:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))

    in_domain: list[tuple[str, float]] = []
    out_domain: list[tuple[str, float]] = []

    print(f"{'id':<5} {'loại':<14} {'best cosine':>11}  câu hỏi")
    print("-" * 96)
    for case in dataset:
        score = best_dense_score(case["question"])
        bucket = out_domain if case.get("type") == "out_of_domain" else in_domain
        bucket.append((case["id"], score))
        print(f"{case['id']:<5} {case.get('type', ''):<14} {score:>11.4f}  {case['question'][:58]}")

    lows = [score for _, score in in_domain]
    highs = [score for _, score in out_domain]

    print("\n" + "=" * 96)
    print(f"In-domain  (n={len(lows)}):  min={min(lows):.4f}  mean={statistics.mean(lows):.4f}  max={max(lows):.4f}")
    print(f"Out-domain (n={len(highs)}): min={min(highs):.4f}  mean={statistics.mean(highs):.4f}  max={max(highs):.4f}")

    separable = min(lows) > max(highs)
    print(f"\nHai phân bố tách rời: {'CÓ' if separable else 'KHÔNG'}")

    if separable:
        # Đặt ngưỡng vào giữa khe hở để chịu được nhiễu của câu hỏi mới.
        recommended = round((min(lows) + max(highs)) / 2, 3)
        print(f"  khe hở: {max(highs):.4f} .. {min(lows):.4f}  (rộng {min(lows) - max(highs):.4f})")
    else:
        # Không tách được thì chọn ngưỡng cực đại hoá số quyết định đúng.
        candidates = sorted({round(score, 3) for _, score in in_domain + out_domain})
        best, recommended = -1, 0.0
        for candidate in candidates:
            correct = sum(1 for s in lows if s >= candidate) + sum(1 for s in highs if s < candidate)
            if correct > best:
                best, recommended = correct, candidate
        total = len(lows) + len(highs)
        print(f"  ngưỡng tốt nhất phân loại đúng {best}/{total} câu")
        print("  -> cosine tuyệt đối KHÔNG đủ để phát hiện ngoài domain với model này.")

    # Ngưỡng cực đại hoá độ chính xác chưa chắc là ngưỡng nên chạy thật. Bỏ sót
    # một câu ngoài domain thì còn safe refusal ở tầng generation đỡ; nhưng đẩy
    # nhầm một câu ĐÚNG domain sang fallback là mất luôn câu trả lời tốt. Vì vậy
    # điểm vận hành lấy dưới câu in-domain yếu nhất một khoảng an toàn.
    margin = 0.01
    operating = round(min(lows) - margin, 3)
    caught = sum(1 for s in highs if s < operating)
    print(f"\nNgưỡng cực đại độ chính xác : {recommended}")
    print(f"Điểm vận hành đề xuất       : {operating}"
          f"  (= in-domain thấp nhất {min(lows):.4f} trừ biên {margin})")
    print(f"  - không đẩy nhầm câu in-domain nào sang fallback")
    print(f"  - bắt được {caught}/{len(highs)} câu ngoài domain; phần còn lại do safe refusal xử lý")
    print(f"(mặc định của repo là 0.3 — với model này thì fallback sẽ không bao giờ chạy)")
    recommended = operating

    OUTPUT.write_text(
        json.dumps(
            {
                "embedding_model": "intfloat/multilingual-e5-small",
                "top_k": TOP_K,
                "in_domain": [{"id": i, "best_dense_score": s} for i, s in in_domain],
                "out_of_domain": [{"id": i, "best_dense_score": s} for i, s in out_domain],
                "in_domain_min": min(lows),
                "out_of_domain_max": max(highs),
                "separable": separable,
                "recommended_threshold": recommended,
                "repo_default": 0.3,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Đã ghi {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
