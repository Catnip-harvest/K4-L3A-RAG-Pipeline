"""A/B evaluation: dense-only (Config A) so với hybrid + RRF (Config B).

Hai config chỉ khác nhau đúng một biến — ``use_reranking``. Golden dataset,
generator, prompt, top_k, threshold và corpus đều giữ nguyên. Nếu đổi nhiều hơn
một biến thì chênh lệch metric không còn nói lên điều gì về RRF.

Bốn metric, tất cả đều **tất định và không phụ thuộc thang đo**:

``context_recall``
    Tỉ lệ ``context_keys`` (chuỗi đáp án nguyên văn) xuất hiện trong ngữ cảnh
    top-k. Đây là metric nghiêm nhất và là metric khó gian lận nhất: hoặc bằng
    chứng có trong context, hoặc không.

``context_precision``
    Average precision@k theo ``gold_doc_ids``. Phạt việc lấy đúng tài liệu
    nhưng xếp nó dưới ba chunk nhiễu — tức là đo cả thứ hạng chứ không chỉ
    sự có mặt.

``faithfulness``
    Tỉ lệ từ mang nghĩa trong câu trả lời có mặt trong context, tính trung bình
    theo câu. Một câu bịa sẽ mang theo những từ không hề có trong context.
    Cố ý KHÔNG dùng cosine: ``multilingual-e5-small`` nén mọi similarity vào
    dải 0.78–0.92, nên ngưỡng cosine ở đây là vô nghĩa.

``answer_relevance``
    Token-level F1 giữa câu trả lời và ``expected_answer``, kiểu SQuAD.

Hai case ``out_of_domain`` KHÔNG tính vào bốn metric trên — chúng không có đáp
án đúng để so. Chúng được báo cáo riêng bằng tỉ lệ từ chối, vì đó mới là hành
vi đúng cần đo.

Chạy:
    python scripts/run_evaluation.py                 # đầy đủ, gọi LLM
    python scripts/run_evaluation.py --no-llm        # chỉ metric retrieval, không tốn quota
    python scripts/run_evaluation.py --limit 3       # thử nhanh
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task9_retrieval_pipeline import retrieve_with_trace  # noqa: E402
from src.task10_generation import REFUSAL, generate_with_trace  # noqa: E402


ROOT = Path(__file__).parent.parent
DATASET = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
RESULTS = ROOT / "group_project" / "evaluation" / "results.json"

TOP_K = 5

CONFIGS = {
    "A": {"label": "dense-only", "use_reranking": False, "balanced": False},
    "B": {"label": "hybrid + RRF", "use_reranking": True, "balanced": False},
    "C": {"label": "hybrid + RRF, can bang doc_type", "use_reranking": True, "balanced": True},
}

# A/B bat buoc cua rubric la A vs B. C la thi nghiem cua phan khuyen nghi:
# no doi THEM mot bien nen khong duoc tron vao ket luan A/B.
REQUIRED_AB = ("A", "B")

_TOKEN = re.compile(r"\w+", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+|\n+")

# Từ chức năng tiếng Việt: có mặt ở mọi câu nên không phải bằng chứng về grounding.
_STOPWORDS = {
    "là", "của", "và", "có", "cho", "các", "được", "trong", "với", "những", "một",
    "này", "thì", "ở", "về", "khi", "nào", "bao", "nhiêu", "gì", "ai", "tại",
    "đến", "từ", "theo", "hay", "hoặc", "sẽ", "đã", "phải", "không", "cũng", "để",
    "nếu", "mà", "còn", "như", "vào", "ra", "do", "bởi", "trên", "dưới", "sau",
    "trước", "nhưng", "vì", "nên", "đó", "kia", "ấy", "rằng", "thuộc", "gồm",
}


def normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def content_words(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(normalise(text).lower()) if t not in _STOPWORDS and len(t) > 1}


def context_recall(case: dict, chunks: list[dict]) -> float | None:
    """Tỉ lệ chuỗi đáp án nguyên văn có mặt trong ngữ cảnh top-k."""
    keys = case.get("context_keys") or []
    if not keys:
        return None
    haystack = normalise("\n".join(chunk["content"] for chunk in chunks))
    return sum(1 for key in keys if normalise(key) in haystack) / len(keys)


def context_precision(case: dict, chunks: list[dict]) -> float | None:
    """Average precision@k — thưởng cho việc xếp chunk đúng lên trên."""
    gold = set(case.get("gold_doc_ids") or [])
    if not gold or not chunks:
        return None
    hits, weighted = 0, 0.0
    for rank, chunk in enumerate(chunks, start=1):
        doc = chunk["id"].split("::")[0]
        if doc in gold:
            hits += 1
            weighted += hits / rank
    # Chia cho SỐ CHUNK ĐÚNG lấy được, không phải số tài liệu gold. Gold là tài
    # liệu còn kết quả là chunk, nên nhiều chunk có thể cùng thuộc một tài liệu
    # gold; chia cho len(gold) sẽ cho ra giá trị lớn hơn 1. Công thức này là
    # trung bình của precision@rank tại các vị trí đúng, luôn nằm trong [0, 1].
    return weighted / hits if hits else 0.0


def faithfulness(answer: str, chunks: list[dict]) -> float | None:
    """Trung bình theo câu của tỉ lệ từ mang nghĩa được context hậu thuẫn."""
    if not answer.strip() or not chunks:
        return None
    supported = content_words("\n".join(chunk["content"] for chunk in chunks))
    scores = []
    for sentence in _SENTENCE.split(answer):
        words = content_words(sentence)
        if len(words) < 3:
            continue
        scores.append(len(words & supported) / len(words))
    return statistics.mean(scores) if scores else None


def answer_relevance(answer: str, expected: str) -> float | None:
    """Token-level F1 giữa câu trả lời và đáp án vàng."""
    if not answer.strip():
        return 0.0
    produced, gold = content_words(answer), content_words(expected)
    if not produced or not gold:
        return None
    overlap = len(produced & gold)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(produced), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 25


def run_case(case: dict, config_key: str, use_llm: bool) -> dict:
    """Chạy một câu hỏi qua một config và chấm điểm.

    Lỗi provider (thường gặp nhất là 429 rate limit) được thử lại có chờ. Nếu
    vẫn hỏng thì case bị đánh dấu ``errored`` và KHÔNG được chấm. Chấm nó bằng 0
    sẽ biến một đợt throttling thành kết luận sai về chất lượng retrieval: đúng
    lỗi đã xảy ra trong lần chạy đầu, khi quota của gemini-2.5-flash cạn giữa
    chừng và toàn bộ Config B hiện ra thành "từ chối nhiều hơn".
    """
    use_reranking = CONFIGS[config_key]["use_reranking"]
    balanced = CONFIGS[config_key]["balanced"]
    started = time.perf_counter()

    if use_llm:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result, trace = generate_with_trace(
                case["question"], top_k=TOP_K, use_reranking=use_reranking, balanced=balanced
            )
            if not trace.get("errored"):
                break
            if attempt < MAX_ATTEMPTS:
                wait = BACKOFF_SECONDS * attempt
                print(f"      {case['id']} cfg {config_key}: provider lỗi, chờ {wait}s rồi thử lại")
                time.sleep(wait)
        chunks, answer = result["sources"], result["answer"]
        retrieval_source = result["retrieval_source"]
    else:
        chunks, trace = retrieve_with_trace(
            case["question"], top_k=TOP_K, use_reranking=use_reranking, balanced=balanced
        )
        answer, retrieval_source = "", trace.get("decision", "hybrid")

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    refused = bool(answer) and REFUSAL[:30] in answer

    row = {
        "id": case["id"],
        "type": case.get("type", ""),
        "question": case["question"],
        "config": config_key,
        "answer": answer,
        "refused": refused,
        "retrieval_source": retrieval_source,
        "best_dense_score": round(trace.get("best_dense_score", 0.0), 4),
        "fallback_triggered": any(
            stage.get("name") == "fallback" and stage.get("triggered")
            for stage in trace.get("stages", [])
        ),
        "retrieved_docs": [chunk["id"].split("::")[0] for chunk in chunks],
        "elapsed_ms": elapsed_ms,
        "llm_ms": trace.get("llm_elapsed_ms", 0.0),
        "citations": [c["marker"] for c in trace.get("citations", [])],
        "citations_all_matched": all(c["matched"] for c in trace.get("citations", []) or [{"matched": True}]),
    }

    row["errored"] = bool(trace.get("errored"))
    row["llm_error"] = trace.get("llm_error", "")
    if row["errored"]:
        # Không đo được thì không chấm. Số case bị loại được báo cáo riêng.
        row["metrics"] = {}
        return row

    if case.get("type") == "out_of_domain":
        # Với câu ngoài domain, "đúng" nghĩa là từ chối. Không chấm bốn metric.
        row["metrics"] = {}
        row["correct_refusal"] = refused
        return row

    row["metrics"] = {
        "context_recall": context_recall(case, chunks),
        "context_precision": context_precision(case, chunks),
        "faithfulness": faithfulness(answer, chunks) if use_llm else None,
        "answer_relevance": answer_relevance(answer, case["expected_answer"]) if use_llm else None,
    }
    return row


def average(rows: list[dict], metric: str) -> float | None:
    values = [r["metrics"].get(metric) for r in rows if r["metrics"].get(metric) is not None]
    return round(statistics.mean(values), 4) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-llm", action="store_true", help="Bỏ generation, chỉ chấm retrieval")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ chạy N case đầu")
    parser.add_argument(
        "--configs", default="AB",
        help="Cac config can chay, vi du AB (bat buoc) hoac ABC (them thi nghiem can bang)",
    )
    args = parser.parse_args()

    use_llm = not args.no_llm
    configs = [c for c in args.configs.upper() if c in CONFIGS]
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]

    all_rows: list[dict] = []
    for config_key in configs:
        label = CONFIGS[config_key]["label"]
        print(f"\n{'=' * 90}\nConfig {config_key} — {label}\n{'=' * 90}")
        for case in dataset:
            row = run_case(case, config_key, use_llm)
            all_rows.append(row)
            metrics = row["metrics"]
            summary = "  ".join(
                f"{name[:4]}={value:.2f}" if value is not None else f"{name[:4]}=  - "
                for name, value in metrics.items()
            ) or ("TỪ CHỐI (đúng)" if row.get("correct_refusal") else "KHÔNG từ chối (sai)")
            print(f"  {row['id']:<5} {summary}   dense={row['best_dense_score']:.3f} {row['elapsed_ms']:>7.0f}ms")

    graded = {k: [r for r in all_rows if r["config"] == k and r["type"] != "out_of_domain"] for k in configs}
    ood = {k: [r for r in all_rows if r["config"] == k and r["type"] == "out_of_domain"] for k in configs}

    metric_names = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    baseline = configs[0]

    def fmt(value: float | None) -> str:
        return f"{value:>12.4f}" if value is not None else f"{'—':>12}"

    first = configs[0]
    print(f"\n{'=' * 90}")
    print(f"TỔNG KẾT ({len(graded[first])} case in-domain, {len(ood[first])} case ngoài domain)")
    print("=" * 90)

    # Số case bị loại vì lỗi provider. Nếu cột này khác 0 thì mọi so sánh bên
    # dưới đều đang dựa trên ít dữ liệu hơn, và phải nói rõ trong báo cáo.
    for key in configs:
        errored = [r for r in all_rows if r["config"] == key and r.get("errored")]
        if errored:
            print(f"  [!] Config {key}: {len(errored)} case bị loại do lỗi provider "
                  f"({', '.join(r['id'] for r in errored)})")

    header = f"{'Metric':<20}" + "".join(f"{'Config ' + k:>12}" for k in configs)
    if len(configs) > 1:
        header += "".join(f"{f'Δ {k}-{baseline}':>12}" for k in configs[1:])
    print(header)

    summary: dict = {}
    for name in metric_names:
        values = {k: average(graded[k], name) for k in configs}
        row_summary = dict(values)
        for k in configs[1:]:
            base, other = values[baseline], values[k]
            row_summary[f"delta_{k}"] = (
                round(other - base, 4) if (base is not None and other is not None) else None
            )
        summary[name] = row_summary
        line = f"{name:<20}" + "".join(fmt(values[k]) for k in configs)
        line += "".join(fmt(row_summary.get(f"delta_{k}")) for k in configs[1:])
        print(line)

    averages: dict = {}
    for key in configs:
        values = [summary[n][key] for n in metric_names if summary[n][key] is not None]
        averages[key] = round(statistics.mean(values), 4) if values else None
    for key in configs[1:]:
        base, other = averages[baseline], averages[key]
        averages[f"delta_{key}"] = (
            round(other - base, 4) if (base is not None and other is not None) else None
        )
    summary["average"] = averages
    line = f"{'AVERAGE':<20}" + "".join(fmt(averages[k]) for k in configs)
    line += "".join(fmt(averages.get(f"delta_{k}")) for k in configs[1:])
    print(line)

    for key in configs:
        rows = graded[key]
        latency = round(statistics.mean(r["elapsed_ms"] for r in rows), 1) if rows else 0.0
        refusals = sum(1 for r in ood[key] if r.get("correct_refusal"))
        print(f"\nConfig {key}: latency TB {latency} ms | từ chối đúng {refusals}/{len(ood[key])} câu ngoài domain")

    RESULTS.write_text(
        json.dumps(
            {
                "run_date": date.today().isoformat(),
                "top_k": TOP_K,
                "llm_used": use_llm,
                "configs": CONFIGS,
                "summary": summary,
                "rows": all_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nĐã ghi {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
