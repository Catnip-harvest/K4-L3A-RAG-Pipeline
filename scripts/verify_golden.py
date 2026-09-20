"""Kiểm tra golden dataset có thật sự bám corpus.

Một câu hỏi vàng chỉ có giá trị khi đáp án của nó tồn tại nguyên văn trong tài
liệu. Script này kiểm tra ba điều cho từng case:

1. Mọi ``gold_doc_ids`` tồn tại trong data/standardized/.
2. ``expected_context`` là chuỗi con nguyên văn của một trong các tài liệu gold.
3. Mọi ``context_keys`` cũng vậy — đây là các chuỗi dùng để chấm context recall,
   nên sai một ký tự là điểm sẽ sai suốt cả bài đánh giá.

Case ``out_of_domain`` được bỏ qua ba kiểm tra trên: theo định nghĩa chúng không
có tài liệu gold, và đó mới là điều cần kiểm.

Chạy:
    python scripts/verify_golden.py
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path


ROOT = Path(__file__).parent.parent
STANDARDIZED = ROOT / "data" / "standardized"
DATASET = ROOT / "group_project" / "evaluation" / "golden_dataset.json"


def normalise(text: str) -> str:
    """So khớp bỏ qua khác biệt xuống dòng và khoảng trắng thừa.

    PDF gốc ngắt dòng giữa câu, nên "là 02\nnăm" và "là 02 năm" là cùng một nội
    dung. Chuẩn hoá NFC vì tiếng Việt có hai cách mã hoá dấu khác nhau và MarkItDown
    không phải lúc nào cũng trả về cùng một dạng.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def main() -> int:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    corpus = {
        path.relative_to(STANDARDIZED).as_posix(): normalise(path.read_text(encoding="utf-8"))
        for path in STANDARDIZED.rglob("*.md")
    }
    everything = "\n".join(corpus.values())

    problems: list[str] = []
    in_domain = 0

    for case in dataset:
        case_id = case["id"]
        for field in ("question", "expected_answer", "expected_context"):
            if not str(case.get(field, "")).strip():
                problems.append(f"{case_id}: thiếu field '{field}'")

        if case.get("type") == "out_of_domain":
            if case.get("gold_doc_ids"):
                problems.append(f"{case_id}: case ngoài domain không được có gold_doc_ids")
            continue

        in_domain += 1
        gold_ids = case.get("gold_doc_ids") or []
        if not gold_ids:
            problems.append(f"{case_id}: thiếu gold_doc_ids")
            continue

        missing_docs = [doc for doc in gold_ids if doc not in corpus]
        if missing_docs:
            problems.append(f"{case_id}: không có tài liệu {missing_docs}")
            continue

        gold_text = "\n".join(corpus[doc] for doc in gold_ids)

        expected = normalise(case["expected_context"])
        if expected not in gold_text:
            where = "CÓ trong corpus nhưng ở tài liệu khác" if expected in everything else "KHÔNG có ở đâu cả"
            problems.append(f"{case_id}: expected_context {where}\n        {expected[:90]!r}")

        for key in case.get("context_keys") or []:
            if normalise(key) not in gold_text:
                where = "ở tài liệu khác" if normalise(key) in everything else "không tồn tại"
                problems.append(f"{case_id}: context_key {where}: {key!r}")

        for trap in case.get("trap_doc_ids") or []:
            if trap not in corpus:
                problems.append(f"{case_id}: trap_doc_ids không tồn tại: {trap}")

    print(f"Golden cases        : {len(dataset)}  ({in_domain} in-domain, {len(dataset) - in_domain} out-of-domain)")
    print(f"Tài liệu trong corpus: {len(corpus)}")
    if problems:
        print(f"\n{len(problems)} VẤN ĐỀ:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nTất cả expected_context và context_keys đều khớp nguyên văn với tài liệu gold.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
