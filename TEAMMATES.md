# Thành viên — Lab 08: RAG Pipeline

Lớp K4-L3A · Repository: `Catnip-harvest/K4A-Day08-02563-HoangQuocViet` · Nhánh mặc định: `main`

## Thành viên

| Họ và tên | Mã học viên | Vai trò | Phần việc |
|---|---|---|---|
| Hoàng Quốc Việt | 02563 | Data · Retrieval · Generation/UI · Evaluation | Toàn bộ Task 1–10, `app.py`, `ui/`, golden dataset, hiệu chỉnh threshold, A/B evaluation, báo cáo |

**Bài nộp cá nhân.** Buổi này tôi làm một mình nên không có phân công nhiều
người; bốn vai trò mà hướng dẫn gợi ý (Data, Retrieval, Generation/UI,
Evaluation/Integration) đều do một người thực hiện. Mọi commit trên `main` đều
của tôi và tôi giải thích hoặc chạy lại được bất kỳ phần nào trong buổi demo.

## Quy ước làm việc

- Không commit `.env`. Chỉ `.env.example` được đưa lên repo.
- Không commit `chroma_db/` — dữ liệu dẫn xuất, dựng lại bằng
  `python -m src.task4_chunking_indexing`.
- `data/index/chunks.json` **được** commit: BM25 (Task 6) đọc từ file này để đảm
  bảo dùng đúng tập chunk mà dense đã index. Hai bên chunk riêng thì RRF sẽ gộp
  hai bảng xếp hạng trỏ tới hai tập ID khác nhau.
- Đổi corpus thì phải chạy lại index **và** evaluation. Con số trong
  `RESULT.md` chỉ có nghĩa khi đi kèm đúng phiên bản corpus.

## Thứ tự thực hiện

| # | Việc | File | Bằng chứng |
|---|---|---|---|
| 1 | Chốt chủ đề, thu thập 4 PDF chính sách | `src/task1_collect_legal_docs.py` | `data/landing/legal/MANIFEST.json` |
| 2 | Crawl 10 trang dịch vụ, làm sạch, loại dữ liệu cá nhân | `src/task2_crawl_news.py` | `data/landing/news/*.json` |
| 3 | Chuẩn hoá sang Markdown | `src/task3_convert_markdown.py` | `data/standardized/`, `data/SOURCES.md` |
| 4 | Chunk, embed, index | `src/task4_chunking_indexing.py` | 1009 chunks |
| 5 | Dense + BM25 trên cùng tập chunk | `src/task5`, `src/task6` | `pytest tests/test_contracts.py` |
| 6 | RRF | `src/task7_reranking.py` | test kiểm tra đúng công thức |
| 7 | Fallback vectorless | `src/task8_pageindex_vectorless.py` | cây heading local |
| 8 | Pipeline + cổng ngưỡng | `src/task9_retrieval_pipeline.py` | test dùng dense score, không dùng RRF score |
| 9 | Generation + citation + safe refusal | `src/task10_generation.py` | từ chối đúng 2/2 câu ngoài domain |
| 10 | Chatbot Streamlit + biểu đồ giải thích | `app.py`, `ui/` | |
| 11 | Golden dataset 18 câu, tự kiểm | `group_project/evaluation/golden_dataset.json` | `scripts/verify_golden.py` |
| 12 | Hiệu chỉnh threshold bằng dữ liệu | `scripts/calibrate_threshold.py` | `threshold_calibration.json` |
| 13 | A/B evaluation + phân tích lỗi | `scripts/run_evaluation.py` | `RESULT.md`, `results.json` |

Báo cáo cá nhân: `reports/02563-HoangQuocViet.md`.
