# RAG Pipeline — Quy chế và dịch vụ sinh viên UTC

Nhóm **Logitech** · Lớp K4-L3A · Lab 08

Chatbot hỏi đáp trên bộ tài liệu chính sách và trang dịch vụ của Trường Đại học
Giao thông Vận tải. Hybrid retrieval (dense + BM25 + RRF), fallback vectorless,
câu trả lời có citation truy ngược được về chunk, và từ chối an toàn khi không
đủ bằng chứng.

Thành viên và phân công: [TEAMMATES.md](TEAMMATES.md).

## Corpus

| | Số lượng | Ký tự sau khi làm sạch |
|---|---:|---:|
| Tài liệu chính sách (PDF → Markdown) | 4 | 373.746 |
| Trang dịch vụ (crawl → Markdown) | 10 | 31.246 |
| **Tổng** | **14** | **404.992** |

Nguồn đầy đủ kèm URL, ngày thu thập và ghi chú bản quyền: [data/SOURCES.md](data/SOURCES.md).

Toàn bộ nguồn là tài liệu công khai trên `utc.edu.vn` và các tên miền con.
Dữ liệu cá nhân (số di động và email của cán bộ trong bảng phân công nhân sự) bị
loại khỏi Markdown ở bước chuẩn hoá; PDF gốc trong `data/landing/` giữ nguyên
như trường đã phát hành.

## Chạy thử

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env               # rồi điền GEMINI_API_KEY
```

```bash
# 1. Thu thập và chuẩn hoá (idempotent — chạy lại không tạo bản sao)
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# 2. Index
python -m src.task4_chunking_indexing

# 3. Kiểm tra
pytest -q

# 4. Chatbot
streamlit run app.py
```

`chroma_db/` không được commit vì là dữ liệu dẫn xuất; bước 2 dựng lại nó trong
khoảng hai phút trên CPU. `data/index/chunks.json` **có** trong repo vì Task 6
(BM25) phải đọc đúng tập chunk mà Task 5 đã index.

## Pipeline

```
Query ──┬─► Dense (multilingual-e5-small, cosine)  ─┐
        └─► BM25  (rank_bm25, cùng tập chunk)      ─┴─► RRF (k=60) ─┬─► Generation + citation
                                                                    │
        best dense cosine < threshold ──► Vectorless fallback ──────┘
                                          (cây heading, không vector)
```

Cổng ngưỡng đọc **cosine gốc của dense**, không đọc RRF score. Hai đại lượng
khác thang: cosine nằm trong [0, 1] và có nghĩa tuyệt đối, còn RRF score luôn
quanh 0,016–0,033 bất kể kết quả tốt hay tệ vì nó chỉ mã hoá thứ hạng. Đem RRF
score so với ngưỡng thì fallback sẽ kích hoạt ở mọi câu hỏi.

## Cấu hình thực dùng

| Tham số | Giá trị | Lý do |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 | Giữ mặc định của starter; xem đánh đổi ở Q15 trong `RESULT.md` |
| Embedding | `intfloat/multilingual-e5-small` (384-dim) | Huấn luyện cho retrieval, hiểu prefix `query:`/`passage:`, 470 MB — chạy được trên máy demo |
| LLM | `gemini-3.1-flash-lite` | Provider duy nhất nhóm có key thật |
| `top_k` | 5 | |
| `SCORE_THRESHOLD` | **0.853** | Hiệu chỉnh bằng dữ liệu, **không** dùng mặc định 0.3 — xem bên dưới |
| RRF `k` | 60 | Giá trị gốc trong bài báo Cormack 2009 |

### Vì sao threshold là 0.853 chứ không phải 0.3

`multilingual-e5-small` đặt toàn bộ embedding trong một hình nón hẹp, nên cosine
của **mọi** câu hỏi đều rơi vào dải 0,82–0,92 — kể cả câu hoàn toàn ngoài domain.
Với ngưỡng 0.3 thì fallback không bao giờ chạy.

Đo trên 18 câu golden (`python scripts/calibrate_threshold.py`):

| | n | min | trung bình | max |
|---|---:|---:|---:|---:|
| Trong domain | 16 | 0,8634 | 0,8898 | 0,9236 |
| Ngoài domain | 2 | 0,8240 | 0,8535 | 0,8830 |

Hai phân bố **chồng lấn**: câu ngoài domain Q18 ("trường có đào tạo tiến sĩ
ngành Y khoa không?") đạt 0,8830, cao hơn câu trong domain Q09 (0,8634). Đây là
kết quả đáng chú ý nhất của bài: **cosine tuyệt đối không phải bộ phát hiện
ngoài domain đáng tin với họ model này.**

Ngưỡng vận hành 0.853 đặt dưới câu in-domain yếu nhất một biên 0.01, nên không
đẩy nhầm câu đúng domain nào sang fallback. Nó bắt được câu ngoài domain rõ ràng
(Q17); câu ngoài domain "gần domain" như Q18 được tầng thứ hai xử lý — safe
refusal ở generation, và tầng đó bắt đúng 2/2.

## Kiểm thử

```bash
pytest tests/test_contracts.py -q     # 15 passed — schema, thứ tự, RRF, fallback
pytest tests/test_acceptance.py -q    # 5 passed — corpus, golden dataset, báo cáo
python scripts/verify_golden.py       # mọi expected_context khớp nguyên văn corpus
```

`scripts/verify_golden.py` là bước tự kiểm bắt buộc: acceptance test chỉ đếm số
case và kiểm tra field không rỗng, nó **không** xác minh đáp án vàng có thật
trong tài liệu. Script này làm việc đó.

## Đánh giá

Kết quả đầy đủ: [group_project/evaluation/RESULT.md](group_project/evaluation/RESULT.md).

```bash
python scripts/calibrate_threshold.py          # hiệu chỉnh ngưỡng
python scripts/run_evaluation.py --configs AB  # A/B bắt buộc
python scripts/run_evaluation.py --configs ABC # thêm thí nghiệm cân bằng doc_type
python scripts/run_evaluation.py --no-llm      # chỉ metric retrieval, không tốn quota
```

## Cấu trúc repo

```
src/task1..3      thu thập và chuẩn hoá corpus
src/task4         chunk, embed, index vào ChromaDB
src/task5..6      dense search và BM25 trên cùng tập chunk
src/task7         RRF
src/task8         fallback vectorless (PageIndex API, hoặc cây heading local)
src/task9         pipeline + cổng ngưỡng
src/task10        generation, citation, safe refusal
app.py + ui/      chatbot Streamlit và các biểu đồ giải thích quyết định
scripts/          hiệu chỉnh ngưỡng, kiểm tra golden dataset, chạy evaluation
data/landing/     file gốc chưa xử lý
data/standardized/ Markdown đã làm sạch — đầu vào của index
data/index/       tập chunk dùng chung cho dense và BM25
```
