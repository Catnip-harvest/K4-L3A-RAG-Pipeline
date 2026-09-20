# Individual contribution report

## Thông tin

- Họ và tên: Hoàng Quốc Việt
- Mã học viên: 02563
- Nhóm: làm cá nhân (buổi này tôi làm một mình, xem `TEAMMATES.md`)
- Repository/branch: `Catnip-harvest/K4A-Day08-02563-HoangQuocViet`, nhánh `feat/rag-pipeline-utc` → `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 1–3: corpus | 4 PDF chính sách UTC + 10 trang dịch vụ; lọc boilerplate theo chuỗi ≥3 dòng; loại dữ liệu cá nhân | `6469e2a`, `data/SOURCES.md` | Done |
| Task 4: chunk + index | 1009 chunk, e5-small 384 chiều, prefix `query:`/`passage:` do người gọi gắn | `03701a1` | Done |
| Task 5–6: dense + BM25 | Cùng tập chunk qua `data/index/chunks.json` | `03701a1` | Done |
| Task 7: RRF | Copy item trước khi đổi score, không mutate danh sách vào | `03701a1` | Done |
| Task 8: fallback | PageIndex API khi có key; cây heading local khi không | `03701a1` | Done |
| Task 9: pipeline | Cổng ngưỡng đọc cosine gốc; fuse đúng một lần | `03701a1` | Done |
| Task 10: generation | Citation, safe refusal, tách `refused` khỏi `errored` | `0a52bdf` | Done |
| Chatbot + đồ hoạ | Streamlit, sơ đồ pipeline SVG theo số liệu thật, biểu đồ dịch chuyển thứ hạng | `656a9fa` | Done |
| Golden dataset + evaluation | 18 case, script tự kiểm, hiệu chỉnh ngưỡng, A/B + thí nghiệm C | `fbd11b5`, `RESULT.md` | Done |
| README, TEAMMATES | | `039e416` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Không dùng `SCORE_THRESHOLD = 0.3` như starter, mà đo rồi
   chọn **0.853**.
   **Lý do/evidence:** `multilingual-e5-small` đặt mọi embedding trong một hình
   nón hẹp. Đo trên 18 câu golden: in-domain 0.8634–0.9236, out-of-domain
   0.8240–0.8830. Với ngưỡng 0.3 thì fallback **không bao giờ** chạy — nhánh
   code đó tồn tại nhưng chết. Số liệu ở `threshold_calibration.json`.
   **Trade-off:** hai phân bố chồng lấn nên không có ngưỡng nào đúng hết. Tôi
   chọn điểm dưới câu in-domain yếu nhất một biên 0.01: chấp nhận bỏ sót câu
   "gần domain" (Q18 đạt 0.8830, cao hơn cả câu in-domain Q09) để đổi lấy việc
   không đẩy nhầm câu đúng domain nào sang fallback. Phần bỏ sót do safe refusal
   ở generation gánh, và thực đo nó bắt đúng 2/2.

2. **Quyết định:** Trong `generate_with_trace`, tách `refused` (model đọc context
   rồi kết luận thiếu căn cứ) khỏi `errored` (chưa gọi được model).
   **Lý do/evidence:** lần chạy A/B đầu tiên báo hybrid+RRF tệ hơn hẳn dense-only
   (faithfulness −0.185). Sai. `gemini-2.5-flash` giới hạn 20 request/ngày, một
   lượt A/B cần 36; Config A dùng hết quota nên Config B nhận HTTP 429, và code
   lúc đó ghi lỗi provider thành "model từ chối". Sau khi tách hai trạng thái và
   thêm retry, chênh lệch thật là **−0.0044**, tức là không đáng kể.
   **Trade-off:** harness chậm hơn vì phải chờ và thử lại, và một số case có thể
   bị loại khỏi phép chấm thay vì có điểm. Tôi chấp nhận mất dữ liệu điểm còn
   hơn có số liệu sai — báo cáo ghi rõ số case bị loại (lần chạy cuối: 0).

## Kiểm thử và kết quả

- Test: `pytest -q` → **20 passed** (15 contract + 5 acceptance).
- Tự kiểm: `scripts/verify_golden.py` xác nhận 18/18 case có `expected_context`
  và `context_keys` khớp **nguyên văn** với tài liệu gold. Acceptance test chỉ
  đếm field không rỗng nên không thay được bước này.
- A/B (16 case in-domain, `top_k=5`): Config A trung bình **0.6710**, Config B
  **0.6666**, chênh **−0.0044**. Kết luận: **RRF không cải thiện trên corpus này.**
- Thí nghiệm khuyến nghị: Config C (cân bằng `doc_type`) trung bình **0.7053**,
  **+0.0343** so với A, `context_precision` **+0.0810**, và nhanh hơn 1195 ms/câu.
- Ngoài domain: 2/2 câu bị từ chối đúng ở cả ba config.

**Lỗi đã phát hiện và cách xử lý:**

- *BM25 trả rỗng trên corpus nhỏ.* `BM25Okapi` tính `idf = log(N−n+0.5) − log(n+0.5)`;
  với 2 tài liệu và token xuất hiện ở 1 tài liệu thì idf = 0, nên mọi score
  bằng 0 kể cả khi khớp hoàn hảo. Đổi điều kiện loại từ `score <= 0` sang
  "không khớp token nào".
- *`context_precision` vượt 1.0.* Tôi chia cho số tài liệu gold, trong khi kết
  quả là chunk và nhiều chunk cùng thuộc một tài liệu. Sửa thành chia cho số
  chunk đúng lấy được.
- *`\b` bị heredoc nuốt thành ký tự backspace `\x08`* khi vá regex lọc số điện
  thoại, làm pattern không bao giờ khớp. Phát hiện bằng `repr()` của dòng đó.
- *Application Control chặn `pandas 3.0.6`* (`DLL load failed while importing
  reshape`). Ghim `3.0.5` thay vì tắt Application Control.
- *Câu hỏi thử không dấu* ("Ky tuc xa...") cho kết quả rác và suýt làm tôi kết
  luận retrieval hỏng. Với đúng dấu, dense trả đúng chunk ở hạng 1 với 0.8997.

## Điều còn hạn chế

- **Corpus lệch nặng và tôi chưa sửa tận gốc.** 90.6% chunk là legal, riêng Sổ
  tay K60 chiếm 34.6%. Trang `phong-bao-ve.md` chỉ có 7/1009 chunk nên Q09 hỏng
  ở cả ba config. Config C giảm nhẹ triệu chứng chứ không chữa nguyên nhân.
- **Q12 vẫn sai với faithfulness 1.00** — hai tài liệu chính thức mâu thuẫn,
  cùng online. Cần metadata `document_version`/`supersedes`, chưa làm.
- **Bốn metric là bản tự viết, không phải RAGAS.** Đổi lại được tính tất định
  và không tốn quota, nhưng `faithfulness` và `answer_relevance` là xấp xỉ theo
  từ vựng chứ không phải phán đoán ngữ nghĩa.
- **Conversation memory chưa được đo**, nên tôi không tự tính nó vào bonus.

**Nếu có thêm thời gian, việc đầu tiên tôi làm:** chia 4 tài liệu legal theo
Điều/Chương trước khi chunk. Nó nhắm đúng trần `context_recall` 0.72–0.75 mà cả
ba config đều chạm, và nó làm citation chỉ được tới "Điều 16" thay vì "chunk-274".

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích
hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-20
- Tên thành viên: Hoàng Quốc Việt
