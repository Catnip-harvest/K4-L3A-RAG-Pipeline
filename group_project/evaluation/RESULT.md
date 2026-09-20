# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | Harness tự viết, `scripts/run_evaluation.py`. Bốn metric tất định, không dùng LLM-judge — xem "Định nghĩa metric" bên dưới |
| Evaluator model                    | Không có. Chấm bằng khớp chuỗi và token F1, nên chạy lại cho đúng cùng một con số |
| Generator model                    | `gemini-3.1-flash-lite` (temperature 0.3, top_p 0.9) |
| Embedding model                    | `intfloat/multilingual-e5-small`, 384 chiều, cosine, prefix `query:` / `passage:` |
| Corpus version/commit              | `039e416` — 14 tài liệu, 404.992 ký tự, 1009 chunk |
| Golden dataset size                | 18 case: 16 trong domain được chấm + 2 ngoài domain đo hành vi từ chối |
| `top_k`                            | 5 |
| Fallback threshold and calibration | **0.853**, hiệu chỉnh bằng `scripts/calibrate_threshold.py` trên 16 câu trong domain và 2 câu ngoài domain. Mặc định 0.3 của repo không dùng được — xem "Hiệu chỉnh threshold" |
| Case bị loại do lỗi provider       | 0 |

### Định nghĩa metric

Bốn metric đều **tất định và không phụ thuộc thang đo**. Chọn cách này thay vì
LLM-judge vì chạy lại phải ra đúng con số cũ, và vì quota free tier không đủ để
chấm 54 lượt sinh + 54 lượt chấm trong một buổi.

| Metric | Cách tính |
|---|---|
| `context_recall` | Tỉ lệ `context_keys` (chuỗi đáp án nguyên văn) xuất hiện trong ngữ cảnh top-5. Nghiêm nhất trong bốn metric: hoặc bằng chứng có, hoặc không |
| `context_precision` | Average precision@5 theo `gold_doc_ids`, chia cho số chunk đúng lấy được. Phạt việc lấy đúng tài liệu nhưng xếp nó dưới các chunk nhiễu |
| `faithfulness` | Trung bình theo câu của tỉ lệ từ mang nghĩa trong câu trả lời có mặt trong context. Một câu bịa mang theo từ không hề có trong context |
| `answer_relevance` | Token-level F1 giữa câu trả lời và `expected_answer`, kiểu SQuAD |

Hai case ngoài domain không được chấm bốn metric — chúng không có đáp án đúng để
so. Chúng được báo cáo riêng bằng tỉ lệ từ chối, vì đó mới là hành vi cần đo.

`scripts/verify_golden.py` xác nhận mọi `expected_context` và `context_keys` đều
khớp **nguyên văn** với tài liệu gold trước khi chạy đánh giá.

## Configurations

- **Config A — dense-only:** chỉ `semantic_search`, lấy top-5 trực tiếp. `use_reranking=False`.
- **Config B — hybrid + RRF:** `semantic_search` và `lexical_search` mỗi bên lấy 10 ứng viên, fuse một lần bằng RRF (k=60), lấy top-5. `use_reranking=True`.
- **Config C — hybrid + RRF, cân bằng `doc_type`:** như B nhưng dense chạy **riêng trong từng `doc_type`** rồi đưa cả hai bảng xếp hạng vào RRF cùng bảng BM25. Đây là **thí nghiệm của phần khuyến nghị**, không thuộc A/B bắt buộc, vì nó đổi thêm một biến.

A và B chỉ khác nhau đúng một biến `use_reranking`. Golden dataset, generator,
prompt, `top_k`, threshold và corpus giữ nguyên.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |   0.8899 |   0.8937 |   +0.0038 |
| Answer relevance  |   0.4620 |   0.4679 |   +0.0059 |
| Context recall    |   0.7500 |   0.7188 |   −0.0312 |
| Context precision |   0.5822 |   0.5860 |   +0.0038 |
| **Average**       | **0.6710** | **0.6666** | **−0.0044** |

Câu ngoài domain: cả hai config từ chối đúng **2/2**.

Thí nghiệm bổ sung (không thuộc A/B bắt buộc):

| Metric | Config A | Config C | Delta C−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.8899 | 0.9354 | **+0.0455** |
| Answer relevance | 0.4620 | 0.5036 | **+0.0416** |
| Context recall | 0.7500 | 0.7188 | −0.0312 |
| Context precision | 0.5822 | 0.6632 | **+0.0810** |
| **Average** | 0.6710 | **0.7053** | **+0.0343** |

## A/B comparison

- **Cấu hình tốt hơn: không có.** Chênh lệch B−A là **−0.0044** trên 16 case —
  nhỏ hơn nhiều so với mức nhiễu của một bộ 16 câu. Kết luận trung thực là
  **hybrid + RRF không cải thiện gì trên corpus này**, chứ không phải "B tốt hơn
  một chút".

- **Evidence:** ba metric nhích lên trong khoảng +0.004 đến +0.006, nhưng
  `context_recall` **giảm** 0.0312. Tức là RRF kéo thêm được vài chunk đúng tài
  liệu, đồng thời đẩy mất một số chunk chứa chuỗi đáp án. Nhìn vào thành phần
  kết quả thì rõ vì sao: **tỉ lệ chunk `news` trong top-5 của A và B giống hệt
  nhau, 23/80 (28.8%)**. BM25 không đưa được tài liệu mới nào vào; nó chỉ hoán
  vị thứ tự trong đúng nhóm chunk mà dense đã chọn.

- **Vì sao:** corpus lệch nặng. 914/1009 chunk (**90.6%**) là legal, riêng
  *Sổ tay sinh viên K60* chiếm **34.6%** toàn bộ index. Trang trả lời được câu
  hỏi dịch vụ thường chỉ có 7–15 chunk. Cả dense lẫn BM25 đều rút từ cùng một
  cái hồ lệch đó, nên fuse hai bảng xếp hạng cùng bị lệch không sửa được lệch.
  RRF chỉ hợp nhất thứ hạng; nó không tạo ra ứng viên mới.

- **Trade-off latency/cost:** A 5473 ms/câu, B 5396 ms/câu — như nhau, vì phần
  lớn thời gian là gọi LLM (A 4261 ms, B 5355 ms) chứ không phải retrieval.
  BM25 trên 1009 chunk mất dưới 5 ms. Về cost thì hai config bằng nhau: cùng
  một lượt sinh cho mỗi câu. Nói cách khác **B không đắt hơn, nó chỉ không giúp gì**.

- **Config C thì khác:** +0.0343 trung bình, riêng `context_precision` +0.0810,
  và **nhanh hơn** (4278 ms so với 5473 ms). Tỉ lệ chunk `news` trong top-5 tăng
  từ 28.8% lên **38.8%**. Đây mới là thay đổi có tác dụng.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
| 1 | Q11 — "Đơn vị nào chủ trì việc đánh giá kết quả học tập của sinh viên?" | B | 0.12 | 0.00 | 0.00 | 0.00 | retrieval | Top-5 **toàn bộ** là legal (`so-tay-sinh-vien-k60`, `qc-dao-tao-thac-si-2021`). Trang gold `phong-dao-tao-dai-hoc.md` có 11 chunk, phải cạnh tranh với 914 chunk quy chế cũng nói về "đánh giá kết quả học tập". Đây là lỗi corpus imbalance, không phải lỗi câu hỏi |
| 2 | Q09 — "Sinh viên bị mất tài sản trong khuôn viên trường thì báo cho đơn vị nào?" | B | 0.38 | 0.00 | 0.00 | 0.00 | data + retrieval | `phong-bao-ve.md` chỉ có **7 chunk trên 1009** (0.7%). Corpus cũng không chứa cụm "mất tài sản" ở đâu cả, nên BM25 không có gì để bám và dense phải nhảy hai bước ngữ nghĩa ("mất đồ" → "an ninh trật tự") |
| 3 | Q12 — "Ai quản lý ký túc xá của trường?" | B | 1.00 | 0.11 | 0.00 | 0.00 | data | **Hai câu trả lời chính thức mâu thuẫn, cùng crawl một ngày.** Trang Ban Quản lý KTX vẫn tự giới thiệu là đang hoạt động; trang Phòng Chăm sóc người học ghi rằng QĐ 2109/QĐ-ĐHGTVT đã sáp nhập đơn vị đó. Retrieval xếp trang cũ lên đầu ở **cả ba config**. Faithfulness 1.00 là điểm đáng sợ nhất trong bảng: câu trả lời trung thành tuyệt đối với context, và context thì đã lỗi thời |
| 4 | Q14 — "Đơn vị nào chịu trách nhiệm thu học phí?" | B | 0.88 | 0.29 | 0.00 | 0.00 | retrieval | Mức học phí nằm ở tài liệu legal, đơn vị thu nằm ở trang news. Top-5 toàn `so-tay-sinh-vien-k60` — câu hỏi bắc cầu hai loại nguồn thì nhóm đông chunk luôn thắng |
| 5 | Q13 — "Học bổng và vay vốn tín dụng liên hệ đơn vị nào?" | B | 0.98 | 0.23 | 0.50 | 0.00 | retrieval | Đáp án nằm ở hai tài liệu; chỉ lấy được một nửa (recall 0.50) |

**Config C sửa được bao nhiêu:** Q11 từ trung bình **0.031 lên 0.769** (recall
0.00 → 1.00, và cả `phong-dao-tao-dai-hoc.md` lẫn trang bẫy
`phong-quan-ly-chat-luong.md` đều vào top-5, đúng như thiết kế câu hỏi mong đợi).
Q13 từ 0.426 lên 0.530. **Q09, Q12 và Q14 không cải thiện** — xem phần khuyến nghị.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
| 1 | **Bật cân bằng `doc_type` làm mặc định** (Config C): chạy dense riêng trong từng `doc_type` rồi đưa các bảng xếp hạng vào RRF | Q11 từ 0.031 → 0.769. Chunk `news` trong top-5 tăng 28.8% → 38.8%. Trung bình +0.0343, precision +0.0810, latency giảm 1195 ms | Đưa average từ 0.671 lên ~0.705 mà không tốn thêm chi phí. Cách sửa này hoạt động được chính vì RRF dùng **thứ hạng**: một chunk hạng 1 trong nhóm 95 chunk news đấu sòng phẳng với chunk hạng 1 trong nhóm 914 chunk legal | `python scripts/run_evaluation.py --configs AC` — so `context_precision` và tỉ lệ `news` trong `retrieved_docs` |
| 2 | **Thêm metadata `document_version` / `supersedes` / `status`, và lọc bỏ tài liệu đã bị thay thế trước khi chấm điểm** | Q12 sai ở **cả ba config** với faithfulness 1.00. Không có cấu hình chunking hay retrieval nào sửa được: cả hai tài liệu đều đang online và đều là nguồn chính thức | Sửa dứt điểm lớp lỗi nguy hiểm nhất — câu trả lời tự tin, có trích dẫn, và sai. Đây là lỗi duy nhất trong bảng mà người dùng không có cách nào tự phát hiện | Đánh dấu `ban-quan-ly-ky-tuc-xa.md` là `superseded_by: phong-cham-soc-nguoi-hoc`, chạy lại Q12 và kiểm tra top-1 đổi sang tài liệu đúng |
| 3 | **Chia nhỏ 4 tài liệu legal theo Điều/Chương trước khi chunk, thay vì cắt phẳng 500 ký tự** | Q14 và Q09 vẫn hỏng ở Config C. Cân bằng `doc_type` chỉ sửa được tỉ lệ giữa hai nhóm; bên trong nhóm legal, *Sổ tay K60* (349 chunk) vẫn át ba tài liệu còn lại | Giúp câu hỏi bắc cầu legal↔news (Q14) và câu cần nhảy ngữ nghĩa (Q09). Cũng làm citation chỉ được tới "Điều 16" thay vì "chunk-274" | Chạy lại toàn bộ với chunker theo heading, so `context_recall` của Q09/Q13/Q14 |

Lưu ý về `context_recall`: cả ba config đều **0.7188 hoặc 0.7500**, tức là
khoảng một phần tư số câu không lấy được đủ chuỗi đáp án vào top-5. Đây là trần
của corpus và chunking hiện tại, không phải của chiến lược fusion — đó là lý do
khuyến nghị số 3 nhắm vào chunking chứ không nhắm vào retrieval.

### Hiệu chỉnh threshold

| | n | min | trung bình | max |
|---|---:|---:|---:|---:|
| Trong domain | 16 | 0.8634 | 0.8898 | 0.9236 |
| Ngoài domain | 2 | 0.8240 | 0.8535 | 0.8830 |

`multilingual-e5-small` đặt toàn bộ embedding trong một hình nón hẹp nên cosine
của **mọi** câu hỏi đều rơi vào dải 0.82–0.92. Với mặc định 0.3 của repo,
fallback không bao giờ chạy.

Hai phân bố **chồng lấn**: câu ngoài domain Q18 ("trường có đào tạo tiến sĩ
ngành Y khoa không?") đạt 0.8830, **cao hơn** câu trong domain Q09 (0.8634). Kết
luận: cosine tuyệt đối không phải bộ phát hiện ngoài domain đáng tin với họ
model này.

Điểm vận hành **0.853** đặt dưới câu in-domain yếu nhất một biên 0.01, nên không
đẩy nhầm câu đúng domain nào sang fallback. Nó bắt được câu ngoài domain rõ ràng
(Q17, 0.8240); câu "gần domain" như Q18 do tầng thứ hai xử lý — safe refusal ở
generation. Đo thực tế: **2/2 câu ngoài domain bị từ chối đúng ở cả ba config.**

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Cân bằng `doc_type` trong RRF (Config C) | Config A dense-only | average **+0.0343**, precision **+0.0810**, faithfulness +0.0455 | **−1195 ms/câu**, cost không đổi | Có tác dụng. Nên bật mặc định |
| Conversation memory cho câu hỏi nối tiếp | Không có lịch sử | Chưa đo định lượng | +~200 token/lượt | Đã cài trong `generate_with_trace(history=...)`, giới hạn 4 lượt gần nhất và chỉ dùng để hiểu đại từ, không làm nguồn dữ kiện. Chạy được trong UI; **chưa có phép đo nên không tự tính bonus** |
| UI citation/source highlighting | Không có | Không áp dụng | Không đổi | `[n]` bấm được, cuộn tới và làm nổi thẻ nguồn tương ứng; từ khoá truy vấn được bôi `<mark>` trong nội dung chunk |
| Đổi generator `gemini-2.5-flash-lite` → `gemini-3.1-flash-lite` | 2.5-flash-lite | Sửa 1/3 ca từ chối sai trong phép thử nhanh | Không đổi | 2.5-flash-lite từ chối Q02 dù chuỗi đáp án **có** trong context. Chọn model bằng đo, không bằng mặc định |

### Một lỗi đo suýt thành kết luận sai

Lần chạy A/B đầu tiên báo hybrid + RRF **tệ hơn hẳn** dense-only
(faithfulness −0.185, answer relevance −0.181). Sai.

`gemini-2.5-flash` giới hạn **20 request/ngày** ở free tier, còn một lượt A/B
cần 36. Config A dùng hết quota, nên mọi câu của Config B trả về HTTP 429 — và
code lúc đó ghi nhận lỗi provider thành "model từ chối trả lời". Config B trông
như đang từ chối liên tục.

Đã sửa ba chỗ: `generate_with_trace` phân biệt `refused` (model đọc context rồi
quyết định) với `errored` (chưa gọi được model); harness thử lại có chờ khi gặp
lỗi provider; và case vẫn lỗi thì bị **loại khỏi phép chấm** thay vì bị cho 0
điểm. Lần chạy báo cáo ở đây có **0 case bị loại**.

Bài học đáng ghi: một lỗi hạ tầng có thể giả dạng thành kết quả khoa học, và nó
giả dạng theo hướng *có vẻ hợp lý* — "thêm BM25 làm nhiễu context nên model từ
chối nhiều hơn" là một câu chuyện nghe rất xuôi tai. Các metric retrieval không
gọi LLM (`context_recall`, `context_precision`) không hề bị ảnh hưởng, và chính
chúng là thứ cho thấy có gì đó không khớp.
