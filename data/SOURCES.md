# Nguồn dữ liệu — provenance của corpus

**Chủ đề:** Dịch vụ sinh viên và quy chế đào tạo — Trường Đại học Giao thông Vận tải (UTC).

Toàn bộ corpus lấy từ các tên miền công khai của `utc.edu.vn` (cổng chính và
các trang đơn vị/phân hiệu). Không tài liệu nào nằm sau tường đăng nhập, và
không có bước nào vượt qua WAF hay hạn chế truy cập.

`characters` đếm phần nội dung của file trong `data/standardized/`, **không**
tính khối metadata ở đầu file, tức là số ký tự còn lại sau khi đã làm sạch.

| doc_id | title | type | source URL | retrieved | characters | licence |
|---|---|---|---|---|---|---|
| `de-an-tuyen-sinh-2025` | Đề án tuyển sinh đại học 2025 | legal | <https://tuyensinh.utc.edu.vn/sites/ts.utc.edu.vn/files/29.5.25_%C4%90%E1%BB%81%20%C3%A1n%20tuy%E1%BB%83n%20sinh%202025.pdf> | 2026-09-20 | 115,982 | public source, official university website |
| `qc-dao-tao-thac-si-2021` | Quy chế tuyển sinh và đào tạo trình độ thạc sĩ (2021) | legal | <https://www.utc.edu.vn/Upload/FilePost/2022/06/09/qc-thac-si-2021.pdf> | 2026-09-20 | 91,151 | public source, official university website |
| `so-tay-ho-tro-co-van-hoc-tap` | Sổ tay hỏi đáp hỗ trợ cố vấn học tập và sinh viên | legal | <https://dee.utc.edu.vn/sites/dee.utc.edu.vn/files/Sotay%20Hotro%20CVHT-SV%20DDT%20V1.0.pdf> | 2026-09-20 | 25,132 | public source, official university website |
| `so-tay-sinh-vien-k60` | Sổ tay sinh viên K60 | legal | <https://fme.utc.edu.vn/sites/fme.utc.edu.vn/files/SO%20TAY%20SINH%20VIEN%20K60.pdf> | 2026-09-20 | 140,719 | public source, official university website |
| `phong-bao-ve` | Phòng Bảo vệ | news | <https://www.utc.edu.vn/gioi-thieu/phong-bao-ve> | 2026-09-20 | 1,927 | public source, official university website |
| `phong-cong-tac-chinh-tri-va-sinh-vien` | Phòng chăm sóc người học | news | <https://www.utc.edu.vn/gioi-thieu/phong-cong-tac-chinh-tri-va-sinh-vien> | 2026-09-20 | 2,652 | public source, official university website |
| `phong-dao-tao-dai-hoc` | Phòng Đào tạo đại học | news | <https://www.utc.edu.vn/gioi-thieu/phong-dao-tao-dai-hoc> | 2026-09-20 | 3,684 | public source, official university website |
| `trung-tam-dao-tao-truc-tuyen-utc` | Trung tâm Đào tạo trực tuyến UTC | news | <https://www.utc.edu.vn/gioi-thieu/trung-tam-dao-tao-truc-tuyen-utc> | 2026-09-20 | 3,299 | public source, official university website |
| `phong-ke-hoach-tai-chinh` | Phòng Kế hoạch - Tài chính | news | <https://www.utc.edu.vn/gioi-thieu/phong-ke-hoach-tai-chinh> | 2026-09-20 | 4,456 | public source, official university website |
| `ban-quan-ly-ky-tuc-xa` | Ban Quản lý ký túc xá | news | <https://www.utc.edu.vn/gioi-thieu/ban-quan-ly-ky-tuc-xa> | 2026-09-20 | 2,159 | public source, official university website |
| `phong-phap-che-va-kiem-soat-noi-bo` | Phòng Pháp chế và kiểm soát nội bộ | news | <https://www.utc.edu.vn/gioi-thieu/phong-phap-che-va-kiem-soat-noi-bo> | 2026-09-20 | 2,645 | public source, official university website |
| `phong-quan-ly-chat-luong` | Phòng Quản lý chất lượng | news | <https://www.utc.edu.vn/gioi-thieu/phong-quan-ly-chat-luong> | 2026-09-20 | 2,884 | public source, official university website |
| `trung-tam-thong-tin-thu-vien` | Trung tâm thông tin thư viện | news | <https://www.utc.edu.vn/gioi-thieu/trung-tam-thong-tin-thu-vien> | 2026-09-20 | 3,407 | public source, official university website |
| `tram-y-te` | Trạm Y tế | news | <https://www.utc.edu.vn/gioi-thieu/tram-y-te> | 2026-09-20 | 2,450 | public source, official university website |

**Tổng cộng:** 4 văn bản quy chế (372,984 ký tự) + 10 trang bài viết (29,563 ký tự) = 14 tài liệu, 402,547 ký tự.

## Cách tái lập

```bash
python -m src.task1_collect_legal_docs   # tải PDF + ghi MANIFEST.json
python -m src.task2_crawl_news           # crawl 10 trang + làm sạch
python -m src.task3_convert_markdown     # chuẩn hoá về Markdown
```

Cả ba bước đều *idempotent*: chạy lại ghi đè tại chỗ, không nhân bản file.
`data/landing/legal/MANIFEST.json` lưu SHA-256 và dung lượng từng PDF nên
kiểm chứng được file trên đĩa đúng là file đã tải từ URL ghi trong bảng.

## Xử lý dữ liệu cá nhân

Các trang giới thiệu đơn vị có bảng **danh sách cán bộ kèm số di động và
email cá nhân**. Bộ crawl cắt bỏ toàn bộ khối này (mốc `ĐỘI NGŨ` / `DANH SÁCH
CÁN BỘ` / tiêu đề cột `Họ tên`), che số điện thoại trên dòng có `ĐT:`, và bỏ
dòng chứa địa chỉ `@utc.edu.vn` không thuộc danh sách hộp thư đơn vị. Nhờ vậy
**không có số di động hay email cá nhân nào của cán bộ đi vào repo**; phần giữ
lại chỉ gồm địa chỉ, số tổng đài và hộp thư chung của đơn vị.

Hai ngoại lệ có chủ ý, đều nằm trong văn bản chính thức do trường tự công bố:

* Tên người đứng đầu đơn vị (vd `Trưởng phòng: ...`) trong khối THÔNG TIN
  CHUNG — thông tin tổ chức công khai, không kèm liên hệ cá nhân.
* Hotline tuyển sinh in trong Đề án tuyển sinh 2025 và Sổ tay sinh viên —
  số điện thoại được công bố chính để thí sinh và sinh viên gọi đến.

