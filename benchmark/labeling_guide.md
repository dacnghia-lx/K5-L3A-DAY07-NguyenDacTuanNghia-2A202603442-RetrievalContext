# Hướng dẫn gán nhãn audit agent (Day 7)

Nhãn nằm ở `results/agent/<split>/<config>/reviews.json`, đúng định dạng `starter/agent_metrics.py` cần:
mỗi phần tử là `{query_id, answer_grade, citation_support, abstention_correct, note}`.

> **Lưu ý minh bạch:** nhãn được tạo **với sự hỗ trợ của AI** (Claude), bằng cách đọc từng trace: câu hỏi,
> `expected_answer` và `evidence_quotes` của benchmark, câu trả lời của agent, và **toàn văn** từng chunk được trích dẫn
> trong `data/chunks.json`. Sau đó sinh viên đã review 26 mục cần phán đoán (2026-11-14):
> 24 mục đồng ý, 2 mục sửa 0 → 1 (test A1 t01, dev A5 d6-q19). Các nhãn còn lại vẫn là nhãn có AI hỗ trợ; số liệu agent trong báo cáo
> phụ thuộc trực tiếp vào các nhãn này. Đáp án chuẩn (gold) chỉ dùng để chấm, không bao giờ đưa cho agent.

## Câu có đáp án (`answerable = true`)

| answer_grade | Ý nghĩa |
|---|---|
| 2 | Trả lời đúng **và** mọi nhận định chính đều được các chunk đã trích dẫn hỗ trợ |
| 1 | Đúng một phần, **hoặc** đúng nhưng một nhận định chính không được chunk trích dẫn hỗ trợ (trích sai chunk) |
| 0 | Sai / không có căn cứ, **hoặc** agent từ chối (`status == "abstain"` bắt buộc = 0) |

- `abstention_correct`: `null`.
- Quy ước áp dụng khi chấm: chi tiết phụ (ví dụ nhắc thêm phiên bản, câu so sánh bên lề) đúng theo tài liệu nhưng
  không nằm trong chunk trích dẫn thì **không** hạ điểm; lý do bịa ra (khác tài liệu) hoặc câu trả lời tự mâu thuẫn thì hạ điểm.

## Câu không có đáp án (`answerable = false`)

- `answer_grade`: `null`.
- `abstention_correct`: `true` chỉ khi `status == "abstain"`; câu trả lời tự bịa nội dung là `false`.

## citation_support

- Một giá trị boolean cho **mỗi** phần tử của danh sách `citations` trong trace, đúng thứ tự.
- `true` chỉ khi nội dung chunk đó thực sự hỗ trợ câu / đoạn mà nó được gắn vào; ID hợp lệ nhưng nội dung không liên quan là `false`.
- Không có citation → danh sách rỗng `[]`.

## note

Lý do ngắn bằng tiếng Anh (≤ 20 từ), ví dụ "Abstained although answerable; evidence was in context."
