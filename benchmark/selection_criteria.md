# Tiêu chí chọn cấu hình Day 7 (chốt 2026-11-14, trước mọi lần chạy Day 7)

Chỉ dùng **dev** để chọn. Thứ tự ưu tiên:
1. Retrieval: Evidence Hit@3 trên 20 câu dev có đáp án (query gốc, cùng định nghĩa hit).
2. Context Evidence Rate trên dev (context cuối gửi model).
3. Hit@1 dev.
4. Thời gian truy vấn trung bình (ms) — chi phí retrieval đều = 0 (local).
Với agent (chạy bằng LLM thật): grounded answer trung bình /2 dev, abstention đúng trên 3 câu thiếu evidence dev, số câu có đáp án bị từ chối nhầm, rồi số token / thời gian.

Sau khi chốt, chạy test **một lần** cho baseline (dense) và cấu hình đã chọn; không chỉnh gì dựa trên test.

## Kết quả chọn trên dev (retrieval, ghi trước khi chạy test)

Dev (20 câu có đáp án): Evidence Hit@3 hòa ở 18/20 giữa R3, R4, R5, R6. Tiêu chí 2 (Context Evidence Rate): R5 = 19/20, các cấu hình còn lại ≤ 18/20 → **chọn R5** (hybrid RRF + filter + gộp trùng + context top-5 có chunk lân cận, budget 6000 ký tự).
Baseline để so sánh trên test: **R1** (dense Day 6 + filter = cấu hình Day 6 đã chọn).

## Chọn cấu hình agent trên dev

Quy tắc (chốt trước khi chạy test): grounded answer trung bình cao nhất → hòa thì abstention đúng nhiều hơn → ít từ chối nhầm hơn → ít `prompt_tokens` hơn (`auto_summary.json`).
Nhãn audit: `results/agent/dev/<cfg>/reviews.json` (gán nhãn có AI hỗ trợ, xem `benchmark/labeling_guide.md`); số liệu từ `metrics.json` và `auto_summary.json`. Dev = 20 câu có đáp án + 3 câu không có đáp án, model gpt-4o-mini, 0 lỗi provider.

| Cấu hình | Retrieval | Grounded mean (/2) | 0 / 1 / 2 | Citation validity | Abstention đúng | Từ chối nhầm | Context evidence | Prompt tokens | Chi phí (USD) | Latency TB (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 (baseline) | R1 | 0.75 | 11 / 3 / 6 | 8/12 | 3/3 | 10/20 | 11/20 | 90548 | 0.015813 | 5.71 |
| A2 | R5 | 1.30 | 7 / 0 / 13 | 21/22 | 3/3 | 7/20 | 17/20 | 133368 | 0.022873 | 6.09 |
| A3 | R5 (1 search) | 1.20 | 8 / 0 / 12 | 18/18 | 3/3 | 7/20 | 17/20 | 100522 | 0.017885 | 5.86 |
| A4 | R5 + sửa citation | **1.85** | 1 / 1 / 18 | 25/28 | 3/3 | 1/20 | 17/20 | 143800 | 0.024912 | 6.26 |
| A5 | R1 + sửa citation | 1.55 | 2 / 5 / 13 | 19/22 | 3/3 | 1/20 | 11/20 | 81572 | 0.014694 | 5.03 |

**Chọn A4** (grounded mean cao nhất 1.85, không cần dùng tiêu chí phụ). Sau khi sinh viên review nhãn (2026-11-14), d6-q19 của A5 được sửa 0 → 1 nên A5 đổi từ 1.50 (3/4/13) thành 1.55 (2/5/13); áp lại quy tắc, A4 vẫn được chọn. Baseline so sánh trên test: **A1**.
Test chạy một lần: `python -m scripts.run_agent --split test --configs A1,A4 --sleep 1`.
