# Báo cáo Day 7

Sinh viên: Nguyễn Đắc Tuấn Nghĩa — MSSV 2A202603442. Mọi con số dưới đây lấy từ file do script sinh ra (`results/…`) hoặc từ nhãn audit (`results/agent/<split>/<cfg>/reviews.json`).

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: pydoc topics chính thức của Python 3.11 và 3.13 (`data/raw/pydoc_topics_py3.11.py`, `data/raw/pydoc_topics_py3.13.py`), giấy phép PSF-2.0 (`data/raw/PSF_LICENSE.txt`). Tiếp nối Day 6: 468 trang (topic/mục), 1415 chunk theo cấu hình Day 6 B (theo đoạn ≤ 1000 ký tự + dòng heading), lưu ở `data/chunks.json` (`results/index_info.json`).
- Vector store, embedding model, phiên bản và số chiều: Chroma 1.5.9, collection `day06_B_para1000` (cosine, HNSW); embedding WordLlama `l2_supercat` phiên bản 0.4.0.post1, 256 chiều. Dựng lại bằng `python -m scripts.build_index` (sync 1415 chunk, kiểm tra re-ingest 0 chunk mới).
- Agent model, cách chạy và phân chia dev/test: `gpt-4o-mini` (temperature 0) qua API tương thích OpenAI, tool calling thật (`src/agent.py`, `scripts/run_agent.py`). Benchmark `benchmark/questions.json`: dev = 20 câu Day 6 có đáp án + 3 câu thiếu bằng chứng (du1–du3); test = 6 câu mới có đáp án (t01–t06) + 2 câu thiếu bằng chứng (tu1–tu2). Benchmark viết trước mọi lần chạy Day 7.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: mỗi câu có đáp án có `evidence_quotes` trích nguyên văn từ tài liệu; `scripts/verify_benchmark.py` tìm mọi chunk/trang chứa trích dẫn đó để tạo `gold`. Câu thiếu bằng chứng được kiểm tra không có chunk nào chứa đáp án. Retrieval chấm tự động (Hit@1, Evidence Hit@3, Context Evidence Rate theo `gold`); câu trả lời agent chấm bằng nhãn audit theo `benchmark/labeling_guide.md` (grounded 0/1/2, citation support từng citation, abstention). Gold chỉ dùng để chấm, không đưa cho agent.

## Retrieval và thử nghiệm

Số liệu từ `results/retrieval/dev/summary.md` và `results/retrieval/test/summary.md` (dev N = 20, test N = 6 câu có đáp án; chi phí retrieval = 0 vì chạy local).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (R1) | Dense Chroma Day 6, top-3, filter `python_version` do người dùng nêu, không rerank, context 3 chunk / 4000 ký tự | Cấu hình Day 6 đã chọn (Day 6 C) | dev | 9/20 (45%) | 12/20 (60%) | 12/20 (60%) | 5.02 ms / 0 USD |
| Phương pháp 2 (R2) | BM25 (k1=1.5, b=0.75, tokenizer giữ định danh), top-3, filter, context 3 / 4000 | Tài liệu đầy định danh/keyword → khớp từ vựng tốt hơn embedding tĩnh yếu | dev | 15/20 (75%) | 17/20 (85%) | 17/20 (85%) | 1.79 ms / 0 USD |
| Phương pháp 3 (R3) | Hybrid RRF dense + BM25 (pool 30, k=60), top-3, filter, context 3 / 4000 | Hợp nhất rank giữ được cả thắng lợi của BM25 lẫn dense | dev | 14/20 (70%) | 18/20 (90%) | 18/20 (90%) | 12.82 ms / 0 USD |
| Cải thiện 1 (R4) | R3 + gộp chunk trùng nội dung (bản 3.11/3.13, compound vs topic riêng) | Miss q05/q08 của R3 có top-3 bị lấp bởi bản sao giống hệt → gộp trùng giải phóng chỗ | dev | 14/20 (70%) | 18/20 (90%) | 18/20 (90%) | 16.72 ms / 0 USD |
| Cải thiện 2 (R5) | R4 + context top-5 + chunk lân cận, budget 6000 ký tự | Evidence nằm ở chunk kề bên chunk trúng (q05) → đưa vào context | dev | 14/20 (70%) | 18/20 (90%) | 19/20 (95%) | 16.04 ms / 0 USD (context TB 5454 ký tự) |
| Cấu hình đã chọn (R5) | như trên | chốt trên dev | test | 4/6 (67%) | 6/6 (100%) | 6/6 (100%) | 17.69 ms / 0 USD (context TB 5309 ký tự) |
| Baseline (R1) | như trên | so sánh sau khi chốt | test | 3/6 (50%) | 3/6 (50%) | 3/6 (50%) | 6.24 ms / 0 USD (context TB 2665 ký tự) |

R2, R3 là phương pháp retrieval mới (lexical, hybrid); R4, R5 là cải thiện trên R3 (R5 đồng thời đổi chiến lược chọn context). Ngoài ra có R0 (dense không filter, kiểm tra tái lập Day 6) và R6 (R2 + gộp trùng, ablation) trong `results/retrieval/dev/summary.md`.

**Chấm context thủ công trên test** (BENCHMARK.md §2; 0 = không có evidence, 1 = có nhưng thiếu điều kiện hoặc nguồn, 2 = đủ evidence và nguồn). Chấm trên context cuối của retrieval (R1, R5: `context_ids` trong `results/retrieval/test/R*.json`) và của agent (A1, A4: `final_context_ids` trong trace). Chi tiết, lý do và chunk ID: `results/retrieval/test/context_grades.json`. Nhãn này được soạn **có hỗ trợ của AI**; sinh viên đã review 3 mục mức 1 (R1 t01, R1 t03, A1 t03) và đồng ý.

| Query ID | R1 | R5 | A1 | A4 | Lý do chính |
|---|---|---|---|---|---|
| t01 | 1 | 2 | 0 | 2 | R1 chỉ có `types#20:c000` (return trong generator → StopIteration) nhưng thiếu `StopIteration.value`; A1 chỉ có chunk async generator; R5/A4 có `return#1:c000` |
| t02 | 2 | 2 | 2 | 2 | `nonlocal#1:c000` nêu rõ SyntaxError |
| t03 | 1 | 2 | 1 | 2 | R1/A1 chỉ có chunk câu lệnh `with` (false → with re-raise, true → chặn), thiếu câu "__exit__() should not reraise"; R5 có `context-managers#1:c001`, A4 có `specialnames#23:c001` |
| t04 | 2 | 2 | 2 | 2 | `typesseq-mutable#1:c006` / `typesseq#4:c006` (3.11, đúng filter) |
| t05 | 0 | 2 | 0 | 2 | R1/A1 không có câu về giá trị của `__type_params__` cho đối tượng không generic (A1 chỉ có `compound#32:c001` nói thuộc tính liệt kê type parameter); R5/A4 có `specialattrs#1:c000` |
| t06 | 2 | 2 | 2 | 2 | `exceptions#1:c002` / `execmodel#…:c002`: message không thuộc API |

| Cấu hình | Mức 0 | Mức 1 | Mức 2 | Context Evidence Rate tự động |
|---|---|---|---|---|
| R1 (baseline retrieval) | 1 | 2 | 3 | 3/6 |
| R5 (retrieval đã chọn) | 0 | 0 | 6 | 6/6 |
| A1 (agent baseline) | 2 | 1 | 3 | 3/6 |
| A4 (agent đã chọn) | 0 | 0 | 6 | 6/6 |

Số câu mức 2 trùng với Context Evidence Rate tự động ở cả 4 cấu hình. Ba trường hợp mức 1 (R1 t01, R1 t03, A1 t03) đều là câu mà kiểm tra tự động báo "không có evidence": context có thông tin liên quan nhưng không chứa câu evidence gốc, nên chỉ đủ một phần.

Hai lỗi dev, nguyên nhân, cách thử và lý do chọn:

1. **d6-q05** ("How do I stop Python from showing the original exception as the cause…"): R1 top-3 = `raise#1:c002` (3.11), `raise#1:c002` (3.13), `exceptions#1:c002` — hai slot bị chiếm bởi hai bản giống hệt, còn câu "Exception chaining can be explicitly suppressed by specifying None in the from clause" nằm ở chunk kề `raise#1:c003`. Thử R4 (gộp trùng) — vẫn miss@3; R5 thêm chunk lân cận vào context nên Context Evidence Rate có câu này. Ở agent, A1 (dense) trả lời sai ("raise không dùng from"), A2/A3/A4 (R5) trả lời đúng `from None` có citation `raise#1:c003`.
2. **d6-q08** ("Do the and/or operators always return True or False?"): câu evidence ("…return the last evaluated argument") nằm ở topic boolean operations, nhưng cả R1 và R5 đều lấy `truth#1:c000` (Truth Value Testing) và các chunk so sánh; đây là miss duy nhất về context của R5 trên dev. Câu hỏi diễn đạt bằng "True or False", trùng từ với trang truth value hơn trang toán tử. Không sửa được bằng gộp trùng/lân cận; tuy vậy `truth#1:c000` có câu "the Boolean operations or and and always return one of their operands" nên agent với R5 vẫn trả lời đúng (nhãn 2), còn A1 từ chối.

Lý do chọn R5: Evidence Hit@3 hòa 18/20 giữa R3, R4, R5, R6; theo tiêu chí thứ 2 (Context Evidence Rate) R5 = 19/20 cao nhất (`benchmark/selection_criteria.md`).

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: hai tool — `search(query, python_version?)` (model tự suy ra filter phiên bản từ câu hỏi) và `final_answer(answer, citations, abstain)`. Tối đa `max_searches` = 3 lượt search (A3: 1); khi hết lượt, tool trả "Search limit reached" và yêu cầu trả lời hoặc từ chối. Dừng khi: `final_answer`; model trả lời không gọi tool 2 lần (→ abstain); vượt số lượt (→ abstain). Context: mỗi lần search thêm chunk mới (top `context_k`, không lặp ID) vào context, cắt theo budget 8000 ký tự. Citation phải là ID chunk đã được hiển thị **và** xuất hiện dạng `[id]` trong câu trả lời; nếu không, A1–A3 chuyển thành abstain (`invalid_or_missing_citations`), A4/A5 được phép sửa 1 lần (`citation_repairs = 1`) với thông báo lỗi cụ thể, quy tắc kiểm tra giữ nguyên. Lỗi provider được retry và ghi `status = error` (0 lỗi trong mọi lần chạy).

Bảng tổng hợp (từ `metrics.json` + `auto_summary.json`; "Số lượt gọi" = tổng số search / tổng số LLM call):

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A1 (R1 dense, baseline) / dev | 23/23 | 43 / 78 | 11/3/6 — 0.75 | 8/12 (67%) | 10/10 câu answered | 3/3 | 10/20 |
| A2 (R5) / dev | 23/23 | 38 / 76 | 7/0/13 — 1.30 | 21/22 (95%) | 13/13 | 3/3 | 7/20 |
| A3 (R5, 1 search) / dev | 23/23 | 23 / 63 | 8/0/12 — 1.20 | 18/18 (100%) | 13/13 | 3/3 | 7/20 |
| A4 (R5 + sửa citation) / dev | 23/23 | 36 / 80 | 1/1/18 — **1.85** | 25/28 (89%) | 19/19 | 3/3 | 1/20 |
| A5 (R1 + sửa citation) / dev | 23/23 | 39 / 76 | 2/5/13 — 1.55 | 19/22 (86%) | 19/19 | 3/3 | 1/20 |
| A1 (baseline) / test | 8/8 | 16 / 26 | 3/1/2 — 0.83 | 3/5 (60%) | 4/4 | 1/2 | 3/6 |
| A4 (đã chọn) / test | 8/8 | 13 / 26 | 0/0/6 — **2.00** | 9/9 (100%) | 6/6 | 2/2 | 0/6 |

Chọn A4 trên dev theo quy tắc chốt trước (grounded mean cao nhất, xem `benchmark/selection_criteria.md`), sau đó chạy test một lần cho A1 và A4.

Kết quả từng query trên test:

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t01 (A1) | có đáp án | 2 search "return statement in generator function"; context không có evidence | StopIteration, nhưng nói sai "non-empty return là syntax error trong generator"; `[return#1:c001]` (async generator) | 1 — đúng một phần (StopIteration) nhưng có nhận định sai và chunk trích về async generator (sinh viên sửa 0 → 1 khi review) |
| t01 (A4) | có đáp án | 1 search; context có evidence | StopIteration + StopIteration.value; `[py3.11/return#1:c000]` | 2 — đúng, được chunk hỗ trợ |
| t02 (A1) | có đáp án | 1 search "nonlocal statement name not bound"; context có evidence | abstain (`invalid_or_missing_citations`) | 0 — từ chối nhầm do định dạng citation |
| t02 (A4) | có đáp án | 1 search; context có evidence | SyntaxError; `[py3.13/nonlocal#1:c000][py3.13/naming#3:c003]` | 2 — cả hai chunk hỗ trợ |
| t03 (A1) | có đáp án | 1 search "__exit__ method re-raise exception"; context không có evidence | abstain (`invalid_or_missing_citations`) | 0 — từ chối, chunk evidence không được lấy |
| t03 (A4) | có đáp án | 1 search; context có evidence | Không re-raise, trả giá trị true để chặn; `[py3.13/specialnames#23:c001][py3.11/specialnames#23:c001]` | 2 — đúng, được hỗ trợ |
| t04 (A1) | có đáp án (filter 3.11) | 2 search, filter 3.11; context có evidence | Xóa sạch sequence; `[py3.11/typesseq-mutable#1:c006]` | 2 — đúng, được hỗ trợ |
| t04 (A4) | có đáp án (filter 3.11) | 2 search, filter 3.11; context có evidence | Xóa sạch sequence; `[py3.11/typesseq#4:c006]` | 2 — đúng, được hỗ trợ |
| t05 (A1) | có đáp án (filter 3.13) | 3 search "__type_params__…", filter 3.13; context không có evidence | abstain (`invalid_or_missing_citations`) | 0 — từ chối, chunk evidence không được lấy |
| t05 (A4) | có đáp án (filter 3.13) | 1 search, filter 3.13; context có evidence | Tuple rỗng; `[py3.13/specialattrs#1:c000]` | 2 — đúng, được hỗ trợ |
| t06 (A1) | có đáp án | 1 search; context có evidence | Không, message không thuộc API; `[py3.11/execmodel#8:c002][py3.13/exceptions#1:c002]` | 2 — cả hai chunk hỗ trợ |
| t06 (A4) | có đáp án | 1 search; context có evidence | Như trên; `[py3.11/exceptions#1:c002][py3.13/exceptions#1:c002]` | 2 — cả hai chunk hỗ trợ |
| tu1 (A1) | thiếu bằng chứng | 3 search về `sys.getrecursionlimit` | Trả lời "1000" với `[py3.11/execmodel#8:c001]` (không liên quan) | abstention sai — bịa từ kiến thức ngoài |
| tu1 (A4) | thiếu bằng chứng | 3 search cùng nội dung | abstain | abstention đúng |
| tu2 (A1) | thiếu bằng chứng | 3 search `int_max_str_digits` | abstain | abstention đúng |
| tu2 (A4) | thiếu bằng chứng | 3 search | abstain | abstention đúng |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí: trace `results/agent/{dev,test}/<cfg>/traces.json` (mọi event: llm_call, search kèm query/filter/rank/score, context IDs, usage, latency); nhãn `results/agent/{dev,test}/<cfg>/reviews.json`; metric `metrics.json` và `auto_summary.json` cùng thư mục. Lệnh: `python -m scripts.run_agent --split dev --configs A1,A2,A3,A4,A5`, `python -m scripts.run_agent --split test --configs A1,A4 --sleep 1`, rồi `python starter/agent_metrics.py --cases … --traces … --reviews … --split <split> --output …/metrics.json`. Chi phí/thời gian (gpt-4o-mini):

| Cấu hình / split | Prompt tokens | Completion tokens | Chi phí (USD) | Latency TB / câu (s) | Context cuối TB (ký tự) | Context evidence |
|---|---|---|---|---|---|---|
| A1 / dev | 90548 | 3719 | 0.015813 | 5.71 | 4143 | 11/20 |
| A2 / dev | 133368 | 4779 | 0.022873 | 6.09 | 7399 | 17/20 |
| A3 / dev | 100522 | 4677 | 0.017885 | 5.86 | 7319 | 17/20 |
| A4 / dev | 143800 | 5572 | 0.024912 | 6.26 | 7394 | 17/20 |
| A5 / dev | 81572 | 4096 | 0.014694 | 5.03 | 3524 | 11/20 |
| A1 / test | 29717 | 1305 | 0.005241 | 5.63 | 4244 | 3/6 |
| A4 / test | 45315 | 1729 | 0.007835 | 5.93 | 7236 | 6/6 |

- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
  - **Trước/sau.** Dev: A1 → A4 grounded mean 0.75 → 1.85, từ chối nhầm 10/20 → 1/20. Test (chạy một lần sau khi chốt): A1 0.83 → A4 2.00, từ chối nhầm 3/6 → 0/6, abstention đúng 1/2 → 2/2. Tách tác động: chỉ đổi retrieval (A1 → A2) tăng 0.75 → 1.30; chỉ thêm sửa citation trên dense (A1 → A5) tăng 0.75 → 1.55; cả hai (A4) cho 1.85. Giảm còn 1 search (A3) làm grounded giảm từ 1.30 xuống 1.20 so với A2.
  - **Lỗi từ chối nhầm do định dạng citation (động lực của A4/A5).** Trong trace dev, lần dừng `invalid_or_missing_citations` xảy ra ở 8/23 câu với A1, 6/23 với A2, 7/23 với A3: model đưa ID vào `citations` nhưng không viết `[id]` trong câu trả lời (ví dụ d6-q09 A1: câu trả lời "SyntaxError" đúng nhưng bị chuyển thành abstain). A4/A5 cho phép sửa 1 lần: 6 yêu cầu sửa trên dev cho mỗi cấu hình, và không còn lần dừng `invalid_or_missing_citations` nào. Trên test, A1 vẫn có 3 lần như vậy, A4 có 3 yêu cầu sửa và trả lời được cả 6 câu.
  - **d6-q11 (walrus) — A4 từ chối nhầm.** Câu hỏi không nêu phiên bản nhưng model tự đặt filter `python_version = 3.11` cho cả 3 lần search; topic "Assignment expressions" chỉ có trong pydoc 3.13 (`B:py3.13/assignment-expressions#1:c000`), nên filter loại mất evidence → abstain. A1 thử filter 3.13 ở lần 2 và trả lời đúng; A2 tìm thấy chunk này ở rank 1 ở lần search thứ 3 (filter 3.13), nhưng context đã đầy budget 8000 ký tự sau lần search đầu (7758 ký tự, 9 chunk 3.11 kèm chunk lân cận) nên chunk không được thêm vào context → vẫn từ chối. Đây là lỗi suy luận filter của agent cộng với chính sách context "ai đến trước giữ chỗ", không phải lỗi index; hướng sửa (chưa làm): không đặt filter khi câu hỏi không nêu phiên bản, và cho chunk mới có rank cao thay thế chunk cũ khi hết budget.
  - Các lỗi còn lại của A4 trên dev (nhãn < 2): d6-q16 nhãn 1 (đúng "có" nhưng chunk trích là ví dụ match, không nói binding còn sau khối); citation không hỗ trợ ở d6-q03 (`try#3:c001`), d6-q14 (`debugger#2:c005`) và d6-q16 (`match#2:c002`).
  - **Đánh đổi.** A4 tốn nhiều token nhất trên dev (143800 prompt tokens, 0.024912 USD, 6.26 s/câu) do context R5 dài hơn (~7400 ký tự so với ~4100 của A1) và thêm lượt sửa citation; so với A1 chi phí dev tăng khoảng 1.6 lần. A5 rẻ nhất (0.014694 USD, 5.03 s) nhưng grounded thấp hơn (1.55) vì dense hay bỏ sót evidence (context evidence 11/20). Retrieval R5 chậm hơn R1 (16.04 ms so với 5.02 ms trên dev) nhưng không đáng kể so với thời gian LLM (giây).
  - **Giới hạn.** Benchmark nhỏ (dev 23 câu, test 8 câu) nên chênh lệch một vài câu đã đổi metric đáng kể; test chỉ 6 câu có đáp án. Chỉ dùng một model (gpt-4o-mini, temperature 0) và chạy mỗi cấu hình một lần, chưa đo phương sai. Dev dùng lại 20 câu Day 6 (đã thấy kết quả Day 6) nên chỉ dùng làm dev. Nhãn audit được soạn **có hỗ trợ của AI**; sinh viên đã review 26 mục cần phán đoán (24 đồng ý, 2 sửa 0 → 1), các nhãn còn lại vẫn là nhãn có AI hỗ trợ. Trên Windows cần đặt `PYTHONUTF8=1` để một test của starter đọc file UTF-8 đúng.

### Khai báo sử dụng AI

Claude (trợ lý lập trình AI) đã hỗ trợ viết code (`src/`, `scripts/`), soạn benchmark, gán nhãn audit (đọc từng trace cùng toàn văn chunk được trích) và viết báo cáo này. Mọi số liệu lấy từ file kết quả do script sinh ra. Nhãn được soạn có AI hỗ trợ; sinh viên đã review 26 mục cần phán đoán (ngày 2026-11-14): 24 đồng ý, 2 sửa 0 → 1 (test A1 t01, dev A5 d6-q19). Các nhãn còn lại vẫn là nhãn có AI hỗ trợ; sinh viên chịu trách nhiệm về toàn bộ nội dung nộp.

## Thử thách nâng cao (nếu thực hiện)

Không làm.

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản: không làm.
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý: không làm.
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev: không làm.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | — | — | không làm |
| Ảnh, sơ đồ | — | — | không làm |
| Biểu đồ | — | — | không làm |
| Multi-hop | — | — | không làm |
| Text | — | — | không làm |
| Thiếu bằng chứng | — | — | không làm |
| **Toàn bộ** | — | — | không làm |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống: không làm.
- Giới hạn, cách tái lập và đường dẫn benchmark: không làm.
