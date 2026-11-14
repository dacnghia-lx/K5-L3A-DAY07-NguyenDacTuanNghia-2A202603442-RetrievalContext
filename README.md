# Bài nộp Day 7 — Nguyễn Đắc Tuấn Nghĩa (2A202603442)

> Phần đề bài gốc nằm dưới đường kẻ. Báo cáo: `report/individual/2A202603442_NguyenDacTuanNghia.md`.

**Tiếp nối Day 6:** cùng corpus (pydoc topics Python 3.11 + 3.13, PSF-2.0), cùng chunk (Day 6 B: theo đoạn ≤1000 ký tự + dòng heading, 1415 chunk), cùng embedding WordLlama `l2_supercat` 256 chiều và Chroma 1.5.9 collection `day06_B_para1000` (cosine, HNSW). Agent dùng LLM thật qua API tương thích OpenAI với tool calling; các kết quả trong repo chạy bằng OpenAI `gpt-4o-mini` (temperature 0). Có thể đổi sang Gemini/OpenRouter… qua `LLM_BASE_URL`, `LLM_MODEL` trong `.env` (xem `.env.example`).

## Chạy lại

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.build_index            # dựng lại index Day 6 trong repo này + data/chunks.json
python -m scripts.verify_benchmark       # kiểm chứng evidence / câu thiếu bằng chứng
python -m unittest discover -s tests -v  # test mock (chỉ kiểm tra code)
python -m scripts.eval_retrieval --split dev                 # 7 cấu hình retrieval R0–R6
python -m scripts.eval_retrieval --split test --configs R1,R5
# Agent: copy .env.example -> .env, điền LLM_API_KEY
python -m scripts.check_llm
python -m scripts.run_agent --split dev --configs A1,A2,A3,A4,A5 --sleep 1
python -m scripts.run_agent --split test --configs A1,A4 --sleep 1
python starter/agent_metrics.py --cases results/agent/test/A4/cases.json --traces results/agent/test/A4/traces.json --reviews results/agent/test/A4/reviews.json --split test --output results/agent/test/A4/metrics.json
```

| Nội dung | Đường dẫn |
|---|---|
| Code | `src/` (`retrieval.py`, `agent.py`, `llm.py`, `pipeline.py`, `corpus.py`, `day6/` = module Day 6) · `scripts/` |
| Dữ liệu | `data/raw/` (nguồn + PSF license), `data/chunks.json` (chunk Day 6 ở schema Day 7) |
| Benchmark | `benchmark/questions.json`, `benchmark/selection_criteria.md`, `benchmark/labeling_guide.md` |
| Thử nghiệm | `artifacts/experiment_log.csv`, `results/retrieval/{dev,test}/`, `results/agent/{dev,test}/<config>/` |

## Kết quả agent

Model `gpt-4o-mini` (temperature 0). Trên Windows đặt `PYTHONUTF8=1` trước mọi lệnh Python (PowerShell: `$env:PYTHONUTF8="1"`); một test của starter cần biến này để đọc file UTF-8 đúng.

```bash
python -m scripts.run_agent --split dev --configs A1,A2,A3,A4,A5
python -m scripts.run_agent --split test --configs A1,A4 --sleep 1
python starter/agent_metrics.py --cases results/agent/dev/A4/cases.json --traces results/agent/dev/A4/traces.json --reviews results/agent/dev/A4/reviews.json --split dev --output results/agent/dev/A4/metrics.json
python starter/agent_metrics.py --cases results/agent/test/A4/cases.json --traces results/agent/test/A4/traces.json --reviews results/agent/test/A4/reviews.json --split test --output results/agent/test/A4/metrics.json
```

Nhãn audit (`reviews.json`) gán theo `benchmark/labeling_guide.md` (soạn có AI hỗ trợ; sinh viên đã review 26 mục cần phán đoán: 24 đồng ý, 2 sửa 0 → 1). Chọn **A4** (R5 hybrid + gộp trùng + chunk lân cận, 1 lần sửa citation) trên dev theo `benchmark/selection_criteria.md`.

| Cấu hình / split | Grounded mean (/2) | Citation hợp lệ | Abstention đúng | Từ chối nhầm | Chi phí (USD) |
|---|---|---|---|---|---|
| A1 baseline / dev | 0.75 | 8/12 | 3/3 | 10/20 | 0.015813 |
| A4 đã chọn / dev | 1.85 | 25/28 | 3/3 | 1/20 | 0.024912 |
| A1 baseline / test | 0.83 | 3/5 | 1/2 | 3/6 | 0.005241 |
| A4 đã chọn / test | 2.00 | 9/9 | 2/2 | 0/6 | 0.007835 |

---

# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
