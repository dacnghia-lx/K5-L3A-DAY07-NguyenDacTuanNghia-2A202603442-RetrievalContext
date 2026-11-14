| Config | Mô tả | Hit@1 | Evidence Hit@3 | Context Evidence Rate | MRR@10 | Context TB (ký tự) | Latency TB | Chi phí |
|---|---|---|---|---|---|---|---|---|
| R0 | Dense Day 6 (B), không filter | 9/20 (45%) | 12/20 (60%) | 12/20 (60%) | 0.547 | 2966 | 4.94 ms | 0 |
| R1 | Dense Day 6 + filter phiên bản (= Day 6 C) — baseline | 9/20 (45%) | 12/20 (60%) | 12/20 (60%) | 0.559 | 2928 | 5.02 ms | 0 |
| R2 | BM25 (tokenizer giữ định danh) + filter | 15/20 (75%) | 17/20 (85%) | 17/20 (85%) | 0.802 | 2887 | 1.79 ms | 0 |
| R3 | Hybrid RRF dense+BM25 (pool 30, k=60) + filter | 14/20 (70%) | 18/20 (90%) | 18/20 (90%) | 0.783 | 2914 | 12.82 ms | 0 |
| R4 | R3 + gộp chunk trùng nội dung (3.11/3.13, compound) | 14/20 (70%) | 18/20 (90%) | 18/20 (90%) | 0.797 | 2934 | 16.72 ms | 0 |
| R5 | R4 + context top-5 + chunk lân cận, budget 6000 ký tự | 14/20 (70%) | 18/20 (90%) | 19/20 (95%) | 0.797 | 5454 | 16.04 ms | 0 |
| R6 | R2 + gộp trùng (ablation) | 15/20 (75%) | 18/20 (90%) | 18/20 (90%) | 0.817 | 2908 | 5.84 ms | 0 |
