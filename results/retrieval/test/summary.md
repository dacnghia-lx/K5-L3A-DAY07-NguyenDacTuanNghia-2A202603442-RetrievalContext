| Config | Mô tả | Hit@1 | Evidence Hit@3 | Context Evidence Rate | MRR@10 | Context TB (ký tự) | Latency TB | Chi phí |
|---|---|---|---|---|---|---|---|---|
| R1 | Dense Day 6 + filter phiên bản (= Day 6 C) — baseline | 3/6 (50%) | 3/6 (50%) | 3/6 (50%) | 0.582 | 2665 | 6.24 ms | 0 |
| R5 | R4 + context top-5 + chunk lân cận, budget 6000 ký tự | 4/6 (67%) | 6/6 (100%) | 6/6 (100%) | 0.806 | 5309 | 17.69 ms | 0 |
