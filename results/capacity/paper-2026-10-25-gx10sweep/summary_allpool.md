| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| mav-n1-gx3-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 3 | ok | 276.521 | [4.05, 4.595, 4.186] | 4.186 | 31.123 | 13.8 | [7, 5, 5] | 798.3 |
| mav-n4-gx3-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 3 | ok | 242.418 | [] | None | None | None | [6, 16, 11] | 438.6 |
| mav-n8-gx3-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 3 | ok | 254.447 | [] | None | None | None | [12, 8, 5] | 528.6 |
| kimi-n1-gx3-1 | kimi-k2.6-iq1_s | 3 | ok | 308.765 | [1.719, 1.829] | 1.774 | 3.132 | 333.8 | [8, 11, 6] | 734.3 |
| kimi-n8-gx3-1 | kimi-k2.6-iq1_s | 3 | ok | 334.641 | [] | None | None | None | [7, 8, 5] | 991.8 |
| ds671-n1-gx3-1 | deepseek-v3.1-ud-tq1_0 | 3 | load_failed | 1077.433 | [] | None | None | None | [21, 13, 20] | 2314.8 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mav-n1-gx3-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 1 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 4.186 | 238.9 | 0.0 | 13.8 | 13.8 | 0.0 | 1.0 |
| mav-n4-gx3-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 4 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 9.696 | 412.6 | 1.31 | 24.6 | 6.1 | 3.0 | 0.983 |
| mav-n8-gx3-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 8 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 15.167 | 528.5 | 4.303 | 19.5 | 2.4 | 6.0 | 0.986 |
| kimi-n1-gx3-1 | kimi-k2.6-iq1_s | 1 | ok | 2 (2 ok, 0 not concurrent, 0 skewed) | 1.774 | 564.3 | 0.0 | 333.8 | 333.8 | 0.0 | 1.0 |
| kimi-n8-gx3-1 | kimi-k2.6-iq1_s | 8 | ok | 2 (2 ok, 0 not concurrent, 0 skewed) | 4.986 | 1615.0 | 22.388 | 787.9 | 97.9 | 2.5 | 0.983 |
| ds671-n1-gx3-1 | deepseek-v3.1-ud-tq1_0 | 1 | load_failed | 0 (0 ok, 0 not concurrent, 0 skewed) | None | None | None | None | None | None | None |
