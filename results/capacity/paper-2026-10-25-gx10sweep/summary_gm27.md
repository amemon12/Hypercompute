| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| gm27-n1-gxA-1 | gemma-3-27b-it-q4_k_m | 0 | ok | 14.073 | [12.343, 12.338, 12.344] | 12.343 | 396.128 | 0.0 | [] | 44.8 |
| gm27-n4-gxA-1 | gemma-3-27b-it-q4_k_m | 0 | ok | 4.05 | [] | None | None | None | [] | 44.9 |
| gm27-n8-gxA-1 | gemma-3-27b-it-q4_k_m | 0 | ok | 4.049 | [] | None | None | None | [] | 50.2 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gm27-n1-gxA-1 | gemma-3-27b-it-q4_k_m | 1 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 12.343 | 81.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| gm27-n4-gxA-1 | gemma-3-27b-it-q4_k_m | 4 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 45.354 | 88.2 | 0.009 | 0.0 | 0.0 | 0.0 | 0.998 |
| gm27-n8-gxA-1 | gemma-3-27b-it-q4_k_m | 8 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 84.986 | 94.3 | 0.037 | 0.0 | 0.0 | 0.0 | 0.994 |
