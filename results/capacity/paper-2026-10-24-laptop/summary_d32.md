| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| d32-g14A-1 | qwen2.5-32b-instruct-q4_k_m | 0 | aborted_by_guard | 20.069 | [] | None | None | None | [7, 5] | 23.9 |
| d32-g14P2-1 | qwen2.5-32b-instruct-q4_k_m | 2 | aborted_by_guard | 36.121 | [] | None | None | None | [7, 7] | 53.0 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d32-g14A-1 | qwen2.5-32b-instruct-q4_k_m | 1 | aborted_by_guard | 0 (0 ok, 0 not concurrent, 0 skewed) | None | None | None | None | None | None | None |
| d32-g14P2-1 | qwen2.5-32b-instruct-q4_k_m | 1 | aborted_by_guard | 0 (0 ok, 0 not concurrent, 0 skewed) | None | None | None | None | None | None | None |
