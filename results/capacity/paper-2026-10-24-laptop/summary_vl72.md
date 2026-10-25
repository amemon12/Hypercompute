| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| vl72-g14A-1 | qwen2.5-vl-72b-instruct-q4_k_m | 0 | ok | 34.08 | [0.141, 0.141] | 0.141 | 4.642 | 26896.3 | [7, 6] | 1224.5 |
| vl72-g14P2-1 | qwen2.5-vl-72b-instruct-q4_k_m | 2 | ok | 100.16 | [0.367, 0.364] | 0.365 | 7.221 | 8956.0 | [6, 6] | 1426.8 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vl72-g14A-1 | qwen2.5-vl-72b-instruct-q4_k_m | 1 | ok | 2 (2 ok, 0 not concurrent, 0 skewed) | 0.141 | 7093.4 | 0.0 | 26896.3 | 26896.3 | 0.0 | 1.0 |
| vl72-g14P2-1 | qwen2.5-vl-72b-instruct-q4_k_m | 1 | ok | 2 (2 ok, 0 not concurrent, 0 skewed) | 0.365 | 2736.5 | 0.0 | 8956.0 | 8956.0 | 0.0 | 1.0 |
