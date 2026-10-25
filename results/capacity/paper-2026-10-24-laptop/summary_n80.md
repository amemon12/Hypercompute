| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| n80-g14A-1 | qwen3-next-80b-a3b-instruct-q3_k_m | 0 | ok | 22.043 | [14.75, 24.415, 18.91] | 18.91 | 78.966 | 6.0 | [6, 7] | 76.2 |
| n80-g14P2-1 | qwen3-next-80b-a3b-instruct-q3_k_m | 2 | ok | 180.376 | [8.915, 10.21, 10.89] | 10.21 | 29.744 | 5.1 | [6, 6] | 243.1 |
| n80-g14A-2 | qwen3-next-80b-a3b-instruct-q3_k_m | 0 | ok | 18.031 | [15.526, 20.169, 34.467] | 20.169 | 84.834 | 14.2 | [6, 7] | 55.8 |
| n80-g14P2-2-refused | qwen3-next-80b-a3b-instruct-q3_k_m | 2 | refused | None | [] | None | None | None | [6, 5] | 1.8 |
| n80-g14P2-2 | qwen3-next-80b-a3b-instruct-q3_k_m | 2 | refused | None | [] | None | None | None | [7, 6] | 1.6 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n80-g14A-1 | qwen3-next-80b-a3b-instruct-q3_k_m | 1 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 18.91 | 52.9 | 0.0 | 6.0 | 6.0 | 0.0 | 1.0 |
| n80-g14P2-1 | qwen3-next-80b-a3b-instruct-q3_k_m | 1 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 10.21 | 97.9 | 0.0 | 5.1 | 5.1 | 0.0 | 1.0 |
| n80-g14A-2 | qwen3-next-80b-a3b-instruct-q3_k_m | 1 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 20.169 | 49.6 | 0.0 | 14.2 | 14.2 | 0.0 | 1.0 |
| n80-g14P2-2-refused | qwen3-next-80b-a3b-instruct-q3_k_m | 1 | refused | 0 (0 ok, 0 not concurrent, 0 skewed) | None | None | None | None | None | None | None |
| n80-g14P2-2 | qwen3-next-80b-a3b-instruct-q3_k_m | 1 | refused | 0 (0 ok, 0 not concurrent, 0 skewed) | None | None | None | None | None | None | None |

Errors
- n80-g14P2-2-refused: the gateway rejected --layout: HTTP 422 layout_device: mac-a (6dad6cf7) has no verified RPC device RPC0; its worker runs -d MTL0
- n80-g14P2-2: the gateway rejected --layout: HTTP 422 layout_device: mac-a (6dad6cf7) has no verified RPC device RPC0; its worker runs -d MTL0
