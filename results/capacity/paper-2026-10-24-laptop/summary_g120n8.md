| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| g120n8-g14A-1 | gpt-oss-120b-mxfp4 | 0 | ok | 40.082 | [] | None | None | None | [6, 7] | 593.7 |
| g120n8-g14P2-1 | gpt-oss-120b-mxfp4 | 2 | ok | 122.199 | [] | None | None | None | [6, 8] | 437.3 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g120n8-g14A-1 | gpt-oss-120b-mxfp4 | 8 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 12.485 | 644.8 | 0.042 | 447.3 | 56.0 | 0.0 | 0.999 |
| g120n8-g14P2-1 | gpt-oss-120b-mxfp4 | 8 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 16.234 | 495.9 | 0.027 | 75.9 | 9.4 | 0.0 | 0.999 |
