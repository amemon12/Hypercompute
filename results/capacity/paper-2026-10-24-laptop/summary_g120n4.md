| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| g120n4-g14A-1 | gpt-oss-120b-mxfp4 | 0 | ok | 44.105 | [] | None | None | None | [32, 31] | 517.2 |
| g120n4-g14P2-1 | gpt-oss-120b-mxfp4 | 2 | ok | 128.205 | [] | None | None | None | [6, 7] | 445.2 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g120n4-g14A-1 | gpt-oss-120b-mxfp4 | 4 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 8.08 | 495.1 | 0.023 | 358.8 | 89.7 | 0.0 | 0.999 |
| g120n4-g14P2-1 | gpt-oss-120b-mxfp4 | 4 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 11.514 | 346.0 | 0.018 | 59.7 | 15.0 | 0.0 | 0.999 |
