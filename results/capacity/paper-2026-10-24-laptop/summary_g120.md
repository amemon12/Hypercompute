| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| g120-g14A-1 | gpt-oss-120b-mxfp4 | 0 | ok | 40.085 | [3.614, 3.882, 3.99] | 3.882 | 3.921 | 251.4 | [6, 7] | 351.5 |
| g120-g14P2-1 | gpt-oss-120b-mxfp4 | 2 | ok | 112.175 | [7.586, 9.071, 9.562] | 9.071 | 28.643 | 7.5 | [5, 6] | 968.4 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g120-g14A-1 | gpt-oss-120b-mxfp4 | 1 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 3.882 | 257.6 | 0.0 | 251.4 | 251.4 | 0.0 | 1.0 |
| g120-g14P2-1 | gpt-oss-120b-mxfp4 | 1 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 9.071 | 110.2 | 0.0 | 7.5 | 7.5 | 0.0 | 1.0 |
