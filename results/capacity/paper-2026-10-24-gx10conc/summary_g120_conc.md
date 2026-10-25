| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| g120-n1-gxA-1 | gpt-oss-120b-mxfp4 | 0 | ok | 56.099 | [53.293, 54.213, 54.847] | 54.213 | 362.996 | None | [] | 75.4 |
| g120-n4-gxA-1 | gpt-oss-120b-mxfp4 | 0 | ok | 60.112 | [] | None | None | None | [] | 85.7 |
| g120-n8-gxA-1 | gpt-oss-120b-mxfp4 | 0 | ok | 56.106 | [] | None | None | None | [] | 87.6 |
| g120-n8-gxA-2 | gpt-oss-120b-mxfp4 | 0 | ok | 56.104 | [] | None | None | None | [] | 87.4 |
| g120-n4-gxA-2 | gpt-oss-120b-mxfp4 | 0 | ok | 58.107 | [] | None | None | None | [] | 84.4 |
| g120-n1-gxA-2 | gpt-oss-120b-mxfp4 | 0 | ok | 60.114 | [55.158, 55.245, 54.97] | 55.158 | 367.333 | None | [] | 78.4 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g120-n1-gxA-1 | gpt-oss-120b-mxfp4 | 1 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 54.213 | 18.4 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| g120-n4-gxA-1 | gpt-oss-120b-mxfp4 | 4 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 109.572 | 36.5 | 0.029 | 0.0 | 0.0 | 2.0 | 0.986 |
| g120-n8-gxA-1 | gpt-oss-120b-mxfp4 | 8 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 142.495 | 56.4 | 0.04 | 0.0 | 0.0 | 5.0 | 0.989 |
| g120-n8-gxA-2 | gpt-oss-120b-mxfp4 | 8 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 150.234 | 53.6 | 0.04 | 0.0 | 0.0 | 5.0 | 0.988 |
| g120-n4-gxA-2 | gpt-oss-120b-mxfp4 | 4 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 111.446 | 35.7 | 0.029 | 0.0 | 0.0 | 1.0 | 0.987 |
| g120-n1-gxA-2 | gpt-oss-120b-mxfp4 | 1 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 55.158 | 18.1 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
