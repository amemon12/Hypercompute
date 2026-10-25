| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| mav-n1-gxA-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 0 | ok | 198.768 | [9.186, 9.647, 10.523] | 9.647 | 44.511 | 55.6 | [] | 250.0 |
| mav-n4-gxA-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 0 | ok | 210.359 | [] | None | None | None | [] | 285.1 |
| mav-n8-gxA-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 0 | ok | 214.359 | [] | None | None | None | [12, 23, 5] | 316.8 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mav-n1-gxA-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 1 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 9.647 | 103.7 | 0.0 | 55.6 | 55.6 | 0.0 | 1.0 |
| mav-n4-gxA-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 4 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 30.292 | 132.6 | 0.167 | 62.9 | 15.7 | 2.0 | 0.984 |
| mav-n8-gxA-1 | llama-4-maverick-17b-128e-instruct-ud-q2_k_xl | 8 | ok | 3 (3 ok, 0 not concurrent, 0 skewed) | 44.443 | 180.4 | 0.011 | 86.6 | 10.8 | 0.0 | 0.999 |
