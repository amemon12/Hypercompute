| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| d70-g14A-1 | llama-3.3-70b-instruct-q4_k_m | 0 | ok | 42.252 | [0.165, 0.165] | 0.165 | 6.361 | 23454.2 | [6, 6] | 1097.2 |
| d70-g14P2-1 | llama-3.3-70b-instruct-q4_k_m | 2 | ok | 96.157 | [0.901, 0.895] | 0.898 | 12.514 | 2761.2 | [6, 6] | 1162.5 |
| d70-g14A-2 | llama-3.3-70b-instruct-q4_k_m | 0 | aborted_by_guard | 38.079 | [] | None | None | None | [9, 12] | 605.6 |
| d70-g14P2-2-refused | llama-3.3-70b-instruct-q4_k_m | 2 | refused | None | [] | None | None | None | [7, 8] | 1.9 |
| d70-g14P2-2 | llama-3.3-70b-instruct-q4_k_m | 2 | refused | None | [] | None | None | None | [5, 6] | 1.5 |

Batches (N=1: each rep as a batch of one)

| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | missed first iteration (median) | window coverage (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d70-g14A-1 | llama-3.3-70b-instruct-q4_k_m | 1 | ok | 2 (2 ok, 0 not concurrent, 0 skewed) | 0.165 | 6061.0 | 0.0 | 23454.2 | 23454.2 | 0.0 | 1.0 |
| d70-g14P2-1 | llama-3.3-70b-instruct-q4_k_m | 1 | ok | 2 (2 ok, 0 not concurrent, 0 skewed) | 0.898 | 1113.4 | 0.0 | 2761.2 | 2761.2 | 0.0 | 1.0 |
| d70-g14A-2 | llama-3.3-70b-instruct-q4_k_m | 1 | aborted_by_guard | 1 (0 ok, 0 not concurrent, 0 skewed) | None | None | None | None | None | None | None |
| d70-g14P2-2-refused | llama-3.3-70b-instruct-q4_k_m | 1 | refused | 0 (0 ok, 0 not concurrent, 0 skewed) | None | None | None | None | None | None | None |
| d70-g14P2-2 | llama-3.3-70b-instruct-q4_k_m | 1 | refused | 0 (0 ok, 0 not concurrent, 0 skewed) | None | None | None | None | None | None | None |

Errors
- d70-g14P2-2-refused: the gateway rejected --layout: HTTP 422 layout_order: only the last stage is on this machine: peers come first and this machine's GPU last, as llama.cpp orders them. The host CPU is no stage: an MoE model keeps the routed experts its GPU cannot hold there, as the moe_offload tier does
- d70-g14P2-2: the gateway rejected --layout: HTTP 422 layout_order: only the last stage is on this machine: peers come first and this machine's GPU last, as llama.cpp orders them. The host CPU is no stage: an MoE model keeps the routed experts its GPU cannot hold there, as the moe_offload tier does
