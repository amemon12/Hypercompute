| label | model | Macs | status | load s | decode tok/s (reps) | decode median | prefill median | SSD MiB/token | RTT ms | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 235b-hs-local-w | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 0 | ok | 46.392 | [1.66, 1.418, 1.637] | 1.637 | 1.479 | 466.1 | [51, 112, 67] | 485.8 |
| 235b-hs-D-1-refused | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 3 | refused | None | [] | None | None | None | [] | 1.3 |
| 235b-hs-D-1 | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 3 | refused | None | [] | None | None | None | [] | 1.1 |
| 235b-hs-C1-1-refused | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 3 | refused | None | [] | None | None | None | [] | 1.1 |
| 235b-hs-C1-1 | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 3 | refused | None | [] | None | None | None | [] | 1.0 |
| 235b-hs-local-1 | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 0 | ok | 48.102 | [1.304, 1.418, 1.232] | 1.304 | 1.543 | 589.3 | [] | 515.7 |
| 235b-hs-C1-2-refused | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 3 | refused | None | [] | None | None | None | [] | 1.4 |
| 235b-hs-C1-2 | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 3 | refused | None | [] | None | None | None | [] | 1.2 |
| 235b-hs-D-2-refused | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 3 | refused | None | [] | None | None | None | [] | 1.0 |
| 235b-hs-D-2 | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 3 | refused | None | [] | None | None | None | [] | 1.0 |
| 235b-hs-local-2 | qwen3-235b-a22b-instruct-2507-ud-q2_k_xl | 0 | ok | 42.437 | [1.365, 1.2, 1.236] | 1.236 | 1.44 | 681.6 | [] | 508.5 |

Errors
- 235b-hs-D-1-refused: the gateway rejected --layout: HTTP 422 layout_node: 4c2991743c154be896d10a0b53ba4d2b is not an eligible peer of this request (joined, verified, within node_ids, and a strategy other than local_only)
- 235b-hs-D-1: the gateway rejected --layout: HTTP 422 layout_node: 4c2991743c154be896d10a0b53ba4d2b is not an eligible peer of this request (joined, verified, within node_ids, and a strategy other than local_only)
- 235b-hs-C1-1-refused: the gateway rejected --layout: HTTP 422 layout_node: 4c2991743c154be896d10a0b53ba4d2b is not an eligible peer of this request (joined, verified, within node_ids, and a strategy other than local_only)
- 235b-hs-C1-1: the gateway rejected --layout: HTTP 422 layout_node: 4c2991743c154be896d10a0b53ba4d2b is not an eligible peer of this request (joined, verified, within node_ids, and a strategy other than local_only)
- 235b-hs-C1-2-refused: the gateway rejected --layout: HTTP 422 layout_node: 4c2991743c154be896d10a0b53ba4d2b is not an eligible peer of this request (joined, verified, within node_ids, and a strategy other than local_only)
- 235b-hs-C1-2: the gateway rejected --layout: HTTP 422 layout_node: 4c2991743c154be896d10a0b53ba4d2b is not an eligible peer of this request (joined, verified, within node_ids, and a strategy other than local_only)
- 235b-hs-D-2-refused: the gateway rejected --layout: HTTP 422 layout_node: 4c2991743c154be896d10a0b53ba4d2b is not an eligible peer of this request (joined, verified, within node_ids, and a strategy other than local_only)
- 235b-hs-D-2: the gateway rejected --layout: HTTP 422 layout_node: 4c2991743c154be896d10a0b53ba4d2b is not an eligible peer of this request (joined, verified, within node_ids, and a strategy other than local_only)
