"""Mock pool shaped like the real one on 2026-10-24 (gateway records in the main repo's results/capacity/).

Machine names, contribution budgets and layer counts come from the recorded Kimi-K2.6 pooled run. Mock speeds:
Kimi and DeepSeek use the R6 pooled medians, gpt-oss 120B the R6 GX10-alone median, and the 0.5B numbers are
placeholders. Every answer the mock returns starts with [mock].
"""

from __future__ import annotations

GIB = 1024**3

HOST = {"id": "local", "display_name": "gx10-49d1 (this machine)", "device": "CUDA0", "max_bytes": 84886034022,
        "pool": {"id": "uma", "kind": "unified", "total_bytes": 130593898496}}

PEERS = [
    {"id": "6dad6cf75b2e4cf197b7bc7f4d0a67e4", "display_name": "mac-a", "device": "MTL0", "max_bytes": 12348030976,
     "pool": {"id": "uma", "kind": "unified", "total_bytes": 17179869184}, "rtt_ms": 17},
    {"id": "d4e65f83c4be4a41aad722e40983b08b", "display_name": "mac-b", "device": "MTL0", "max_bytes": 12348030976,
     "pool": {"id": "uma", "kind": "unified", "total_bytes": 17179869184}, "rtt_ms": 17},
    {"id": "4c2991743c154be896d10a0b53ba4d2b", "display_name": "mac-azim", "device": "MTL0", "max_bytes": 8053063680,
     "pool": {"id": "uma", "kind": "unified", "total_bytes": 17179869184}, "rtt_ms": 84,
     "mode": "relay"},  # mac-azim stayed on the Iroh relay in every R6 run
    {"id": "8f169e3549fc41fbb14f6876a19d704c", "display_name": "g14", "device": "CUDA0", "max_bytes": 9663676416,
     "pool": {"id": "vram:CUDA0", "kind": "discrete_vram", "total_bytes": 12884901888}, "rtt_ms": 17},
]

# id, display name, family, params_b, quantization, bytes, n_layer, moe, mock decode tok/s, mock prefill tok/s, role
MODELS = [
    ("kimi-k2.6-iq1_s", "Kimi K2.6", "kimi", 1000.0, "IQ1_S", 207404948128, 61, True, 1.54, 3.30, "capacity_candidate"),
    ("deepseek-v3.1-ud-tq1_0", "DeepSeek V3.1", "deepseek", 671.0, "UD-TQ1_0", 170499768256, 61, True, 1.50, 6.36,
     "capacity_candidate"),
    ("gpt-oss-120b-mxfp4", "gpt-oss 120B", "gpt-oss", 120.0, "MXFP4", 63387346208, 36, True, 56.0, 382.0, "standard"),
    ("qwen2.5-0.5b-instruct-q4_k_m", "Qwen2.5 0.5B Instruct", "qwen", 0.5, "Q4_K_M", 491400032, 24, False, 210.0,
     2400.0, "plumbing"),
]
