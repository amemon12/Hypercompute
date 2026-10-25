# Image input, live on the GX10 (2026-10-25 02:50 to 03:00 UTC)

feat/vlm-mmproj at 0b25813 on the ASUS GX10 (NVIDIA GB10, 121 GiB unified memory), alone, no peers. llama.cpp
b11770 (39da130fc) source build with libmtmd. The host ran from a separate worktree (~/hc-vlm) with
HYPERCOMPUTE_CONFIG_DIR pointing at its catalog; the original host was restored afterwards.

## Steps and results

| Step | Result |
| --- | --- |
| `models pull gemma-3-27b-it-q4_k_m` | model (16546404992 B) and projector (857739168 B) downloaded and SHA-verified in about 3 min, each with its marker |
| `vision_check.py --preflight` | `PASS catalog` |
| `run_big.py --strategy local_only --keep` | plan tier gpu, 62 layers on CUDA0, `engine_options.mmproj: true`, reason "~1.8 GiB of CUDA0 held back for the vision projector"; ready in 4.1 s (weights in page cache) |
| `vision_check.py --home ~/.hypercompute` | 5 of 5 PASS (below) |
| known-answer images | both correct (below) |

```
PASS deployment  gemma-3-27b-it-q4_k_m on this machine + 0 peer(s)
PASS projector   -mmdev CUDA0 (local), peers none
PASS log         CLIP using CUDA0 backend; loaded multimodal model; compute 121 MiB on CUDA0 against the 1024 MiB prior
PASS image       The screenshot shows the Hypercompute interface, a platform for distributed computing, ...
PASS text        ready
```

Known answers, through `hypercompute ask-image`:

- A generated 512x512 PNG (red circle, blue square in the top-left corner): `circle=red, square=blue,
  corner=top left`. Correct.
- An archived dashboard overview (capture omitted for privacy), asked for the product name and the GPU of "this machine":
  `product=Hypercompute, gpu=NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Correct.

For new checks, `vision_check.py` defaults to `docs/screenshots/03-choose-model.png`. The results below describe
the original validation image, not the replacement default.

## Measurements

3 reps each, cold prompt (`cache_prompt: false`), 128 output tokens (`ignore_eos`), temperature 0, same deployment:

| Request | prompt tokens | prompt time | decode | wall |
| --- | --- | --- | --- | --- |
| screenshot + "Describe this screenshot in detail." | 273 | 1051 to 1057 ms | 12.08 to 12.09 tok/s | 11.6 s |
| text only, similar question | 21 | 268 to 271 ms | 12.15 to 12.17 tok/s | 10.8 s |

`run_big` text reps (128 prompt, 64 output): decode 12.11 to 12.12 tok/s, prefill 394 to 403 tok/s.

- An image costs about 0.8 s more prompt time (256 image tokens for Gemma 3, plus the encoder); decode is
  unchanged within 1 percent.
- The projector ran on CUDA0 (`CLIP using CUDA0 backend`); gemma3 projector, image size 896, warmup at 896x896.
- Projector compute buffer: 121.25 MiB on CUDA0 and 9.19 MiB on the CPU, against the planner's 1024 MiB prior
  (PROJECTOR_COMPUTE_EST). Gemma 3 always resizes to 896x896, so its buffer is fixed; Qwen2.5-VL takes larger
  images and was not measured, so the prior stays.
- Memory: the deployment's llama-server held 18121 MiB (model, KV, projector and buffers). The planner estimated
  17.96 GiB for the layers plus 1.8 GiB reserved for the projector, 19.76 GiB in all, against 17.70 GiB used.
- The rental's Nemotron llama-server (snap, root, port 8346) held 189 MiB of GPU memory throughout, asleep.

## Not covered

The pooled case (projector on the host, layers on RPC peers) is covered by unit tests only. A run of it is in
the G14 handover (docs/VISION.md, Live check).
