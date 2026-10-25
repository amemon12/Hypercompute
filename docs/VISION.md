# Image input

A pooled vision-language model can answer questions about an image. The language model splits across peers over
llama.cpp RPC as usual; the vision projector (the `mmproj` GGUF that turns an image into tokens) runs on the
coordinator only.

Status: run live on the GX10 alone with Gemma 3 27B (all checks pass, decode unchanged with an image; see
[VISION-GX10-LIVE.md](VISION-GX10-LIVE.md)). The pooled case is unit-tested, not yet run live.

## How it works

1. **Catalog.** A GGUF entry may list an `mmproj` file (name, url, sha256, size_bytes) at the entry's own revision.
   `qwen2.5-vl-72b-instruct-q4_k_m` (unsloth `mmproj-F16.gguf`, 1.41 GB) and `gemma-3-27b-it-q4_k_m` (bartowski
   `mmproj-google_gemma-3-27b-it-f16.gguf`, 0.86 GB) have one; both were checked against the HF tree API.
2. **Download.** `uv run hypercompute models pull <model>` (or the dashboard) downloads the projector with the model
   and checks its SHA-256 and size. It has its own marker, `.verified-mmproj.json`, so a model downloaded before
   this change stays usable for text, and pulling it again fetches only the projector. Peers never receive it.
3. **Engine.** With a verified projector, llama-server starts with `--mmproj <file> -mmdev <device>`: this
   machine's last GPU in the plan (the one holding the output head), or `-mmdev none` (host CPU) when the plan has
   no local GPU. `-mmdev` is required: at b11770 the projector otherwise follows the first `--device` entry
   (`common/arg.cpp`), and Hypercompute lists the RPC peers first, so it would land on a peer.
4. **Memory.** llama-server runs with `--fit off`, so the planner takes the projector file plus a 1 GiB prior for
   the vision encoder's compute buffer off that device's budget before splitting layers. A plan that never saw the
   reservation (capacity optimizer, measured profile) gets a warning when it leaves less room than that. The real
   compute buffer is in the llama-server log as `compute buffer size`, after the `clip_model_loader` lines.
5. **Gateway.** `/v1/chat/completions` accepts OpenAI-style `image_url` parts in user messages, checks them, and
   forwards the body unchanged. A text-only request is forwarded exactly as before. If the ready model was started
   without a projector, an image request gets 400 `vision_unavailable`.

```json
{"messages": [{"role": "user", "content": [
  {"type": "text", "text": "What is on the table?"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}]}],
 "max_tokens": 256}
```

From the host itself:

```bash
uv run hypercompute ask-image photo.jpg "What is on the table?"
```

It prints the answer, the model, llama-server's prompt and decode timings (the prompt tokens include the image's
tokens) and the wall time of the request.

## Live check

`scripts/vision_check.py` checks a running gateway without planning, deploying or stopping anything. The gateway
reads its catalog at startup from `HYPERCOMPUTE_CONFIG_DIR` (default: the checkout's `configs/`), so first confirm
it lists the projector:

```bash
uv run --no-sync python scripts/vision_check.py --port 8001 --preflight gemma-3-27b-it-q4_k_m
```

Plan and deploy after that, then check the ready deployment: the projector's device in the engine argv and in
llama-server's log, one image question and one text chat. It prints one PASS or FAIL line per check:

```bash
uv run --no-sync python scripts/vision_check.py --port 8001 --home C:/Users/operator/.hypercompute/iroh-session
```

## Limits

- PNG, JPEG or GIF only: what llama.cpp's stb_image decodes. WebP needs an ffmpeg the hosts do not ship.
- Base64 data URLs only. The gateway refuses `http(s)` URLs (llama-server would fetch them from the host),
  `file://` paths, raw base64, and audio or video parts.
- At most 4 images per request, 8 MiB and 64 megapixels each (the size is read from the image header, because
  llama.cpp decodes the full image before resizing it), all within the 16 MiB chat body limit: about 12 MiB of
  images in total. A chat client that resends earlier images with every turn reaches the 4-image cap.
- Only this machine (loopback) gets the 16 MiB body limit; other clients keep 4 MiB, since the gateway buffers a
  body before it checks the owner token and only this machine may chat.
- llama.cpp engine only. vLLM, EXO and AirLLM deployments answer `vision_unavailable`.
- The stream tier (hcstream) runs without the projector, because hcstream bounds host RAM and does not count it.
  The plan says so.
- Each image costs prompt tokens (Gemma 3: 256 per image; Qwen2.5-VL: more for larger images), so leave room in
  `ctx_size`. With a projector loaded, llama-server turns off context shift and cache reuse. A model that needs
  non-causal attention over the image caps its image tokens at the ubatch size (`-ub`, 512 by default) and logs a
  warning.
- The 1 GiB compute-buffer prior is an estimate until a live load measures it.
- The plan decides whether the projector loads (`engine_options.mmproj`), so a plan made before the projector
  was downloaded deploys text-only; plan again after `models pull`. Tune trials and replans, which do not go
  through that path, load a downloaded projector without reserving room for it.
- The reservation comes off this machine's last GPU in device order; on a host with several GPUs whose plan skips
  that one, the projector's real device only gets the warning.
- Text-only benchmarks of a VLM change once its projector is downloaded: the deployment reserves memory for it and
  loads it, so the layout can differ. To keep text-only numbers comparable, measure before pulling the projector,
  or delete the projector file and `.verified-mmproj.json` from the model's directory.
