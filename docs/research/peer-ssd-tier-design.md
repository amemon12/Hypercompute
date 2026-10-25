# Peer SSD tier: every machine pages its own share from its own SSD

Design, 2026-10-22, branch `codex/iroh-transport`. Nothing in this document has been built or run on a Mac. It turns
three research reports (Task A: RPC devices and host routing; Task B: macOS feasibility, partly done; Task C: placement
and cost model) plus `CHECKPOINT-2026-10-22.md`, `hybrid-capacity-implementation.md` and `hybrid-llama-audit.md` into
something we can build in stages.

Tags used throughout:

- **[V]** verified in source or repo at the cited line, by this document or by the Task A/B reports, or observed by
  one of the read-only commands listed in section 13.
- **[M]** measured earlier on this hardware; the evidence file is named.
- **[E]** estimate from the Task C cost model (`model.py`, path in section 13). Not a measurement.
- **[A]** assumption. Each one is listed in section 12 with the stage that checks it.

Source aliases (all read-only research copies, never executed):

- `RPC/` = `C:\Users\operator\AppData\Local\Temp\claude\C--\9e2c80d5-50ca-460d-8443-8db45359fcde\scratchpad\rpc`
  (b11770 `ggml-rpc.cpp`, `rpc-server.cpp`, `log.h`, `ext/`; Task A checked by MD5 that it matches the b11770 tree)
- `LL/` = `...\scratchpad\llsrc` (b11770 llama and ggml sources)
- `MR/` = `...\scratchpad\peerstream\mac-release\x` (macOS arm64 b11770 release, extracted as data; the archive's
  sha256 matches `configs/engines.lock.json:82-84`)
- Repo paths are relative to `C:\Users\operator\.hypercompute\worktrees\iroh-transport`.

## 0. Recommendation

**Build it, in stages, starting tonight. Only the first three stages fit tonight, and the demo that gains most is the
Macs-only one, not the G14-hosted one.**

- **Mechanism.** A server-only patch to `ggml-rpc.cpp` ("peerstream"), built from the pinned b11770 source on a Mac.
  It backs the peer's CPU device with read-only file mappings of the RPC cache files instead of private memory. The
  wire protocol does not change and the host keeps the stock release.
- **Per peer, for MoE models:** expose the CPU device only (`-d CPU`), give it whole layers, keep them file-backed.
  The Mac's free memory then works as page cache for its own shard, and the owner's apps can take it back. Metal
  stays resident as today and is used for dense models.
- **Seeding:** each peer downloads its own shard from Hugging Face by byte ranges, in parallel, before the run. The
  patched server verifies and names each file by the same FNV-1a hash the host computes at load, so a bad download can
  only cause an upload, never wrong weights.
- **Go tonight:**
  - S0: measure the per-peer hop.
  - S1: the patch plus a 7B gate on one 200 GB Mac.
  - S2: HF range seeding plus one gpt-oss-120b run, G14 host with one file-backed Mac.
- **No-go tonight:**
  - Planner-driven 4-device runs with published numbers.
  - Models bigger than gpt-oss-120b. They need a 200 GB Mac as host, and the Mac-host path has never run
    (`docs/research/mac-host-runbook.md` section 9 says so).
  - Pooled disk, where the host does not hold the whole GGUF.
  - A Windows peer patch.
  - Dense parts on Metal with the experts on the CPU inside one peer.
- **Why this order:**
  - **G14 host.** One Mac memory-only was measured at 4.52 tok/s [M], against 4.05 tok/s for the G14 alone [M]. With
    three file-backed Macs the estimate is 4.1 tok/s at today's 60 ms per-peer cost and 9.5 tok/s at 13 ms [E]. So the
    tier helps a G14-hosted run only after the hop cost is found.
  - **Macs only.** Three Macs memory-only are estimated at 0.9-1.0 tok/s. With the peer tier they reach 2.3-3.0
    tok/s [E]. This is where "join your friend's laptop and run something none of you can run alone" actually holds.

## 1. What the tier adds, and what it cannot

What pools across machines:

- **Disk capacity.** Each peer's SSD holds its shard.
- **Page cache.** Each peer's free RAM caches the hot part of its shard.
- **Compute.**

What does not pool: **SSD bandwidth for one chat.** llama.cpp runs the stages of one token one after another
(`LL/ggml_src_ggml-backend.cpp:1865-1900`, Task C) [V], so per-token fault I/O is a sum over stages, not a maximum.
Overlapping I/O across stages would need new engine work, and this design does not include it.

What the host already does [M]:

- On Windows it streams dense CPU layers (hcstream).
- It keeps routed experts file-backed (`moe_offload`, `services/moe_tier.py`).

This design brings the same property to the peers. Today a peer only takes what fits in the memory it lends.

## 2. Facts the design rests on

| # | Fact | Where |
| --- | --- | --- |
| F1 | The server allocates every buffer from the device's default buffer type. | `RPC/ggml-rpc.cpp:1502-1522` (`:1507-1508`) [V] |
| F2 | The default CPU buffer is `ggml_aligned_malloc`, which is anonymous `vm_allocate` on macOS. | `LL/ggml_src_ggml-backend.cpp:2561-2570`; `RPC/ext/ggml.c:343-362` [V] |
| F3 | A Metal shared buffer is host memory wrapped by `newBufferWithBytesNoCopy`. Its base is that CPU address, but its buffer type reports `is_host=false`. Residency sets keep it wired. | `RPC/ext/ggml_src_ggml-metal_ggml-metal-device.m:2128-2156`, `:2297-2298`; `RPC/ext/ggml_src_ggml-metal_ggml-metal.cpp:289-293` [V]; residency at `metal-device.m:1005-1017`, `:2061-2065` (Task B) [V] |
| F4 | The client asks `SET_TENSOR_HASH` only for weight buffers over 10 MiB. The hash is FNV-1a 64 over exactly the bytes being set, and the request is synchronous. | `RPC/ggml-rpc.cpp:117-118`, `:313-321`, `:844-874`, `:1148-1174` [V] |
| F5 | On a miss the client sends `SET_TENSOR` with `cache_flag=1`. With `-c` the server writes `cache_dir/%016llx` without checking the write, then copies the data. | `RPC/ggml-rpc.cpp:1742-1752` [V] |
| F6 | On a hit the server reads the whole file into a vector, takes the size from the file, and copies it. | `RPC/ggml-rpc.cpp:1818-1835`, `:1855-1880` [V] |
| F7 | When a handler returns false, the server closes the connection, so failures are already fail-closed. | `RPC/ggml-rpc.cpp:2600-2608` [V] |
| F8 | `GET_ALIGNMENT` is answered by the server. The client pads every tensor to that alignment (the RPC buffer type has no `alloc_buffer_n`) and aligns offsets against the real remote base. | `RPC/ggml-rpc.cpp:1524-1534`, `:1125-1128`; `LL/ggml_src_ggml-backend.cpp:57-91` (pad at `:71`); `LL/ggml_src_ggml-alloc.c:53-75` [V] |
| F9 | llama allocates RPC weight buffers through `ggml_backend_alloc_ctx_tensors_from_buft` and tags them WEIGHTS. It uploads each tensor once, at offset 0, with its full size. | `LL/src_llama-model.cpp:1959`, `:1980`; `LL/src_llama-model-loader.cpp:1735`, `:1807` (Task A) [V] |
| F10 | Buffer clears happen only on the output and KV buffers. | `LL/src_llama-context.cpp:2317`; `LL/src_llama-kv-cache.cpp:293`, `:399` [V] |
| F11 | `ggml_backend_cpu_buffer_from_ptr` wraps a pointer as `CPU_Mapped` and never frees it. | `LL/ggml_src_ggml-backend.cpp:2537-2538`, `:2628-2630` [V] |
| F12 | Copies between two devices on the same server go through a synchronous `COPY_TENSOR`. Between Metal shared and CPU it returns false, and the client then goes through the host. | `RPC/ggml-rpc.cpp:930-951`, `:2009-2048` [V] |
| F13 | rpc-server's only flags are `-H -p -t -d -c`. Threads default to `hardware_concurrency()/2`. The default devices exclude CPU and ACCEL. The cache directory is `$LLAMA_CACHE/rpc`. | `RPC/rpc-server.cpp:94-112`, `:190-208`, `:64-82`, `:240-251` [V] |
| F14 | `GGML_RPC_DEBUG` 1 logs connections and allocations, 2 logs every command, 3 logs message detail. | `RPC/log.h:11-49` [V] |
| F15 | The Mac release `ggml-rpc-server` names `@rpath/libggml-rpc.0.dylib` and `@loader_path`, and links `/usr/lib/librdma.dylib`. Every binary checked is ad-hoc and linker-signed (CodeDirectory flags `0x20002`), with no hardened runtime and no library validation. | string and CodeDirectory inspection of `MR/` (section 13) [V] |
| F16 | The repo already builds the pinned commit on macOS with Metal and RPC, and can point the agent at a source build. | `scripts/build_llama_macos.sh:168-184`; `backend/hypercompute/engine.py:34`, `:182-185` (`HYPERCOMPUTE_LLAMA_BIN_DIR`) [V] |
| F17 | The agent starts the worker with `-H -p -d`. With `--cache` it adds `-c` and `LLAMA_CACHE=<HOME>/rpc-cache`. It passes no `-t`. | `backend/hypercompute/agent/join.py:193-197` [V] |
| F18 | Disk free is measured once at join (`<HOME>` volume) and heartbeats do not refresh it. Only the host gets a storage pool. | `agent/inventory.py:216`, `:227-240`; `capacity/pools.py:203-210` [V] |
| F19 | `build_args` emits the moe `-ot` before `--rpc`. | `adapters/llama_rpc.py:154`, `:160` [V] |
| F20 | The pinned gpt-oss file on Hugging Face supports byte ranges: a 302 to a CDN, then `206` with `content-range: bytes 0-15/63387346208`. | HEAD request, section 13 [V] |
| F21 | gpt-oss-120b MXFP4 per layer: 5 tensors over 10 MiB totalling 1,717,125,120 B (`attn_q`, `attn_output`, `ffn_{gate,up,down}_exps`) and 14 small ones totalling 9,087,488 B. All 36 layers: 61.82 GB over the threshold, 0.327 GB small. The header ends at 13,022,240 B. Every large tensor is a multiple of 16 KiB. Qwen2.5-7B Q4_K_M layer 0: 3 large (132.1 MB), 9 small (17.0 MB), and some large tensors are not 16 KiB multiples. | `gguf.read_tensor_table` on the local files (section 13) [V] |

## 3. The peer mechanism (peerstream)

### 3.1 Why a patched server built from source

| Option | Verdict |
| --- | --- |
| Interposer (`DYLD_INSERT_LIBRARIES`, wrapper dylib) | Rejected. The CPU allocation happens inside libggml-base (F2) and cannot be interposed across dylibs. Task B's Mach-O and interposer checks were not finished. |
| Replace only `libggml-rpc.0.dylib` next to the release dylibs | Possible (F15: loaded by `@rpath`, no library validation), but our build would have to match the release's options exactly. The release links `librdma` (F15), and a mismatch is an ABI risk [A]. Kept as a fallback. |
| **Full source build of b11770 with the patch, selected with `HYPERCOMPUTE_LLAMA_BIN_DIR`** | **Chosen.** One toolchain, no mixed binaries, and it uses F16. Cost: Xcode Command Line Tools plus cmake on the build Mac. |

The patch changes only server-side code in `ggml/src/ggml-rpc/ggml-rpc.cpp`. The client code and the message structs
are untouched, so a stock Windows host (RPC proto 8.0.0) talks to a patched Mac. With `HC_PEERSTREAM` unset the server
behaves as stock, except for the two small robustness fixes P6b and P9 below.

### 3.2 Exact launch

MoE peer, the default for this tier:

```sh
env LLAMA_CACHE="$HYPERCOMPUTE_HOME/rpc-cache" \
    HC_PEERSTREAM=1 \
    HC_PEERSTREAM_DEVICES=CPU \
    HC_PEERSTREAM_DISK_FLOOR=16106127360 \
    HC_PEERSTREAM_STAGING="$HYPERCOMPUTE_HOME/rpc-cache/staging" \
  "$HYPERCOMPUTE_LLAMA_BIN_DIR/ggml-rpc-server" -H 127.0.0.1 -p 50052 -d CPU -t 4 -c
```

Notes on the launch:

- `-H 127.0.0.1` is what the agent uses under Iroh (`cli.py:288`) [V].
- The floor is 15 GiB.
- `-t 4` is a starting value for the 4 performance cores. It is not a measurement [A]; S1 measures 4 against 6.

Two-device peer (Metal-resident span, then a file-backed span):

- `-d MTL0,CPU`, plus `HC_PEERSTREAM_UMA_COPY=1`.
- Device order on the peer is span order on the host: MTL0 must come first.

For gates, add `GGML_RPC_DEBUG=1` to get the allocation lines (F14).

The agent change goes in `agent/join.py:193-197`:

- `-d` comes from `contribution.devices`.
- `-t` comes from a new `contribution.threads`.
- The `HC_PEERSTREAM_*` variables come from the new contribution fields (section 5).
- `-c` is forced on whenever any device is file-backed.

### 3.3 Patch specification

Line anchors are in the stock file `RPC/ggml-rpc.cpp`. The patch is about 250 lines [E].

- **P1, configuration** (in `ggml_backend_rpc_start_server`, `:2762-2795`).
  - Read the environment variables:
    - `HC_PEERSTREAM`
    - `HC_PEERSTREAM_DEVICES` (comma list of device names)
    - `HC_PEERSTREAM_DISK_FLOOR` (bytes)
    - `HC_PEERSTREAM_STAGING` (directory)
    - `HC_PEERSTREAM_ADVICE` (`normal`, `random`, `sequential` or `willneed`; default `normal`)
    - `HC_PEERSTREAM_UMA_COPY`
    - `HC_PEERSTREAM_LOG` (`tensor` or `summary`)
  - Resolve each named device to its index in `backends`. Each one must have a host default buffer type
    (`ggml_backend_buft_is_host`, so CPU only).
  - Refuse to start, with an ERROR line and no listening socket, if:
    - a device name is unknown,
    - a named device is not a host device,
    - `cache_dir` is null (no `-c`),
    - or `getpagesize()` is not a power of two.
  - Print a banner line after `local cache`:
    `  peerstream     : CPU (page 16384, floor 15.00 GiB, staging on, uma-copy off, advice normal, v1)`.
    The page size is read at runtime, never hardcoded; 16 KiB on Apple Silicon is [A].
  - Store the settings in one static struct. The server is a single process with one client at a time (Task A).
- **P2, alignment** (`get_alignment`, `:1524-1534`). For a file-backed device, return `max(stock, page)`.
  - Through F8 this places every tensor on a page boundary and pads it to whole pages.
  - It costs less than 16 KiB per tensor, and nothing for gpt-oss (F21).
- **P3, allocation** (`alloc_buffer`, `:1502-1522`). For a file-backed device:
  - `p = mmap(NULL, round_up(size, page), PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, -1, 0)`.
  - Wrap it with `ggml_backend_cpu_buffer_from_ptr(p, size)` (F11).
  - Record `{buffer -> (p, len, device)}` in a new `rpc_server` member map.
  - Untouched pages cost no physical memory. On macOS this needs virtual overcommit [A].
  - Compute and KV buffers on this device take the same path and simply behave like malloc.
  - Other devices are stock, but their device index is also recorded, for P7.
- **P4, free.** In `free_buffer` (`:1560-1578`) and `~rpc_server` (`:2428-2431`), after `ggml_backend_buffer_free`,
  `munmap` the recorded range. `CPU_Mapped` does not free its memory (F11), so without this the regions leak.
- **P5, mapping on a hash hit** (`set_tensor_hash`, `:1838-1883`). Conditions:
  - the buffer is file-backed,
  - `offset == 0`,
  - `tensor->data % page == 0`,
  - `data + round_up(nbytes, page) <= base + size`.

  Steps:
  1. Look up `cache_dir/<hash>`.
  2. If it is absent, look in each immediate subdirectory of the staging root (`<root>/<model-id>/`, section 4) for a
     file named exactly after the tensor whose size is `ggml_nbytes`. FNV-1a it in 8 MiB reads. If the hash matches,
     `rename` it to `cache_dir/<hash>`; the staging root and the cache are on the same volume, so this is a rename and
     not a copy. On a mismatch, log WARN, leave the file where it is, and try the next subdirectory. The hash, not the
     name, decides, so the same tensor name in two models is harmless.
  3. Require `st_size == ggml_nbytes(tensor)`. Any other size is answered as a miss. This fixes the stock
     truncated-file hazard of F6 for these devices.
  4. `mmap(data, round_up(nbytes, page), PROT_READ, MAP_PRIVATE|MAP_FIXED, fd, 0)`, check that the result equals
     `data`, then `close(fd)`.
  5. Apply `madvise` with the chosen advice, and answer `result=1`.

  If the conditions fail (an unaligned or partial set), use the stock read-and-copy and count it as `copied`.
  - `PROT_READ` is deliberate. A write into a weight would crash loudly instead of silently turning into private
    copy-on-write memory. F10 says no such write happens.
  - Mapping past end-of-file inside the last page reads zeros [A]. No page lies wholly past end-of-file.
- **P6, cache write on a miss** (`set_tensor`, `:1698-1754`), for a file-backed device with `cache_flag`, `offset == 0`
  and `size == nbytes`:
  1. Check the disk floor: `statvfs(cache_dir)` must leave `f_bavail*f_frsize - size >= floor`.
  2. Write `<hash>.part.<pid>.<n>` and check the stream state and the byte count.
  3. `rename` it to `<hash>`.
  4. Map it as in P5.

  Any failure returns false, which fails the load closed through F7. If the tensor were silently copied instead, the
  peer would hold weights in private memory the planner never admitted.
  - **P6b**, for stock devices: the same temp-file-and-rename write, and a floor violation skips the cache write but
    still copies. The tensor is in memory either way, so nothing is lost.
- **P7, same-server copy** (`copy_tensor`, `:2009-2048`). Only with `HC_PEERSTREAM_UMA_COPY=1`. If
  `ggml_backend_buffer_copy_tensor` returns false and both buffers belong to this server, each either a host buffer or
  a Metal buffer with a non-null base, then `memcpy(dst->data, src->data, nbytes)` and `result=1`.
  - The function already synchronized every backend at `:2010`.
  - Stock Metal `set_tensor` on shared buffers is itself a `memcpy` (`metal-device.m:2333-2336`, Task A) [V].
  - Unified memory makes the base a CPU address (F3). A buffer created with `GGML_METAL_SHARED_BUFFERS_DISABLE` would
    be private and must be excluded. The agent never sets that variable [A].
  - Effect: an MTL0-to-CPU crossing costs one synchronous round trip instead of two (a failed `COPY_TENSOR`, then
    `GET_TENSOR`, then `SET_TENSOR`; F12).
- **P8, evidence.**
  - Per tensor at INFO when `HC_PEERSTREAM_LOG=tensor`:
    `peerstream: map <name> <MiB> <hash> hit|staged|uploaded`.
  - One summary per load at the first `graph_compute` after any `set_tensor*`:
    `peerstream: CPU mapped <n> tensors <GiB>, copied <n> <MiB>, staged ok <n> bad <n>, uploads <n> <GiB>, unaligned <n>`.
- **P9, start-up hygiene.** Delete stale `^[0-9a-f]{16}\.part\..*` files in `cache_dir`.
- **P10, name safety.** Staging lookups use the tensor name the client sent (`deserialize_tensor` sets it at `:1693`).
  Accept only `[A-Za-z0-9._-]{1,63}` that does not start with `.`. Anything else counts as a miss.

### 3.4 What a peer holds, by kind

| Kind | Where it lives | Counted against |
| --- | --- | --- |
| Large tensors (over 10 MiB) of file-backed layers | Clean file pages from the peer's SSD, dropped under pressure and read again on the next fault [A] | Peer `ssd` pool (disk), plus the page-cache estimate (speed only) |
| Small tensors (10 MiB or less) of file-backed layers, sent whole by the host (F4) | Anonymous memory (P3 region) | Peer `uma` pool, private. About 9.1 MB per gpt-oss layer (F21) |
| KV and compute buffers of the CPU device | Anonymous memory | `uma`, private. KV is about 8.4 MB per gpt-oss layer at ctx 4096 [E] |
| Metal-resident layers (two-device peers, dense models) | Wired Metal shared memory (F3) | `uma`, private |

No repacked copy is made on the peer [A]. The host picks buffer types for an RPC device from that device's own list,
which holds only the RPC buffer type (Task A, `RPC/ggml-rpc.cpp:1318-1347`). The CPU backend's repack type is
therefore never used, and nothing duplicates the mapped bytes. The speed cost of skipping repack on ARM is unknown
[A], and S2 measures it.

### 3.5 Deliberately out of scope

- **Paging Metal weights.**
  - Metal buffers are wrapped at allocation and kept resident by residency sets (F3).
  - Paging them would need small per-range Metal buffers made resident around each command buffer, which is
    hcstream's slot design ported to Metal [A]. That is a separate project.
- **Dense parts on Metal with the experts on the CPU inside one peer.** Technically expressible (section 6.4), but
  every layer crosses MTL0 to CPU and back. Each crossing is a client-driven synchronous command (F12), so it costs
  `2 x layers x RTT` per token.
  - At the measured 7-11 ms hotspot RTT [M], 12 layers cost about 216 ms per token [E].
  - The optimizer may consider it only when the measured RTT is 0.5 ms or less.
  - A real fix is a server-side scheduler behind one virtual device, which is research.
  - The `meta` backend in `RPC/ext/meta.cpp` splits tensors along axes across devices. It is not this.
- **A Windows peer.** It would need a patched `ggml-rpc.dll`; the Windows release ships it separately (listing of
  `~\.hypercompute\engines\llama-b11770-windows-x64-cuda13.4`) [V]. The G14 also has about 15 GiB of free disk, so
  there is nothing worth building tonight.

### 3.6 Build and distribution on macOS

`scripts/build_llama_macos.sh` gains `--patch FILE` and `--build-dir DIR`:

- The script stops if the checkout has tracked changes (`:161-163`), so the patch is applied after that check.
- It builds into `build-peerstream` instead of `build-hypercompute` (`:168`).
- It records `sha256(patch)` beside the binaries.

The patch lives at `native/peerstream/peerstream.patch`, with a README and the MIT notice for the llama.cpp lines it
contains.

Running the build:

- Build on one 200 GB Mac. It needs Xcode CLT and cmake; the script's own checks report what is missing (`:103-136`).
- Then `export HYPERCOMPUTE_LLAMA_BIN_DIR=<build-peerstream>/bin` (F16).
- The join path accepts a source build [V]:
  - The release comparison is skipped when either side reports no release (`cli.py:284-287`;
    `services/nodes.py:384-385`).
  - The RPC protocol must still match (`services/nodes.py:382-383`). The patch keeps 8.0.0.
  - The local engine smoke test must pass (`cli.py:281`).
- Copying the `bin` directory to the other Macs is plausible: same arch, and ad-hoc signing like the release (F15).
  A copy that arrives by AirDrop carries `com.apple.quarantine`, which must be removed with `xattr -dr`. That is the
  owner's call on their own build [A].

## 4. Seeding each peer's shard from Hugging Face

### 4.1 Names

- The host decides the name: it hashes exactly the `nbytes` of each large tensor it sets (F4, F9). The cache file is
  `<LLAMA_CACHE>/rpc/<lowercase %016x of FNV-1a 64>`.
- A peer never needs to compute a name itself. It stages each tensor by its GGUF name, and the patched server (P5)
  hashes the staged file the first time the host asks for that hash, then renames it into the cache.
- So the content decides the name:
  - A corrupt or truncated download gets a different hash. The host never asks for it, and the tensor is uploaded
    from the host instead: slower, but correct.
  - Duplicate files are harmless.
- FNV-1a 64 reference vectors for the tests: `""` gives `cbf29ce484222325`, `"a"` gives `af63dc4c8601ec8c`, `"foobar"`
  gives `85944171f73967e8` (section 13) [V].

### 4.2 Flow (`backend/hypercompute/agent/seed.py`, CLI `hypercompute seed`, and a control-API call)

1. **Input** (from the host's plan, or by hand tonight):
   - the model id, which gives repo, revision, file names, sizes and sha256 from `configs/models*.yaml`,
     e.g. `configs/models.example.yaml:398-415`;
   - the peer's file-backed layer list.
2. **Tensor table.**
   - Download each shard's header range `[0, data_start)` (13.0 MB for gpt-oss, F21; start with 32 MiB and double if
     parsing fails) into `staging/<model-id>/.hdr/<file>`.
   - Make that file sparse at the full size (`truncate`) and run the existing `gguf.read_tensor_table`
     (`gguf.py:283-301`), which gives the shard, absolute offset and exact `nbytes`.
3. **Selection.** Tensors of the assigned layers with `nbytes > 10 MiB`. Smaller ones are sent whole by the host anyway
   (F4).
4. **Download.** For each tensor:
   - GET `Range: bytes=offset-(offset+nbytes-1)` into `staging/<model-id>/<name>.part`.
   - Resume from the `.part` size.
   - Check the length, then rename to `staging/<model-id>/<name>`.
   - Run 4 at a time.
   - On 403 or expiry, resolve the URL again: the CDN redirect is signed (F20).
   - Check the disk floor before each tensor and stop cleanly if it would be crossed.
5. **Progress and report.** Bytes done, bytes per second and ETA go into the heartbeat. Staged bytes per model go into
   the inventory (section 5).

Sizes for gpt-oss-120b per file-backed layer: 1.717 GB staged, 9.1 MB still sent by the host (F21).

Time for 14 layers (24.0 GB):

- about 16 min at 25 MB/s [A];
- seed on normal Wi-Fi, never on the phone hotspot, which is cellular data;
- the three Macs share one uplink, so their seeds add up.

Without seeding, the stock first load uploads at about 100 s per GB (521 s for 5.18 GB [M], checkpoint). That is about
40 min for one 14-layer peer, and peers load one after another.

### 4.3 Alternatives kept for later

- Seeding from the host over the LAN, which is useful when there is no internet.
- A tracker, so a second peer can copy a shard from the first.

## 5. Inventory, contribution and pool changes

**Contribution** (`ContributionLimits`, `schemas.py:147-154`):

- `file_backed_devices: list[str] = []`
- `disk_cap_bytes: int | None`: the most cache bytes this peer will hold.
- `disk_floor_bytes: int`: default `max(15 GiB, 10% of the volume)`.
- `threads: int | None`

New `join` flags (`cli.py:539-544`):

```sh
hypercompute join --devices CPU --cache --stream-from-disk --max-disk 120GiB --disk-floor 15GiB --threads 4
```

`--stream-from-disk` sets `file_backed_devices=["CPU"]` and requires `--cache`.

**Inventory** (`agent/inventory.py`, `schemas.HostInventory`):

- `rpc_cache: RpcCacheInfo | None`:
  - path, volume total and free (`shutil.disk_usage` on `config.paths().rpc_cache`);
  - bytes in `rpc/`;
  - staged bytes per model id, and verified tensors per model id. The agent learns these from the P8 lines it already
    tails through `proc.read_log` (`join.py:204-205`).
- `peerstream: PeerStreamInfo | None`, parsed from the P1 banner line with a new `_RPC_PEERSTREAM_RE` in
  `adapters/llama_output.py` (next to `:31-34`) and a new `RpcBanner.peerstream` field. Fields: devices, page size,
  floor, uma-copy, version.
- **Page cache, observed:**
  - `vm_stat`, file-backed pages times the page size [A: field name];
  - `psutil.virtual_memory().available`;
  - `sysctl vm.swapusage`, used for the swap guard in section 8.
- `fresh_measurements` (`inventory.py:227-240`) also refreshes disk free, cache bytes and swap. Today only RAM and VRAM
  are refreshed.

**Pools** (`capacity/pools.py`):

- `_peer_pools` (`:136-163`) adds a `(node, "ssd")` storage pool when `file_backed_devices` is not empty:
  - `free` = cache volume free;
  - `reserve` = disk floor;
  - `cap` = `disk_cap_bytes - bytes already cached`.
  - If any of these was not observed, the pool is excluded with that reason.
- Admission: a file-backed stage's new bytes go to `ssd`; its private bytes go to `uma`. Bytes already cached or staged
  cost no disk.
- The page cache is not an admission pool. It is an estimate for the speed model only:
  `C_peer = max(0, uma.free - reserve - private demand on uma)`, provenance `estimate`.
  - On Azim's Mac: 10.6 GB free [M, Task C] minus a 4 GiB reserve minus about 0.5 GB private gives about 5.8 GB [E].
- `services/deployments.remote_budgets` (`:172-200`) today shares `max_bytes` across RPC devices by their reported free
  memory. A file-backed device must instead get:
  - a private budget (KV, compute, small tensors),
  - a disk budget,
  - a page-cache estimate,
  - and `residency="file_backed"` on its `DeviceBudget`.

## 6. Planner and optimizer changes

### 6.1 Device model

`DeviceBudget` (planner) and `PeerSide` (`capacity/optimizer.py:61-72`) gain:

- `residency`
- `disk_admissible_bytes`
- `cached_bytes_by_unit`
- `page_cache_bytes_est`
- `fault_ms_per_mb`
- `hop_ms`, as a `Coefficient` with provenance

For a file-backed stage, `_assignment` and `_layer_capacity` (`services/planner.py:141-160`) split the bytes using the
tensor table:

- private = the small tensors of the stage + KV + compute + overhead, and must fit the private budget;
- file = the large tensors, padded to whole pages, and the part not yet cached must fit the disk budget.

The 10 MiB threshold is `HASH_THRESHOLD` (F4); keep it as a constant with that citation.

### 6.2 Stages and device order

Per peer the optimizer picks one of three shapes:

| Shape | Peer devices (`-d`) | Per-token hop | When |
| --- | --- | --- | --- |
| CPU file-backed span | `CPU` | `H` | MoE. Task C's "explicit cache" case, its best for both hosts [E] |
| MTL0 resident span then CPU file-backed span | `MTL0,CPU` | `H + 1 RTT` with P7, `H + 2 RTT` stock | Dense layers that fit the peer's page cache, or a measured gain |
| MTL0 resident span | `MTL0` | `H` | Dense models; today's behaviour |

Global device names are `RPC0..` in `--rpc` endpoint order and, within an endpoint, in the server's `-d` order (Task A,
`RPC/ggml-rpc.cpp:3150-3186`); `services/deployments.assign_rpc_names` (`:209`) already numbers them this way.

The layout is:

```
--device RPC(peer1 MTL0?),RPC(peer1 CPU),RPC(peer2 ...),...,CUDA0|MTL0(host)
--split-mode layer
--tensor-split <whole-layer counts>
```

Host CPU experts use the existing `-ot ...=CPU` (`moe_offload_args`, `llama_rpc.py:324-342`). Every peer stage is whole
layers.

The optimizer's exact DP already handles "zero or more peers, each a contiguous span"
(`hybrid-capacity-implementation.md`, optimizer contract). The change lets one peer contribute up to two consecutive
spans on two devices that share its `uma` pool, and charges the extra RTT.

### 6.3 Per-peer MoE cost (replaces the no-reuse rule)

For a file-backed stage holding layers `S`:

```
W     = sum(experts(i) for i in S)                     # file-backed routed-expert bytes
D     = sum(dense(i)   for i in S)                     # read every token, assumed hot in cache
C     = page_cache_bytes_est - D
miss  = exp(-C / (kappa * W))                          # kappa measured 0.176 [M fit], pessimistic 0.35 [A]
ssd   = f * W * miss                                   # f = expert_used / expert_count
ms    = c_cpu * (D + f * W) / 1e9  +  fault_ms_per_mb * ssd / 1e6  +  hop_ms (+ RTT if a second device)
```

- Fault I/O is added to compute, not overlapped with it: mmap faults stall the compute threads. `cpu_stage_ms`'s `max`
  (`capacity/costs.py:399-401`) is for hcstream slots only (Task C).
- The same model replaces the host's `miss = 1 - C/W` (`services/moe_tier.py:128` in the working tree, which another
  session is editing).

Coefficients:

| Coefficient | Value | Provenance |
| --- | --- | --- |
| Mac CPU compute `c_cpu` | 25 ms/GB | [A] |
| Mac fault rate | 0.90 ms/MB, same as the G14 fit | [A] |
| `hop_ms` | 60 ms per peer stage today | [M fit, Task C] |
| `HOP_MS` (`services/planner.py:49`) | 4.0 | Replaced by the per-peer `hop_ms` |
| `ROUND_TRIPS_PER_TOKEN` (`capacity/costs.py:51`) | 2 | Replaced by the per-peer `hop_ms` |

Each measured value is stored in `~/.hypercompute/capacity/measurements.json` with provenance.

Worked example [E]: a CPU-only Mac with 14 gpt-oss layers in the Mac-host configuration.

| Term | Value |
| --- | --- |
| `W` | 23.75 GB |
| `D` | 0.42 GB |
| `C` | 5.42 GB |
| `miss` | 0.27 |
| `ssd` | 0.20 GB/token |
| Compute | 29 ms |
| Faults | 183 ms |
| Hop | 60 ms |
| **Stage total** | **about 271 ms** |

Faults dominate. That is why S1 and S2 must measure the Mac fault rate and test `HC_PEERSTREAM_ADVICE`: Task C puts a
0.5 ms/MB Mac at 4.3 tok/s for the Macs-only cluster.

For dense layers that are file-backed, the miss is 0 if `W <= C`. Otherwise every byte misses once per token, because
LRU loses under cyclic access [A]. The optimizer only chooses that when nothing else fits, and labels it.

### 6.4 `-ot` to a peer CPU buffer type (expressible, off by default)

This is the "dense on Metal, experts on peer CPU" variant. For a peer at `127.0.0.1:50052` with `-d MTL0,CPU` and
layers 0-5 on its MTL0:

```
-ot "^blk\.([0-5])\.ffn_(up|down|gate_up|gate)_(ch|)exps\.=RPC1[127.0.0.1:50052]"
```

The buffer type name is `RPC<server-local index>[<endpoint>]` (Task A, `RPC/ggml-rpc.cpp:1318-1347`). The same format
appears in real host logs (`adapters/llama_output.py:8`) [V].

Requirements:

1. `--rpc` before `-ot`, because arguments are parsed left to right (Task A, `LL/common_arg.cpp:822-880`). Move the
   `--rpc` emission in `build_args` ahead of the tier arguments (F19).
2. The CPU device in `--device` even with a 0 count, or its weights have no backend and the scheduler aborts (Task A,
   `LL/src_llama-context.cpp:421-427`).
3. No comma in the pattern. `gguf.number_alternation` guarantees this (`gguf.py:371-373`).

Cost `2 x |S| x RTT` per token (section 3.5). The optimizer considers this shape only when the measured RTT is 0.5 ms
or less.

### 6.5 Load cost and layout stability

- Cold load: (uncached and unstaged large bytes) at the measured upload rate, plus staged bytes at the peer hash rate
  (about 1 GB/s [A]), plus the host's read and hash of every peer byte (about 1 GB/s hash [A]).
- Tie-break plans that reuse seeded shards. Seeding follows the plan (section 4.2 step 1), so a different split later
  turns some hits into uploads.
- `OptimizeResult.explain()` gets one line per peer stage, with figures like section 6.3: units, disk and cached
  bytes, private bytes against admissible, cache estimate, miss, SSD per token, and compute, fault and hop terms, each
  with provenance.

## 7. Host modes

- **Mode A, host-full (tonight).** The host holds the whole GGUF. Peers hold cache copies of their shards, so those
  bytes exist twice. Stock host binaries.
  - At each load the host reads and hashes every byte it places on a peer (F4, F9).
  - The host's disk therefore bounds the model: about 75 GB on the G14 if gpt-oss is deleted, about 190 GB on a
    200 GB Mac, about 55 GB on Azim's Mac (Task C) [E].
- **A Mac as host.**
  - Same mechanism. The host runs `moe_offload` with Metal for the dense parts, head and KV, and its own experts
    file-backed through `llama_mmap` (`mmap(PROT_READ, MAP_SHARED)`, `LL/src_llama-mmap.cpp:485`, `:495`, Task B) [V].
  - That path is being prepared by another session and has never run (`docs/research/mac-host-runbook.md`).
  - Use a 200 GB Mac. The runbook names Azim's Mac as host and requires 62 GiB free (its section 1.1), but Azim has
    about 60 GB, which is also what Task C flagged.
- **Mode B, host-sparse (later; this is what "pooled disk" means).** The host keeps only the header and its own
  layers, in a sparse GGUF. This needs a host-side client patch: for tensors listed in a hash manifest (built from the
  peers' P8 lines), send `SET_TENSOR_HASH` from the manifest without touching the data, and abort on a miss instead of
  uploading zeros from a hole.
  - That patch is a replacement `ggml-rpc.dll` on Windows and `libggml-rpc` on macOS, each with its own gate.
  - It lets the G14, the best GPU, host a model bigger than its 15 GiB of free disk.
  - Pooled raw disk is about 475 GB (Task C) [E].

## 8. Safety

- **Fail closed.**
  - The server refuses to start on bad configuration (P1).
  - A file-backed device that cannot get its file fails the load (P6, F7). It never silently falls back to private
    memory.
  - The host deployment fails, and the existing fallback and replan logic takes over.
- **Disk floors.**
  - Enforced by the seeder before each range and by the server before each cache write (`statvfs`).
  - Default `max(15 GiB, 10%)`. That leaves room for macOS swapfiles, which grow on the boot volume [A], and for APFS
    snapshots.
  - Azim's Mac (about 60 GB free) keeps 15 GiB, leaving about 44 GB, which is 25 gpt-oss layers.
- **Contributor reserve.**
  - The reserve keeps its meaning for private memory (`pools.py` admission).
  - File-backed bytes are not private. Page cache is reclaimable, so the owner's apps take it back first and the cost
    is slower tokens, not the owner's RAM [A]. S2 checks this by opening heavy apps during a decode.
- **Swap.**
  - Clean file pages are dropped, not swapped [A]. Private demand stays small (section 3.4).
  - The agent records `vm.swapusage` when the worker starts. If swap grows by more than 1 GiB, or free disk falls
    below the floor, it marks the node `pressure`; `run_big` stops, as it already does on the host with
    `--max-swap-growth-gib`, `run_big.py:771`. Later the replan controller drops the peer.
- **Cache hygiene.**
  - Writes use a temp file and a rename (P6, P6b), so a renamed-over or deleted file keeps the inode alive for existing
    mappings [A].
  - Purge already runs only after the worker stops (`cli.py:367-378`).
  - Exclude `rpc-cache` from Time Machine (`tmutil addexclusion`) and from Spotlight [A].
- **Security.**
  - Tensor names come from the client, so P10 limits staging lookups to safe names.
  - ggml RPC remains unsafe on an open network (`hybrid-llama-audit.md:137`). Only Iroh-authenticated hosts reach it.
- **Power and data.** Macs stay on AC (runbook rule). Seed on Wi-Fi, not on the hotspot.

## 9. Verification from logs

**Host** (`llama-server -lv 4`):

- `load_tensors: RPCn[ep] model buffer size = X MiB` (`LL/src_llama-model.cpp:2009`; parsed by `_BUFFER_RE`,
  `llama_output.py:36-39`). X must equal the plan's large plus small bytes for that device, padded to whole pages, to
  within 1%.
- `llama_kv_cache: RPCn[ep] KV buffer size` and `sched_reserve: RPCn[ep] compute buffer size`
  (`LL/src_llama-kv-cache.cpp:291`, `LL/src_llama-context.cpp:842`). Their sum, plus small tensors, gives the peer's
  private bytes.
- Layer assignment lines (`LL/src_llama-model.cpp:1677`) and `-ot` override lines (`LL/src_llama-model-loader.cpp:1254`)
  are debug level, so a gate needs `-lv 5` (`llama_output.py:11-14`).
- Load time with every tensor seeded is about the host's hashing time. Any upload shows up as minutes.

**Peer** (worker log):

- The P1 banner line.
- P8 per-tensor lines and the summary. Pass:
  - `mapped` bytes equal the planned large bytes;
  - `copied` is only the small tensors;
  - `uploads 0` when seeded;
  - `unaligned 0`;
  - `staged bad 0`.
- With `GGML_RPC_DEBUG=1`, the `alloc_buffer` lines with device and size (`RPC/ggml-rpc.cpp:1514-1519`).

**Peer OS:**

- `footprint` or `vmmap --summary` on the rpc-server PID. Physical footprint must be about the private estimate, not
  gigabytes of weights; mapped-file resident memory grows toward the cache size [A: tool output].
- The `vm_stat` pageins delta per decoded token, compared with the predicted SSD per token.
- `vm.swapusage` unchanged.

**Correctness A/B:** the same layout and devices with stock and with patched peers must give bit-identical tokens at
temperature 0. Same kernels and same bytes, so this is exact; CPU against CUDA would not be.

## 10. Staged build plan

Windows-runnable tests use `uv run --no-sync pytest -q -p no:cacheprovider`. Real-Mac gates say what must be shown.

**S0, measure the per-peer hop (no code, tonight, 45 min [A]).**

- G14 plus one Mac on the hotspot. Run the L2 recipe (`docs/testplan/open_questions.md:177`, commands at `:370-372`)
  on the 0.5B with 0, 1 and 2 Mac layers.
- Repeat over plain LAN TCP without Iroh, on a network without client isolation, to split tunnel time from network
  time.
- Run 16 decode tokens with `GGML_RPC_DEBUG=2` on the host to count synchronous commands per token per stage.
- Hypotheses, none tested:
  - the Python bridge (`transport/iroh.py:380-415`, being edited in the working tree);
  - Wi-Fi power save;
  - a synchronous `GET_ALLOC_SIZE` per token (`RPC/ggml-rpc.cpp:1036-1060`);
  - CPU contention.
- Exit: `hop_ms` per peer stored as measured, plus a written decision whether to fix the bridge before adding Macs.

**S1, patch and 7B gate (tonight, about 3 h [A]).**

Code:

- `native/peerstream/peerstream.patch` (P1-P10)
- the build-script flags (3.6)
- the banner parser
- agent arguments and environment

Tests:

- `git apply --check` of the patch against `RPC/ggml-rpc.cpp`
- banner parse fixture (`tests/fixtures/`, the existing `*_b11770.*` style)
- `join.py` argument and environment builder for every contribution shape

Real-Mac gates on one 200 GB Mac:

1. The build succeeds at commit `39da130f`, and the banner shows `v8.0.0` plus the peerstream line.
2. With `HC_PEERSTREAM` unset, Qwen2.5-7B gives tokens identical to the release `ggml-rpc-server`.
   - Use the 7B, not the 0.5B: no 0.5B layer tensor is over 10 MiB, so nothing would be mapped. The 7B has 3 per
     layer (F21).
3. Host G14 with 8 layers of the 7B on the Mac's `-d CPU`: patched tokens identical to stock with the same layout.
   - P8 shows 24 mapped and 72 copied tensors (expected from F21's layer-0 counts); footprint shows only the
     private bytes.
   - This also covers partial last pages, since some 7B tensors are not 16 KiB multiples (F21).
4. Failure cases:
   - A truncated cache file becomes a miss and an upload.
   - A floor set above the free space makes the load fail closed with a clear line.
   - A bad `HC_PEERSTREAM_DEVICES` stops the server before it listens.
5. Disconnecting the host unmaps everything (`vmmap` shows no cache files mapped).
6. Measure `-t 4` against `-t 6` on decode, and the Mac CPU ms per GB.

**S2, seeding plus one-peer gpt-oss (tonight, about 2 h [A]).**

Code: `agent/seed.py`, `hypercompute seed`, and the inventory `rpc_cache` fields.

Tests:

- range selection from a fixture GGUF;
- `.part` resume;
- length check and rename;
- floor refusal;
- 403 re-resolve (mocked HTTP);
- FNV reference vectors.

Real gate:

1. Seed 5 gpt-oss layers (8.6 GB) on one Mac over Wi-Fi.
2. Run the G14 host with that Mac `-d CPU` file-backed and the rest as today.
3. Check:
   - P8 `staged ok 25`, `uploads 0`;
   - output coherent;
   - decode, SSD per token on both machines, and Mac pageins per token;
   - compare with the 4.52 tok/s memory-only pair [M].
4. Pressure test: open heavy apps on the Mac during a decode and record memory pressure, swap and the slowdown.

**S3, inventory, pools and contribution (next day).**

Section 5 code. Unit tests:

- the peer `ssd` pool, and its exclusion reasons;
- admission split into private and disk for a file-backed device;
- the heartbeat refreshes disk and swap;
- `--stream-from-disk` validation.

Real: `join --stream-from-disk` shows in `doctor` and in the host's node view.

**S4, planner and optimizer (next day; the planner files are under another session's edits, so land after they
commit).**

Section 6 code. Unit tests:

- the κ miss model reproduces Task C's calibration: 205 and 134 MB/token predicted, against 232 and 121 measured;
- the file-backed fit;
- device list and tensor split for the three peer shapes;
- `--rpc` before `-ot` in `build_args`;
- the optimizer never splits a layer inside one peer while RTT is over 0.5 ms;
- an oracle check against brute force on small models, as for the existing DP.

Real: `run_big --plan-only` explains each peer stage, then one run where predicted and measured are within the stated
error.

**S5, demo runs.**

- **(a)** G14 host plus 3 Macs, gpt-oss-120b, peers file-backed, after S0's fix. Record decode, first token, SSD per
  token per machine, and participation.
- **(b)** A 200 GB Mac host, the other two Macs file-backed, and the G14 as a stock memory-only CUDA peer, with a model
  over 59 GiB (for example GLM-4.5-Air Q4_K_M at 73.0 GB, or Qwen3-235B UD-Q2_K_XL at 88.8 GB; Task C sizes). This
  needs the Mac-host path to pass first.

**S6, later.**

- Mode B (host-sparse) with the host client patch on Windows and macOS.
- Metal-side paging.
- A server-side multi-device peer.
- A Windows peer patch.
- LAN seeding.

## 11. Machine roles tonight

| Machine | Role | Why |
| --- | --- | --- |
| G14 (RTX 5070 Ti 12 GiB, 31 GiB RAM, about 15 GiB free disk) | Host for gpt-oss-120b only | Has the verified file, the fastest dense compute, and about 19.5 GB of page cache [M]. Cannot hold any other big model. Not a peerstream peer (no Windows patch, no disk). For S5(b) it joins as a stock memory-only CUDA peer. |
| Mac A (200 GB) | Build Mac and first file-backed peer, later the host for models over 59 GiB | Has the disk for both. Installs CLT and cmake once. |
| Mac B (200 GB) | File-backed peer | Gets the copied or rebuilt binaries. |
| Azim's Mac (about 60 GB) | File-backed peer, not host | 15 GiB floor leaves about 44 GB of shards. The gpt-oss GGUF alone is 63.4 GB. |

## 12. Assumption register

| # | Assumption | Checked in |
| --- | --- | --- |
| A1 | `mmap(MAP_FIXED\|MAP_PRIVATE, PROT_READ)` of a file over part of our own anonymous mapping works on XNU and is undone by `munmap` | S1 gate 3, gate 5 |
| A2 | Apple Silicon page size is 16 KiB (P1 prints the real value) | S1 gate 1 |
| A3 | Clean mapped file pages are dropped under pressure and faulted back from SSD; owner apps win | S2 pressure test |
| A4 | macOS overcommits the P3 anonymous reservation without charging it | S1 gate 3 footprint |
| A5 | CPU kernels on the M4 run MXFP4 at usable speed without repack; Mac CPU about 25 ms/GB | S1 gate 6, S2 |
| A6 | Mac fault reads cost about 0.9 ms/MB, and `HC_PEERSTREAM_ADVICE` can improve that | S2 |
| A7 | The 60 ms per-peer excess is removable to about 13 ms | S0 |
| A8 | HF ranges sustain about 25 MB/s per Mac, and signed redirects can be resolved again | S2 |
| A9 | A source build needs only CLT and cmake, and the copied `bin` runs after quarantine removal | S1 gate 1, then Mac B |
| A10 | P7's `memcpy` between Metal shared and CPU buffers is correct after `sync_all_backends` | Only when a two-device peer is first used (S4) |
| A11 | FNV at about 1 GB/s on M4 and G14 (single thread) | S1, S2 load timings |
| A12 | Tool output formats (`vm_stat`, `footprint`, `vmmap`, `vm.swapusage`) match the parsers | S1 |
| A13 | `tmutil addexclusion` and Spotlight `.noindex` work without sudo for a user path | S3 |
| A14 | Swapfiles grow on the boot volume and a 15 GiB floor is enough | S2 pressure test |
| A15 | κ = 0.176 from two G14 runs on similar prompts holds on Macs and for other prompts | S2, S5 |

## 13. What was run for this document

All read-only. Nothing downloaded or executed from a third party.

1. Reads of the pinned sources and the repo files cited above.
2. String and CodeDirectory inspection of `MR/ggml-rpc-server`, `libggml.0.27.0.dylib`, `libggml-rpc.0.27.0.dylib`,
   `libggml-base.0.27.0.dylib` and `llama-server`. These were byte reads with `grep -a` and `python -I`, and none of
   them was executed.
3. One `curl -I` with `Range: bytes=0-15` against the pinned Hugging Face URL (`configs/models.example.yaml:413`).
   Headers only.
4. `uv run --no-sync python ...\scratchpad\tensor_split.py` over the local gpt-oss and Qwen2.5-7B GGUFs, using the
   repo's `gguf.read_tensor_table` (header only).
5. FNV-1a 64 reference vectors computed in Python.
6. Task C model and its output, used for every [E] number:
   `...\scratchpad\taskc_model\model.py` and `...\scratchpad\taskc_model\out.txt`.

## 14. Go/no-go

**GO tonight for S0, S1 and S2.** They need one 200 GB Mac with CLT and cmake, the G14 with the gpt-oss file it already
has, and normal Wi-Fi for seeding. They produce real evidence:

- the per-peer hop;
- bit-identical tokens from a file-backed peer;
- Mac fault cost and page-cache behaviour;
- one-peer gpt-oss with a seeded shard.

**NO-GO tonight** for:

- publishing 4-device numbers;
- anything bigger than gpt-oss-120b;
- pooled disk;
- a Windows peer;
- the dense-on-Metal and experts-on-CPU split inside one peer.

**Conditional stretch tonight.** If S0 brings the hop under about 20 ms and S1 and S2 pass, run G14 plus 3 file-backed
Macs on gpt-oss with a hand-written layout as a preview. Label it a preview, not a planner result.
