# EXO audit for the Hypercompute EXO adapter

Audited 2026-10-21 (UTC) through the authenticated `gh` CLI
(`gh api -H "Accept: application/vnd.github.raw" repos/exo-explore/exo/contents/PATH?ref=SHA`), the GitHub
issues/PR API, the Hugging Face model API, the Homebrew cask API, one HTTP `HEAD` on the DMG URL and the
Microsoft WSL networking docs. No EXO code was cloned, built, installed or run. Line numbers refer to files at
the pinned SHA unless stated.

This file extends `later_engines_and_prior_art.md` section A. It does not repeat that section's license,
route list or acknowledged-vs-ready summary; it adds install/run, networking, Linux, full API shapes,
failure semantics and an adapter recipe.

Status words:
- VERIFIED: read in source, lock file, API output or docs at the stated ref.
- INFERRED: follows from the code, not exercised on hardware.
- REPORTED: stated in an upstream issue or PR by a third party, not checked by us.
- UNVERIFIED: could not be confirmed, with the reason.

---

## 0. TL;DR

1. **Pin: keep `bc412e2230f395e652c23426a5eb88dfa515769b`.** It is still the tip of `main` (committer date
   2026-10-13T20:11:46Z; repo `pushed_at` 2026-10-13T20:11:49Z; checked 2026-10-21T04:58Z). There is no
   `staging` branch. Install with `uv sync --locked` so the git deps resolve to the revisions in `uv.lock`.
2. **The packaged app cannot join a source node at this pin.** `EXO-latest.dmg` is v1.0.71
   (Last-Modified 2026-04-23, same as the GitHub release and the Homebrew cask). It predates the
   libp2p-to-zenoh rewrite (#2132, 2026-06-03), so its discovery protocol and default namespace differ.
   There is no pip package (PyPI `exo` is an unrelated yeast-genomics tool).
3. **macOS from source needs** full Xcode (Metal toolchain, MLX is compiled from git), `uv`, Python 3.13
   (uv-managed), Node >= 18 to build the dashboard, and stable Rust (edition 2024) because `exo-rs` is a
   maturin/pyo3 crate that `uv sync` builds.
4. **Discovery is LAN-only**: IPv6 link-local multicast `ff12::e0a1:de89` on UDP 52413, then zenoh TCP on
   52414, then each worker probes every other node's HTTP API (`GET :52415/node_id`) to build topology
   edges. Static peers (`--bootstrap-peers`) are disabled at this pin. No authentication anywhere.
5. **Namespace isolation** is `--namespace` / `EXO_ZENOH_NAMESPACE` (default = the installed exo version,
   `0.3.70`). It is an 8-byte blake3 prefix compared in a cleartext UDP Hello; it isolates, it does not
   authenticate. Restrict membership with a host firewall.
6. **Linux CPU-only is not usable out of the box at this pin**: the `mlx-cpu` extra locks a fork `mlx`
   0.32.0 wheel with PyPI `mlx-cpu` 0.31.2, reported to fail on `import mlx.core` (open #2359; fix PR #2385
   open, not mergeable). Linux GPU is not supported (README: "exo runs on CPU on Linux"). Native Windows is
   impossible (`import resource`, uv environments darwin/linux only).
7. **"Ready" means every runner of the instance is `RunnerReady` or `RunnerRunning` in `/state/runners`.**
   `/instance/await` only proves the instance exists. EXO already runs a 50-token warmup before
   `RunnerReady`.
8. **Use `GET /instance/previews?model_id=..&node_ids=A&node_ids=B` then `POST /instance {instance}`**
   for explicit node selection. The instance id is known up front. `POST /place_instance` recomputes the
   placement on the master and returns no instance id.
9. **A node leaving deletes the instance; there is no "failed instance" state.** After 15 s of silence the
   master emits `NodeTimedOut` + `InstanceDeleted`; open streams end with an SSE error object and
   `data: [DONE]`. No automatic re-placement.
10. **Every node downloads the whole model repo** (`resolve_allow_patterns` returns `["*"]`), even for a
    pipeline shard. Built-in model cards default to `trust_remote_code = true`.

---

## 1. Pin recommendation

| Item | Evidence | Status |
|---|---|---|
| `main` tip = `bc412e2230f3...` | `gh api repos/exo-explore/exo/commits/main` returned the same SHA; `compare/bc412e2...main` = `identical, ahead=0 behind=0` | VERIFIED |
| Last 8 commits on main are app/dashboard fixes dated 2026-10-13 (#2410-#2427) | commits API | VERIFIED |
| Discovery fixes are included: #2320, #2327, #2346 (2026-10-12), #2419 (2026-10-13) | `commits?path=rust/networking/src/discovery.rs` | VERIFIED |
| zenoh migration commit `09f9ea313f` "libp2p -> zenoh (#2132)" 2026-06-03 | same listing | VERIFIED |
| v1.0.71 tag `fd707de` still uses libp2p | `rust/networking/Cargo.toml@fd707de` contains `libp2p = { workspace = true, features = ["full"] }`; `compare fd707de...bc412e2` = ahead 136 | VERIFIED |
| Version string | `pyproject.toml:3` `version = "0.3.70"`; `src/exo/__init__.py` `__version__ = version("exo")` | VERIFIED |

Recommendation: keep `bc412e2`. Run the identical commit on every node. Because the default namespace is
the version string (`0.3.70`), two different commits with the same version would still discover each other
and may then mis-parse each other's messages (`router.py:94-100` drops unreadable messages). Set
`EXO_ZENOH_NAMESPACE=hc-bc412e2-<random>` to enforce same-commit and isolate the cluster at once.

Reproducibility of third-party sources (all VERIFIED reachable on 2026-10-21):

| Dependency | Declared (pyproject / Cargo.toml) | Locked |
|---|---|---|
| `mlx` (darwin) | git `ml-explore/mlx` rev `0e3ff364...` (`pyproject.toml:84`) | `uv.lock:1430-1432` version `0.32.4.dev20261003+0e3ff3643` |
| `mlx` (linux x86_64) | fork wheel `rltakashige/mlx-jaccl-fix-small-recv` release `mlx_cuda` (`pyproject.toml:86`) | `uv.lock:1403-1411` `0.32.0`, sha256 `58a83a19...` |
| `mlx-lm` | git `rltakashige/mlx-lm` **branch** `leo/deepseek-v4` (`pyproject.toml:96`) | `uv.lock:1547-1549` rev `6a3df6cd` |
| `mflux` | git `evanev7/mflux` **branch** `exo2` (`pyproject.toml:97`) | `uv.lock:1337-1339` rev `0fdd4cca` |
| `zenoh` 1.9.0 | git `evanev7/zenoh` **branch** `exo` (`Cargo.toml:63-89`) | `Cargo.lock` rev `aa219b84` (ancestor of current branch tip `2fed58ef`) |

The branches can move; only the lock files pin them. Hypercompute should install with
`uv sync --locked ...` and consider mirroring the fork wheels.

---

## 2. macOS Apple Silicon: install and run

### 2.1 Packaged options (VERIFIED)

| Option | Facts | Usable with a source node at the pin? |
|---|---|---|
| DMG `https://assets.exolabs.net/EXO-latest.dmg` | `HEAD`: `Content-Length: 412972413`, `Last-Modified: Thu, 23 Apr 2026 15:04:06 GMT` = v1.0.71 | No: pre-zenoh (libp2p) |
| `brew install --cask exo` | cask `version 1.0.71`, `url .../EXO-1.0.71.dmg`, `depends_on arch arm64, macos >= 15`, `auto_updates: true` | No, same build |
| GitHub releases | latest `v1.0.71` 2026-04-23, asset `EXO-1.0.71.dmg` | No |
| pip | PyPI `exo` 0.1.2 is "Quality Control and Visualization utilities for Yeast Epigenome Project" (CEGRcode); `exo-rs` is not on PyPI | Not available |
| Nix | `nix run .#exo` from a checkout (README:126-130); Cachix cache `exo` | Yes in principle (UNVERIFIED, not run) |

Other app facts: README:81 says "The macOS app requires macOS Tahoe 26.2 or later" while the cask says
macOS >= 15 (conflict, UNVERIFIED which is right). README:91 says the app "will ask for permission to
modify system settings and install a new Network profile"; the cask uninstall stanza removes a LaunchDaemon
`io.exo.networksetup`. The app's default namespace is its build tag or `CFBundleShortVersionString`
(`app/EXO/EXO/ExoProcessController.swift:512-528`), which differs from the source default `0.3.70`.
**Recommendation: do not use the app on friends' Macs; run the pinned source build under the Hypercompute
agent.**

### 2.2 Prerequisites (from source)

| Requirement | Why | Evidence |
|---|---|---|
| Full Xcode (not only CLT) + Metal toolchain | MLX is compiled from git on darwin, needs the Metal compiler | README:140 "provides the Metal ToolChain required for MLX compilation"; `pyproject.toml:84`; CI runs `xcodebuild -downloadComponent MetalToolchain` (`.github/workflows/pipeline.yml:57`); merged PR #2404 "build MLX from upstream main on macOS" |
| CMake >= 3.25 | MLX build backend | `ml-explore/mlx@0e3ff36 pyproject.toml` `requires = ["setuptools>=80", "cmake>=3.25", ...]`; uv installs it as a build dependency (INFERRED) |
| uv >= 0.8.6 | `[tool.uv] required-version = ">=0.8.6"` | `pyproject.toml:186` |
| Python `==3.13.*` | `requires-python` | `pyproject.toml:6`; `.python-version` = `3.13`; uv can download it |
| Rust stable, edition 2024 (so >= 1.85) | `exo-rs` is built by maturin (`rust/exo_rs/pyproject.toml:1-3` `requires = ["maturin>=1.0,<2.0"]`), a uv workspace member (`pyproject.toml:78-82`) | `Cargo.toml:7` `edition = "2024"`; nix uses `fenix ... stable` (`rust/parts.nix:7`). No separate `maturin` install needed: uv runs the build backend (INFERRED) |
| Node + npm (README Linux section says >= 18) | dashboard build; the API refuses to start without a dashboard dir (see 2.6) | README:147, 176, 196 |
| `macmon` (optional) | GPU/power/temp; without it "exo falls back to psutil and monitors memory only" | README:157-167, pinned rev `a1cd06b6...` |

### 2.3 Commands (from README:169-183, hardened for Hypercompute)

```bash
# once per Mac
xcode-select -p                                   # must point into Xcode.app, not CommandLineTools
xcodebuild -downloadComponent MetalToolchain      # Xcode 26+; INFERRED from CI, harmless if present
brew install uv node
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # stable toolchain

git clone https://github.com/exo-explore/exo ~/hc/exo
cd ~/hc/exo && git checkout bc412e2230f395e652c23426a5eb88dfa515769b
(cd dashboard && npm ci && npm run build)         # README uses npm install; npm ci honours package-lock.json
uv sync --locked --extra mlx                      # builds MLX from source and exo_rs via maturin

# run a node (foreground; Hypercompute agent supervises it)
export EXO_ZENOH_NAMESPACE="hc-bc412e2-<16 random hex>"     # same on every node
uv run --frozen exo -v --api-port 52415
```

`uv run --frozen` avoids re-resolving at start. `.venv/bin/exo` also works (`[project.scripts] exo =
"exo.main:main"`, `pyproject.toml:32-33`).

Process facts:
- One exo per cache dir: a PID file at `EXO_CACHE_HOME/exo.pid` is taken before anything starts
  (`main.py:303-308`, `constants.py:82`). Two nodes on one machine need different `EXO_HOME` and ports
  (INFERRED; the discovery socket uses `SO_REUSEPORT` on unix, `discovery.rs:72-74`).
- The API port is taken first so "port in use" fails fast: `Port 52415 is already in use - is exo already
  running? Use --api-port to choose another.` (`api/listening_socket.py:7-14`).
- `SIGINT`/`SIGTERM` trigger a clean shutdown; a second signal calls `sys.exit(1)` (`main.py:183-204`).
- macOS will prompt for Local Network access and incoming connections. If Local Network is blocked,
  discovery logs "peer discovery could not send on any network interface ... allow it in System Settings >
  Privacy & Security > Local Network" every 60 s (`discovery.rs:309-317`).

### 2.4 CLI flags (`src/exo/main.py:420-563`, VERIFIED)

| Flag | Default | Effect |
|---|---|---|
| `-v` / `-q` | 0 | verbosity |
| `-m`, `--force-master` | off | election seniority 1,000,000 (`main.py:154`) |
| `--no-api` | API on | no dashboard **and no API**; see 2.6 for why this breaks multi-node |
| `--api-port` | 52415 | dashboard + API port |
| `--no-worker` | worker on | coordinate only, run no models |
| `--no-downloads` | on | disable the download coordinator |
| `--offline` | `EXO_OFFLINE` | local models only |
| `--legacy-daemon` | off | double-fork daemon |
| `--namespace` | `EXO_ZENOH_NAMESPACE` or `EXO_LIBP2P_NAMESPACE` or `__version__` (`main.py:407-417`) | discovery namespace |
| `--zenoh-port` | 52414 | TCP port nodes connect on |
| `--discovery-port` | 52413 | UDP discovery port |
| `--bootstrap-peers` / `EXO_BOOTSTRAP_PEERS` | hidden | **refused**: `raise ValueError("Bootstrap peers has been temporarily removed")` (`main.py:373-374`) |
| `--no-batch`, `--fast-synch`, `--no-fast-synch` | hidden | developer options |

### 2.5 Environment variables (VERIFIED in source)

| Variable | Effect | Evidence |
|---|---|---|
| `EXO_ZENOH_NAMESPACE` | discovery namespace; `EXO_LIBP2P_NAMESPACE` still honoured with a deprecation warning | `main.py:364-368, 407-417` |
| `EXO_HOME` | config, data and cache all become `Path.home() / EXO_HOME` (an absolute value wins by pathlib rules) | `constants.py:7-22` |
| `XDG_CONFIG_HOME` / `XDG_DATA_HOME` / `XDG_CACHE_HOME` | Linux only; on macOS everything is `~/.exo` | `constants.py:16-22` |
| `EXO_DEFAULT_MODELS_DIR` | first writable model dir; default `~/.exo/models` (macOS), `~/.local/share/exo/models` (Linux) | `constants.py:30-35`, README:314 |
| `EXO_MODELS_DIRS` | extra writable dirs, colon-separated, first with enough free space wins | `constants.py:45-57`, `download_utils.py:181-221` |
| `EXO_MODELS_READ_ONLY_DIRS` | pre-staged models, searched first, never deleted | `constants.py:45-57`, `download_utils.py:152-166` |
| `EXO_OFFLINE` | no internet checks | `constants.py:106`, `main.py:428` |
| `EXO_DASHBOARD_DIR` | dashboard dir, `Path.home() / value` | `constants.py:63-71` |
| `EXO_RESOURCES_DIR` | built-in model cards dir, `Path.home() / value` | `constants.py:59-62` |
| `EXO_ENABLE_IMAGE_MODELS` | show image cards | `constants.py:102-104` |
| `EXO_TRACING_ENABLED` | tracing | `constants.py:108` |
| `EXO_MAX_CONCURRENT_REQUESTS` | default 8 | `constants.py:112` |
| `ENABLE_DISAGGREGATION` | prefill/decode instance links | `constants.py:110` |
| `OVERRIDE_MEMORY_MB` | non-darwin only: report this as `ram_available` | `info_gatherer.py:511-520` |
| `EXO_FAST_SYNCH` | MLX `MLX_METAL_FAST_SYNCH` override | `main.py:380-386`, `runner/bootstrap.py:76` |
| `HF_TOKEN`, `HF_HOME`, `HF_ENDPOINT` | HF auth and mirror | `download/huggingface_utils.py:61-80` |
| `HTTPS_PROXY` / `HTTP_PROXY`, `SSL_CERT_FILE` | download session | `download_utils.py:555-563` |

### 2.6 Ports, bind addresses and the dashboard

| Port | Proto | Bind | Purpose | Evidence |
|---|---|---|---|---|
| 52415 | TCP | `0.0.0.0` (IPv4 only, not configurable) | API + dashboard, **also used by peers for topology probes** | `listening_socket.py:4, 23-27` |
| 52414 | TCP | `tcp/[::]:52414` | zenoh router sessions | `rust/networking/src/lib.rs:31` |
| 52413 | UDP | `[::]:52413`, joins `ff12::e0a1:de89` on every interface with a non-loopback IPv6 address | discovery Hello/WhatsUp | `discovery.rs:19, 61-128` |
| random 49153-65535 | TCP | ring: self `0.0.0.0:<port>` | MLX ring backend, one port per instance (`ephemeralPort`) | `utils/ports.py:4-6`, `placement_utils.py:381-428` |
| random 49153-65535 | TCP | coordinator `0.0.0.0` | MLX JACCL coordinator (RDMA only) | `placement.py:386-391` |

Security posture (VERIFIED): API has `allow_origins=["*"]` CORS (`api/main.py:380-387`) and no auth;
zenoh config sets no TLS, auth or ACL (`lib.rs:23-52`); MLX ring sockets are plain TCP. Anyone who can
reach these ports can join the cluster, create/delete instances or trigger model downloads
(`POST /models/add`). The chat adapter also fetches `http(s)` image URLs server-side
(`adapters/chat_completions.py:82, 96`).

**Disabling the dashboard.** There is no flag. `API.__init__` mounts `StaticFiles(directory=dashboard_dir())`
at `/` (`api/main.py:308-315`); `find_dashboard()` raises `FileNotFoundError("Unable to locate dashboard
assets - you probably forgot to run ...")` when `dashboard/build/index.html` is missing
(`dashboard_path.py:34-40`). So:
- Option 1: build it (cheap, ~one `npm ci && npm run build`).
- Option 2 (INFERRED): set `EXO_DASHBOARD_DIR` to any existing empty directory. Starlette only checks the
  directory exists; `/` then returns 404 while API routes, registered first (`api/main.py:306-315`), work.
- `--no-api` is **not** a substitute on a worker node. Topology edges are created only when a worker reaches
  another node's API: `_poll_connection_updates` calls `check_reachable(..., api_port=self.api_port)` every
  10 s (`worker/main.py:569-612`), which does `GET http://<ip>:<api_port>/node_id` and checks the returned
  id (`utils/info_gatherer/net_profile.py:17-79`). A node without an API never gets inbound edges, so it can
  never be in a multi-node cycle (INFERRED). For the same reason **every node must use the same
  `--api-port`**.

### 2.7 File locations (macOS defaults; VERIFIED in `constants.py`)

| Path | Content |
|---|---|
| `~/.exo/models/<org>--<name>/` | model files (`ModelId.normalize()` replaces `/` with `--`, `common.py:33-42`) |
| `~/.exo/models/caches/<org>--<name>/` | metadata cache (`download_utils.py:235-239`) |
| `~/.exo/custom_model_cards/*.toml` | cards added by `POST /models/add` or fetched on demand |
| `~/.exo/exo_log/exo.log`, `~/.exo/exo_log/runner_log/` | logs |
| `~/.exo/exo.pid` | PID lock |
| `~/.exo/event_log/` | event log |

Node ids are **not persistent**: `get_node_zid` returns `NodeId(os.urandom(16).hex().lstrip("0"))` with the
comment "TODO(evan): bring back node id persistence" (`routing/router.py:248-256`). Hypercompute must map
machines to node ids at every start (use `GET http://<machine>:52415/node_id`, which returns that machine's
own id as a bare JSON string; `api/main.py:390`. `docs/api.md:17-28` wrongly calls it the master id and
shows an object).

### 2.8 How nodes discover each other (VERIFIED in Rust and Python)

1. Every second, each node sends a UDP `Hello { magic "EXO", kind 0, nonce[8], namespace[8] }` to
   `[ff12::e0a1:de89]:52413` on each interface with a non-loopback IPv6 address (`discovery.rs:19-20,
   282-322, 467-476`). `namespace[8]` is `blake3(namespace)[..8]` (`lib.rs:61-65`).
2. A node with the same 8 bytes replies unicast with `WhatsUp { nonce, zid[16], port_le }`
   (`discovery.rs:203-245`). A different namespace or version is dropped and logged at most every 10 min
   ("found another exo at ... that can't join this cluster", `discovery.rs:214-226`).
3. The node with the **greater** zid dials the other over zenoh TCP using the IPv6 link-local source address
   and the advertised port (`lib.rs:76-105`). zenoh runs in `router` mode with multicast scouting off and
   `scouting/gossip/multihop = true` (`lib.rs:29-35`).
4. zenoh liveliness tokens `live/<zid>` produce `Discovered` / `Expired` events (`swarm.rs:79-115`). zenoh's
   lease is 10 s (`discovery.rs:32` comment).
5. Election picks a master (every node starts with a master and an API; `main.py:138-162`).
6. Each worker reports memory/network/backends and probes peers' APIs to create directed topology edges
   (2.6). Placement needs a cycle (`topology.get_cycles()`, `shared/topology.py:214-224`): for two nodes,
   edges in both directions.

Consequences:
- Nodes must share one L2 segment where IPv6 link-local multicast passes (Wi-Fi client isolation or
  multicast filtering breaks it), plus IPv4 reachability for the API probes. Tailscale/VPN or internet
  clusters cannot form at this pin (no static peers, no unicast discovery).
- After a process stall of 5 s or more, discovery ignores peers for 12 s so stale sessions close
  (`discovery.rs:29-36, 325-352`).

### 2.9 Restricting a cluster to specific machines

| Layer | Mechanism | Strength |
|---|---|---|
| Cluster membership | unique `EXO_ZENOH_NAMESPACE` per Hypercompute cluster | isolation only: the 8-byte hash travels in cleartext and can be replayed (`discovery.rs:214`) |
| Cluster membership | host firewall allowlist: UDP 52413, TCP 52414, TCP 52415, TCP 49153-65535 only from peer IPs | the only real control (INFERRED) |
| Static peer list | none (`--bootstrap-peers` refused, `main.py:373-374`) | n/a |
| Per instance | `GET /instance/previews?model_id=..&node_ids=A&node_ids=B`, then `POST /instance` with the chosen preview | exact node set (section 4.4) |
| Per node role | `--no-worker` (never runs runners), `--no-downloads` | coarse |

---

## 3. Linux and WSL2 (CPU only)

### 3.1 GPU vs CPU at `bc412e2`

| Question | Answer | Evidence |
|---|---|---|
| Official Linux status | CPU only. "Currently, exo runs on CPU on Linux. GPU support for Linux platforms is under development." | README:247, 583 (VERIFIED) |
| Platform tiers | Tier 1 = Apple Silicon macOS only. "Linux CUDA Support" and "Linux CPU Support" are under **Planned** | `PLATFORMS.md:3-23` (VERIFIED) |
| CUDA code paths exist | extras `mlx-cuda12` / `mlx-cuda13` (fork wheels); a node advertises `MlxCuda` and `Vllm` if NVML finds a GPU | `pyproject.toml:63-72`; `info_gatherer.py:357-383` (VERIFIED) |
| Can a Linux GPU node be sized correctly? | No: Linux nodes report system RAM via psutil (`info_gatherer.py:451-452, 511-520`); open PR #2216 "report CUDA GPU VRAM as node memory on Linux" | VERIFIED (code) / REPORTED (PR) |
| RTX 5070 Ti (sm_120) under WSL2 via MLX CUDA | not attempted | UNVERIFIED; do not plan on it |
| CPU device selection | the `mlx-cpu` backend has no GPU, so MLX runs on CPU; Metal-only calls are guarded (`utils_mlx.py:805-806`) | INFERRED |
| Mixed Mac + Linux ring | placement allows it: `MlxRing` accepts `MlxMetal`, `MlxCuda`, `MlxCpu`, every node always lists `MlxCpu` (`placement.py:66-69, 279-300`; `info_gatherer.py:377`) | placement VERIFIED; runtime UNVERIFIED |
| Native Windows | impossible: `main.py:4` `import resource` (Unix-only); `[tool.uv] environments = ["sys_platform == 'darwin'", "sys_platform == 'linux'"]` (`pyproject.toml:188`); PLATFORMS.md "Longer term" | VERIFIED |
| CI on Linux | Nix builds for `x86_64-linux` and `aarch64-linux`; pytest runs on macOS only | `.github/workflows/pipeline.yml:17-23, 106-117` (VERIFIED) |

### 3.2 Linux CPU install is broken out of the box

- `uv.lock:1403-1411` resolves linux x86_64 `mlx` to the fork wheel
  `mlx-0.32.0-cp313-cp313-manylinux_2_35_x86_64.whl`; `uv.lock:1438-1444` resolves `mlx-cpu` to PyPI
  `0.31.2` (VERIFIED).
- Open issue #2359 (2026-09-30, REPORTED): with that pair, `import mlx.core` fails with
  `undefined symbol ... ThreadLocalStream ...`; bumping `mlx-cpu` to 0.32.0 alone gives another undefined
  symbol.
- Open issue #2382 (2026-10-02, REPORTED): `uv sync --extra mlx-cpu` needs `python3-dev` to build
  `miniaudio`, and after all workarounds "inference runs on a single CPU thread only".
- Open PR #2385 "fix: align Linux MLX CPU dependencies" (VERIFIED open, `mergeable=false`): pins
  `mlx-cpu==0.32.0` and restricts the fork `mlx` wheels to the CUDA extras so the CPU extra takes PyPI
  `mlx==0.32.0` + `mlx-cpu==0.32.0`. Its own test plan says full install and inference "were not tested".
- Open issue #2271 (2026-08-20, REPORTED): two Ubuntu nodes on one LAN never discover each other. It predates
  the 2026-10-12 discovery fixes; current status UNVERIFIED.

### 3.3 What a WSL2 Ubuntu 24.04 CPU node would need

1. `.wslconfig`: `[wsl2] networkingMode=mirrored` (Windows 11 22H2+). Microsoft lists mirrored-mode benefits
   as "IPv6 support", "Multicast support" and "Connect to WSL directly from your local area network (LAN)"
   (learn.microsoft.com/windows/wsl/networking, VERIFIED). Default NAT mode gives the VM its own address
   that the Mac cannot reach and no LAN multicast, so discovery cannot work there (INFERRED).
2. Hyper-V firewall: `Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
   -DefaultInboundAction Allow` or per-port `New-NetFirewallHyperVRule` rules (same MS page), plus Windows
   Defender Firewall rules, for UDP 52413, TCP 52414, 52415 and 49153-65535.
3. `.wslconfig` `memory=` to size the VM, and `OVERRIDE_MEMORY_MB` if EXO should report less than the VM has.
4. Packages: `python3-dev build-essential nodejs npm` (REPORTED in #2382 / PR #2385), `uv`, rustup stable.
   glibc 2.39 satisfies the `manylinux_2_35` wheels (INFERRED).
5. A local patch equivalent to PR #2385 in a quarantine checkout, then `uv lock` and
   `uv sync --locked --extra mlx-cpu`. This departs from the upstream lock (UNVERIFIED to work).
6. Same commit, same namespace, same `--api-port` as the Macs; API must stay enabled.
7. Whether IPv6 link-local multicast and zenoh's link-local TCP dial actually work through mirrored mode is
   UNVERIFIED (no WSL test was run).

**Verdict for Hypercompute:** a WSL2 CPU node is a plumbing test at best. In a pipeline ring with the M4 Air
it would be the slowest stage (and possibly single-threaded per #2382), which contradicts the
"low latency, high tokens/s" goal. Keep Windows/NVIDIA capacity on the llama.cpp RPC path; use EXO only for
Mac-only instances.

---

## 4. HTTP API (`src/exo/api/main.py` @ `bc412e2`)

### 4.1 JSON conventions (important for the adapter)

- Cluster types derive from `FrozenModel` with `alias_generator=to_camel`, `validate_by_name=True`,
  `extra="forbid"`, `strict=True` (`utils/pydantic_ext.py:11-18`). Through FastAPI they serialize
  **camelCase** (`instanceId`, `shardAssignments`, `nodeToRunner`, `inBytes`). The official harness reads
  them that way (`tools/src/exo_tools/harness.py:48-66, 552-568`).
- Union members derive from `TaggedModel` and serialize as `{"ClassName": {...}}`
  (`pydantic_ext.py:21-33`), for example `{"MlxRingInstance": {...}}`, `{"RunnerReady": {...}}`,
  `{"DownloadOngoing": {...}}`, `{"PipelineShardMetadata": {...}}`.
- API request/response types in `api/types/api.py` are plain `BaseModel` (snake_case):
  `model_id`, `instance_meta`, `memory_delta_by_node`, `command_id`, `model_card`.
- So responses mix both: a `PlacementPreview` is snake_case on the outside and camelCase + tagged inside
  `instance`. The `/instance/await` SSE payload uses `model_dump_json()` without `by_alias`, so its nested
  instance is probably **snake_case** (INFERRED). Do not parse it; read `/state`.
- Errors: `{"error": {"message": str, "type": str, "param": null, "code": int}}` (`api/main.py:368-378`,
  `api.py:21-29`).

### 4.2 Route table (registrations `api/main.py:390-455`, VERIFIED)

| Method | Path | Handler line | Use in Hypercompute |
|---|---|---|---|
| GET | `/node_id` | 390 (lambda) | map machine to node id |
| GET | `/state`, `/state/{path:path}` | 443-444 -> 457-473 | cluster state, sub-paths by camelCase key |
| GET | `/events` | 445 -> 1070-1078 | debug: recent applied events (JSON array, up to 10,000) |
| GET | `/models`, `/v1/models` | 403-404 -> 1843-1876 | model catalogue with sizes; `?status=downloaded` |
| POST | `/models/add` | 405 -> 1878-1908 | do not use (arbitrary HF repos) |
| DELETE | `/models/custom/{model_id:path}` | 406 | |
| GET | `/models/search` | 407 | |
| GET | `/instance/previews` | 394 -> 564-688 | placement preview, supports `node_ids` |
| GET | `/instance/placement` | 393 -> 523-562 | single placement, no `node_ids` |
| POST | `/instance` | 391 -> 498-521 | create from a preview (preferred) |
| POST | `/place_instance` | 392 -> 475-496 | server-side placement (no instance id returned) |
| GET | `/instance/await` | 395 -> 695-735 | existence wait only |
| GET | `/instance/{instance_id}` | 396 -> 690-693 | 404 when gone |
| DELETE | `/instance/{instance_id}` | 397 -> 737-749 | delete |
| POST | `/v1/chat/completions` | 408-410 -> 973-1005 | chat (SSE or JSON) |
| POST | `/bench/chat/completions` | 411-413 -> 1007-1042 | benchmark variant with `generation_stats` |
| POST | `/v1/cancel/{command_id}` | 424 -> 794-812 | cancel a generation |
| POST | `/download/start` | 446 -> 2255-2263 | pre-download a shard on a node |
| DELETE | `/download/{node_id}/{model_id:path}` | 447 -> 2265-2277 | delete files on a node |
| POST | `/download/cancel` | 448 -> 2279-2288 | cancel a download |
| POST | `/v1/messages`, `/v1/responses`, `/ollama/...` | 422-441 | not needed |

### 4.3 Listing models and sizes

`GET /models[?status=downloaded]` returns `ModelList` (`api.py:32-56`):

```python
class ModelListModel(BaseModel):
    id: str
    object: str = "model"
    ...
    context_length: int = Field(default=0)
    storage_size_megabytes: int = Field(default=0)
    supports_tensor: bool = Field(default=False)
    tasks: list[str] = Field(default=[])
    is_custom: bool = Field(default=False)
    family / quantization / base_model: str
    capabilities: list[str]
    reasoning_dialect: ReasoningDialect = "none"
```

- `storage_size_megabytes = card.storage_size.in_mb` = `round(bytes / 1024**2)` (`api/main.py:1863`,
  `memory.py:35-38`). It is the weights size from the model card, the same number placement uses.
- `?status=downloaded` keeps models with a `DownloadCompleted` on **any** node (`api/main.py:1847-1853`).
- Cards come from `resources/inference_model_cards/*.toml` plus `~/.exo/custom_model_cards`
  (`model_cards.py:38-41, 94-97`). Image cards are hidden unless `EXO_ENABLE_IMAGE_MODELS=true`.
- Exact bytes per model: `GET /state` does not hold cards for built-ins; use the `model_card.storageSize.inBytes`
  inside a placement preview instance (`runnerToShard.*.modelCard`) or the `model_card` echoed by
  `POST /instance`.
- `ModelCard.load` on an unknown id falls back to fetching `config.json` and `model.safetensors.index.json`
  from HF and saves a custom card (`model_cards.py:229-238`), so a typo in `model_id` triggers network I/O.

`ModelCard` fields (`model_cards.py:162-182`): `model_id, storage_size{in_bytes}, n_layers, hidden_size,
supports_tensor, num_key_value_heads, tasks, components, family, quantization, base_model, capabilities,
backends, reasoning_dialect, context_length, uses_cfg, trust_remote_code = True, is_custom, vision,
sampling_defaults`.

### 4.4 Placement preview

`GET /instance/previews?model_id=<id>[&node_ids=<A>&node_ids=<B>...]`

```python
async def get_placement_previews(self, model_id: ModelId,
        node_ids: Annotated[list[NodeId] | None, Query()] = None) -> PlacementPreviewResponse:
```

Response (`api.py:280-291`):

```python
class PlacementPreview(BaseModel):
    model_id: ModelId
    sharding: Sharding            # "Pipeline" | "Tensor"
    instance_meta: InstanceMeta   # "MlxRing" | "MlxJaccl"
    instance: Instance | None = None
    memory_delta_by_node: dict[str, int] | None = None
    error: str | None = None
class PlacementPreviewResponse(BaseModel):
    previews: list[PlacementPreview]
```

Behaviour (VERIFIED, `api/main.py:564-688`; `master/placement.py:191-413`):
- Empty topology returns `{"previews": []}`.
- It tries every `(sharding, instance_meta, min_nodes)` for `min_nodes` in 1..N and dedupes by node count.
- `node_ids` becomes `required_nodes`: only cycles that contain **all** of them are considered
  (`placement.py:220-228`, error "The chosen nodes aren't all connected to each other"). Among fitting cycles
  the **smallest** are kept (`placement_utils.py:40-44`), so `node_ids=[A,B]` yields exactly `{A,B}` when they
  form a cycle, and `node_ids=[A]` with a small model yields a single-node instance on A.
- A single-node result is silently forced to `Pipeline` + `MlxRing` (`placement.py:347-354`) although the
  preview keeps the requested label. Check the instance tag (`MlxRingInstance`) and shard tag
  (`PipelineShardMetadata`), not the label.
- `memory_delta_by_node` is an **even split** of `storage_size` by node count (`api/main.py:654-661`), not the
  real layer split. The real per-node weight bytes are `storage * (end_layer - start_layer) / n_layers`
  (`placement.py:151-158`); layers are split in proportion to each node's `ram_available`
  (`placement_utils.py:47-75, 96-122`).
- Memory checks use weights only (`filter_cycles_by_memory`, `placement_utils.py:21-37`), no KV-cache headroom.
  Loading runners' memory is counted as used (`placement.py:161-188`).
- Error strings worth surfacing: "No nodes are available", "Needs N connected nodes, but at most M are
  connected to each other", "Waiting for the nodes to report their memory", "Not enough memory: ... needs X GB,
  but connected nodes have at most Y GB free between them", "No cycle where every node supports a backend in
  [...]", "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available" (`placement.py:121-137, 210-320`).

`GET /instance/placement?model_id=&sharding=Pipeline&instance_meta=MlxRing&min_nodes=1` returns one
`Instance` or 400 with the reason (`api/main.py:523-562`). It has no `node_ids`.

Instance shape (`shared/types/worker/instances.py:15-39`, `runners.py:86-98`, `shards.py:10-84`), as JSON:

```json
{"MlxRingInstance": {
  "instanceId": "uuid",
  "shardAssignments": {
    "modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "runnerToShard": {"<runnerId>": {"PipelineShardMetadata": {
        "modelCard": {...}, "deviceRank": 0, "worldSize": 2,
        "startLayer": 0, "endLayer": 9, "nLayers": 16, "immediateException": false, "shouldTimeout": null}}},
    "nodeToRunner": {"<nodeId>": "<runnerId>"}},
  "hostsByNode": {"<nodeId>": [{"ip": "0.0.0.0", "port": 51234}, {"ip": "192.168.1.20", "port": 51234}]},
  "ephemeralPort": 51234}}
```

(Field names VERIFIED from the models and the harness; exact values illustrative. `MlxJacclInstance` has
`jacclDevices` and `jacclCoordinators` instead of `hostsByNode`/`ephemeralPort`.)

### 4.5 Creating an instance

Preferred: `POST /instance` with body `{"instance": <preview.instance verbatim>}` (`CreateInstanceParams`,
`api.py:276-277`). This is what the dashboard does (`dashboard/src/routes/+page.svelte:1637-1643`).

```python
async def create_instance(self, payload: CreateInstanceParams) -> CreateInstanceResponse:
    ...
    if required_memory > available_memory:
        raise HTTPException(status_code=400, detail=f"Insufficient memory to create instance. ...")
    command = CreateInstance(instance=instance)
```

- The only check is total model size vs the **sum** of `ram_available` over all nodes
  (`api/main.py:498-521, 1834-1841`). The master adds the instance as given, `# TODO: validate against
  topology` (`placement.py:72-79`; `master/main.py:505-514`), so the preview's `instanceId` is the id that
  will appear in state. Do not post the same preview twice.
- Response `CreateInstanceResponse {message: "Command received.", command_id, model_card}` (`api.py:298-301`).

Alternative: `POST /place_instance` with `PlaceInstanceParams {model_id, sharding="Pipeline",
instance_meta="MlxRing", min_nodes=1}` (`api.py:269-273`). The API validates with `get_placement` (400 with
reason) and the master recomputes placement independently (`master/main.py:489-504`); the response has no
instance id, so callers must diff `/state/instances` (the harness does this, `harness.py:575-627`). It cannot
pin nodes.

### 4.6 Cluster state

`GET /state` returns `State` (`shared/types/state.py:30-72`). camelCase keys:

| Key | Type | Use |
|---|---|---|
| `instances` | `{instanceId: Instance}` | tagged instances |
| `runners` | `{runnerId: RunnerStatus}` | tagged statuses (below) |
| `downloads` | `{nodeId: [DownloadProgress]}` | per-node download progress |
| `tasks` | `{taskId: Task}` | generation and bookkeeping tasks |
| `lastSeen` | `{nodeId: datetime}` | liveness |
| `topology` | `{"nodes": [nodeId], "connections": {src: {sink: [{"sinkMultiaddr": ...}]}}}` | `TopologySnapshot` (`shared/topology.py:23-29`) |
| `nodeIdentities` | `{nodeId: {modelId, chipId, friendlyName, osVersion, osBuildVersion}}` | `profiling.py:78-85` |
| `nodeMemory` | `{nodeId: {ramTotal, ramAvailable, swapTotal, swapAvailable}}` each `{inBytes}` | `profiling.py:13-17` |
| `nodeDisk` | `{nodeId: {total, available}}` | |
| `nodeSystem` | `{nodeId: {gpuUsage, temp, sysPower, pcpuUsage, ecpuUsage}}` | |
| `nodeNetwork` | `{nodeId: {interfaces: [{name, ipAddress, interfaceType}]}}` | |
| `nodeBackends` | `{nodeId: ["MlxCpu", "MlxMetal", ...]}` | `backends.py:4-8` |
| `nodeRdmaCtl`, `nodeThunderbolt`, `nodeThunderboltBridge`, `thunderboltBridgeCycles` | RDMA / TB | |
| `customModelCards`, `instanceLinks`, `prefillServerPorts`, `lastEventAppliedIdx` | | |

`GET /state/<a>/<b>/...` walks `state.model_dump(by_alias=True)` (`api/main.py:457-473`), for example
`/state/instances/<iid>`, `/state/runners/<rid>`, `/state/downloads/<nodeId>`, `/state/nodeMemory`; a missing
key returns 404 `unable to find path 'a.b' in state json`. Sub-path reads are separate requests and not
atomic with each other.

Runner statuses (`runners.py:25-83`) and their meaning from the worker planner (`worker/plan.py:45-323`):

| Tag | Fields | Meaning | Hypercompute phase |
|---|---|---|---|
| `RunnerIdle` | | runner process created; download runs while Idle (`plan.py:146-163`) | `downloading` if `downloads` has Pending/Ongoing for this model on this node, else `starting` |
| `RunnerConnecting` / `RunnerConnected` | | MLX distributed group init, multi-node only (`plan.py:166-213`) | `connecting` |
| `RunnerLoading` | `layersLoaded`, `totalLayers` | weights loading (`utils_mlx.py:177-184`) | `loading` (progress bar) |
| `RunnerLoaded` | | weights in memory | `loading` |
| `RunnerWarmingUp` | | built-in 50-token warmup (`runner.py:277-292`, `generate.py:401-418`) | `warming` |
| `RunnerReady` | `prefillServerPort` | can take tasks | `ready` |
| `RunnerRunning` | | generating | `ready` (busy) |
| `RunnerShuttingDown` / `RunnerShutdown` | | stopping; `RunnerShutdown` removes the runner from state (`apply.py:305-317`) | `stopping` |
| `RunnerFailed` | `errorMessage`, `diagnostics` | process died, lost peers, or stalled (`supervisor.py:391-512`) | `failed` |

Tasks are dispatched only when **all** runners of the instance are `RunnerReady` or `RunnerRunning`
(`plan.py:319-323`). The official harness waits for `"RunnerReady" in runner` for every runner id in
`runnerToShard` and treats `RunnerFailed` as fatal (`harness.py:69-124`).

### 4.7 `/instance/await`

`GET /instance/await?model_id=<id>&timeout_seconds=<0..300>` streams SSE and emits
`data: {"type":"ready","instance":{...}}` as soon as **any** instance with that model id exists in API state,
or `{"type":"timeout","message":"No instance found for model ..."}` (`api/main.py:695-735`,
`api.py:310-317`). It does not look at runners or downloads. Use it only as "placed".

### 4.8 Deleting

`DELETE /instance/{instance_id}`: 404 `Instance not found` if absent, else
`DeleteInstanceResponse {message: "Command received.", command_id, instance_id}` (`api/main.py:737-749`,
`api.py:304-307`). The master cancels pending/running tasks of that instance and emits `InstanceDeleted`
(`placement.py:427-464`); download coordinators cancel now-unneeded downloads (`master/main.py:475-488`);
workers shut runners down (`plan.py:71-79`). Confirm with `GET /instance/{id}` returning 404 (the harness
allows 30 s, `harness.py:127-140, 630-637`).

### 4.9 Chat completions

Request `ChatCompletionRequest` (`api.py:223-249`): `model` (must equal an instance's model id exactly,
`api/main.py:1044-1068`), `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `stop`, `seed`,
`stream`, `stream_options`, `enable_thinking`, `reasoning_effort`, `tools`, `tool_choice`, `logprobs`,
`top_logprobs`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `n`, `user`.
`max_tokens` maps to `max_output_tokens` (`adapters/chat_completions.py:157`); always send it.

Errors before streaming: 404 `{"error":{"message":"No instance found for model <id>",...}}`.

Streaming response (`stream: true`), `text/event-stream`, headers `Cache-Control: no-cache`,
`Connection: close`, `X-Accel-Buffering: no` (`api/main.py:983-997`). Event sequence
(`api/keepalive.py:9-34`; `adapters/chat_completions.py:218-292`):

```text
: keep-alive                                   <- sent immediately, then every 10 s of silence
: prefill_progress {"PrefillProgressChunk":{...processed_tokens, total_tokens...}}   (comment, INFERRED key case)
data: {"id":"<command_id>","object":"chat.completion","created":...,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}
data: {... "delta":{"role":"assistant","reasoning_content":"..."} ...}          <- thinking tokens
data: {... "choices":[{"index":0,"delta":{...},"finish_reason":"stop"}],"usage":{"prompt_tokens":..,"completion_tokens":..,"total_tokens":..,...}}
: generation_stats {"prompt_tps":..,"generation_tps":..,"prompt_tokens":..,"generation_tokens":..,"peak_memory_usage":{...},"prefix_cache_hit":"none"}   (only if stats were attached)
data: [DONE]
```

- `id` is the EXO `command_id`; use it for `POST /v1/cancel/{id}`. Closing the HTTP connection also cancels
  (`TaskCancelled` in the stream cleanup, `api/main.py:844-848`).
- `object` is `"chat.completion"` even for chunks (`api.py:155`), not OpenAI's `chat.completion.chunk`.
- `usage` is attached to the final chunk regardless of `stream_options` (`chat_completions.py:279-286`).
- Tool calls arrive as one `delta.tool_calls` chunk with `finish_reason: "tool_calls"`, then `[DONE]`.
- Errors mid-stream: `data: {"error":{"message":"...","type":"InternalServerError","code":500}}` then
  `data: [DONE]` (`chat_completions.py:233-243`). HTTP status stays 200.
- Non-streaming: one JSON `ChatCompletionResponse`; on a generation error the body is an `ErrorResponse` with
  HTTP 500 (`api/collected_response.py:10-42`).
- `GenerationStats` is created on the last token in the MLX generator (`generate.py:761-768`); whether the
  default (batched) path forwards it on `/v1/chat/completions` is UNVERIFIED; `/bench/chat/completions`
  returns it as `generation_stats` in JSON (`api/main.py:1007-1042`).
- Acceptance: if no `TaskCreated`/first chunk is seen, the API resends after 10, 15 and 30 s, then emits the
  error "The cluster didn't accept the request. Please check that the model is running and try again."
  (`api/main.py:232-237, 2208-2244`). A request sent while runners are still loading is accepted and waits
  as a pending task (`master/main.py:307-352`).
- If two instances serve the same model, the master picks the one with the fewest in-flight tasks, ignoring
  readiness (`master/main.py:315-345`). Keep one instance per model id.

### 4.10 Download progress

`GET /state/downloads/<nodeId>` returns a list of tagged items (`shared/types/worker/downloads.py:12-52`):

| Tag | Fields (camelCase) |
|---|---|
| `DownloadPending` | `nodeId`, `shardMetadata`, `modelDirectory`, `downloaded{inBytes}`, `total{inBytes}` |
| `DownloadOngoing` | ... `downloadProgress{total, downloaded, downloadedThisSession, completedFiles, totalFiles, speed, etaMs, files{...}}` |
| `DownloadCompleted` | ... `total{inBytes}`, `readOnly` |
| `DownloadFailed` | ... `errorMessage` |

Match on `shardMetadata.<Tag>.modelCard.modelId`. Downloads start automatically when an instance is created
(`plan.py:135-163`; `worker/main.py:389-430`). An already complete local copy short-circuits to
`DownloadCompleted` (`worker/main.py:393-414`). Explicit pre-download: `POST /download/start` with
`{"targetNodeId": "<node>", "shardMetadata": <tagged shard from a preview>}` (camelCase, as the dashboard
sends it, `dashboard/src/lib/stores/app.svelte.ts:3300-3310`); response `{"commandId": "..."}` (INFERRED key
case). **Each node downloads the full repo**: `resolve_allow_patterns` returns `["*"]` with the comment
"'Smart' downloads are disabled" (`download_utils.py:845-850`).

---

## 5. Small test models and storage

Built-in cards at the pin (`resources/inference_model_cards/`, VERIFIED) and HF API facts (VERIFIED
2026-10-21):

| Model id | Bytes (card) | `/models` MB | Layers | hidden / kv heads | Tensor | HF license, gated | Repo files |
|---|---|---|---|---|---|---|---|
| `mlx-community/Qwen3-0.6B-4bit` | 342,884,352 | 327 | 28 | 1024 / 8 | false | apache-2.0, not gated | 11 files, no `.py` |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | 729,808,896 | 696 | 16 | 2048 / 8 | true | llama3.2, not gated | 8 files, no `.py` |
| `mlx-community/Qwen3.5-2B-MLX-8bit` | 2,662,787,264 | 2539 | 24 | 2048 / 2 | true | apache-2.0, not gated | 12 files |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | 4,637,851,648 | 4423 | 32 | 4096 / 8 | true | llama3.1, not gated | 8 files |

All four list `backends = ["MlxMetal", "MlxCuda", "MlxCpu"]`. Qwen3 cards declare
`capabilities = ["text", "thinking", "thinking_toggle"]`; send `"enable_thinking": false` for latency tests
(mapping via `resolve_reasoning_params`, INFERRED). README uses `Llama-3.2-1B-Instruct-4bit` as its example.

Recommended Hypercompute smoke tests:
- single Mac: `mlx-community/Qwen3-0.6B-4bit` (fastest download, 327 MB);
- two Macs, pipeline: `mlx-community/Llama-3.2-1B-Instruct-4bit` with `node_ids=A,B` (16 layers, tensor
  capable too).

Storage: `~/.exo/models/mlx-community--Qwen3-0.6B-4bit/` on macOS, `~/.local/share/exo/models/...` on Linux,
first writable dir in `EXO_MODELS_DIRS` with space, or a pre-staged copy under `EXO_MODELS_READ_ONLY_DIRS`
(searched first, `download_utils.py:152-178`). Every node of an instance stores the whole repo.

Security: `ModelCard.trust_remote_code` defaults to `True` (`model_cards.py:179`), only cards fetched by
`fetch_from_hf` set it to `False` (`model_cards.py:261`). Hypercompute should allowlist model ids and never
expose `POST /models/add`.

---

## 6. Failure semantics

There is no instance-level status field. Failure shows up as (a) runner statuses, (b) the instance
disappearing, (c) stream error objects.

### 6.1 A node leaves mid-generation (crash, sleep, cable/Wi-Fi drop)

| When (approx.) | What happens | Evidence |
|---|---|---|
| t + up to 5 s | The surviving rank's supervisor sees an MLX ring transport error in runner diagnostics and stops the runner: in-flight tasks get `ErrorChunk("Runner shutdown before completing command (<cause>)")`, status `RunnerFailed` | `supervisor.py:58-64, 391-430, 432-512` |
| t + 90 s (fallback) | if rank 0 is generating and silent: "Runner made no progress for 90s while generating" | `supervisor.py:62-64, 416-424` |
| t + 10 s | zenoh lease expires, liveliness `Expired` | `discovery.rs:32`, `swarm.rs:79-93` |
| t + 15 s (+1 s check) | master: node silent > `NODE_SILENCE_TIMEOUT` (15 s) -> `NodeTimedOut`; every instance with a runner on a node outside the connected set -> `InstanceDeleted` | `master/main.py:87-100, 620-663` |
| on `InstanceDeleted` | instance, its runners and bookkeeping tasks leave state (`apply.py:242-287`); node's memory/network/disk entries and topology node go (`apply.py:341-406`); the API closes the request streams of that instance (`api/main.py:2074-2075, 2096-2108`) | VERIFIED |
| stream end | `_token_chunk_stream` yields `ErrorChunk("The model instance serving this request stopped before it finished")` (`api/main.py:227-229, 836-842`), so SSE ends with `data: {"error":{...}}` + `data: [DONE]` | VERIFIED |
| afterwards | no re-placement. The upstream resilience test deletes all instances and places a new 1-node instance by hand (`tests/test_resilience.py:30-45`) | VERIFIED |

If the node that left was the **master**, a new election runs; nodes "follow the new master without tearing
anything down" (`main.py:228-235`) but the API ends all in-flight requests (`api/main.py:346-356`). If the
node that left hosts the API Hypercompute talks to, the HTTP connection simply breaks. Talk to the API of the
most stable node (the Mac that owns the deployment).

### 6.2 A runner fails while its node stays up

`RunnerFailed{errorMessage, diagnostics}` is reported (`supervisor.py:497-507`). Peers' runners of the same
instance are shut down (`plan.py:86-96`). The worker recreates its runner with exponential backoff
(`KeyedBackoff(base=0.5, cap=10.0)`: 0.5, 1, 2, 4, 8 s; `worker/main.py:114-117`, `utils/keyed_backoff.py`)
and after `EXO_MAX_INSTANCE_RETRIES = 5` attempts asks the master to delete the instance
(`constants.py:114`, `worker/main.py:346-367`). Attempts reset when the runner reaches `RunnerReady`
(`worker/main.py:192-200`). `RunnerFailed` is therefore **transient** in state (replaced by `RunnerIdle` on
retry); a poller can miss it. The durable signal is the instance's 404.

### 6.3 Detecting failure from the API

| Signal | Call |
|---|---|
| instance gone (node left, retries exhausted, someone deleted it) | `GET /state/instances/<iid>` or `GET /instance/<iid>` -> 404 |
| runner failed (with reason) | `GET /state/runners/<rid>` -> `{"RunnerFailed": {"errorMessage": ..., "diagnostics": [...]}}` |
| download failed | `GET /state/downloads/<nodeId>` -> `DownloadFailed.errorMessage` |
| node gone or stale | `GET /state/topology` (`nodes`), `GET /state/lastSeen` (age > 15 s) |
| request failed | SSE `data: {"error":...}` before `[DONE]`; JSON mode HTTP 500 `ErrorResponse` |
| history | `GET /events` (recent `RunnerStatusUpdated`, `InstanceDeleted`, `NodeTimedOut`; JSON key case INFERRED) |

---

## 7. Adapter recipe for Hypercompute

Assumptions: Hypercompute talks to the API of the deployment's primary Mac, `BASE=http://<mac-ip>:52415`
(LAN-only, firewall-restricted). All nodes run `bc412e2` with the same `EXO_ZENOH_NAMESPACE` and
`--api-port 52415`. Model ids come from an allowlist. Every call has a timeout; every 4xx/5xx body is parsed
as `{"error":{"message":...}}`.

Helper: `untag(x) = next(iter(x.values()))`, `tag(x) = next(iter(x))` for `{"ClassName": {...}}` values.

### (a) Check cluster nodes

```text
for each machine m Hypercompute wants in the instance:
    GET http://<m.ip>:52415/node_id              -> "<nodeId>"   (bare JSON string; ids change on every exo restart)
GET BASE/state/topology                          -> {"nodes":[...], "connections":{src:{sink:[...]}}}
GET BASE/state/nodeMemory                        -> {nodeId:{"ramAvailable":{"inBytes":N},...}}
GET BASE/state/nodeBackends                      -> {nodeId:["MlxCpu","MlxMetal"]}
GET BASE/state/nodeIdentities                    -> {nodeId:{"friendlyName":...,"chipId":...}}
```

Pass when every wanted node id is in `topology.nodes`, has a `nodeMemory` entry (placement needs it,
`harness.py:552-572`), lists `MlxMetal` (Apple-only policy), and for N > 1 there are connections in both
directions between consecutive nodes. Otherwise report "waiting for nodes" and retry with backoff; a fresh
node usually needs a few seconds for discovery plus up to 10 s for the first edge probe
(`worker/main.py:612`).

### (b) Preview placement on specific nodes

```text
GET BASE/instance/previews?model_id=<M>&node_ids=<A>&node_ids=<B>
```

Pick the first preview with `error == null`, `instance_meta == "MlxRing"`,
`tag(instance) == "MlxRingInstance"`, `sharding == "Pipeline"` (or `"Tensor"` when wanted and N > 1),
`set(untag(instance).shardAssignments.nodeToRunner) == {A, B}`, and every shard tag matching the sharding.
If none qualifies, show the `error` strings (section 4.4). Compute the real per-node weight bytes from each
shard's `startLayer/endLayer/nLayers` and `modelCard.storageSize.inBytes`, not `memory_delta_by_node`.

Optional: pre-download so "download" and "load" are separate phases:
`POST BASE/download/start {"targetNodeId": node, "shardMetadata": runnerToShard[nodeToRunner[node]]}` per
node, then poll `GET BASE/state/downloads/<node>` until `DownloadCompleted` for M (or `DownloadFailed`).

### (c) Create the instance

```text
GET  BASE/state/instances                        -> refuse if another instance already serves <M>
POST BASE/instance   {"instance": <chosen preview.instance, verbatim>}
     200 {"message":"Command received.","command_id":"...","model_card":{...}}
     400 {"error":{"message":"Insufficient memory ..."}}
iid  = untag(preview.instance).instanceId
rids = keys(untag(preview.instance).shardAssignments.runnerToShard)
```

Phase = `acknowledged`.

### (d) Wait until truly ready

Poll every 0.5 s (one `GET BASE/state` per tick gives an atomic view; sub-paths are fine if state is large):

```text
inst = state.instances[iid]
  missing and never seen   -> still "acknowledged"; give up after ~30 s (command lost) and retry (c) once
  missing after being seen -> FAILED ("instance deleted": node left or 5 runner retries exhausted)
for rid in rids:
  st = state.runners.get(rid)          # absent = not reported yet
  RunnerFailed                -> FAILED(errorMessage, diagnostics); DELETE the instance (do not wait for 5 retries)
  RunnerIdle                  -> "downloading" if state.downloads[node] has Pending/Ongoing for M (progress = downloaded/total), else "starting"
  RunnerConnecting|Connected  -> "connecting"
  RunnerLoading               -> "loading" layersLoaded/totalLayers
  RunnerLoaded|RunnerWarmingUp-> "warming"
READY when every rid is RunnerReady or RunnerRunning
```

Deadlines: download budget from model size and measured speed (`downloadProgress.speed`, `etaMs`); load
budget 400 s, matching the supervisor's `initialize_timeout` default (`supervisor.py:216`). Do not rely on
`GET /instance/await`; at most use it as the "placed" signal.

### (e) Warm up

EXO already ran a 50-token greedy warmup per runner before `RunnerReady`. Hypercompute's own warmup checks
the full API -> master -> runner path and records TTFT:

```text
POST BASE/v1/chat/completions
{"model":"<M>","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0,"stream":false,"enable_thinking":false}
200 -> ok (record wall time); 404 -> instance vanished, go to FAILED; 500 -> error body, FAILED
```

### (f) Stream chat

```text
POST BASE/v1/chat/completions   {..., "stream": true, "max_tokens": <required>}
read lines:
  ""                       -> event boundary
  ":" prefix               -> comment: "keep-alive" | "prefill_progress {...}" | "generation_stats {...}" (optional metrics)
  "data: [DONE]"           -> end
  "data: {json}"           -> if "error" in json: FAILED(json.error.message), expect [DONE] next
                              else first time: save json.id as command_id
                                   delta.content / delta.reasoning_content / delta.tool_calls
                                   finish_reason (+ usage) on the last data chunk
```

Measure TTFT at the first `data:` with non-empty content, not at the first byte (the first byte is the
immediate `: keep-alive`). Client-side idle timeout: about 35 s with no line at all (keep-alives arrive every
10 s). User cancel: `POST BASE/v1/cancel/<command_id>` or close the connection. In parallel, poll
`GET BASE/state/instances/<iid>` every 2-5 s during serving; a 404 means the deployment is lost.

### (g) Delete the instance

```text
DELETE BASE/instance/<iid>      -> 200 {"message":"Command received.","command_id":"...","instance_id":"<iid>"} | 404 already gone
poll GET BASE/instance/<iid> every 0.5 s until 404 (30 s budget)
optional: confirm no runner of rids remains in GET BASE/state/runners and nodeMemory.ramAvailable recovers
optional: DELETE BASE/download/<nodeId>/<M> to reclaim disk (files are kept by default)
```

---

## 8. Hypercompute-specific conclusions

- EXO is Mac-only for the adapter. With today's hardware (one M4 MacBook Air 16 GB), an EXO deployment is a
  single-node instance; multi-node EXO needs two or more Macs on the same LAN.
- The Windows laptop cannot be an EXO node, and a WSL2 CPU node would slow a Mac ring. Cross-OS layer split
  stays on llama.cpp RPC.
- Run EXO under the Hypercompute agent from the pinned source build (not the libp2p-era app), with a per-cluster
  namespace and a firewall allowlist; never expose 52413-52415 or the ring port range beyond trusted peers.
- The adapter's "ready" must come from `/state/runners`, its "failed" from `RunnerFailed` or the instance's
  404, and it must expect an SSE error object rather than an HTTP error for mid-stream failures.
- Keep one EXO instance per model id, always send `max_tokens`, and allowlist model ids
  (`trust_remote_code` defaults to true for built-in cards).

---

## 9. UNVERIFIED items

| Item | Reason |
|---|---|
| Exact macOS minimum for the source build; whether the Metal toolchain download is needed on the friend's Xcode | not run; inferred from README and CI |
| `EXO_DASHBOARD_DIR=<empty dir>` keeps the API working without a built dashboard | Starlette behaviour inferred, not run |
| A node with `--no-api` can never join a multi-node instance | inferred from the probe code |
| Snake_case payload inside `/instance/await` and camelCase `commandId` from `/download/start` | inferred from pydantic config, not observed |
| Whether `generation_stats` appears on `/v1/chat/completions` streams | batch path not traced end to end |
| Linux CPU node works after the PR #2385 dependency fix; multi-core use | upstream reports say no or untested; nothing run |
| WSL2 mirrored mode passes the `ff12::` discovery multicast and zenoh's link-local dial | no WSL test |
| Mixed Metal + CPU MLX ring at runtime | placement allows it; not run |
| Linux + CUDA (including sm_120) through `mlx-cuda12/13` | not supported per README; not tested |
| Two nodes of different commits but the same `0.3.70` version string interoperate | not tested; avoid by namespace |
| App requirement macOS 26.2 (README) vs cask `macos >= 15` | conflicting sources |

---

## 10. Evidence index (all at `bc412e2` unless noted)

Read via `gh api` (raw): `README.md`, `PLATFORMS.md`, `docs/api.md`, `pyproject.toml`, `uv.lock` (grep),
`Cargo.toml`, `Cargo.lock` (grep), `justfile`, `flake.nix` (grep), `python/parts.nix` (grep),
`rust/parts.nix` (grep), `rust/exo_rs/{pyproject.toml,src/networking.rs}`,
`rust/networking/{Cargo.toml,src/lib.rs,src/swarm.rs,src/discovery.rs}`,
`src/exo/{__init__.py,main.py}`, `src/exo/api/{main.py,listening_socket.py,keepalive.py,collected_response.py}`,
`src/exo/api/types/api.py`, `src/exo/api/adapters/chat_completions.py`,
`src/exo/shared/{constants.py,apply.py,topology.py}` (parts), `src/exo/shared/models/model_cards.py`,
`src/exo/shared/types/{state,common,memory,profiling,backends,topology,chunks}.py`,
`src/exo/shared/types/worker/{instances,runners,shards,downloads}.py`,
`src/exo/master/{main.py,placement.py,placement_utils.py}` (parts), `src/exo/worker/{main.py,plan.py}` (parts),
`src/exo/worker/runner/{supervisor.py,runner.py}` (parts), `src/exo/worker/engines/mlx/utils_mlx.py` (parts),
`src/exo/worker/engines/mlx/generator/generate.py` (parts),
`src/exo/worker/runner/llm_inference/batch_generator.py` (grep),
`src/exo/download/{download_utils.py,huggingface_utils.py}` (parts),
`src/exo/utils/{dashboard_path.py,ports.py,keyed_backoff.py,pydantic_ext.py}`,
`src/exo/utils/info_gatherer/{info_gatherer.py,system_info.py,net_profile.py}` (parts),
`src/exo/routing/router.py` (parts), `src/exo/shared/tests/{test_namespace.py,test_dashboard_lookup.py}`,
`tests/test_resilience.py`, `tools/src/exo_tools/{client.py,harness.py}`,
`resources/inference_model_cards/` (4 cards), `app/EXO/EXO/ExoProcessController.swift` (parts),
`app/EXO/EXO.xcodeproj/project.pbxproj` (grep), `.github/workflows/{pipeline.yml,build-app.yml}` (parts),
`dashboard/src/lib/stores/app.svelte.ts` and `dashboard/src/routes/+page.svelte` (grep).
Other: `ml-explore/mlx@0e3ff36 pyproject.toml, setup.py` (grep); `rust/networking/Cargo.toml@fd707de`;
issues #2359, #2382, #2271 and PR #2385, #2216, #2404 (GitHub API); HF model API for the four models;
`formulae.brew.sh/api/cask/exo.json`; `HEAD https://assets.exolabs.net/EXO-latest.dmg`;
`learn.microsoft.com/en-us/windows/wsl/networking`.

---

## Addendum (critic)

Completeness review of the readiness logic in §7(d). That logic has three hidden assumptions:

1. The runner ids taken from the preview stay valid.
2. An instance that vanishes from the polled node's `/state` was really deleted.
3. Nothing stale reappears after an EXO restart.

All three were checked in source at `bc412e2` (`gh api -H "Accept: application/vnd.github.raw"
"repos/exo-explore/exo/contents/<path>?ref=bc412e2230f395e652c23426a5eb88dfa515769b"`, plus GitHub code search on `main`, which
is the same commit). Nothing was run.

### E1. Runner ids are fixed for the life of an instance (VERIFIED)

- The worker creates its runner, and re-creates it after a failure, from the instance itself:
  `runner_id = instance.shard_assignments.node_to_runner.get(node_id, None)` and then
  `BoundInstance(instance=instance, bound_runner_id=runner_id, bound_node_id=node_id)` (`src/exo/worker/plan.py:99-132`).
- So the `rids` from §7(c) stay valid through all 5 backoff retries. A `RunnerFailed` -> `RunnerIdle` -> ... -> `RunnerReady`
  sequence happens under the same id.
- The worker also holds back while another rank of the same instance is `RunnerFailed`, unless its own runner failed too
  (`plan.py:115-123`). The worker's attempt counter resets only on `RunnerReady` for that runner id (`worker/main.py:192-200`).

### E2. An election does not empty the API's state (VERIFIED)

- On a new master every node calls `self.api.follow_new_master(result.won_clock)` (`src/exo/main.py:290-293`). Its docstring:
  "keep ours until its snapshot replaces it, so the running models stay usable" (`api/main.py:346-356`). It only ends in-flight
  streams.
- `API.reset()` does `self.state = State()` (`api/main.py:330-344`), but `main.py` never calls it. The only API calls there are
  `run`, `state`, `follow_new_master` and `unpause` (lines 194-211, 290-296), and code search found no other production caller.
- The API's state is replaced wholesale only by `_apply_snapshot` (`api/main.py:2079`). The master sends that snapshot to a
  node that asks for events older than its in-memory window (`_serve_event_log_request`, `master/main.py:718-735`).
- A newly elected master starts from the cluster state that node already held (`initial_state=self._cluster_state()`,
  `main.py:206-211, 261`).
- So an instance leaves `/state/instances` only through `InstanceDeleted`, or through a snapshot of a master state that no
  longer has it. "Missing after being seen -> FAILED" in §7(d) is sound.

### E3. During an election, commands block but state reads do not (VERIFIED)

- `_pause_on_new_election` sets `paused = True` when an election message carries a newer clock (`api/main.py:2125-2133`).
- `_forward` then waits on `paused_ev` before sending any command (`2191-2196`), until `unpause` runs. Commands include
  `POST /instance`, `DELETE /instance/...` and chat.
- `GET /state` (`get_state`, `457-473`) is not gated. It keeps serving the last applied state, which may be stale for the
  length of the election.
- Adapter rule: give command POSTs their own generous timeout (>= 30 s), separate from the 0.5 s state poll. Treat a slow
  POST as "election in progress", not as a dead cluster, while `/node_id` still answers.

### E4. No event log survives a restart (VERIFIED; corrects §2.7)

- At startup the node deletes the old log directories:
  - `shutil.rmtree(_OLD_MASTER_EVENT_LOG_DIR)` for `EXO_EVENT_LOG_DIR / "master"` (`main.py:47-48, 178-181`)
  - `shutil.rmtree(_OLD_API_EVENT_LOG_DIR)` for `.../"api"` (`api/main.py:224-225, 1972-1975`)
- The API begins from `State()` (`api/main.py:285`). `DiskEventLog` is referenced only by `src/exo/utils/disk_event_log.py`
  and its test (code search).
- The `~/.exo/event_log/` row in §2.7 therefore describes a removed feature.
- Together with non-persistent node ids (§2.7), a restarted single-Mac EXO always comes back with **zero instances**. The
  adapter must re-create the instance after any EXO restart and must never reuse an old `iid`.

### E5. Open questions found by this review and not resolved here

| Gap | Why it matters | Status |
|---|---|---|
| Time from `POST /instance` until the instance shows up in the polled node's `/state` | §7(d) gives up after "~30 s", which has no source basis | UNVERIFIED; measure |
| Whether `generation_stats` arrives on `/v1/chat/completions` streams | tokens/s telemetry without client-side timing | still UNVERIFIED (§9) |
| `GET /state` with an empty path returns the pydantic `State` directly (`api/main.py:457-459`), so its camelCase rests on FastAPI's default `by_alias` encoding | sub-paths explicitly use `model_dump(by_alias=True)` (`461`) | INFERRED for the empty path; the official harness reads camelCase from it (§4.1) |
| Size of a full `GET /state` while downloads report per-file progress | a 0.5 s poll of the whole state may be heavy on a busy cluster; sub-paths are the fallback | UNVERIFIED |
