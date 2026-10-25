# hcpeer: file-backed CPU weights on macOS peers

Status (2026-10-22, hcpeer v4): the patched `libggml-rpc` builds from the pinned b11770 sources, cross-compiled on the
Windows host with zig, and passes every static check against the real macOS release. The hcpeer code itself has run
on Linux (`selftest`, 79 checks, real files, mmap, rename, fork). **It has not run on a Mac.** `build_hcpeer_mac.py
selftest` and `smoke` are the first macOS runtime tests and have to be run at a Mac.

v3 (review fix): cache saves are atomic with or without `HC_PEER_MMAP` (stock wrote them in place), and with it on a
cache hit whose file cannot be the data asked for is deleted and answered as a miss instead of being served.

v4 (review fix): with `HC_PEER_MMAP=1` a save also has to keep the cache directory within `HC_PEER_CACHE_MAX_GIB` (a
managed peer's `--max-disk`), and every mapped file gets its access time set to now. A mapped file holds its disk
until it is unmapped, so the managed agent's LRU eviction cannot bring an over-full cache back under its cap; hcpeer
now never saves, and so never maps, past it.

## What it does

A Mac peer started as `ggml-rpc-server -d MTL0,CPU -c` exposes its CPU as a second RPC device. Stock b11770 copies
every weight the host sends to that device into anonymous memory (`ggml_aligned_malloc`, which is `vm_allocate` on
macOS), so the CPU device can hold at most what the Mac can keep resident, and anything beyond that goes to swap.

With hcpeer installed and `HC_PEER_MMAP=1`, a weight that reaches a CPU buffer through the RPC tensor cache is mapped
from its cache file `<LLAMA_CACHE>/rpc/<fnv64 hex>` instead. Untouched pages are clean and file-backed, so macOS can
drop them under memory pressure and fault them back in from the Mac's SSD. The cache files already exist for this
(a peer joined with `--cache` runs the worker with `-c` and `LLAMA_CACHE=<HOME>/rpc-cache`), and the same file can
back the tensor on every later load.

The installed library also makes the server's own cache files safe for an unattended Mac:

- every cache file the server saves (for any device, with or without `HC_PEER_MMAP`) is written to a new file
  `.hcpeer-<hash>.<pid>.tmp` in the cache directory (opened `O_CREAT | O_EXCL | O_NOFOLLOW`, so nothing found at that
  name is ever truncated, reused or followed), written in full (1 GiB pieces, short writes retried), fsynced and
  closed with every step checked, and only then renamed over `<hash>`; on any failure the temporary file is removed,
  nothing is left under the hash name, and the payload is still copied into the tensor (the host just misses the
  cache next time). Stock truncates and rewrites `<hash>` in place without checking the write, so a server killed or
  out of disk mid-write left a short file under the hash name, which every later load took as the whole weight
  (stock copies it, hcpeer would map it), and a rewrite under a live mapping changes the mapped tensor or kills the
  server with SIGBUS (measured in the self-test). With `HC_PEER_MMAP` unset the save is otherwise stock: no disk
  floor, and the stock `[set_tensor] saved to` line for every save;
- temporary files of servers that no longer run are removed at start (on or off, whenever the server runs with `-c`);
- with `HC_PEER_MMAP=1`, a cache hit (`SET_TENSOR_HASH`) whose file cannot hold the data asked for is deleted and
  answered as a miss, so the client resends the data and the atomic save writes a complete file; the tensor is left
  untouched. This applies to every buffer type, the Metal copy path included. `SET_TENSOR_HASH` carries the tensor,
  the offset and the hash but no size, so "the data asked for" is: the rest of the tensor from the offset (the b11770
  model loader sends every weight whole at offset 0, and never in chunks to an RPC device, which has no host buffer),
  or, for a tensor of three or more dimensions, a whole number of its outermost slices starting on a slice boundary
  (the b11770 MoE cache fills its 3-D expert banks with `slot * expert_size, n * expert_size`). A file that cannot be
  opened (other than a missing one, which stays a plain miss) is also a miss. With `HC_PEER_MMAP` unset this check
  does not run and the hit is stock;
- with `HC_PEER_MMAP=1`, a save is skipped (the payload is still copied) when it would leave less than
  `HC_PEER_CACHE_MIN_FREE_GIB` free on the cache volume, default 20, the same floor as `agent/seed_cache.py`; `0`
  disables it. Stock saves without any limit;
- with `HC_PEER_MMAP=1` and `HC_PEER_CACHE_MAX_GIB` set, a save is also skipped (the payload is still copied, into
  anonymous memory) when it would take the cache directory past that many GiB, a plain decimal number; any other value
  caps at 0, so a garbled cap saves nothing, and unset means no cap. The directory is measured on every save, as
  `agent/seed_cache.py` `cache_usage` counts it (every regular file, without the file the save replaces), so the
  agent's evictions and seed downloads count as they are now. The cap is checked at save time because a mapped file
  keeps its disk blocks until it is unmapped: nothing past the cap is ever saved, so nothing past it is ever mapped;
- with `HC_PEER_MMAP=1`, every file mapped into a tensor (save or hit) gets its access time set to now with
  `futimens` (the write time is kept, so the agent's crash purge by mtime is unaffected). A mapped hit reads at most
  the partial last page, and the agent ranks files for eviction by `max(atime, mtime)`, so without this a file in use
  could rank oldest;
- `[hcpeer]` log lines are a token bucket (512 lines burst, then one a second) with running totals in every line, so
  a steady stream of fallbacks (MoE expert banks at unaligned slots, for example) cannot fill the log.

Metal buffers are never mapped: they are wrapped in an `MTLBuffer` with a residency set at allocation, and remapping
memory under them is unsafe. Everything that is not a CPU-buffer weight arriving through the cache keeps the stock
copy.

## The patch

`native/hcpeer/hcpeer.patch` is the whole change to `ggml-rpc.cpp` (16 added lines, nothing removed, server side
only, no wire change). The logic lives in `native/hcpeer/src/hcpeer.cpp`.

| Hook | Where | Effect |
| --- | --- | --- |
| `hcpeer_alignment` | `rpc_server::get_alignment` | reports the page size (16384 on Apple Silicon) instead of 32 for the CPU buffer type, so the host places every tensor on a page boundary |
| `hcpeer_save_cached` | `rpc_server::set_tensor`, before the stock cache write | cache miss: atomic save (under the disk floor and the cache cap when enabled), then maps the file for a CPU tensor when enabled (`HCPEER_DONE`) or copies (`HCPEER_COPY`, stock write skipped). On macOS it never returns `HCPEER_STOCK`, so the stock in-place `std::ofstream` write is unreachable; only an unsupported platform still runs it |
| `hcpeer_map_cached` | `rpc_server::set_tensor_hash`, before the stock read | cache hit, when enabled: a file of the wrong size is deleted and the patch answers `result = 0` (`HCPEER_MISS`); a good file is mapped for a CPU tensor (`HCPEER_DONE`, `result = 1`); anything else (`HCPEER_STOCK`: disabled, no such file, Metal, mapping fell back) runs the stock read and copy |
| `hcpeer_log_banner` | `ggml_backend_rpc_start_server` | removes stale temporary files when there is a cache directory, then one banner line, `  hcpeer mmap    : on (v4, page 16384 B, ...)` or `: off (...)` |
| `#include "hcpeer.h"` | top | |

Why this works without a protocol change: the host's allocator aligns tensor offsets to the alignment the server
reports, against the real remote base from `BUFFER_GET_BASE` (`ggml-alloc.c` `aligned_offset`), and the CPU buffer
base comes from `vm_allocate`, which is page aligned (`ggml.c` `ggml_aligned_malloc`, `TARGET_OS_OSX`). The client only
sends `SET_TENSOR_HASH` (and, on a miss, `SET_TENSOR` with the cache flag) for `WEIGHTS` buffers and tensors over
10 MiB (`rpc_use_hash_cache`), so only large weights are ever mapped. Older hosts and agents keep working: they see a
larger alignment and nothing else.

A tensor is mapped only when all of these hold, otherwise the stock copy runs:

- `HC_PEER_MMAP=1` in the server's environment, the server runs with `-c`, and the build is for macOS (`__APPLE__`);
- the tensor's buffer type is exactly `ggml_backend_cpu_buffer_type()` (the CPU device's default, never Metal);
- the bytes `[data + offset, + size)` lie inside both the tensor and its buffer, and start on a page boundary;
- the cache file opens, is a regular file and has exactly the size just sent (save path) or a size the request can
  mean (hit path, see above; a file that fails this is not copied either, it is deleted and the hit is a miss).

Mapping: the whole pages are mapped with `mmap(dst, head, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE, fd, 0)`;
the partial last page is read with `pread` into the anonymous memory already there. The tail is read before the
mapping, so a failure leaves nothing half done. If `mmap` fails, the range is re-mapped as anonymous memory (or the
server aborts if even that fails) and the stock copy runs over it. Freeing the buffer is unchanged:
`ggml_aligned_free` calls `vm_deallocate` on the whole range, which removes the file mappings with it.

`PROT_READ | PROT_WRITE` rather than `PROT_READ`: with `MAP_PRIVATE` a write never reaches the file, it copies the
one page it touches, which is what the stock server would have done to its anonymous copy (checked in the
self-test). Read-only pages would turn any later write (`BUFFER_CLEAR`, `MEMSET_TENSOR`, a MoE bank slot being
refilled, a stock-path copy over a mapped range) into a crash of an unattended server.

Log lines (stderr, which the agent captures in `agent/logs/ggml-rpc-server.log`):

```
  hcpeer mmap    : on (v4, page 16384 B, CPU weights mapped from the cache, cache saves keep 20 GiB free, cache capped
                   at 30 GiB)    (one line; "no cache size cap" when HC_PEER_CACHE_MAX_GIB is unset)
[set_tensor] saved to '<cache>/rpc/<hash>'
[hcpeer] mapped <tensor>, <MiB> MiB from <cache>/rpc/<hash> (<n> tensors, <GiB> GiB mapped since start)
[hcpeer] <tensor>: <why>, copying instead (<path>, <n> fallbacks so far, <m> log lines dropped)
[hcpeer] <tensor>: removed the cache file, <bytes> bytes cannot be the data asked for at offset <o> of a <n>-byte
         tensor, answering a cache miss (<path>, <n> rejected so far, <m> log lines dropped)    (one line)
[hcpeer] not caching <cache>/rpc/<hash>: the cache volume would drop below HC_PEER_CACHE_MIN_FREE_GIB free (<n> so far)
[hcpeer] not caching <cache>/rpc/<hash>: the cache directory would grow past HC_PEER_CACHE_MAX_GIB (<n> so far)
[hcpeer] <tensor>: cannot set the access time of <path>, LRU eviction may pick it: <error>
[hcpeer] removed <n> temporary cache files left by stopped servers
```

## The release it replaces

Read from the pinned `llama-b11770-bin-macos-arm64.tar.gz` (sha256 `83790ddc...43e7`) with
`native/hcpeer/hcpeer_macho.py`, which also re-hashes every code page of each signature:

| File | Facts |
| --- | --- |
| `ggml-rpc-server` | arm64 executable, minos 13.3, rpath `@loader_path`, loads `@rpath/libggml-rpc.0.dylib`; signature `adhoc, linker-signed`, no team ID, no hardened runtime, no library validation, no entitlements |
| `libggml-rpc.0.27.0.dylib` | id `@rpath/libggml-rpc.0.dylib` current 0.27.0 compat 0.0.0, sha256 `547ac08b...452d`; loads `libggml-base.0`, `libc++.1`, `libSystem.B`, and weakly `/usr/lib/librdma.dylib` (built with the Apple RDMA transport); 90 exports; ad-hoc linker-signed |
| `libggml-rpc.0.dylib`, `libggml-rpc.dylib` | symlinks to the file above (the engine installer keeps them: `tarfile` `filter="data"`) |
| importers | only `libggml.0.27.0.dylib` imports from it, one symbol: `_ggml_backend_rpc_reg`. `ggml-rpc-server` finds the server entry at run time through `ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server")` |

Since the server is ad-hoc signed without the hardened runtime, it loads a dylib with a different ad-hoc signature;
library validation does not apply. `verify` refuses a server with library validation, or with the hardened runtime
and no `disable-library-validation` entitlement, team ID or not.

## Building

The prebuilt `native/hcpeer/build/libggml-rpc.dylib` (with `libggml-rpc.dylib.sha256` and `build.json`) is tracked
by git through `native/hcpeer/.gitignore` (the root `build/` rule ignores everything else in that folder), so a Mac
needs no toolchain. `tests/test_hcpeer_build.py` fails if `build.json` does not match the current sources, so a
source edit without a rebuild cannot be committed with a green suite.

Cross build (any host, used here):

```
uv run --no-project --with ziglang==0.16.0 python scripts/build_hcpeer_mac.py cross [--archive <release .tar.gz>]
```

Without `--archive` or `--engine-dir` it downloads the pinned release archive and checks its sha256 against
`configs/engines.lock.json`. It extracts four files into `native/hcpeer/build/release/`, links against the
release's own `libggml-base.0.27.0.dylib`, writes the dylib, `libggml-rpc.dylib.sha256` and `build.json` (source
hashes over LF line endings, so Windows and Mac checkouts agree), and runs the static check. Result for v3 with zig
0.16.0 (`-target aarch64-macos.13.3`, stripped, relative source paths): 798,720 bytes, **sha256
`6c703f2e4834299ff685a5c32b2db35778616649b35729939732ed0836beccd4`**, identical over two clean rebuilds, no compiler
warning at all with `-Wall -Wextra` (v2 was 798,592 bytes, `8a6af323...d99d`).

Native build on a Mac (needs Apple clang from `xcode-select --install`, which opens a GUI dialog):

```
uv run --no-sync python scripts/build_hcpeer_mac.py native [--engine-dir <bin dir>]
```

Same sources and flags, `-arch arm64 -mmacosx-version-min=13.3 -dynamiclib`, linked against the installed engine's
`libggml-base`; ld64 ad-hoc signs arm64 output itself. zig also runs on macOS, so `cross --engine-dir <bin dir>` works
on a Mac without Xcode.

Differences from the release library, both builds: no RDMA-over-Thunderbolt transport (irrelevant over Wi-Fi and
Iroh; a non-RDMA server answers HELLO with zero capabilities and prints `transport : TCP`, which the agent's banner
parser accepts). The zig build additionally links libc++ statically and exports 14 weak libc++ definitions
(`operator new`/`delete` and two templates); if dyld coalesces other images onto them they still call
`malloc`/`free`, like the system ones.

## Verified here, not verified

Verified on Windows and in WSL (evidence below):

- the hcpeer code at runtime on Linux x86-64 (`selftest`, in WSL Ubuntu 24.04, both g++ 13.3 and zig 0.16 clang with
  `-Wall -Wextra -Werror`): 56 checks with hcpeer on and 9 with it off, all passing: save then map, aligned hit maps,
  pages backed by the cache file (`/proc/self/maps`), unaligned hit falls back leaving the tensor untouched, a write
  to a mapped page never reaches the file, re-saving a name keeps an older mapping's bytes (rename), in-place
  truncation of a mapped file kills the reader with SIGBUS (why saves are atomic), outside-buffer tensors fall back,
  an offset past the tensor is left to the stock checks and keeps the file, a missing file is a silent miss, a
  sub-page tensor is saved and copied, a non-CPU buffer gets an atomic save but is never mapped, the disk floor skips
  the save and writes nothing, an unusable cache directory copies instead of crashing, a symlink and a hard link
  planted at the temporary name are neither followed nor truncated (`O_EXCL`), 4000 fallbacks in a burst log at most
  512 lines and the next line counts every dropped one, stale temporary files of dead servers are removed and live
  ones kept, a malformed floor falls back to 20 GiB;
- v3 checks in the same run: a short cache file (the first part of a payload, as an interrupted stock save leaves it)
  for a CPU tensor and for another buffer type is removed and answered as a miss with the tensor untouched and the
  rejection logged; a file larger than the tensor likewise; the resend saves a complete file that the next hit maps;
  a directory under a hash name is a miss and is left alone; on a 3-D bank a two-slot file at a slot boundary is
  mapped, while a short or part-slot file is removed; a 1-D tensor only accepts a file holding all of it. Off: the
  save is atomic, ignores the floor, is logged with the stock line and copies; a short file is left to the stock path
  as before; the banner removes dead servers' temporary files;
- v4, g++ 13.3 in WSL Ubuntu 24.04 (`/tmp`, ext4 relatime), 70 checks on and 9 off, all passing. New: with a sparse
  file filling the cache directory to half a payload under a 1 GiB cap, the save is skipped, logged, nothing is left
  and the tensor stays anonymous; at exactly the cap it is saved and mapped; re-saving a name counts only the new
  bytes; a cap of 0 saves nothing; four malformed caps show `cache capped at 0 GiB`; unset shows `no cache size cap`;
  a mapped hit sets a file's atime (set to tomorrow first, which relatime never moves on its own) to now and keeps its
  mtime. Off ignores the cap. The pre-fix `hcpeer.cpp` built with the same harness fails 10 on checks (the cap, the
  banner, the atime) and the off banner check;
- v4 cross build (zig 0.16, offline) passes the static check against the release (`verify_ok`), and the self-test
  harness compiles and links for arm64 macOS with `-Werror` and no warning;
- the new checks catch the old behaviour: with the size rule disabled (mutant) 7 of them fail, with `O_TRUNC` in
  place of `O_EXCL` the hard-link check fails, and the off run passes for both mutants;
- the self-test also compiles and links for arm64 macOS with zig, exactly as `selftest` builds it on a Mac (`-Werror`,
  no `HCPEER_POSIX_TEST`, using `vm_allocate` like ggml);
- the smoke client against a real stock b11770 `ggml-rpc-server` (Windows, CPU only, the v2 client before the
  short-file probe was added): protocol 8 HELLO, alignment, alloc, cache save, two hash hits, three read-backs equal,
  a 50,152-byte cache file named by the FNV-1a hash; the v3 client (with the probe) has run only against the
  in-process fake server in `tests/test_hcpeer_build.py`;
- the patched sources compile for arm64 macOS with `-Wall -Wextra` and no warning at all (v3 rebuilt twice here);
- Mach-O layout: arm64 `MH_DYLIB`, install name and versions equal to the release's, rpath `@loader_path`, minos 13.3;
- the ad-hoc code signature is present and every page hash matches the file;
- symbols, checked against the full extracted release (14 Mach-O importers): every release importer's symbols and
  every `_ggml_*` API export of the original are exported; all 48 imports bound to `@rpath/libggml-base.0.dylib` are
  exported by the release's `libggml-base`; the other 174 bind to `libSystem.B`; no flat-namespace imports;
- the vendored upstream files match b11770 byte for byte (sha256 pinned in `tests/test_hcpeer_build.py`).

Not verified (needs a Mac):

- that dyld loads it and `ggml-rpc-server` starts (`smoke` checks this first);
- the two patch lines that answer `HCPEER_MISS` as `result = 0` inside the real server: the self-test drives
  `hcpeer.cpp` only, and no patched server has run anywhere (`smoke` now plants a short file and checks the miss,
  the removal and the log line);
- `MAP_FIXED` over a `vm_allocate` range on XNU and its SIGBUS/copy-on-write behaviour (`selftest` checks this);
- that clean mapped pages are really dropped under pressure and re-read, the page-fault read rate on an M4 Air SSD,
  and the decode speed of a CPU stage reading from mapped weights;
- macOS behaviour of the static libc++ copy next to the system one;
- anything with a real model.

## Running it

hcpeer runs from a copy of the engine, picked per agent with `HYPERCOMPUTE_LLAMA_BIN_DIR`, so the stock engine stays
stock, other agents and `hypercompute engine` keep using it, and nothing needs sudo. This replaces the in-place
`install` below for the peers. At each Mac, from the repo checkout after pulling, with the agent stopped (Ctrl+C in its
terminal, or `uv run hypercompute leave`):

```
# 1. make the clone, or re-check it if it is already current; the path is the only line on stdout
CLONE=$(bash scripts/capacity/hcpeer_mac_install.sh) && echo "$CLONE"
# 2. only if step 1 printed a path: join with hcpeer on (ticket and code from `uv run hypercompute invite` on the host)
HYPERCOMPUTE_LLAMA_BIN_DIR="$CLONE" HC_PEER_MMAP=1 uv run hypercompute join --ticket <T> --code <C> \
  --devices MTL0,CPU --managed --devices-allowed MTL0,CPU --max-disk 80GiB --max-memory 10GiB --reserve 4GiB --yes
```

A peer already joined with `--devices MTL0` (with the two variables set) gets its CPU device from the host:

```
uv run hypercompute capacity peers                                  # on the host: the node ids
uv run hypercompute capacity worker --node <id> --devices MTL0,CPU  # the peer restarts its worker with both
```

`hcpeer_mac_install.sh` (bash 3.2, plain user, macOS tools only):

- finds the stock bin dir the way `build_hcpeer_mac.py` does (`$HYPERCOMPUTE_HOME/engines/llama-b11770-macos-arm64/`
  plus `bin_subdir` from its marker; `--engine-dir` overrides) and refuses one whose real `libggml-rpc` file is not
  the release library in `build.json` (`release_original_sha256`), including one with hcpeer installed in place
  (`build_hcpeer_mac.py rollback` restores it);
- checks `native/hcpeer/build/libggml-rpc.dylib` against `libggml-rpc.dylib.sha256` (`--dylib` overrides);
- copies the bin dir to `<bin dir>.staging.<pid>` (`cp -cR`, an APFS clone, else `cp -R`; symlinks stay symlinks),
  drops a copied engine marker so the clone is never taken for the installed engine, and renames the hcpeer library
  over the clone's real file behind `libggml-rpc.dylib` (`libggml-rpc.0.dylib`, the name dyld loads, must be the
  same file; links may only point inside the directory), so the file gets a new inode;
- runs `codesign --verify` on it and re-signs ad hoc (`codesign -s - -f`) only if that fails, then removes
  `com.apple.quarantine` from every file and directory that has it;
- starts the staged `ggml-rpc-server -H 127.0.0.1 -p 47152 -d CPU -c` (next ports 47153-47156 if that one is taken)
  with `HC_PEER_MMAP=1`, a 0 GiB floor, a scratch `LLAMA_CACHE` and `DYLD_*` unset, waits for it to listen, stops it,
  and requires `hcpeer mmap    : on (v4` or a later version;
- writes `.hcpeer-clone` (source dir, the stock and hcpeer sha256s, the installed file and its sha256) and moves the
  staging dir to `<bin dir>-hcpeer`.

A second run with the same stock engine and dylib finds the clone current and only smokes it again; a changed stock
engine, a new dylib or an edited clone is rebuilt. It never writes inside the stock bin dir and refuses to replace a
`-hcpeer` directory without `.hcpeer-clone`. `hypercompute engine install` replaces the whole install dir, the clone
with it: run the script again afterwards. The clone shows as `[source-build]` in the join plan (the override path);
the host's release check still passes, since the release comes from `llama-server --version`.

What the agent does with `HC_PEER_MMAP=1` in its environment on macOS (`agent/hcpeer.py`, `Agent.worker_command`):

- turns the tensor cache on, so the worker gets `-c` and `LLAMA_CACHE=<HOME>/rpc-cache`, and sets `HC_PEER_MMAP=1`
  in the worker's environment itself;
- with `--managed --max-disk`, sets `HC_PEER_CACHE_MAX_GIB` to `max_disk_bytes // GiB` only once a banner showed
  hcpeer v4 or later: the first start of a session learns the version, and the agent restarts that worker once with
  the cap; later starts (host restarts with `capacity worker`, crash revivals) pass it from the start. A v3 or stock
  library gets no cap;
- prints before the consent prompt that the worker's CPU device maps cached tensor files from disk, and after the
  start what the banner said, for example `hcpeer v4 on: CPU weights are mapped from the cache, cache capped at 80
  GiB.` or a warning that the engine has no hcpeer (then the CPU device copies weights into memory as stock does).

The opt-in lives on the Mac: the host's `capacity worker` changes devices and threads, and the restarted worker keeps
hcpeer as the contributor set it. To stop using hcpeer, stop sharing and join again without the two variables; to
remove the clone, `rm -rf "$CLONE"` after that. Check a running worker with
`grep 'hcpeer mmap' ~/.hypercompute/agent/logs/ggml-rpc-server.log`.

## What a Mac must run after pulling

The older in-place route, kept for reference: it replaces the library inside the stock engine. At the Mac, before
leaving it unattended, from the repo checkout (the prebuilt dylib comes with the pull):

```
# 0. stop the agent if it is running (uv run hypercompute leave), so the worker restarts with the new library
# 1. static check against this Mac's engine
uv run --no-sync python scripts/build_hcpeer_mac.py verify
# 2. optional, about a minute, downloads the ziglang wheel: the hcpeer code on this Mac's kernel
uv run --no-project --with ziglang==0.16.0 python scripts/build_hcpeer_mac.py selftest
# 3. runtime test of the candidate, then install (keeps the original as libggml-rpc.0.27.0.dylib.hcpeer-orig),
#    then the decisive test of the installed copy
uv run --no-sync python scripts/build_hcpeer_mac.py smoke --dylib native/hcpeer/build/libggml-rpc.dylib
uv run --no-sync python scripts/build_hcpeer_mac.py install
uv run --no-sync python scripts/build_hcpeer_mac.py smoke
# 4. join with the CPU device and the cache; the agent passes its environment to ggml-rpc-server
HC_PEER_MMAP=1 uv run hypercompute join --ticket <T> --code <C> --devices MTL0,CPU --cache --max-memory 10GiB \
  --reserve 4GiB --yes
```

Only if the dylib is missing from the checkout (or `verify` reports a different engine): build it on the Mac with
`uv run --no-project --with ziglang==0.16.0 python scripts/build_hcpeer_mac.py cross --engine-dir "$BIN"`, where
`BIN` is `~/.hypercompute/engines/llama-b11770-macos-arm64/llama-b11770` (or under `$HYPERCOMPUTE_HOME`).

`smoke` starts the engine's `ggml-rpc-server` on 127.0.0.1 with `-d CPU -c`, `HC_PEER_MMAP=1` and a 0 GiB floor in a
temporary cache, sends three 50 KB tensors (cache save, aligned cache hit, unaligned cache hit), then plants half of
another payload under that payload's hash (what an interrupted stock save leaves) and asks for it at the aligned
tensor, reads the three tensors back and stops the server. It passes only if the banner says `on`, the CPU alignment
is the page size, both real hits are answered, the planted hit is a miss and its file is gone, all three read back
unchanged (so the rejected file did not touch the tensor), the log shows exactly two `mapped` lines, one `copying
instead` and one `answering a cache miss`, the saved cache file is complete and no temporary file is left. A pre-v3
library fails `short_file_missed`. With `--dylib` the candidate is loaded through `DYLD_LIBRARY_PATH`
so it is tested before it replaces anything (if that run fails only on `banner_on`, dyld did not pick the candidate
up; the installed-copy smoke is the decisive one). If the installed-copy smoke fails, run `rollback` and join without
`HC_PEER_MMAP` (stock behaviour). Every command prints JSON and exits 0 (ok), 1 (checks failed) or 2 (error).

Without `HC_PEER_MMAP=1` the installed library behaves like stock b11770 except for the two build differences above
and its cache saves (atomic, fsynced, stale temporary files removed at start), which only ever leave a complete file
or none, so installing it is not otherwise a behaviour change: no disk floor, no size check on hits, nothing mapped.
Without `--devices ...,CPU` nothing is mapped (the atomic saves, the size check on hits and the floor still apply
while `HC_PEER_MMAP=1`).

## Install and rollback

`install` refuses unless the static check passes and the current file (or the existing backup) is the pinned release
file by sha256. The first time it copies the original to a `.part` file, checks its sha256 and renames it to
`libggml-rpc.0.27.0.dylib.hcpeer-orig` (a short backup would make rollback impossible). It copies the new library to
`libggml-rpc.0.27.0.dylib.hcpeer-tmp`, checks that the copy has the sha256 the static check saw, and renames it over
`libggml-rpc.0.27.0.dylib`, so the file gets a new inode (rewriting a loaded, signed Mach-O in place can get the next
process that maps it killed by the code-signing check). The symlinks keep pointing at the same name. A running
`ggml-rpc-server` keeps the library it loaded until it restarts. It writes `.hcpeer.json` (sha256, source, time).

`rollback` renames the backup back (new inode again), removes `.hcpeer.json`, and is idempotent. `status` reports
whether the library is stock or hcpeer by sha256. An engine reinstall (`hypercompute engine install`) replaces the
whole directory, which also reverts to stock.

Never truncate or rewrite a cache file in place while a server has it mapped: a mapped page past the new end of file
raises SIGBUS (the self-test shows it). Replacing by rename (hcpeer's own saves, and `agent/seed_cache.py`: part file,
verify, `os.replace`) and deleting (`leave --purge-cache`, after the worker stops) are safe, because a mapping keeps
its own reference to the old file. A deleted but still mapped file keeps its disk space until the worker frees that
buffer or restarts, so eviction cannot reclaim a mapped file. The managed agent's `guard_disk` (`agent/managed.py`,
every 15 s, through `seed_cache.evict_cache`) holds `--max-disk` and the free floor by unlinking the least recently used
cache files and counts every unlink as freed; for a mapped file nothing comes back, while the cache's measured size
drops. While files are mapped, eviction therefore enforces neither limit, and the deleted but mapped bytes are counted
by neither side's directory scan. hcpeer v4 keeps both limits at save time instead (nothing below the floor or past
`HC_PEER_CACHE_MAX_GIB` is saved, so nothing is mapped there) and marks mapped files as just used so LRU takes files
not in use first. Free space can still fall under the floor for other reasons (swap, the user's own files), and then
the guard deletes mapped files that free nothing; the agent side has to stop that (below).

## What the agent would need (1 and 2 wired, see Running it)

1. Install step: done as a clone instead (`scripts/capacity/hcpeer_mac_install.sh`), run by the contributor while
   the agent is stopped; the stock engine is never changed, so there is nothing to roll back.
2. Environment: done (`agent/hcpeer.py`, `Agent.worker_command`): `HC_PEER_MMAP=1` together with `-c`, and with
   `--max-disk` the cap `HC_PEER_CACHE_MAX_GIB` = `max_disk_bytes // GiB` once a banner shows `(v4` or later.
   `HC_PEER_CACHE_MIN_FREE_GIB` is not passed: the consent floor is always `MIN_FREE_BYTES` (20 GiB), which is
   hcpeer's default. The worker inherits the agent's environment, so a join with `HC_PEER_MMAP=1` refuses a shell
   that already sets `HC_PEER_CACHE_MAX_GIB` or `HC_PEER_CACHE_MIN_FREE_GIB` (`cli.py`, `hcpeer.inherited_limits`).
3. Report it: parse `^\s*hcpeer mmap\s*:\s*(on|off)` and the page size from the banner next to `parse_rpc_banner`
   (another engineer owns `adapters/llama_output.py`), and send it in the node inventory as an optional field
   (for example `rpc_cpu_file_backed: bool`, `rpc_cpu_alignment: int`), so older hosts ignore it.
4. Evidence: read the running totals from the last `[hcpeer] mapped` and `copying instead` lines (lines may be
   dropped under a flood, the totals are not), so a run can show that the CPU stage was file-backed.
5. A remote control to restart the worker with or without `HC_PEER_MMAP`, so the host can switch it off overnight
   without someone at the Mac.

## Requested changes in files others own

- `services/planner.py`, `services/moe_tier.py`: when a peer reports `rpc_cpu_file_backed`, its CPU device's capacity
  is bounded by its cache disk, not its RAM: plan CPU-device weights against the smaller of free disk minus the 20 GiB
  floor and `--max-disk` minus the cache's current size (hcpeer v4 copies anything past either into RAM), count
  only page cache (not resident memory) for speed, and price those bytes with a fault-read rate (assumed about
  0.9 ms/MB until measured on an M4 Air). Do not split one layer between a peer's MTL0 and CPU (each crossing is a
  synchronous copy through the host). MoE expert banks on a peer's CPU device map only slots at page-aligned offsets
  (for gpt-oss-120b, `expert_size` 4,406,400 B is not a multiple of 16 KiB, so only slot 0); the rest are copies.
- `services/deployments.py`: the CPU device's GET_DEVICE_MEMORY still reports physical RAM; do not reject a placement
  on it for exceeding that when the node is file-backed. Seed the peer's cache first (`seed_cache.py`), so the load
  maps instead of uploading.
- `adapters/llama_output.py`: the banner regex above (the extra line sits before `Devices:` and matches none of the
  existing patterns).
- `scripts/capacity/run_big.py`, `hostmon.py`: record per peer whether hcpeer was on and its mapped bytes, so runs
  with and without it can be compared.
- `agent/managed.py` `guard_disk`, `agent/seed_cache.py` `evict_cache`: while a worker runs with `HC_PEER_MMAP=1`,
  do not unlink cache files it has mapped (protect the names in its `[hcpeer] mapped` and `saved to` lines since it
  started, or a list hcpeer would write), count deleted but still mapped bytes as used, and end a floor pass whose
  unlinks did not raise `disk_free` instead of deleting the next batch every 15 s.

## Risks and limits

- Cache files written before hcpeer was installed (by stock, which writes in place) may be short if a server was
  killed or ran out of disk while writing. With `HC_PEER_MMAP=1` such a file is deleted and re-fetched on its first
  hit, unless its length happens to pass the size rule: for a tensor of three or more dims (expert weights, MoE cache
  banks) a file cut exactly on an expert boundary is accepted. With `HC_PEER_MMAP` unset hits are stock and trust any
  file under its hash name. If a Mac's cache has any doubt, move `<HOME>/rpc-cache/rpc` aside before joining (the next
  load re-uploads and re-saves atomically). Only a content hash check would close this fully; it was left out because
  FNV-1a over every hit would read each mapped weight in full at load time, which mapping exists to avoid.
- With `HC_PEER_MMAP=1`, a cached partial write of a tensor of one or two dims (any offset and size other than "the
  rest of the tensor") is deleted on every hit and resent, so it never benefits from the cache. b11770 sends no such
  writes to an RPC device of more than 10 MiB (the hash cache threshold): weights go whole, MoE cache fills are whole
  slots of 3-D banks.
- Compute buffers on the peer's CPU device are padded to 16 KiB per tensor (alignment applies to every buffer of that
  device), a few tens of MB at most.
- Weights at or below 10 MiB, and the partial last page of each mapped tensor, stay in anonymous memory.
- The first token after a load faults every touched page in from the SSD.
- `fsync` per saved tensor adds a little to first loads, with or without `HC_PEER_MMAP` (later loads hit the cache and
  write nothing).
- The zig build exports weak libc++ symbols; the native build does not.
- No RDMA transport in either build.

## Evidence

- Build output: `native/hcpeer/build/libggml-rpc.dylib`, `build.json`, `libggml-rpc.dylib.sha256` (tracked).
- Session scratchpad (`review_tools/`): `cross_review_1.json`, `cross_review_2.json` (two identical rebuilds),
  `verify_full_release_review.json` (static check against the full release), `selftest_linux.json` (the Linux
  self-test output); `review_winrpc/stock_smoke.json` (smoke client against the real stock server). These are v2.
- v3, session scratchpad `hcpeer_v3/`: `selftest_gxx.txt` (g++ self-test, 56 + 9 checks), `mutate.txt` (the two
  mutants and the zig-built static self-test), `cross_1.json` (cross build and static check), `build_v2.json` (the
  v2 build record it replaced); `rebuild2/` (second, byte-identical v3 build), `hcpeer_posix_test_macos` (the arm64
  macOS self-test binary, built, not run).
- v4, session scratchpad `hcpeer_cap/`: `selftest_before.txt` and `selftest_after.txt` (the g++ self-test of the
  pre-fix and fixed `hcpeer.cpp` with the v4 harness), `cross_v4.json` (cross build and static check),
  `zig_macos_selftest.log` and `hcpeer_posix_test_macos` (arm64 macOS self-test binary, built, not run), `before/`
  (the v3 sources, `build.json` and dylib it replaced).
- Sources: pristine `native/hcpeer/third_party/llama.cpp/` (b11770, commit `39da130f`), patched
  `native/hcpeer/src/ggml-rpc.cpp`, `native/hcpeer/hcpeer.patch`, self-test `native/hcpeer/test/hcpeer_posix_test.cpp`.
- Tests: `uv run --no-sync pytest tests/test_hcpeer_build.py -q -p no:cacheprovider`.
