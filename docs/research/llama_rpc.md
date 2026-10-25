# llama.cpp RPC audit (pinned release b11770)

Source: `ggml-org/llama.cpp` tag `b11770` = commit `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb`
(committed 2026-10-20T15:47:44Z, "ggml-cuda : support ssm_conv kernel size 7 (#30873)").
Fetched read-only through `gh api repos/ggml-org/llama.cpp/contents/<path>?ref=<sha>`. No code was built or run.

Delta check: `gh api repos/ggml-org/llama.cpp/compare/39da130...9a3b91c6801e68f412fcf1b6f8e153aa562b8536`
returned `status=ahead ahead=12 behind=0`, and no changed file path contains `rpc`. So the handover's
audit commit and this tag have the same RPC code.

Files read in full: `ggml/src/ggml-rpc/ggml-rpc.cpp` (3194 lines), `ggml/include/ggml-rpc.h` (35),
`tools/rpc/rpc-server.cpp` (268), `tools/rpc/README.md` (137), `ggml/src/ggml-rpc/transport.h` (39),
`ggml/src/ggml-rpc/CMakeLists.txt` (52), `tools/rpc/CMakeLists.txt` (20), `ggml/src/ggml-rpc/log.h`/`log.cpp`,
`tests/test-rpc-multi-server.{cpp,sh}`.
Read in part: `ggml/src/ggml-rpc/transport.cpp` (TCP path, caps and socket setup, lines 1-212 and 530-809),
`ggml/src/ggml-metal/ggml-metal-device.m`, `ggml/src/ggml-metal/ggml-metal.cpp`, `ggml/src/ggml-cpu/ggml-cpu.cpp`,
`common/arg.cpp`, `ggml/src/ggml-backend-meta.cpp`, `src/llama.cpp`, `ggml/src/ggml.c`, `ggml/include/ggml.h`,
`ggml/include/ggml-backend.h`.

Line numbers below refer to the files at that commit. Paths without a directory mean `ggml/src/ggml-rpc/`.

---

## TL;DR for Hypercompute planning

1. **One client per server at a time.** The accept loop is sequential and `listen()` uses a backlog of 1.
   A second `llama-server` that connects while one is active stalls in the HELLO handshake until the first client
   disconnects. Hypercompute has to give out exclusive leases on each worker's `ggml-rpc-server`.
2. **Losing a worker mid-run kills the client.** Every send or receive failure on the client goes through
   `RPC_STATUS_ASSERT` -> `GGML_ABORT` -> `abort()`. If the worker is unreachable at startup, `--rpc` aborts during
   argument parsing. There are no socket timeouts or keepalives, so a silent network drop can hang the client
   instead. The host has to supervise `llama-server` as a child process and run its own watchdog.
3. **Version check is weak.** The client only rejects `major != 8 || minor > 0`. The server never checks the
   client's version, only the HELLO payload size (24 bytes). The patch number and build tag are never compared at
   runtime, so Hypercompute has to enforce "same tag everywhere" itself.
4. **Builds of the same tag on different platforms are meant to interoperate** (CUDA, Metal and CPU together,
   as the README diagram shows). This assumes 64-bit little-endian hosts and no override of `GGML_MAX_NAME`.
5. **`--cache` stores weight tensors larger than 10 MiB**, keyed by a 64-bit FNV-1a hash, in
   `$LLAMA_CACHE/rpc`, `%LOCALAPPDATA%\llama.cpp\rpc`, `~/Library/Caches/llama.cpp/rpc` or
   `${XDG_CACHE_HOME:-~/.cache}/llama.cpp/rpc`. The files are never deleted and have no size cap.
6. **There is no authentication.** Any peer that can reach the port can allocate memory, read and write tensors,
   and make the server listen on `0.0.0.0:<port+1000>` or connect to any host (`COMM_INIT`).
   Bind to a private or overlay interface only.
7. **Metal reports `recommendedMaxWorkingSetSize` as total memory**, and free memory as that total minus this
   process's `currentAllocatedSize`. The CPU device on macOS and Linux reports free memory equal to total.

---

## 1. Server accept loop, concurrency, disconnect cleanup

### Verified: sequential, one client at a time

`ggml_backend_rpc_start_server` (ggml-rpc.cpp:2762-2831) creates the backends once, then:

```cpp
// ggml-rpc.cpp:2817-2826
while (true) {
    auto client_socket = server_socket->accept();
    if (client_socket == nullptr) {
        LOG_ERROR("Failed to accept client connection\n");
        return;
    }
    LOG_INFO("Accepted client connection\n");
    rpc_serve_client(backends, cache_dir, client_socket);
    LOG_INFO("Client connection closed\n");
}
```

`rpc_serve_client` runs on the accept thread and returns only when the client disconnects or a protocol error
occurs. There are no threads, no `select`/`poll` and no queueing of sessions. The listen backlog is 1:

```cpp
// transport.cpp:745
if (listen(sockfd, 1) < 0) {
```

The client side reflects this. ggml-rpc.cpp:3004-3005 says *"a server processes its socket sequentially"*.

### Second client while one is active

- Verified from code: the server does not call `accept()` again until `rpc_serve_client` returns. A second
  connection is not served while the first is active.
- Verified from code: the client's first action is `negotiate_hello`. It sends `RPC_CMD_HELLO` and then blocks
  in `recv_data` waiting for the reply (ggml-rpc.cpp:423-430, 390-405). That receive has no timeout.
  Grepping transport.cpp for `timeout|KEEPALIVE|RCVTIMEO` finds only the RDMA queue-pair attribute
  `a.timeout = 14` at line 391, and no socket option. So the second client hangs until the first one
  disconnects, and then gets served.
- In `llama-server` this happens during argument parsing. `--rpc` calls `add_rpc_devices`
  (common/arg.cpp:2683-2693 -> 1169-1188) -> `ggml_backend_rpc_add_server` -> `ggml_backend_rpc_get_device_count`
  -> `get_dispatcher` -> `rpc_dispatcher::start` (ggml-rpc.cpp:3150-3159, 3136-3141, 708-724, 638-658).
  The second `llama-server` therefore appears frozen at startup.
- UNVERIFIED (OS TCP stack, not llama.cpp code, and not tested): what happens once more connections arrive than
  the backlog of 1 can queue. Linux usually drops the SYN, so the client's `connect()` retries and eventually
  times out. Windows usually answers with RST, so `connect()` fails and the client hits
  `GGML_ABORT("Failed to connect to %s\n")`.

### Disconnect and resource lifetime (verified)

- Per-connection state: `rpc_serve_client` builds a fresh `rpc_server server(backends, cache_dir);`
  (ggml-rpc.cpp:2436). That object owns the `buffers` set, `stored_graphs` (graph cache) and `comm_states`
  (ggml-rpc.cpp:1446-1451).
- When the function returns, `~rpc_server` frees every buffer the client allocated:
  ```cpp
  // ggml-rpc.cpp:2428-2432
  rpc_server::~rpc_server() {
      for (auto buffer : buffers) {
          ggml_backend_buffer_free(buffer);
      }
  }
  ```
  The graph cache and comm peer sockets are destroyed along with the object, so a clean disconnect returns
  device memory.
- Process-lifetime state: the `ggml_backend_t` backends are created once (ggml-rpc.cpp:2782) and reused for
  every client. The `rpc_transport_shutdown()` and `ggml_backend_free` code after the loop (2827-2830) only runs
  if `accept()` fails, because the function returns early in that case. A server that is working normally never
  exits on its own and has to be killed.
- Risk (inferred from code, not tested): the destructor does not call `sync_all_backends()` before freeing
  buffers. Graph compute on the server is asynchronous (`ggml_backend_graph_compute_async`, 2191). If a client
  drops while a GPU graph is still running, its buffers may be freed underneath the running kernel.
- Any handler that returns `false` (validation failure, unknown command, short read) makes `rpc_serve_client`
  `return`, which closes the connection (for example 2474-2482 and 2500-2512). The client then aborts (see §3).

### Supervisor-relevant process behaviour (verified)

- Without `-d`, `-H` or `-p` the server binds `127.0.0.1:50052` (rpc-server.cpp:94-100).
- `-H` must be an IPv4 literal: `inet_addr(host) == INADDR_NONE` -> "Invalid host address" (transport.cpp:733-736).
  Hostnames such as `localhost` and IPv6 are not accepted. The socket is `AF_INET` only (transport.cpp:725).
  The client side resolves hostnames with `gethostbyname` (transport.cpp:764), IPv4 only.
- Exit codes are misleading. Bind failure, a bad host or an accept failure only logs and returns from
  `start_server_fn`, and then `main` does `return 0;` (rpc-server.cpp:266-267). An unknown CLI argument prints
  usage and calls `exit(0)` (rpc-server.cpp:160-165). Only `-d <unknown>`, "No devices found", a cache directory
  error, or a missing RPC backend return 1. **A supervisor cannot use exit code 0 as proof that the server
  started. Probe the port, or parse stderr for `Starting RPC server`.**
- Log sink: `LOG_INFO/WARN/ERROR` map to `GGML_LOG_*` (log.h:43-45), and the default ggml callback writes to
  stderr (ggml.c:314-319). `Accepted client connection` and `Client connection closed` (2823, 2825) are printed
  at default verbosity and can drive a busy/idle indicator. `GGML_RPC_DEBUG=1..3` adds handshake, transport and
  per-command traces (log.h:11-35, README.md:114-136).
- `SO_REUSEADDR` is set on the listen socket (transport.cpp:705-709, 729). On Windows that option allows
  another socket to bind the same port. UNVERIFIED whether two `ggml-rpc-server` instances on one Windows port
  would both start without error. Hypercompute should hold a lockfile per port.

---

## 2. HELLO handshake, versioning, transport negotiation, cross-build compatibility

### Wire framing (verified)

```
request : | cmd (1 byte) | request_size (8 bytes) | request_data |      // ggml-rpc.cpp:372-386
response: | response_size (8 bytes) | response_data |                  // ggml-rpc.cpp:388-405
```

The structs are `#pragma pack(push, 1)` (ggml-rpc.cpp:34, 272). `RPC_CMD_HELLO` is pinned to 14:
`static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");` (ggml-rpc.cpp:85).

### Structs (verified)

```cpp
// ggml-rpc.cpp:124-134 ; RPC_CONN_CAPS_SIZE = 24 (transport.h:11)
struct rpc_msg_hello_req { uint8_t conn_caps[RPC_CONN_CAPS_SIZE]; };
struct rpc_msg_hello_rsp { uint8_t major; uint8_t minor; uint8_t patch; uint8_t padding;
                           uint8_t conn_caps[RPC_CONN_CAPS_SIZE]; };
```

The request payload is 24 bytes and the response payload is 28 bytes.

### Version checks (verified)

- Client side (ggml-rpc.cpp:432-436):
  ```cpp
  if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
      LOG_ERROR("RPC server version mismatch: %d.%d.%d\n", ...);
      return false;
  }
  ```
  A `false` return becomes `GGML_ABORT("RPC handshake failed for %s\n")` (ggml-rpc.cpp:652-654). The patch
  number is ignored. A server with a lower minor version is accepted.
- Server side (ggml-rpc.cpp:2437-2456): the first byte must be `RPC_CMD_HELLO`, otherwise
  `"Expected HELLO command, update client"` and the connection closes. The payload size must equal
  `sizeof(rpc_msg_hello_req)`, otherwise `"HELLO request size mismatch ... client needs upgrade to protocol v8.x"`.
  **The server never inspects the client's version.** The request carries none, only caps.
- A second HELLO after the handshake closes the session (`case RPC_CMD_HELLO: return;`, 2485-2488).
- Current version: `RPC_PROTO_MAJOR_VERSION 8`, `MINOR 0`, `PATCH 0` (ggml-rpc.h:9-11). The server prints
  `Starting RPC server v8.0.0` (2769-2772). The README sample output still shows `v3.0.0` (README.md:61), which is
  stale.

### Transport negotiation (verified)

- The client fills `request.conn_caps` with `sock->get_caps()` (423-427). The server replies with its own caps
  (2466) and then both sides call `update_caps(peer_caps)` (2472 server, 438 client).
- `get_caps` zero-fills the 24 bytes. It probes RDMA only if built with `GGML_RPC_RDMA` and `GGML_RPC_NO_RDMA`
  is unset (transport.cpp:588-616). `update_caps` falls back to TCP with no error if either side has no RDMA
  (all-zero caps) or if RDMA activation fails (transport.cpp:618-649). `transport_name()` returns "RDMA" or "TCP"
  (686-688).
- Windows builds never have RDMA. `CMakeLists.txt:29-31` forces `GGML_RPC_RDMA OFF` when the platform is neither
  APPLE nor non-WIN32. Windows nodes always speak TCP.

### `GGML_OP_COUNT` static_assert (verified meaning)

```cpp
// ggml-rpc.h:13-15
static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
```

This check runs only at compile time. It reminds maintainers to bump the patch version whenever the op enum
changes, because ops travel as raw enum integers (`result.op = tensor->op;` ggml-rpc.cpp:781; server
`result->op = (ggml_op) tensor->op;` 1687). Types also travel raw (`result.type = tensor->type;` 761). The server
validates only `type < GGML_TYPE_COUNT` (1643) and does not validate the op range. **Because the patch number is
never checked at runtime, two builds of different tags that both report 8.0.x will connect and may silently run
the wrong op if their enums differ.**

### Same tag on different platforms: interoperable, with conditions

Verified facts that support interoperability:
- Every wire struct uses fixed-width fields and is packed (34-272). `rpc_tensor` size is checked to be a multiple
  of 8 (54).
- Backend-specific sizing is asked of the server, not assumed: `GET_ALIGNMENT`, `GET_MAX_SIZE`, and
  `GET_ALLOC_SIZE` for quantized tensors with `ne[0] % 512 != 0` plus ops that may expand
  (1006-1027, 1034-1123). This covers CUDA's 512 padding (803-806).
- `GRAPH_CACHE_MAX = 1024` "client and server must use the same value" (120-122). The same tag guarantees that.
- The README architecture shows a CUDA host, a Metal host and a CPU device serving one client (README.md:11-34).

Conditions (inferred from code, not tested):
- **64-bit only.** Size prefixes are written as `size_t` (`send_data(&msg_size, sizeof(msg_size))`, 324, and
  `&input_size`, 379) and read as `uint64_t` (334-335, 394-395).
- **Little-endian only.** Structs are `memcpy`'d with no byte-order conversion. macOS arm64, Windows x64 and
  Linux x64/arm64 all qualify.
- **Same compile-time limits.** The `rpc_tensor` layout depends on `GGML_MAX_DIMS` (4), `GGML_MAX_SRC` (10),
  `GGML_MAX_OP_PARAMS` (64) and `GGML_MAX_NAME` (64). `GGML_MAX_NAME` sits behind `#ifndef` and can be overridden
  (ggml.h:222-230), so every node must be built without overriding it.
- The client device always reports type GPU (`// TODO: obtain value from the server`, 2851-2856), and
  `supports_op` always returns `true` (2888-2893). The scheduler therefore assumes the remote backend supports
  every op. UNVERIFIED what a backend does with an unsupported op, but server compute failure is fatal (§3).

Conclusion: two builds of b11770 on macOS arm64 (Metal) and Windows x64 (CUDA or CPU) should interoperate over
TCP. This is a design inference and has not been hardware-validated here.

---

## 3. Client behaviour when a server is lost (mid-load or mid-inference)

### Verified: the client aborts the whole process. It does not return an error.

All traffic goes through one `rpc_dispatcher` thread per endpoint (708-724). Each command is checked like this:

```cpp
// ggml-rpc.cpp:31
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// ggml-rpc.cpp:680-686 (rpc_dispatcher::work)
if (msg_ptr->output) {
    bool status = send_rpc_cmd(sock, msg_ptr->cmd, ..., msg_ptr->output, msg_ptr->output_size);
    RPC_STATUS_ASSERT(status);
} else {
    bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size);
    RPC_STATUS_ASSERT(status);
}
```

`GGML_ABORT` -> `ggml_abort()` prints the message and a backtrace to stderr and calls `abort()`
(ggml.c:253-272). The process dies. On POSIX that is SIGABRT. UNVERIFIED: the exact Windows exit code
(the CRT default is usually 3).

Connection setup also aborts (`rpc_dispatcher::start`, 638-658):
```cpp
sock = socket_t::connect(host.c_str(), port);
if (sock == nullptr) { GGML_ABORT("Failed to connect to %s\n", endpoint.c_str()); }
if (!negotiate_hello(sock)) { GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str()); }
```

By phase:

| Phase | Trigger | Result (verified unless marked) |
|---|---|---|
| Startup (`--rpc` parse) | worker down or refusing | `GGML_ABORT("Failed to connect ...")` at arg-parse time (arg.cpp:1184-1186 -> 650) |
| Startup | version mismatch | `GGML_ABORT("RPC handshake failed ...")` (652-654) |
| Startup | worker busy with another client | **hang** in HELLO recv (no timeout), see §1 |
| Model load (`SET_TENSOR`, `SET_TENSOR_HASH`, `ALLOC_BUFFER`, ...) | worker process dies (FIN/RST) | `recv`/`send` fails -> `RPC_STATUS_ASSERT` -> abort |
| Inference (`GRAPH_COMPUTE` is fire-and-forget via `send_async`, 1266; reads via `GET_TENSOR`) | worker dies | the next send or recv fails in the dispatcher thread -> abort |
| Any phase | silent link loss (Wi-Fi drop, sleep, cable pull, no FIN/RST) | **likely hang**: blocking `recv` with no timeout or keepalive; callers wait on `future.wait()` with no timeout (554-556, 577-578). UNVERIFIED how long the OS takes to surface an error (Linux tcp_retries2 is about 15 min if unacked data is pending; a pure recv wait can block indefinitely) |

Related verified facts:
- `recv_data` treats `n == 0` as an error ("recv returned 0 (peer closed?)") and `n < 0` as failure
  (transport.cpp:570-585). The TCP path has no retry or reconnect logic.
- A failed server allocation is not fatal. `ALLOC_BUFFER` returning `remote_ptr == 0` makes the client
  return `nullptr` with a `LOG_WARN` (990-1003). This is the one recoverable server-side failure.
- A failed server compute is fatal to the **server**:
  `GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");`
  (2192, 2211, 2349). The server abort then makes the client abort.
- Possible extra server-death path (inferred, not tested): neither transport.cpp, rpc-server.cpp, ggml-rpc.cpp
  nor ggml.c mentions `SIGPIPE`, `MSG_NOSIGNAL` or `SO_NOSIGPIPE` (grep came back empty), and `send(fd, ..., 0)`
  uses flags 0 (transport.cpp:548). On Linux and macOS, writing to a socket the client already closed raises
  SIGPIPE, whose default action terminates the process. A client that vanishes in the middle of a response
  (for example `GET_TENSOR`) could therefore kill `ggml-rpc-server`. Windows has no SIGPIPE.

Planning consequence: run `llama-server` as a supervised child, treat its death as a session failure, probe
TCP and HELLO before launching, keep an application-level heartbeat and token-progress watchdog that kills a hung
`llama-server`, and restart workers that die.

---

## 4. `--cache` (`-c`)

### What is cached (verified)

- Only **weight** tensors over **10 MiB**:
  ```cpp
  // ggml-rpc.cpp:117-118
  const size_t HASH_THRESHOLD = 10 * 1024 * 1024;
  // ggml-rpc.cpp:844-846
  static bool rpc_use_hash_cache(const ggml_tensor * tensor, size_t size) {
      return size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
  }
  ```
  Activations are excluded on purpose (comment at 840-843).
- Protocol (848-873): the client computes a 64-bit **FNV-1a** hash of the whole tensor payload
  (`fnv_hash`, 313-321) and sends `SET_TENSOR_HASH {tensor, offset, hash}`. If the server has a file with that
  name, it loads it into the tensor and replies `result=1`, and the client skips the upload. Otherwise the client
  sends `SET_TENSOR` with `cache_flag=1`, and the server writes the payload to `<cache_dir>/<016x hash>`
  (1742-1751).
- The client always takes the hash path for large weights, even when the server runs without `-c`. That server
  answers "miss" (`get_cached_file` returns false when `cache_dir` is null, 1819-1821) and ignores `cache_flag`.
  The cost is one extra round trip per large tensor.

### Where (verified, rpc-server.cpp:64-83 and 243)

| Condition | Directory |
|---|---|
| `LLAMA_CACHE` set | `$LLAMA_CACHE/rpc` (no `llama.cpp` component: early `return cache_directory;`) |
| Windows | `%LOCALAPPDATA%\llama.cpp\rpc` (throws if `LOCALAPPDATA` is unset) |
| macOS | `$HOME/Library/Caches/llama.cpp/rpc` |
| Linux and other | `$XDG_CACHE_HOME/llama.cpp/rpc`, else `$HOME/.cache/llama.cpp/rpc` |

The README's "`$HOME/.cache/llama.cpp/rpc`" (README.md:96) is correct only for Linux. The directory is created at
startup, and failure returns 1 (rpc-server.cpp:242-249).

### Persistence and integrity (verified unless marked)

- Files persist after the server exits. No code in ggml-rpc.cpp or rpc-server.cpp deletes, evicts or caps cache
  files (grep for `remove|unlink|evict` matches nothing about files; the only eviction is the in-memory graph
  cache). The cache grows without bound, about one model's weights per distinct model.
- The write is a plain `std::ofstream ofs(cache_file, std::ios::binary); ofs.write(...)` with no temp file,
  rename or fsync (1747-1749). Risk (inferred): a crash mid-write leaves a truncated file. On a later hit,
  `get_cached_file` loads whatever size the file has (1829-1834), bounds-checked only against the buffer
  (1868-1879). That is silent weight corruption.
- FNV-1a 64 is not collision-resistant. Risk (inferred): with a cache directory shared between sessions, a
  malicious client could plant colliding content that a later client's weights resolve to.
- The cache is keyed only by content hash, not by model or offset, so identical tensors are shared across
  models.

---

## 5. Multi-device servers, `-d`, defaults, memory reporting

### One server, several devices (verified)

- `-d/--device` takes names separated by `,` or `/` (regex `[,/]+`, rpc-server.cpp:132-147). Each name is
  resolved with `ggml_backend_dev_by_name` (174). On an unknown name it prints the available devices
  (name, description, total and free MiB) to **stdout** and returns empty, then `main` prints "No devices found"
  and returns 1 (176-187, 235-238).
- Default when `-d` is absent: **every device whose type is neither CPU nor ACCEL** (so GPU and IGPU), and only
  if there are none, the CPU device (rpc-server.cpp:191-208). README.md:36-37 says the same.
- The server builds one backend per device (ggml-rpc.cpp:2776-2795). `-t/--threads` (default
  `hardware_concurrency()/2`, rpc-server.cpp:98) is applied through `ggml_backend_set_n_threads` to whichever
  backend exposes it (CPU).
- A GPU and the CPU can be served together, for example `-d CUDA0,CPU`. Device names: CUDA `CUDA<n>`
  (README.md:65-71), Metal `MTL<n>` (ggml-metal-device.m:1302).
- Every request carries a `uint32_t device` index, which is checked against `backends.size()`
  (for example 1463-1465, 2116-2118).

### Client view (verified)

- One TCP connection (dispatcher) per endpoint, shared by all of that endpoint's devices (708-724).
- `ggml_backend_rpc_add_server` asks `DEVICE_COUNT` and creates client devices named `RPC<global index>` with
  description = endpoint. The index comes from a process-wide counter, so it does not restart per server
  (3150-3183). Backends created with `ggml_backend_rpc_init` are named `RPC<dev>[<endpoint>]` (1351).
- `GGML_RPC_MAX_SERVERS 16` is defined in ggml-rpc.h:17 but is not referenced in ggml-rpc.cpp or in any other
  file read for this audit.
- llama.cpp splits weights and KV "in proportion to each device's available memory" unless `--tensor-split`
  is given (README.md:83).

### `GET_DEVICE_MEMORY` (verified)

The server returns `ggml_backend_dev_memory()` of the underlying device (ggml-rpc.cpp:2413-2426).

- **Metal** (ggml-metal-device.m:1501-1511):
  ```objc
  *total     = dev->mtl_device.recommendedMaxWorkingSetSize;
  size_t cur = dev->mtl_device.currentAllocatedSize;
  // it's possible to allocate more than `recommendedMaxWorkingSetSize`
  *free      = *total > cur ? *total - cur : 0;
  ```
  "total" is therefore Apple's recommended working set, which is below physical unified RAM. "free" subtracts only
  this process's Metal allocations: it ignores other apps' memory pressure and also shrinks as the RPC server
  itself allocates. The Metal device type is GPU (ggml-metal.cpp:692-694), so it is exposed by default.
  `GET_MAX_SIZE` comes from the buffer type. Metal sets `max_buffer_size = maxBufferLength`
  (ggml-metal-device.m:1287). UNVERIFIED that the Metal buft's get_max_size returns exactly that field.
- **CPU** (ggml-cpu.cpp:365-382): on Windows, `ullTotalPhys` and `ullAvailPhys`. On other platforms,
  `total = _SC_PHYS_PAGES * _SC_PAGE_SIZE` and **`free = total`** (comment: "free system memory is ill-defined").
  A CPU-only Mac or Linux worker therefore overstates free memory.
- **CUDA**: not opened in this audit. UNVERIFIED (expected `cudaMemGetInfo`).

Planning consequence: Hypercompute's resource discovery should query memory itself (for example `vm_stat` and
`sysctl hw.memsize` on macOS, NVML on NVIDIA) and pass an explicit `--tensor-split` instead of trusting the
proportional default.

---

## 6. Security

### Verified

- **No authentication, authorization, encryption or allowlist.** The only gate is the HELLO size check, which is
  not a secret.
- Non-loopback warning (rpc-server.cpp:224-232), printed whenever `host != "127.0.0.1"`:
  ```
  WARNING: Host ('%s') is != '127.0.0.1'
           Never expose the RPC server to an open network!
           This is an experimental feature and is not secure!
  ```
  The README makes the same point: "functionality is fragile and insecure. Never run the RPC server on an open
  network or in a sensitive environment!" (README.md:3-5).
- Deserialization checks that exist:
  - An unknown command `>= RPC_CMD_COUNT` disconnects (2478-2482).
  - Fixed-size requests must match the exact size (`recv_msg` with `size != msg_size` fails, 333-342).
  - `deserialize_tensor`: `type < GGML_TYPE_COUNT`, `blck_size != 0`, the buffer pointer must be in this
    connection's `buffers` set (otherwise it is nulled), and `[data, data+nbytes)` must lie inside the buffer,
    with an overflow check (1641-1685).
  - `set_tensor`, `get_tensor`, `set_tensor_hash`, `memset_tensor`, `set_tensor_2d`, `get_tensor_2d` and
    `copy_tensor` all bounds-check offset and size against the buffer and, for 2d/memset, the tensor, including
    overflow checks (1609-1628, 1729-1739, 1771-1811, 1867-1879, 1941-1953, 1979-2002, 2027-2042).
  - `graph_compute` validates header and array lengths before reading (2109-2135). `create_node` rejects
    unknown ids and a non-null `data` without a buffer (2050-2104).
  - `free_buffer`, `buffer_clear` and `buffer_get_base` require the pointer to be in the `buffers` set
    (1548-1591).
- Limits that are **missing**:
  - **No maximum message size.** Variable-length messages do `input.resize(size)` for any client-supplied
    64-bit size, catching only `std::bad_alloc` (344-356). `MAX_CHUNK_SIZE = 1 GiB` (transport.h:10) only splits
    individual `send()`/`recv()` calls (transport.cpp:547, 572) and does not cap a message. A peer can force a
    huge zero-filled allocation, which is a memory exhaustion DoS.
  - `ALLOC_BUFFER` accepts any size on any device (1502-1522).
  - The server returns raw server pointers to the client (`remote_ptr = reinterpret_cast<uint64_t>(buffer)`,
    1512; `base_ptr`, 1556), which leaks addresses.
  - A failed graph compute triggers `GGML_ASSERT`, so a crafted graph can crash the server (2192, 2211).
  - **`COMM_INIT` opens new network surface regardless of `-H`.** Rank 0 does
    `socket_t::create_server("0.0.0.0", request.port)` on a port the client chooses (2237). Rank 1 connects to a
    client-supplied `host[64]:port` and retries 100 times at 50 ms (2256-2266). A peer can therefore make the
    server listen on all interfaces, or connect outward to any address.
- UNVERIFIED (not analysed): whether `ggml_set_name(result, tensor->name)` (1693) is safe for a non-terminated
  64-byte wire name, and whether the error log at 1678-1682 can over-read `tensor->name`.

Planning consequence: bind workers only to a private overlay (Tailscale or WireGuard IP) or LAN IP, firewall
50052 and 51052 (comm port = port + 1000) to the session's hosts, never use `0.0.0.0` on untrusted networks, and
put Hypercompute's own authenticated control plane in front of who is allowed to connect.

---

## 7. Server-to-server, RDMA, multi-server features

### Pairwise allreduce over a direct server-to-server link (verified)

- New commands `COMM_INIT`, `COMM_ALLREDUCE` and `COMM_FREE` (78-80, 251-270, 2224-2411). Data moves between
  the two servers and "never passes through the client" (2967-2969).
- Exposed through `get_proc_address` names `ggml_backend_comm_init`, `ggml_backend_comm_free` and
  `ggml_backend_comm_allreduce_tensor` (3098-3114). They are used by the meta device
  (ggml-backend-meta.cpp:1857-1869), which llama.cpp creates for `-sm tensor` (src/llama.cpp:161-178, 193-219).
  arg.cpp:2821-2826 labels `tensor` as "parallelized, EXPERIMENTAL". The default split mode is `layer`
  (pipelined).
- Constraints (2993-3065): exactly **2** backends, both RPC, on **different endpoints** ("multiple ranks on
  endpoint ... are not supported"). The comm port is rank 0's port + 1000. Rank 1 connects to rank 0's
  `--rpc` host, so endpoints must be mutually reachable ("do not bind the servers to 127.0.0.1 when they run on
  different machines", 3016-3017). `GGML_RPC_NO_COMM` disables it. The server side requires `world == 2`
  (2226).
- F32 partials with 32768 or more elements are sent as bf16 on the wire (2314-2318).
- The comm link uses the same caps negotiation, so it can upgrade to RDMA (2222-2223).
- MVP relevance: not needed. The default `-sm layer` over 2 or more RPC servers uses only client<->server links.
  Tensor parallel across two Macs over Thunderbolt is a later-phase option.

### RDMA (verified from CMake, transport.cpp and README)

- Build-time auto-detect: Linux `libibverbs` (RoCEv2 or IB), macOS `librdma` (RDMA over Thunderbolt 5,
  macOS 26.2+, `rdma_ctl enable` from Recovery). The `GGML_RPC_RDMA` option defaults ON when the library is
  found. **Windows is forced OFF** (CMakeLists.txt:13-52, README.md:98-112). Apple builds link `librdma`
  weakly so the binary still loads on older macOS (CMakeLists.txt:38-45).
- Runtime: negotiated in HELLO caps and falls back to TCP silently. Disable with `GGML_RPC_NO_RDMA=1` on either
  peer. Linux device selection through `GGML_RDMA_DEV` and `GGML_RDMA_GID` (transport.cpp:215-216). RDMA is used
  only when the connection is made over the RDMA-capable interface (README.md:107).
- The server logs `transport : TCP (RDMA auto-negotiate enabled)` or `transport : TCP` (2803-2807).
  The negotiated transport per connection is visible at `GGML_RPC_DEBUG>=1` ("client connected, transport: %s",
  2473).

### `test-rpc-multi-server` (verified)

- Built only when `LLAMA_BUILD_TESTS AND UNIX AND NOT GGML_BACKEND_DL` (tools/rpc/CMakeLists.txt:6-16).
- It is a regression test, not a feature. It starts two CPU `ggml-rpc-server`s on 127.0.0.1 and checks that a
  tensor allocated on server A, when sent in a graph to server B, is not dereferenced by B (B's log must not
  contain "invalid data ptr"), and that cached alloc sizes are keyed by `nb[]`
  (tests/test-rpc-multi-server.sh:31-43, .cpp:14-63). It does confirm that one client process can hold several
  servers at once and that cross-server pointers are zeroed (`serialize_tensor` with a dispatcher, 762-772).
- Its `wait_for_port` loop (sh:18-29) is a usable pattern for Hypercompute's readiness probe.

### Other client-side behaviour worth knowing (verified)

- Graph caching: the client sends `GRAPH_RECOMPUTE {device, uid}` instead of re-serializing a graph the server
  already holds (1242-1268). `FREE_BUFFER` invalidates every stored graph on the server (1568-1575).
- Busy-spin dispatcher: on aarch64 it spins with `yield` for 100 ms after the last command, but only while
  `busy_spin_users > 0`, which is only set by comm (410-418, 660-674, 3028-3030).

---

## Exact native commands implied by this audit (for the MVP slice)

Worker (Windows + NVIDIA, built with `-DGGML_CUDA=ON -DGGML_RPC=ON` at b11770):
```
ggml-rpc-server.exe -H <worker-LAN-or-tailnet-IPv4> -p 50052 -d CUDA0 -c
```
Worker (macOS Apple Silicon, build with `-DGGML_RPC=ON`). `GGML_METAL` defaults ON under `if (APPLE)`
and `GGML_RPC` defaults OFF (ggml/CMakeLists.txt:95-96, 239, 250):
```
GGML_RPC_NO_RDMA=1 ./ggml-rpc-server -H <worker-IPv4> -p 50052 -d MTL0 -c
```
Host (any platform, built with `-DGGML_RPC=ON`):
```
llama-server -m <model.gguf> -ngl 99 --rpc <workerA-ip>:50052,<workerB-ip>:50052 --host 127.0.0.1 --port 8080
```
Notes: `-H` must be an IPv4 literal. Readiness has to come from a port and HELLO probe, not the exit code. Each
worker must be leased to exactly one host session. `GGML_RPC_DEBUG=1` on both sides plus `-lv 5` on the client
shows handshake and transport (README.md:130-136). These commands are derived from the source and have **not
been hardware-validated** here.

---

## Addendum (critic)

Critic pass 2026-10-20, same ref `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb`, files read with
`gh api -H "Accept: application/vnd.github.raw" "repos/ggml-org/llama.cpp/contents/<path>?ref=<sha>"`. Nothing built or run.
Gateway scope checked: (a) launch with `--rpc` + `--device`/`--tensor-split`, (b) parse load logs, (c) proxy streaming chat,
(d) probe workers with `--rpc host:port --list-devices`.

### Open items in this file

| Item | Matters for | Status after this pass |
|---|---|---|
| How long a `llama-server` keeps its TCP connection to a worker (lease semantics, probe collisions) | (a), (d) | RESOLVED in C1 below. Not asked in the original audit, and it refines TL;DR item 1 |
| CUDA `GET_DEVICE_MEMORY` semantics (§5 "UNVERIFIED (expected `cudaMemGetInfo`)") | (a) default split, (d) probe numbers | RESOLVED in C2 below |
| Windows exit code after `GGML_ABORT` -> `abort()` | supervisor error mapping | Still UNVERIFIED. It is CRT behaviour, not llama.cpp source. Needs a hardware run |
| OS behaviour beyond the listen backlog of 1, and `SO_REUSEADDR` double bind on Windows | worker leases, lockfile | Still UNVERIFIED. This is OS TCP-stack behaviour |
| Metal buft `get_max_size` | very large single tensors on Mac workers | Still UNVERIFIED. Low priority for the MVP models |
| "Exact native commands" host line uses `-ngl 99 --port 8080`, leaves `--fit` on and passes no `-ts` | (a) | Inconsistent with llama_placement.md §4 (`--fit off -ngl all -ts a,b`) and with llama_server.md (default port is 9931). Use the combined launch line in llama_server.md "Addendum (critic)" |

### C1. RPC connections are short-lived until RPC buffers exist (verified)

`get_dispatcher` caches each dispatcher only as a `std::weak_ptr`:

```cpp
// ggml/src/ggml-rpc/ggml-rpc.cpp:708-722
static std::shared_ptr<rpc_dispatcher> get_dispatcher(const std::string & endpoint) {
    ...
    static std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;   // :711
    ...
    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);          // TCP connect + HELLO
```

A dispatcher, and so its socket, lives only while something holds a `shared_ptr` to it. Only two structs keep one:
`ggml_backend_rpc_context` (backend, ggml-rpc.cpp:297-302) and `ggml_backend_rpc_buffer_context` (buffer, 304-308).
The device context (281-286) and the buffer-type context (288-294) store only the endpoint string.

Several short calls take a temporary `shared_ptr` and drop it:
- `ggml_backend_rpc_get_device_count` (3136-3141), during `--rpc` parsing: `auto dispatcher = get_dispatcher(endpoint); ... send(RPC_CMD_DEVICE_COUNT ...); return ...;`
- `ggml_backend_rpc_get_device_memory` (1374-1384). The device `get_memory` (2845-2849) calls it.
- `ggml_backend_rpc_buffer_type` (1328): GET_ALIGNMENT and GET_MAX_SIZE, after which the buft is cached forever in `buft_map`.
- `GET_ALLOC_SIZE` (1108).

When that temporary goes out of scope, `~rpc_dispatcher` (699-706) sets `running = false`, interrupts the queue, sets
`sock = nullptr` (which closes the socket) and joins the thread. `alloc_buffer` (981-1003) and `ggml_backend_rpc_init`
(1349-1358) keep the dispatcher. So from the first real RPC buffer onward, each endpoint has one persistent connection,
and it closes only after every RPC buffer and backend has been freed.

Consequences for the gateway:
1. Probe cost. `llama-server --rpc A:50052 --list-devices` opens **1 + N** separate TCP connections to A, where N is the
   number of devices A exposes. One carries `DEVICE_COUNT` during `--rpc` parsing. Then each `ggml_backend_dev_memory` call
   in `common_print_available_devices` (common/arg.cpp:1162-1166) gets its own connection. Each connection does a full
   HELLO, and each makes the worker print one `Accepted client connection` / `Client connection closed` pair (2823, 2825).
   When the handler then calls `exit(0)` (arg.cpp:2753-2754), no dispatcher is still alive.
2. Leases cannot be inferred from connections. A normal launch also opens and closes several connections before the model
   is resident: the parse-time `DEVICE_COUNT`, the device-memory query behind `using device ... MiB free`, the
   free-memory default split, and every `--fit` probe memory query. Only after that does it hold one persistent
   connection. **Between those connections the worker's sequential accept loop is free**, so another client can be
   served in the gap. If that client is a second `llama-server` that allocates buffers, the first one's next connect
   blocks in HELLO with no timeout (§1) until the second one exits. Hypercompute must enforce exclusive worker leases in
   its own control plane. "Worker currently connected" is not a lease.
3. A worker busy/idle indicator built on the `Accepted`/`closed` log lines will flap during probes and model load. Use
   the gateway's lease table instead, or "a connection has stayed open for several seconds".
4. A `--list-devices` probe against a worker that is serving another session's persistent connection **hangs**. It does
   not fail. Run every probe with a hard wall-clock timeout plus process kill, and never probe a worker that is leased to
   a running session.
5. Correction to llama_placement.md §1.4 ("their TCP connection was still opened at parse time"): the parse-time
   connection to an RPC server that is not named in `--device` is opened **and closed** during parsing. INFERRED: no
   persistent connection follows, because no buffer or backend is created for a device outside `model->devices`.

### C2. CUDA device memory as seen by `--list-devices`, `--fit` and the default split (verified)

The rpc-server answers `GET_DEVICE_MEMORY` with its device's `ggml_backend_dev_memory` (§5). For CUDA that is
`ggml_backend_cuda_device_get_memory`, `ggml/src/ggml-cuda/ggml-cuda.cu:5448-5490`:

```cpp
cudaError_t err = cudaMemGetInfo(free, total);                                  // :5451
if (err != cudaSuccess) {
    GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", ...);      // :5454
    *free = 0; *total = 0; return;
}
#if defined(__linux__) && !defined(GGML_USE_HIP)
    bool is_uma = prop.integrated > 0 || ggml_cuda_unified_memory_enabled();    // GGML_CUDA_ENABLE_UNIFIED_MEMORY, :156
    if (is_uma) { ... *free = (size_t)available_memory_kb * 1024; ... }         // /proc/meminfo, :5474-5478
#endif
const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
*free  /= share_count;
*total /= share_count;                                                          // :5484-5487
```

- Windows/NVIDIA worker: `free` and `total` come straight from `cudaMemGetInfo`. INFERRED from CUDA runtime semantics,
  not from llama.cpp source: this is device-wide free memory, so other processes using the GPU (desktop, games) are
  already subtracted. Metal is different: its "free" subtracts only the rpc-server's own allocations (§5).
- On a `cudaMemGetInfo` failure the device still appears, with `0 MiB, 0 MiB free`. A 0/0 device is skipped by `--fit`
  (`did not report memory; --fit will not use it`, llama_placement.md §4). In the free-memory default split it falls
  back to host memory (llama_placement.md §1.5). The gateway should reject a probe result with `total == 0` instead of
  scheduling layers onto that device.
- Linux integrated or UMA CUDA (the source comment names DGX Spark; the GB10-based GX10 is INFERRED to take this path):
  `free` becomes `/proc/meminfo` `MemAvailable`, or free hugepages times the page size when hugetlbfs is configured
  (5374-5440). `total` stays the `cudaMemGetInfo` value. Such a device reports type `IGPU` (5492-5501). On a host it is
  added to the model device list only if no discrete GPU exists (llama_placement.md §1.3). On a worker it is exposed by
  default (`tools/rpc/rpc-server.cpp:191-200` accepts every type except CPU and ACCEL).
- Virtual CUDA devices that share one physical GPU split both numbers by the share count.
- Gateway implication: `--list-devices` MiB values are live and floored to whole MiB. They cannot be compared across
  platforms (CUDA is device-wide, Metal is per-process working set, CPU on POSIX is free = total). Compute
  `--tensor-split` from Hypercompute's own per-platform model with explicit margins rather than from raw free MiB.
