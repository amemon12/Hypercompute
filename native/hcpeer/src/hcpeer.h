#pragma once

// hcpeer: file-backed CPU weights for ggml-rpc-server on macOS (Hypercompute, server side only, no protocol change).
//
// With HC_PEER_MMAP=1 and a cache directory (rpc-server -c), a weight that the client sends through the RPC tensor
// cache (SET_TENSOR_HASH hit, or SET_TENSOR with the cache flag) to a tensor in a CPU buffer is mapped from its cache
// file with mmap(MAP_FIXED | MAP_PRIVATE) instead of copied into the buffer's anonymous memory. Untouched pages stay
// clean and file-backed, so macOS can drop them under memory pressure and fault them back in from the SSD. Metal and
// every other buffer type keep the stock copy. Any failure falls back to the stock copy.
//
// Every cache file the server saves (any device, with or without HC_PEER_MMAP) is written to a new temporary file,
// fsynced and renamed into place, so a killed server or a full disk never leaves a short file under a hash name, and
// a tensor mapped from an older file keeps its pages. While enabled, a save also has to leave the cache volume
// HC_PEER_CACHE_MIN_FREE_GIB (default 20) free and keep the cache directory within HC_PEER_CACHE_MAX_GIB (unset: no
// cap), because a mapped file holds its disk until it is unmapped and no eviction can win that back. Every mapped file
// gets its access time set to now, so LRU eviction takes the files not in use first. A cache hit whose file cannot
// hold the data asked for is deleted and answered as a miss, so the client resends the data.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

#define HCPEER_VERSION "4"

// What the caller does after a hook.
enum hcpeer_action {
    HCPEER_STOCK = 0,  // hcpeer did nothing: run the stock code
    HCPEER_DONE  = 1,  // the tensor holds the data: answer as the stock success path does
    HCPEER_COPY  = 2,  // the cache file is dealt with: copy the payload into the tensor, skip the stock cache write
    HCPEER_MISS  = 3,  // the cache file was unusable (and removed): answer SET_TENSOR_HASH as a cache miss
};

// True when HC_PEER_MMAP=1 and the platform is supported (macOS). Read once.
bool hcpeer_enabled();

// Prints one banner line, "  hcpeer mmap    : on (...)" or "  hcpeer mmap    : off (...)". With a cache directory it
// first removes the temporary cache files that stopped servers left there (on or off: saves are atomic either way).
void hcpeer_log_banner(const char * cache_dir);

// The alignment to report for a device's default buffer type: at least the page size for the CPU buffer type when
// enabled with a cache directory, so the client places every weight on a page boundary. Otherwise `alignment`.
size_t hcpeer_alignment(ggml_backend_buffer_type_t buft, size_t alignment, bool has_cache_dir);

// SET_TENSOR_HASH, which carries no size: check cache_dir/<hash as %016x> against the bytes the request can mean (the
// rest of the tensor from offset, or whole outermost slices of a tensor of three or more dims, as MoE cache bank fills
// send), then map it over a CPU tensor's bytes. HCPEER_DONE when the tensor now reads from the file (a served hit),
// HCPEER_MISS when the file cannot be that data (deleted) or cannot be read, HCPEER_STOCK otherwise (disabled, no such
// file, another buffer type, or mapping fell back), which leaves the tensor untouched for the stock path.
int hcpeer_map_cached(ggml_tensor * tensor, uint64_t offset, const char * cache_dir, uint64_t hash);

// SET_TENSOR with the cache flag: save cache_dir/name atomically, then, when enabled, map it over a CPU tensor's bytes.
// HCPEER_DONE when the tensor reads from the saved file, HCPEER_COPY otherwise (disabled, saved for another device,
// save skipped by the floor or the cap or failed, or mapping fell back). HCPEER_STOCK only where hcpeer is unsupported.
int hcpeer_save_cached(ggml_tensor * tensor, uint64_t offset, const void * data, size_t size, const char * cache_dir,
                       const char * name);
