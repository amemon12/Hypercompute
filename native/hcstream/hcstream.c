/*
 * hcstream: bounded CPU weight residency for stock llama.cpp b11770 on Windows.
 *
 * Built as a proxy ggml.dll. The 16 ggml.dll exports are thin wrappers around the real library
 * (renamed ggml_hcorig.dll in the overlay engine directory). When HC_STREAM_CONFIG names a config
 * file, the first wrapper call patches import tables so that:
 *
 *   - llama_init_from_model installs a ggml scheduler eval callback on the new context;
 *   - llama_free drains the reader pool, restores the original (file-mapped) tensor pointers and
 *     frees every buffer this extension allocated;
 *   - optionally, GetProcAddress("PrefetchVirtualMemory") in llama.dll returns a no-op, so the
 *     loader does not pull the whole GGUF into the file cache.
 *
 * Layers named in the config are either "resident" (copied once into private memory with
 * unbuffered reads) or "stream" (read from the GGUF with FILE_FLAG_NO_BUFFERING into one of a
 * fixed number of slot buffers just before their first consumer runs). Streamed bytes never
 * enter the OS file cache, so physical SSD reads are real and the memory bound is
 * slots * slot_bytes + resident bytes. Weight tensors keep their CPU_Mapped buffer, so the
 * scheduler still assigns their ops to the CPU backend; only tensor->data is rebound.
 *
 * Correctness argument for slot reuse: the eval callback is asked about each node before any node
 * of the pending range runs, and the scheduler synchronizes after every range it computes. We
 * track which streamed layers are referenced by pending (asked, not yet computed) nodes and never
 * evict a slot holding one of them. With two or more slots we stop (need = true) when a second
 * layer enters the pending range, so at most two layers are pinned. With one slot every streamed
 * consumer is a stop, so nothing is pinned when the next layer is bound. An evicted layer's
 * tensors point back at the file mapping, which always holds correct bytes.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

#define HC_VERSION 1
#define HC_ALIGN 4096ull
#define HC_MAX_SHARDS 64
#define HC_MAX_LAYERS 1024
#define HC_MAX_READERS 16
#define HC_PTR_CAP 16384 /* power of two, > 2x weight tensors of any supported model */
#define HC_MAX_NEED 4

enum { KIND_NONE = 0, KIND_RESIDENT = 1, KIND_STREAM = 2 };
enum { SLOT_FREE = 0, SLOT_LOADING = 1, SLOT_READY = 2, SLOT_ERROR = 3 };

typedef struct {
    char name[GGML_MAX_NAME];
    int layer;
    int shard;
    uint64_t offset; /* absolute file offset of the tensor data */
    uint64_t nbytes; /* exact ggml_nbytes */
    uint64_t span;   /* bytes until the next tensor (>= nbytes) */
    uint64_t slot_off; /* offset of the tensor data inside its layer buffer */
    struct ggml_tensor *t; /* registered model tensor, NULL until the graph references it */
    void *orig;            /* file-mapped data pointer at registration */
    int verified;
} hc_tensor;

typedef struct {
    int shard;
    uint64_t file_off; /* aligned */
    uint64_t len;      /* aligned */
    uint64_t buf_off;  /* aligned position inside the layer buffer */
} hc_run;

typedef struct {
    int kind;
    int first_tensor, n_tensors;
    hc_run *runs;
    int n_runs;
    uint64_t buf_bytes; /* aligned bytes needed to hold the layer */
    char *resident_buf;
    int slot;            /* stream slot holding the layer (loading or ready) or -1 */
    int bound;           /* tensors currently point into the slot */
    uint64_t last_bound; /* bind tick, for victim choice */
} hc_layer;

typedef struct {
    char *buf;
    uint64_t cap;
    int layer;
    volatile LONG state;
    volatile LONG pending_chunks;
    volatile LONG failed;
    uint64_t epoch;
} hc_slot;

typedef struct {
    char *dst;
    int shard;
    uint64_t file_off;
    uint64_t len;
    hc_slot *slot; /* NULL for resident loads */
    volatile LONG *pending;
    volatile LONG *failed;
    uint64_t epoch;
} hc_task;

typedef struct hc_state hc_state;

typedef struct {
    hc_state *st;
    int index;
    HANDLE files[HC_MAX_SHARDS];
} hc_reader;

struct hc_state {
    /* config */
    int slots_n;
    uint64_t max_slot_bytes, max_resident_bytes;
    int readers_n;
    uint64_t chunk_bytes;
    int no_prefetch;
    int io_error_mmap_fallback;
    WCHAR *stats_wpath; /* stats destination, NULL when stats are off */
    DWORD stats_interval_ms;
    uint64_t fault_error_after_bytes;
    DWORD fault_delay_us;
    int n_shards;
    WCHAR *shard_paths[HC_MAX_SHARDS];
    int n_layers;
    hc_layer layers[HC_MAX_LAYERS];
    hc_tensor *tensors;
    int n_tensors;
    int *stream_order; /* streamed layer ids in ascending order */
    int n_stream;

    /* runtime */
    struct llama_context *ctx;
    ggml_backend_sched_eval_callback inner_cb;
    void *inner_ud;
    ggml_abort_callback inner_abort;
    void *inner_abort_ud;
    struct ggml_tensor *last_ask_node;
    int last_inner_need;
    int started, shut_down;
    volatile LONG fatal;
    hc_slot *slots;
    uint64_t slot_bytes;
    int pending[HC_MAX_NEED * 2];
    int n_pending;
    int newest_layer;
    uint64_t tick;

    struct { const void *key; int idx; } ptrs[HC_PTR_CAP]; /* idx >= 0 tensor, -1 unmanaged */
    int *name_index; /* open addressing over tensors by name hash */
    int name_cap;

    /* reader pool */
    SRWLOCK qlock;
    CONDITION_VARIABLE qcv;   /* readers wait for tasks */
    CONDITION_VARIABLE donecv; /* decode thread waits for slot completion */
    hc_task *queue;
    int q_cap, q_head, q_len;
    volatile LONG inflight;
    volatile LONG stop;
    HANDLE threads[HC_MAX_READERS];
    hc_reader readers[HC_MAX_READERS];
    HANDLE stats_thread;
    HANDLE stats_stop;
    DWORD stats_thread_id;

    /* counters */
    volatile LONG last_error;
    volatile LONG64 last_error_offset;
    volatile LONG64 bytes_read, read_ns, chunks, stall_ns, layer_loads, sync_loads, prefetch_loads,
        prefetch_hits, binds, evictions, passes, io_errors, cancelled_tasks, resident_bytes,
        verify_failures, mmap_fallbacks, prefetch_noop_calls, prefetch_noop_bytes, fault_injected,
        slot_bytes_committed;
};

/* ------------------------------------------------------------------------------------------ */
/* original ggml.dll and logging                                                                */

static HMODULE g_self;
static HMODULE g_orig;
static INIT_ONCE g_orig_once = INIT_ONCE_STATIC_INIT;
static INIT_ONCE g_hook_once = INIT_ONCE_STATIC_INIT;
static hc_state *g_cfg;    /* parsed config, NULL when the capability is off */
static hc_state *g_active; /* attached to a live context */
static int g_no_prefetch;
static volatile LONG64 g_prefetch_noop_calls, g_prefetch_noop_bytes;
static LARGE_INTEGER g_qpf;

typedef struct llama_context *(*init_from_model_fn)(struct llama_model *, struct llama_context_params);
typedef void (*llama_free_fn)(struct llama_context *);
typedef FARPROC(WINAPI *get_proc_fn)(HMODULE, LPCSTR);
typedef bool (*buffer_is_host_fn)(ggml_backend_buffer_t);
typedef const char *(*buffer_name_fn)(ggml_backend_buffer_t);
typedef size_t (*nbytes_fn)(const struct ggml_tensor *);

static init_from_model_fn g_real_init;
static llama_free_fn g_real_free;
static get_proc_fn g_real_get_proc;
static buffer_is_host_fn g_buffer_is_host;
static buffer_name_fn g_buffer_name;
static nbytes_fn g_nbytes;

static void hc_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("hcstream: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
}

#define HC_DIE_JOIN_MS 2000
#define HC_STATS_RENAME_TRIES 100 /* 10 ms apart: how long a lifecycle snapshot waits for a reader to close the file */

static volatile LONG g_dying;

static void hc_write_stats(hc_state *st, const char *state);
static int hc_stop_stats(hc_state *st, DWORD wait_ms);

/*
 * Log the reason, write the final "fatal" stats snapshot and exit with code 86.
 *
 * Deadlock reasoning for the final snapshot:
 *   - The stats thread is told to stop and joined with a bounded wait before the fatal write, so its periodic
 *     "running" write cannot land on top of the fatal one. hc_stop_stats never waits on the calling thread, so
 *     hc_die is safe on the stats thread itself.
 *   - The stats thread takes none of this file's locks: hc_write_stats reads every counter without qlock. A caller
 *     holding qlock therefore cannot block it. (No hc_die call is made under qlock today; hc_enqueue_layer releases
 *     it first.) The only locks the stats thread takes are the CRT's stdio and heap locks, which no caller holds
 *     across a call into this file. The join is still bounded, so a stats thread stuck in file I/O delays the exit
 *     by at most HC_DIE_JOIN_MS.
 *   - If the join times out, the stats thread is still alive. It cannot clobber the fatal snapshot: every write
 *     goes through its own per-state temporary file, and a non-fatal write is dropped once st->fatal is set
 *     (checked again just before the rename).
 *   - The fatal rename is retried for up to HC_STATS_RENAME_TRIES x 10 ms while a reader holds the stats file open
 *     (see hc_publish_stats), so the whole fatal path is bounded by the join, that retry and file system latency.
 *   - Only the first failing thread writes the snapshot. A second caller waits longer than the first one's bounded
 *     waits for it to end the process and then exits itself.
 */
static __declspec(noreturn) void hc_die(hc_state *st, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("hcstream: FATAL: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
    if (InterlockedCompareExchange(&g_dying, 1, 0) != 0) {
        Sleep(HC_DIE_JOIN_MS * 2 + HC_STATS_RENAME_TRIES * 10);
        ExitProcess(86);
    }
    if (st) {
        InterlockedExchange(&st->fatal, 1);
        hc_stop_stats(st, HC_DIE_JOIN_MS);
        hc_write_stats(st, "fatal");
    }
    ExitProcess(86);
}

static uint64_t hc_now_ns(void) {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (uint64_t)((double)c.QuadPart * 1e9 / (double)g_qpf.QuadPart);
}

static BOOL CALLBACK hc_load_orig(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once; (void)param; (void)ctx;
    WCHAR path[MAX_PATH];
    DWORD n = GetModuleFileNameW(g_self, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return FALSE;
    WCHAR *slash = wcsrchr(path, L'\\');
    if (!slash) return FALSE;
    wcscpy_s(slash + 1, MAX_PATH - (slash + 1 - path), L"ggml_hcorig.dll");
    g_orig = LoadLibraryW(path);
    QueryPerformanceFrequency(&g_qpf);
    return g_orig != NULL;
}

static FARPROC hc_orig(const char *name) {
    if (!InitOnceExecuteOnce(&g_orig_once, hc_load_orig, NULL, NULL) || !g_orig) hc_die(NULL, "cannot load ggml_hcorig.dll next to the proxy");
    FARPROC p = GetProcAddress(g_orig, name);
    if (!p) hc_die(NULL, "ggml_hcorig.dll lacks %s", name);
    return p;
}

/* ------------------------------------------------------------------------------------------ */
/* config parsing                                                                              */

static uint64_t hc_align_down(uint64_t v) { return v & ~(HC_ALIGN - 1); }
static uint64_t hc_align_up(uint64_t v) { return (v + HC_ALIGN - 1) & ~(HC_ALIGN - 1); }

static uint32_t hc_hash_str(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) h = (h ^ (uint8_t)*s) * 16777619u;
    return h;
}

static int hc_find_name(hc_state *st, const char *name) {
    uint32_t mask = (uint32_t)st->name_cap - 1;
    for (uint32_t i = hc_hash_str(name) & mask;; i = (i + 1) & mask) {
        int idx = st->name_index[i];
        if (idx < 0) return -1;
        if (strcmp(st->tensors[idx].name, name) == 0) return idx;
    }
}

/* UTF-8 to UTF-16; NULL for invalid UTF-8 or out of memory. */
static WCHAR *hc_widen(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    WCHAR *w = (WCHAR *)malloc(sizeof(WCHAR) * (size_t)n);
    if (w && MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, n) != n) {
        free(w);
        w = NULL;
    }
    return w;
}

/* UTF-16 to UTF-8, for messages only; NULL on failure. */
static char *hc_narrow(const WCHAR *w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if (n <= 0) return NULL;
    char *s = (char *)malloc((size_t)n);
    if (s && WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL) != n) {
        free(s);
        s = NULL;
    }
    return s;
}

static int cmp_tensor_offset(const void *a, const void *b) {
    const hc_tensor *x = (const hc_tensor *)a, *y = (const hc_tensor *)b;
    if (x->layer != y->layer) return x->layer - y->layer;
    if (x->shard != y->shard) return x->shard - y->shard;
    return x->offset < y->offset ? -1 : x->offset > y->offset;
}

/* Merge each layer's tensors into aligned file runs and lay them out inside one layer buffer. */
static void hc_build_runs(hc_state *st) {
    qsort(st->tensors, (size_t)st->n_tensors, sizeof(hc_tensor), cmp_tensor_offset);
    for (int i = 0; i < st->n_tensors; i++) {
        hc_tensor *t = &st->tensors[i];
        hc_layer *L = &st->layers[t->layer];
        if (L->n_tensors == 0) L->first_tensor = i;
        L->n_tensors++;
    }
    for (int l = 0; l < st->n_layers; l++) {
        hc_layer *L = &st->layers[l];
        L->slot = -1;
        if (!L->n_tensors) continue;
        L->runs = (hc_run *)calloc((size_t)L->n_tensors, sizeof(hc_run));
        if (!L->runs) hc_die(st, "out of memory building runs");
        uint64_t pos = 0;
        for (int k = 0; k < L->n_tensors; k++) {
            hc_tensor *t = &st->tensors[L->first_tensor + k];
            uint64_t a = hc_align_down(t->offset), b = hc_align_up(t->offset + t->span);
            hc_run *r = L->n_runs ? &L->runs[L->n_runs - 1] : NULL;
            if (r && r->shard == t->shard && a <= r->file_off + r->len) {
                if (b > r->file_off + r->len) {
                    pos += b - (r->file_off + r->len);
                    r->len = b - r->file_off;
                }
            } else {
                r = &L->runs[L->n_runs++];
                r->shard = t->shard;
                r->file_off = a;
                r->len = b - a;
                r->buf_off = pos;
                pos += r->len;
            }
            t->slot_off = r->buf_off + (t->offset - r->file_off);
        }
        L->buf_bytes = pos;
    }
}

/* wpath is the config file name as given in the environment. The file itself is UTF-8 text. */
static hc_state *hc_parse_config(const WCHAR *wpath) {
    char *path = hc_narrow(wpath); /* for messages */
    if (!path) hc_die(NULL, "HC_STREAM_CONFIG is not a valid path");
    FILE *f = _wfopen(wpath, L"rb");
    if (!f) hc_die(NULL, "cannot open HC_STREAM_CONFIG %s", path);
    hc_state *st = (hc_state *)calloc(1, sizeof(hc_state));
    if (!st) hc_die(NULL, "out of memory");
    st->slots_n = 2;
    st->readers_n = 4;
    st->chunk_bytes = 8ull << 20;
    st->stats_interval_ms = 500;
    st->newest_layer = -1;
    int cap = 4096, line_no = 0, header = 0, ended = 0;
    st->tensors = (hc_tensor *)calloc((size_t)cap, sizeof(hc_tensor));
    if (!st->tensors) hc_die(NULL, "out of memory");
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        line_no++;
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (!n || line[0] == '#') continue;
        char key[64] = {0};
        int consumed = 0;
        if (sscanf(line, "%63s %n", key, &consumed) < 1) continue;
        const char *rest = line + consumed;
        unsigned long long a = 0, b = 0, c = 0, d = 0, e = 0;
        if (!strcmp(key, "hcstream")) {
            if (sscanf(rest, "%llu", &a) != 1 || a != HC_VERSION) hc_die(NULL, "config line %d: unsupported version", line_no);
            header = 1;
        } else if (!header) {
            hc_die(NULL, "config line %d: missing 'hcstream 1' header", line_no);
        } else if (!strcmp(key, "mode")) {
            if (strcmp(rest, "stream") != 0) hc_die(NULL, "config line %d: unknown mode '%s'", line_no, rest);
        } else if (!strcmp(key, "slots") && sscanf(rest, "%llu", &a) == 1 && a >= 1 && a <= 8) {
            st->slots_n = (int)a;
        } else if (!strcmp(key, "max_slot_bytes") && sscanf(rest, "%llu", &a) == 1) {
            st->max_slot_bytes = a;
        } else if (!strcmp(key, "max_resident_bytes") && sscanf(rest, "%llu", &a) == 1) {
            st->max_resident_bytes = a;
        } else if (!strcmp(key, "readers") && sscanf(rest, "%llu", &a) == 1 && a >= 1 && a <= HC_MAX_READERS) {
            st->readers_n = (int)a;
        } else if (!strcmp(key, "chunk_bytes") && sscanf(rest, "%llu", &a) == 1 && a >= HC_ALIGN && a % HC_ALIGN == 0) {
            st->chunk_bytes = a;
        } else if (!strcmp(key, "no_prefetch") && sscanf(rest, "%llu", &a) == 1) {
            st->no_prefetch = a != 0;
        } else if (!strcmp(key, "on_io_error")) {
            if (!strcmp(rest, "mmap")) st->io_error_mmap_fallback = 1;
            else if (strcmp(rest, "abort") != 0) hc_die(NULL, "config line %d: on_io_error must be abort or mmap", line_no);
        } else if (!strcmp(key, "stats")) {
            free(st->stats_wpath);
            st->stats_wpath = NULL;
            if (rest[0] && !(st->stats_wpath = hc_widen(rest))) hc_die(NULL, "config line %d: stats path is not valid UTF-8", line_no);
        } else if (!strcmp(key, "stats_interval_ms") && sscanf(rest, "%llu", &a) == 1 && a >= 50) {
            st->stats_interval_ms = (DWORD)a;
        } else if (!strcmp(key, "fault_read_error_after_bytes") && sscanf(rest, "%llu", &a) == 1) {
            st->fault_error_after_bytes = a;
        } else if (!strcmp(key, "fault_read_delay_us") && sscanf(rest, "%llu", &a) == 1) {
            st->fault_delay_us = (DWORD)a;
        } else if (!strcmp(key, "shard")) {
            int idx = -1, off = 0;
            if (sscanf(rest, "%d %n", &idx, &off) < 1 || idx != st->n_shards || idx >= HC_MAX_SHARDS || !rest[off])
                hc_die(NULL, "config line %d: shards must be listed in order with a path", line_no);
            st->shard_paths[st->n_shards++] = hc_widen(rest + off);
        } else if (!strcmp(key, "layer")) {
            char kind[16] = {0};
            if (sscanf(rest, "%llu %15s", &a, kind) != 2 || a >= HC_MAX_LAYERS) hc_die(NULL, "config line %d: bad layer", line_no);
            int k = !strcmp(kind, "resident") ? KIND_RESIDENT : !strcmp(kind, "stream") ? KIND_STREAM : !strcmp(kind, "none") ? KIND_NONE : -1;
            if (k < 0) hc_die(NULL, "config line %d: layer kind must be resident, stream or none", line_no);
            st->layers[a].kind = k;
            if ((int)a + 1 > st->n_layers) st->n_layers = (int)a + 1;
        } else if (!strcmp(key, "tensor")) {
            char name[GGML_MAX_NAME + 8] = {0};
            if (sscanf(rest, "%llu %llu %llu %llu %llu %71s", &a, &b, &c, &d, &e, name) != 6 || a >= HC_MAX_LAYERS ||
                (int)b >= st->n_shards || d == 0 || e < d || strlen(name) >= GGML_MAX_NAME)
                hc_die(NULL, "config line %d: bad tensor entry", line_no);
            if (st->n_tensors == cap) {
                cap *= 2;
                st->tensors = (hc_tensor *)realloc(st->tensors, sizeof(hc_tensor) * (size_t)cap);
                if (!st->tensors) hc_die(NULL, "out of memory");
            }
            hc_tensor *t = &st->tensors[st->n_tensors++];
            memset(t, 0, sizeof *t);
            strcpy(t->name, name);
            t->layer = (int)a;
            t->shard = (int)b;
            t->offset = c;
            t->nbytes = d;
            t->span = e;
            if ((int)a + 1 > st->n_layers) st->n_layers = (int)a + 1;
        } else if (!strcmp(key, "end")) {
            ended = 1;
            break;
        } else {
            hc_die(NULL, "config line %d: unknown or invalid entry '%s'", line_no, key);
        }
    }
    fclose(f);
    if (!ended) hc_die(NULL, "config %s is truncated (no 'end' line)", path);
    free(path);
    for (int i = 0; i < st->n_shards; i++)
        if (!st->shard_paths[i]) hc_die(NULL, "shard %d path is not valid UTF-8", i);
    hc_build_runs(st);
    st->name_cap = 1;
    while (st->name_cap < st->n_tensors * 2) st->name_cap <<= 1;
    st->name_index = (int *)malloc(sizeof(int) * (size_t)st->name_cap);
    if (!st->name_index) hc_die(NULL, "out of memory");
    memset(st->name_index, 0xff, sizeof(int) * (size_t)st->name_cap);
    uint32_t mask = (uint32_t)st->name_cap - 1;
    for (int i = 0; i < st->n_tensors; i++) {
        if (hc_find_name(st, st->tensors[i].name) >= 0) hc_die(NULL, "duplicate tensor %s", st->tensors[i].name);
        uint32_t h = hc_hash_str(st->tensors[i].name) & mask;
        while (st->name_index[h] >= 0) h = (h + 1) & mask;
        st->name_index[h] = i;
    }
    st->stream_order = (int *)calloc((size_t)st->n_layers + 1, sizeof(int));
    if (!st->stream_order) hc_die(NULL, "out of memory");
    uint64_t resident = 0;
    for (int l = 0; l < st->n_layers; l++) {
        hc_layer *L = &st->layers[l];
        if (L->kind != KIND_NONE && !L->n_tensors) hc_die(NULL, "layer %d is managed but lists no tensors", l);
        if (L->kind == KIND_STREAM) {
            st->stream_order[st->n_stream++] = l;
            if (L->buf_bytes > st->slot_bytes) st->slot_bytes = L->buf_bytes;
        } else if (L->kind == KIND_RESIDENT) {
            resident += L->buf_bytes;
        }
    }
    if (st->max_slot_bytes && st->slot_bytes > st->max_slot_bytes)
        hc_die(NULL, "largest streamed layer needs %llu bytes per slot, over the admitted %llu", (unsigned long long)st->slot_bytes,
               (unsigned long long)st->max_slot_bytes);
    if (st->max_resident_bytes && resident > st->max_resident_bytes)
        hc_die(NULL, "resident layers need %llu bytes, over the admitted %llu", (unsigned long long)resident,
               (unsigned long long)st->max_resident_bytes);
    return st;
}

/* ------------------------------------------------------------------------------------------ */
/* reader pool                                                                                  */

static HANDLE hc_open_shard(hc_state *st, hc_reader *r, int shard) {
    if (!r->files[shard]) {
        HANDLE h = CreateFileW(st->shard_paths[shard], GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                               OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
        if (h == INVALID_HANDLE_VALUE) return NULL;
        r->files[shard] = h;
    }
    return r->files[shard];
}

/* Record the error the fatal line reports. With several failing readers the last one wins. */
static void hc_note_error(hc_state *st, DWORD err, uint64_t off) {
    InterlockedExchange64(&st->last_error_offset, (LONG64)off);
    InterlockedExchange(&st->last_error, (LONG)err);
}

static int hc_read_chunk(hc_state *st, hc_reader *r, hc_task *t) {
    HANDLE h = hc_open_shard(st, r, t->shard);
    if (!h) {
        hc_note_error(st, GetLastError(), t->file_off);
        return 0;
    }
    if (st->fault_delay_us) {
        uint64_t until = hc_now_ns() + (uint64_t)st->fault_delay_us * 1000ull;
        while (hc_now_ns() < until) Sleep(0);
    }
    uint64_t done = 0;
    while (done < t->len) {
        if (st->fault_error_after_bytes && (uint64_t)st->bytes_read + done >= st->fault_error_after_bytes) {
            InterlockedIncrement64(&st->fault_injected);
            /* reported like a real device error, so the fatal line names an error code and the failing offset */
            hc_note_error(st, ERROR_READ_FAULT, t->file_off + done);
            return 0;
        }
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        uint64_t off = t->file_off + done;
        ov.Offset = (DWORD)off;
        ov.OffsetHigh = (DWORD)(off >> 32);
        DWORD want = (DWORD)((t->len - done) > (1ull << 30) ? (1ull << 30) : (t->len - done)), got = 0;
        if (!ReadFile(h, t->dst + done, want, &got, &ov)) {
            DWORD err = GetLastError();
            if (err == ERROR_HANDLE_EOF) break;
            hc_note_error(st, err, off);
            return 0;
        }
        done += got;
        if (got < want) break; /* an unbuffered read stops short only at end of file; the rest is padding */
    }
    InterlockedAdd64(&st->bytes_read, (LONG64)done);
    return 1;
}

static DWORD WINAPI hc_reader_main(LPVOID arg) {
    hc_reader *r = (hc_reader *)arg;
    hc_state *st = r->st;
    for (;;) {
        AcquireSRWLockExclusive(&st->qlock);
        while (!st->q_len && !st->stop) SleepConditionVariableSRW(&st->qcv, &st->qlock, INFINITE, 0);
        if (!st->q_len && st->stop) {
            ReleaseSRWLockExclusive(&st->qlock);
            break;
        }
        hc_task t = st->queue[st->q_head];
        st->q_head = (st->q_head + 1) % st->q_cap;
        st->q_len--;
        InterlockedIncrement(&st->inflight);
        ReleaseSRWLockExclusive(&st->qlock);

        int ok = 1;
        if (st->stop || (t.slot && t.slot->epoch != t.epoch)) {
            InterlockedIncrement64(&st->cancelled_tasks);
            ok = 0;
        } else {
            uint64_t t0 = hc_now_ns();
            ok = hc_read_chunk(st, r, &t);
            InterlockedAdd64(&st->read_ns, (LONG64)(hc_now_ns() - t0));
            InterlockedIncrement64(&st->chunks);
            if (!ok) InterlockedIncrement64(&st->io_errors);
        }
        if (!ok) InterlockedExchange(t.failed, 1);
        AcquireSRWLockExclusive(&st->qlock);
        if (InterlockedDecrement(t.pending) == 0 && t.slot) t.slot->state = t.slot->failed ? SLOT_ERROR : SLOT_READY;
        InterlockedDecrement(&st->inflight);
        WakeAllConditionVariable(&st->donecv);
        ReleaseSRWLockExclusive(&st->qlock);
    }
    for (int i = 0; i < HC_MAX_SHARDS; i++)
        if (r->files[i]) CloseHandle(r->files[i]);
    return 0;
}

/* Queue the reads that fill buf with layer L. front != 0 puts them ahead of queued prefetches. */
static void hc_enqueue_layer(hc_state *st, hc_layer *L, char *buf, hc_slot *slot, volatile LONG *pending, volatile LONG *failed,
                             int front) {
    int count = 0;
    for (int i = 0; i < L->n_runs; i++) count += (int)((L->runs[i].len + st->chunk_bytes - 1) / st->chunk_bytes);
    AcquireSRWLockExclusive(&st->qlock);
    if (st->q_len + count > st->q_cap) {
        ReleaseSRWLockExclusive(&st->qlock);
        hc_die(st, "reader queue overflow (%d + %d > %d)", st->q_len, count, st->q_cap);
    }
    InterlockedAdd(pending, count);
    hc_task *tasks = (hc_task *)_alloca(sizeof(hc_task) * (size_t)count);
    int k = 0;
    for (int i = 0; i < L->n_runs; i++) {
        hc_run *r = &L->runs[i];
        for (uint64_t off = 0; off < r->len; off += st->chunk_bytes) {
            hc_task *t = &tasks[k++];
            t->dst = buf + r->buf_off + off;
            t->shard = r->shard;
            t->file_off = r->file_off + off;
            t->len = (r->len - off) < st->chunk_bytes ? (r->len - off) : st->chunk_bytes;
            t->slot = slot;
            t->pending = pending;
            t->failed = failed;
            t->epoch = slot ? slot->epoch : 0;
        }
    }
    if (front) {
        for (int i = count - 1; i >= 0; i--) {
            st->q_head = (st->q_head - 1 + st->q_cap) % st->q_cap;
            st->queue[st->q_head] = tasks[i];
        }
    } else {
        for (int i = 0; i < count; i++) st->queue[(st->q_head + st->q_len + i) % st->q_cap] = tasks[i];
    }
    st->q_len += count;
    WakeAllConditionVariable(&st->qcv);
    ReleaseSRWLockExclusive(&st->qlock);
}

static void hc_wait_counter(hc_state *st, volatile LONG *pending) {
    AcquireSRWLockExclusive(&st->qlock);
    while (*pending > 0) SleepConditionVariableSRW(&st->donecv, &st->qlock, INFINITE, 0);
    ReleaseSRWLockExclusive(&st->qlock);
}

/* ------------------------------------------------------------------------------------------ */
/* binding                                                                                      */

static void hc_set_data(hc_tensor *t, char *base) {
    if (t->t) t->t->data = base ? base + t->slot_off : t->orig;
}

static void hc_verify(hc_state *st, hc_tensor *t) {
    if (t->verified || !t->t) return;
    size_t n = t->nbytes < 64 ? (size_t)t->nbytes : 64;
    if (memcmp(t->t->data, t->orig, n) != 0) {
        InterlockedIncrement64(&st->verify_failures);
        hc_die(st, "streamed bytes for %s differ from the file mapping (bad offset table?)", t->name);
    }
    t->verified = 1;
}

static void hc_bind_layer_buf(hc_state *st, hc_layer *L, char *buf) {
    for (int k = 0; k < L->n_tensors; k++) {
        hc_tensor *t = &st->tensors[L->first_tensor + k];
        hc_set_data(t, buf);
        if (buf) hc_verify(st, t);
    }
}

static void hc_io_failure(hc_state *st, int layer) {
    if (!st->io_error_mmap_fallback)
        hc_die(st, "unbuffered read failed for layer %d (win32 error %ld at offset %lld)", layer, (long)st->last_error,
               (long long)st->last_error_offset);
    InterlockedIncrement64(&st->mmap_fallbacks);
}

static void hc_load_resident(hc_state *st, hc_layer *L, int layer) {
    if (L->resident_buf) return;
    char *buf = (char *)VirtualAlloc(NULL, (SIZE_T)L->buf_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buf) hc_die(st, "cannot commit %llu bytes for resident layer %d", (unsigned long long)L->buf_bytes, layer);
    volatile LONG pending = 0, failed = 0;
    hc_enqueue_layer(st, L, buf, NULL, &pending, &failed, 1);
    hc_wait_counter(st, &pending);
    if (failed) {
        VirtualFree(buf, 0, MEM_RELEASE);
        hc_io_failure(st, layer);
        return;
    }
    L->resident_buf = buf;
    InterlockedAdd64(&st->resident_bytes, (LONG64)L->buf_bytes);
    hc_bind_layer_buf(st, L, buf);
}

static int hc_is_pinned(const int *pinned, int n, int layer) {
    for (int i = 0; i < n; i++)
        if (pinned[i] == layer) return 1;
    return 0;
}

static void hc_evict_slot(hc_state *st, hc_slot *s) {
    if (s->layer < 0) return;
    hc_layer *old = &st->layers[s->layer];
    hc_bind_layer_buf(st, old, NULL);
    old->slot = -1;
    old->bound = 0;
    s->layer = -1;
    InterlockedIncrement64(&st->evictions);
}

/* A slot not holding a pinned layer and not loading; prefers the least recently bound layer. */
static hc_slot *hc_pick_victim(hc_state *st, const int *pinned, int n_pinned) {
    hc_slot *best = NULL;
    uint64_t best_tick = UINT64_MAX;
    for (int i = 0; i < st->slots_n; i++) {
        hc_slot *s = &st->slots[i];
        if (s->state == SLOT_LOADING) continue;
        if (s->layer >= 0 && hc_is_pinned(pinned, n_pinned, s->layer)) continue;
        uint64_t tick = s->layer < 0 ? 0 : st->layers[s->layer].last_bound;
        if (tick < best_tick) {
            best = s;
            best_tick = tick;
        }
    }
    return best;
}

static void hc_start_load(hc_state *st, hc_slot *s, int layer, int front) {
    hc_evict_slot(st, s);
    s->layer = layer;
    s->epoch++;
    s->failed = 0;
    s->pending_chunks = 0;
    s->state = SLOT_LOADING;
    st->layers[layer].slot = (int)(s - st->slots);
    InterlockedIncrement64(&st->layer_loads);
    hc_enqueue_layer(st, &st->layers[layer], s->buf, s, &s->pending_chunks, &s->failed, front);
}

static void hc_bind_stream(hc_state *st, int layer, const int *pinned, int n_pinned) {
    hc_layer *L = &st->layers[layer];
    uint64_t t0 = hc_now_ns();
    if (L->slot >= 0) {
        hc_slot *s = &st->slots[L->slot];
        if (s->state == SLOT_LOADING) {
            InterlockedIncrement64(&st->prefetch_hits);
            hc_wait_counter(st, &s->pending_chunks);
        }
    } else {
        hc_slot *s = hc_pick_victim(st, pinned, n_pinned);
        if (!s) hc_die(st, "no free slot for layer %d (%d slots, %d pinned)", layer, st->slots_n, n_pinned);
        InterlockedIncrement64(&st->sync_loads);
        hc_start_load(st, s, layer, 1);
        hc_wait_counter(st, &s->pending_chunks);
    }
    InterlockedAdd64(&st->stall_ns, (LONG64)(hc_now_ns() - t0));
    hc_slot *s = &st->slots[L->slot];
    if (s->state == SLOT_ERROR) {
        hc_evict_slot(st, s);
        s->state = SLOT_FREE;
        hc_io_failure(st, layer);
        return;
    }
    L->last_bound = ++st->tick;
    st->newest_layer = layer;
    if (L->bound) return; /* already pointing into the ready slot: nothing to rebind */
    hc_bind_layer_buf(st, L, s->buf);
    L->bound = 1;
    InterlockedIncrement64(&st->binds);
    if (st->n_stream && layer == st->stream_order[0]) InterlockedIncrement64(&st->passes);
}

/* Start bounded lookahead reads into slots no pending node can still reference. */
static void hc_prefetch(hc_state *st, const int *pinned, int n_pinned) {
    if (st->slots_n < 2 || st->newest_layer < 0 || !st->n_stream) return;
    int pos = 0;
    while (pos < st->n_stream && st->stream_order[pos] != st->newest_layer) pos++;
    int keep[HC_MAX_NEED * 2 + 8];
    int n_keep = 0;
    for (int i = 0; i < n_pinned && n_keep < (int)(sizeof keep / sizeof keep[0]); i++) keep[n_keep++] = pinned[i];
    for (int ahead = 1; ahead < st->slots_n && ahead < st->n_stream; ahead++) {
        int layer = st->stream_order[(pos + ahead) % st->n_stream];
        if (n_keep < (int)(sizeof keep / sizeof keep[0])) keep[n_keep++] = layer;
        if (st->layers[layer].slot >= 0) continue;
        hc_slot *s = hc_pick_victim(st, keep, n_keep);
        if (!s) break;
        InterlockedIncrement64(&st->prefetch_loads);
        hc_start_load(st, s, layer, 0);
    }
}

/* ------------------------------------------------------------------------------------------ */
/* scheduler callback                                                                            */

static int hc_register(hc_state *st, struct ggml_tensor *w) {
    uint32_t h = (uint32_t)(((uintptr_t)w >> 4) * 2654435761u) & (HC_PTR_CAP - 1);
    for (;; h = (h + 1) & (HC_PTR_CAP - 1)) {
        if (st->ptrs[h].key == w) return st->ptrs[h].idx;
        if (!st->ptrs[h].key) break;
    }
    int idx = hc_find_name(st, w->name);
    /* Only weight-like names are cached; weight structs outlive every graph, so their addresses never alias an
       activation, but caching every activation pointer would grow the table without bound. */
    if (idx < 0 && strncmp(w->name, "blk.", 4) != 0 && strncmp(w->name, "output", 6) != 0 &&
        strncmp(w->name, "token_embd", 10) != 0)
        return -1;
    if (idx >= 0) {
        hc_tensor *t = &st->tensors[idx];
        hc_layer *L = &st->layers[t->layer];
        if (L->kind == KIND_NONE) {
            idx = -1;
        } else {
            if (!w->buffer || !g_buffer_is_host(w->buffer))
                hc_die(st, "%s is not in host memory; the plan placed a managed layer on an accelerator", w->name);
            const char *bname = g_buffer_name(w->buffer);
            if (!bname || !strstr(bname, "Mapped"))
                hc_die(st, "%s is in buffer %s, not a file mapping; use --load-mode mmap and --no-repack", w->name, bname ? bname : "?");
            if ((uint64_t)g_nbytes(w) != t->nbytes)
                hc_die(st, "%s has %llu bytes, the manifest says %llu", w->name, (unsigned long long)g_nbytes(w),
                       (unsigned long long)t->nbytes);
            t->t = w;
            t->orig = w->data;
            if (L->kind == KIND_RESIDENT && L->resident_buf) {
                hc_set_data(t, L->resident_buf);
                hc_verify(st, t);
            } else if (L->kind == KIND_STREAM && L->bound) {
                hc_set_data(t, st->slots[L->slot].buf);
                hc_verify(st, t);
            }
        }
    }
    int used = 0;
    for (int i = 0; i < HC_PTR_CAP; i++) used += st->ptrs[i].key != NULL;
    if (used >= HC_PTR_CAP / 2) hc_die(st, "tensor pointer table is full");
    st->ptrs[h].key = w;
    st->ptrs[h].idx = idx;
    return idx;
}

static void hc_start(hc_state *st) {
    st->started = 1;
    HMODULE base = GetModuleHandleW(L"ggml-base.dll");
    if (!base) hc_die(st, "ggml-base.dll is not loaded");
    g_buffer_is_host = (buffer_is_host_fn)(void *)GetProcAddress(base, "ggml_backend_buffer_is_host");
    g_buffer_name = (buffer_name_fn)(void *)GetProcAddress(base, "ggml_backend_buffer_name");
    g_nbytes = (nbytes_fn)(void *)GetProcAddress(base, "ggml_nbytes");
    if (!g_buffer_is_host || !g_buffer_name || !g_nbytes) hc_die(st, "ggml-base.dll lacks expected exports");
    if (st->n_stream) {
        st->slots = (hc_slot *)calloc((size_t)st->slots_n, sizeof(hc_slot));
        if (!st->slots) hc_die(st, "out of memory for the slot table");
        for (int i = 0; i < st->slots_n; i++) {
            st->slots[i].buf = (char *)VirtualAlloc(NULL, (SIZE_T)st->slot_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (!st->slots[i].buf) hc_die(st, "cannot commit slot %d (%llu bytes)", i, (unsigned long long)st->slot_bytes);
            st->slots[i].cap = st->slot_bytes;
            st->slots[i].layer = -1;
            InterlockedAdd64(&st->slot_bytes_committed, (LONG64)st->slot_bytes);
        }
    }
    uint64_t max_chunks = 0;
    for (int l = 0; l < st->n_layers; l++) {
        hc_layer *L = &st->layers[l];
        uint64_t c = 0;
        for (int i = 0; i < L->n_runs; i++) c += (L->runs[i].len + st->chunk_bytes - 1) / st->chunk_bytes;
        if (c > max_chunks) max_chunks = c;
    }
    st->q_cap = (int)(max_chunks * (uint64_t)(st->slots_n + 2) + 16);
    st->queue = (hc_task *)calloc((size_t)st->q_cap, sizeof(hc_task));
    if (!st->queue) hc_die(st, "out of memory for the reader queue");
    InitializeSRWLock(&st->qlock);
    InitializeConditionVariable(&st->qcv);
    InitializeConditionVariable(&st->donecv);
    for (int i = 0; i < st->readers_n; i++) {
        st->readers[i].st = st;
        st->readers[i].index = i;
        st->threads[i] = CreateThread(NULL, 0, hc_reader_main, &st->readers[i], 0, NULL);
        if (!st->threads[i]) hc_die(st, "cannot start reader thread");
    }
    hc_log("attached: %d streamed layers in %d slots of %llu bytes, readers %d, chunk %llu", st->n_stream, st->slots_n,
           (unsigned long long)st->slot_bytes, st->readers_n, (unsigned long long)st->chunk_bytes);
}

static int hc_pending_has(hc_state *st, int layer) {
    for (int i = 0; i < st->n_pending; i++)
        if (st->pending[i] == layer) return 1;
    return 0;
}

static bool hc_ask(hc_state *st, struct ggml_tensor *node) {
    int need[HC_MAX_NEED];
    int n_need = 0;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor *s = node->src[i];
        if (!s) continue;
        struct ggml_tensor *w = s->view_src ? s->view_src : s;
        int idx = hc_register(st, w);
        if (idx < 0) continue;
        int layer = st->tensors[idx].layer;
        hc_layer *L = &st->layers[layer];
        if (L->kind == KIND_RESIDENT) {
            if (!L->resident_buf) hc_load_resident(st, L, layer);
            continue;
        }
        int dup = 0;
        for (int k = 0; k < n_need; k++) dup |= need[k] == layer;
        if (!dup && n_need < HC_MAX_NEED) need[n_need++] = layer;
    }
    if (!n_need) goto fix_views;

    int pinned[HC_MAX_NEED * 2];
    int n_pinned = 0;
    for (int i = 0; i < st->n_pending; i++) pinned[n_pinned++] = st->pending[i];
    for (int k = 0; k < n_need; k++)
        if (!hc_is_pinned(pinned, n_pinned, need[k])) pinned[n_pinned++] = need[k];
    for (int k = 0; k < n_need; k++) {
        hc_layer *L = &st->layers[need[k]];
        if (L->slot < 0 || st->slots[L->slot].state != SLOT_READY || st->tensors[L->first_tensor].t == NULL ||
            !hc_pending_has(st, need[k]))
            hc_bind_stream(st, need[k], pinned, n_pinned);
        else
            L->last_bound = ++st->tick;
    }
    st->n_pending = 0;
    for (int i = 0; i < n_pinned; i++) st->pending[st->n_pending++] = pinned[i];
    hc_prefetch(st, pinned, n_pinned);

fix_views:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor *s = node->src[i];
        if (s && s->view_src && s->view_src->data) s->data = (char *)s->view_src->data + s->view_offs;
    }
    if (!n_need) return false;
    return st->slots_n == 1 || st->n_pending >= 2;
}

static bool hc_after(hc_state *st) {
    st->n_pending = 0;
    if (st->newest_layer >= 0) {
        int keep = st->newest_layer;
        hc_prefetch(st, &keep, 1);
    }
    return true;
}

static bool hc_cb_eval(struct ggml_tensor *t, bool ask, void *ud) {
    hc_state *st = (hc_state *)ud;
    if (st->shut_down) return st->inner_cb ? st->inner_cb(t, ask, st->inner_ud) : false;
    if (!st->started) hc_start(st);
    if (ask) {
        int inner = st->inner_cb ? st->inner_cb(t, true, st->inner_ud) : 0;
        int mine = hc_ask(st, t);
        st->last_ask_node = t;
        st->last_inner_need = inner;
        return inner || mine;
    }
    bool ok = hc_after(st);
    if (st->inner_cb && t == st->last_ask_node && st->last_inner_need) ok = st->inner_cb(t, false, st->inner_ud) && ok;
    return ok;
}

static bool hc_abort_cb(void *ud) {
    hc_state *st = (hc_state *)ud;
    if (st->fatal) return true;
    return st->inner_abort ? st->inner_abort(st->inner_abort_ud) : false;
}

/* ------------------------------------------------------------------------------------------ */
/* statistics                                                                                    */

/* "<stats>.<state>.tmp" on the stack: no heap allocation, and one temporary file per state, so the stats thread
   ("running") and the threads writing "attached", "passthrough", "fatal" or "shutdown" never share a file. */
static WCHAR *hc_stats_tmp(const hc_state *st, const char *state, WCHAR *buf) {
    size_t n = wcslen(st->stats_wpath), k = strlen(state);
    memcpy(buf, st->stats_wpath, sizeof(WCHAR) * n);
    buf[n] = L'.';
    for (size_t i = 0; i < k; i++) buf[n + 1 + i] = (WCHAR)(unsigned char)state[i];
    memcpy(buf + n + 1 + k, L".tmp", sizeof(WCHAR) * 5);
    return buf;
}

/*
 * Replace the stats file with tmp; 1 when it was published. A reader that holds the stats file open without
 * FILE_SHARE_DELETE (Python's open(), for one) makes MoveFileExW fail with ERROR_ACCESS_DENIED, so a lifecycle snapshot
 * is retried for up to `tries` x 10 ms. A "running" snapshot gets one try: the next interval writes another. On failure
 * the temporary file is removed and *err holds the last win32 error (0 when a fatal snapshot took over).
 */
static int hc_publish_stats(hc_state *st, const WCHAR *tmp, int final_fatal, int tries, DWORD *err) {
    *err = 0;
    for (int i = 0; i < tries; i++) {
        /* checked before every attempt: a stats thread that outlived hc_die's bounded join must not replace the
           fatal snapshot */
        if (st->fatal && !final_fatal) {
            *err = 0;
            break;
        }
        if (MoveFileExW(tmp, st->stats_wpath, MOVEFILE_REPLACE_EXISTING)) return 1;
        *err = GetLastError();
        if (i + 1 < tries) Sleep(10);
    }
    DeleteFileW(tmp);
    return 0;
}

/* Called from the stats thread ("running" only) and from lifecycle points. Takes no lock of this file (see hc_die).
   Only lifecycle snapshots log a failure, so the stats thread never touches stderr. */
static void hc_write_stats(hc_state *st, const char *state) {
    if (!st || !st->stats_wpath) return;
    int final_fatal = strcmp(state, "fatal") == 0;
    int lifecycle = strcmp(state, "running") != 0;
    if (st->fatal && !final_fatal) return; /* the fatal snapshot is the last word */
    WCHAR *tmp = hc_stats_tmp(st, state, (WCHAR *)_alloca(sizeof(WCHAR) * (wcslen(st->stats_wpath) + strlen(state) + 6)));
    FILE *f = _wfopen(tmp, L"wb");
    if (!f) {
        if (lifecycle) hc_log("warning: cannot create the '%s' stats snapshot (win32 error %lu)", state, (unsigned long)GetLastError());
        return;
    }
    uint64_t resident_planned = 0, stream_pass = 0;
    for (int l = 0; l < st->n_layers; l++) {
        if (st->layers[l].kind == KIND_RESIDENT) resident_planned += st->layers[l].buf_bytes;
        if (st->layers[l].kind == KIND_STREAM) stream_pass += st->layers[l].buf_bytes;
    }
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    uint64_t unix_ms = ((((uint64_t)ft.dwHighDateTime) << 32) | ft.dwLowDateTime) / 10000ull - 11644473600000ull;
    int bad = fprintf(f,
            "{\"version\":%d,\"pid\":%lu,\"state\":\"%s\",\"updated_unix_ms\":%llu,\"slots\":%d,\"slot_bytes\":%llu,"
            "\"slot_bytes_committed\":%lld,\"readers\":%d,\"chunk_bytes\":%llu,\"no_prefetch\":%s,\"streamed_layers\":%d,\"stream_bytes_per_pass\":%llu,"
            "\"resident_bytes_planned\":%llu,\"resident_bytes\":%lld,\"bytes_read\":%lld,\"read_ns\":%lld,\"chunks\":%lld,"
            "\"stall_ns\":%lld,\"layer_loads\":%lld,\"sync_loads\":%lld,\"prefetch_loads\":%lld,\"prefetch_hits\":%lld,"
            "\"binds\":%lld,\"evictions\":%lld,\"passes\":%lld,\"io_errors\":%lld,\"cancelled_tasks\":%lld,"
            "\"verify_failures\":%lld,\"mmap_fallbacks\":%lld,\"fault_injected\":%lld,\"prefetch_noop_calls\":%lld,"
            "\"prefetch_noop_bytes\":%lld,\"inflight\":%ld,\"queued\":%d}\n",
            HC_VERSION, GetCurrentProcessId(), state, (unsigned long long)unix_ms, st->slots_n, (unsigned long long)st->slot_bytes,
            (long long)st->slot_bytes_committed, st->readers_n, (unsigned long long)st->chunk_bytes, st->no_prefetch ? "true" : "false", st->n_stream,
            (unsigned long long)stream_pass, (unsigned long long)resident_planned, (long long)st->resident_bytes,
            (long long)st->bytes_read, (long long)st->read_ns, (long long)st->chunks, (long long)st->stall_ns,
            (long long)st->layer_loads, (long long)st->sync_loads, (long long)st->prefetch_loads, (long long)st->prefetch_hits,
            (long long)st->binds, (long long)st->evictions, (long long)st->passes, (long long)st->io_errors,
            (long long)st->cancelled_tasks, (long long)st->verify_failures, (long long)st->mmap_fallbacks,
            (long long)st->fault_injected, (long long)g_prefetch_noop_calls, (long long)g_prefetch_noop_bytes,
            (long)st->inflight, st->q_len) < 0;
    bad |= fclose(f) != 0;
    if (bad) {
        DeleteFileW(tmp);
        if (lifecycle) hc_log("warning: cannot write the '%s' stats snapshot", state);
        return;
    }
    DWORD err = 0;
    if (!hc_publish_stats(st, tmp, final_fatal, lifecycle ? HC_STATS_RENAME_TRIES : 1, &err) && lifecycle && err)
        hc_log("warning: the '%s' stats snapshot was not published (win32 error %lu)", state, (unsigned long)err);
}

static DWORD WINAPI hc_stats_main(LPVOID arg) {
    hc_state *st = (hc_state *)arg;
    while (WaitForSingleObject(st->stats_stop, st->stats_interval_ms) == WAIT_TIMEOUT) hc_write_stats(st, "running");
    return 0;
}

/* Signal the stats thread and wait up to wait_ms for it to exit; 1 when it is gone or never ran. On the stats
   thread itself this only signals (a thread cannot join itself), and the handles stay open until process exit. */
static int hc_stop_stats(hc_state *st, DWORD wait_ms) {
    if (!st->stats_thread) return 1;
    SetEvent(st->stats_stop);
    if (GetCurrentThreadId() == st->stats_thread_id) return 0;
    if (WaitForSingleObject(st->stats_thread, wait_ms) != WAIT_OBJECT_0) return 0;
    CloseHandle(st->stats_thread);
    CloseHandle(st->stats_stop);
    st->stats_thread = NULL;
    st->stats_stop = NULL;
    st->stats_thread_id = 0;
    return 1;
}

/* ------------------------------------------------------------------------------------------ */
/* hooks                                                                                         */

static void hc_reset_runtime(hc_state *st);

static void hc_shutdown(hc_state *st) {
    if (st->shut_down) return;
    st->shut_down = 1;
    if (st->started) {
        AcquireSRWLockExclusive(&st->qlock);
        st->stop = 1;
        for (int i = 0; i < st->slots_n && st->slots; i++) st->slots[i].epoch++; /* stale queued reads are skipped */
        WakeAllConditionVariable(&st->qcv);
        ReleaseSRWLockExclusive(&st->qlock);
        for (int i = 0; i < st->readers_n; i++)
            if (st->threads[i]) {
                WaitForSingleObject(st->threads[i], INFINITE);
                CloseHandle(st->threads[i]);
            }
        for (int l = 0; l < st->n_layers; l++) hc_bind_layer_buf(st, &st->layers[l], NULL);
        for (int i = 0; i < st->slots_n && st->slots; i++) {
            if (st->slots[i].buf && VirtualFree(st->slots[i].buf, 0, MEM_RELEASE))
                InterlockedAdd64(&st->slot_bytes_committed, -(LONG64)st->slots[i].cap);
            st->slots[i].buf = NULL;
            st->slots[i].layer = -1;
        }
        for (int l = 0; l < st->n_layers; l++)
            if (st->layers[l].resident_buf) {
                VirtualFree(st->layers[l].resident_buf, 0, MEM_RELEASE);
                st->layers[l].resident_buf = NULL;
                st->layers[l].slot = -1;
            }
        st->resident_bytes = 0;
    }
    hc_stop_stats(st, INFINITE);
    hc_write_stats(st, "shutdown");
    hc_log("detached: buffers released, tensors point at the file mapping again");
    hc_reset_runtime(st);
}

/* Forget everything tied to the freed context (tensor pointers may belong to a freed model) so the next context,
   for example the real one after a --fit probe context, attaches from a clean state. Counters stay cumulative. */
static void hc_reset_runtime(hc_state *st) {
    free(st->slots);
    st->slots = NULL;
    free(st->queue);
    st->queue = NULL;
    st->q_cap = st->q_head = st->q_len = 0;
    st->inflight = 0;
    st->stop = 0;
    memset(st->threads, 0, sizeof st->threads);
    memset(st->readers, 0, sizeof st->readers);
    memset(st->ptrs, 0, sizeof st->ptrs);
    for (int i = 0; i < st->n_tensors; i++) {
        st->tensors[i].t = NULL;
        st->tensors[i].orig = NULL;
        st->tensors[i].verified = 0;
    }
    for (int l = 0; l < st->n_layers; l++) {
        st->layers[l].slot = -1;
        st->layers[l].bound = 0;
        st->layers[l].last_bound = 0;
    }
    st->ctx = NULL;
    st->inner_cb = NULL;
    st->inner_abort = NULL;
    st->last_ask_node = NULL;
    st->n_pending = 0;
    st->newest_layer = -1;
    st->tick = 0;
    st->started = 0;
    st->shut_down = 0;
}

static int hc_manages_layers(const hc_state *st) {
    for (int l = 0; l < st->n_layers; l++)
        if (st->layers[l].kind != KIND_NONE) return 1;
    return 0;
}

static void hc_start_stats(hc_state *st) {
    st->stats_stop = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (st->stats_stop) st->stats_thread = CreateThread(NULL, 0, hc_stats_main, st, 0, &st->stats_thread_id);
    if (st->stats_thread) return;
    DWORD err = GetLastError();
    if (st->stats_stop) CloseHandle(st->stats_stop);
    st->stats_stop = NULL;
    hc_log("warning: cannot start the stats thread (win32 error %lu); only lifecycle snapshots are written",
           (unsigned long)err);
}

static struct llama_context *hc_init_from_model(struct llama_model *model, struct llama_context_params params) {
    /* a config without managed layers only switches off the mapping prefetch */
    hc_state *st = (g_cfg && !g_active && hc_manages_layers(g_cfg)) ? g_cfg : NULL;
    if (st) {
        st->inner_cb = params.cb_eval;
        st->inner_ud = params.cb_eval_user_data;
        st->inner_abort = params.abort_callback;
        st->inner_abort_ud = params.abort_callback_data;
        params.cb_eval = hc_cb_eval;
        params.cb_eval_user_data = st;
        params.abort_callback = hc_abort_cb;
        params.abort_callback_data = st;
        if (params.op_offload) hc_log("warning: op offload is on; prompt batches may copy streamed weights through the mapping");
    } else if (g_cfg && g_active) {
        hc_log("warning: a second context was created; it runs without streaming");
    }
    struct llama_context *ctx = g_real_init(model, params);
    if (st && ctx) {
        st->ctx = ctx;
        g_active = st;
        if (st->stats_wpath) hc_start_stats(st);
        hc_write_stats(st, "attached");
    } else if (ctx && g_cfg && !hc_manages_layers(g_cfg)) {
        /* the model is mapped by now, so this snapshot carries the mapping prefetch no-op counters */
        hc_write_stats(g_cfg, "passthrough");
    }
    return ctx;
}

static void hc_free(struct llama_context *ctx) {
    if (g_active && g_active->ctx == ctx) {
        hc_shutdown(g_active);
        g_active = NULL;
    }
    g_real_free(ctx);
}

static BOOL WINAPI hc_prefetch_noop(HANDLE process, ULONG_PTR n, PWIN32_MEMORY_RANGE_ENTRY entries, ULONG flags) {
    (void)process; (void)flags;
    InterlockedIncrement64(&g_prefetch_noop_calls);
    for (ULONG_PTR i = 0; i < n; i++) InterlockedAdd64(&g_prefetch_noop_bytes, (LONG64)entries[i].NumberOfBytes);
    return TRUE;
}

static FARPROC WINAPI hc_get_proc(HMODULE m, LPCSTR name) {
    if (g_no_prefetch && (ULONG_PTR)name > 0xFFFF && strcmp(name, "PrefetchVirtualMemory") == 0) return (FARPROC)(void *)hc_prefetch_noop;
    return g_real_get_proc(m, name);
}

static int hc_patch_iat(HMODULE mod, const char *dll, const char *func, void *repl, void **orig) {
    BYTE *base = (BYTE *)mod;
    IMAGE_NT_HEADERS64 *nt = (IMAGE_NT_HEADERS64 *)(base + ((IMAGE_DOS_HEADER *)base)->e_lfanew);
    IMAGE_DATA_DIRECTORY dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress) return 0;
    int patched = 0;
    for (IMAGE_IMPORT_DESCRIPTOR *imp = (IMAGE_IMPORT_DESCRIPTOR *)(base + dir.VirtualAddress); imp->Name; imp++) {
        if (_stricmp((const char *)(base + imp->Name), dll) != 0) continue;
        IMAGE_THUNK_DATA64 *names = (IMAGE_THUNK_DATA64 *)(base + (imp->OriginalFirstThunk ? imp->OriginalFirstThunk : imp->FirstThunk));
        IMAGE_THUNK_DATA64 *iat = (IMAGE_THUNK_DATA64 *)(base + imp->FirstThunk);
        for (; names->u1.AddressOfData; names++, iat++) {
            if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal)) continue;
            IMAGE_IMPORT_BY_NAME *ibn = (IMAGE_IMPORT_BY_NAME *)(base + names->u1.AddressOfData);
            if (strcmp((const char *)ibn->Name, func) != 0) continue;
            DWORD old;
            if (!VirtualProtect(&iat->u1.Function, sizeof(ULONGLONG), PAGE_READWRITE, &old)) return -1;
            if (orig && !*orig) *orig = (void *)(uintptr_t)iat->u1.Function;
            iat->u1.Function = (ULONGLONG)(uintptr_t)repl;
            VirtualProtect(&iat->u1.Function, sizeof(ULONGLONG), old, &old);
            patched++;
        }
    }
    return patched;
}

static BOOL CALLBACK hc_install(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once; (void)param; (void)ctx;
    /* the wide variant, so a config under a non-ASCII home directory is found */
    DWORD cap = GetEnvironmentVariableW(L"HC_STREAM_CONFIG", NULL, 0);
    if (cap == 0) return TRUE; /* capability off: pure pass-through */
    WCHAR *path = (WCHAR *)malloc(sizeof(WCHAR) * cap);
    if (!path) hc_die(NULL, "out of memory reading HC_STREAM_CONFIG");
    DWORD n = GetEnvironmentVariableW(L"HC_STREAM_CONFIG", path, cap);
    if (n == 0) { /* set but empty: off, as before */
        free(path);
        return TRUE;
    }
    if (n >= cap) hc_die(NULL, "HC_STREAM_CONFIG changed while it was read");
    g_cfg = hc_parse_config(path);
    free(path);
    g_no_prefetch = g_cfg->no_prefetch;
    HMODULE llama = GetModuleHandleW(L"llama.dll");
    if (!llama) hc_die(g_cfg, "llama.dll is not loaded");
    g_real_init = (init_from_model_fn)(void *)GetProcAddress(llama, "llama_init_from_model");
    g_real_free = (llama_free_fn)(void *)GetProcAddress(llama, "llama_free");
    if (!g_real_init || !g_real_free) hc_die(g_cfg, "llama.dll lacks llama_init_from_model or llama_free");
    const WCHAR *users[] = {L"llama-common.dll", L"llama-server-impl.dll", L"llama-cli-impl.dll", L"llama-completion-impl.dll",
                            L"llama-bench-impl.dll"};
    int hooked_init = 0;
    for (size_t i = 0; i < sizeof users / sizeof users[0]; i++) {
        HMODULE m = GetModuleHandleW(users[i]);
        if (!m) continue;
        hooked_init += hc_patch_iat(m, "llama.dll", "llama_init_from_model", (void *)hc_init_from_model, NULL) > 0;
        hc_patch_iat(m, "llama.dll", "llama_free", (void *)hc_free, NULL);
    }
    if (!hooked_init) hc_die(g_cfg, "no loaded module imports llama_init_from_model; unsupported llama.cpp layout");
    if (g_no_prefetch) {
        void *orig = NULL;
        if (hc_patch_iat(llama, "kernel32.dll", "GetProcAddress", (void *)hc_get_proc, &orig) <= 0 || !orig)
            hc_die(g_cfg, "cannot hook GetProcAddress in llama.dll");
        g_real_get_proc = (get_proc_fn)orig;
    }
    hc_log("enabled (%d tensors, %d streamed layers, %d slots, prefetch %s)", g_cfg->n_tensors, g_cfg->n_stream, g_cfg->slots_n,
           g_no_prefetch ? "off" : "on");
    /* No context will attach, so no thread writes stats. One snapshot now shows the config was taken; another is
       written when a context is created, after the model mapping (and its prefetch call) exists. */
    if (!hc_manages_layers(g_cfg)) hc_write_stats(g_cfg, "passthrough");
    return TRUE;
}

static void hc_hook(void) { InitOnceExecuteOnce(&g_hook_once, hc_install, NULL, NULL); }

/* ------------------------------------------------------------------------------------------ */
/* ggml.dll exports                                                                             */

#define HC_EXPORT __declspec(dllexport)
#define HC_FWD(ret, name, params, args)                       \
    HC_EXPORT ret name params {                               \
        typedef ret(*fn_t) params;                            \
        hc_hook();                                            \
        return ((fn_t)(void *)hc_orig(#name)) args;           \
    }
#define HC_FWD_VOID(name, params, args)                       \
    HC_EXPORT void name params {                              \
        typedef void(*fn_t) params;                           \
        hc_hook();                                            \
        ((fn_t)(void *)hc_orig(#name)) args;                  \
    }

HC_FWD(ggml_backend_dev_t, ggml_backend_dev_by_name, (const char *name), (name))
HC_FWD(ggml_backend_dev_t, ggml_backend_dev_by_type, (enum ggml_backend_dev_type type), (type))
HC_FWD(size_t, ggml_backend_dev_count, (void), ())
HC_FWD(ggml_backend_dev_t, ggml_backend_dev_get, (size_t index), (index))
HC_FWD_VOID(ggml_backend_device_register, (ggml_backend_dev_t device), (device))
HC_FWD(ggml_backend_t, ggml_backend_init_best, (void), ())
HC_FWD(ggml_backend_t, ggml_backend_init_by_name, (const char *name, const char *params), (name, params))
HC_FWD(ggml_backend_t, ggml_backend_init_by_type, (enum ggml_backend_dev_type type, const char *params), (type, params))
HC_FWD(ggml_backend_reg_t, ggml_backend_load, (const char *path), (path))
HC_FWD_VOID(ggml_backend_load_all, (void), ())
HC_FWD_VOID(ggml_backend_load_all_from_path, (const char *dir_path), (dir_path))
HC_FWD(ggml_backend_reg_t, ggml_backend_reg_by_name, (const char *name), (name))
HC_FWD(size_t, ggml_backend_reg_count, (void), ())
HC_FWD(ggml_backend_reg_t, ggml_backend_reg_get, (size_t index), (index))
HC_FWD_VOID(ggml_backend_register, (ggml_backend_reg_t reg), (reg))
HC_FWD_VOID(ggml_backend_unload, (ggml_backend_reg_t reg), (reg))

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        g_self = inst;
        DisableThreadLibraryCalls(inst);
    }
    return TRUE;
}
