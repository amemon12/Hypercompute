// Runtime test of native/hcpeer/src/hcpeer.cpp on a POSIX host (Linux, or macOS).
//
//   c++ -std=c++17 -O1 -Wall -Wextra -Werror -Wno-unused-function -DHCPEER_POSIX_TEST
//       -Inative/hcpeer/src -Inative/hcpeer/third_party/llama.cpp
//       native/hcpeer/src/hcpeer.cpp native/hcpeer/test/hcpeer_posix_test.cpp -o hcpeer_posix_test
//   ./hcpeer_posix_test <empty scratch dir> on|off
//
// Stands in for ggml-base with a page-aligned anonymous "CPU buffer" (what vm_allocate gives ggml on macOS) and
// drives every hook against real files, mmap, rename and fork. One line per check; exit 1 on any failure.

#include "hcpeer.h"

#include "ggml-backend-impl.h"

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(__APPLE__)
#    include <mach/mach.h>
// libSystem (libproc); declared here because zig's bundled macOS headers may not carry libproc.h
extern "C" int proc_regionfilename(int pid, uint64_t address, void * buffer, uint32_t buffersize);
#endif

// ----------------------------------------------------------------------------- ggml-base stand-ins

static ggml_backend_buffer_type g_cpu_buft   = {};
static ggml_backend_buffer_type g_metal_buft = {};
static std::string              g_log;

extern "C" {

size_t ggml_nbytes(const struct ggml_tensor * t) {  // ggml.c formula for a block size of 1, type size 1 (I8)
    size_t n = 1;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        n += (size_t) (t->ne[i] - 1) * t->nb[i];
    }
    return n;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t b) { return b->context; }
size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t b) { return b->size; }
ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t b) { return b->buft; }
ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) { return &g_cpu_buft; }
const char * ggml_get_name(const struct ggml_tensor * t) { return t->name; }

void ggml_log_internal(enum ggml_log_level, const char * format, ...) {
    char line[2048];
    va_list args;
    va_start(args, format);
    vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    g_log += line;
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "ggml_abort %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
}

}  // extern "C"

// ----------------------------------------------------------------------------- helpers

static int g_failures = 0;

static void check(bool ok, const char * what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    g_failures += ok ? 0 : 1;
}

static size_t page() { return (size_t) sysconf(_SC_PAGESIZE); }

static std::string hex_name(uint64_t hash) {
    char name[17];
    snprintf(name, sizeof(name), "%016" PRIx64, hash);
    return name;
}

static std::vector<char> random_bytes(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::vector<char> out(n);
    for (char & c : out) {
        c = (char) (gen() & 0xFF);
    }
    return out;
}

static std::vector<char> read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static bool exists(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static size_t count(const std::string & text, const char * needle) {
    size_t n = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) {
        n++;
    }
    return n;
}

static bool no_tmp_files(const std::string & dir) {
    DIR * d = opendir(dir.c_str());
    bool clean = d != nullptr;
    for (struct dirent * e = d != nullptr ? readdir(d) : nullptr; e != nullptr; e = readdir(d)) {
        clean = clean && std::strncmp(e->d_name, ".hcpeer-", 8) != 0;
    }
    if (d != nullptr) {
        closedir(d);
    }
    return clean;
}

// The file backing the mapping that holds addr, as the kernel reports it ("" when anonymous or unknown).
static std::string backing_file(const void * addr) {
#if defined(__APPLE__)
    char path[4 * PATH_MAX];
    const int n = proc_regionfilename(getpid(), (uint64_t) (uintptr_t) addr, path, sizeof(path));
    return n > 0 ? std::string(path, (size_t) n) : std::string();
#else
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        uintptr_t lo = 0, hi = 0;
        if (sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR, &lo, &hi) == 2 && lo <= (uintptr_t) addr &&
            (uintptr_t) addr < hi) {
            const size_t slash = line.find('/');
            return slash == std::string::npos ? std::string() : line.substr(slash);
        }
    }
    return std::string();
#endif
}

struct fixture {
    std::string          dir;
    char *               base = nullptr;
    size_t               size = 0;
    ggml_backend_buffer  cpu  = {};
    ggml_backend_buffer  gpu  = {};
};

static ggml_tensor make_tensor(ggml_backend_buffer * buffer, char * data, size_t nbytes, const char * name) {
    ggml_tensor t = {};
    t.type   = GGML_TYPE_I8;
    t.buffer = buffer;
    t.data   = data;
    t.ne[0]  = (int64_t) nbytes;
    t.ne[1] = t.ne[2] = t.ne[3] = 1;
    t.nb[0]  = 1;
    t.nb[1] = t.nb[2] = t.nb[3] = nbytes;
    snprintf(t.name, sizeof(t.name), "%s", name);
    return t;
}

// Page-aligned zeroed memory from the allocator ggml's CPU buffers use: vm_allocate on macOS, anonymous mmap here.
static char * alloc_buffer(size_t size) {
#if defined(__APPLE__)
    vm_address_t addr = 0;
    return vm_allocate(mach_task_self(), &addr, size, VM_FLAGS_ANYWHERE) == KERN_SUCCESS ? (char *) addr : nullptr;
#else
    void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    return p == MAP_FAILED ? nullptr : (char *) p;
#endif
}

static fixture make_fixture(const char * dir_arg) {
    fixture f;
    char real[PATH_MAX];
    f.dir  = realpath(dir_arg, real) != nullptr ? std::string(real) : std::string(dir_arg);
    f.size = 96 * page();
    f.base = alloc_buffer(f.size);
    f.cpu  = {{}, &g_cpu_buft, f.base, f.size, GGML_BACKEND_BUFFER_USAGE_WEIGHTS};
    f.gpu  = {{}, &g_metal_buft, f.base, f.size, GGML_BACKEND_BUFFER_USAGE_WEIGHTS};
    return f;
}

// ----------------------------------------------------------------------------- checks with HC_PEER_MMAP=1

// A pid that no longer runs.
static pid_t dead_pid() {
    const pid_t child = fork();
    if (child == 0) {
        _exit(0);
    }
    waitpid(child, nullptr, 0);
    return child;
}

static void write_file(const std::string & path, const char * data, size_t n) {
    std::ofstream(path, std::ios::binary).write(data, (std::streamsize) n);
}

static void check_banner_and_stale_files(fixture & f) {
    const pid_t child = dead_pid();
    const std::string hash = hex_name(0xabcdef0123456789ull);
    const std::string dead = f.dir + "/.hcpeer-" + hash + "." + std::to_string(child) + ".tmp";
    const std::string mine = f.dir + "/.hcpeer-" + hash + "." + std::to_string(getpid()) + ".tmp";
    const std::string live = f.dir + "/.hcpeer-" + hash + ".1.tmp";
    const std::string seed = f.dir + "/.seed-" + hash + ".part";
    for (const std::string & p : {dead, mine, live, seed}) {
        std::ofstream(p) << "x";
    }
    hcpeer_log_banner(f.dir.c_str());
    check(!exists(dead) && !exists(mine), "banner removes temporary files of stopped servers");
    check(exists(live) && exists(seed), "banner keeps a running server's temporary file and other files");
    unlink(live.c_str());
    unlink(seed.c_str());
    check(g_log.find("hcpeer mmap    : on (v4, page " + std::to_string(page()) + " B") != std::string::npos,
          "banner says on with the page size");
    g_log.clear();
    hcpeer_log_banner(nullptr);
    check(g_log.find("HC_PEER_MMAP=1 needs -c") != std::string::npos, "banner without -c says off");
    for (const char * bad : {"abc", "-5", "", "12x"}) {
        setenv("HC_PEER_CACHE_MIN_FREE_GIB", bad, 1);
        g_log.clear();
        hcpeer_log_banner(f.dir.c_str());
        check(g_log.find("keep 20 GiB free") != std::string::npos, "a malformed floor falls back to 20 GiB");
    }
    setenv("HC_PEER_CACHE_MIN_FREE_GIB", "0", 1);
}

static void check_alignment() {
    check(hcpeer_alignment(&g_cpu_buft, 32, true) == page(), "CPU alignment is the page size with -c");
    check(hcpeer_alignment(&g_cpu_buft, 32, false) == 32, "CPU alignment unchanged without -c");
    check(hcpeer_alignment(&g_metal_buft, 32, true) == 32, "other buffer types keep their alignment");
}

static void check_save_hit_and_cow(fixture & f) {
    const size_t n = 3 * page() + 1000;
    const std::vector<char> a = random_bytes(n, 1);
    const std::string name = hex_name(0x1111222233334444ull), path = f.dir + "/" + name;
    ggml_tensor t0 = make_tensor(&f.cpu, f.base, n, "t0");
    check(hcpeer_save_cached(&t0, 0, a.data(), n, f.dir.c_str(), name.c_str()) == HCPEER_DONE, "save maps a CPU tensor");
    check(memcmp(f.base, a.data(), n) == 0, "saved tensor reads back the payload");
    check(read_file(path) == a, "cache file holds exactly the payload");
    check(no_tmp_files(f.dir), "no temporary file left after a save");
    check(backing_file(f.base) == path, "saved tensor pages are backed by the cache file");

    ggml_tensor t1 = make_tensor(&f.cpu, f.base + 8 * page(), n, "t1");
    check(hcpeer_map_cached(&t1, 0, f.dir.c_str(), 0x1111222233334444ull) == HCPEER_DONE, "aligned hit maps");
    check(memcmp(t1.data, a.data(), n) == 0, "aligned hit reads back the payload");
    check(backing_file(t1.data) == path, "hit pages are backed by the cache file");

    ggml_tensor t2 = make_tensor(&f.cpu, f.base + 16 * page() + 64, n, "t2");
    check(hcpeer_map_cached(&t2, 0, f.dir.c_str(), 0x1111222233334444ull) == HCPEER_STOCK,
          "unaligned hit falls back");
    check(std::vector<char>(f.base + 16 * page() + 64, f.base + 16 * page() + 64 + n) == std::vector<char>(n, 0),
          "fallback leaves the tensor untouched for the stock copy");
    check(g_log.find("not page aligned") != std::string::npos, "fallback is logged");

    ((char *) t1.data)[0] ^= 0x5A;
    check(read_file(path) == a, "a write to a mapped page never reaches the file (MAP_PRIVATE)");
    check(memcmp(f.base, a.data(), n) == 0, "a write to one tensor does not leak into another mapping");
}

static void check_resave_keeps_old_mapping(fixture & f) {
    const size_t n = 3 * page() + 1000;
    const std::vector<char> a = random_bytes(n, 1), b = random_bytes(n, 2);
    const std::string name = hex_name(0x1111222233334444ull), path = f.dir + "/" + name;
    ggml_tensor t3 = make_tensor(&f.cpu, f.base + 24 * page(), n, "t3");
    check(hcpeer_save_cached(&t3, 0, b.data(), n, f.dir.c_str(), name.c_str()) == HCPEER_DONE,
          "saving the same name again maps the new file");
    check(memcmp(f.base, a.data(), n) == 0, "a tensor mapped from the replaced file keeps its bytes (rename)");
    check(memcmp(t3.data, b.data(), n) == 0 && read_file(path) == b, "the new tensor and file hold the new bytes");
}

// Why saves are atomic: truncating a mapped cache file in place kills the server on its next read.
static void check_in_place_truncation_is_fatal(fixture & f) {
    const pid_t child = fork();
    if (child == 0) {
        const size_t n = 3 * page() + 1000;
        const std::vector<char> c = random_bytes(n, 3);
        const std::string name = hex_name(0x5555666677778888ull), path = f.dir + "/" + name;
        ggml_tensor t = make_tensor(&f.cpu, f.base + 32 * page(), n, "t4");
        if (hcpeer_save_cached(&t, 0, c.data(), n, f.dir.c_str(), name.c_str()) != HCPEER_DONE) {
            _exit(3);
        }
        if (truncate(path.c_str(), 0) != 0) {
            _exit(4);
        }
        volatile char sink = ((volatile char *) t.data)[page()];
        (void) sink;
        _exit(0);
    }
    int status = 0;
    waitpid(child, &status, 0);
    const bool killed = WIFSIGNALED(status) && (WTERMSIG(status) == SIGBUS || WTERMSIG(status) == SIGSEGV);
    check(killed, "in-place truncation of a mapped file kills the reader (SIGBUS)");
}

static void check_fallbacks(fixture & f) {
    const size_t n = 3 * page() + 1000;
    const std::vector<char> a = random_bytes(n, 1);
    const std::string path = f.dir + "/" + hex_name(0x1111222233334444ull);
    ggml_tensor outside = make_tensor(&f.cpu, f.base + f.size + page(), n, "outside");
    check(hcpeer_map_cached(&outside, 0, f.dir.c_str(), 0x1111222233334444ull) == HCPEER_STOCK,
          "a tensor outside its buffer falls back");
    ggml_tensor offset = make_tensor(&f.cpu, f.base + 44 * page(), n, "offset");
    check(hcpeer_map_cached(&offset, n, f.dir.c_str(), 0x1111222233334444ull) == HCPEER_STOCK && exists(path),
          "an offset past the tensor is left to the stock checks and keeps the file");
    const size_t logged = g_log.size();
    check(hcpeer_map_cached(&offset, 0, f.dir.c_str(), 0x9999999999999999ull) == HCPEER_STOCK &&
          g_log.size() == logged, "a missing cache file is a silent miss for the stock path");

    ggml_tensor tiny = make_tensor(&f.cpu, f.base + 48 * page(), 1000, "tiny");
    const std::string tiny_name = hex_name(0x0000000000000001ull);
    check(hcpeer_save_cached(&tiny, 0, a.data(), 1000, f.dir.c_str(), tiny_name.c_str()) == HCPEER_COPY &&
          read_file(f.dir + "/" + tiny_name).size() == 1000, "a tensor under one page is saved and copied");

    ggml_tensor gpu = make_tensor(&f.gpu, f.base + 52 * page(), n, "metal");
    const std::string gpu_name = hex_name(0x0000000000000002ull);
    check(hcpeer_save_cached(&gpu, 0, a.data(), n, f.dir.c_str(), gpu_name.c_str()) == HCPEER_COPY &&
          read_file(f.dir + "/" + gpu_name) == a, "another buffer type gets an atomic save and the stock copy");
    check(hcpeer_map_cached(&gpu, 0, f.dir.c_str(), 0x0000000000000002ull) == HCPEER_STOCK,
          "another buffer type is never mapped: a good file goes to the stock copy");
    check(backing_file(gpu.data).empty(), "another buffer type's memory stays anonymous");
}

static void check_floor_and_bad_dir(fixture & f) {
    const size_t n = 3 * page() + 1000;
    const std::vector<char> a = random_bytes(n, 1);
    const std::string name = hex_name(0x0000000000000003ull);
    ggml_tensor t = make_tensor(&f.cpu, f.base + 56 * page(), n, "floor");
    setenv("HC_PEER_CACHE_MIN_FREE_GIB", "1048576", 1);
    check(hcpeer_save_cached(&t, 0, a.data(), n, f.dir.c_str(), name.c_str()) == HCPEER_COPY &&
          !exists(f.dir + "/" + name) && no_tmp_files(f.dir), "below the free-space floor nothing is written");
    check(g_log.find("HC_PEER_CACHE_MIN_FREE_GIB") != std::string::npos, "the skipped save is logged");
    setenv("HC_PEER_CACHE_MIN_FREE_GIB", "0", 1);
    check(hcpeer_save_cached(&t, 0, a.data(), n, "/nonexistent/hcpeer", name.c_str()) == HCPEER_COPY,
          "an unusable cache directory means copy, not crash");

    const std::string victim = f.dir + "/victim";
    std::ofstream(victim) << "keep";
    const std::string tmp = f.dir + "/.hcpeer-" + name + "." + std::to_string(getpid()) + ".tmp";
    check(symlink(victim.c_str(), tmp.c_str()) == 0, "plant a symlink at the temporary file name");
    check(hcpeer_save_cached(&t, 0, a.data(), n, f.dir.c_str(), name.c_str()) == HCPEER_COPY &&
          read_file(victim) == std::vector<char>{'k', 'e', 'e', 'p'} && !exists(f.dir + "/" + name),
          "a symlink at the temporary name is not followed");
    unlink(tmp.c_str());
    check(link(victim.c_str(), tmp.c_str()) == 0, "plant a hard link at the temporary file name");
    check(hcpeer_save_cached(&t, 0, a.data(), n, f.dir.c_str(), name.c_str()) == HCPEER_COPY &&
          read_file(victim) == std::vector<char>{'k', 'e', 'e', 'p'} && !exists(f.dir + "/" + name),
          "an existing file at the temporary name is never truncated or reused (O_EXCL)");
    unlink(tmp.c_str());
    unlink(victim.c_str());
}

// The review finding: a short file under a hash name (an interrupted stock save, or one from before hcpeer v3) must
// never be served. It is removed and answered as a miss, so the client resends and the atomic save replaces it.
static void check_short_files_are_misses(fixture & f) {
    const size_t n = 3 * page() + 1000;
    const std::vector<char> a = random_bytes(n, 5);
    const std::string name = hex_name(0x0000000000000005ull), path = f.dir + "/" + name;
    ggml_tensor cpu = make_tensor(&f.cpu, f.base + 36 * page(), n, "short_cpu");
    write_file(path, a.data(), n - 5000);
    check(hcpeer_map_cached(&cpu, 0, f.dir.c_str(), 0x5ull) == HCPEER_MISS && !exists(path),
          "a short cache file for a CPU tensor is removed and answered as a miss");
    check(std::vector<char>(f.base + 36 * page(), f.base + 36 * page() + n) == std::vector<char>(n, 0) &&
          backing_file(cpu.data).empty(), "a rejected file leaves the tensor untouched");
    check(g_log.find("bytes cannot be the data asked for") != std::string::npos &&
          g_log.find("answering a cache miss") != std::string::npos, "the rejection is logged");
    ggml_tensor gpu = make_tensor(&f.gpu, f.base + 36 * page(), n, "short_metal");
    write_file(path, a.data(), n - 5000);
    check(hcpeer_map_cached(&gpu, 0, f.dir.c_str(), 0x5ull) == HCPEER_MISS && !exists(path),
          "a short cache file for another buffer type (stock copy path) is removed and answered as a miss");
    write_file(path, a.data(), n);
    ggml_tensor small = make_tensor(&f.cpu, f.base + 40 * page(), 2 * page(), "small");
    check(hcpeer_map_cached(&small, 0, f.dir.c_str(), 0x5ull) == HCPEER_MISS && !exists(path),
          "a cache file larger than the tensor is removed and answered as a miss");
    check(hcpeer_save_cached(&cpu, 0, a.data(), n, f.dir.c_str(), name.c_str()) == HCPEER_DONE &&
          read_file(path) == a && hcpeer_map_cached(&cpu, 0, f.dir.c_str(), 0x5ull) == HCPEER_DONE &&
          memcmp(cpu.data, a.data(), n) == 0, "the resend saves a complete file that the next hit maps");
    const std::string dir_name = f.dir + "/" + hex_name(0x6ull);
    check(mkdir(dir_name.c_str(), 0755) == 0 && hcpeer_map_cached(&cpu, 0, f.dir.c_str(), 0x6ull) == HCPEER_MISS &&
          exists(dir_name), "a directory under a hash name is a miss and is left alone");
}

// A 3-D tensor like a MoE cache bank: `slots` slots of `slot` bytes, filled a whole slot at a time.
static ggml_tensor make_bank(ggml_backend_buffer * buffer, char * data, size_t slot, int64_t slots) {
    ggml_tensor t = make_tensor(buffer, data, slot, "bank");
    t.ne[2] = slots;
    t.nb[2] = slot;
    t.nb[3] = slot * (size_t) slots;
    return t;
}

static void check_bank_fills(fixture & f) {
    const size_t slot = 2 * page();
    const std::vector<char> two = random_bytes(2 * slot, 6);
    const std::string name = hex_name(0x0000000000000007ull), path = f.dir + "/" + name;
    ggml_tensor bank = make_bank(&f.cpu, f.base + 64 * page(), slot, 4);
    write_file(path, two.data(), two.size());
    check(hcpeer_map_cached(&bank, slot, f.dir.c_str(), 0x7ull) == HCPEER_DONE &&
          memcmp(f.base + 64 * page() + slot, two.data(), two.size()) == 0,
          "a whole-slot fill at a slot boundary of a 3-D bank is served from its file");
    write_file(path, two.data(), two.size() - 100);
    check(hcpeer_map_cached(&bank, slot, f.dir.c_str(), 0x7ull) == HCPEER_MISS && !exists(path),
          "a short bank fill file is removed and answered as a miss");
    write_file(path, two.data(), slot + slot / 2);
    check(hcpeer_map_cached(&bank, slot, f.dir.c_str(), 0x7ull) == HCPEER_MISS && !exists(path),
          "a bank fill file of part of a slot is removed and answered as a miss");
    write_file(path, two.data(), slot);
    ggml_tensor flat = make_tensor(&f.cpu, f.base + 64 * page(), 4 * slot, "flat");
    check(hcpeer_map_cached(&flat, 0, f.dir.c_str(), 0x7ull) == HCPEER_MISS && !exists(path),
          "a 1-D tensor only accepts a file holding all of it from offset");
}

// A sparse file of `size` bytes: it counts toward the cache directory's size without using the test volume.
static bool sparse_file(const std::string & path, off_t size) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return false;
    }
    const bool sized = ftruncate(fd, size) == 0;
    return close(fd) == 0 && sized;
}

// The review finding: a mapped file keeps its disk until it is unmapped, so the agent's eviction cannot bring the
// cache back under --max-disk once hcpeer mapped past it. A save that would take the cache directory past
// HC_PEER_CACHE_MAX_GIB is skipped and copied, so nothing past the cap is ever saved or mapped.
static void check_cache_cap(fixture & f) {
    const size_t n = 3 * page() + 1000;
    const off_t gib = (off_t) 1 << 30;
    const std::vector<char> a = random_bytes(n, 8), b = random_bytes(n, 9);
    const std::string dir = f.dir + "/capped", name = hex_name(0x8ull), other = hex_name(0x9ull);
    const std::string path = dir + "/" + name, filler = dir + "/ffffffffffffffff";
    ggml_tensor t = make_tensor(&f.cpu, f.base + 76 * page(), n, "capped");
    check(mkdir(dir.c_str(), 0755) == 0 && sparse_file(filler, gib - (off_t) n / 2),
          "fill the capped cache directory to half a payload under 1 GiB");
    setenv("HC_PEER_CACHE_MAX_GIB", "1", 1);
    g_log.clear();
    check(hcpeer_save_cached(&t, 0, a.data(), n, dir.c_str(), name.c_str()) == HCPEER_COPY && !exists(path) &&
          no_tmp_files(dir) && backing_file(t.data).empty(), "a save past HC_PEER_CACHE_MAX_GIB is skipped and copied");
    check(g_log.find("past HC_PEER_CACHE_MAX_GIB") != std::string::npos, "the capped save is logged");
    check(sparse_file(filler, gib - (off_t) n) &&
          hcpeer_save_cached(&t, 0, a.data(), n, dir.c_str(), name.c_str()) == HCPEER_DONE && read_file(path) == a &&
          backing_file(t.data) == path, "a save that fills the cache exactly to the cap is saved and mapped");
    check(hcpeer_save_cached(&t, 0, b.data(), n, dir.c_str(), name.c_str()) == HCPEER_DONE && read_file(path) == b,
          "replacing a file of the same name counts only the new bytes against the cap");
    setenv("HC_PEER_CACHE_MAX_GIB", "0", 1);
    check(hcpeer_save_cached(&t, 0, a.data(), n, dir.c_str(), other.c_str()) == HCPEER_COPY &&
          !exists(dir + "/" + other), "a cap of 0 GiB saves nothing");
    for (const char * bad : {"abc", "-5", "", "1.5"}) {
        setenv("HC_PEER_CACHE_MAX_GIB", bad, 1);
        g_log.clear();
        hcpeer_log_banner(dir.c_str());
        check(g_log.find("cache capped at 0 GiB") != std::string::npos, "a malformed cap caps at 0 GiB");
    }
    unsetenv("HC_PEER_CACHE_MAX_GIB");
    g_log.clear();
    hcpeer_log_banner(dir.c_str());
    check(g_log.find("no cache size cap") != std::string::npos, "banner: no cap when HC_PEER_CACHE_MAX_GIB is unset");
    unlink(filler.c_str());
}

// A mapped hit reads at most the partial last page, so nothing but hcpeer marks the file as used. The agent's LRU
// ranks files by max(atime, mtime): a mapped file must look newest, and its mtime (crash purges read it) must stay.
static void check_mapped_files_rank_as_used(fixture & f) {
    const std::string dir = f.dir + "/capped", path = dir + "/" + hex_name(0x8ull);
    const time_t now = time(nullptr);
    // atime in the future: Linux relatime then never updates it on its own, so only hcpeer's futimens can move it
    const struct timespec times[2] = {{now + 86400, 0}, {1000000000, 0}};
    ggml_tensor t = make_tensor(&f.cpu, f.base + 84 * page(), 3 * page() + 1000, "used");
    check(utimensat(AT_FDCWD, path.c_str(), times, 0) == 0, "age the cache file: atime tomorrow, mtime in 2001");
    check(hcpeer_map_cached(&t, 0, dir.c_str(), 0x8ull) == HCPEER_DONE, "the cached file maps");
    struct stat st;
    check(stat(path.c_str(), &st) == 0 && st.st_atime >= now - 60 && st.st_atime <= now + 60 &&
          st.st_mtime == 1000000000, "a mapped file's atime is set to now and its mtime is kept");
}

static void check_log_is_bounded(fixture & f) {
    ggml_tensor t = make_tensor(&f.cpu, f.base + 16 * page() + 64, 3 * page() + 1000, "flood");
    const size_t before = count(g_log, "copying instead");
    for (int i = 0; i < 4000; i++) {
        hcpeer_map_cached(&t, 0, f.dir.c_str(), 0x1111222233334444ull);
    }
    const size_t logged = count(g_log, "copying instead") - before;
    check(logged > 0 && logged <= 512, "4000 fallbacks in a burst log at most the 512-line burst budget");
    sleep(2);  // the bucket refills one line a second
    hcpeer_map_cached(&t, 0, f.dir.c_str(), 0x1111222233334444ull);
    const size_t at = g_log.rfind("fallbacks so far, ");
    size_t dropped = 0;
    if (at != std::string::npos) {
        sscanf(g_log.c_str() + at, "fallbacks so far, %zu log lines dropped", &dropped);
    }
    check(dropped >= 4000 - logged, "the next logged line counts every dropped line");
}

static int run_on(fixture & f) {
    setenv("HC_PEER_MMAP", "1", 1);
    setenv("HC_PEER_CACHE_MIN_FREE_GIB", "0", 1);
    unsetenv("HC_PEER_CACHE_MAX_GIB");
    check(hcpeer_enabled(), "HC_PEER_MMAP=1 enables hcpeer");
    check_alignment();
    check_banner_and_stale_files(f);
    check_save_hit_and_cow(f);
    check_resave_keeps_old_mapping(f);
    check_in_place_truncation_is_fatal(f);
    check_fallbacks(f);
    check_short_files_are_misses(f);
    check_bank_fills(f);
    check_floor_and_bad_dir(f);
    check_cache_cap(f);
    check_mapped_files_rank_as_used(f);
    check_log_is_bounded(f);
    return g_failures == 0 ? 0 : 1;
}

// Off, hcpeer keeps the stock behaviour except that cache saves are atomic (no disk floor, a log line per save).
static int run_off(fixture & f) {
    setenv("HC_PEER_MMAP", "0", 1);
    setenv("HC_PEER_CACHE_MIN_FREE_GIB", "1048576", 1);  // a floor no disk meets: off must ignore it
    setenv("HC_PEER_CACHE_MAX_GIB", "0", 1);             // and a cap nothing fits under
    check(!hcpeer_enabled(), "HC_PEER_MMAP=0 leaves hcpeer off");
    check(hcpeer_alignment(&g_cpu_buft, 32, true) == 32, "off: alignment unchanged");
    const std::string dead = f.dir + "/.hcpeer-" + hex_name(0xabcdef0123456789ull) + "." +
                             std::to_string(dead_pid()) + ".tmp";
    std::ofstream(dead) << "x";
    hcpeer_log_banner(f.dir.c_str());
    check(g_log.find("off (v4, set HC_PEER_MMAP=1") != std::string::npos, "off: banner says off");
    check(!exists(dead), "off: the banner removes temporary files of stopped servers");
    const size_t n = 3 * page() + 1000;
    const std::vector<char> a = random_bytes(n, 1);
    const std::string path = f.dir + "/1111222233334444";
    ggml_tensor t = make_tensor(&f.cpu, f.base, n, "off");
    check(hcpeer_save_cached(&t, 0, a.data(), n, f.dir.c_str(), "1111222233334444") == HCPEER_COPY &&
          read_file(path) == a && no_tmp_files(f.dir), "off: the save is atomic, ignores floor and cap, then copies");
    check(g_log.find("[set_tensor] saved to '" + path + "'") != std::string::npos, "off: the save is logged as stock");
    check(hcpeer_map_cached(&t, 0, f.dir.c_str(), 0x1111222233334444ull) == HCPEER_STOCK, "off: hit is stock");
    write_file(path, a.data(), n - 5000);
    check(hcpeer_map_cached(&t, 0, f.dir.c_str(), 0x1111222233334444ull) == HCPEER_STOCK && exists(path),
          "off: a short file is left to the stock path as before");
    check(backing_file(f.base).empty(), "off: nothing is mapped");
    return g_failures == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <empty scratch dir> on|off\n", argv[0]);
        return 2;
    }
    fixture f = make_fixture(argv[1]);
    if (f.base == nullptr) {
        fprintf(stderr, "cannot allocate the test buffer\n");
        return 2;
    }
    const int rc = strcmp(argv[2], "on") == 0 ? run_on(f) : run_off(f);
    printf("%s: %d failures\n", rc == 0 ? "PASS" : "FAIL", g_failures);
    return rc;
}
