#include "hcpeer.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// macOS is the shipped target. HCPEER_POSIX_TEST builds the same code on Linux for native/hcpeer/test only.
#if defined(__APPLE__) || defined(HCPEER_POSIX_TEST)
#    include <dirent.h>
#    include <fcntl.h>
#    include <signal.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <sys/statvfs.h>
#    include <unistd.h>
#    define HCPEER_SUPPORTED 1
#else
#    define HCPEER_SUPPORTED 0
#endif

namespace {

bool env_enabled() {
    const char * value = std::getenv("HC_PEER_MMAP");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool is_cpu_buffer(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && ggml_backend_buffer_get_type(buffer) == ggml_backend_cpu_buffer_type();
}

constexpr uint64_t DEFAULT_MIN_FREE_GIB = 20;  // agent/seed_cache.py MIN_FREE_BYTES
constexpr uint64_t NO_CACHE_CAP         = UINT64_MAX;

// A plain decimal number of GiB (at most 1 PiB) as bytes in *bytes. False for anything else, nullptr included.
bool parse_gib(const char * value, uint64_t * bytes) {
    if (value == nullptr || !std::isdigit((unsigned char) value[0])) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long gib = std::strtoull(value, &end, 10);
    if (errno != 0 || *end != '\0' || gib > (1ull << 20)) {
        return false;
    }
    *bytes = (uint64_t) gib << 30;
    return true;
}

// HC_PEER_CACHE_MIN_FREE_GIB as bytes: a plain decimal number of GiB, 0 disables the floor, anything else the default.
uint64_t min_free_bytes() {
    uint64_t bytes = 0;
    return parse_gib(std::getenv("HC_PEER_CACHE_MIN_FREE_GIB"), &bytes) ? bytes : DEFAULT_MIN_FREE_GIB << 30;
}

// HC_PEER_CACHE_MAX_GIB as bytes: the consented size of the cache directory (a managed peer's --max-disk, in whole
// GiB). Unset is no cap. Any value that is not a plain decimal number caps at 0, so a garbled cap saves nothing.
uint64_t max_cache_bytes() {
    const char * value = std::getenv("HC_PEER_CACHE_MAX_GIB");
    uint64_t bytes = 0;
    if (value == nullptr) {
        return NO_CACHE_CAP;
    }
    return parse_gib(value, &bytes) ? bytes : 0;
}

// The banner's cap clause.
std::string cap_clause() {
    const uint64_t cap = max_cache_bytes();
    if (cap == NO_CACHE_CAP) {
        return "no cache size cap";
    }
    return "cache capped at " + std::to_string(cap >> 30) + " GiB";
}

#if HCPEER_SUPPORTED

struct hcpeer_stats {
    size_t   tensors       = 0;
    uint64_t bytes         = 0;
    size_t   fallbacks     = 0;
    size_t   rejected      = 0;
    size_t   save_problems = 0;
    size_t   suppressed    = 0;
};

// Only the server thread touches these: rpc-server serves one client and one command at a time.
hcpeer_stats g_stats;

constexpr double MIB = 1024.0 * 1024.0;

// Log lines are a token bucket (a whole model load is logged, a steady stream is held to one line a second), so an
// unattended server cannot fill its log. Every line carries running totals, so dropped lines lose no counts.
constexpr double LOG_BURST       = 512.0;
constexpr double LOG_PER_SECOND  = 1.0;
double           g_log_tokens    = LOG_BURST;
auto             g_log_refilled  = std::chrono::steady_clock::now();

bool take_log_line() {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - g_log_refilled).count();
    g_log_refilled = now;
    g_log_tokens = std::min(LOG_BURST, g_log_tokens + elapsed * LOG_PER_SECOND);
    if (g_log_tokens < 1.0) {
        g_stats.suppressed++;
        return false;
    }
    g_log_tokens -= 1.0;
    return true;
}

// The tensor bytes [offset, offset + size) as a pointer, or nullptr when they leave the tensor or its buffer.
char * target_bytes(const ggml_tensor * tensor, uint64_t offset, size_t size) {
    const uint64_t  nbytes = ggml_nbytes(tensor);
    const uintptr_t base   = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uintptr_t end    = base + ggml_backend_buffer_get_size(tensor->buffer);
    const uintptr_t data   = (uintptr_t) tensor->data;
    if (size == 0 || offset > nbytes || size > nbytes - offset) {
        return nullptr;
    }
    if (data < base || data > end || offset > end - data || size > end - data - offset) {
        return nullptr;
    }
    return (char *) (data + offset);
}

void report_fallback(const ggml_tensor * tensor, const char * path, const char * why, int err) {
    g_stats.fallbacks++;
    if (!take_log_line()) {
        return;
    }
    GGML_LOG_WARN("[hcpeer] %s: %s%s%s, copying instead (%s, %zu fallbacks so far, %zu log lines dropped)\n",
                  ggml_get_name(tensor), why, err ? ": " : "", err ? std::strerror(err) : "", path, g_stats.fallbacks,
                  g_stats.suppressed);
}

void report_rejected(const ggml_tensor * tensor, const char * path, const char * why, int err) {
    g_stats.rejected++;
    if (!take_log_line()) {
        return;
    }
    GGML_LOG_WARN("[hcpeer] %s: %s%s%s, answering a cache miss (%s, %zu rejected so far, %zu log lines dropped)\n",
                  ggml_get_name(tensor), why, err ? ": " : "", err ? std::strerror(err) : "", path, g_stats.rejected,
                  g_stats.suppressed);
}

void report_mapped(const ggml_tensor * tensor, const char * path, size_t size) {
    g_stats.tensors++;
    g_stats.bytes += size;
    if (!take_log_line()) {
        return;
    }
    GGML_LOG_INFO("[hcpeer] mapped %s, %.2f MiB from %s (%zu tensors, %.2f GiB mapped since start)\n",
                  ggml_get_name(tensor), size / MIB, path, g_stats.tensors, g_stats.bytes / (MIB * 1024.0));
}

void report_save_problem(const std::string & path, const char * why, int err) {
    g_stats.save_problems++;
    if (!take_log_line()) {
        return;
    }
    GGML_LOG_WARN("[hcpeer] not caching %s: %s%s%s (%zu so far)\n", path.c_str(), why, err ? ": " : "",
                  err ? std::strerror(err) : "", g_stats.save_problems);
}

size_t page_size() {
    static const long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? (size_t) page : 0;
}

bool read_exact(int fd, char * dst, size_t len, off_t at) {
    while (len > 0) {
        const ssize_t n = pread(fd, dst, len, at);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n == 0) {
            errno = 0;  // end of file: the file shrank since fstat, no errno to report
            return false;
        }
        if (n < 0) {
            return false;
        }
        dst += n;
        len -= (size_t) n;
        at  += n;
    }
    return true;
}

// write(2) on macOS rejects a single call above INT_MAX bytes, so large tensors go out in 1 GiB pieces.
bool write_all(int fd, const char * src, size_t len) {
    while (len > 0) {
        const ssize_t n = write(fd, src, std::min(len, (size_t) 1 << 30));
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            if (n == 0) {
                errno = EIO;
            }
            return false;
        }
        src += n;
        len -= (size_t) n;
    }
    return true;
}

// Map the first `size` bytes of fd over dst: whole pages with mmap, the partial last page read into the anonymous
// memory already there (mapping it would cover the next tensor's bytes). On false, *why says what failed and the
// caller copies the whole range again, so a half-done state never survives.
bool map_over(int fd, char * dst, size_t size, const char ** why, int * err) {
    const size_t page = page_size();
    if (page == 0 || (uintptr_t) dst % page != 0) {
        *why = "tensor data is not page aligned (client allocated before HC_PEER_MMAP was on?)";
        return false;
    }
    const size_t head = size - size % page;
    if (head == 0) {
        *why = "smaller than one page";
        return false;
    }
    if (!read_exact(fd, dst + head, size - head, (off_t) head)) {
        *why = "reading the partial last page failed";
        *err = errno;
        return false;
    }
    // PROT_WRITE on a private mapping never reaches the file: a stray write copies that one page, as stock would.
    void * mapped = mmap(dst, head, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        *why = "mmap failed";
        *err = errno;
        // Make sure the range is plain writable memory again before the stock copy runs over it.
        if (mmap(dst, head, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0) == MAP_FAILED) {
            GGML_ABORT("[hcpeer] cannot restore anonymous memory after a failed mmap: %s", std::strerror(errno));
        }
        return false;
    }
    return true;
}

// A cache file opened for reading and its size, or -1 with *why and *err set (*err is ENOENT for no such file).
int open_cache_file(const char * path, uint64_t * size, const char ** why, int * err) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        *why = "cannot open the cache file";
        *err = errno;
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        *why = "the cache file is not a regular file";
        *err = 0;
        close(fd);
        return -1;
    }
    *size = (uint64_t) st.st_size;
    return fd;
}

// Set a mapped cache file's access time to now and keep its write time (crash purges read that). A mapped hit reads at
// most the partial last page, and the agent's LRU eviction ranks files by max(atime, mtime): unlinking a mapped file
// frees nothing until the mapping goes, so a file in use has to rank newest. A failure only costs that ranking.
void mark_used(const ggml_tensor * tensor, int fd, const char * path) {
    const struct timespec times[2] = {{0, UTIME_NOW}, {0, UTIME_OMIT}};
    if (futimens(fd, times) == 0) {
        return;
    }
    const int err = errno;
    if (take_log_line()) {
        GGML_LOG_WARN("[hcpeer] %s: cannot set the access time of %s, LRU eviction may pick it: %s\n",
                      ggml_get_name(tensor), path, std::strerror(err));
    }
}

// Map the first `size` bytes of fd over the tensor from offset, logging the result. The caller closes fd: the mapping
// keeps its own reference to the file.
bool map_fd(ggml_tensor * tensor, uint64_t offset, int fd, size_t size, const char * path) {
    const char * why = "the cache file does not fit the tensor";
    int err = 0;
    char * dst = target_bytes(tensor, offset, size);
    if (dst != nullptr && map_over(fd, dst, size, &why, &err)) {
        mark_used(tensor, fd, path);
        report_mapped(tensor, path, size);
        return true;
    }
    report_fallback(tensor, path, why, err);
    return false;
}

// SET_TENSOR: map the file just saved, which must still hold exactly the `size` bytes that were sent.
bool map_saved(ggml_tensor * tensor, uint64_t offset, const char * path, size_t size) {
    const char * why = nullptr;
    int err = 0;
    uint64_t have = 0;
    const int fd = open_cache_file(path, &have, &why, &err);
    if (fd < 0) {
        report_fallback(tensor, path, why, err);
        return false;
    }
    bool mapped = false;
    if (have != size) {
        report_fallback(tensor, path, "the cache file size differs from the data just sent", 0);
    } else {
        mapped = map_fd(tensor, offset, fd, size, path);
    }
    close(fd);
    return mapped;
}

// SET_TENSOR_HASH carries no size, so a cache file has to hold bytes the request can mean: the rest of the tensor from
// offset (the model loader sends every weight whole, at offset 0), or for a tensor of three or more dims a whole number
// of its outermost slices from a slice boundary (MoE cache bank fills send slot * expert size, n * expert size).
bool plausible_size(const ggml_tensor * tensor, uint64_t offset, uint64_t size) {
    const uint64_t nbytes = ggml_nbytes(tensor);
    if (size == 0 || offset >= nbytes || size > nbytes - offset) {
        return false;
    }
    if (size == nbytes - offset) {
        return true;
    }
    int outer = GGML_MAX_DIMS - 1;
    while (outer > 0 && tensor->ne[outer] <= 1) {
        outer--;
    }
    const uint64_t slice = tensor->nb[outer];
    return outer >= 2 && slice > 0 && offset % slice == 0 && size % slice == 0;
}

// A cache file that cannot be the data asked for (a short file an interrupted stock save left, for example) is removed,
// so the client's resend saves a complete one, and the hit is answered as a miss.
int reject_file(const ggml_tensor * tensor, uint64_t offset, const char * path, uint64_t have) {
    const bool removed = unlink(path) == 0;
    const int err = removed ? 0 : errno;
    char why[256];
    snprintf(why, sizeof(why), "%s the cache file, %" PRIu64 " bytes cannot be the data asked for at offset %" PRIu64
             " of a %zu-byte tensor", removed ? "removed" : "cannot remove", have, offset, ggml_nbytes(tensor));
    report_rejected(tensor, path, why, err);
    return HCPEER_MISS;
}

// SET_TENSOR_HASH with hcpeer on: see hcpeer_map_cached. Only a missing file is left silently to the stock path.
int serve_hit(ggml_tensor * tensor, uint64_t offset, const char * path) {
    const char * why = nullptr;
    int err = 0;
    uint64_t have = 0;
    const int fd = open_cache_file(path, &have, &why, &err);
    if (fd < 0) {
        if (err == ENOENT) {
            return HCPEER_STOCK;  // a plain cache miss, which the stock path reports
        }
        report_rejected(tensor, path, why, err);
        return HCPEER_MISS;
    }
    int action = HCPEER_STOCK;
    if (offset >= ggml_nbytes(tensor)) {
        report_fallback(tensor, path, "the offset is past the end of the tensor", 0);  // the stock checks decide
    } else if (!plausible_size(tensor, offset, have)) {
        action = reject_file(tensor, offset, path, have);
    } else if (is_cpu_buffer(tensor->buffer) && map_fd(tensor, offset, fd, (size_t) have, path)) {
        action = HCPEER_DONE;
    }
    close(fd);
    return action;
}

// True when the cache volume keeps the free-space floor after `size` more bytes. Unknown free space counts as full.
bool has_room(const std::string & dir, const std::string & path, size_t size) {
    struct statvfs st;
    if (statvfs(dir.c_str(), &st) != 0) {
        report_save_problem(path, "cannot read the free space of the cache volume", errno);
        return false;
    }
    const uint64_t avail = (uint64_t) st.f_bavail * (uint64_t) st.f_frsize;
    const uint64_t floor = min_free_bytes();
    if (avail < floor || avail - floor < size) {
        report_save_problem(path, "the cache volume would drop below HC_PEER_CACHE_MIN_FREE_GIB free", 0);
        return false;
    }
    return true;
}

// The bytes in the regular files of dir other than `skip`, counted as the agent's cache_usage counts them. False with
// errno set when dir cannot be listed.
bool dir_usage(const std::string & dir, const char * skip, uint64_t * used) {
    DIR * d = opendir(dir.c_str());
    if (d == nullptr) {
        return false;
    }
    uint64_t total = 0;
    struct stat st;
    errno = 0;
    for (struct dirent * e = readdir(d); e != nullptr; errno = 0, e = readdir(d)) {
        if (std::strcmp(e->d_name, skip) != 0 && fstatat(dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW) == 0 &&
            S_ISREG(st.st_mode)) {
            total += (uint64_t) st.st_size;
        }
    }
    const int err = errno;
    closedir(d);
    errno = err;
    *used = total;
    return err == 0;
}

// True when the cache directory stays within HC_PEER_CACHE_MAX_GIB with `size` bytes saved as `name` (an older file of
// that name is replaced, so it does not count). The directory is measured on every save, so the agent's evictions and
// seed downloads count as they are now. A save past the cap is skipped and copied: a mapped file keeps its disk until
// it is unmapped, and the agent's eviction cannot win that back, so nothing past the cap may be saved and mapped.
bool under_cap(const std::string & dir, const char * name, const std::string & path, size_t size) {
    const uint64_t cap = max_cache_bytes();
    if (cap == NO_CACHE_CAP) {
        return true;
    }
    uint64_t used = 0;
    if (!dir_usage(dir, name, &used)) {
        report_save_problem(path, "cannot measure the cache directory against HC_PEER_CACHE_MAX_GIB", errno);
        return false;
    }
    if (used > cap || cap - used < size) {
        report_save_problem(path, "the cache directory would grow past HC_PEER_CACHE_MAX_GIB", 0);
        return false;
    }
    return true;
}

// Write dir/name as a new file renamed into place: no reader ever sees a partial file under the hash name, and a
// tensor already mapped from an older file of that name keeps its pages. False leaves no file behind. The temporary
// name is this process's own (the banner removed any stale one), and O_EXCL never reuses or follows anything found
// there. `keep_free` applies HC_PEER_CACHE_MIN_FREE_GIB and HC_PEER_CACHE_MAX_GIB (hcpeer on); off, saves are
// unbounded and logged as stock does.
bool save_atomic(const std::string & dir, const char * name, const void * data, size_t size, bool keep_free) {
    const std::string path = dir + "/" + name;
    if (keep_free && !has_room(dir, path, size)) {
        return false;
    }
    if (keep_free && !under_cap(dir, name, path, size)) {
        return false;
    }
    const std::string tmp = dir + "/.hcpeer-" + name + "." + std::to_string((long) getpid()) + ".tmp";
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0644);
    if (fd < 0) {
        report_save_problem(path, "cannot create the temporary file", errno);
        return false;
    }
    const char * why = nullptr;
    if (!write_all(fd, (const char *) data, size)) {
        why = "writing failed";
    } else if (fsync(fd) != 0) {
        why = "fsync failed";
    }
    int err = why != nullptr ? errno : 0;
    if (close(fd) != 0 && why == nullptr) {
        why = "close failed";
        err = errno;
    }
    if (why == nullptr && rename(tmp.c_str(), path.c_str()) != 0) {
        why = "rename failed";
        err = errno;
    }
    if (why != nullptr) {
        unlink(tmp.c_str());
        report_save_problem(path, why, err);
        return false;
    }
    if (!keep_free || take_log_line()) {
        GGML_LOG_INFO("[set_tensor] saved to '%s'\n", path.c_str());
    }
    return true;
}

// The pid in a temporary file name ".hcpeer-<16 hex>.<pid>.tmp", or -1 for any other name.
long tmp_owner(const char * name) {
    const size_t len = std::strlen(name);
    const size_t prefix = 8, hex = 16, suffix = 4;
    if (len < prefix + hex + 2 + suffix || std::strncmp(name, ".hcpeer-", prefix) != 0 ||
        std::strcmp(name + len - suffix, ".tmp") != 0 || name[prefix + hex] != '.') {
        return -1;
    }
    for (size_t i = prefix; i < prefix + hex; i++) {
        if (!std::isxdigit((unsigned char) name[i])) {
            return -1;
        }
    }
    long pid = 0;
    for (size_t i = prefix + hex + 1; i < len - suffix; i++) {
        if (!std::isdigit((unsigned char) name[i]) || pid > 100000000) {
            return -1;
        }
        pid = pid * 10 + (name[i] - '0');
    }
    return pid > 0 ? pid : -1;
}

// Remove the temporary files of servers that no longer run. Called once at start, before this server saves anything,
// so a file carrying this server's own pid is a leftover of an earlier process that had the same pid.
void remove_stale_tmp(const char * dir) {
    DIR * d = opendir(dir);
    if (d == nullptr) {
        return;
    }
    size_t removed = 0;
    for (struct dirent * e = readdir(d); e != nullptr; e = readdir(d)) {
        const long pid = tmp_owner(e->d_name);
        if (pid <= 0) {
            continue;
        }
        const bool gone = pid == (long) getpid() || (kill((pid_t) pid, 0) != 0 && errno == ESRCH);
        if (gone && unlinkat(dirfd(d), e->d_name, 0) == 0) {
            removed++;
        }
    }
    closedir(d);
    if (removed > 0) {
        GGML_LOG_INFO("[hcpeer] removed %zu temporary cache files left by stopped servers\n", removed);
    }
}

#else

size_t page_size() {
    return 0;
}

bool map_saved(ggml_tensor *, uint64_t, const char *, size_t) {
    return false;
}

int serve_hit(ggml_tensor *, uint64_t, const char *) {
    return HCPEER_STOCK;
}

bool save_atomic(const std::string &, const char *, const void *, size_t, bool) {
    return false;
}

void remove_stale_tmp(const char *) {}

#endif  // HCPEER_SUPPORTED

}  // namespace

bool hcpeer_enabled() {
    static const bool enabled = HCPEER_SUPPORTED && env_enabled();
    return enabled;
}

void hcpeer_log_banner(const char * cache_dir) {
    if (HCPEER_SUPPORTED && cache_dir != nullptr) {
        remove_stale_tmp(cache_dir);  // saves are atomic with hcpeer on or off
    }
    if (!HCPEER_SUPPORTED) {
        GGML_LOG_INFO("  hcpeer mmap    : off (v%s, unsupported on this platform)\n", HCPEER_VERSION);
    } else if (!hcpeer_enabled()) {
        GGML_LOG_INFO("  hcpeer mmap    : off (v%s, set HC_PEER_MMAP=1 to enable)\n", HCPEER_VERSION);
    } else if (cache_dir == nullptr) {
        GGML_LOG_INFO("  hcpeer mmap    : off (v%s, HC_PEER_MMAP=1 needs -c)\n", HCPEER_VERSION);
    } else {
        GGML_LOG_INFO("  hcpeer mmap    : on (v%s, page %zu B, CPU weights mapped from the cache, cache saves keep "
                      "%" PRIu64 " GiB free, %s)\n", HCPEER_VERSION, page_size(), min_free_bytes() >> 30,
                      cap_clause().c_str());
    }
}

size_t hcpeer_alignment(ggml_backend_buffer_type_t buft, size_t alignment, bool has_cache_dir) {
    if (!hcpeer_enabled() || !has_cache_dir || buft != ggml_backend_cpu_buffer_type()) {
        return alignment;
    }
    const size_t page = page_size();
    return page > alignment ? page : alignment;
}

int hcpeer_map_cached(ggml_tensor * tensor, uint64_t offset, const char * cache_dir, uint64_t hash) {
    if (!hcpeer_enabled() || cache_dir == nullptr) {
        return HCPEER_STOCK;
    }
    char name[17];
    snprintf(name, sizeof(name), "%016" PRIx64, hash);
    const std::string path = std::string(cache_dir) + "/" + name;
    return serve_hit(tensor, offset, path.c_str());
}

int hcpeer_save_cached(ggml_tensor * tensor, uint64_t offset, const void * data, size_t size, const char * cache_dir,
                       const char * name) {
    if (!HCPEER_SUPPORTED || cache_dir == nullptr || name == nullptr) {
        return HCPEER_STOCK;
    }
    const bool on = hcpeer_enabled();
    if (!save_atomic(cache_dir, name, data, size, on) || !on || !is_cpu_buffer(tensor->buffer)) {
        return HCPEER_COPY;
    }
    const std::string path = std::string(cache_dir) + "/" + name;
    return map_saved(tensor, offset, path.c_str(), size) ? HCPEER_DONE : HCPEER_COPY;
}
