"""RPC cache seeding: FNV-1a 64 against an independent implementation, GGUF tensor selection and host-side hashing,
ranged downloads with atomic publish and resume, disk caps, progress, catalog validation and path refusal."""

from __future__ import annotations

import functools
import os
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from hypercompute import gguf
from hypercompute.agent import seed_cache
from hypercompute.agent.managed import load_catalog, resolve_sources, safe_file_name
from hypercompute.agent.seed_cache import (
    FNV_OFFSET,
    FetchResult,
    FetchTask,
    ManagedError,
    SeedRunner,
    SeedSource,
    SeedStopped,
    cache_name,
    fetch_tensor,
    fnv1a64,
    fnv1a64_file,
    part_path,
)
from hypercompute.schemas import ModelArtifact, ModelFile, SeedSpec, SeedTensor
from hypercompute.services import peer_seed
from hypercompute.services.peer_seed import (
    HashCache,
    ManagedRefusal,
    build_seed,
    hash_ranges,
    parse_layers,
    select_tensors,
)

REPO, REV = "org/model-GGUF", "a" * 40
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/model.gguf"
JOB = "b" * 32


def ref_fnv(data: bytes) -> int:
    """FNV-1a 64 written from the definition (offset basis, xor then multiply by the prime, mod 2^64)."""
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) % (1 << 64)
    return h


# --------------------------------------------------------------------------- FNV


@pytest.mark.parametrize("data,expected", [(b"", 0xCBF29CE484222325), (b"a", 0xAF63DC4C8601EC8C),
                                           (b"foobar", 0x85944171F73967E8)])
def test_fnv_matches_published_vectors(data, expected):
    assert fnv1a64(data) == expected == ref_fnv(data)


@pytest.mark.parametrize("n", [1, 7, 8, 9, 15, 16, 17, 1000, 4099])
def test_fnv_matches_reference_and_continues_across_chunks(n):
    data = os.urandom(n)
    assert fnv1a64(data) == ref_fnv(data)
    cut = n // 3
    assert fnv1a64(data[cut:], fnv1a64(data[:cut])) == ref_fnv(data)
    assert fnv1a64(memoryview(data)) == ref_fnv(data)


def test_fnv_file_range_and_short_file(tmp_path):
    blob = os.urandom(10_000)
    path = tmp_path / "f.bin"
    path.write_bytes(blob)
    assert fnv1a64_file(path, 123, 4567) == ref_fnv(blob[123:123 + 4567])
    with pytest.raises(ManagedError, match="ends"):
        fnv1a64_file(path, 9_000, 2_000)  # never a hash of fewer bytes than asked
    stop = tmp_path / "stop"
    stop.touch()
    with pytest.raises(SeedStopped):
        fnv1a64_file(path, 0, 100, str(stop))


# --------------------------------------------------------------------------- synthetic GGUF


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path: Path, tensors: list[tuple[str, int]], pad_byte: int = 0xAA) -> Path:
    """GGUF v3, F32 1-D tensors of `n` elements each, random data, alignment 32 with non-zero padding."""
    kv = _s("general.architecture") + struct.pack("<I", 8) + _s("llama")
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1) + kv
    data, infos = b"", b""
    for name, n in tensors:
        infos += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<IQ", 0, len(data))
        data += os.urandom(4 * n)
        data += bytes([pad_byte]) * ((-len(data)) % 32)
    head += infos
    head += b"\0" * ((-len(head)) % 32)
    path.write_bytes(head + data)
    return path


TENSORS = [("token_embd.weight", 3000), ("blk.0.attn_q.weight", 64), ("blk.0.ffn_up.weight", 2050),
           ("blk.1.ffn_up.weight", 1024), ("blk.1.ffn_down.weight", 2100), ("blk.2.ffn_up.weight", 2050),
           ("output.weight", 3000)]
THRESHOLD = 4096  # bytes; 1024 F32 elements is exactly the threshold and must be left out (ggml: size > threshold)


def fake_models(paths: list[Path], status: str = "downloaded", model_id: str = "m1") -> SimpleNamespace:
    files = [ModelFile(name=p.name, url=f"https://huggingface.co/{REPO}/resolve/{REV}/{p.name}", sha256="0" * 64,
                       size_bytes=p.stat().st_size) for p in paths]
    art = ModelArtifact(id=model_id, display_name="m", family="llama", params_b=1.0, quantization="F32", repo=REPO,
                        revision=REV, license="mit", files=files)
    return SimpleNamespace(get=lambda mid: art, view=lambda mid: SimpleNamespace(status=status),
                           file_paths=lambda mid: list(paths), artifact=art)


def test_select_uses_exact_nbytes_over_threshold_in_the_layers(tmp_path):
    path = write_gguf(tmp_path / "a.gguf", TENSORS)
    table = gguf.read_tensor_table([path])
    chosen, skipped = select_tensors(table, [0, 1], THRESHOLD)
    assert [t.name for t in chosen] == ["blk.0.ffn_up.weight", "blk.1.ffn_down.weight"]
    assert not skipped
    up = chosen[0]
    assert up.nbytes == 8200 and up.span == 8224  # padding is not part of the tensor the client hashes


def test_build_seed_hashes_host_bytes_like_the_rpc_client(tmp_path):
    path = write_gguf(tmp_path / "a.gguf", TENSORS)
    blob = path.read_bytes()
    build = build_seed(fake_models([path]), "m1", [0, 1, 2], cache_dir=tmp_path / "hc", workers=1,
                       threshold=THRESHOLD)
    table = {t.name: t for t in gguf.read_tensor_table([path])}
    want = [table[n] for n in ("blk.0.ffn_up.weight", "blk.1.ffn_down.weight", "blk.2.ffn_up.weight")]
    assert [(t.offset, t.nbytes) for t in build.tensors] == [(w.offset, w.nbytes) for w in want]
    for t in build.tensors:
        assert t.fnv64 == f"{ref_fnv(blob[t.offset:t.offset + t.nbytes]):016x}"
        assert t.file == "a.gguf"
    assert (build.repo, build.revision) == (REPO, REV)
    assert build.bytes_total == sum(w.nbytes for w in want)


def test_build_seed_maps_split_shards_to_their_catalog_files(tmp_path):
    a = write_gguf(tmp_path / "m-00001-of-00002.gguf", TENSORS[:3])
    b = write_gguf(tmp_path / "m-00002-of-00002.gguf", TENSORS[3:])
    build = build_seed(fake_models([a, b]), "m1", [0, 1, 2], cache_dir=None, workers=1, threshold=THRESHOLD)
    by_file = {t.file for t in build.tensors}
    assert by_file == {a.name, b.name}
    for t in build.tensors:
        blob = (tmp_path / t.file).read_bytes()
        assert t.fnv64 == f"{ref_fnv(blob[t.offset:t.offset + t.nbytes]):016x}"


def test_hash_cache_skips_rehash_until_the_file_changes(tmp_path, monkeypatch):
    path = write_gguf(tmp_path / "a.gguf", TENSORS)
    calls = []
    real = peer_seed.hash_ranges

    def counting(items, workers, stop_file=None):
        calls.append(len(items))
        return real(items, workers, stop_file)

    monkeypatch.setattr(peer_seed, "hash_ranges", counting)
    first = build_seed(fake_models([path]), "m1", [0, 1], cache_dir=tmp_path / "hc", workers=1, threshold=THRESHOLD)
    second = build_seed(fake_models([path]), "m1", [0, 1], cache_dir=tmp_path / "hc", workers=1, threshold=THRESHOLD)
    assert calls == [2] and first == second
    os.utime(path, ns=(time.time_ns(), time.time_ns() + 10**9))  # a re-download changes the stamp
    build_seed(fake_models([path]), "m1", [0, 1], cache_dir=tmp_path / "hc", workers=1, threshold=THRESHOLD)
    assert calls == [2, 2]
    assert HashCache.key("a.gguf", 1, 2) == "a.gguf:1:2"


def test_concurrent_seeds_of_one_model_keep_each_others_hashes(tmp_path, monkeypatch):
    """Regression: each build_seed saved only the hashes it had loaded plus its own, so seeding three peers at once
    (run_big --seed-peers) kept the last peer's hashes and the next run hashed the others again; the shared tmp file
    name also made one of two overlapping saves fail on Windows."""
    path = write_gguf(tmp_path / "a.gguf", TENSORS)
    real, both_loaded = peer_seed.hash_ranges, threading.Barrier(2, timeout=5)

    def together(items, workers, stop_file=None):
        both_loaded.wait()  # both builds read the cache file before either one saves
        return real(items, workers, stop_file)

    monkeypatch.setattr(peer_seed, "hash_ranges", together)
    models = fake_models([path])
    with ThreadPoolExecutor(2) as pool:
        builds = list(pool.map(lambda layer: build_seed(models, "m1", [layer], cache_dir=tmp_path / "hc", workers=1,
                                                        threshold=THRESHOLD), [0, 1]))
    seeded = [t for b in builds for t in b.tensors]
    cache = HashCache(tmp_path / "hc" / "m1.json", {"a.gguf": path})
    assert len(seeded) == 2
    assert [cache.get(HashCache.key(t.file, t.offset, t.nbytes)) for t in seeded] == [t.fnv64 for t in seeded]
    assert [p.name for p in (tmp_path / "hc").iterdir()] == ["m1.json"]  # no tmp file left behind


def test_hash_ranges_on_worker_processes_matches_sequential(tmp_path):
    blob = os.urandom(300_000)
    path = tmp_path / "f.bin"
    path.write_bytes(blob)
    items = [(str(path), 0, 100_000), (str(path), 100_000, 150_000), (str(path), 7, 9)]
    assert hash_ranges(items, workers=2) == [ref_fnv(blob[o:o + n]) for _, o, n in items]


def test_build_seed_refusals(tmp_path):
    path = write_gguf(tmp_path / "a.gguf", TENSORS)
    with pytest.raises(ManagedRefusal) as exc:
        build_seed(fake_models([path], status="downloading"), "m1", [0], cache_dir=None, workers=1)
    assert exc.value.code == "model_not_downloaded"
    with pytest.raises(ManagedRefusal) as exc:
        build_seed(fake_models([path]), "m1", [7, 8], cache_dir=None, workers=1, threshold=THRESHOLD)
    assert exc.value.code == "nothing_to_seed"


def test_parse_layers():
    assert parse_layers("30-35") == [30, 31, 32, 33, 34, 35]
    assert parse_layers("7") == [7]
    for bad in ("35-30", "a-b", "1,2", "-3"):
        with pytest.raises(ValueError):
            parse_layers(bad)


# --------------------------------------------------------------------------- ranged fetch -> cache file


class RangeServer:
    """Serves byte ranges of a blob like the Hugging Face CDN; records each Range header."""

    def __init__(self, blob: bytes, *, fail_first: int = 0, ignore_range: bool = False, cut: int | None = None):
        self.blob, self.fail_first, self.ignore_range, self.cut = blob, fail_first, ignore_range, cut
        self.ranges: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range", "")
        self.ranges.append(rng)
        if self.fail_first:
            self.fail_first -= 1
            return httpx.Response(503)
        if self.ignore_range:
            return httpx.Response(200, content=self.blob)
        a, b = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", rng).groups())
        body = self.blob[a:b + 1]
        if self.cut is not None:
            body, self.cut = body[: self.cut], None  # the connection drops once, mid-range
        return httpx.Response(206, content=body, headers={"Content-Range": f"bytes {a}-{b}/{len(self.blob)}"})


def client_for(server) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(server), follow_redirects=True)


def task_for(blob: bytes, cache: Path, offset: int = 100, nbytes: int = 50_000, fnv: int | None = None,
             stop_file: str | None = None) -> FetchTask:
    real = ref_fnv(blob[offset:offset + nbytes])
    return FetchTask(url=URL, offset=offset, nbytes=nbytes, fnv64=real if fnv is None else fnv, cache_dir=str(cache),
                     stop_file=stop_file, retries=2)


@pytest.fixture
def blob():
    return os.urandom(120_000)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(seed_cache, "time", SimpleNamespace(sleep=lambda s: None))  # retries back off instantly


def test_fetch_writes_exactly_the_range_under_its_hash_name(tmp_path, blob):
    task = task_for(blob, tmp_path)
    server = RangeServer(blob)
    result = fetch_tensor(task, client_for(server))
    final = tmp_path / cache_name(task.fnv64)
    assert result.status == "written" and result.downloaded == task.nbytes
    assert final.read_bytes() == blob[100:100 + 50_000]
    assert re.fullmatch(r"[0-9a-f]{16}", final.name)
    assert not part_path(tmp_path, task.fnv64).exists()
    assert server.ranges == ["bytes=100-50099"]
    again = fetch_tensor(task, client_for(server))
    assert again.status == "present" and len(server.ranges) == 1  # idempotent: nothing downloaded twice


def test_fetch_resumes_a_part_file(tmp_path, blob):
    task = task_for(blob, tmp_path)
    part_path(tmp_path, task.fnv64).write_bytes(blob[100:100 + 20_000])
    server = RangeServer(blob)
    result = fetch_tensor(task, client_for(server))
    assert result.status == "written" and result.downloaded == 30_000
    assert server.ranges == ["bytes=20100-50099"]
    assert (tmp_path / cache_name(task.fnv64)).read_bytes() == blob[100:50_100]


def test_fetch_recovers_from_a_dropped_connection_and_a_503(tmp_path, blob):
    task = task_for(blob, tmp_path)
    server = RangeServer(blob, fail_first=1, cut=10_000)
    result = fetch_tensor(task, client_for(server))
    assert result.status == "written"
    assert server.ranges == ["bytes=100-50099", "bytes=100-50099", "bytes=10100-50099"]


def test_hash_mismatch_never_publishes(tmp_path, blob):
    task = task_for(blob, tmp_path, fnv=0x1234)
    result = fetch_tensor(task, client_for(RangeServer(blob)))
    assert result.status == "failed" and "hash mismatch" in result.error
    assert not (tmp_path / cache_name(0x1234)).exists()
    assert not part_path(tmp_path, 0x1234).exists()


def test_server_ignoring_the_range_fails_without_writing(tmp_path, blob):
    task = task_for(blob, tmp_path)
    result = fetch_tensor(task, client_for(RangeServer(blob, ignore_range=True)))
    assert result.status == "failed" and "ignored the byte range" in result.error
    assert not (tmp_path / cache_name(task.fnv64)).exists()


def test_redirect_to_plain_http_is_refused(tmp_path, blob):
    task = task_for(blob, tmp_path)
    server = RangeServer(blob)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "huggingface.co":
            return httpx.Response(302, headers={"Location": "http://cdn.example/x"})
        return server(request)

    result = fetch_tensor(task, client_for(handler))
    assert result.status == "failed" and "non-https" in result.error
    assert not (tmp_path / cache_name(task.fnv64)).exists()


def test_truncated_cache_file_is_replaced_atomically(tmp_path, blob):
    task = task_for(blob, tmp_path)
    final = tmp_path / cache_name(task.fnv64)
    final.write_bytes(blob[100:5_000])  # an rpc-server write cut short: right name, wrong size
    result = fetch_tensor(task, client_for(RangeServer(blob)))
    assert result.status == "written" and final.read_bytes() == blob[100:50_100]


def test_failed_download_leaves_no_final_file(tmp_path, blob):
    task = task_for(blob, tmp_path)
    result = fetch_tensor(task, client_for(RangeServer(blob, fail_first=10)))
    assert result.status == "failed"
    assert not (tmp_path / cache_name(task.fnv64)).exists()


def test_stop_file_cancels(tmp_path, blob):
    stop = tmp_path / "stop"
    stop.touch()
    result = fetch_tensor(task_for(blob, tmp_path, stop_file=str(stop)), client_for(RangeServer(blob)))
    assert result.status == "cancelled"


def test_a_second_download_of_a_tensor_never_shares_its_part_file(tmp_path, blob, monkeypatch):
    """Regression: two downloads of one tensor (a stopping job's and a new job's, or an earlier agent's orphaned pool
    worker) appended to one part file, and each checked only the hash of the bytes it streamed itself, so the
    tensor's cache name got a mixed, oversized file that rpc-server would load as the weights."""
    monkeypatch.setattr(seed_cache, "CHUNK", 10_000)
    task = task_for(blob, tmp_path)
    server, wrote_some, go_on = RangeServer(blob), threading.Event(), threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        response = server(request)
        if len(server.ranges) > 1:
            return response

        def held():  # the first download stalls after its first chunk
            yield response.content[:10_000]
            wrote_some.set()
            go_on.wait(5)
            yield response.content[10_000:]

        return httpx.Response(206, headers={"Content-Range": response.headers["content-range"]}, content=held())

    with ThreadPoolExecutor(1) as pool:
        running = pool.submit(fetch_tensor, task, client_for(handler))
        assert wrote_some.wait(5)
        second = fetch_tensor(task, client_for(handler))
        go_on.set()
        first = running.result(10)
    assert first.status == "written"
    assert second.status == "failed" and "another download of this tensor" in second.error
    assert (tmp_path / cache_name(task.fnv64)).read_bytes() == blob[100:50_100]
    assert [p.name for p in tmp_path.iterdir()] == [cache_name(task.fnv64)]  # no part or lock file left behind


# --------------------------------------------------------------------------- the runner: progress and disk caps


def spec_for(blob: bytes, ranges: list[tuple[int, int]], part: int = 1, parts: int = 1, job: str = JOB) -> SeedSpec:
    tensors = [SeedTensor(file="model.gguf", offset=o, nbytes=n, fnv64=f"{ref_fnv(blob[o:o + n]):016x}")
               for o, n in ranges]
    return SeedSpec(job_id=job, model_id="m1", repo=REPO, revision=REV, part=part, parts=parts, tensors=tensors)


SOURCES = {"model.gguf": SeedSource("model.gguf", URL, 120_000)}


def make_runner(tmp_path, blob, *, max_disk=10**9, free=10**15, min_free=0, server=None):
    server = server or RangeServer(blob)
    client = client_for(server)
    return SeedRunner(tmp_path / "rpc", max_disk, min_free, workers=2,
                      executor_factory=lambda n: ThreadPoolExecutor(n),
                      fetch=functools.partial(fetch_tensor, client=client), disk_free=lambda p: free, poll_s=0.05)


def wait_state(runner: SeedRunner, states=("done", "failed", "cancelled"), timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        p = runner.progress()
        if p is not None and p.state in states:
            return p
        time.sleep(0.02)
    raise AssertionError(f"runner stuck: {runner.progress()}")


def test_runner_downloads_reports_progress_and_skips_present(tmp_path, blob):
    runner = make_runner(tmp_path, blob)
    ranges = [(0, 30_000), (30_000, 30_000), (60_000, 40_000)]
    (tmp_path / "rpc").mkdir()
    (tmp_path / "rpc" / cache_name(ref_fnv(blob[0:30_000]))).write_bytes(blob[0:30_000])  # already cached
    assert "3 tensors queued" in runner.submit(spec_for(blob, ranges), SOURCES)
    p = wait_state(runner)
    runner.close()
    assert p.state == "done" and p.error is None
    assert (p.tensors_total, p.tensors_done, p.tensors_present, p.tensors_failed) == (3, 3, 1, 0)
    assert p.bytes_done == p.bytes_total == 100_000
    for o, n in ranges:
        assert (tmp_path / "rpc" / cache_name(ref_fnv(blob[o:o + n]))).read_bytes() == blob[o:o + n]


def test_runner_waits_for_every_part_and_dedups_resent_parts(tmp_path, blob):
    runner = make_runner(tmp_path, blob)
    runner.submit(spec_for(blob, [(0, 10_000)], part=1, parts=2), SOURCES)
    time.sleep(0.3)
    p = runner.progress()
    assert p.state == "running" and p.parts_received == 1 and p.tensors_done == 1
    assert "already received" in runner.submit(spec_for(blob, [(0, 10_000)], part=1, parts=2), SOURCES)
    runner.submit(spec_for(blob, [(10_000, 10_000)], part=2, parts=2), SOURCES)
    assert wait_state(runner).state == "done"
    runner.close()


def test_runner_respects_max_disk(tmp_path, blob):
    runner = make_runner(tmp_path, blob, max_disk=45_000)
    runner.submit(spec_for(blob, [(0, 30_000), (30_000, 30_000)]), SOURCES)
    p = wait_state(runner, ("failed",))
    runner.close()
    assert "--max-disk" in p.error
    files = [f for f in (tmp_path / "rpc").iterdir() if not f.name.startswith(".")]
    assert sum(f.stat().st_size for f in files) <= 45_000


def test_runner_keeps_the_free_space_floor(tmp_path, blob):
    runner = make_runner(tmp_path, blob, free=50_000, min_free=40_000)
    runner.submit(spec_for(blob, [(0, 30_000)]), SOURCES)
    p = wait_state(runner, ("failed",))
    runner.close()
    assert "free on the cache disk" in p.error and p.tensors_done == 0
    assert not any((tmp_path / "rpc").glob("[0-9a-f]" * 16))


def test_runner_cancel_and_replacement(tmp_path, blob):
    runner = make_runner(tmp_path, blob)
    runner.submit(spec_for(blob, [(0, 10_000)], parts=3), SOURCES)
    assert runner.cancel() is True
    assert runner.progress().state == "cancelled"
    with pytest.raises(ManagedError, match="cancelled"):
        runner.submit(spec_for(blob, [(10_000, 10_000)], part=2, parts=3), SOURCES)
    runner.submit(spec_for(blob, [(20_000, 10_000)], job="c" * 32), SOURCES)  # a new job replaces it
    p = wait_state(runner)
    runner.close()
    assert p.job_id == "c" * 32 and p.state == "done"


def test_runner_reports_failed_tensors(tmp_path, blob):
    runner = make_runner(tmp_path, blob)
    spec = spec_for(blob, [(0, 10_000)])
    bad = SeedSpec(**{**spec.model_dump(), "tensors": [{**spec.tensors[0].model_dump(), "fnv64": "0" * 15 + "1"}]})
    runner.submit(bad, SOURCES)
    p = wait_state(runner)
    runner.close()
    assert p.state == "failed" and p.tensors_failed == 1 and "hash mismatch" in p.failures[0].error


def test_late_part_of_a_stopped_job_is_refused_not_reported_done(tmp_path, blob):
    """Regression: a part arriving after the disk cap stopped the job used to restart it, and the job then ended
    `done` although the tensors the cap had dropped were never seeded."""
    runner = make_runner(tmp_path, blob, max_disk=45_000)
    runner.submit(spec_for(blob, [(0, 30_000), (30_000, 30_000)], part=1, parts=2), SOURCES)
    assert "--max-disk" in wait_state(runner, ("failed",)).error
    with pytest.raises(ManagedError, match="--max-disk"):
        runner.submit(spec_for(blob, [(60_000, 10_000)], part=2, parts=2), SOURCES)
    assert "already received" in runner.submit(spec_for(blob, [(0, 30_000)], part=1, parts=2), SOURCES)
    time.sleep(0.2)
    p = runner.progress()
    runner.close()
    assert p.state == "failed" and "--max-disk" in p.error and p.parts_received == 1


class HeldFetch:
    """A fetch that holds each download until released, ends early once its job's stop file appears, and records
    how many downloads of one tensor ran at the same time."""

    def __init__(self) -> None:
        self.release, self.lock = threading.Event(), threading.Lock()
        self.running: dict[int, int] = {}
        self.calls: list[int] = []
        self.most = 0

    def __call__(self, task: FetchTask) -> FetchResult:
        with self.lock:
            self.calls.append(task.fnv64)
            self.running[task.fnv64] = self.running.get(task.fnv64, 0) + 1
            self.most = max(self.most, self.running[task.fnv64])
        try:
            while not self.release.wait(0.01):
                if task.stop_file is not None and os.path.exists(task.stop_file):
                    return FetchResult(task.fnv64, "cancelled")
            return FetchResult(task.fnv64, "written", task.nbytes)
        finally:
            with self.lock:
                self.running[task.fnv64] -= 1


def test_a_job_stopped_by_a_disk_limit_stops_its_downloads_before_a_new_job_takes_them(tmp_path):
    """Regression: a job the disk cap stopped kept its downloads running and was not counted as stopping, so the
    owner's smaller retry started the same tensor while the old pool worker still appended to its part file."""
    fetch = HeldFetch()
    runner = SeedRunner(tmp_path / "rpc", 1_500, 0, workers=2, executor_factory=lambda n: ThreadPoolExecutor(n),
                        fetch=fetch, disk_free=lambda p: 10**15, poll_s=0.02)
    one = SeedTensor(file="model.gguf", offset=0, nbytes=1_000, fnv64="1" * 16)
    two = SeedTensor(file="model.gguf", offset=1_000, nbytes=1_000, fnv64="2" * 16)

    def spec(job: str, tensors: list[SeedTensor]) -> SeedSpec:
        return SeedSpec(job_id=job, model_id="m1", repo=REPO, revision=REV, part=1, parts=1, tensors=tensors)

    runner.submit(spec(JOB, [one, two]), SOURCES)
    assert "--max-disk" in wait_state(runner, ("failed",)).error  # one is downloading, two would pass the cap
    runner.submit(spec("c" * 32, [one]), SOURCES)  # the owner seeds fewer layers, as the refusal says
    deadline = time.monotonic() + 5
    while len(fetch.calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    fetch.release.set()
    p = wait_state(runner)
    runner.close()
    assert fetch.calls == [int("1" * 16, 16)] * 2 and fetch.most == 1
    assert p.job_id == "c" * 32 and p.state == "done"


def cache_file(cache: Path, data: bytes, age_s: float) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / cache_name(ref_fnv(data))
    path.write_bytes(data)
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def test_seeding_evicts_the_oldest_other_cache_files_to_stay_under_the_cap(tmp_path, blob):
    cache = tmp_path / "rpc"
    oldest = cache_file(cache, os.urandom(20_000), age_s=7200)
    newer = cache_file(cache, os.urandom(20_000), age_s=60)
    part = cache / ".seed-00000000000000ff.part"
    part.write_bytes(b"x" * 5_000)  # another job's resumable download: never evicted
    runner = make_runner(tmp_path, blob, max_disk=70_000)
    runner.submit(spec_for(blob, [(0, 30_000)]), SOURCES)
    p = wait_state(runner)
    runner.close()
    assert p.state == "done" and p.tensors_done == 1
    assert not oldest.exists() and newer.exists() and part.exists()
    assert seed_cache.cache_usage(cache) <= 70_000


def test_evict_cache_spares_part_files_and_kept_names(tmp_path):
    cache = tmp_path / "rpc"
    a = cache_file(cache, b"a" * 1000, age_s=300)
    b = cache_file(cache, b"b" * 1000, age_s=200)
    c = cache_file(cache, b"c" * 1000, age_s=100)
    (cache / ".seed-0000000000000001.part").write_bytes(b"p" * 1000)
    (cache / "notes.txt").write_bytes(b"n" * 1000)
    assert seed_cache.evict_cache(cache, 1500, keep={a.name}) == 2000
    assert a.exists() and not b.exists() and not c.exists()
    assert (cache / ".seed-0000000000000001.part").exists() and (cache / "notes.txt").exists()
    assert seed_cache.evict_cache(cache, 0) == 0 and seed_cache.evict_cache(tmp_path / "missing", 10) == 0
    assert seed_cache.bytes_over_limits(used=100, free=1000, max_bytes=80, min_free=0) == 20
    assert seed_cache.bytes_over_limits(used=0, free=1000, max_bytes=10**6, min_free=900, extra=300) == 200
    assert seed_cache.bytes_over_limits(used=10, free=10**6, max_bytes=100, min_free=10) == 0


def test_purge_recent_drops_what_a_killed_worker_may_have_cut_short(tmp_path):
    cache = tmp_path / "rpc"
    old = cache_file(cache, b"o" * 1000, age_s=3600)
    fresh = cache_file(cache, b"f" * 1000, age_s=5)
    part = cache / ".seed-0000000000000001.part"
    part.write_bytes(b"p")
    assert seed_cache.purge_recent(cache, time.time() - 120) == 1
    assert old.exists() and not fresh.exists() and part.exists()
    assert seed_cache.purge_recent(tmp_path / "missing", 0) == 0


def test_cache_files_go_least_recently_used_first_with_mtime_where_reads_are_not_recorded(tmp_path):
    cache = tmp_path / "rpc"
    now = time.time()
    hot = cache_file(cache, b"h" * 1000, age_s=3000)
    os.utime(hot, (now - 5, now - 3000))  # written long ago, read at the last load
    cold = cache_file(cache, b"c" * 1000, age_s=2000)
    noatime = cache_file(cache, b"n" * 1000, age_s=1000)
    os.utime(noatime, (now - 9000, now - 1000))  # atime older than the write: the volume does not record reads
    (cache / ("e" * 16)).mkdir()  # a directory under a hash-shaped name is not a cache file
    assert [f.path for f in seed_cache._finished_cache_files(cache)] == [cold, noatime, hot]
    assert seed_cache.evict_cache(cache, 1500) == 2000 and hot.exists() and not cold.exists()
    assert seed_cache.purge_recent(cache, now - 60) == 0  # a read is not a write: hot was not cut short by a crash


class GoneEntry:
    """A listed directory entry whose file was removed before its stat (DirEntry.stat raises on POSIX)."""

    def __init__(self, name: str) -> None:
        self.name, self.path = name, f"/gone/{name}"

    def is_file(self, follow_symlinks: bool = True) -> bool:
        return True

    def stat(self, follow_symlinks: bool = True) -> os.stat_result:
        raise FileNotFoundError(self.path)


def test_eviction_skips_files_that_vanish_or_cannot_be_deleted(tmp_path, monkeypatch):
    assert seed_cache._cache_entry(GoneEntry("0" * 16)) is None and seed_cache._entry_size(GoneEntry("x")) == 0
    cache = tmp_path / "rpc"
    locked = cache_file(cache, b"l" * 1000, age_s=300)  # oldest, but open in rpc-server (Windows refuses)
    gone = cache_file(cache, b"g" * 1000, age_s=200)  # removed by another pass between listing and deleting
    spare = cache_file(cache, b"s" * 1000, age_s=100)
    real_unlink = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self == locked:
            raise PermissionError(13, "in use by another process", str(self))
        if self == gone:
            real_unlink(self)
            raise FileNotFoundError(str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert seed_cache.evict_cache(cache, 1500) == 1000  # only what this pass deleted counts
    assert locked.exists() and not gone.exists() and not spare.exists()
    assert seed_cache.cache_usage(cache) == 1000


# --------------------------------------------------------------------------- catalog validation


def catalog_entry() -> dict[str, ModelArtifact]:
    f = ModelFile(name="model.gguf", url=URL, sha256="0" * 64, size_bytes=120_000)
    art = ModelArtifact(id="m1", display_name="m", family="llama", params_b=1.0, quantization="F32", repo=REPO,
                        revision=REV, license="mit", files=[f])
    return {"m1": art}


def seed(tensors: list[dict], **over) -> SeedSpec:
    base = {"job_id": JOB, "model_id": "m1", "repo": REPO, "revision": REV, "part": 1, "parts": 1,
            "tensors": tensors}
    return SeedSpec(**{**base, **over})


T = {"file": "model.gguf", "offset": 0, "nbytes": 100, "fnv64": "0" * 16}


def test_sources_come_from_the_catalog_url():
    assert resolve_sources(seed([T]), catalog_entry())["model.gguf"].url == URL


@pytest.mark.parametrize("change,match", [
    ({"model_id": "other"}, "not in this machine's catalog"),
    ({"revision": "b" * 40}, "differs"),
    ({"repo": "evil/repo"}, "differs"),
    ({"tensors": [{**T, "file": "../../etc/passwd"}]}, "unsafe file name"),
    ({"tensors": [{**T, "file": "sub/model.gguf"}]}, "unsafe file name"),
    ({"tensors": [{**T, "file": "..\\model.gguf"}]}, "unsafe file name"),
    ({"tensors": [{**T, "file": "other.gguf"}]}, "is not a file of"),
    ({"tensors": [{**T, "offset": 119_950}]}, "past the end"),
])
def test_seed_requests_outside_the_catalog_are_refused(change, match):
    with pytest.raises(ManagedError, match=match):
        resolve_sources(seed(**{"tensors": [T], **change}), catalog_entry())


def test_non_huggingface_catalog_url_is_refused():
    cat = catalog_entry()
    art = cat["m1"]
    bad = art.model_copy(update={"files": [art.files[0].model_copy(update={"url": "https://evil.example/x.gguf"})]})
    with pytest.raises(ManagedError, match="resolve URL"):
        resolve_sources(seed([T]), {"m1": bad})


def with_url(url: str) -> dict[str, ModelArtifact]:
    art = catalog_entry()["m1"]
    return {"m1": art.model_copy(update={"files": [art.files[0].model_copy(update={"url": url})]})}


def test_a_subfolder_in_the_pinned_resolve_url_is_accepted():
    # unsloth and others keep each quantization in a folder; the host stores the GGUF flat, so `path` stays unset
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/UD-Q2_K_XL/model.gguf"
    assert resolve_sources(seed([T]), with_url(url))["model.gguf"].url == url


@pytest.mark.parametrize("tail", ["../model.gguf", "UD/../model.gguf", "UD/other.gguf", "UD/.model.gguf",
                                  "UD//model.gguf", "UD\\model.gguf", "model.gguf?x=1", "a/b c/model.gguf"])
def test_a_resolve_url_must_end_in_this_file_through_safe_folders(tail):
    with pytest.raises(ManagedError, match="resolve URL"):
        resolve_sources(seed([T]), with_url(f"https://huggingface.co/{REPO}/resolve/{REV}/{tail}"))


def test_real_catalog_qwen235_entry_resolves():
    cat = load_catalog()
    art = cat["qwen3-235b-a22b-instruct-2507-ud-q2_k_xl"]
    for f in art.files:
        spec = seed([{**T, "file": f.name}], model_id=art.id, repo=art.repo, revision=art.revision)
        assert resolve_sources(spec, cat)[f.name].url == f.url


def test_real_catalog_gpt_oss_entry_resolves():
    cat = load_catalog()
    art = cat["gpt-oss-120b-mxfp4"]
    spec = seed([{**T, "file": art.files[0].name}], model_id=art.id, repo=art.repo, revision=art.revision)
    src = resolve_sources(spec, cat)[art.files[0].name]
    assert src.url == art.files[0].url and src.url.startswith("https://huggingface.co/ggml-org/")


def test_real_catalog_deepseek_v31_entry_resolves():
    cat = load_catalog()
    art = cat["deepseek-v3.1-ud-tq1_0"]
    (f,) = art.files
    spec = seed([{**T, "file": f.name}], model_id=art.id, repo=art.repo, revision=art.revision)
    src = resolve_sources(spec, cat)[f.name]
    assert src.url == f.url == f"https://huggingface.co/unsloth/DeepSeek-V3.1-GGUF/resolve/{art.revision}/{f.name}"


def test_real_catalog_kimi_k26_entry_resolves():
    cat = load_catalog()
    art = cat["kimi-k2.6-iq1_s"]
    assert [f.name for f in art.files] == [f"moonshotai_Kimi-K2.6-IQ1_S-0000{i}-of-00006.gguf" for i in range(1, 7)]
    assert sum(f.size_bytes for f in art.files) == 207404948128  # the HF tree API total of the six parts
    base = f"https://huggingface.co/bartowski/moonshotai_Kimi-K2.6-GGUF/resolve/{art.revision}/moonshotai_Kimi-K2.6-IQ1_S/"
    for f in art.files:
        spec = seed([{**T, "file": f.name}], model_id=art.id, repo=art.repo, revision=art.revision)
        assert resolve_sources(spec, cat)[f.name].url == f.url == base + f.name


def test_safe_file_name():
    assert safe_file_name("gpt-oss-120b-MXFP4.gguf")
    for bad in ("", "..", "a/../b", "/abs", "C:x", ".hidden", "a\\b", "a b"):
        assert not safe_file_name(bad)


def test_fnv_offset_constant():
    assert FNV_OFFSET == 0xCBF29CE484222325
