"""Engine variant selection, archive safety, contribution defaults, units and address classification."""

import io
import zipfile

import pytest
from factories import engine_info, make_inventory
from hypercompute import engine, netutil
from hypercompute.agent.contribution import default_contribution, default_max_bytes, eligible_devices
from hypercompute.schemas import EngineDevice
from hypercompute.units import GiB, MiB, parse_size


@pytest.fixture
def lock():
    return engine.load_lock()


def test_lock_pins_one_release_for_every_variant(lock):
    assert lock.release == "b11770" and lock.rpc_proto == "8.0.0"
    assert lock.commit.startswith("39da130f")
    for v in lock.variants.values():
        assert v.assets and all(len(a.sha256) == 64 and lock.release in a.url for a in v.assets)


@pytest.mark.parametrize("os_name,arch,nv,expected", [
    ("macos", "arm64", None, "macos-arm64"),
    ("windows", "x86_64", 591, "windows-x64-cuda13.4"),
    ("windows", "x86_64", None, "windows-x64-cpu"),
    ("windows", "x86_64", 560, "windows-x64-cpu"),
    ("linux", "x86_64", 575, "linux-x64-cuda12.8"),
    ("linux", "x86_64", 590, "linux-x64-cuda13.4"),
    ("linux", "arm64", 580, "linux-arm64-cuda13.4"),
])
def test_variant_choice(lock, os_name, arch, nv, expected):
    variant, reason = engine.choose_variant(lock, os_name, arch, nv)
    assert variant.key == expected and reason


def test_unsupported_platform_gets_build_instructions(lock):
    with pytest.raises(engine.EngineError) as e:
        engine.choose_variant(lock, "macos", "x86_64", None)
    assert "build_llama" in (e.value.fix or "")


def test_zip_slip_rejected(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../evil.txt", "x")
    archive = tmp_path / "evil.zip"
    archive.write_bytes(buf.getvalue())
    with pytest.raises(engine.EngineError):
        engine._safe_extract(archive, tmp_path / "out")
    assert not (tmp_path.parent / "evil.txt").exists()


def test_only_allowlisted_binaries(lock):
    with pytest.raises(engine.EngineError):
        engine.exe_name("cmd")


def test_download_verifies_sha256(tmp_path):
    import hashlib

    import httpx

    payload = b"hello engine"

    def client():
        return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload)))

    good = hashlib.sha256(payload).hexdigest()
    out = engine.download_verified("https://example.invalid/a.zip", tmp_path / "a.zip", good, len(payload),
                                   client=client())
    assert out.read_bytes() == payload
    with pytest.raises(engine.EngineError):
        engine.download_verified("https://example.invalid/b.zip", tmp_path / "b.zip", "0" * 64, len(payload),
                                 client=client())
    assert not (tmp_path / "b.zip").exists() and not (tmp_path / "b.zip.part").exists()


def test_uma_pool_counted_once():
    mac = make_inventory("macos", engine=engine_info(devices=[EngineDevice(
        name="MTL0", description="Apple M4", total_bytes=10 * GiB, free_bytes=10 * GiB)], accelerator="metal"),
        ram_total=16 * GiB)
    # CPU and MTL0 both live on the 'uma' pool: sharing both must not double the budget
    assert default_max_bytes(mac, ["MTL0", "CPU"]) == default_max_bytes(mac, ["MTL0"]) == 10 * GiB


def test_discrete_gpu_default_and_cpu_fallback():
    pc = make_inventory()
    devices, fallback = eligible_devices(pc)
    assert devices == ["CUDA0"] and not fallback
    c = default_contribution(pc)
    assert c.max_bytes == 12226 * MiB - 512 * MiB and c.enforcement == "soft_admission"
    no_gpu = make_inventory(engine=engine_info(devices=[]), vram_total=None)
    devices, fallback = eligible_devices(no_gpu)
    assert devices == ["CPU"] and fallback


@pytest.mark.parametrize("text,value", [("8GiB", 8 * GiB), ("512 MiB", 512 * MiB), ("2g", 2 * GiB),
                                        ("1000", 1000), ("1.5GB", 1_500_000_000)])
def test_parse_size(text, value):
    assert parse_size(text) == value


def test_parse_size_rejects_garbage():
    with pytest.raises(ValueError):
        parse_size("lots")


@pytest.mark.parametrize("ip,private", [("10.19.175.205", True), ("192.168.1.4", True), ("172.18.48.1", True),
                                        ("100.101.102.103", True), ("169.254.21.125", False), ("8.8.8.8", False),
                                        ("127.0.0.1", False), ("not-an-ip", False)])
def test_private_lan_classification(ip, private):
    assert netutil.is_private_lan(ip) is private


# --- regression: link-local direct links (finding 12) ---------------------------------------------------------------

@pytest.mark.parametrize("ip,ok", [("169.254.10.3", True), ("10.19.175.205", True), ("100.101.102.103", True),
                                   ("8.8.8.8", False), ("127.0.0.1", False), ("fe80::1", False), ("junk", False)])
def test_lan_or_link_local_classification(ip, ok):
    assert netutil.is_lan_or_link_local(ip) is ok


def _iface(name, ips, kind="ethernet", up=True):
    from hypercompute.schemas import NetworkInterface

    return NetworkInterface(name=name, ipv4=ips, is_up=up, speed_mbps=None, kind=kind)


def test_lan_addresses_lists_link_local_only_when_nothing_else(monkeypatch):
    monkeypatch.setattr(netutil, "route_source_ip", lambda *a, **k: None)
    monkeypatch.setattr(netutil, "list_interfaces", lambda: [
        _iface("Ethernet 2", ["169.254.10.2"]), _iface("down", ["169.254.9.9"], up=False)])
    assert netutil.lan_addresses() == ["169.254.10.2"]
    monkeypatch.setattr(netutil, "list_interfaces", lambda: [
        _iface("Ethernet 2", ["169.254.10.2"]), _iface("Wi-Fi", ["192.168.1.4"], kind="wifi")])
    assert netutil.lan_addresses() == ["192.168.1.4"]


def test_resolve_ipv4_returns_an_address_not_a_name():
    assert netutil.resolve_ipv4("localhost") == "127.0.0.1"
    assert netutil.resolve_ipv4("10.0.0.7") == "10.0.0.7"
    assert netutil.resolve_ipv4("no-such-host.invalid") is None
