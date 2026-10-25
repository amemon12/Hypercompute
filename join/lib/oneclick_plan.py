"""Device, memory cap, cache and managed choices, and the join command (PACKAGING-SPEC 2.4 steps 5 to 12)."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from oneclick_core import GIB, Ctx, Stop, ask_gib, ask_name, ask_yes, clean_name, parse_gib
from oneclick_invite import CODE_TOKEN, TICKET_TOKEN

DEVICE_RE = re.compile(r"[A-Za-z0-9]{1,16}")
RELAY_RE = re.compile(r"https://[^\s\"'<>`]{1,500}")

Inventory = Mapping[str, Any]
# The NVIDIA driver the GPU engine needs, by (os.system, os.machine), as engine.choose_variant picks it. Below it the
# installer picks the CPU build again, so `engine install --force` would loop. Linux arm64 has no minimum there.
CUDA_MIN_DRIVER = {("windows", "x86_64"): 580, ("linux", "x86_64"): 570}
# What --managed adds on top of downloads (agent/managed.py consent_lines, agent/join.py print_plan over Iroh).
MANAGED_RESTART = ("the host may also restart this computer's worker on {device} and change how many CPU threads it "
                   "uses, from 1 up to all {cores} cores")
MANAGED_UNATTENDED = ("if the host goes away, this computer keeps waiting for it for up to 2 hours, and a crashed "
                      "worker is started again (at most 3 times in 30 minutes)")


def managed_terms(device: str) -> tuple[str, str]:
    return MANAGED_RESTART.format(device=device, cores=max(1, os.cpu_count() or 1)), MANAGED_UNATTENDED


def _pools(inv: Inventory) -> list[Mapping[str, Any]]:
    return list(inv.get("memory_pools") or [])


def ram_pool(inv: Inventory) -> Mapping[str, Any] | None:
    return next((p for p in _pools(inv) if p.get("id") in ("ram", "uma")), None)


def find_device(inv: Inventory, device: str) -> Mapping[str, Any] | None:
    return next((d for d in inv.get("devices") or [] if d.get("id") == device), None)


def pool_for(inv: Inventory, device: str) -> Mapping[str, Any] | None:
    """The device's pool through devices[].pool_id. The CPU uses the ram or uma pool."""
    dev = find_device(inv, device) if device != "CPU" else None
    if dev is not None:
        pool = next((p for p in _pools(inv) if p.get("id") == dev.get("pool_id")), None)
        if pool is not None:
            return pool
    return ram_pool(inv)


def pool_total(pool: Mapping[str, Any], dev: Mapping[str, Any] | None) -> int:
    return int(pool.get("total_bytes") or (dev or {}).get("total_bytes") or 0)


def default_cap_bytes(pool: Mapping[str, Any], dev: Mapping[str, Any] | None = None) -> int:
    total = pool_total(pool, dev)
    if pool.get("kind") == "discrete_vram":
        return int(total * 0.80)  # below the agent's total - 512 MiB: a laptop GPU also drives the display
    if pool.get("kind") == "unified":
        return max(0, min(int(total * 0.65), total - 6 * GIB))
    return max(0, min(int(total * 0.40), total - 8 * GIB))


def default_reserve_bytes(inv: Inventory) -> int:
    total = int((ram_pool(inv) or {}).get("total_bytes") or 16 * GIB)
    return max(4 * GIB, int(total * 0.20))


def engine_names(inv: Inventory) -> list[str]:
    return [str(d.get("name")) for d in (inv.get("engine") or {}).get("devices") or []]


def nvidia_present(ctx: Ctx, inv: Inventory) -> bool:
    if not ctx.dry_run:
        return ctx.nvidia_ok()
    found = any(d.get("backend") == "cuda" or str(d.get("vendor") or "").lower().startswith("nvidia")
                for d in inv.get("devices") or [])
    ctx.note(f"would run: nvidia-smi -L (assumed to {'work' if found else 'fail'}, from the inventory devices)")
    return found


def nvidia_driver_major(inv: Inventory) -> int | None:
    """The major number of the NVIDIA driver nvidia-smi reported in the inventory, e.g. 581 for "581.57"."""
    for dev in inv.get("devices") or []:
        if dev.get("backend") == "cuda" or str(dev.get("vendor") or "").lower().startswith("nvidia"):
            head = str(dev.get("driver_version") or "").split(".")[0].strip()
            if head.isdigit():
                return int(head)
    return None


def engine_fix(inv: Inventory) -> str:
    """How to get a GPU engine: a driver update when the driver is too old for one, else a forced reinstall."""
    osinfo = inv.get("os") or {}
    need = CUDA_MIN_DRIVER.get((osinfo.get("system"), osinfo.get("machine")))
    major = nvidia_driver_major(inv)
    if need is not None and major is not None and major < need:
        return (f"Your NVIDIA driver ({major}) is too old for the GPU engine. Update it to {need} or newer, then "
                "close this window and run: uv run hypercompute engine install")
    return "To fix that, close this window and run: uv run hypercompute engine install --force"


def pick_device(ctx: Ctx, inv: Inventory) -> str:
    names = engine_names(inv)
    override = ctx.env.get("HC_DEVICE")
    if override:
        if DEVICE_RE.fullmatch(override) and (override in names or override == "CPU"):
            return override
        raise Stop(f"HC_DEVICE={override[:16]} is not a device the engine can use. It can use: "
                   f"{', '.join([*names, 'CPU'])}.")
    osinfo = inv.get("os") or {}
    if osinfo.get("system") == "macos" and osinfo.get("machine") == "arm64" and "MTL0" in names:
        return "MTL0"
    if nvidia_present(ctx, inv):
        if "CUDA0" in names:
            return "CUDA0"
        variant = (inv.get("engine") or {}).get("variant") or "unknown"
        ctx.say(f"This computer has an NVIDIA GPU, but the installed engine ({variant}) cannot see it.")
        ctx.say(engine_fix(inv))
        if not ask_yes(ctx, "Share the CPU instead? It is slow. [y/N]", False):
            raise Stop("Not sharing. Nothing was changed.")
        return "CPU"
    ctx.say("No supported GPU found. Sharing the CPU is slow.")
    return "CPU"


@dataclass(frozen=True)
class Memory:
    max_gib: int
    reserve_gib: int
    pool_text: str


def plan_memory(ctx: Ctx, inv: Inventory, device: str) -> Memory:
    pool, dev = pool_for(inv, device), find_device(inv, device)
    if pool is None or pool_total(pool, dev) <= 0:
        raise Stop("Could not read this computer's memory from the inventory.")
    cap = default_cap_bytes(pool, dev) // GIB
    if cap < 1:
        raise Stop("This computer does not have enough memory to share.")
    total_gib = pool_total(pool, dev) // GIB
    chosen = ask_gib(ctx, f"Memory the host may use [{cap} GiB]", cap, 1, total_gib, "HC_MAX_MEMORY")
    if chosen > cap:
        ctx.say(f"Note: {chosen} GiB is above the safe default of {cap} GiB. This computer may slow down.")
    reserve = default_reserve_bytes(inv) // GIB
    if "HC_RESERVE" in ctx.env:
        ram_gib = int((ram_pool(inv) or {}).get("total_bytes") or 0) // GIB
        reserve = ask_gib(ctx, f"Memory kept free for you [{reserve} GiB]", reserve, 0, max(ram_gib, 1), "HC_RESERVE")
    free = pool.get("free_bytes")
    free_text = f", {free / GIB:.1f} GiB free now" if isinstance(free, int) else ""
    text = f"{pool.get('id')} ({pool.get('kind')}), {pool_total(pool, dev) / GIB:.1f} GiB total{free_text}"
    return Memory(chosen, reserve, text)


def ask_managed(ctx: Ctx, inv: Inventory, device: str) -> int | None:
    """Disk GiB the host may fill with downloads, or None for off. N = min(80, floor(disk free GiB - 30))."""
    free = inv.get("disk_free_bytes")
    if not isinstance(free, int):
        return None
    limit = min(80, math.floor(free / GIB - 30))
    if "HC_MAX_DISK" in ctx.env:
        top = max(1, math.floor(free / GIB - 20))
        limit = ask_gib(ctx, "Disk the host may fill", limit, 1, top, "HC_MAX_DISK")
    elif limit < 10:
        if ctx.env.get("HC_MANAGED") == "1":
            ctx.say("Not enough free disk for host downloads, so they stay off.")
        return None
    ctx.say("With host downloads on, " + "; ".join(managed_terms(device)) + ".")
    prompt = (f"Let the host download this computer's share of a model from huggingface.co into that cache "
              f"(up to {limit} GiB)? [y/N]")
    return limit if ask_yes(ctx, prompt, False, "HC_MANAGED") else None


def relay_url(ctx: Ctx) -> str | None:
    url = ctx.env.get("HC_RELAY_URL")
    if not url:
        return None
    if not RELAY_RE.fullmatch(url):
        raise Stop("HC_RELAY_URL must be an https:// address.")
    return url


@dataclass(frozen=True)
class JoinPlan:
    name: str
    device: str
    max_gib: str
    reserve_gib: str
    cache: bool
    disk_gib: str | None  # host downloads (managed) are on when set
    relay_url: str | None
    pool_text: str


def compose_join(python: str, plan: JoinPlan) -> list[str]:
    """Spec 2.4 step 12. The ticket and code stay placeholders; only the supervisor fills them in."""
    cmd = [python, "-m", "hypercompute.cli", "join", "--ticket", TICKET_TOKEN, "--code", CODE_TOKEN,
           "--name", plan.name, "--devices", plan.device, "--max-memory", f"{plan.max_gib}GiB",
           "--reserve", f"{plan.reserve_gib}GiB"]
    if plan.cache:
        cmd.append("--cache")
    cmd.append("--yes")
    if plan.disk_gib is not None:
        cmd += ["--managed", "--devices-allowed", plan.device, "--max-disk", f"{plan.disk_gib}GiB"]
    if plan.relay_url:
        cmd += ["--relay-url", plan.relay_url]
    return cmd


def ask_cache(ctx: Ctx) -> bool:
    cache = ask_yes(ctx, "Keep received model pieces on disk so the next load is quicker? [Y/n]", True, "HC_CACHE")
    if cache:
        ctx.say(f"The cache lives in {ctx.home / 'rpc-cache'} and stays there after you stop sharing.")
    return cache


def build_plan(ctx: Ctx, inv: Inventory | None) -> JoinPlan:
    if inv is None:
        return placeholder_plan(ctx)
    device = pick_device(ctx, inv)
    ctx.say(f"Device to share: {device}")
    memory = plan_memory(ctx, inv, device)
    name = ask_name(ctx, str(inv.get("hostname") or ""))
    cache = ask_cache(ctx)
    disk = ask_managed(ctx, inv, device)
    if disk is not None and not cache:
        ctx.say("Host downloads go into the cache, so the cache is on.")
        cache = True
    return JoinPlan(name, device, str(memory.max_gib), str(memory.reserve_gib), cache,
                    None if disk is None else str(disk), relay_url(ctx), memory.pool_text)


def placeholder_plan(ctx: Ctx) -> JoinPlan:
    """Dry run without --inventory-file: what the inventory decides stays a placeholder unless an HC_* sets it."""
    env = ctx.env
    disk = str(parse_gib(env.get("HC_MAX_DISK", "")) or "<D>") if env.get("HC_MANAGED") == "1" else None
    return JoinPlan(clean_name(env.get("HC_NAME", "")) or "<hostname>", env.get("HC_DEVICE") or "<device>",
                    str(parse_gib(env.get("HC_MAX_MEMORY", "")) or "<N>"),
                    str(parse_gib(env.get("HC_RESERVE", "")) or "<R>"), ask_cache(ctx), disk, relay_url(ctx),
                    "(read from the inventory on a real run)")
