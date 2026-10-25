"""hcbench: write a benchmark plan, run it through the paper driver, and report the results.

  uv run --no-sync python bench/hcbench.py plan --tag d70 --model llama-3.3-70b-instruct-q4_k_m --host g14 \
      --peer <node id> --peer <node id> --order APAP --prompt-tokens 64 --output-tokens 32 --reps 2 \
      --seed-peers --out results/capacity/paper-<date>-<name>/plan_d70.json
  uv run --no-sync python bench/hcbench.py run PLAN.json --index results/capacity/paper-<date>-<name>/index_d70.jsonl \
      --dry-run          (prints what would run; a real run needs --yes)
  uv run --no-sync python bench/hcbench.py report [--check]

`plan` checks every row with run_big.py's own parse_args. `run` only starts scripts/capacity/paper/matrix.py, which
runs each row with run_big.py against the gateway on 127.0.0.1:8001; hcbench itself opens no connection. `report`
reads the index files and run records and writes bench/RESULTS.md (bench/hcbench_report.py).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PureWindowsPath
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True  # run_big and batchlib live in a checkout a live campaign uses: leave no .pyc there

import hcbench_report  # noqa: E402

NAME = re.compile(r"^[a-z0-9]+$")  # tags and host names: no dashes, so no label prefix catches another group
NODE_ID = re.compile(r"^[a-f0-9]{32}$")  # schemas.py:164
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ORDER = re.compile(r"^[AP]{1,8}$")
INDEX_NAME = re.compile(r"^index_[a-z0-9_]+\.jsonl$")
PLAN_KEYS = {"label": str, "model": str, "nodes": list, "args": list, "timeout_s": (int, float)}
MATRIX_REL = Path("scripts") / "capacity" / "paper" / "matrix.py"
RUN_BIG_REL = Path("scripts") / "capacity" / "run_big.py"
CTX_DEFAULT = 4096
FACTS = ("matrix.py talks to the gateway on 127.0.0.1:8001 (matrix.py:42); start the host with --port 8001.",
         "A row that ends infeasible or refused is retried once after 60 s and leaves an extra <label>-refused "
         "record in the index (matrix.py:193 to 200).")


class HcbenchError(ValueError):
    """A refused command: bad flags, a bad plan, or an index that is not safe to write."""


# ----------------------------------------------------------------------------- plan


def _num(value: float) -> str:
    return f"{value:g}"


def row_args(opts: argparse.Namespace, pooled: bool) -> list[str]:
    """run_big flags of one row, in the order the campaign plans use (paper-2026-10-24-laptop/plan_*_row.json)."""
    args = ["--prompt-tokens", str(opts.prompt_tokens), "--output-tokens", str(opts.output_tokens),
            "--warmup", str(opts.warmup), "--reps", str(opts.reps), "--load-timeout", _num(opts.load_timeout)]
    if opts.ctx != CTX_DEFAULT:
        args += ["--ctx", str(opts.ctx)]
    if opts.max_commit_growth_gib is not None:
        args += ["--max-commit-growth-gib", _num(opts.max_commit_growth_gib)]
    if opts.concurrency != 1:
        args += ["--concurrency", str(opts.concurrency)]
    if opts.no_capacity_mode == "all" or (opts.no_capacity_mode == "pooled" and pooled):
        args += ["--no-capacity-mode"]
    if pooled and opts.seed_peers:
        args += ["--seed-peers", "--seed-timeout", _num(opts.seed_timeout)]
    if pooled and opts.layout:
        args += ["--layout", Path(opts.layout).resolve().as_posix()]
    return args + list(opts.extra_arg or [])


def build_plan(opts: argparse.Namespace) -> list[dict[str, Any]]:
    """One matrix row per letter of --order: A alone (no nodes), P pooled onto every --peer."""
    peers = list(opts.peer or [])
    counts = {"A": opts.start - 1, "P": opts.start - 1}
    rows = []
    for letter in opts.order:
        counts[letter] += 1
        pooled = letter == "P"
        group = f"{opts.host}P{len(peers)}" if pooled else f"{opts.host}A"
        rows.append({"label": f"{opts.tag}-{group}-{counts[letter]}", "model": opts.model,
                     "nodes": peers if pooled else [], "args": row_args(opts, pooled),
                     "timeout_s": opts.timeout_pooled if pooled else opts.timeout_alone})
    return rows


def check_plan_flags(opts: argparse.Namespace) -> None:
    for flag, value in (("--tag", opts.tag), ("--host", opts.host)):
        if not NAME.match(value):
            raise HcbenchError(f"{flag} {value!r}: use lowercase letters and digits only (no dashes)")
    if not MODEL_ID.match(opts.model):
        raise HcbenchError(f"--model {opts.model!r} is not a catalog id")
    if not ORDER.match(opts.order):
        raise HcbenchError("--order is A and P only, 1 to 8 letters")
    peers = opts.peer or []
    bad = [p for p in peers if not NODE_ID.match(p)]
    if bad:
        raise HcbenchError(f"--peer takes 32 lowercase hex node ids: {', '.join(bad)}")
    if len(set(peers)) != len(peers):
        raise HcbenchError("--peer lists a node twice")
    if "P" in opts.order and not peers:
        raise HcbenchError("--order has pooled rows (P) but no --peer was given")
    if opts.start < 1:
        raise HcbenchError("--start counts from 1")
    if min(opts.load_timeout, opts.timeout_alone, opts.timeout_pooled) <= 0:
        raise HcbenchError("--load-timeout, --timeout-alone and --timeout-pooled must be positive")


def layout_warnings(path: str) -> list[str]:
    """Stages refused in R7: a peer stage on an RPC device (n80), a local stage on CPU (d70)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HcbenchError(f"--layout {path}: {exc}") from exc
    stages = data.get("stages") if isinstance(data, dict) else data
    out = []
    for i, st in enumerate(stages if isinstance(stages, list) else []):
        if not isinstance(st, dict):
            continue
        node, device = str(st.get("node_id", "")), str(st.get("device", ""))
        if node != "local" and device.upper().startswith("RPC"):
            out.append(f"warning: layout stage {i} names {device} for peer {node[:8]}: use the device its worker "
                       "serves (MTL0, CUDA0); R7 refused this (index_n80_refused_layout_device.jsonl)")
        if node == "local" and device.upper() == "CPU":
            out.append(f"warning: layout stage {i} puts layers on the local CPU: the gateway refuses a host CPU "
                       "stage (index_d70_refused_layout_cpu_stage.jsonl)")
    return out


_RUN_BIG: list[ModuleType] = []


def run_big_module(repo: Path = REPO) -> ModuleType:
    """scripts/capacity/run_big.py, loaded once; loading it only adds sys.path entries and imports."""
    if not _RUN_BIG:
        spec = importlib.util.spec_from_file_location("hcbench_run_big", repo / RUN_BIG_REL)
        if spec is None or spec.loader is None:
            raise HcbenchError(f"cannot load {repo / RUN_BIG_REL}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # its dataclasses look their module up by name
        spec.loader.exec_module(module)
        _RUN_BIG.append(module)
    return _RUN_BIG[0]


def matrix_argv(row: Mapping[str, Any], home: str) -> list[str]:
    """The run_big flags matrix.py passes for one row (matrix.py:143 to 149), without the uv prefix."""
    argv = ["--model", row["model"], "--min-available-gib", "0", "--home", home]
    for nid in row.get("nodes", []):
        argv += ["--node", nid]
    if not row.get("nodes"):
        argv += ["--strategy", "local_only"]
    return argv + list(row.get("args", []))


def validate_row(row: Mapping[str, Any], home: str) -> None:
    """run_big's own parse_args on the argv matrix would build; its error message rejects the plan."""
    module = run_big_module()
    err = io.StringIO()
    saved = sys.argv[0]
    sys.argv[0] = "run_big.py"
    try:
        with contextlib.redirect_stderr(err):
            module.parse_args(matrix_argv(row, home))
    except SystemExit as exc:
        lines = [ln for ln in err.getvalue().splitlines() if "error:" in ln] or [f"exit {exc.code}"]
        raise HcbenchError(f"{row['label']}: {lines[-1].strip()}") from exc
    finally:
        sys.argv[0] = saved


def cmd_plan(opts: argparse.Namespace) -> int:
    check_plan_flags(opts)
    out = Path(opts.out)
    if out.exists() and not opts.force:
        raise HcbenchError(f"{out} exists: pass --force to overwrite it")
    warnings = layout_warnings(opts.layout) if opts.layout else []
    plan = build_plan(opts)
    home = str(Path.home() / ".hypercompute" / "iroh-session")
    for row in plan:
        validate_row(row, home)
    for w in warnings:
        print(w, file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(plan, indent=1) + "\n")
    print("| label | peers | args |\n|---|---|---|")
    for row in plan:
        print(f"| {row['label']} | {len(row['nodes'])} | {' '.join(row['args'])} |")
    print(f"wrote {out} ({len(plan)} rows)")
    return 0


# ----------------------------------------------------------------------------- run


def preset_env(name: str | None, home: str, path: str) -> dict[str, str]:
    """Environment of a known host. g14: matrix.py's own defaults (matrix.py:22 to 24). gx10: exactly what
    scripts/capacity/paper/gx10/matrix.sh:3 to 6 exports."""
    if name == "g14":
        win = PureWindowsPath(home)  # the G14 is a Windows host
        return {"HC_PAPER_REPO": str(win / ".hypercompute" / "worktrees" / "live-host"),
                "HC_PAPER_HOME": str(win / ".hypercompute" / "iroh-session"),
                "HC_PAPER_UV": str(win / ".local" / "bin" / "uv.exe")}
    if name == "gx10":
        return {"PATH": f"{home}/.local/bin:/usr/local/cuda/bin:{path}",
                "HYPERCOMPUTE_LLAMA_BIN_DIR": f"{home}/.hypercompute/src/llama.cpp/build-hypercompute/bin",
                "HC_PAPER_REPO": f"{home}/hypercompute", "HC_PAPER_HOME": f"{home}/.hypercompute",
                "HC_PAPER_UV": f"{home}/.local/bin/uv", "HC_PAPER_PROBE_MACS": "0"}
    return {}


def run_env(opts: argparse.Namespace, base: Mapping[str, str]) -> dict[str, str]:
    """The driver's environment: the caller's, then the preset, then the flags (matrix.py:20 to 26)."""
    env = {**base, **preset_env(opts.preset, str(Path.home()), base.get("PATH", ""))}
    for flag, key in (("repo", "HC_PAPER_REPO"), ("home", "HC_PAPER_HOME"), ("uv", "HC_PAPER_UV"),
                      ("llama_bin_dir", "HYPERCOMPUTE_LLAMA_BIN_DIR")):
        if getattr(opts, flag):
            env[key] = str(getattr(opts, flag))
    if opts.no_probe_macs:
        env["HC_PAPER_PROBE_MACS"] = "0"
    return env


def load_plan(path: Path) -> list[dict[str, Any]]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HcbenchError(f"cannot read plan {path}: {exc}") from exc
    if not isinstance(plan, list) or not plan:
        raise HcbenchError(f"{path}: a plan is a non-empty JSON list of rows")
    for i, row in enumerate(plan):
        bad = [k for k, t in PLAN_KEYS.items() if not isinstance(row, dict) or not isinstance(row.get(k), t)]
        if bad:
            raise HcbenchError(f"{path}: row {i} lacks or mistypes {', '.join(bad)}")
    return plan


def check_index(index: Path, append: bool) -> None:
    if not INDEX_NAME.match(index.name):
        raise HcbenchError(f"{index.name}: an index is named index_<lowercase letters, digits, _>.jsonl")
    tags = [t for t in hcbench_report.ARCHIVED if t in index.name]
    if tags:
        raise HcbenchError(f"{index.name}: names holding {tags[0]} are for archived bad rows (NUMBERS section 2)")
    if index.is_file() and index.stat().st_size > 0 and not append:
        raise HcbenchError(f"{index} already has records: pass --append to add to it")


@contextlib.contextmanager
def index_lock(index: Path) -> Iterator[Path]:
    """An exclusive <index>.lock for the whole run, so two drivers never append to one index."""
    lock = index.with_name(index.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise HcbenchError(f"{lock} exists: another run holds this index (delete the lock if it is stale)") from exc
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        yield lock
    finally:
        lock.unlink(missing_ok=True)


def wait_for(proc: subprocess.Popen) -> int:
    """Wait for matrix to exit. Ctrl+C reaches matrix and run_big in the same console; run_big stops its deployment
    on interrupt, so keep waiting for the exit code instead of leaving them behind."""
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            print("interrupt: waiting for matrix.py to stop its row", file=sys.stderr, flush=True)


def cmd_run(opts: argparse.Namespace) -> int:
    plan_path, index = Path(opts.plan).resolve(), Path(opts.index).resolve()
    plan = load_plan(plan_path)
    check_index(index, opts.append)
    env = run_env(opts, os.environ)
    repo = Path(env["HC_PAPER_REPO"]) if "HC_PAPER_REPO" in env else REPO
    matrix = repo / MATRIX_REL
    cmd = [sys.executable, "-I", str(matrix), str(plan_path), str(index)]
    if not opts.dry_run and not opts.yes:
        raise HcbenchError("a real run starts deployments on the gateway: pass --yes (or --dry-run to look first)")
    for fact in FACTS:
        print(fact)
    if opts.dry_run:
        uv = env.get("HC_PAPER_UV", str(Path.home() / ".local" / "bin" / "uv.exe"))
        home = env.get("HC_PAPER_HOME", str(Path.home() / ".hypercompute" / "iroh-session"))
        print(f"driver: {shlex.join(cmd)}")
        print(f"rows run in: {env.get('HC_PAPER_REPO', 'matrix.py default, the G14 live-host checkout')}")
        for row in plan:
            argv = [uv, "run", "--no-sync", "python", RUN_BIG_REL.as_posix(), *matrix_argv(row, home)]
            print(f"{row['label']}: {shlex.join(argv)}")
        return 0
    if not matrix.is_file():
        raise HcbenchError(f"{matrix} not found (set --repo)")
    index.parent.mkdir(parents=True, exist_ok=True)
    with index_lock(index):
        print(f"driver: {shlex.join(cmd)}", flush=True)
        return wait_for(subprocess.Popen(cmd, env=env, cwd=repo))


# ----------------------------------------------------------------------------- report and the parser


def cmd_report(opts: argparse.Namespace) -> int:
    return hcbench_report.report_main(opts.glob, opts.results_root, Path(opts.pairs), Path(opts.out), opts.check)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hcbench.py", description=(__doc__ or "").split("\n", 1)[0])
    sub = p.add_subparsers(dest="command", required=True)
    pl = sub.add_parser("plan", help="write a matrix plan (JSON rows for scripts/capacity/paper/matrix.py)")
    pl.add_argument("--tag", required=True)
    pl.add_argument("--model", required=True)
    pl.add_argument("--host", required=True, help="host name in the labels, e.g. g14 or gx")
    pl.add_argument("--peer", action="append", help="32 hex node id of a peer for the pooled rows (repeatable)")
    pl.add_argument("--order", default="AP", help="A (alone) and P (pooled), e.g. APAP; at most 8")
    pl.add_argument("--start", type=int, default=1,
                    help="first k of the labels, so a later plan for the same groups continues the numbering")
    for flag, default in (("--prompt-tokens", 128), ("--output-tokens", 64), ("--warmup", 1), ("--reps", 3),
                          ("--ctx", CTX_DEFAULT), ("--concurrency", 1)):
        pl.add_argument(flag, type=int, default=default)
    pl.add_argument("--seed-peers", action="store_true", help="pooled rows seed their peers first")
    pl.add_argument("--seed-timeout", type=float, default=3600.0)
    pl.add_argument("--layout", help="pinned layout JSON for the pooled rows")
    pl.add_argument("--no-capacity-mode", choices=("none", "pooled", "all"), default="none")
    pl.add_argument("--max-commit-growth-gib", type=float, default=None)
    pl.add_argument("--load-timeout", type=float, default=3600.0)
    pl.add_argument("--timeout-alone", type=int, default=7200)
    pl.add_argument("--timeout-pooled", type=int, default=9000)
    pl.add_argument("--extra-arg", action="append", help="one more run_big argument for every row (repeatable)")
    pl.add_argument("--out", required=True)
    pl.add_argument("--force", action="store_true", help="overwrite --out")
    pl.set_defaults(func=cmd_plan)
    r = sub.add_parser("run", help="run a plan through scripts/capacity/paper/matrix.py (needs --yes)")
    r.add_argument("plan")
    r.add_argument("--index", required=True, help="results/capacity/paper-<date>-<name>/index_<name>.jsonl")
    r.add_argument("--preset", choices=("g14", "gx10"))
    r.add_argument("--repo", help="checkout run_big runs in (HC_PAPER_REPO)")
    r.add_argument("--home", help="the gateway's HYPERCOMPUTE_HOME (HC_PAPER_HOME)")
    r.add_argument("--uv", help="uv executable (HC_PAPER_UV)")
    r.add_argument("--no-probe-macs", action="store_true", help="skip the SSH probes of the Macs")
    r.add_argument("--llama-bin-dir", help="HYPERCOMPUTE_LLAMA_BIN_DIR, for the cooldown check on a GB10")
    r.add_argument("--append", action="store_true", help="add to an index that already has records")
    r.add_argument("--dry-run", action="store_true", help="print each row's run_big command and exit")
    r.add_argument("--yes", action="store_true", help="really start the rows")
    r.set_defaults(func=cmd_run)
    rep = sub.add_parser("report", help="write bench/RESULTS.md from the index files")
    rep.add_argument("--glob", default=hcbench_report.DEFAULT_GLOB, help="index files, relative to the repo root")
    rep.add_argument("--results-root", help="where run dirs live (default $HC_PAPER_REPO or this repo)")
    rep.add_argument("--pairs", default=str(HERE / "pairs.json"))
    rep.add_argument("--out", default=str(HERE / "RESULTS.md"))
    rep.add_argument("--check", action="store_true", help="exit 1 when RESULTS.md is out of date; write nothing")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    opts = build_parser().parse_args(argv)
    try:
        return opts.func(opts)
    except HcbenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
