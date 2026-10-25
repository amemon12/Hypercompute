#!/usr/bin/env bash
# Install the pinned vLLM into its own venv on the Linux file system: inside a WSL2 distro (Windows hosts) or on a
# native Linux host with an NVIDIA GPU. Facts and pins: docs/research/vllm.md (vLLM v0.32.0 = 5df63a7).
#
# Usage: bash install_vllm_wsl.sh [--force]
#   --force   delete and recreate the venv before installing
# Env: HC_VLLM_ROOT overrides the install root (default: $HOME/.hypercompute-vllm; never under /mnt).
#      HC_VLLM_UV_CACHE overrides the uv cache used for this install (deleted once the packages are in the venv).
#
# Idempotent: re-running reuses uv, the venv and already installed packages. The last line it prints is
# "HC_VLLM_RESULT {json}" (versions, venv path, CUDA device capability), which `hypercompute engine install
# --engine vllm` records in ~/.hypercompute/engines/vllm/installed.json. It never serves a model; the only GPU
# use is torch.cuda.get_device_capability() and one tiny Triton kernel that proves vLLM's JIT compiles here.
set -euo pipefail

VLLM_VERSION=0.32.0     # PyPI wheel is CUDA 13.0 and pins torch==2.13.0 (release notes L21, requirements/cuda.txt:L7)
TORCH_BACKEND=cu130     # torch 2.13.0 cu130 wheels include sm_120 (pytorch build_env_setup.py:L71-84)
PYTHON_VERSION=3.12     # vLLM 0.32.0 requires-python >=3.11,<3.15; Ubuntu 24.04 ships 3.12
UV_VERSION=0.12.21      # used only when no uv with --torch-backend is present; sha256 from the GitHub release
UV_SHA256_X86_64=23f02075b652bb1df64178cfae41b5caf160822e720e2663568f3f5d63bc52c0
UV_SHA256_AARCH64=030b69227b40af8c1981b7301793dc66e71ed3c796ea8688209dd268bd91ec51
ROOT=${HC_VLLM_ROOT:-$HOME/.hypercompute-vllm}
VENV=$ROOT/venv
UV_CACHE=${HC_VLLM_UV_CACHE:-$ROOT/uv-cache}  # this install's own uv cache, deleted once the packages are in the venv

log() { printf '[hypercompute-vllm] %s\n' "$*"; }
die() {
  printf '[hypercompute-vllm] ERROR: %s\n' "$1" >&2
  [ $# -ge 2 ] && printf '[hypercompute-vllm] fix:   %s\n' "$2" >&2
  exit 1
}

FORCE=0
while [ $# -gt 0 ]; do
  case $1 in
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" "usage: bash install_vllm_wsl.sh [--force]" ;;
  esac
done

check_host() {
  [ "$(uname -s)" = Linux ] || die "vLLM runs on Linux only (found $(uname -s))" \
    "on Windows run this inside a WSL2 distro; on macOS vLLM needs the separate vllm-metal plugin"
  ARCH=$(uname -m)
  case $ARCH in
    x86_64) UV_SHA256=$UV_SHA256_X86_64 ;;
    aarch64) UV_SHA256=$UV_SHA256_AARCH64 ;;
    *) die "no vLLM CUDA wheel for $ARCH" "use an x86_64 or aarch64 Linux host" ;;
  esac
  case $ROOT in
    /mnt/*) die "install root $ROOT is on the Windows file system" \
      "keep vLLM and the model cache in the Linux file system (unset HC_VLLM_ROOT)" ;;
  esac
  WSL=false
  grep -qi microsoft /proc/version 2>/dev/null && WSL=true
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found, so no NVIDIA GPU is visible here" \
    "WSL2: install or update the NVIDIA driver on Windows and run 'wsl --update' (never install a Linux NVIDIA driver inside WSL); Linux: install the NVIDIA driver (R580 or newer for CUDA 13)"
  log "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
  # Triton and torch.compile build small C helpers with the system compiler every time vLLM starts
  command -v gcc >/dev/null 2>&1 || command -v cc >/dev/null 2>&1 || die "no C compiler (gcc) found" \
    "sudo apt-get update && sudo apt-get install -y build-essential   (vLLM's Triton kernels compile C helpers)"
}

has_headers() {  # python: Triton's C helpers include Python.h from the interpreter's include dir
  "$1" -c 'import os, sys, sysconfig; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))'
}

uv_ok() {  # grep reads all input: `grep -q` could SIGPIPE uv and fail the pipeline under pipefail
  [ -x "$1" ] && "$1" pip install --help 2>/dev/null | grep -- --torch-backend >/dev/null
}

download() {  # url dest
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -O "$2" "$1"
  else die "neither curl nor wget is installed" "sudo apt-get install -y curl"
  fi
}

find_uv() {
  UV=$HOME/.local/bin/uv
  uv_ok "$UV" && return 0
  if command -v uv >/dev/null 2>&1 && uv_ok "$(command -v uv)"; then UV=$(command -v uv); return 0; fi
  local name=uv-$ARCH-unknown-linux-gnu tmp
  tmp=$(mktemp -d)
  log "installing uv $UV_VERSION to $HOME/.local/bin"
  download "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$name.tar.gz" "$tmp/uv.tar.gz"
  echo "$UV_SHA256  $tmp/uv.tar.gz" | sha256sum -c --status - || { rm -rf "$tmp"; die "uv download failed its sha256 check" \
    "retry; if it fails again, check the network for a proxy rewriting downloads"; }
  tar -xzf "$tmp/uv.tar.gz" -C "$tmp"
  mkdir -p "$HOME/.local/bin"
  install -m 0755 "$tmp/$name/uv" "$tmp/$name/uvx" "$HOME/.local/bin/"
  rm -rf "$tmp"
  UV=$HOME/.local/bin/uv
  uv_ok "$UV" || die "uv at $UV does not support --torch-backend" "delete $UV and re-run this script"
}

make_venv() {
  if [ "$FORCE" = 1 ] && [ -d "$VENV" ]; then
    log "removing $VENV (--force)"
    rm -rf "$VENV"
  fi
  mkdir -p "$ROOT"
  if [ -x "$VENV/bin/python" ] && ! has_headers "$VENV/bin/python"; then
    # an older install used the distro's Python, which has no Python.h without python3.X-dev (needs sudo)
    log "$VENV has no Python headers, so vLLM's Triton kernels cannot compile; recreating it on uv's Python"
    rm -rf "$VENV"
  fi
  if [ ! -x "$VENV/bin/python" ]; then
    log "creating $VENV (uv-managed Python $PYTHON_VERSION, which ships the headers Triton compiles against)"
    "$UV" venv "$VENV" --python "$PYTHON_VERSION" --python-preference only-managed --seed
  fi
  has_headers "$VENV/bin/python" || die "the Python in $VENV has no Python.h" \
    "re-run with --force; if it persists, install the headers: sudo apt-get install -y python$PYTHON_VERSION-dev"
}

install_vllm() {
  log "installing vllm==$VLLM_VERSION with the $TORCH_BACKEND PyTorch backend (several GB on the first run)"
  # Not ~/.cache/uv: there the ~8 GB of wheels would outlive the venv (a --force rebuild, a deleted root) as a hidden
  # copy that the WSL virtual disk never gives back. A failed run keeps this cache, so a retry resumes.
  UV_CACHE_DIR="$UV_CACHE" "$UV" pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION" \
    --torch-backend="$TORCH_BACKEND"
  rm -rf "$UV_CACHE"
  KERNELS=ok
  log "fetching FlashInfer's precompiled kernels (vllm download-kernels; ~9 GB installed)"
  # download-kernels runs `python -m pip install`; without this pip also keeps a ~2.7 GB copy of the wheels
  if ! PATH=$VENV/bin:$PATH PIP_NO_CACHE_DIR=1 "$VENV/bin/vllm" download-kernels; then
    KERNELS=failed
    log "warning: vllm download-kernels failed; vLLM will compile kernels at startup (slow first start on Blackwell)"
  fi
}

report() {
  # A file, not stdin: Triton reads a @jit function's source from its file, so a kernel defined on stdin cannot compile
  local smoke=$ROOT/install-smoke.py rc
  cat > "$smoke" <<'PY'
import json, os, platform
import torch
import vllm

ok = torch.cuda.is_available()
cap = torch.cuda.get_device_capability(0) if ok else None
triton_error = None
if ok:  # vLLM JIT-compiles Triton kernels at startup; prove the compiler, headers and libcuda all work here
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _bump(x_ptr):
            tl.store(x_ptr, tl.load(x_ptr) + 1)

        x = torch.zeros(1, device="cuda")
        _bump[(1,)](x)
        torch.cuda.synchronize()
        if x.item() != 1:
            triton_error = f"the test kernel returned {x.item()}, expected 1"
    except Exception as exc:  # report the cause; the installer refuses to record a broken install
        triton_error = f"{type(exc).__name__}: {str(exc)[-400:]}"
result = {"vllm": vllm.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
          "cuda_available": ok, "capability": f"{cap[0]}.{cap[1]}" if cap else None,
          "device": torch.cuda.get_device_name(0) if ok else None, "python": platform.python_version(),
          "root": os.environ["HC_ROOT"], "venv": os.environ["HC_VENV"], "uv": os.environ["HC_UV"],
          "wsl": os.environ["HC_WSL"] == "true", "kernel": platform.release(), "machine": platform.machine(),
          "kernels": os.environ["HC_KERNELS"], "triton_ok": ok and triton_error is None, "triton_error": triton_error}
print(f"vllm {result['vllm']}, torch {result['torch']} (CUDA {result['torch_cuda']}), "
      f"device {result['device']}, capability {cap}")
print("HC_VLLM_RESULT " + json.dumps(result))
if ok and triton_error:
    print(f"[hypercompute-vllm] ERROR: Triton cannot compile kernels here: {triton_error}")
    raise SystemExit(4)
raise SystemExit(0 if ok else 3)
PY
  set +e
  HC_ROOT=$ROOT HC_VENV=$VENV HC_UV="$("$UV" --version)" HC_WSL=$WSL HC_KERNELS=$KERNELS "$VENV/bin/python" "$smoke"
  rc=$?
  set -e
  rm -f "$smoke"
  return $rc
}

check_host
find_uv
log "uv: $("$UV" --version) at $UV"
make_venv
install_vllm
set +e
report
rc=$?
set -e
case $rc in
  0) ;;
  4) die "vLLM's Triton kernels cannot compile in this environment (see the error above)" \
       "check gcc works ('gcc --version') and re-run with --force" ;;
  *) die "PyTorch in $VENV cannot see the GPU" \
       "WSL2: update the Windows NVIDIA driver and run 'wsl --update'; Linux: check 'nvidia-smi' and the driver version (R580+)" ;;
esac
