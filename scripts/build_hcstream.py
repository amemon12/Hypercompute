"""Build the hcstream proxy ggml.dll (native/hcstream) with the Zig C toolchain from PyPI.

  uv run --no-project --with ziglang python scripts/build_hcstream.py

No Visual Studio or CUDA toolkit is needed: the proxy is plain C against the pinned b11770 headers and only calls
kernel32 plus functions it resolves at runtime from the release's own DLLs. Output: native/hcstream/build/ggml.dll.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "native" / "hcstream"
OUT = SRC / "build" / "ggml.dll"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "ziglang", "cc", "-target", "x86_64-windows-gnu", "-O2", "-shared",
           "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter", "-Wno-dll-attribute-on-redeclaration",
           f"-I{SRC / 'third_party' / 'llama.cpp'}", "-o", str(OUT), str(SRC / "hcstream.c"), "-lkernel32"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout + result.stderr, file=sys.stderr)
        return result.returncode
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"built {OUT.relative_to(ROOT).as_posix()} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
