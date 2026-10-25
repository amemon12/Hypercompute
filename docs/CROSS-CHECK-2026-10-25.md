# Hypercompute cross-check, 2026-10-25

Source baseline: `4bf581eb59a5d1cbb5370c0dd3c49a53de4dccde`, cloned from private GitHub main. Work is on
`codex/judge-cross-check-20261025`. The existing local checkout had uncommitted changes and was left intact.
`uv sync` succeeded before the handoff was read. The GX10 checkout at `77e17f3` had identical runtime sources to
the baseline; their only difference was the handoff document.

## Test suites

The handoff's seven suites were rerun. Existing test assertions were not weakened or removed. New regression
tests exercise the failures fixed during this check.

| Suite | Command | Passed | Skipped | Deselected | Duration |
|---|---|---:|---:|---:|---:|
| Backend | `uv run python -m pytest -q` | 2152 | 5 | 8 | 498.64 s |
| Packaging | `uv run python -m pytest join/tests sdk/python/tests bench/tests app/tests -q` | 542 | 0 | 0 | 98.46 s |
| Fetch agent | In `integrations/fetch-agent`: `uv sync`, then `uv run python -m pytest -q` | 131 | 0 | 0 | 13.34 s |
| Droid autotune | In `integrations/droid-autotune`: `uv sync`, then `uv run python -m pytest -q` | 98 | 0 | 0 | 9.43 s |
| x402 proxy | In `integrations/x402-proxy`: `uv sync`, then `uv run python -m pytest -q` | 58 | 0 | 0 | 5.81 s |
| Convex bridge | In `integrations/convex-room/bridge`: `../../../.venv/Scripts/python.exe -m pytest` | 36 | 0 | 0 | 6.35 s |
| Convex functions | In `integrations/convex-room`: `npm ci`, then `npx vitest run` | 37 | 0 | 0 | 1.62 s |

Total: **3054 passed, 5 skipped, 8 deselected, zero failed** across the seven required suites.

The backend result has one fewer pass and one additional skip than the handoff. The fresh clone lacks a prebuilt
native library. The five skips are:

- `tests/test_hcpeer_build.py:522`: prebuilt cross-compiled library absent.
- `tests/test_hcpeer_build.py:715`: native test requires macOS or Linux and a C++ compiler.
- `tests/test_vllm.py:486`: worker launch/stop needs Linux, Bash and `setsid`.
- `tests/test_vllm_install.py:53`: explicitly skipped on Windows.
- `tests/test_vllm_install.py:314`: requires GNU `find` and a POSIX filesystem.

Eight hardware-marked cases were deselected by the repository's normal pytest configuration. The real-device SDK
checks below complement that portable suite; they do not turn deselected cases into passes.

Additional checks: dashboard 104 tests passed across 12 files; website 16 tests passed; both production builds and
Convex typechecking passed. The website build and tests were repeated after the copy edits. Changed SDK files pass
Ruff. Fetch-agent `uv sync --check` confirms its lockfile and environment agree.

Environment: Windows, Python 3.12.10 for the backend/packaging/bridge, Python 3.13.12 selected by the integration
projects, Node 22.22.2, npm 11.10.0, uv 0.10.6. No mainnet x402 activity was performed.

## Fixed during verification

- Fetch-agent's documented plain `uv sync` omitted pytest. Its default development dependency group now installs
  the test tools. Its stale lockfile was regenerated for the project's existing Python requirement.
- Dashboard Vitest incorrectly collected the nested website's `node:test` suite. Dashboard discovery now stays
  in its own `src`; the website has its own `npm test` command.
- `seed_and_run.py` waited for an hour after the host reported that a small model had no tensors large enough to
  seed. It now recognizes that precise empty-seed response, transfers the small tensors during loading, respects
  the current job identifier, stops on real failure, and refuses deployment after a timeout. The timeout includes
  busy-peer retries. Eighteen new SDK regressions cover these paths and cleanup.
- `--keep-running` retains a deployment only after readiness, chat and streaming succeed. It prints the SDK command
  to stop that exact deployment. `--seed-timeout=SECONDS` bounds preparation.
- The app now shows a model preloaded through the SDK, including readiness and named layer assignments. It avoids
  concurrent starts and does not claim ownership of an externally loaded deployment. It handles stop during loading,
  retains terminal failure detail, ignores tuning trials and creates its host log directory on a fresh account.
  Eleven new app regression cases cover those changes.
- The app removes its launch token from the address bar, masks invite input, stops displaying partial invitation
  codes, and reports clipboard errors accurately. Its introductory text no longer implies every pooled model is
  impossible to run on the host alone.
- Explicit Git attributes enforce LF for `.command` files and CRLF for `.cmd` files.

## Fresh clone smoke

Initial clone of the baseline into a temporary directory, with no Git hardlinks:

- All 32 tracked `.sh` and `.command` files passed Git Bash `bash -n`; none contained CR bytes.
- All six `.cmd` launchers were CRLF, with no bare LF or bare CR.
- Importing `hypercompute_sdk` directly from `sdk/python/src` exposed callable `Pool.plan`, `deploy`, `wait_ready`,
  `chat`, `stream`, and `stop`.
- A real `app/hcapp.py --port 0 --no-browser` process served the page and all six referenced assets, including the
  CSS fonts and background. Each returned HTTP 200.
- All nine implemented API routes rejected unauthenticated requests with HTTP 403; authenticated state returned
  200. The launch token was captured privately and never included in the report or test output.
- `bench/hcbench.py --help` exited zero.

The same app/SDK/assets/API smoke passed against the changed working tree. A final temporary clone of committed
source `db996b90056f95c3dab17dc5ebb0d3330a00ab87`, created with `core.autocrlf=true` and no hardlinks, then completed
its own `uv sync` in a fresh virtual environment. All checks above passed again: 32 shell files, six CRLF command
files, six SDK methods, page and six assets, nine unauthenticated API rejections, authenticated state and benchmark
help. The clone's working tree was clean. The subsequent commit only records these verification results.

## Real machines and SDK

The GX10 was reached directly from the Windows computer through the existing SSH identity. Windows OpenSSH rejected
the key's existing ACL; the installed Git SSH client worked without changing key permissions. No sudo password was
requested or used. Its gateway was already running, with mac-a, mac-b and g14 eligible and no active deployment.

| Check | Result |
|---|---|
| `quickstart.py qwen2.5-0.5b-instruct-q4_k_m` | Passed: feasible GPU plan, all three peers plus GX10, ready, chat, streamed answer, stopped |
| Original small-model `seed_and_run.py` | Reproduced the empty-seed hang; stopped before deployment |
| Fixed small-model `seed_and_run.py` | Passed: three legitimate seed skips after 15 s, ready after 6 s, chat and stream returned, all four devices, stopped |
| Mac join command | mac-a stopped sharing with cache retained, then the actual `hypercompute-join.command` ran over SSH; exit zero and sharing confirmed |
| First DeepSeek 671B rehearsal | Feasible moe_offload, seeded 180 s, gateway ready after 350.428424 s, coherent chat; old 48-token stream returned reasoning only and exposed the budget bug |
| Targeted SDK stream recheck | Passed against the same real deployment with 400 tokens: visible count from one to ten, 133 chunks |
| Live app chat | Passed: actual visible streamed answer from all four machines; capture retained for the video |
| Final corrected DeepSeek example | Passed end to end: seed 105 s, readiness poll 152 s, coherent chat, visible streamed count, four devices; retained ready with keep-running |

These are functional checks, not new benchmark comparisons. The fixed SDK example and app were copied to isolated
temporary paths on the GX10 for testing; its existing checkout and running gateway were preserved. The private
launch URL and invitation stayed out of captured output. The Mac action was a remote script invocation, not a
filmed physical double-click.

The final corrected example ended successfully at 12:04:39 UTC. Its plan is g14 3 layers, mac-b 2, mac-a 2,
GX10 54, on `moe_offload`. This differs from the earlier recorded rehearsal because the planner used the current
memory readings. The retained deployment is `5cdc693f91f747b888481aea3a265e2e`; the app and same-port SSH tunnel
are running, and the authenticated presenter tab is open. Stop this exact deployment from the GX10 when done:

```bash
PYTHONPATH=sdk/python/src uv run --no-sync python -c "from hypercompute_sdk import Pool; Pool().stop('5cdc693f91f747b888481aea3a265e2e')"
```

The raw functional run output is retained beside the local video as `deepseek-sdk-run.txt`. The video's timing
labels refer to the first rehearsal, retained as `deepseek-sdk-first-run.txt`, not this later warm-cache rerun.

## Security and privacy

Two pattern-based scanners examined current files and all reachable Git history: 1353 tracked files, 111 commits,
1768 unique blobs, 91,051,870 bytes. Gitleaks v8.30.1 was checksum-verified, ran with full redaction, and did not honor
inline ignore comments. Its 30 current and 35 historical findings were all classified: request idempotency
identifiers and synthetic test credentials. A separate 13-family scan found only test, mock and template values;
an additional Iroh-ticket scan found zero candidates. No real key, token or invitation was detected. Pattern scans
cannot prove that an arbitrary unknown secret is absent. No history rewrite was performed. The final staged-tree
scan covered 1359 files and found the same 30 classified candidates, with no new finding. The committed candidate
had 112 reachable commits, 1809 unique blobs and 91,828,834 blob bytes. The previous complete history scan plus a
zero-finding scan of the new commit covers that candidate. The final documentation-only commit contains this masked
verification record and no new source or runtime values.

The privacy scan covered 97 tracked files under `frontend/website` plus the fetch-agent README. After cleanup there
were no home paths or personal laptop-name patterns. Original path findings removed from documentation:

- `join/README.md:131`: `[home/worktree path masked]`.
- `sdk/python/README.md:136`: `[home/worktree path masked]`.
- `docs/pitch/NUMBERS.md:44-49`: `[home interpreter and temporary paths masked]`.
- `docs/demo-app/THEME.md:6`: `[home/worktree path masked]`.

Masked scanner candidates and classifications are listed in the appendix. No candidate value is reproduced.

## Documentation and copy

README, join README, SDK README, bench README, NUMBERS and E2E-CHECK were checked against the implementation.
The six requested documents had 62 resolvable relative file/image links. All literal repository-root paths in
those documents exist. Fifteen README files also passed their relative-link check. Fifteen CLI help routes,
quickstart/app/bench help and seven benchmark command examples parsed successfully.

The benchmark report initially checked 636 values with zero differences. The retired cross-session 671B pair
was then removed from active comparisons in `bench/pairs.json`, and the report regenerated. Final result: 611
quoted values checked with zero differences. The benchmark suite passed again, 72 tests in 39.53 s. Controlled
benchmark tables were preserved. Missing scratchpad-only reproduction commands were replaced
with tracked commands. Stale development-branch language, the superseded 671B decode-gain claim, and contradictory
unmeasured-status text were removed from the primary documentation. R6's slower pooled decode and R8's parity are
now described with their separate conditions.

Public website and README prose was simplified and unnecessary third-party repository references removed. Required
attribution, dependency URLs, executable identifiers, model identifiers and historical technical evidence remain.
This cleanup does not remove or conceal the software's actual dependencies. The live public website was inspected
but was not deployed or edited externally.

## Demo deliverables and remaining gates

The [judge runbook](demo/JUDGE-RUNBOOK.md) gives setup, preload commands, a timed story, exact stopping instructions
and peer/network/GX10 fallbacks. The [video shot list](demo/VIDEO-SHOTLIST.md) separates authentic capture from an
edited reconstruction and identifies the unfilmed physical Mac double-click.

The local OBS recording is **168.033333 seconds (2 minutes 48 seconds)**, H.264, 1920 by 1080 at 30 fps, 8,562,423 bytes, with no audio track. It is a captioned edit of authentic screenshots, not continuous live footage. Nine representative frames were reviewed, including the loading, chat and closing segments; no credential was visible. Full-file FFmpeg decoding exited zero with no errors. OBS recording is stopped. The physical Mac double-click remains unfilmed and is explicitly labeled.

Local artifact, relative to the parent workspace: output/hypercompute-judge-20261025/hypercompute-judge-rehearsal.mp4. SHA-256: 6107e2c361b70323d10c99bdf6de139d2597f70e698ac53636127cec73391075. Editable source, source-image hashes and frame reviews are beside the video. The recording is outside Git.
Publication and any push or merge to main require final approval. No push, merge, site deployment or video
publication was performed as part of the local verification.

## Masked scanner appendix

- `integrations/convex-room/tests/setup.ts:9`: `[MASKED]`, synthetic test credential.
- `results/20261021T020602Z-local_only-g14-cuda.json:481`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T105813Z-local_only-vllm-gw-vllm-awq.json:874`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T020708Z-all_selected-loopback-cuda+rpccpu.json:783`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T020959Z-all_selected-pre-peer-loss.json:783`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T020917Z-all_selected-pre-crash.json:783`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T023106Z-all_selected-win-cuda+wsl-linux-cpu-bench.json:697`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T022529Z-all_selected-7b-win-cuda+wsl-linux-cpu.json:705`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T021253Z-all_selected-pre-leave.json:777`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T023004Z-local_only-g14-cuda-bench.json:623`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T022339Z-all_selected-win-cuda+wsl-linux-cpu.json:697`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T021126Z-local_only-7b-g14-cuda.json:711`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T104344Z-all_selected-gw-prekill.json:1124`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T031942Z-all_selected-final-7b-cuda+wsl.json:929`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T023703Z-all_selected-pre-pause.json:921`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T031812Z-local_only-final-7b-local.json:853`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T021239Z-all_selected-7b-cuda+rpccpu-loopback.json:785`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T031348Z-all_selected-pre-hardkill-jobobject.json:969`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T031751Z-all_selected-final-0.5b-cuda+wsl.json:921`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T104215Z-all_selected-gw-0.5b-split-order.json:1124`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T104014Z-auto-gw-14b-auto.json:921`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T110149Z-local_only-vllm-gw-vllm-awq.json:899`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T032156Z-all_selected-final-7b-3machines.json:977`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T105620Z-local_only-airllm-gw-airllm-0.5b.json:809`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T104432Z-all_selected-gw-preleave.json:1124`: `[MASKED]`, request idempotency identifier, not a credential.
- `results/20261021T103904Z-auto-gw-7b-auto.json:819`: `[MASKED]`, request idempotency identifier, not a credential.
- `sdk/python/tests/test_sdk_errors.py:196`: `[MASKED]`, synthetic test credential.
- `sdk/python/tests/fake_gateway.py:15`: `[MASKED]`, synthetic test credential.
- `sdk/python/tests/test_sdk_auth.py:12`: `[MASKED]`, synthetic test credential.
- `sdk/python/tests/test_sdk_auth.py:13`: `[MASKED]`, synthetic test credential.

Additional custom-pattern reviews:
- `integrations/droid-autotune/autotune/static/index.html:111`: `[MASKED]`, token template sentinel replaced at runtime by `autotune/web.py:83`.
- `integrations/fetch-agent/mock_gateway/server.py:28`: `[MASKED]`, a mock prefix concatenated with a random UUID; not a complete credential.
