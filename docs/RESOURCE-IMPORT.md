# Repository snapshot

This repository contains the application, SDK, launchers, website, integrations, native extensions, documentation, and benchmark resources together in one snapshot commit. Earlier development history is not imported.

Personal hostnames and user-directory components in historical benchmark artifacts use neutral identifiers. Numeric benchmark fields are unchanged. Commit IDs retained inside those artifacts identify the builds used for the recorded experiments; they are not revisions in this repository. Repository links point to the corresponding files on `main`.

Eight archived screenshots were omitted because they contained personal machine names or invitation codes. The remaining screenshots and benchmark records are retained. Historical notes still describe the original experiments, and do not imply that they were rerun during this import.

Paths in historical runbooks and experiment scripts are examples and must be configured for the current host. Use `bench/hcbench.py` and its environment overrides for new benchmark runs. Third-party license and attribution notices are retained.

## Import validation

- Python packaging, SDK, app, benchmark, host-parsing, and vision checks: 604 passed, 11 skipped. Command: `uv run python -m pytest join/tests sdk/python/tests bench/tests app/tests tests/test_fix_agent.py tests/test_vision_check.py -q`.
- Dashboard tests: 104 passed. Website calculation tests: 16 passed. Both production builds succeeded.
- Python syntax: 340 files checked. Shell syntax and LF endings: 32 scripts checked.
- All 441 parseable JSON and JSONL resources retained their numeric values.
- Text scans found no remaining removed personal identifiers; screenshot OCR identified the eight omitted images. Pattern scanning found no credential candidates beyond test fixtures and examples. These checks are bounded scans, not a guarantee that an unknown secret cannot exist.

The full backend and real-hardware suites were not rerun for this snapshot import.
