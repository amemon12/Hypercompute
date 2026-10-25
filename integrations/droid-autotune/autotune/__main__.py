"""python -m autotune serve | mock-gateway"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

import uvicorn

from . import config
from .gateway import GatewayClient
from .runlog import RunStore
from .runner import DroidRunner, Runner, ScriptedRunner, sweep_calls
from .service import AutotuneService
from .web import create_app


def serve(args: argparse.Namespace) -> int:
    settings = config.load()
    if args.gateway:
        settings = dataclasses.replace(settings, gateway=args.gateway)
    if args.runner:
        settings = dataclasses.replace(settings, runner=args.runner)
    ready, why = settings.droid_ready
    if settings.runner == "droid" and not ready:
        logging.warning("Droid is not ready (%s); runs will fail until it is", why)

    def factory() -> Runner:
        if settings.runner == "scripted":
            return ScriptedRunner(sweep_calls)
        return DroidRunner(settings)

    service = AutotuneService(settings, GatewayClient(settings.gateway), factory, RunStore(settings.runs_dir),
                              runner_name=settings.runner)
    app = create_app(service)
    print(f"autotune: http://127.0.0.1:{settings.port}  gateway {settings.gateway}  runner {settings.runner}",
          flush=True)
    uvicorn.run(app, host="127.0.0.1", port=settings.port, log_level="warning")
    return 0


def mock(args: argparse.Namespace) -> int:
    from mock_gateway.app import create_app as mock_app

    app = mock_app(args.host_kind, verify_s=args.verify_s, replace_delay_s=args.replace_delay_s, peers=args.peer)
    print(f"mock gateway ({args.host_kind}): http://127.0.0.1:{args.port}  controls at /mock", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per poll otherwise
    parser = argparse.ArgumentParser(prog="python -m autotune")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="watch the gateway and serve the approval page")
    s.add_argument("--gateway", help="gateway URL (loopback), default AUTOTUNE_GATEWAY or http://127.0.0.1:8000")
    s.add_argument("--runner", choices=["droid", "scripted"], help="scripted = no Droid, for rehearsing the page")
    s.set_defaults(fn=serve)
    m = sub.add_parser("mock-gateway", help="run the mock gateway (mock priors, no real hosts)")
    m.add_argument("--host-kind", choices=["g14", "gx10"], default="g14")
    m.add_argument("--port", type=int, default=8010)
    m.add_argument("--peer", action="append", default=[], help="a peer already joined at start (repeatable)")
    m.add_argument("--verify-s", type=float, default=2.0)
    m.add_argument("--replace-delay-s", type=float, default=4.0)
    m.set_defaults(fn=mock)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
