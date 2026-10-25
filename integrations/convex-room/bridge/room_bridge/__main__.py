"""python -m room_bridge [--env-file PATH] run | check | open | close | clear"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from .config import BridgeConfig, ConfigError, read_env_file
from .gateway import GatewayState
from .pool import build_pool
from .room_api import RoomApiError
from .service import BridgeService

log = logging.getLogger("room_bridge")


def load_config(env_file: str | None) -> BridgeConfig | None:
    env = dict(os.environ)
    if env_file:
        env = {**read_env_file(Path(env_file)), **{k: v for k, v in env.items() if k.startswith("HC_")}}
    return BridgeConfig.from_env(env)


async def _admin(cfg: BridgeConfig, action: str) -> int:
    service, clients = BridgeService.create(cfg)
    try:
        if action == "check":
            state: GatewayState = await service.gateway.state()
            view = build_pool(state, cfg.slots)
            print(f"gateway {cfg.gateway_url}: {'reachable' if state.reachable else 'UNREACHABLE'}, "
                  f"{view.payload['deploymentState']}, model {view.payload['modelId']}, {view.slots} slot(s)")
            for m in view.payload["machines"]:
                print(f"  {m['name']:<20} {m['device']:<8} {m['layers']:>3} layers  {m['path']}")
            result = await service.api.pool(view.payload)
            print(f"convex {cfg.site_url}: room {cfg.room_code} {'open' if result.get('open') else 'closed'}")
            return 0
        print(await service.api.admin(action))
        return 0
    except RoomApiError as exc:
        print(f"convex refused: {exc}", file=sys.stderr)
        return 2
    finally:
        for c in clients:
            await c.aclose()


async def _run(cfg: BridgeConfig) -> None:
    service, clients = BridgeService.create(cfg)
    try:
        await service.run()
    finally:
        for c in clients:
            await c.aclose()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="room_bridge", description=__doc__)
    p.add_argument("--env-file", help="KEY=VALUE file (git-ignored); HC_* variables in the environment win")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("command", choices=["run", "check", "open", "close", "clear"], nargs="?", default="run")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its INFO lines would print every request URL
    try:
        cfg = load_config(args.env_file)
    except (ConfigError, OSError) as exc:
        print(f"room bridge not started: {exc}", file=sys.stderr)
        return 2
    if cfg is None:
        print("room bridge off: set HC_CONVEX_SITE_URL, HC_ROOM_SECRET and HC_ROOM_CODE to turn it on",
              file=sys.stderr)
        return 0
    if args.command != "run":
        return asyncio.run(_admin(cfg, args.command))
    try:
        asyncio.run(_run(cfg))
    except KeyboardInterrupt:
        log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
