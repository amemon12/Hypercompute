"""Python client for a Hypercompute gateway running on this machine."""

from .client import DEFAULT_URL, Pool
from .errors import (
    BodyTooLarge,
    ChatInterrupted,
    ConflictError,
    DeploymentFailed,
    EngineError,
    GatewayUnreachable,
    HypercomputeError,
    NotFoundError,
    OwnerAuthError,
    ReadyTimeout,
    RequestError,
    StreamError,
    UnavailableError,
)
from .types import ChatChunk

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_URL",
    "BodyTooLarge",
    "ChatChunk",
    "ChatInterrupted",
    "ConflictError",
    "DeploymentFailed",
    "EngineError",
    "GatewayUnreachable",
    "HypercomputeError",
    "NotFoundError",
    "OwnerAuthError",
    "Pool",
    "ReadyTimeout",
    "RequestError",
    "StreamError",
    "UnavailableError",
    "__version__",
]
