# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-overhead structured events for cross-stage runtime observability.

The event stream is intentionally observation-only: it does not expose a
control API and must never affect scheduling.  All timestamps use the host
monotonic clock, which makes events emitted by the local stage processes
directly comparable after their logs are merged.
"""

from __future__ import annotations

import itertools
import json
import os
import socket
import threading
import time
from collections.abc import Mapping
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_EVENT_PREFIX = "[OmniRuntimeEvent]"
_ENABLE_ENV = "OMNI_RUNTIME_OBSERVABILITY"
_LEGACY_ENABLE_ENV = "VLLM_OMNI_RUNTIME_OBSERVABILITY"
_STATE_ENABLE_ENV = "OMNI_RUNTIME_STATE"
_STATE_ENDPOINT_ENV = "OMNI_RUNTIME_STATE_ENDPOINT"
_event_sequence = itertools.count()
_event_lock = threading.Lock()
_transport_socket: socket.socket | None = None
_transport_pid: int | None = None
_transport_lock = threading.Lock()


def _send_to_runtime_state_collector(payload: bytes) -> None:
    """Best-effort, non-blocking delivery to the local live-state collector."""
    endpoint = os.getenv(_STATE_ENDPOINT_ENV)
    if not endpoint:
        return
    try:
        host, raw_port = endpoint.rsplit(":", 1)
        port = int(raw_port)
    except (ValueError, TypeError):
        return

    global _transport_pid
    global _transport_socket
    pid = os.getpid()
    with _transport_lock:
        if _transport_socket is None or _transport_pid != pid:
            if _transport_socket is not None:
                _transport_socket.close()
            _transport_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _transport_socket.setblocking(False)
            _transport_pid = pid
        try:
            _transport_socket.sendto(payload, (host, port))
        except (BlockingIOError, OSError):
            # The event sequence lets the collector detect a missing datagram.
            return


def runtime_observability_enabled() -> bool:
    """Return whether the structured runtime event stream is enabled."""
    value = os.getenv(_ENABLE_ENV, os.getenv(_LEGACY_ENABLE_ENV, ""))
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    state_value = os.getenv(_STATE_ENABLE_ENV, "")
    if state_value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    # A live scheduling policy cannot operate safely without the same event
    # stream.  Enabling the policy therefore implies observability.
    policy = os.getenv("OMNI_RUNTIME_POLICY", "").strip().lower()
    return policy in {"watermark", "dynamic", "deadline"}


def runtime_observability_logging_enabled() -> bool:
    """Return whether events should also be copied into the INFO log."""
    value = os.getenv(_ENABLE_ENV, os.getenv(_LEGACY_ENABLE_ENV, ""))
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_runtime_request_id(fallback: str, *metadata_sources: Any) -> str:
    """Resolve the original request ID carried through stage metadata."""
    for metadata in metadata_sources:
        if not isinstance(metadata, Mapping):
            continue
        for key in ("global_request_id", "request_id"):
            value = metadata.get(key)
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if value:
                return str(value)
    return str(fallback)


def emit_runtime_event(
    event: str,
    *,
    request_id: str,
    stage: str,
    unit: str | None = None,
    delta: int | float | None = None,
    inventory: int | float | None = None,
    **fields: Any,
) -> None:
    """Emit one process-safe JSON event.

    ``delta`` describes the change caused by this event and ``inventory`` is
    the caller's post-event local inventory.  Cross-process inventories are
    reconstructed from events; no shared mutable state is introduced here.
    """
    if not runtime_observability_enabled():
        return

    # Assigning the sequence under a lock but sending after releasing it lets
    # concurrent threads send N+1 before N.  The collector correctly treats
    # that as an invalid transport trace.  Serialize construction and send so
    # the wire order agrees with the per-process sequence order.
    with _event_lock:
        sequence = next(_event_sequence)
        payload: dict[str, Any] = {
            "event": event,
            "request_id": str(request_id),
            "stage": stage,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_ns": time.time_ns(),
            "pid": os.getpid(),
            "sequence": sequence,
        }
        if unit is not None:
            payload["unit"] = unit
        if delta is not None:
            payload["delta"] = delta
        if inventory is not None:
            payload["inventory"] = inventory
        payload.update(fields)
        rendered = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        _send_to_runtime_state_collector(rendered.encode("utf-8"))
    if runtime_observability_logging_enabled():
        logger.info("%s %s", _EVENT_PREFIX, rendered)


__all__ = [
    "emit_runtime_event",
    "resolve_runtime_request_id",
    "runtime_observability_enabled",
    "runtime_observability_logging_enabled",
]
