# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime speculative-depth policy for multi-stage Omni serving.

The policy is intentionally kept outside the scheduler.  The process-local
runtime-state collector owns the latest cross-stage state and answers small
localhost UDP queries from the stage-0 scheduler.  A missing, stale, or
invalid response always produces a bounded fallback depth.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import socket
import time
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

RUNTIME_POLICY_ENV = "OMNI_RUNTIME_POLICY"
RUNTIME_POLICY_FALLBACK_K_ENV = "OMNI_RUNTIME_POLICY_FALLBACK_K"
RUNTIME_POLICY_STARTUP_K_ENV = "OMNI_RUNTIME_POLICY_STARTUP_K"
RUNTIME_POLICY_CRITICAL_SLACK_MS_ENV = "OMNI_RUNTIME_POLICY_CRITICAL_SLACK_MS"
RUNTIME_POLICY_EMERGENCY_K1_SLACK_MS_ENV = "OMNI_RUNTIME_POLICY_EMERGENCY_K1_SLACK_MS"
RUNTIME_POLICY_LOW_SLACK_MS_ENV = "OMNI_RUNTIME_POLICY_LOW_SLACK_MS"
RUNTIME_POLICY_HIGH_SLACK_MS_ENV = "OMNI_RUNTIME_POLICY_HIGH_SLACK_MS"
RUNTIME_POLICY_MAX_FRESHNESS_MS_ENV = "OMNI_RUNTIME_POLICY_MAX_FRESHNESS_MS"
RUNTIME_POLICY_QUERY_TIMEOUT_MS_ENV = "OMNI_RUNTIME_POLICY_QUERY_TIMEOUT_MS"
RUNTIME_POLICY_MIN_RESIDENCY_STEPS_ENV = "OMNI_RUNTIME_POLICY_MIN_RESIDENCY_STEPS"
RUNTIME_POLICY_HYSTERESIS_MS_ENV = "OMNI_RUNTIME_POLICY_HYSTERESIS_MS"
RUNTIME_POLICY_CALL_FIXED_COST_MS_ENV = "OMNI_RUNTIME_POLICY_CALL_FIXED_COST_MS"
RUNTIME_POLICY_DRAFT_TOKEN_COST_MS_ENV = "OMNI_RUNTIME_POLICY_DRAFT_TOKEN_COST_MS"
RUNTIME_POLICY_SHARED_DOWNSTREAM_DEVICE_ENV = "OMNI_RUNTIME_POLICY_THINKER_SHARES_DOWNSTREAM_DEVICE"
RUNTIME_POLICY_PROFILE_KS_ENV = "OMNI_RUNTIME_POLICY_PROFILE_KS"
RUNTIME_POLICY_PROFILE_RESIDENCY_ENV = "OMNI_RUNTIME_POLICY_PROFILE_RESIDENCY"
RUNTIME_STATE_ENDPOINT_ENV = "OMNI_RUNTIME_STATE_ENDPOINT"

_ENABLED_POLICY_NAMES = {"watermark", "dynamic", "deadline"}
_SUPPORTED_DEPTHS = (0, 1, 3, 7)


def runtime_policy_enabled() -> bool:
    value = os.getenv(RUNTIME_POLICY_ENV, "").strip().lower()
    return value in _ENABLED_POLICY_NAMES


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.3f", name, raw, default)
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%r; using %s", name, raw, default)
    return default


def clamp_speculative_depth(value: int, maximum: int) -> int:
    """Clamp a configured depth to a supported value no larger than maximum."""
    maximum = max(0, int(maximum))
    requested = max(0, min(int(value), maximum))
    candidates = [depth for depth in _SUPPORTED_DEPTHS if depth <= maximum]
    if maximum not in candidates:
        candidates.append(maximum)
    return max(depth for depth in candidates if depth <= requested)


@dataclass(frozen=True)
class RuntimePolicyConfig:
    fallback_k: int = 1
    startup_k: int = 1
    critical_slack_ms: float = 0.0
    emergency_k1_slack_ms: float = -400.0
    low_slack_ms: float = 250.0
    high_slack_ms: float = 1000.0
    max_freshness_ms: float = 250.0
    query_timeout_ms: float = 2.0
    min_residency_steps: int = 8
    hysteresis_ms: float = 100.0
    call_fixed_cost_ms: float = 0.15
    draft_token_cost_ms: float = 0.10
    thinker_shares_downstream_device: bool = True

    @classmethod
    def from_env(cls) -> RuntimePolicyConfig:
        critical = _read_float_env(
            RUNTIME_POLICY_CRITICAL_SLACK_MS_ENV,
            0.0,
        )
        emergency_k1 = min(
            critical,
            _read_float_env(
                RUNTIME_POLICY_EMERGENCY_K1_SLACK_MS_ENV,
                -400.0,
            ),
        )
        low = max(
            critical,
            _read_float_env(RUNTIME_POLICY_LOW_SLACK_MS_ENV, 250.0),
        )
        high = max(
            low,
            _read_float_env(RUNTIME_POLICY_HIGH_SLACK_MS_ENV, 1000.0),
        )
        return cls(
            fallback_k=max(
                0,
                _read_int_env(RUNTIME_POLICY_FALLBACK_K_ENV, 1),
            ),
            startup_k=max(
                0,
                _read_int_env(RUNTIME_POLICY_STARTUP_K_ENV, 1),
            ),
            critical_slack_ms=critical,
            emergency_k1_slack_ms=emergency_k1,
            low_slack_ms=low,
            high_slack_ms=high,
            max_freshness_ms=max(
                0.0,
                _read_float_env(RUNTIME_POLICY_MAX_FRESHNESS_MS_ENV, 250.0),
            ),
            query_timeout_ms=max(
                0.1,
                _read_float_env(RUNTIME_POLICY_QUERY_TIMEOUT_MS_ENV, 2.0),
            ),
            min_residency_steps=max(
                1,
                _read_int_env(
                    RUNTIME_POLICY_MIN_RESIDENCY_STEPS_ENV,
                    8,
                ),
            ),
            hysteresis_ms=max(
                0.0,
                _read_float_env(RUNTIME_POLICY_HYSTERESIS_MS_ENV, 100.0),
            ),
            call_fixed_cost_ms=max(
                0.0,
                _read_float_env(
                    RUNTIME_POLICY_CALL_FIXED_COST_MS_ENV,
                    0.15,
                ),
            ),
            draft_token_cost_ms=max(
                0.0,
                _read_float_env(
                    RUNTIME_POLICY_DRAFT_TOKEN_COST_MS_ENV,
                    0.10,
                ),
            ),
            # Unknown placement is treated conservatively as shared. The Omni
            # engine injects an exact value into the stage-0 runtime env after
            # resolving the deployment topology.
            thinker_shares_downstream_device=_read_bool_env(
                RUNTIME_POLICY_SHARED_DOWNSTREAM_DEVICE_ENV,
                True,
            ),
        )


@dataclass
class _RequestDepthState:
    k: int
    residency_steps: int
    last_seen_ns: int


class WatermarkSpeculativePolicy:
    """Choose K from downstream inventory and playable-audio slack.

    The low-watermark behavior distinguishes two cases:

    * downstream work is already buffered: reduce Thinker work so Talker and
      Code2Wav can drain it;
    * downstream is empty: use the full draft window so Thinker releases
      another verified burst instead of starving the pipeline.
    """

    def __init__(self, config: RuntimePolicyConfig | None = None) -> None:
        self.config = config or RuntimePolicyConfig.from_env()
        self._request_depth_state: dict[str, _RequestDepthState] = {}
        self._profile_batch_index = 0
        raw_profile_ks = os.getenv(RUNTIME_POLICY_PROFILE_KS_ENV, "")
        try:
            self._profile_ks = tuple(int(item.strip()) for item in raw_profile_ks.split(",") if item.strip())
        except ValueError:
            logger.warning(
                "Invalid %s=%r; disabling K profiling sequence",
                RUNTIME_POLICY_PROFILE_KS_ENV,
                raw_profile_ks,
            )
            self._profile_ks = ()
        self._profile_residency = max(
            1,
            _read_int_env(RUNTIME_POLICY_PROFILE_RESIDENCY_ENV, 32),
        )

    def _cost_adjust_depth(
        self,
        decision: dict[str, Any],
        maximum_k: int,
    ) -> dict[str, Any]:
        """Reject a shallow K when repeated-call cost exceeds its saved work.

        Lower K shortens the current draft/verifier call, but advancing one
        maximum-K-sized window then needs more calls.  The two configurable
        profile terms make that tradeoff explicit instead of assuming K1 is
        always cheaper than K7.
        """
        if maximum_k <= 0:
            return {
                **decision,
                "raw_k": 0,
                "k": 0,
                "cost_adjusted": False,
            }
        raw_k = max(1, min(int(decision["k"]), maximum_k))
        result = {
            **decision,
            "raw_k": raw_k,
            "cost_adjusted": False,
        }
        if raw_k >= maximum_k or maximum_k <= 1:
            return result

        reason = str(decision.get("reason", ""))
        if reason.startswith("emergency_slack") or reason.startswith("fallback"):
            return result
        if reason not in {
            "startup_inventory_drain",
            "critical_slack_cost_aware",
            "low_slack_drain_downstream",
            "medium_slack",
        }:
            return result

        config = self.config
        supported = sorted(
            {depth for depth in (*_SUPPORTED_DEPTHS, maximum_k) if raw_k <= depth <= maximum_k and depth > 0}
        )
        chosen_k = maximum_k
        chosen_saved_ms = 0.0
        chosen_extra_call_ms = 0.0
        for candidate_k in supported:
            saved_ms = (maximum_k - candidate_k) * config.draft_token_cost_ms
            extra_calls = max(
                0,
                math.ceil(maximum_k / candidate_k) - 1,
            )
            extra_call_ms = extra_calls * config.call_fixed_cost_ms
            if saved_ms >= extra_call_ms or candidate_k == maximum_k:
                chosen_k = candidate_k
                chosen_saved_ms = saved_ms
                chosen_extra_call_ms = extra_call_ms
                break

        if chosen_k != raw_k:
            result["k"] = chosen_k
            result["cost_adjusted"] = True
            result["reason"] = f"cost_guard:{reason}"
        result["estimated_current_call_saved_ms"] = chosen_saved_ms
        result["estimated_extra_call_cost_ms"] = chosen_extra_call_ms
        return result

    def _hysteresis_allows(
        self,
        *,
        current_k: int,
        candidate_k: int,
        slack_ms: float | None,
    ) -> bool:
        if slack_ms is None or self.config.hysteresis_ms <= 0.0:
            return True
        margin = self.config.hysteresis_ms
        if candidate_k < current_k:
            boundary = self.config.low_slack_ms if candidate_k <= 1 else self.config.high_slack_ms
            return slack_ms <= boundary - margin
        boundary = self.config.low_slack_ms if current_k <= 1 else self.config.high_slack_ms
        return slack_ms >= boundary + margin

    def _stabilize_depth(
        self,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        request_id = str(decision["request_id"])
        candidate_k = int(decision["k"])
        original_reason = str(decision.get("reason", "unknown"))
        base_reason = str(decision.get("base_reason", original_reason))
        now_ns = time.monotonic_ns()
        state = self._request_depth_state.get(request_id)

        force_transition = (
            base_reason
            in {
                "startup_refill_downstream",
                "downstream_starvation_refill",
                "target_only",
            }
            or base_reason.startswith("critical_slack")
            or base_reason.startswith("emergency_slack")
            or base_reason.startswith("fallback")
        )
        held_by: str | None = None
        if state is None:
            state = _RequestDepthState(candidate_k, 1, now_ns)
            self._request_depth_state[request_id] = state
        elif candidate_k == state.k:
            state.residency_steps += 1
            state.last_seen_ns = now_ns
        else:
            slack = decision.get("deadline_slack_ms")
            slack_ms = None if slack is None else float(slack)
            if not force_transition and state.k >= 3 and state.residency_steps < self.config.min_residency_steps:
                held_by = "residency"
            elif not force_transition and not self._hysteresis_allows(
                current_k=state.k,
                candidate_k=candidate_k,
                slack_ms=slack_ms,
            ):
                held_by = "hysteresis"

            if held_by is None:
                state.k = candidate_k
                state.residency_steps = 1
            else:
                candidate_k = state.k
                state.residency_steps += 1
            state.last_seen_ns = now_ns

        result = {
            **decision,
            "candidate_k": int(decision["k"]),
            "k": candidate_k,
            "residency_steps": state.residency_steps,
            "held_by": held_by,
        }
        if held_by is not None:
            result["reason"] = f"{held_by}_hold:{original_reason}"
        return result

    def _prune_request_depth_state(self, active_request_ids: set[str]) -> None:
        # A request can be temporarily absent from one scheduler batch, so do
        # not delete it immediately.  Bound stale state without coupling the
        # policy to scheduler lifecycle messages.
        cutoff_ns = time.monotonic_ns() - 60_000_000_000
        stale = [
            request_id
            for request_id, state in self._request_depth_state.items()
            if request_id not in active_request_ids and state.last_seen_ns < cutoff_ns
        ]
        for request_id in stale:
            self._request_depth_state.pop(request_id, None)

    def decide_request(
        self,
        *,
        request_id: str,
        snapshot: dict[str, Any] | None,
        downstream_active: bool,
        thinker_shares_downstream_device: bool,
        maximum_k: int,
    ) -> dict[str, Any]:
        config = self.config
        if not downstream_active:
            return {
                "request_id": request_id,
                "k": maximum_k,
                "reason": "target_only",
                "ready_for_policy": True,
                "downstream_starved": False,
                "thinker_shares_downstream_device": (thinker_shares_downstream_device),
            }

        if snapshot is None:
            return {
                "request_id": request_id,
                # Until the first downstream event arrives, the only useful
                # action is to release work as quickly as possible.
                "k": maximum_k,
                "reason": "startup_refill_downstream",
                "ready_for_policy": False,
                "downstream_starved": thinker_shares_downstream_device,
                "thinker_shares_downstream_device": (thinker_shares_downstream_device),
            }

        if not bool(snapshot.get("transport_valid", False)):
            desired_k = config.fallback_k if thinker_shares_downstream_device else maximum_k
            return {
                "request_id": request_id,
                "k": clamp_speculative_depth(desired_k, maximum_k),
                "reason": "fallback_transport_invalid",
                "ready_for_policy": False,
                "downstream_starved": False,
                "thinker_shares_downstream_device": (thinker_shares_downstream_device),
            }

        freshness_ms = snapshot.get("freshness_ms")
        if freshness_ms is not None and float(freshness_ms) > config.max_freshness_ms:
            desired_k = config.fallback_k if thinker_shares_downstream_device else maximum_k
            return {
                "request_id": request_id,
                "k": clamp_speculative_depth(desired_k, maximum_k),
                "reason": "fallback_state_stale",
                "ready_for_policy": False,
                "freshness_ms": float(freshness_ms),
                "downstream_starved": False,
                "thinker_shares_downstream_device": (thinker_shares_downstream_device),
            }

        talker = snapshot.get("talker", {})
        code2wav = snapshot.get("code2wav", {})
        playout = snapshot.get("playout", {})
        fifo_credit = int(talker.get("fifo_credit_rows", 0) or 0)
        pending_frames = int(code2wav.get("pending_total_frames", 0) or 0)
        downstream_inventory = fifo_credit > 0 or pending_frames > 0
        common = {
            "freshness_ms": freshness_ms,
            "fifo_credit_rows": fifo_credit,
            "code2wav_pending_frames": pending_frames,
            "downstream_starved": (thinker_shares_downstream_device and not downstream_inventory),
            "thinker_shares_downstream_device": (thinker_shares_downstream_device),
        }

        # Reducing Thinker K only makes compute available to the downstream
        # stages when they contend for the same physical device. On a disjoint
        # device it merely slows production and increases Talker starvation.
        if not thinker_shares_downstream_device:
            return {
                "request_id": request_id,
                "k": maximum_k,
                "reason": "separate_device_keep_refilling",
                "ready_for_policy": bool(snapshot.get("ready_for_policy", False)),
                **common,
            }

        # A low-watermark controller may yield only while there is work to
        # drain. Empty Talker/Code2Wav inventory is a hard refill condition:
        # use the full draft window to release the next verified burst.
        if not downstream_inventory:
            return {
                "request_id": request_id,
                "k": maximum_k,
                "reason": "downstream_starvation_refill",
                "ready_for_policy": bool(snapshot.get("ready_for_policy", False)),
                **common,
            }

        if not bool(snapshot.get("ready_for_policy", False)):
            return {
                "request_id": request_id,
                "k": clamp_speculative_depth(config.startup_k, maximum_k),
                "reason": "startup_inventory_drain",
                "ready_for_policy": False,
                **common,
            }

        slack_ms = float(playout.get("deadline_slack_ms", 0.0) or 0.0)
        if slack_ms <= config.emergency_k1_slack_ms:
            # The current DSpark runner requires at least one speculative
            # position. Passing K=0 reaches upstream kernels that normalize by
            # the speculative length and can terminate the engine core with a
            # divide error. Reserve K1 for a near-empty playable buffer; an
            # ordinary low watermark still passes through the call-cost guard.
            desired_k = 1
            reason = "emergency_slack_min_safe_depth"
        elif slack_ms <= config.critical_slack_ms:
            desired_k = 1
            reason = "critical_slack_cost_aware"
        elif slack_ms <= config.low_slack_ms:
            desired_k = 1
            reason = "low_slack_drain_downstream"
        elif slack_ms <= config.high_slack_ms:
            desired_k = 3
            reason = "medium_slack"
        else:
            desired_k = maximum_k
            reason = "high_slack"

        return {
            "request_id": request_id,
            "k": clamp_speculative_depth(desired_k, maximum_k),
            "reason": reason,
            "ready_for_policy": True,
            "freshness_ms": freshness_ms,
            "deadline_slack_ms": slack_ms,
            "playable_buffer_ms": float(playout.get("playable_buffer_ms", 0.0) or 0.0),
            **common,
        }

    def decide_batch(
        self,
        *,
        requests: list[dict[str, Any]],
        snapshots: dict[str, dict[str, Any] | None],
        maximum_k: int,
    ) -> dict[str, Any]:
        maximum_k = max(0, int(maximum_k))
        if self._profile_ks:
            profile_slot = (self._profile_batch_index // self._profile_residency) % len(self._profile_ks)
            requested_k = self._profile_ks[profile_slot]
            profile_k = clamp_speculative_depth(requested_k, maximum_k)
            self._profile_batch_index += 1
            per_request = [
                {
                    "request_id": str(item["request_id"]),
                    "k": profile_k,
                    "raw_k": profile_k,
                    "candidate_k": profile_k,
                    "reason": f"profile_forced_k{profile_k}",
                    "ready_for_policy": True,
                    "residency_steps": ((self._profile_batch_index - 1) % self._profile_residency) + 1,
                    "held_by": None,
                    "cost_adjusted": False,
                }
                for item in requests
            ]
            return {
                "batch_k": profile_k,
                "reason": f"profile_forced_k{profile_k}",
                "maximum_k": maximum_k,
                "per_request": per_request,
            }
        raw_per_request = [
            self.decide_request(
                request_id=str(item["request_id"]),
                snapshot=snapshots.get(str(item["request_id"])),
                downstream_active=bool(item.get("downstream_active", True)),
                thinker_shares_downstream_device=bool(
                    item.get(
                        "thinker_shares_downstream_device",
                        self.config.thinker_shares_downstream_device,
                    )
                ),
                maximum_k=maximum_k,
            )
            for item in requests
        ]
        per_request = []
        for decision in raw_per_request:
            with_base_reason = {
                **decision,
                "base_reason": str(decision.get("reason", "unknown")),
            }
            cost_adjusted = self._cost_adjust_depth(
                with_base_reason,
                maximum_k,
            )
            per_request.append(self._stabilize_depth(cost_adjusted))
        self._prune_request_depth_state({str(item["request_id"]) for item in requests})
        if per_request:
            # The Omni DSpark proposer supports a ragged request-major draft
            # layout.  ``batch_k`` is therefore only the rectangular output
            # width / maximum work of any one request; it is not applied to
            # every request.  In particular, one starved stream requesting K7
            # no longer promotes unrelated K1/K3 streams to K7.
            batch_k = max(int(item["k"]) for item in per_request)
            depths = sorted({int(item["k"]) for item in per_request})
            reason = "ragged_request_depths:" + ",".join(map(str, depths))
        else:
            batch_k = clamp_speculative_depth(
                self.config.fallback_k,
                maximum_k,
            )
            reason = "fallback_empty_batch"
        return {
            "batch_k": batch_k,
            "reason": reason,
            "maximum_k": maximum_k,
            "per_request": per_request,
        }


class RuntimePolicyClient:
    """Synchronous fail-safe query client used by the stage-0 scheduler."""

    def __init__(
        self,
        *,
        endpoint: str | None,
        config: RuntimePolicyConfig | None = None,
    ) -> None:
        self.config = config or RuntimePolicyConfig.from_env()
        self._endpoint = self._parse_endpoint(endpoint)
        self._socket: socket.socket | None = None
        self._query_sequence = itertools.count()
        self._last_k_by_request: dict[str, int] = {}

    def _remember_response_depths(self, response: dict[str, Any]) -> None:
        for item in response.get("per_request", []):
            if not isinstance(item, dict) or "request_id" not in item:
                continue
            self._last_k_by_request[str(item["request_id"])] = int(item.get("k", 1))
        # Keep client-side failover state bounded for long-running servers.
        while len(self._last_k_by_request) > 8192:
            self._last_k_by_request.pop(next(iter(self._last_k_by_request)))

    @staticmethod
    def _parse_endpoint(endpoint: str | None) -> tuple[str, int] | None:
        if not endpoint:
            return None
        try:
            host, raw_port = endpoint.rsplit(":", 1)
            return host, int(raw_port)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_env(cls) -> RuntimePolicyClient:
        return cls(endpoint=os.getenv(RUNTIME_STATE_ENDPOINT_ENV))

    def _fallback(
        self,
        *,
        requests: list[dict[str, Any]],
        maximum_k: int,
        reason: str,
        query_latency_us: float,
    ) -> dict[str, Any]:
        fallback_k = clamp_speculative_depth(
            (self.config.fallback_k if self.config.thinker_shares_downstream_device else maximum_k),
            maximum_k,
        )
        per_request = [
            {
                "request_id": str(item["request_id"]),
                "k": self._last_k_by_request.get(
                    str(item["request_id"]),
                    fallback_k,
                ),
                "reason": reason,
                "ready_for_policy": False,
                "downstream_starved": False,
                "thinker_shares_downstream_device": (self.config.thinker_shares_downstream_device),
            }
            for item in requests
        ]
        return {
            "batch_k": max(
                (int(item["k"]) for item in per_request),
                default=fallback_k,
            ),
            "reason": reason,
            "maximum_k": maximum_k,
            "per_request": per_request,
            "fallback": True,
            "query_latency_us": query_latency_us,
        }

    def decide(
        self,
        *,
        requests: list[dict[str, Any]],
        maximum_k: int,
    ) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        if self._endpoint is None:
            return self._fallback(
                requests=requests,
                maximum_k=maximum_k,
                reason="fallback_collector_unavailable",
                query_latency_us=0.0,
            )

        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.connect(self._endpoint)

        query_id = next(self._query_sequence)
        query = {
            "message_type": "policy_query",
            "query_id": query_id,
            "maximum_k": int(maximum_k),
            "thinker_shares_downstream_device": (self.config.thinker_shares_downstream_device),
            "requests": requests,
        }
        deadline_ns = started_ns + int(self.config.query_timeout_ms * 1_000_000)
        try:
            self._socket.send(json.dumps(query, separators=(",", ":")).encode("utf-8"))
            while True:
                remaining_s = (deadline_ns - time.monotonic_ns()) / 1_000_000_000.0
                if remaining_s <= 0:
                    raise TimeoutError
                self._socket.settimeout(remaining_s)
                payload = self._socket.recv(65535)
                response = json.loads(payload)
                if (
                    isinstance(response, dict)
                    and response.get("message_type") == "policy_response"
                    and response.get("query_id") == query_id
                ):
                    break
            response["fallback"] = False
            response["query_latency_us"] = (time.monotonic_ns() - started_ns) / 1000.0
            self._remember_response_depths(response)
            return response
        except (OSError, TimeoutError, json.JSONDecodeError):
            return self._fallback(
                requests=requests,
                maximum_k=maximum_k,
                reason="fallback_policy_query_failed",
                query_latency_us=(time.monotonic_ns() - started_ns) / 1000.0,
            )


__all__ = [
    "RUNTIME_POLICY_ENV",
    "RUNTIME_POLICY_SHARED_DOWNSTREAM_DEVICE_ENV",
    "RuntimePolicyClient",
    "RuntimePolicyConfig",
    "WatermarkSpeculativePolicy",
    "clamp_speculative_depth",
    "runtime_policy_enabled",
]
