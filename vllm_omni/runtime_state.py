# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Live per-request state reconstructed from runtime observability events.

Worker processes send structured events to a localhost datagram socket and
the client process reduces them into request snapshots.  When explicitly
enabled, the same collector answers read-only policy queries from the stage-0
scheduler; all scheduling mutations remain inside that scheduler.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from vllm_omni.runtime_observability import runtime_observability_enabled
from vllm_omni.runtime_policy import (
    WatermarkSpeculativePolicy,
    runtime_policy_enabled,
)

logger = init_logger(__name__)

RUNTIME_STATE_ENDPOINT_ENV = "OMNI_RUNTIME_STATE_ENDPOINT"
DEADLINE_GUARD_MS_ENV = "OMNI_RUNTIME_DEADLINE_GUARD_MS"
RUNTIME_STATE_RCVBUF_BYTES_ENV = "OMNI_RUNTIME_STATE_RCVBUF_BYTES"
RUNTIME_EVENT_TRACE_PATH_ENV = "OMNI_RUNTIME_EVENT_TRACE_PATH"
RUNTIME_EVENT_TRACE_EVENTS_ENV = "OMNI_RUNTIME_EVENT_TRACE_EVENTS"
_KERNEL_RMEM_MAX_PATH = Path("/proc/sys/net/core/rmem_max")
_POLICY_RESPONSE_REQUEST_FIELDS = {
    "request_id",
    "k",
    "reason",
    "deadline_slack_ms",
    "playable_buffer_ms",
    "fifo_credit_rows",
    "code2wav_pending_frames",
    "raw_k",
    "candidate_k",
    "residency_steps",
    "held_by",
    "cost_adjusted",
    "estimated_current_call_saved_ms",
    "estimated_extra_call_cost_ms",
}


def _runtime_state_rcvbuf_bytes(*, fallback: int) -> int:
    raw = os.getenv(RUNTIME_STATE_RCVBUF_BYTES_ENV)
    if raw is None:
        try:
            raw = _KERNEL_RMEM_MAX_PATH.read_text().strip()
        except OSError:
            return fallback
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using %d bytes",
            RUNTIME_STATE_RCVBUF_BYTES_ENV,
            raw,
            fallback,
        )
        return fallback
    if value <= 0:
        logger.warning(
            "%s must be positive, got %d; using %d bytes",
            RUNTIME_STATE_RCVBUF_BYTES_ENV,
            value,
            fallback,
        )
        return fallback
    return value


@dataclass
class StageForwardState:
    calls: int = 0
    host_model_call_ns: int = 0
    scheduled_tokens: int = 0
    last_call_monotonic_ns: int = 0

    def update(self, event: dict[str, Any]) -> None:
        self.calls += 1
        self.host_model_call_ns += int(event.get("delta", 0) or 0)
        self.scheduled_tokens += int(event.get("scheduled_tokens", 0) or 0)
        self.last_call_monotonic_ns = max(
            self.last_call_monotonic_ns,
            int(event.get("monotonic_ns", 0) or 0),
        )

    def merge(self, other: StageForwardState) -> None:
        self.calls += other.calls
        self.host_model_call_ns += other.host_model_call_ns
        self.scheduled_tokens += other.scheduled_tokens
        self.last_call_monotonic_ns = max(
            self.last_call_monotonic_ns,
            other.last_call_monotonic_ns,
        )


@dataclass
class RequestRuntimeState:
    request_id: str
    first_event_monotonic_ns: int = 0
    last_event_monotonic_ns: int = 0

    talker_text_credit: int = 0
    talker_text_enqueued_rows: int = 0
    talker_text_consumed_rows: int = 0
    talker_text_bootstrap_rows: int = 0
    talker_wait_events: int = 0
    talker_wait_time_ms: float = 0.0
    talker_finished: bool = False

    talker_codec_produced_frames: int = 0
    code2wav_pending_total_frames: int = 0
    code2wav_pending_new_frames: int = 0
    code2wav_enqueued_new_frames: int = 0
    code2wav_scheduled_new_frames: int = 0
    code2wav_finished: bool = False

    waveform_released_samples: int = 0
    waveform_sample_rate: int = 0
    first_waveform_release_monotonic_ns: int = 0

    terminal_seen: bool = False
    terminal_arrived_samples: int = 0
    terminal_consumed_samples: int = 0
    terminal_buffer_samples: int = 0
    terminal_minimum_buffer_samples: int | None = None
    terminal_unmet_samples: int = 0
    terminal_sample_rate: int = 0
    terminal_cumulative_unmet_ms: float = 0.0

    stage_forwards: dict[str, StageForwardState] = field(default_factory=dict)
    policy_decisions: int = 0
    policy_applied_k: int | None = None
    policy_desired_k: int | None = None
    policy_raw_k: int | None = None
    policy_candidate_k: int | None = None
    policy_residency_steps: int = 0
    policy_reason: str | None = None
    policy_fallback_decisions: int = 0
    policy_cost_adjusted_decisions: int = 0
    policy_residency_holds: int = 0
    policy_hysteresis_holds: int = 0
    policy_query_latency_us: float = 0.0
    policy_k_histogram: dict[int, int] = field(default_factory=dict)

    def touch(self, event: dict[str, Any]) -> None:
        timestamp = int(event.get("monotonic_ns", 0) or 0)
        if timestamp <= 0:
            return
        if self.first_event_monotonic_ns <= 0:
            self.first_event_monotonic_ns = timestamp
        else:
            self.first_event_monotonic_ns = min(
                self.first_event_monotonic_ns,
                timestamp,
            )
        self.last_event_monotonic_ns = max(
            self.last_event_monotonic_ns,
            timestamp,
        )

    def apply(self, event: dict[str, Any]) -> None:
        self.touch(event)
        event_name = str(event.get("event", ""))

        if event_name == "talker_text_bootstrap":
            self.talker_text_bootstrap_rows += int(event.get("delta", 0) or 0)
            self.talker_text_credit = int(event.get("inventory", self.talker_text_credit) or 0)
        elif event_name == "talker_text_enqueue":
            self.talker_text_enqueued_rows += int(event.get("delta", 0) or 0)
            self.talker_text_credit = int(event.get("inventory", self.talker_text_credit) or 0)
        elif event_name == "talker_text_consume":
            self.talker_text_consumed_rows += abs(int(event.get("delta", 0) or 0))
            self.talker_text_credit = int(event.get("inventory", self.talker_text_credit) or 0)
        elif event_name == "talker_text_wait_end":
            self.talker_wait_events += 1
            self.talker_wait_time_ms += float(event.get("delta", 0.0) or 0.0) * 1000.0
        elif event_name == "talker_text_summary":
            self.talker_text_credit = int(event.get("inventory", 0) or 0)
            self.talker_text_enqueued_rows = int(event.get("enqueued_rows", 0) or 0)
            self.talker_text_consumed_rows = int(event.get("consumed_rows", 0) or 0)
            self.talker_text_bootstrap_rows = int(event.get("bootstrap_rows", 0) or 0)
            self.talker_wait_events = int(event.get("wait_events", 0) or 0)
            self.talker_wait_time_ms = float(event.get("wait_time_s", 0.0) or 0.0) * 1000.0
            self.talker_finished = bool(event.get("upstream_finished", False))
        elif event_name == "talker_codec_produce":
            self.talker_codec_produced_frames += int(event.get("delta", 0) or 0)
        elif event_name == "code2wav_codec_enqueue":
            self.code2wav_pending_total_frames = int(event.get("inventory", 0) or 0)
            self.code2wav_pending_new_frames = int(event.get("pending_new_frames", 0) or 0)
            self.code2wav_enqueued_new_frames += int(event.get("new_frames", 0) or 0)
        elif event_name == "code2wav_codec_schedule":
            self.code2wav_pending_total_frames = int(event.get("inventory", 0) or 0)
            self.code2wav_pending_new_frames = int(event.get("pending_new_frames", 0) or 0)
            self.code2wav_scheduled_new_frames += int(event.get("new_frames", 0) or 0)
        elif event_name == "code2wav_codec_summary":
            self.code2wav_pending_total_frames = int(event.get("inventory", 0) or 0)
            self.code2wav_pending_new_frames = int(event.get("pending_new_frames", 0) or 0)
            self.code2wav_enqueued_new_frames = int(event.get("enqueued_new_frames", 0) or 0)
            self.code2wav_scheduled_new_frames = int(event.get("scheduled_new_frames", 0) or 0)
            self.code2wav_finished = True
        elif event_name == "code2wav_waveform_release":
            released = int(event.get("delta", 0) or 0)
            self.waveform_released_samples += released
            self.waveform_sample_rate = int(event.get("sample_rate", self.waveform_sample_rate) or 0)
            if released > 0 and self.first_waveform_release_monotonic_ns <= 0:
                self.first_waveform_release_monotonic_ns = int(event.get("monotonic_ns", 0) or 0)
        elif event_name == "terminal_audio_arrive":
            self.terminal_seen = True
            self.terminal_arrived_samples = int(
                event.get(
                    "cumulative_arrived_samples",
                    self.terminal_arrived_samples + int(event.get("delta", 0) or 0),
                )
                or 0
            )
            self.terminal_buffer_samples = int(event.get("inventory", 0) or 0)
            if self.terminal_minimum_buffer_samples is None:
                self.terminal_minimum_buffer_samples = self.terminal_buffer_samples
            else:
                self.terminal_minimum_buffer_samples = min(
                    self.terminal_minimum_buffer_samples,
                    self.terminal_buffer_samples,
                )
            self.terminal_sample_rate = int(event.get("sample_rate", self.terminal_sample_rate) or 0)
        elif event_name == "terminal_playout_advance":
            self.terminal_seen = True
            self.terminal_consumed_samples += int(event.get("consumed_samples", 0) or 0)
            self.terminal_unmet_samples += int(event.get("unmet_samples", 0) or 0)
            self.terminal_buffer_samples = int(event.get("inventory", 0) or 0)
            if self.terminal_minimum_buffer_samples is not None:
                self.terminal_minimum_buffer_samples = min(
                    self.terminal_minimum_buffer_samples,
                    self.terminal_buffer_samples,
                )
            self.terminal_cumulative_unmet_ms = float(
                event.get("cumulative_unmet_ms", self.terminal_cumulative_unmet_ms) or 0.0
            )
        elif event_name == "terminal_playout_summary":
            self.terminal_seen = True
            self.terminal_arrived_samples = int(event.get("arrived_samples", 0) or 0)
            self.terminal_consumed_samples = int(event.get("consumed_samples", 0) or 0)
            self.terminal_unmet_samples = int(event.get("unmet_samples", 0) or 0)
            self.terminal_buffer_samples = int(event.get("inventory", 0) or 0)
            if self.terminal_minimum_buffer_samples is not None:
                self.terminal_minimum_buffer_samples = min(
                    self.terminal_minimum_buffer_samples,
                    self.terminal_buffer_samples,
                )
            self.terminal_sample_rate = int(event.get("sample_rate", self.terminal_sample_rate) or 0)
            self.terminal_cumulative_unmet_ms = float(event.get("cumulative_unmet_ms", 0.0) or 0.0)
        elif event_name in {"ar_model_forward", "code2wav_model_forward"}:
            stage = str(event.get("stage", "unknown"))
            self.stage_forwards.setdefault(stage, StageForwardState()).update(event)
        elif event_name == "speculative_depth_decision":
            applied_k = int(event.get("inventory", 0) or 0)
            self.policy_decisions += 1
            self.policy_applied_k = applied_k
            self.policy_desired_k = int(event.get("desired_k", applied_k) or 0)
            raw_k = event.get("raw_k")
            if raw_k is not None:
                self.policy_raw_k = int(raw_k)
            candidate_k = event.get("candidate_k")
            if candidate_k is not None:
                self.policy_candidate_k = int(candidate_k)
            self.policy_residency_steps = int(event.get("residency_steps", 0) or 0)
            self.policy_reason = str(event.get("policy_reason", "unknown"))
            self.policy_fallback_decisions += int(bool(event.get("fallback", False)))
            self.policy_cost_adjusted_decisions += int(bool(event.get("cost_adjusted", False)))
            held_by = event.get("held_by")
            self.policy_residency_holds += int(held_by == "residency")
            self.policy_hysteresis_holds += int(held_by == "hysteresis")
            self.policy_query_latency_us += float(event.get("query_latency_us", 0.0) or 0.0)
            self.policy_k_histogram[applied_k] = self.policy_k_histogram.get(applied_k, 0) + 1

    def merge(self, other: RequestRuntimeState) -> None:
        """Merge state observed before an internal-to-external ID mapping."""
        if other.first_event_monotonic_ns > 0:
            if self.first_event_monotonic_ns <= 0:
                self.first_event_monotonic_ns = other.first_event_monotonic_ns
            else:
                self.first_event_monotonic_ns = min(
                    self.first_event_monotonic_ns,
                    other.first_event_monotonic_ns,
                )
        self.last_event_monotonic_ns = max(
            self.last_event_monotonic_ns,
            other.last_event_monotonic_ns,
        )

        self.talker_text_credit = other.talker_text_credit or self.talker_text_credit
        self.talker_text_enqueued_rows += other.talker_text_enqueued_rows
        self.talker_text_consumed_rows += other.talker_text_consumed_rows
        self.talker_text_bootstrap_rows += other.talker_text_bootstrap_rows
        self.talker_wait_events += other.talker_wait_events
        self.talker_wait_time_ms += other.talker_wait_time_ms
        self.talker_finished = self.talker_finished or other.talker_finished

        self.talker_codec_produced_frames += other.talker_codec_produced_frames
        self.code2wav_pending_total_frames = other.code2wav_pending_total_frames or self.code2wav_pending_total_frames
        self.code2wav_pending_new_frames = other.code2wav_pending_new_frames or self.code2wav_pending_new_frames
        self.code2wav_enqueued_new_frames += other.code2wav_enqueued_new_frames
        self.code2wav_scheduled_new_frames += other.code2wav_scheduled_new_frames
        self.code2wav_finished = self.code2wav_finished or other.code2wav_finished

        self.waveform_released_samples += other.waveform_released_samples
        self.waveform_sample_rate = other.waveform_sample_rate or self.waveform_sample_rate
        if other.first_waveform_release_monotonic_ns > 0:
            if self.first_waveform_release_monotonic_ns <= 0:
                self.first_waveform_release_monotonic_ns = other.first_waveform_release_monotonic_ns
            else:
                self.first_waveform_release_monotonic_ns = min(
                    self.first_waveform_release_monotonic_ns,
                    other.first_waveform_release_monotonic_ns,
                )

        if other.terminal_seen:
            self.terminal_seen = True
            self.terminal_arrived_samples += other.terminal_arrived_samples
            self.terminal_consumed_samples += other.terminal_consumed_samples
            self.terminal_buffer_samples = other.terminal_buffer_samples
            if other.terminal_minimum_buffer_samples is not None:
                if self.terminal_minimum_buffer_samples is None:
                    self.terminal_minimum_buffer_samples = other.terminal_minimum_buffer_samples
                else:
                    self.terminal_minimum_buffer_samples = min(
                        self.terminal_minimum_buffer_samples,
                        other.terminal_minimum_buffer_samples,
                    )
            self.terminal_unmet_samples += other.terminal_unmet_samples
            self.terminal_sample_rate = other.terminal_sample_rate or self.terminal_sample_rate
            self.terminal_cumulative_unmet_ms += other.terminal_cumulative_unmet_ms

        for stage, stage_state in other.stage_forwards.items():
            self.stage_forwards.setdefault(stage, StageForwardState()).merge(stage_state)
        self.policy_decisions += other.policy_decisions
        if other.policy_applied_k is not None:
            self.policy_applied_k = other.policy_applied_k
        if other.policy_desired_k is not None:
            self.policy_desired_k = other.policy_desired_k
        if other.policy_raw_k is not None:
            self.policy_raw_k = other.policy_raw_k
        if other.policy_candidate_k is not None:
            self.policy_candidate_k = other.policy_candidate_k
        self.policy_residency_steps = max(
            self.policy_residency_steps,
            other.policy_residency_steps,
        )
        if other.policy_reason is not None:
            self.policy_reason = other.policy_reason
        self.policy_fallback_decisions += other.policy_fallback_decisions
        self.policy_cost_adjusted_decisions += other.policy_cost_adjusted_decisions
        self.policy_residency_holds += other.policy_residency_holds
        self.policy_hysteresis_holds += other.policy_hysteresis_holds
        self.policy_query_latency_us += other.policy_query_latency_us
        for depth, count in other.policy_k_histogram.items():
            self.policy_k_histogram[depth] = self.policy_k_histogram.get(depth, 0) + count

    def snapshot(
        self,
        *,
        now_monotonic_ns: int,
        deadline_guard_ms: float,
        transport_valid: bool,
        dropped_event_lower_bound: int,
    ) -> dict[str, Any]:
        sample_rate = self.terminal_sample_rate or self.waveform_sample_rate
        if self.terminal_seen and sample_rate > 0:
            playable_buffer_ms = self.terminal_buffer_samples * 1000.0 / sample_rate
            playable_buffer_source = "terminal"
        elif sample_rate > 0 and self.waveform_released_samples > 0 and self.first_waveform_release_monotonic_ns > 0:
            released_ms = self.waveform_released_samples * 1000.0 / sample_rate
            elapsed_ms = max(
                0.0,
                (now_monotonic_ns - self.first_waveform_release_monotonic_ns) / 1_000_000.0,
            )
            playable_buffer_ms = max(0.0, released_ms - elapsed_ms)
            playable_buffer_source = "server_estimate"
        else:
            playable_buffer_ms = 0.0
            playable_buffer_source = "unavailable"

        deadline_slack_ms = playable_buffer_ms - deadline_guard_ms
        minimum_playable_buffer_ms = (
            self.terminal_minimum_buffer_samples * 1000.0 / sample_rate
            if sample_rate > 0 and self.terminal_minimum_buffer_samples is not None
            else None
        )
        freshness_ms = (
            max(0.0, (now_monotonic_ns - self.last_event_monotonic_ns) / 1_000_000.0)
            if self.last_event_monotonic_ns > 0
            else None
        )
        return {
            "request_id": self.request_id,
            "updated_monotonic_ns": self.last_event_monotonic_ns,
            "freshness_ms": freshness_ms,
            "transport_valid": transport_valid,
            "dropped_event_lower_bound": dropped_event_lower_bound,
            "ready_for_policy": transport_valid and sample_rate > 0 and playable_buffer_source != "unavailable",
            "talker": {
                "fifo_credit_rows": self.talker_text_credit,
                "enqueued_rows": self.talker_text_enqueued_rows,
                "consumed_rows": self.talker_text_consumed_rows,
                "bootstrap_rows": self.talker_text_bootstrap_rows,
                "wait_events": self.talker_wait_events,
                "wait_time_ms": self.talker_wait_time_ms,
                "codec_frames_produced": self.talker_codec_produced_frames,
                "finished": self.talker_finished,
            },
            "code2wav": {
                "pending_total_frames": self.code2wav_pending_total_frames,
                "pending_new_frames": self.code2wav_pending_new_frames,
                "enqueued_new_frames": self.code2wav_enqueued_new_frames,
                "scheduled_new_frames": self.code2wav_scheduled_new_frames,
                "waveform_released_samples": self.waveform_released_samples,
                "finished": self.code2wav_finished,
            },
            "playout": {
                "sample_rate": sample_rate,
                "arrived_samples": self.terminal_arrived_samples,
                "consumed_samples": self.terminal_consumed_samples,
                "buffer_samples": self.terminal_buffer_samples,
                "unmet_samples": self.terminal_unmet_samples,
                "cumulative_unmet_ms": self.terminal_cumulative_unmet_ms,
                "playable_buffer_ms": playable_buffer_ms,
                "minimum_playable_buffer_ms": minimum_playable_buffer_ms,
                "playable_buffer_source": playable_buffer_source,
                "deadline_guard_ms": deadline_guard_ms,
                "deadline_slack_ms": deadline_slack_ms,
                "deadline_missed": self.terminal_unmet_samples > 0,
            },
            "model_forwards": {
                stage: {
                    "calls": value.calls,
                    "host_model_call_ms": value.host_model_call_ns / 1_000_000.0,
                    "scheduled_tokens": value.scheduled_tokens,
                    "last_call_monotonic_ns": value.last_call_monotonic_ns,
                }
                for stage, value in sorted(self.stage_forwards.items())
            },
            "policy": {
                "decisions": self.policy_decisions,
                "applied_k": self.policy_applied_k,
                "desired_k": self.policy_desired_k,
                "raw_k": self.policy_raw_k,
                "candidate_k": self.policy_candidate_k,
                "residency_steps": self.policy_residency_steps,
                "reason": self.policy_reason,
                "fallback_decisions": self.policy_fallback_decisions,
                "cost_adjusted_decisions": (self.policy_cost_adjusted_decisions),
                "residency_holds": self.policy_residency_holds,
                "hysteresis_holds": self.policy_hysteresis_holds,
                "mean_query_latency_us": (
                    self.policy_query_latency_us / self.policy_decisions if self.policy_decisions > 0 else 0.0
                ),
                "k_histogram": {str(depth): count for depth, count in sorted(self.policy_k_histogram.items())},
            },
        }


class RuntimeStateStore:
    """Thread-safe event reducer and snapshot registry."""

    def __init__(self, *, deadline_guard_ms: float = 0.0) -> None:
        self.deadline_guard_ms = max(0.0, deadline_guard_ms)
        self._lock = threading.RLock()
        self._requests: dict[str, RequestRuntimeState] = {}
        self._aliases: dict[str, str] = {}
        self._last_sequence_by_pid: dict[int, int] = {}
        self._transport_valid = True
        self._dropped_event_lower_bound = 0
        self._out_of_order_events = 0
        self._malformed_events = 0

    def mark_malformed_event(self) -> None:
        with self._lock:
            self._malformed_events += 1
            self._transport_valid = False

    def _record_sequence(self, event: dict[str, Any]) -> None:
        try:
            pid = int(event.get("pid", -1))
            sequence = int(event.get("sequence", -1))
        except (TypeError, ValueError):
            self._transport_valid = False
            return
        if pid < 0 or sequence < 0:
            self._transport_valid = False
            return
        last_sequence = self._last_sequence_by_pid.get(pid)
        if last_sequence is None and sequence > 0:
            self._dropped_event_lower_bound += sequence
            self._transport_valid = False
        elif last_sequence is not None:
            if sequence > last_sequence + 1:
                self._dropped_event_lower_bound += sequence - last_sequence - 1
                self._transport_valid = False
            elif sequence <= last_sequence:
                self._out_of_order_events += 1
                self._transport_valid = False
        self._last_sequence_by_pid[pid] = max(sequence, last_sequence or 0)

    def _canonical_id(self, request_id: str) -> str:
        while request_id in self._aliases and self._aliases[request_id] != request_id:
            request_id = self._aliases[request_id]
        return request_id

    def apply_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._record_sequence(event)
            raw_request_id = str(event.get("request_id", "unknown"))
            if event.get("event") == "request_id_map":
                internal_request_id = str(event.get("internal_request_id", "") or "")
                root_request_id = self._canonical_id(raw_request_id)
                if internal_request_id:
                    self._aliases[internal_request_id] = root_request_id
                    existing = self._requests.pop(internal_request_id, None)
                    if existing is not None:
                        root_state = self._requests.setdefault(
                            root_request_id,
                            RequestRuntimeState(root_request_id),
                        )
                        root_state.merge(existing)
                self._requests.setdefault(
                    root_request_id,
                    RequestRuntimeState(root_request_id),
                ).touch(event)
                return

            request_id = self._canonical_id(raw_request_id)
            state = self._requests.setdefault(
                request_id,
                RequestRuntimeState(request_id),
            )
            state.apply(event)

    def snapshot(self, request_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            now_ns = time.monotonic_ns()
            if request_id is not None:
                canonical_id = self._canonical_id(str(request_id))
                state = self._requests.get(canonical_id)
                request_snapshots = (
                    {
                        canonical_id: state.snapshot(
                            now_monotonic_ns=now_ns,
                            deadline_guard_ms=self.deadline_guard_ms,
                            transport_valid=self._transport_valid,
                            dropped_event_lower_bound=self._dropped_event_lower_bound,
                        )
                    }
                    if state is not None
                    else {}
                )
            else:
                request_snapshots = {
                    key: value.snapshot(
                        now_monotonic_ns=now_ns,
                        deadline_guard_ms=self.deadline_guard_ms,
                        transport_valid=self._transport_valid,
                        dropped_event_lower_bound=self._dropped_event_lower_bound,
                    )
                    for key, value in sorted(self._requests.items())
                }
            return {
                "transport": {
                    "valid": self._transport_valid,
                    "dropped_event_lower_bound": self._dropped_event_lower_bound,
                    "out_of_order_events": self._out_of_order_events,
                    "malformed_events": self._malformed_events,
                    "source_processes": len(self._last_sequence_by_pid),
                },
                "deadline_guard_ms": self.deadline_guard_ms,
                "requests": request_snapshots,
            }

    def snapshots_for_policy(
        self,
        request_ids: list[str],
    ) -> dict[str, dict[str, Any] | None]:
        """Return snapshots keyed by the IDs used in a scheduler query."""
        with self._lock:
            now_ns = time.monotonic_ns()
            snapshots: dict[str, dict[str, Any] | None] = {}
            for raw_request_id in request_ids:
                request_id = str(raw_request_id)
                canonical_id = self._canonical_id(request_id)
                state = self._requests.get(canonical_id)
                snapshots[request_id] = (
                    state.snapshot(
                        now_monotonic_ns=now_ns,
                        deadline_guard_ms=self.deadline_guard_ms,
                        transport_valid=self._transport_valid,
                        dropped_event_lower_bound=self._dropped_event_lower_bound,
                    )
                    if state is not None
                    else None
                )
            return snapshots


class RuntimeStateCollector:
    """Localhost event collector owned by the AsyncOmni client process."""

    def __init__(self, *, deadline_guard_ms: float) -> None:
        self.store = RuntimeStateStore(deadline_guard_ms=deadline_guard_ms)
        self.policy = WatermarkSpeculativePolicy() if runtime_policy_enabled() else None
        raw_trace_path = os.getenv(RUNTIME_EVENT_TRACE_PATH_ENV, "").strip()
        raw_trace_events = os.getenv(
            RUNTIME_EVENT_TRACE_EVENTS_ENV,
            "speculative_action_complete",
        )
        self._event_trace_events = {event.strip() for event in raw_trace_events.split(",") if event.strip()}
        self._event_trace_file = None
        self._event_trace_records_since_flush = 0
        if raw_trace_path:
            trace_path = Path(raw_trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._event_trace_file = trace_path.open(
                "a",
                encoding="utf-8",
            )
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.requested_rcvbuf_bytes = _runtime_state_rcvbuf_bytes(
            fallback=self._socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
        )
        self._socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            self.requested_rcvbuf_bytes,
        )
        self.kernel_rcvbuf_bytes = self._socket.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
        )
        self._socket.bind(("127.0.0.1", 0))
        self._socket.settimeout(0.1)
        host, port = self._socket.getsockname()
        self.endpoint = f"{host}:{port}"
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="omni-runtime-state",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload, peer = self._socket.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                break
            if self._stop.is_set():
                break
            try:
                event = json.loads(payload)
                if not isinstance(event, dict):
                    raise TypeError("runtime event is not a JSON object")
            except (json.JSONDecodeError, TypeError):
                self.store.mark_malformed_event()
                continue
            if event.get("message_type") == "policy_query":
                self._answer_policy_query(event, peer)
                continue
            if self._event_trace_file is not None and (
                not self._event_trace_events or str(event.get("event", "")) in self._event_trace_events
            ):
                try:
                    self._event_trace_file.write(
                        json.dumps(
                            event,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    self._event_trace_records_since_flush += 1
                    if self._event_trace_records_since_flush >= 256:
                        self._event_trace_file.flush()
                        self._event_trace_records_since_flush = 0
                except OSError:
                    logger.exception("[OmniRuntimeState] failed to write event trace")
                    self._event_trace_file.close()
                    self._event_trace_file = None
            self.store.apply_event(event)

    def _answer_policy_query(
        self,
        query: dict[str, Any],
        peer: tuple[str, int],
    ) -> None:
        if self.policy is None:
            return
        raw_requests = query.get("requests", [])
        if not isinstance(raw_requests, list):
            return
        shares_downstream_device = bool(
            query.get(
                "thinker_shares_downstream_device",
                self.policy.config.thinker_shares_downstream_device,
            )
        )
        requests = [
            {
                **item,
                "thinker_shares_downstream_device": (shares_downstream_device),
            }
            for item in raw_requests
            if isinstance(item, dict) and item.get("request_id") is not None
        ]
        request_ids = [str(item["request_id"]) for item in requests]
        try:
            maximum_k = max(0, int(query.get("maximum_k", 0)))
            decision = self.policy.decide_batch(
                requests=requests,
                snapshots=self.store.snapshots_for_policy(request_ids),
                maximum_k=maximum_k,
            )
            compact_per_request = [
                {key: value for key, value in item.items() if key in _POLICY_RESPONSE_REQUEST_FIELDS}
                for item in decision.get("per_request", [])
                if isinstance(item, dict)
            ]
            response = {
                "message_type": "policy_response",
                "query_id": query.get("query_id"),
                **decision,
                "per_request": compact_per_request,
            }
            self._socket.sendto(
                json.dumps(response, separators=(",", ":")).encode("utf-8"),
                peer,
            )
        except (OSError, TypeError, ValueError):
            logger.exception("[OmniRuntimePolicy] failed to answer policy query")

    def snapshot(self, request_id: str | None = None) -> dict[str, Any]:
        return self.store.snapshot(request_id)

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._socket.sendto(b"{}", self._socket.getsockname())
        except OSError:
            pass
        self._thread.join(timeout=1.0)
        self._socket.close()
        if self._event_trace_file is not None:
            self._event_trace_file.flush()
            self._event_trace_file.close()
            self._event_trace_file = None


_collector_lock = threading.Lock()
_collector: RuntimeStateCollector | None = None
_collector_users = 0


def acquire_runtime_state_collector() -> RuntimeStateCollector | None:
    """Start or retain the process-local collector when observation is enabled."""
    global _collector
    global _collector_users

    if not runtime_observability_enabled():
        return None
    with _collector_lock:
        if _collector is None:
            guard_raw = os.getenv(DEADLINE_GUARD_MS_ENV, "0")
            try:
                deadline_guard_ms = float(guard_raw)
            except ValueError:
                logger.warning(
                    "Invalid %s=%r; using 0 ms",
                    DEADLINE_GUARD_MS_ENV,
                    guard_raw,
                )
                deadline_guard_ms = 0.0
            _collector = RuntimeStateCollector(deadline_guard_ms=deadline_guard_ms)
            os.environ[RUNTIME_STATE_ENDPOINT_ENV] = _collector.endpoint
            logger.info(
                "[OmniRuntimeState] collector=%s deadline_guard_ms=%.3f "
                "requested_rcvbuf_bytes=%d kernel_rcvbuf_bytes=%d",
                _collector.endpoint,
                deadline_guard_ms,
                _collector.requested_rcvbuf_bytes,
                _collector.kernel_rcvbuf_bytes,
            )
        _collector_users += 1
        return _collector


def release_runtime_state_collector(
    collector: RuntimeStateCollector | None,
) -> None:
    """Release a collector acquired by an OmniBase instance."""
    global _collector
    global _collector_users

    if collector is None:
        return
    with _collector_lock:
        if collector is not _collector:
            return
        _collector_users = max(0, _collector_users - 1)
        if _collector_users > 0:
            return
        endpoint = _collector.endpoint
        _collector.stop()
        _collector = None
        if os.getenv(RUNTIME_STATE_ENDPOINT_ENV) == endpoint:
            os.environ.pop(RUNTIME_STATE_ENDPOINT_ENV, None)


__all__ = [
    "DEADLINE_GUARD_MS_ENV",
    "RUNTIME_STATE_ENDPOINT_ENV",
    "RUNTIME_STATE_RCVBUF_BYTES_ENV",
    "RequestRuntimeState",
    "RuntimeStateCollector",
    "RuntimeStateStore",
    "acquire_runtime_state_collector",
    "release_runtime_state_collector",
]
