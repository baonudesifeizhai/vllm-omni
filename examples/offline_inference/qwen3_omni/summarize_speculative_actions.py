#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize measured speculative actions and optional downstream state."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return ordered[index]


def _single_positive_depth(histogram: dict[str, Any]) -> int | None:
    depths = [int(depth) for depth, count in histogram.items() if int(count or 0) > 0 and int(depth) > 0]
    return depths[0] if len(depths) == 1 else None


def _summarize_group(actions: list[dict[str, Any]]) -> dict[str, Any]:
    call_ms = [float(action.get("action_wall_ms", 0.0)) for action in actions]
    drafted = sum(int(action.get("drafted_tokens", 0) or 0) for action in actions)
    accepted = sum(int(action.get("accepted_draft_tokens", 0) or 0) for action in actions)
    released = sum(int(action.get("released_tokens", 0) or 0) for action in actions)
    request_calls = sum(int(action.get("decode_requests", 0) or 0) for action in actions)
    return {
        "calls": len(actions),
        "request_calls": request_calls,
        "batch_size_p50": _percentile(
            [float(action.get("batch_size", 0) or 0) for action in actions],
            0.5,
        ),
        "call_wall_ms_p50": _percentile(call_ms, 0.5),
        "call_wall_ms_p95": _percentile(call_ms, 0.95),
        "draft_accept_rate": accepted / drafted if drafted else None,
        "accepted_draft_len": accepted / request_calls if request_calls else None,
        "released_len": released / request_calls if request_calls else None,
        "batch_wall_ms_per_released_token": (sum(call_ms) / released if released else None),
        "drafted_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "released_tokens": released,
    }


def _summarize_largest_pure_decode_batch(
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Summarize calls at the largest observed all-decode batch size.

    Tail batches become smaller as requests finish, while prefill/decode
    mixtures include unrelated multimodal work.  Keeping this slice separate
    makes K-to-K call-cost comparisons use the same batch shape.
    """
    pure_decode = [
        action
        for action in actions
        if int(action.get("decode_requests", 0) or 0) > 0
        and int(action.get("decode_requests", 0) or 0) == int(action.get("batch_size", 0) or 0)
    ]
    if not pure_decode:
        return None
    batch_size = max(int(action.get("batch_size", 0) or 0) for action in pure_decode)
    matched = [action for action in pure_decode if int(action.get("batch_size", 0) or 0) == batch_size]
    return {
        "batch_size": batch_size,
        **_summarize_group(matched),
    }


def summarize_actions(actions: list[dict[str, Any]]) -> dict[str, Any]:
    decode_actions = [
        action
        for action in actions
        if action.get("event") == "speculative_action_complete" and int(action.get("decode_requests", 0) or 0) > 0
    ]
    steady_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    transitions: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for action in decode_actions:
        verify_k = _single_positive_depth(action.get("verify_k_histogram", {}))
        proposal_k = _single_positive_depth(action.get("proposal_k_histogram", {}))
        if verify_k is None or proposal_k is None:
            continue
        if verify_k == proposal_k and int(action.get("transition_requests", 0) or 0) == 0:
            steady_by_k[verify_k].append(action)
        elif verify_k != proposal_k:
            transitions[(verify_k, proposal_k)].append(action)

    steady_medians_by_k_batch: dict[tuple[int, int], float] = {}
    steady_medians_by_k: dict[int, float] = {}
    for depth, group in steady_by_k.items():
        steady_medians_by_k[depth] = statistics.median(float(action["action_wall_ms"]) for action in group)
        by_batch: dict[int, list[float]] = defaultdict(list)
        for action in group:
            by_batch[int(action.get("batch_size", 0) or 0)].append(float(action["action_wall_ms"]))
        for batch_size, durations in by_batch.items():
            steady_medians_by_k_batch[(depth, batch_size)] = statistics.median(durations)

    transition_summary: dict[str, Any] = {}
    for (verify_k, proposal_k), group in sorted(transitions.items()):
        deltas: list[float] = []
        for action in group:
            batch_size = int(action.get("batch_size", 0) or 0)
            baseline = steady_medians_by_k_batch.get(
                (verify_k, batch_size),
                steady_medians_by_k.get(verify_k),
            )
            if baseline is not None:
                deltas.append(float(action["action_wall_ms"]) - baseline)
        transition_summary[f"K{verify_k}->K{proposal_k}"] = {
            **_summarize_group(group),
            "matched_switch_delta_ms_p50": _percentile(deltas, 0.5),
            "matched_switch_delta_ms_p95": _percentile(deltas, 0.95),
        }

    return {
        "recorded_actions": len(actions),
        "decode_actions": len(decode_actions),
        "steady": {
            f"K{depth}": {
                **_summarize_group(group),
                "largest_pure_decode_batch": (_summarize_largest_pure_decode_batch(group)),
            }
            for depth, group in sorted(steady_by_k.items())
        },
        "transitions": transition_summary,
    }


def _summarize_stage_group(actions: list[dict[str, Any]]) -> dict[str, Any]:
    call_ms = [float(action.get("action_wall_ms", 0.0)) for action in actions]
    return {
        "calls": len(actions),
        "call_wall_ms_mean": (sum(call_ms) / len(call_ms) if call_ms else None),
        "call_wall_ms_p50": _percentile(call_ms, 0.5),
        "call_wall_ms_p95": _percentile(call_ms, 0.95),
        "scheduled_tokens": sum(int(action.get("scheduled_tokens", 0) or 0) for action in actions),
    }


def summarize_stage_actions(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize stage scheduler round trips, preserving batch shape."""
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        if action.get("event") != "omni_stage_action_complete":
            continue
        by_stage[str(action.get("stage", "unknown"))].append(action)

    output: dict[str, Any] = {}
    for stage, group in sorted(by_stage.items()):
        by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for action in group:
            by_batch[int(action.get("batch_size", 0) or 0)].append(action)
        output[stage] = {
            "all": _summarize_stage_group(group),
            "by_batch_size": {
                str(batch_size): _summarize_stage_group(batch_group)
                for batch_size, batch_group in sorted(by_batch.items())
            },
        }
    return output


def summarize_downstream(state: dict[str, Any]) -> dict[str, Any]:
    requests = list(state.get("requests", {}).values())

    def values(path: tuple[str, ...]) -> list[float]:
        output: list[float] = []
        for request in requests:
            current: Any = request
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if current is not None:
                output.append(float(current))
        return output

    talker_wait = values(("talker", "wait_time_ms"))
    talker_forward = values(("model_forwards", "talker", "host_model_call_ms"))
    code2wav_forward = values(("model_forwards", "code2wav", "host_model_call_ms"))
    unmet = values(("playout", "cumulative_unmet_ms"))
    return {
        "requests": len(requests),
        "talker_upstream_wait_ms_p50": _percentile(talker_wait, 0.5),
        "talker_upstream_wait_ms_p95": _percentile(talker_wait, 0.95),
        "talker_host_forward_ms_per_request_p50": _percentile(
            talker_forward,
            0.5,
        ),
        "code2wav_host_forward_ms_per_request_p50": _percentile(
            code2wav_forward,
            0.5,
        ),
        "playout_unmet_ms_p50": _percentile(unmet, 0.5),
        "playout_unmet_ms_p95": _percentile(unmet, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("actions", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    actions = [json.loads(line) for line in args.actions.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {
        "actions": summarize_actions(actions),
        "stages": summarize_stage_actions(actions),
    }
    if args.state is not None:
        summary["downstream"] = summarize_downstream(json.loads(args.state.read_text(encoding="utf-8")))
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
