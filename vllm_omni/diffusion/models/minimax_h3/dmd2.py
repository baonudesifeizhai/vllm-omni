# SPDX-License-Identifier: Apache-2.0
"""Inference contract for MiniMax H3 DMD2 student checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _validate_sigma_schedule(name: str, values: Any, num_inference_steps: int) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"MiniMax H3 DMD2 {name} must be a list")
    sigmas = tuple(float(value) for value in values)
    expected = num_inference_steps + 1
    if len(sigmas) != expected:
        raise ValueError(
            f"MiniMax H3 DMD2 {name} must contain {expected} values "
            f"for {num_inference_steps} inference steps, got {len(sigmas)}"
        )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in sigmas):
        raise ValueError(f"MiniMax H3 DMD2 {name} values must be finite and in [0, 1]")
    if not math.isclose(sigmas[0], 1.0, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(f"MiniMax H3 DMD2 {name} must start at 1.0")
    if not math.isclose(sigmas[-1], 0.0, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(f"MiniMax H3 DMD2 {name} must end at 0.0")
    if any(current <= following for current, following in zip(sigmas, sigmas[1:])):
        raise ValueError(f"MiniMax H3 DMD2 {name} must be strictly decreasing")
    return sigmas


@dataclass(frozen=True)
class MiniMaxH3DMD2Config:
    """Exact few-step schedule exported with an H3 DMD2 student.

    H3 denoises video and audio jointly but uses a separate time coordinate for
    each modality.  Consequently, the generic DMD2 integer-timestep contract
    cannot represent an H3 checkpoint faithfully.
    """

    num_inference_steps: int
    video_sigmas: tuple[float, ...]
    audio_sigmas: tuple[float, ...]
    solver: str = "ode"

    @classmethod
    def from_model_index(cls, model_index: Mapping[str, Any]) -> MiniMaxH3DMD2Config | None:
        block = model_index.get("dmd2_config")
        if block is None:
            return None
        if not isinstance(block, Mapping):
            raise ValueError("MiniMax H3 dmd2_config must be an object")

        method = str(block.get("method", "dmd2")).strip().lower()
        if method != "dmd2":
            raise ValueError(f"MiniMax H3 dmd2_config.method must be 'dmd2', got {method!r}")
        solver = str(block.get("solver", "ode")).strip().lower()
        if solver != "ode":
            raise ValueError("MiniMax H3 DMD2 currently supports only the deterministic ODE solver")

        raw_steps = block.get("num_inference_steps")
        if isinstance(raw_steps, bool) or not isinstance(raw_steps, int) or raw_steps <= 0:
            raise ValueError("MiniMax H3 DMD2 num_inference_steps must be a positive integer")
        video_sigmas = _validate_sigma_schedule("video_sigmas", block.get("video_sigmas"), raw_steps)
        audio_sigmas = _validate_sigma_schedule("audio_sigmas", block.get("audio_sigmas"), raw_steps)
        return cls(
            num_inference_steps=raw_steps,
            video_sigmas=video_sigmas,
            audio_sigmas=audio_sigmas,
            solver=solver,
        )


__all__ = ["MiniMaxH3DMD2Config"]
