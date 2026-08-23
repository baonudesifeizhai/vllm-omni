# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eligibility checks for the Phase-1 FP8 Host Weight Runtime producer."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.host_weight_runtime import RuntimeMode

logger = init_logger(__name__)


class RuntimeFP8UnavailableError(RuntimeError):
    pass


def _scope_reason(config: object, load_format: str, device: torch.device | None) -> str | None:
    parallel = getattr(config, "parallel_config")
    dp = int(getattr(parallel, "data_parallel_size", 1) or 1)
    sp = int(getattr(parallel, "sequence_parallel_size", 1) or 1)
    if not getattr(config, "host_weight_runtime_root", None):
        return "host_weight_runtime_root is not configured"
    if not getattr(config, "enable_distributed_layerwise_offload", False):
        return "distributed layerwise offload is disabled"
    if not getattr(config, "dlo_use_allgather", True) or max(dp, sp) == 1:
        return "Phase 1 requires multi-rank DLO AllGather"
    if device is None or device.type != "cuda":
        return "the bounded FP8 producer requires CUDA"
    if load_format != "default":
        return "Phase 1 requires load_format='default'"
    if int(getattr(parallel, "tensor_parallel_size", 1)) != 1 or getattr(parallel, "use_hsdp", False):
        return "Phase 1 requires TP=1 without HSDP"
    if getattr(config, "lora_path", None):
        return "Phase 1 supports base weights only"
    return None


def runtime_fp8_requested(config: object, load_format: str, device: torch.device | None) -> bool:
    mode = RuntimeMode(getattr(config, "host_weight_runtime_mode", RuntimeMode.DISABLED.value))
    if mode is RuntimeMode.DISABLED:
        return False
    reason = _scope_reason(config, load_format, device)
    if reason is None:
        return True
    if mode is RuntimeMode.REQUIRED:
        raise RuntimeFP8UnavailableError(reason)
    logger.info("Host Weight Runtime preferred fallback: %s", reason)
    return False


def validate_online_fp8(dit_modules: Sequence[tuple[str, nn.Module]]) -> None:
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )

    methods = [
        getattr(module, "quant_method")
        for _, dit in dit_modules
        for module in dit.modules()
        if getattr(getattr(module, "quant_method", None), "uses_meta_device", False)
    ]
    if not methods or any(not isinstance(method, Fp8PerTensorOnlineLinearMethod) for method in methods):
        raise RuntimeFP8UnavailableError("DiT online quantization is not exclusively per-tensor FP8")


__all__ = [
    "RuntimeFP8UnavailableError",
    "runtime_fp8_requested",
    "validate_online_fp8",
]
