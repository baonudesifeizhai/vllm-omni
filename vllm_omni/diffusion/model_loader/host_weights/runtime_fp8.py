# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Loader transaction for the Phase-1 FP8 Host Weight Runtime path."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan, build_checkpoint_mmap_plan
from vllm_omni.host_weight_runtime import HostWeightLeaseCarrier, RuntimeMode

from .fp8_layout import FINAL_LAYOUT_FP8_POLICY, FinalLayoutFP8ModelPreparation
from .identity_adapter import build_final_layout_identity
from .loader_adapter import (
    build_final_layout_request,
    build_host_weight_runtime,
    build_source_digest_cache,
    prepare_final_layout_sources,
    runtime_mode,
)
from .producers.final_layout_fp8 import FinalLayoutFP8Producer
from .restorer import FinalLayoutTensorRestorer

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
    mode = runtime_mode(config)
    if mode is RuntimeMode.DISABLED:
        return False
    reason = _scope_reason(config, load_format, device)
    if reason is None:
        return True
    if mode is RuntimeMode.REQUIRED:
        raise RuntimeFP8UnavailableError(reason)
    logger.info("Host Weight Runtime preferred fallback: %s", reason)
    return False


def _validate_online_fp8(dit_modules: Sequence[tuple[str, nn.Module]]) -> None:
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


def resolve_runtime_fp8(
    model: nn.Module,
    *,
    config: object,
    load_format: str,
    device: torch.device,
    dit_modules: Sequence[tuple[str, nn.Module]],
    sources: Sequence[object],
    prepare_weights: Callable[..., tuple[Path | str, list[str], bool]],
) -> HostWeightPlan:
    """Resolve one exact artifact and leave its lease owned by DLO."""
    _validate_online_fp8(dit_modules)
    parallel = getattr(config, "parallel_config")
    checkpoint = build_checkpoint_mmap_plan(
        model,
        dit_modules=dit_modules,
        sources=sources,
        model_path=str(getattr(config, "model", "")) or None,
        tensor_parallel_size=int(getattr(parallel, "tensor_parallel_size", 1)),
        use_hsdp=bool(getattr(parallel, "use_hsdp", False)),
        online_quantization=False,
    )
    if checkpoint.plan is None:
        raise RuntimeFP8UnavailableError(checkpoint.fallback_reason or "checkpoint binding failed")

    plan = checkpoint.plan
    owned_sources = tuple(source for source in sources if getattr(source, "prefix", "") in plan.planned_source_prefixes)
    prepared_sources = prepare_final_layout_sources(owned_sources, prepare_weights=prepare_weights)
    runtime = build_host_weight_runtime(config)
    preparation = FinalLayoutFP8ModelPreparation(dit_modules)
    preparation.prepare()
    context = build_final_layout_identity(
        model,
        dit_modules=dit_modules,
        prepared_sources=prepared_sources,
        request=build_final_layout_request(config, dit_modules[0][1], load_format),
        policy=FINAL_LAYOUT_FP8_POLICY,
        source_digest_cache=build_source_digest_cache(runtime),
    )
    producer = FinalLayoutFP8Producer(
        context,
        model,
        dit_modules,
        plan,
        device=device,
    )
    resolution = runtime.resolve(context.identity, producer=producer)
    lease = resolution.lease
    if lease is None:
        raise RuntimeFP8UnavailableError(resolution.report.outcome.value)

    try:
        restore = FinalLayoutTensorRestorer(
            context,
            post_commit=preparation.activate_kernel_views,
        ).plan_restore(model, lease)
        restore.commit()
    except Exception:
        lease.close()
        raise

    logger.info(
        "Resolved runtime FP8 host weights: outcome=%s, identity=%s",
        resolution.report.outcome.value,
        context.identity.key,
    )
    return HostWeightPlan(
        backing_kind="host_weight_runtime",
        bindings={},
        planned_source_prefixes=plan.planned_source_prefixes,
        restored_tensor_names=context.tensor_names,
        lease_carrier=HostWeightLeaseCarrier(lease),
        runtime_mode=runtime_mode(config).value,
    )


__all__ = [
    "RuntimeFP8UnavailableError",
    "resolve_runtime_fp8",
    "runtime_fp8_requested",
]
