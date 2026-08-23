# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small adapters from diffusion loader state to final-layout contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    CanonicalJson,
    HostWeightRuntime,
    HostWeightRuntimeConfig,
    ProductionPolicy,
    RuntimeMode,
    StorageDomainPolicy,
    StorageScope,
)
from vllm_omni.host_weight_runtime.filesystem import detect_storage_class
from vllm_omni.host_weight_runtime.identity import canonical_json

from .contracts import (
    FinalLayoutLoaderIdentity,
    FinalLayoutParallelIdentity,
    FinalLayoutRequest,
    ImplementationIdentity,
    implementation_abi_fingerprint,
)
from .source_identity import NodeSourceDigestCache, PreparedWeightSource, WeightSourceKind

_LOADER_ABI = CanonicalJson.from_value(
    {
        "checkpoint_binding": "component-prefix-exact-v1",
        "implementation": "diffusers-pipeline-loader-final-layout-v1",
    }
)
_LOADER_IMPLEMENTATION = ImplementationIdentity(
    implementation_id="vllm-omni.diffusion.diffusers-pipeline-loader",
    version="1",
    fingerprint=implementation_abi_fingerprint(_LOADER_ABI),
)
_TRANSFORM_ABI = CanonicalJson.from_value({"implementation": "loader-owned-direct-mmap-transform-policy-v1"})
_TRANSFORM_IMPLEMENTATION = ImplementationIdentity(
    implementation_id="vllm-omni.diffusion.direct-mmap-transform",
    version="1",
    fingerprint=implementation_abi_fingerprint(_TRANSFORM_ABI),
)


def runtime_mode(config: object) -> RuntimeMode:
    return RuntimeMode(getattr(config, "host_weight_runtime_mode", RuntimeMode.DISABLED.value))


def build_host_weight_runtime(config: object) -> HostWeightRuntime:
    mode = runtime_mode(config)
    if mode is RuntimeMode.DISABLED:
        return HostWeightRuntime.from_config(HostWeightRuntimeConfig(mode=mode))
    root = Path(getattr(config, "host_weight_runtime_root")).expanduser().resolve()
    domain = StorageDomainPolicy(
        root=root,
        scope=StorageScope.NODE,
        domain_id="node",
        storage_class=detect_storage_class(root if root.exists() else root.parent),
    )
    return HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=mode,
            domain=domain,
            production=ProductionPolicy(allow_local_build=True),
        )
    )


def build_source_digest_cache(runtime: HostWeightRuntime) -> NodeSourceDigestCache:
    domain = runtime.config.domain
    assert domain is not None
    return NodeSourceDigestCache(
        domain.root,
        timeout_seconds=runtime.config.wait.coordination_timeout_seconds,
    )


def _stable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _stable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _stable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_final_layout_request(config: object, dit: nn.Module, load_format: str) -> FinalLayoutRequest:
    model_config = {
        "arch": _stable(getattr(dit, "arch", None)),
        "dtype": str(getattr(config, "dtype", "unknown")),
        "model_class": f"{type(dit).__module__}.{type(dit).__qualname__}",
        "model_config": _stable(getattr(config, "model_config", {})),
    }
    parallel = getattr(config, "parallel_config")
    return FinalLayoutRequest(
        model_id=str(getattr(config, "model")),
        loader=FinalLayoutLoaderIdentity(
            implementation=_LOADER_IMPLEMENTATION,
            model_config_fingerprint=hashlib.sha256(canonical_json(model_config)).hexdigest(),
            weight_transform_fingerprint=_TRANSFORM_IMPLEMENTATION.fingerprint,
        ),
        parallel=FinalLayoutParallelIdentity(
            tensor_parallel_size=int(getattr(parallel, "tensor_parallel_size", 1)),
            use_hsdp=bool(getattr(parallel, "use_hsdp", False)),
        ),
        load_format=load_format,
        adaptation=AdaptationIdentity(),
    )


def prepare_final_layout_sources(
    sources: Sequence[object],
    *,
    prepare_weights: Callable[..., tuple[Path | str, list[str], bool]],
) -> tuple[PreparedWeightSource, ...]:
    prepared: list[PreparedWeightSource] = []
    for source in sources:
        model_or_path = str(getattr(source, "model_or_path"))
        root, files, use_safetensors = prepare_weights(
            model_or_path,
            getattr(source, "subfolder", None),
            getattr(source, "revision", None),
            getattr(source, "fall_back_to_pt", True),
            getattr(source, "allow_patterns_overrides", None),
        )
        if not use_safetensors:
            raise ValueError("runtime FP8 production requires safetensors checkpoint sources")
        prepared.append(
            PreparedWeightSource(
                model_or_path=model_or_path,
                subfolder=getattr(source, "subfolder", None),
                requested_revision=getattr(source, "revision", None),
                prefix=str(getattr(source, "prefix", "")),
                resolved_root=Path(root).expanduser().resolve(),
                weight_files=tuple(Path(path).expanduser().resolve() for path in files),
                use_safetensors=True,
                checkpoint_adapter=_TRANSFORM_IMPLEMENTATION,
                source_kind=(
                    WeightSourceKind.LOCAL_PATH if os.path.isdir(model_or_path) else WeightSourceKind.HUGGING_FACE_HUB
                ),
            )
        )
    return tuple(prepared)


__all__ = [
    "build_final_layout_request",
    "build_host_weight_runtime",
    "build_source_digest_cache",
    "prepare_final_layout_sources",
    "runtime_mode",
]
