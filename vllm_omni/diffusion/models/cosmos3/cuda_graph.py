# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3-owned CUDA graph routine for the cached GEN forward."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.diffusion.cuda_graph import (
    DiffusionCUDAGraphCall,
    DiffusionCUDAGraphStaticInputs,
    clone_graph_value,
    copy_graph_value,
)

_CACHE_INPUT_NAMES = ("cached_kv", "cached_freqs_gen")


def _tensor_tree_refs(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return ("tensor", weakref.ref(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_tensor_tree_refs(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_tensor_tree_refs(item) for item in value))
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, _tensor_tree_refs(item)) for key, item in value.items()))
    return ("value", value)


def _matches_tensor_tree(refs: Any, value: Any) -> bool:
    kind, payload = refs
    if kind == "tensor":
        return isinstance(value, torch.Tensor) and payload() is value
    if kind in ("tuple", "list"):
        expected_type = tuple if kind == "tuple" else list
        return (
            isinstance(value, expected_type)
            and len(payload) == len(value)
            and all(_matches_tensor_tree(item_refs, item) for item_refs, item in zip(payload, value, strict=True))
        )
    if kind == "mapping":
        return (
            isinstance(value, Mapping)
            and len(payload) == len(value)
            and all(key in value and _matches_tensor_tree(item_refs, value[key]) for key, item_refs in payload)
        )
    return payload == value


class Cosmos3CachedGenCUDAGraphRoutine:
    """Own Cosmos3 cache binding and static input updates for GEN replay."""

    def __init__(self, transformer: Any, *, cache_backend: str | None) -> None:
        self.transformer = transformer
        self.cache_backend = cache_backend

    def prepare(self, graph_branch: str, **transformer_kwargs: Any) -> DiffusionCUDAGraphCall:
        cached_kv = self.transformer.cached_kv
        cached_freqs_gen = self.transformer.cached_freqs_gen
        can_capture = True
        fallback_reason = None
        if cached_kv is None or cached_freqs_gen is None:
            can_capture = False
            fallback_reason = "cosmos3_cache_not_ready"
        elif self.cache_backend not in (None, "", "none"):
            can_capture = False
            fallback_reason = f"cache_backend_{self.cache_backend}"
        elif (
            transformer_kwargs.get("action_latents") is not None and transformer_kwargs.get("action_domain_ids") is None
        ):
            can_capture = False
            fallback_reason = "action_domain_ids_not_static"

        gen_kwargs = dict(transformer_kwargs)
        gen_kwargs.pop("text_ids", None)
        gen_kwargs.pop("text_mask", None)
        gen_kwargs["cached_kv"] = cached_kv
        gen_kwargs["cached_freqs_gen"] = cached_freqs_gen
        return DiffusionCUDAGraphCall(
            kwargs=gen_kwargs,
            extra_key={
                "branch": graph_branch,
                "transformer_id": id(self.transformer),
            },
            can_capture=can_capture,
            fallback_reason=fallback_reason,
        )

    def eager(self, graph_branch: str, **transformer_kwargs: Any) -> Any:
        del graph_branch
        return self.transformer(**transformer_kwargs)

    def forward(self, **gen_kwargs: Any) -> Any:
        return self.transformer.forward_cached_gen(**gen_kwargs)

    def create_static_inputs(self, call: DiffusionCUDAGraphCall) -> DiffusionCUDAGraphStaticInputs:
        static_kwargs = {name: clone_graph_value(value) for name, value in call.kwargs.items()}
        cache_sources = {name: _tensor_tree_refs(call.kwargs[name]) for name in _CACHE_INPUT_NAMES}
        return DiffusionCUDAGraphStaticInputs(
            args=[clone_graph_value(value) for value in call.args],
            kwargs=static_kwargs,
            state={"cache_sources": cache_sources},
        )

    def copy_inputs(
        self,
        static_inputs: DiffusionCUDAGraphStaticInputs,
        call: DiffusionCUDAGraphCall,
    ) -> None:
        for static_value, dynamic_value in zip(static_inputs.args, call.args, strict=True):
            copy_graph_value(static_value, dynamic_value)

        cache_sources = static_inputs.state["cache_sources"]
        for name, dynamic_value in call.kwargs.items():
            if name in _CACHE_INPUT_NAMES:
                if _matches_tensor_tree(cache_sources[name], dynamic_value):
                    continue
                copy_graph_value(static_inputs.kwargs[name], dynamic_value)
                cache_sources[name] = _tensor_tree_refs(dynamic_value)
                continue
            copy_graph_value(static_inputs.kwargs[name], dynamic_value)


__all__ = ["Cosmos3CachedGenCUDAGraphRoutine"]
