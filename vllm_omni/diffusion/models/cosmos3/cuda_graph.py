# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph replay for Cosmos3 cached GEN forwards.

This intentionally stays Cosmos3-specific. The cached GEN path has two
requirements that vLLM's generic LLM ``CUDAGraphWrapper`` does not own for us:

* denoising-step tensors must be copied into static CUDA graph buffers;
* ``cached_kv`` / ``cached_freqs_gen`` are stable across denoising steps and
  should only be refreshed when the UND cache changes.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.diffusion.cuda_graph import DiffusionCUDAGraphConfig

logger = init_logger(__name__)

_CACHE_STATIC_INPUT_NAMES = frozenset({"cached_kv", "cached_freqs_gen"})


@dataclass
class _CachedGenGraphEntry:
    graph: torch.cuda.CUDAGraph
    static_kwargs: dict[str, Any]
    static_output: Any
    cache_generation: Hashable | None
    hits: int = 0


class Cosmos3CUDAGraphManager:
    """Owns CUDA graphs for ``Cosmos3Transformer.forward_cached_gen``."""

    def __init__(
        self,
        forward_cached_gen: Callable[..., Any],
        config: DiffusionCUDAGraphConfig | None = None,
    ) -> None:
        self.forward_cached_gen = forward_cached_gen
        self.config = config or DiffusionCUDAGraphConfig()
        self._cache: OrderedDict[tuple[Hashable, ...], _CachedGenGraphEntry] = OrderedDict()
        self.last_call_info: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def clear(self) -> None:
        for entry in self._cache.values():
            entry.graph.reset()
        self._cache.clear()
        self.last_call_info = {"mode": "clear", "cache_size": 0}

    def run_cached_gen(
        self,
        *,
        branch: str,
        cache_generation: Hashable | None,
        **kwargs: Any,
    ) -> Any:
        key = _build_cached_gen_key(branch, kwargs)

        entry = self._cache.get(key)
        if entry is None:
            hidden_states = kwargs["hidden_states"]
            entry = self._capture(kwargs, cache_generation, hidden_states.device)
            self._cache[key] = entry
            self._evict_if_needed()
            entry.hits += 1
            self.last_call_info = {
                "mode": "graph",
                "reason": "capture",
                "cache_size": len(self._cache),
                "key_size": len(key),
            }
            return _return_output(entry.static_output, clone=self.config.clone_outputs)

        self._cache.move_to_end(key)
        self._copy_replay_inputs(entry, kwargs, cache_generation)
        entry.graph.replay()
        entry.hits += 1
        self.last_call_info = {
            "mode": "graph",
            "reason": "hit",
            "cache_size": len(self._cache),
            "key_size": len(key),
            "hits": entry.hits,
        }
        return _return_output(entry.static_output, clone=self.config.clone_outputs)

    def _capture(
        self,
        kwargs: Mapping[str, Any],
        cache_generation: Hashable | None,
        device: torch.device,
    ) -> _CachedGenGraphEntry:
        static_kwargs = {name: _clone_static(value) for name, value in kwargs.items()}

        with torch.no_grad():
            for _ in range(self.config.warmup_steps):
                self.forward_cached_gen(**static_kwargs)
                torch.accelerator.synchronize(device)
                if self.config.clear_cuda_cache_on_capture:
                    torch.accelerator.empty_cache()

        graph = torch.cuda.CUDAGraph()
        graph_pool = current_platform.get_global_graph_pool() if self.config.use_global_graph_pool else None
        with torch.no_grad():
            if graph_pool is None:
                with torch.cuda.graph(graph):
                    static_output = self.forward_cached_gen(**static_kwargs)
            else:
                with torch.cuda.graph(graph, pool=graph_pool):
                    static_output = self.forward_cached_gen(**static_kwargs)

        logger.debug("Captured Cosmos3 cached GEN CUDA graph.")
        return _CachedGenGraphEntry(
            graph=graph,
            static_kwargs=static_kwargs,
            static_output=static_output,
            cache_generation=cache_generation,
        )

    def _copy_replay_inputs(
        self,
        entry: _CachedGenGraphEntry,
        kwargs: Mapping[str, Any],
        cache_generation: Hashable | None,
    ) -> None:
        for name, value in kwargs.items():
            if name in _CACHE_STATIC_INPUT_NAMES:
                continue
            _copy_inputs(entry.static_kwargs[name], value)

        if entry.cache_generation != cache_generation:
            for name in _CACHE_STATIC_INPUT_NAMES:
                if name in kwargs:
                    _copy_inputs(entry.static_kwargs[name], kwargs[name])
            entry.cache_generation = cache_generation

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.config.max_graphs:
            _, entry = self._cache.popitem(last=False)
            entry.graph.reset()


def _build_cached_gen_key(branch: str, kwargs: Mapping[str, Any]) -> tuple[Hashable, ...]:
    key_parts: list[Hashable] = [("branch", branch)]

    for name in (
        "hidden_states",
        "timestep",
        "video_shape",
        "action_latents",
        "action_domain_ids",
        "action_noisy_mask",
        "sound_latents",
        "noisy_frame_mask",
        "control_latents",
        "cached_kv",
        "cached_freqs_gen",
    ):
        if name in kwargs:
            key_parts.append((name, _key_for_value(kwargs[name])))
    return tuple(key_parts)


def _key_for_value(value: Any) -> Hashable:
    if value is None:
        return ("none",)
    if isinstance(value, torch.Tensor):
        device = value.device
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            (device.type, device.index),
            str(value.layout),
            tuple(value.stride()),
        )
    if isinstance(value, (bool, int, float, str, bytes)):
        return (type(value).__name__, value)
    if isinstance(value, tuple):
        return ("tuple", len(value), tuple(_key_for_value(item) for item in value))
    if isinstance(value, list):
        return ("list", len(value), tuple(_key_for_value(item) for item in value))
    raise TypeError(f"unsupported key value {type(value)!r}")


def _clone_static(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_static(item) for item in value)
    if isinstance(value, list):
        return [_clone_static(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _clone_static(item) for key, item in value.items()}
    return value


def _copy_inputs(static_value: Any, dynamic_value: Any) -> None:
    if isinstance(static_value, torch.Tensor):
        static_value.copy_(dynamic_value)
        return
    if isinstance(static_value, tuple):
        for static_item, dynamic_item in zip(static_value, dynamic_value, strict=True):
            _copy_inputs(static_item, dynamic_item)
        return
    if isinstance(static_value, list):
        for static_item, dynamic_item in zip(static_value, dynamic_value, strict=True):
            _copy_inputs(static_item, dynamic_item)
        return
    if isinstance(static_value, dict):
        for key in static_value:
            _copy_inputs(static_value[key], dynamic_value[key])


def _return_output(output: Any, *, clone: bool) -> Any:
    if not clone:
        return output
    return _clone_output(output)


def _clone_output(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_output(item) for item in value)
    if isinstance(value, list):
        return [_clone_output(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _clone_output(item) for key, item in value.items()}
    return value
