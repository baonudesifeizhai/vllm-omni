# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph helpers for diffusion transformer forwards.

The runner is intentionally model-agnostic: it captures a callable for a
specific graph key, owns static input buffers for that key, and replays the
captured graph when later calls have compatible inputs. Model-specific code is
expected to decide whether a call is graph-safe and to pass any extra key parts
that are not visible from the input structure.
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, TypeAlias

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

GraphKey: TypeAlias = tuple[tuple[str, Hashable], ...]


@dataclass
class DiffusionCUDAGraphConfig:
    """Configuration for diffusion CUDA graph capture/replay."""

    enabled: bool = False
    max_graphs: int = 4
    warmup_steps: int = 1
    clone_outputs: bool = True
    use_global_graph_pool: bool = True
    include_non_tensor_inputs: bool = True
    include_tensor_strides: bool = True
    clear_cuda_cache_on_capture: bool = False
    name: str = "diffusion"

    @classmethod
    def from_value(
        cls,
        value: DiffusionCUDAGraphConfig | Mapping[str, Any] | None,
        *,
        enabled: bool | None = None,
    ) -> DiffusionCUDAGraphConfig:
        if value is None:
            config = cls()
        elif isinstance(value, cls):
            config = value
        elif isinstance(value, Mapping):
            valid_fields = {field.name for field in fields(cls)}
            config = cls(**{key: item for key, item in value.items() if key in valid_fields})
        else:
            raise TypeError(
                f"cuda_graph_config must be a DiffusionCUDAGraphConfig, mapping, or None, got {type(value)!r}"
            )
        if enabled:
            config = replace(config, enabled=True)
        return config


@dataclass
class GraphEntry:
    """Captured graph and its static input/output buffers."""

    key: GraphKey
    graph: torch.cuda.CUDAGraph
    static_args: list[Any]
    static_kwargs: dict[str, Any]
    static_output: Any
    hits: int = 0


class GraphKeyBuilder:
    """Build graph keys from nested forward inputs.

    Tensor leaves are keyed by shape, dtype, device, layout, and optionally
    stride. Non-tensor leaves are included when they are hashable primitives or
    simple containers of hashable values. Unsupported objects fall back to
    object identity, which is conservative and avoids accidental graph reuse.
    """

    def __init__(
        self,
        *,
        include_non_tensor_inputs: bool = True,
        include_tensor_strides: bool = True,
    ) -> None:
        self.include_non_tensor_inputs = include_non_tensor_inputs
        self.include_tensor_strides = include_tensor_strides

    def build(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        *,
        extra_key: Hashable | Mapping[str, Hashable] | Sequence[Hashable] | None = None,
    ) -> GraphKey:
        parts: list[tuple[str, Hashable]] = []
        for idx, arg in enumerate(args):
            parts.extend(self._parts_for_value(f"arg{idx}", arg))
        for name in sorted(kwargs):
            parts.extend(self._parts_for_value(name, kwargs[name]))
        if extra_key is not None:
            parts.extend(self._extra_key_parts(extra_key))
        return tuple(parts)

    def first_tensor(self, args: Sequence[Any], kwargs: Mapping[str, Any]) -> torch.Tensor | None:
        for value in args:
            tensor = self._find_tensor(value)
            if tensor is not None:
                return tensor
        for value in kwargs.values():
            tensor = self._find_tensor(value)
            if tensor is not None:
                return tensor
        return None

    def _parts_for_value(self, name: str, value: Any) -> list[tuple[str, Hashable]]:
        if isinstance(value, torch.Tensor):
            return [(name, self._tensor_key(value))]
        if isinstance(value, tuple):
            parts: list[tuple[str, Hashable]] = [(f"{name}.__type__", ("tuple", len(value)))]
            for idx, item in enumerate(value):
                parts.extend(self._parts_for_value(f"{name}.{idx}", item))
            return parts
        if isinstance(value, list):
            parts = [(f"{name}.__type__", ("list", len(value)))]
            for idx, item in enumerate(value):
                parts.extend(self._parts_for_value(f"{name}.{idx}", item))
            return parts
        if isinstance(value, Mapping):
            key_parts: list[tuple[str, Hashable]] = [(f"{name}.__type__", ("dict", len(value)))]
            for key in sorted(value, key=repr):
                key_hash = self._hashable_leaf(key)
                key_parts.append((f"{name}.__key__.{repr(key)}", key_hash))
                key_parts.extend(self._parts_for_value(f"{name}.{repr(key)}", value[key]))
            return key_parts
        if not self.include_non_tensor_inputs:
            return []
        return [(name, self._hashable_leaf(value))]

    def _tensor_key(self, tensor: torch.Tensor) -> Hashable:
        device = tensor.device
        key: list[Hashable] = [
            "tensor",
            tuple(tensor.shape),
            str(tensor.dtype),
            (device.type, device.index),
            str(tensor.layout),
        ]
        if self.include_tensor_strides:
            key.append(tuple(tensor.stride()))
        return tuple(key)

    def _hashable_leaf(self, value: Any) -> Hashable:
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return (type(value).__name__, value)
        if isinstance(value, torch.dtype):
            return ("torch.dtype", str(value))
        if isinstance(value, torch.device):
            return ("torch.device", value.type, value.index)
        if isinstance(value, tuple):
            return ("tuple", tuple(self._hashable_leaf(item) for item in value))
        if isinstance(value, list):
            return ("list", tuple(self._hashable_leaf(item) for item in value))
        if isinstance(value, Mapping):
            return (
                "dict",
                tuple((self._hashable_leaf(key), self._hashable_leaf(value[key])) for key in sorted(value, key=repr)),
            )
        try:
            hash(value)
        except TypeError:
            return ("object", type(value).__module__, type(value).__qualname__, id(value))
        return ("hashable", type(value).__module__, type(value).__qualname__, value)

    def _extra_key_parts(
        self,
        extra_key: Hashable | Mapping[str, Hashable] | Sequence[Hashable],
    ) -> list[tuple[str, Hashable]]:
        if isinstance(extra_key, Mapping):
            return [
                (f"extra.{name}", self._hashable_leaf(value))
                for name, value in sorted(extra_key.items(), key=lambda item: repr(item[0]))
            ]
        if isinstance(extra_key, Sequence) and not isinstance(extra_key, (str, bytes)):
            return [(f"extra.{idx}", self._hashable_leaf(value)) for idx, value in enumerate(extra_key)]
        return [("extra", self._hashable_leaf(extra_key))]

    def _find_tensor(self, value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Mapping):
            values = value.values()
        elif isinstance(value, (tuple, list)):
            values = value
        else:
            return None
        for item in values:
            tensor = self._find_tensor(item)
            if tensor is not None:
                return tensor
        return None


class DiffusionCUDAGraphRunner:
    """Lazy CUDA graph capture/replay for diffusion model callables."""

    def __init__(
        self,
        config: DiffusionCUDAGraphConfig | None = None,
        *,
        key_builder: GraphKeyBuilder | None = None,
    ) -> None:
        self.config = config or DiffusionCUDAGraphConfig()
        self.key_builder = key_builder or GraphKeyBuilder(
            include_non_tensor_inputs=self.config.include_non_tensor_inputs,
            include_tensor_strides=self.config.include_tensor_strides,
        )
        self._cache: OrderedDict[GraphKey, GraphEntry] = OrderedDict()
        self.last_call_info: dict[str, Any] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        for entry in self._cache.values():
            entry.graph.reset()
        self._cache.clear()
        self.last_call_info = {"mode": "clear", "cache_size": 0}

    def run(
        self,
        fn: Callable[..., Any],
        *args: Any,
        graph_extra_key: Hashable | Mapping[str, Hashable] | Sequence[Hashable] | None = None,
        graph_can_capture: bool = True,
        graph_fallback_reason: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run ``fn`` eagerly or through a captured CUDA graph.

        ``graph_extra_key`` is for model-visible state that is not represented by
        input tensor metadata. ``graph_can_capture`` lets model adapters reject a
        call before the generic runner attempts capture.
        """

        if not self.config.enabled:
            return self._eager(fn, "disabled", args, kwargs)
        if not graph_can_capture:
            return self._eager(fn, graph_fallback_reason or "adapter_rejected", args, kwargs)
        first_tensor = self.key_builder.first_tensor(args, kwargs)
        if first_tensor is None:
            return self._eager(fn, "no_tensor_inputs", args, kwargs)
        if first_tensor.device.type != "cuda":
            return self._eager(fn, "non_cuda_inputs", args, kwargs)
        if torch.cuda.is_current_stream_capturing():
            return self._eager(fn, "stream_capturing", args, kwargs)

        key = self.key_builder.build(args, kwargs, extra_key=graph_extra_key)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._capture(key, fn, args, kwargs)
            self._cache[key] = entry
            self._evict_if_needed()
            entry.graph.replay()
            entry.hits += 1
            self.last_call_info = {
                "mode": "graph",
                "reason": "capture",
                "cache_size": len(self._cache),
                "key_size": len(key),
            }
            return self._return_output(entry.static_output)

        self._cache.move_to_end(key)
        self._copy_inputs(entry.static_args, list(args))
        self._copy_inputs(entry.static_kwargs, kwargs)
        entry.graph.replay()
        entry.hits += 1
        self.last_call_info = {
            "mode": "graph",
            "reason": "hit",
            "cache_size": len(self._cache),
            "key_size": len(key),
            "hits": entry.hits,
        }
        return self._return_output(entry.static_output)

    def wrap(
        self,
        fn: Callable[..., Any],
        *,
        extra_key_fn: Callable[..., Hashable | Mapping[str, Hashable] | Sequence[Hashable] | None] | None = None,
        can_capture_fn: Callable[..., tuple[bool, str | None] | bool] | None = None,
    ) -> Callable[..., Any]:
        """Return a drop-in wrapper around ``fn``."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            extra_key = extra_key_fn(*args, **kwargs) if extra_key_fn is not None else None
            can_capture = True
            reason = None
            if can_capture_fn is not None:
                decision = can_capture_fn(*args, **kwargs)
                if isinstance(decision, tuple):
                    can_capture, reason = decision
                else:
                    can_capture = bool(decision)
            return self.run(
                fn,
                *args,
                graph_extra_key=extra_key,
                graph_can_capture=can_capture,
                graph_fallback_reason=reason,
                **kwargs,
            )

        return wrapper

    def _capture(
        self,
        key: GraphKey,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphEntry:
        static_args = [self._clone_static(value) for value in args]
        static_kwargs = {name: self._clone_static(value) for name, value in kwargs.items()}
        device_tensor = self.key_builder.first_tensor(static_args, static_kwargs)
        assert device_tensor is not None

        with torch.no_grad():
            for _ in range(self.config.warmup_steps):
                fn(*static_args, **static_kwargs)
                torch.accelerator.synchronize(device_tensor.device)
                if self.config.clear_cuda_cache_on_capture:
                    torch.accelerator.empty_cache()

        graph = torch.cuda.CUDAGraph()
        pool = self._graph_pool()
        with torch.no_grad():
            if pool is None:
                with torch.cuda.graph(graph):
                    static_output = fn(*static_args, **static_kwargs)
            else:
                with torch.cuda.graph(graph, pool=pool):
                    static_output = fn(*static_args, **static_kwargs)

        logger.debug("Captured %s CUDA graph with key_size=%d", self.config.name, len(key))
        return GraphEntry(
            key=key,
            graph=graph,
            static_args=static_args,
            static_kwargs=static_kwargs,
            static_output=static_output,
        )

    def _eager(
        self,
        fn: Callable[..., Any],
        reason: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        self.last_call_info = {
            "mode": "eager",
            "reason": reason,
            "cache_size": len(self._cache),
        }
        return fn(*args, **kwargs)

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.config.max_graphs:
            _, entry = self._cache.popitem(last=False)
            entry.graph.reset()

    def _graph_pool(self) -> Any:
        if not self.config.use_global_graph_pool:
            return None
        return current_platform.get_global_graph_pool()

    def _return_output(self, output: Any) -> Any:
        if not self.config.clone_outputs:
            return output
        return self._clone_output(output)

    def _clone_static(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, tuple):
            return tuple(self._clone_static(item) for item in value)
        if isinstance(value, list):
            return [self._clone_static(item) for item in value]
        if isinstance(value, Mapping):
            return {key: self._clone_static(item) for key, item in value.items()}
        return value

    def _clone_output(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, tuple):
            return tuple(self._clone_output(item) for item in value)
        if isinstance(value, list):
            return [self._clone_output(item) for item in value]
        if isinstance(value, Mapping):
            return {key: self._clone_output(item) for key, item in value.items()}
        return value

    def _copy_inputs(self, static_value: Any, dynamic_value: Any) -> None:
        if isinstance(static_value, torch.Tensor):
            static_value.copy_(dynamic_value)
            return
        if isinstance(static_value, list):
            for static_item, dynamic_item in zip(static_value, dynamic_value):
                self._copy_inputs(static_item, dynamic_item)
            return
        if isinstance(static_value, tuple):
            for static_item, dynamic_item in zip(static_value, dynamic_value):
                self._copy_inputs(static_item, dynamic_item)
            return
        if isinstance(static_value, dict):
            for key in static_value:
                self._copy_inputs(static_value[key], dynamic_value[key])


__all__ = [
    "DiffusionCUDAGraphConfig",
    "DiffusionCUDAGraphRunner",
    "GraphEntry",
    "GraphKey",
    "GraphKeyBuilder",
]
