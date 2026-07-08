# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph helpers for diffusion transformer forwards.

The manager owns graph capture, replay, caching, and lifecycle. Model routines
own graph-safe input preparation and static-buffer updates.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, Protocol, TypeAlias

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

    def __post_init__(self) -> None:
        if self.max_graphs < 1:
            raise ValueError("max_graphs must be at least 1")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

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
            unknown_fields = set(value) - valid_fields
            if unknown_fields:
                raise ValueError(f"Unknown CUDA graph config fields: {sorted(unknown_fields)}")
            config = cls(**value)
        else:
            raise TypeError(
                f"cuda_graph_config must be a DiffusionCUDAGraphConfig, mapping, or None, got {type(value)!r}"
            )
        if enabled:
            config = replace(config, enabled=True)
        return config


@dataclass
class DiffusionCUDAGraphCall:
    """A model routine's normalized inputs and graph-safety decision."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    extra_key: Hashable | Mapping[str, Hashable] | Sequence[Hashable] | None = None
    can_capture: bool = True
    fallback_reason: str | None = None


@dataclass
class DiffusionCUDAGraphStaticInputs:
    """Static graph inputs owned and updated by a model routine."""

    args: list[Any]
    kwargs: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)


class DiffusionCUDAGraphRoutine(Protocol):
    """Model-owned CUDA graph input and forward contract."""

    def prepare(self, *args: Any, **kwargs: Any) -> DiffusionCUDAGraphCall: ...

    def eager(self, *args: Any, **kwargs: Any) -> Any: ...

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def create_static_inputs(self, call: DiffusionCUDAGraphCall) -> DiffusionCUDAGraphStaticInputs: ...

    def copy_inputs(
        self,
        static_inputs: DiffusionCUDAGraphStaticInputs,
        call: DiffusionCUDAGraphCall,
    ) -> None: ...


@dataclass
class GraphEntry:
    """Captured graph and the model-owned buffers used to replay it."""

    key: GraphKey
    graph: torch.cuda.CUDAGraph
    static_inputs: DiffusionCUDAGraphStaticInputs
    static_output: Any
    hits: int = 0


def clone_graph_value(value: Any) -> Any:
    """Clone tensor leaves while preserving a nested input structure."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_graph_value(item) for item in value)
    if isinstance(value, list):
        return [clone_graph_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: clone_graph_value(item) for key, item in value.items()}
    return value


def copy_graph_value(static_value: Any, dynamic_value: Any) -> None:
    """Copy tensor leaves into an already-compatible static structure."""
    if isinstance(static_value, torch.Tensor):
        static_value.copy_(dynamic_value)
        return
    if isinstance(static_value, (list, tuple)):
        for static_item, dynamic_item in zip(static_value, dynamic_value, strict=True):
            copy_graph_value(static_item, dynamic_item)
        return
    if isinstance(static_value, dict):
        for key in static_value:
            copy_graph_value(static_value[key], dynamic_value[key])


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
        routine: DiffusionCUDAGraphRoutine,
        config: DiffusionCUDAGraphConfig | None = None,
        *,
        key_builder: GraphKeyBuilder | None = None,
    ) -> None:
        self.routine = routine
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
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run the model routine eagerly or through a captured CUDA graph."""

        if not self.config.enabled:
            return self._eager("disabled", args, kwargs)

        call = self.routine.prepare(*args, **kwargs)
        if not call.can_capture:
            return self._eager(call.fallback_reason or "routine_rejected", args, kwargs)
        first_tensor = self.key_builder.first_tensor(call.args, call.kwargs)
        if first_tensor is None:
            return self._eager("no_tensor_inputs", args, kwargs)
        if first_tensor.device.type != "cuda":
            return self._eager("non_cuda_inputs", args, kwargs)
        if torch.cuda.is_current_stream_capturing():
            return self._eager("stream_capturing", args, kwargs)

        key = self.key_builder.build(call.args, call.kwargs, extra_key=call.extra_key)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._capture(key, call)
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
        self.routine.copy_inputs(entry.static_inputs, call)
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

    def _capture(
        self,
        key: GraphKey,
        call: DiffusionCUDAGraphCall,
    ) -> GraphEntry:
        static_inputs = self.routine.create_static_inputs(call)
        device_tensor = self.key_builder.first_tensor(static_inputs.args, static_inputs.kwargs)
        assert device_tensor is not None

        with torch.no_grad():
            for _ in range(self.config.warmup_steps):
                self.routine.forward(*static_inputs.args, **static_inputs.kwargs)
                torch.accelerator.synchronize(device_tensor.device)
                if self.config.clear_cuda_cache_on_capture:
                    torch.accelerator.empty_cache()

        graph = torch.cuda.CUDAGraph()
        pool = self._graph_pool()
        with torch.no_grad():
            if pool is None:
                with torch.cuda.graph(graph):
                    static_output = self.routine.forward(*static_inputs.args, **static_inputs.kwargs)
            else:
                with torch.cuda.graph(graph, pool=pool):
                    static_output = self.routine.forward(*static_inputs.args, **static_inputs.kwargs)

        logger.debug("Captured %s CUDA graph with key_size=%d", self.config.name, len(key))
        return GraphEntry(
            key=key,
            graph=graph,
            static_inputs=static_inputs,
            static_output=static_output,
        )

    def _eager(
        self,
        reason: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        self.last_call_info = {
            "mode": "eager",
            "reason": reason,
            "cache_size": len(self._cache),
        }
        return self.routine.eager(*args, **kwargs)

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
        return clone_graph_value(output)


class DiffusionCUDAGraphManager:
    """Own CUDA graph runners for the routines declared by one pipeline."""

    def __init__(self, config: DiffusionCUDAGraphConfig | None = None) -> None:
        self.config = config or DiffusionCUDAGraphConfig()
        self._routines: dict[str, DiffusionCUDAGraphRoutine] = {}
        self._runners: dict[str, DiffusionCUDAGraphRunner] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def register(self, name: str, routine: DiffusionCUDAGraphRoutine) -> None:
        old_runner = self._runners.get(name)
        if old_runner is not None:
            old_runner.clear()
        self._routines[name] = routine
        self._runners[name] = DiffusionCUDAGraphRunner(routine, replace(self.config, name=name))

    def clear(self) -> None:
        for runner in self._runners.values():
            runner.clear()

    def last_call_info(self, name: str) -> dict[str, Any]:
        return self._runners[name].last_call_info

    def run(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self._runners[name].run(*args, **kwargs)


__all__ = [
    "DiffusionCUDAGraphConfig",
    "DiffusionCUDAGraphCall",
    "DiffusionCUDAGraphManager",
    "DiffusionCUDAGraphRoutine",
    "DiffusionCUDAGraphRunner",
    "DiffusionCUDAGraphStaticInputs",
    "GraphEntry",
    "GraphKey",
    "GraphKeyBuilder",
    "clone_graph_value",
    "copy_graph_value",
]
