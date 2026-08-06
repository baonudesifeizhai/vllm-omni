# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import json
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.utils import get_packed_modules_mapping

logger = init_logger(__name__)

MODEL_OPT_SCALE_SUFFIXES = (
    ".input_scale",
    ".weight_scale",
    ".weight_scale_2",
    ".weight_scale_inv",
)
DEFAULT_PACKED_MODULES_MAPPING = {
    "to_qkv": ("to_q", "to_k", "to_v"),
    "add_kv_proj": ("add_q_proj", "add_k_proj", "add_v_proj"),
    "w13": ("w1", "w3"),
}
FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)
SAFETENSORS_INDEX_FILES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
)
_CheckpointKeyMapper = Callable[[str], str | tuple[str, object] | None]


@dataclass(frozen=True)
class _ModelOptFp8QuantizationPlan:
    """Exact runtime precision decisions derived from checkpoint tensors."""

    quantized_modules: frozenset[str]
    unquantized_modules: frozenset[str]
    source_linear_count: int

    @property
    def modules(self) -> frozenset[str]:
        return self.quantized_modules | self.unquantized_modules


def _modelopt_safetensors_files(checkpoint_dir: Path) -> list[Path]:
    indexes = [checkpoint_dir / name for name in SAFETENSORS_INDEX_FILES if (checkpoint_dir / name).is_file()]
    if len(indexes) > 1:
        raise ValueError(f"Multiple safetensors indexes found in {checkpoint_dir}: {indexes}")
    if indexes:
        with indexes[0].open(encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {indexes[0]}")
        files = [checkpoint_dir / name for name in sorted(set(weight_map.values()))]
    else:
        files = sorted(checkpoint_dir.glob("*.safetensors"))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Safetensors shards referenced by {checkpoint_dir} are missing: {missing}")
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint found in {checkpoint_dir}")
    return files


def _map_checkpoint_target(
    key: str,
    key_mapper: _CheckpointKeyMapper,
    *,
    source_prefix: str,
    target_prefix: str,
    allow_unmapped: bool = False,
) -> str | None:
    remapped = key_mapper(source_prefix + key)
    target = remapped[0] if isinstance(remapped, tuple) else remapped
    if target is None:
        if allow_unmapped:
            return None
        raise ValueError(f"No canonical runtime mapping for checkpoint tensor {key!r}")
    if target_prefix:
        if not target.startswith(target_prefix):
            raise ValueError(f"Mapped tensor {key!r} to {target!r}, outside target prefix {target_prefix!r}")
        target = target[len(target_prefix) :]
    return target


def _build_modelopt_fp8_quantization_plan(
    checkpoint_dir: str | Path,
    key_mapper: _CheckpointKeyMapper,
    *,
    source_prefix: str = "",
    target_prefix: str = "",
    allow_unmapped: bool = False,
    target_module_prefixes: tuple[str, ...] = (),
) -> _ModelOptFp8QuantizationPlan:
    """Derive exact BF16/FP8 runtime groups from safetensors headers.

    Multiple source projections may map to one packed runtime module. Every
    member must use the same precision, otherwise the checkpoint is rejected.
    """

    checkpoint_dir = Path(checkpoint_dir)
    tensor_metadata: dict[str, tuple[str, tuple[int, ...]]] = {}
    for filename in _modelopt_safetensors_files(checkpoint_dir):
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in tensor_metadata:
                    raise ValueError(f"Duplicate safetensors key {key!r} in {checkpoint_dir}")
                tensor_slice = handle.get_slice(key)
                tensor_metadata[key] = (str(tensor_slice.get_dtype()), tuple(tensor_slice.get_shape()))

    target_sources: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for key, (dtype, shape) in tensor_metadata.items():
        if not key.endswith(".weight") or len(shape) != 2:
            continue

        is_fp8 = dtype.startswith("F8_")
        if not is_fp8 and dtype not in {"BF16", "F16", "F32"}:
            raise ValueError(f"Unsupported ModelOpt linear dtype {dtype} for {key!r}")
        scale_key = key.removesuffix(".weight") + ".weight_scale"
        has_scale = scale_key in tensor_metadata
        if is_fp8 != has_scale:
            expected = "present" if is_fp8 else "absent"
            raise ValueError(f"ModelOpt weight_scale must be {expected} for {key!r}, got {has_scale=}")

        target_weight = _map_checkpoint_target(
            key,
            key_mapper,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
            allow_unmapped=allow_unmapped,
        )
        if target_weight is None:
            continue
        if not target_weight.endswith(".weight"):
            raise ValueError(f"Linear weight {key!r} mapped to non-weight target {target_weight!r}")
        target_module = target_weight.removesuffix(".weight")
        if target_module_prefixes and not target_module.startswith(target_module_prefixes):
            continue

        if is_fp8:
            target_scale = _map_checkpoint_target(
                scale_key,
                key_mapper,
                source_prefix=source_prefix,
                target_prefix=target_prefix,
            )
            if target_scale != target_module + ".weight_scale":
                raise ValueError(
                    f"Weight and scale for {key!r} map to different runtime modules: "
                    f"{target_weight!r} vs {target_scale!r}"
                )
        target_sources[target_module].append((key, is_fp8))

    conflicts = {
        target: sources for target, sources in target_sources.items() if len({is_fp8 for _, is_fp8 in sources}) != 1
    }
    if conflicts:
        raise ValueError(f"Packed runtime modules contain mixed BF16/FP8 source projections: {conflicts}")

    quantized = frozenset(target for target, sources in target_sources.items() if sources[0][1])
    unquantized = frozenset(target_sources) - quantized
    plan = _ModelOptFp8QuantizationPlan(
        quantized_modules=quantized,
        unquantized_modules=unquantized,
        source_linear_count=sum(len(sources) for sources in target_sources.values()),
    )
    logger.info(
        "Derived ModelOpt FP8 checkpoint plan from %s: %d source linears -> "
        "%d runtime groups (%d FP8, %d full precision)",
        checkpoint_dir,
        plan.source_linear_count,
        len(plan.modules),
        len(plan.quantized_modules),
        len(plan.unquantized_modules),
    )
    return plan


def _validate_modelopt_fp8_quantization_plan(
    model: nn.Module,
    plan: _ModelOptFp8QuantizationPlan,
) -> None:
    """Verify that runtime LinearBase construction exactly matches the plan."""

    runtime_linears = {name: module for name, module in model.named_modules() if isinstance(module, LinearBase)}
    runtime_names = frozenset(runtime_linears)
    missing_runtime = plan.modules - runtime_names
    unplanned_runtime = runtime_names - plan.modules
    if missing_runtime or unplanned_runtime:
        raise ValueError(
            "ModelOpt checkpoint/runtime linear mapping is incomplete: "
            f"missing_runtime={sorted(missing_runtime)}, unplanned_runtime={sorted(unplanned_runtime)}"
        )

    mismatched: list[str] = []
    for name, module in runtime_linears.items():
        is_unquantized = isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod)
        if is_unquantized != (name in plan.unquantized_modules):
            mismatched.append(name)
    if mismatched:
        raise ValueError(f"Runtime linear precision does not match the ModelOpt checkpoint plan: {mismatched}")


class ModelOptFp8CheckpointConfig:
    """Use exact BF16/FP8 decisions derived from checkpoint tensors."""

    def __init__(self, quant_config: Any, plan: _ModelOptFp8QuantizationPlan):
        self._quant_config = copy.copy(quant_config)
        self._quant_config.exclude_modules = []
        self._plan = plan

    @classmethod
    def from_checkpoint(
        cls,
        quant_config: Any,
        checkpoint_dir: str | Path,
        key_mapper: _CheckpointKeyMapper,
        **plan_kwargs: Any,
    ) -> "ModelOptFp8CheckpointConfig":
        plan = _build_modelopt_fp8_quantization_plan(
            checkpoint_dir,
            key_mapper,
            **plan_kwargs,
        )
        return cls(quant_config, plan)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._quant_config, name)

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Any:
        if isinstance(layer, LinearBase):
            if prefix in self._plan.unquantized_modules:
                return UnquantizedLinearMethod()
            if prefix not in self._plan.quantized_modules:
                raise ValueError(f"No checkpoint precision decision for runtime linear {prefix!r}")
        return self._quant_config.get_quant_method(layer, prefix)

    def validate(self, model: nn.Module) -> None:
        _validate_modelopt_fp8_quantization_plan(model, self._plan)


@dataclass
class _AdaptState:
    scale_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    pending_weights: dict[str, list[tuple[str, str, torch.Tensor, torch.dtype]]] = field(default_factory=dict)
    skipped_scales: int = 0
    dequantized_weights: int = 0


class ModelOptFp8CheckpointAdapter:
    def __init__(self, model: nn.Module, source: object):
        self._loadable_tensors = self._get_model_loadable_tensors(model)
        self._weights_mapper = self._get_weights_mapper(model)
        self._checkpoint_key_mapper = getattr(model, "remap_checkpoint_key", None)
        self._source_label = getattr(source, "prefix", "") or getattr(source, "subfolder", None) or "model"

    @classmethod
    def is_compatible(
        cls,
        source: object,
        quant_config: object | None,
        use_safetensors: bool,
    ) -> bool:
        return use_safetensors and cls._is_transformer_source(source) and cls._is_checkpoint_quant_config(quant_config)

    @staticmethod
    def _is_transformer_source(source: object) -> bool:
        if getattr(source, "subfolder", None) == "transformer":
            return True
        return str(getattr(source, "prefix", "")).startswith("transformer.")

    @staticmethod
    def _is_checkpoint_quant_config(quant_config: object | None) -> bool:
        return (
            quant_config is not None
            and hasattr(quant_config, "get_name")
            and quant_config.get_name() == "modelopt"
            and bool(getattr(quant_config, "is_checkpoint_fp8_serialized", False))
        )

    @staticmethod
    def _get_model_loadable_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
        loadable_tensors: dict[str, torch.Tensor] = {name: param for name, param in model.named_parameters()}
        loadable_tensors.update({name: buffer for name, buffer in model.named_buffers()})
        return loadable_tensors

    @staticmethod
    def _is_scale(name: str) -> bool:
        return name.endswith(MODEL_OPT_SCALE_SUFFIXES)

    @staticmethod
    def _is_fp8_tensor(tensor: torch.Tensor) -> bool:
        return tensor.dtype in FP8_DTYPES

    @staticmethod
    def _get_weight_scale_name(weight_name: str) -> str | None:
        if weight_name.endswith(".weight"):
            return weight_name[: -len(".weight")] + ".weight_scale"
        return None

    @classmethod
    def _get_weights_mapper(cls, model: nn.Module) -> WeightsMapper:
        mapping = {
            packed_name: tuple(shard_names) for packed_name, shard_names in DEFAULT_PACKED_MODULES_MAPPING.items()
        }
        mapping.update(
            {
                str(packed_name): tuple(str(shard_name) for shard_name in shard_names)
                for packed_name, shard_names in get_packed_modules_mapping(model).items()
            }
        )

        orig_to_new_substr = {".to_out.0.": ".to_out."}
        orig_to_new_prefix: dict[str, str] = {}
        for packed_name, shard_names in mapping.items():
            for shard_name in shard_names:
                orig_to_new_substr[f".{shard_name}."] = f".{packed_name}."
                orig_to_new_prefix[f"{shard_name}."] = f"{packed_name}."
        return WeightsMapper(
            orig_to_new_substr=orig_to_new_substr,
            orig_to_new_prefix=orig_to_new_prefix,
        )

    def _resolve_target_and_output_names(self, name: str) -> tuple[str | None, str]:
        if name in self._loadable_tensors:
            return name, name

        if callable(self._checkpoint_key_mapper):
            remapped = self._checkpoint_key_mapper(name)
            candidate, output_name = remapped if isinstance(remapped, tuple) else (remapped, remapped)
            if candidate in self._loadable_tensors:
                return candidate, output_name

        for candidate in self._weights_mapper.apply_list([name]):
            if candidate != name and candidate in self._loadable_tensors:
                # Preserve shard names so downstream packed-weight loaders can route them.
                return candidate, name
        return None, name

    @staticmethod
    def _reshape_weight_scale(scale: torch.Tensor, weight_shape: torch.Size) -> torch.Tensor:
        if scale.numel() == 1:
            return scale.reshape(())
        if len(weight_shape) == 2 and scale.ndim == 1 and scale.shape[0] == weight_shape[0]:
            return scale.reshape(-1, 1)
        if tuple(scale.shape) == tuple(weight_shape):
            return scale
        if (
            len(weight_shape) == 2
            and scale.ndim == 4
            and scale.shape[1] == 1
            and scale.shape[3] == 1
            and weight_shape[0] % scale.shape[0] == 0
            and weight_shape[1] % scale.shape[2] == 0
        ):
            block_n = weight_shape[0] // scale.shape[0]
            block_k = weight_shape[1] // scale.shape[2]
            return scale.expand(scale.shape[0], block_n, scale.shape[2], block_k).reshape(weight_shape)
        raise ValueError(f"Unsupported ModelOpt FP8 weight_scale shape {tuple(scale.shape)} for weight {weight_shape}")

    def _dequantize_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        state: _AdaptState,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        scale_name = self._get_weight_scale_name(name)
        if scale_name is None or scale_name not in state.scale_tensors:
            raise ValueError(f"Missing ModelOpt FP8 weight_scale for full-precision target weight {name!r}")

        scale = state.scale_tensors[scale_name].to(dtype=target_dtype, device=loaded_weight.device)
        scale = self._reshape_weight_scale(scale, loaded_weight.shape)
        return loaded_weight.to(dtype=target_dtype) * scale

    def _flush_pending_weights(
        self,
        scale_name: str,
        state: _AdaptState,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        for (
            weight_name,
            output_name,
            weight_tensor,
            target_dtype,
        ) in state.pending_weights.pop(scale_name, []):
            yield output_name, self._dequantize_weight(weight_name, weight_tensor, state, target_dtype)
            state.dequantized_weights += 1

    def _handle_scale_tensor(
        self,
        name: str,
        output_name: str,
        tensor: torch.Tensor,
        target_name: str | None,
        state: _AdaptState,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        state.scale_tensors[name] = tensor
        if target_name is None:
            state.skipped_scales += 1
        else:
            yield output_name, tensor
        yield from self._flush_pending_weights(name, state)

    def _target_dtype_for_dequantization(
        self,
        tensor: torch.Tensor,
        target_name: str | None,
    ) -> torch.dtype | None:
        if target_name is None or not self._is_fp8_tensor(tensor):
            return None

        target_dtype = self._loadable_tensors[target_name].dtype
        if target_dtype in FP8_DTYPES:
            return None
        return target_dtype

    def _check_full_precision_source_target(
        self,
        name: str,
        tensor: torch.Tensor,
        target_name: str | None,
    ) -> None:
        if target_name is None or not name.endswith(".weight") or self._is_fp8_tensor(tensor):
            return
        if self._loadable_tensors[target_name].dtype in FP8_DTYPES:
            raise ValueError(
                f"Full-precision ModelOpt checkpoint weight {name!r} maps to FP8 runtime parameter "
                f"{target_name!r}; the checkpoint precision plan was not applied"
            )

    def _maybe_dequantize_or_defer_weight(
        self,
        name: str,
        output_name: str,
        tensor: torch.Tensor,
        target_dtype: torch.dtype,
        state: _AdaptState,
    ) -> torch.Tensor | None:
        scale_name = self._get_weight_scale_name(name)
        if scale_name is None:
            raise ValueError(f"Missing ModelOpt FP8 weight_scale name for weight {name!r}")

        if scale_name not in state.scale_tensors:
            state.pending_weights.setdefault(scale_name, []).append((name, output_name, tensor, target_dtype))
            return None

        state.dequantized_weights += 1
        return self._dequantize_weight(name, tensor, state, target_dtype)

    @staticmethod
    def _check_pending_weights(state: _AdaptState) -> None:
        if not state.pending_weights:
            return

        missing_scale_names = ", ".join(repr(name) for name in sorted(state.pending_weights))
        raise ValueError(f"Missing ModelOpt FP8 weight_scale for full-precision target weights: {missing_scale_names}")

    def _log_adaptation_summary(self, state: _AdaptState) -> None:
        if not state.skipped_scales and not state.dequantized_weights:
            return

        logger.info_once(
            "Adapted ModelOpt FP8 %s weights: dequantized %d full-precision weights, skipped %d scale tensors",
            self._source_label,
            state.dequantized_weights,
            state.skipped_scales,
        )

    def adapt(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        state = _AdaptState()

        for name, tensor in weights:
            target_name, output_name = self._resolve_target_and_output_names(name)
            if self._is_scale(name):
                yield from self._handle_scale_tensor(name, output_name, tensor, target_name, state)
                continue

            self._check_full_precision_source_target(name, tensor, target_name)
            target_dtype = self._target_dtype_for_dequantization(tensor, target_name)
            if target_dtype is not None:
                tensor = self._maybe_dequantize_or_defer_weight(name, output_name, tensor, target_dtype, state)
                if tensor is None:
                    continue
            yield output_name, tensor

        self._check_pending_weights(state)
        self._log_adaptation_summary(state)


class ModelOptNvFp4CheckpointAdapter(ModelOptFp8CheckpointAdapter):
    _PRE_QUANT_SCALE_SUFFIX = ".pre_quant_scale"

    @staticmethod
    def _is_checkpoint_quant_config(quant_config: object | None) -> bool:
        return (
            quant_config is not None
            and hasattr(quant_config, "get_name")
            and quant_config.get_name() == "modelopt_fp4"
            and bool(getattr(quant_config, "is_checkpoint_nvfp4_serialized", False))
        )

    def adapt(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        def validated_weights() -> Generator[tuple[str, torch.Tensor], None, None]:
            for name, tensor in weights:
                if name.endswith(self._PRE_QUANT_SCALE_SUFFIX):
                    raise ValueError(
                        f"ModelOpt NVFP4 checkpoint tensor {name!r} is unsupported: "
                        "vLLM 0.25.0 does not consume pre_quant_scale. Export the checkpoint "
                        "with pre-quant scales folded into the weights."
                    )
                yield name, tensor

        yield from super().adapt(validated_weights())


class ModelOptMixedPrecisionCheckpointAdapter(ModelOptFp8CheckpointAdapter):
    @staticmethod
    def _is_checkpoint_quant_config(quant_config: object | None) -> bool:
        return (
            quant_config is not None
            and hasattr(quant_config, "get_name")
            and quant_config.get_name() == "modelopt_mixed"
        )
