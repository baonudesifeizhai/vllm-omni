# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Final-layout tensor ownership shared by diffusion producers and restorers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn

from vllm_omni.host_weight_runtime import TensorKind
from vllm_omni.host_weight_runtime.identity import canonical_json

FINAL_LAYOUT_BF16_MODEL_CONTRACT = "vllm-omni.diffusion.final-layout-bf16-v1"

_SUPPORTED_PARAMETER_DTYPES = {torch.bfloat16, torch.float32}
_SUPPORTED_BUFFER_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
for _dtype_name in (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
):
    if (_dtype := getattr(torch, _dtype_name, None)) is not None:
        _SUPPORTED_BUFFER_DTYPES.add(_dtype)


class FinalLayoutContractError(ValueError):
    """A stable reason that a final-layout consumer must fail closed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RuntimeTensorTarget:
    """One complete final-layout tensor owned by a DiT component."""

    name: str
    tensor: torch.Tensor
    kind: TensorKind

    @property
    def role(self) -> str:
        if self.kind is TensorKind.PARAMETER:
            return "weight"
        return "persistent_buffer"

    @property
    def nbytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


def validate_final_layout_model_contract(
    dit_modules: Sequence[tuple[str, nn.Module]],
) -> tuple[Callable[[], None], ...]:
    """Require an explicit tensor-complete restore contract from every DiT."""
    if not dit_modules:
        raise FinalLayoutContractError(
            "unsupported_model_contract",
            "final-layout BF16 artifacts require at least one DiT component",
        )

    validators: list[Callable[[], None]] = []
    seen_names: set[str] = set()
    for name, module in dit_modules:
        if not name or name in seen_names:
            raise FinalLayoutContractError(
                "ambiguous_ownership",
                f"DiT component name {name!r} is empty or duplicated",
            )
        seen_names.add(name)
        if getattr(module, "host_weight_restore_contract", None) != FINAL_LAYOUT_BF16_MODEL_CONTRACT:
            raise FinalLayoutContractError(
                "unsupported_model_contract",
                f"DiT component {name!r} does not declare {FINAL_LAYOUT_BF16_MODEL_CONTRACT!r}",
            )
        validator = getattr(module, "validate_restored_host_weights", None)
        if not callable(validator):
            raise FinalLayoutContractError(
                "unsupported_model_contract",
                f"DiT component {name!r} does not implement validate_restored_host_weights()",
            )
        validators.append(validator)
    return tuple(validators)


def _named_parameters_with_duplicates(module: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    try:
        return module.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        return module.named_parameters()


def _named_buffers_with_duplicates(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    try:
        return module.named_buffers(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        return module.named_buffers()


def _is_persistent_buffer(module: nn.Module, local_name: str) -> bool:
    parent_path, _, leaf_name = local_name.rpartition(".")
    owner = module.get_submodule(parent_path)
    return leaf_name not in owner._non_persistent_buffers_set


def _resolve_pipeline_tensor(pipeline: nn.Module, runtime_name: str) -> torch.Tensor | None:
    parent_path, _, leaf_name = runtime_name.rpartition(".")
    parent = pipeline.get_submodule(parent_path)
    tensor = parent._parameters.get(leaf_name)
    if tensor is None:
        tensor = parent._buffers.get(leaf_name)
    return tensor


def collect_final_layout_targets(
    pipeline: nn.Module,
    dit_modules: Sequence[tuple[str, nn.Module]],
    *,
    require_materialized: bool,
) -> tuple[RuntimeTensorTarget, ...]:
    """Return the complete, alias-free BF16 DiT ownership boundary.

    Structural collection accepts CPU or meta targets so identity and restore
    planning can run before ordinary materialization. Producer collection also
    requires complete contiguous CPU storages.
    """
    validate_final_layout_model_contract(dit_modules)
    records: dict[str, RuntimeTensorTarget] = {}
    for dit_name, dit_module in dit_modules:
        for local_name, tensor in _named_parameters_with_duplicates(dit_module):
            runtime_name = f"{dit_name}.{local_name}"
            candidate = RuntimeTensorTarget(runtime_name, tensor, TensorKind.PARAMETER)
            existing = records.get(runtime_name)
            if existing is not None and existing.tensor is not tensor:
                raise FinalLayoutContractError(
                    "ambiguous_ownership",
                    f"multiple DiT tensors resolve to {runtime_name!r}",
                )
            records[runtime_name] = candidate
        for local_name, tensor in _named_buffers_with_duplicates(dit_module):
            if not _is_persistent_buffer(dit_module, local_name):
                continue
            runtime_name = f"{dit_name}.{local_name}"
            candidate = RuntimeTensorTarget(runtime_name, tensor, TensorKind.BUFFER)
            existing = records.get(runtime_name)
            if existing is not None and existing.tensor is not tensor:
                raise FinalLayoutContractError(
                    "ambiguous_ownership",
                    f"multiple DiT tensors resolve to {runtime_name!r}",
                )
            records[runtime_name] = candidate

    if not records:
        raise FinalLayoutContractError(
            "ambiguous_ownership",
            "no DiT parameters or persistent buffers were discovered",
        )

    object_owners: dict[int, str] = {}
    storage_owners: dict[tuple[int, int], str] = {}
    for record in records.values():
        tensor = record.tensor
        try:
            pipeline_tensor = _resolve_pipeline_tensor(pipeline, record.name)
        except AttributeError as exc:
            raise FinalLayoutContractError(
                "ambiguous_ownership",
                f"DiT tensor {record.name!r} is not owned by the pipeline",
            ) from exc
        if pipeline_tensor is not tensor:
            raise FinalLayoutContractError(
                "ambiguous_ownership",
                f"DiT tensor {record.name!r} does not resolve to the discovered object",
            )

        object_owner = object_owners.setdefault(id(tensor), record.name)
        if object_owner != record.name:
            raise FinalLayoutContractError(
                "unsupported_alias",
                f"{record.name!r} aliases tensor object {object_owner!r}",
            )
        if hasattr(tensor, "to_local"):
            raise FinalLayoutContractError(
                "unsupported_tensor",
                f"{record.name!r} is a distributed tensor",
            )
        if tensor.layout != torch.strided:
            raise FinalLayoutContractError(
                "unsupported_tensor",
                f"{record.name!r} uses unsupported layout {tensor.layout}",
            )
        if record.kind is TensorKind.PARAMETER and tensor.dtype not in _SUPPORTED_PARAMETER_DTYPES:
            raise FinalLayoutContractError(
                "unsupported_dtype",
                f"{record.name!r} must be BF16 or an explicitly preserved FP32 parameter, got {tensor.dtype}",
            )
        if record.kind is TensorKind.BUFFER and tensor.dtype not in _SUPPORTED_BUFFER_DTYPES:
            raise FinalLayoutContractError(
                "unsupported_dtype",
                f"{record.name!r} uses unsupported buffer dtype {tensor.dtype}",
            )
        if tensor.device.type not in {"cpu", "meta"}:
            raise FinalLayoutContractError(
                "unsupported_tensor",
                f"{record.name!r} must be a CPU or meta tensor, got {tensor.device}",
            )
        if not tensor.is_contiguous():
            raise FinalLayoutContractError(
                "unsupported_tensor",
                f"{record.name!r} is non-contiguous with stride {tensor.stride()}",
            )

        if not require_materialized:
            continue
        if tensor.device.type != "cpu" or tensor.is_meta:
            raise FinalLayoutContractError(
                "unsupported_tensor",
                f"{record.name!r} must be a materialized CPU tensor, got {tensor.device}",
            )
        storage = tensor.untyped_storage()
        if tensor.storage_offset() != 0 or storage.nbytes() != record.nbytes:
            raise FinalLayoutContractError(
                "unsupported_tensor",
                f"{record.name!r} is a view into a larger storage",
            )
        if record.nbytes:
            storage_id = (storage.data_ptr(), storage.nbytes())
            storage_owner = storage_owners.setdefault(storage_id, record.name)
            if storage_owner != record.name:
                raise FinalLayoutContractError(
                    "unsupported_alias",
                    f"{record.name!r} shares storage with {storage_owner!r}",
                )

    # Reject aliases registered outside the DiT ownership boundary. This also
    # covers non-persistent buffers, encoders, VAEs, and resident components.
    for pipeline_name, tensor in chain(
        _named_parameters_with_duplicates(pipeline),
        _named_buffers_with_duplicates(pipeline),
    ):
        owner = object_owners.get(id(tensor))
        if owner is not None and owner != pipeline_name:
            raise FinalLayoutContractError(
                "unsupported_alias",
                f"cached tensor {owner!r} aliases pipeline tensor {pipeline_name!r}",
            )
        if not require_materialized or tensor.device.type != "cpu" or tensor.is_meta or tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        pipeline_storage_owner = storage_owners.get((storage.data_ptr(), storage.nbytes()))
        if pipeline_storage_owner is not None and pipeline_storage_owner != pipeline_name:
            raise FinalLayoutContractError(
                "unsupported_alias",
                f"cached tensor {pipeline_storage_owner!r} shares storage with pipeline tensor {pipeline_name!r}",
            )

    if not any(
        record.kind is TensorKind.PARAMETER and record.tensor.dtype is torch.bfloat16 for record in records.values()
    ):
        raise FinalLayoutContractError(
            "unsupported_dtype",
            "final-layout BF16 representation requires at least one BF16 parameter",
        )

    return tuple(records[name] for name in sorted(records))


def tensor_contract_sha256(records: Sequence[RuntimeTensorTarget]) -> str:
    """Hash the exact structural ownership represented by ordered targets."""
    contract = [
        {
            "dtype": str(record.tensor.dtype),
            "kind": record.kind.value,
            "name": record.name,
            "shape": list(record.tensor.shape),
            "stride": list(record.tensor.stride()),
        }
        for record in records
    ]
    return hashlib.sha256(canonical_json(contract)).hexdigest()


__all__ = [
    "FINAL_LAYOUT_BF16_MODEL_CONTRACT",
    "FinalLayoutContractError",
    "RuntimeTensorTarget",
    "collect_final_layout_targets",
    "tensor_contract_sha256",
    "validate_final_layout_model_contract",
]
