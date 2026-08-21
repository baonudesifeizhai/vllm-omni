# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validation-first restorer for final-layout BF16 diffusion artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm_omni.host_weight_runtime import HostWeightLease, TensorKind, WeightRestorePlan

from .producers.final_layout_bf16 import (
    FINAL_LAYOUT_BF16_REPRESENTATION,
    FINAL_LAYOUT_BF16_RESTORER_SCHEMA,
    validate_final_layout_bf16_identity,
)
from .tensor_layout import collect_final_layout_targets, tensor_contract_sha256, validate_final_layout_model_contract

if TYPE_CHECKING:
    from .identity_adapter import FinalLayoutBF16IdentityContext


@dataclass(frozen=True)
class _Replacement:
    name: str
    parent: nn.Module
    leaf_name: str
    target: torch.Tensor
    source: torch.Tensor
    kind: TensorKind

    def current_target(self) -> torch.Tensor | None:
        if self.kind is TensorKind.PARAMETER:
            return self.parent._parameters.get(self.leaf_name)
        return self.parent._buffers.get(self.leaf_name)


class FinalLayoutBF16RestorePlan(WeightRestorePlan):
    """A one-shot storage-rebinding transaction validated before mutation."""

    def __init__(
        self,
        lease: HostWeightLease,
        replacements: tuple[_Replacement, ...],
        validators: tuple[Callable[[], None], ...],
        source_guard: Callable[[], None],
    ) -> None:
        self._lease = lease
        self._replacements = replacements
        self._validators = validators
        self._source_guard = source_guard
        self._committed = False

    def commit(self) -> None:
        if self._committed:
            raise RuntimeError("final-layout BF16 restore plan was already committed")
        if self._lease.closed:
            raise RuntimeError("cannot commit a final-layout BF16 restore from a closed lease")
        self._source_guard()

        # Revalidate every target before the first mutation. A model rewrite
        # between planning and commit therefore cannot produce a partial mix.
        for replacement in self._replacements:
            current = replacement.current_target()
            if current is not replacement.target:
                raise RuntimeError(f"restore target {replacement.name!r} changed after preflight")
            if (
                tuple(current.shape) != tuple(replacement.source.shape)
                or current.dtype != replacement.source.dtype
                or tuple(current.stride()) != tuple(replacement.source.stride())
                or not replacement.source.is_contiguous()
            ):
                raise RuntimeError(f"restore target {replacement.name!r} changed layout after preflight")

        # Mark first: any unexpected assignment or validator failure makes this
        # plan permanently non-retryable and the partially restored model
        # disposable. Loader fallback is responsible for constructing a fresh
        # model instance in the consumer integration.
        self._committed = True
        for replacement in self._replacements:
            if replacement.target.is_meta:
                if replacement.kind is TensorKind.PARAMETER:
                    replacement.parent._parameters[replacement.leaf_name] = nn.Parameter(
                        replacement.source,
                        requires_grad=replacement.target.requires_grad,
                    )
                else:
                    replacement.parent._buffers[replacement.leaf_name] = replacement.source
            else:
                replacement.target.data = replacement.source
        for validator in self._validators:
            validator()


class FinalLayoutBF16Restorer:
    """Validate one exact lease against a structurally initialized pipeline."""

    def __init__(self, context: FinalLayoutBF16IdentityContext) -> None:
        if not context.dit_names or any(not name for name in context.dit_names):
            raise ValueError("final-layout BF16 restorer requires DiT component names")
        self._context = context

    @property
    def schema(self) -> str:
        return FINAL_LAYOUT_BF16_RESTORER_SCHEMA

    def plan_restore(self, model: object, lease: HostWeightLease) -> FinalLayoutBF16RestorePlan:
        if not isinstance(model, nn.Module):
            raise TypeError("final-layout BF16 restoration requires an nn.Module pipeline")
        if type(model) is not self._context.pipeline_type:
            raise ValueError("restore model implementation differs from the exact identity context")
        if lease.closed:
            raise ValueError("cannot restore from a closed HostWeightLease")
        if lease.identity.canonical_bytes != self._context.identity.canonical_bytes:
            raise ValueError("lease semantic identity differs from the exact restore request")
        if lease.manifest.identity.canonical_bytes != self._context.identity.canonical_bytes:
            raise ValueError("lease manifest identity differs from the exact restore request")
        if lease.identity.producer.restorer_schema != self.schema or lease.manifest.restorer_schema != self.schema:
            raise ValueError("lease restorer schema is incompatible with final-layout BF16 restoration")
        if lease.identity.representation.name != FINAL_LAYOUT_BF16_REPRESENTATION:
            raise ValueError("lease representation is not diffusion final-layout BF16")
        self._context.ensure_sources_unchanged()

        try:
            dit_modules = tuple((name, model.get_submodule(name)) for name in self._context.dit_names)
        except AttributeError as exc:
            raise ValueError("one or more lease-owned DiT modules do not exist in the pipeline") from exc
        if tuple(type(module) for _, module in dit_modules) != self._context.dit_types:
            raise ValueError("restore DiT implementation differs from the exact identity context")
        validators = validate_final_layout_model_contract(dit_modules)
        targets = collect_final_layout_targets(model, dit_modules, require_materialized=False)
        contract_digest = tensor_contract_sha256(targets)
        validate_final_layout_bf16_identity(
            self._context.identity,
            dit_names=self._context.dit_names,
            tensor_contract_digest=contract_digest,
        )

        metadata = lease.manifest.format_metadata.to_value()
        if (
            not isinstance(metadata, dict)
            or metadata.get("component_names") != list(self._context.dit_names)
            or metadata.get("format") != FINAL_LAYOUT_BF16_REPRESENTATION
            or metadata.get("mixed_precision_policy") != "bf16-with-preserved-fp32"
            or metadata.get("tensor_contract_sha256") != contract_digest
        ):
            raise ValueError("lease format metadata differs from the requested DiT ownership")

        manifest_entries = {entry.name: entry for entry in lease.manifest.tensors}
        target_names = {target.name for target in targets}
        if metadata.get("tensor_count") != len(manifest_entries):
            raise ValueError("lease tensor count differs from its final-layout metadata")
        if set(manifest_entries) != target_names or target_names != self._context.tensor_names:
            missing = sorted(target_names - set(manifest_entries))
            unexpected = sorted(set(manifest_entries) - target_names)
            raise ValueError(
                "lease tensor coverage differs from the structurally initialized DiT: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        replacements: list[_Replacement] = []
        source_storages: dict[tuple[int, int], str] = {}
        for target in targets:
            entry = manifest_entries[target.name]
            source = lease.tensors[target.name]
            expected_role = "weight" if target.kind is TensorKind.PARAMETER else "persistent_buffer"
            if entry.kind is not target.kind or entry.role != expected_role:
                raise ValueError(f"lease tensor kind or role differs for {target.name!r}")
            if (
                tuple(source.shape) != tuple(target.tensor.shape)
                or source.dtype != target.tensor.dtype
                or tuple(source.stride()) != tuple(target.tensor.stride())
                or not source.is_contiguous()
                or source.device.type != "cpu"
            ):
                raise ValueError(f"lease tensor layout differs for {target.name!r}")
            if source.numel():
                storage = source.untyped_storage()
                storage_id = (storage.data_ptr(), storage.nbytes())
                owner = source_storages.setdefault(storage_id, target.name)
                if owner != target.name:
                    raise ValueError(f"lease tensor {target.name!r} aliases {owner!r}")
            parent_path, _, leaf_name = target.name.rpartition(".")
            replacements.append(
                _Replacement(
                    name=target.name,
                    parent=model.get_submodule(parent_path),
                    leaf_name=leaf_name,
                    target=target.tensor,
                    source=source,
                    kind=target.kind,
                )
            )

        self._context.ensure_sources_unchanged()
        return FinalLayoutBF16RestorePlan(
            lease,
            tuple(replacements),
            validators,
            self._context.ensure_sources_unchanged,
        )


__all__ = [
    "FinalLayoutBF16RestorePlan",
    "FinalLayoutBF16Restorer",
]
