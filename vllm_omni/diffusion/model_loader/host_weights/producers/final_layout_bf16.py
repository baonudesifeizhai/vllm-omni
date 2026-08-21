# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Post-load producer for final-layout BF16 diffusion DiT weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm_omni.host_weight_runtime import (
    ArtifactWriter,
    CanonicalJson,
    CoordinationScope,
    LookupPhase,
    ProductionMetadata,
    ProductionSourceMode,
    TensorWriteSpec,
    WeightArtifactIdentity,
    WeightProductionSpec,
)

from ..tensor_layout import (
    FinalLayoutContractError,
    RuntimeTensorTarget,
    collect_final_layout_targets,
    tensor_contract_sha256,
    validate_final_layout_model_contract,
)

if TYPE_CHECKING:
    from ..identity_adapter import FinalLayoutBF16IdentityContext

FINAL_LAYOUT_BF16_PRODUCER_ID = "vllm-omni.diffusion.final-layout-bf16"
FINAL_LAYOUT_BF16_VERSION = "1"
FINAL_LAYOUT_BF16_REPRESENTATION = "diffusion-final-layout-bf16"
FINAL_LAYOUT_BF16_MANIFEST_SCHEMA = "diffusion-final-layout-bf16-manifest-v1"
FINAL_LAYOUT_BF16_RESTORER_SCHEMA = "diffusion-final-layout-bf16-restorer-v1"
DEFAULT_SHARD_SIZE_BYTES = 5 * 1024**3


def _identity_metadata(identity: WeightArtifactIdentity) -> tuple[Mapping[str, object], Mapping[str, object]]:
    component = identity.component.metadata.to_value()
    layout = identity.layout.metadata.to_value()
    if not isinstance(component, dict) or not isinstance(layout, dict):
        raise FinalLayoutContractError(
            "identity_incompatible",
            "final-layout BF16 identity metadata must use JSON objects",
        )
    return component, layout


def validate_final_layout_bf16_identity(
    identity: WeightArtifactIdentity,
    *,
    dit_names: Sequence[str],
    tensor_contract_digest: str,
) -> None:
    """Require the exact producer, representation, ownership, and layout ABI."""
    if identity.producer.producer_id != FINAL_LAYOUT_BF16_PRODUCER_ID:
        raise FinalLayoutContractError("identity_incompatible", "identity names a different producer")
    if identity.producer.version != FINAL_LAYOUT_BF16_VERSION:
        raise FinalLayoutContractError("identity_incompatible", "identity producer version is incompatible")
    if identity.producer.manifest_schema != FINAL_LAYOUT_BF16_MANIFEST_SCHEMA:
        raise FinalLayoutContractError("identity_incompatible", "identity manifest schema is incompatible")
    if identity.producer.restorer_schema != FINAL_LAYOUT_BF16_RESTORER_SCHEMA:
        raise FinalLayoutContractError("identity_incompatible", "identity restorer schema is incompatible")
    if identity.representation.name != FINAL_LAYOUT_BF16_REPRESENTATION:
        raise FinalLayoutContractError("identity_incompatible", "identity names a different representation")
    if identity.representation.dtype != str(torch.bfloat16):
        raise FinalLayoutContractError("identity_incompatible", "identity representation dtype is not BF16")
    if identity.component.name != "diffusion-dit" or identity.component.ownership != "complete-final-layout-tensors":
        raise FinalLayoutContractError("identity_incompatible", "identity DiT ownership is incompatible")

    component, layout = _identity_metadata(identity)
    if component.get("component_names") != list(dit_names):
        raise FinalLayoutContractError("identity_incompatible", "identity DiT component names are incompatible")
    if layout.get("tensor_contract_sha256") != tensor_contract_digest:
        raise FinalLayoutContractError("identity_incompatible", "identity tensor contract differs from the model")


def _split_shards(
    records: Sequence[RuntimeTensorTarget],
    max_shard_bytes: int,
) -> tuple[tuple[RuntimeTensorTarget, ...], ...]:
    if max_shard_bytes <= 0:
        raise ValueError("final-layout BF16 shard size must be positive")
    shards: list[tuple[RuntimeTensorTarget, ...]] = []
    current: list[RuntimeTensorTarget] = []
    current_bytes = 0
    for record in records:
        if current and current_bytes + record.nbytes > max_shard_bytes:
            shards.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += record.nbytes
    if current:
        shards.append(tuple(current))
    return tuple(shards)


class FinalLayoutBF16Producer:
    """Publish already-finalized model tensors through a store-scoped writer."""

    def __init__(
        self,
        context: FinalLayoutBF16IdentityContext,
        pipeline: nn.Module,
        dit_modules: Sequence[tuple[str, nn.Module]],
        *,
        max_shard_bytes: int = DEFAULT_SHARD_SIZE_BYTES,
    ) -> None:
        if max_shard_bytes <= 0:
            raise ValueError("final-layout BF16 shard size must be positive")
        self._context = context
        self._pipeline = pipeline
        self._dit_modules = tuple(dit_modules)
        self._max_shard_bytes = max_shard_bytes
        if tuple(name for name, _ in self._dit_modules) != context.dit_names:
            raise ValueError("producer DiT components differ from the exact identity context")
        if type(pipeline) is not context.pipeline_type or tuple(type(module) for _, module in self._dit_modules) != (
            context.dit_types
        ):
            raise ValueError("producer model implementation differs from the exact identity context")
        self._spec = WeightProductionSpec(
            producer_id=FINAL_LAYOUT_BF16_PRODUCER_ID,
            outputs=(context.identity,),
            source_mode=ProductionSourceMode.FINALIZED_MODEL,
            coordination_scope=CoordinationScope.SINGLE_PROCESS,
            lookup_phase=LookupPhase.POST_LOAD_ONLY,
        )

    @property
    def spec(self) -> WeightProductionSpec:
        return self._spec

    def produce(self, writer: ArtifactWriter) -> ProductionMetadata:
        self._context.ensure_sources_unchanged()
        validators = validate_final_layout_model_contract(self._dit_modules)
        records = collect_final_layout_targets(
            self._pipeline,
            self._dit_modules,
            require_materialized=True,
        )
        if frozenset(record.name for record in records) != self._context.tensor_names:
            raise FinalLayoutContractError(
                "tensor_contract_changed",
                "finalized DiT tensor ownership differs from the pre-load identity",
            )
        contract_digest = tensor_contract_sha256(records)
        validate_final_layout_bf16_identity(
            self._context.identity,
            dit_names=self._context.dit_names,
            tensor_contract_digest=contract_digest,
        )
        for validator in validators:
            validator()

        shards = _split_shards(records, self._max_shard_bytes)
        shard_count = len(shards)
        for index, shard in enumerate(shards, start=1):
            specs = tuple(
                TensorWriteSpec(
                    name=record.name,
                    shape=tuple(record.tensor.shape),
                    dtype=record.tensor.dtype,
                    kind=record.kind,
                    role=record.role,
                )
                for record in shard
            )
            file_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            with writer.open_tensor_file(file_name, specs) as output:
                for record in shard:
                    output.write_tensor(record.name, record.tensor.detach())

        self._context.ensure_sources_unchanged()
        return ProductionMetadata(
            producer_schema=FINAL_LAYOUT_BF16_MANIFEST_SCHEMA,
            restorer_schema=FINAL_LAYOUT_BF16_RESTORER_SCHEMA,
            format_metadata=CanonicalJson.from_value(
                {
                    "component_names": list(self._context.dit_names),
                    "format": FINAL_LAYOUT_BF16_REPRESENTATION,
                    "mixed_precision_policy": "bf16-with-preserved-fp32",
                    "tensor_contract_sha256": contract_digest,
                    "tensor_count": len(records),
                    "tensor_layout": "contiguous",
                }
            ),
        )


__all__ = [
    "DEFAULT_SHARD_SIZE_BYTES",
    "FINAL_LAYOUT_BF16_MANIFEST_SCHEMA",
    "FINAL_LAYOUT_BF16_PRODUCER_ID",
    "FINAL_LAYOUT_BF16_REPRESENTATION",
    "FINAL_LAYOUT_BF16_RESTORER_SCHEMA",
    "FINAL_LAYOUT_BF16_VERSION",
    "FinalLayoutBF16Producer",
    "validate_final_layout_bf16_identity",
]
