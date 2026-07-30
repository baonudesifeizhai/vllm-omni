from dataclasses import dataclass, field

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request

from vllm_omni.engine import AdditionalInformationPayload


@dataclass
class OmniNewRequestData(NewRequestData):
    """New request data for omni models with embeddings support.

    Extends NewRequestData to include additional information for direct
    transfer between pipeline stages.

    Note: prompt_embeds is inherited from NewRequestData
    (torch.Tensor | None).

    Args:
        external_req_id: Optional external request ID for tracking
        additional_information: Optional serialized additional information
            dictionary containing tensors or lists
    """

    external_req_id: str | None = None
    additional_information: AdditionalInformationPayload | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "OmniNewRequestData":
        """Create OmniNewRequestData from a Request object.

        Args:
            request: Request object to convert
            block_ids: Tuple of block ID lists for KV cache allocation
            prefill_token_ids: Optional prefill token IDs for v2 model runner

        Returns:
            OmniNewRequestData instance with data from the request
        """
        return cls(
            req_id=request.request_id,
            external_req_id=getattr(request, "external_req_id", None),
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            prompt_is_token_ids=getattr(request, "prompt_is_token_ids", None),
            prefill_token_ids=prefill_token_ids,
            additional_information=getattr(request, "additional_information", None),
        )


@dataclass
class OmniCachedRequestData(CachedRequestData):
    """Cached request data for omni models with embeddings support.

    Args:
        prompt_token_ids: Mapping from request ID to list of prompt token IDs
    """

    prompt_token_ids: dict[str, list[int]]
    additional_information: dict[str, dict | None]


@dataclass
class OmniChunkRecvHandle:
    """Minimal identifier carried from scheduler to runner for chunk-recv
    registration.

    The runner's ``register_chunk_recv`` only consumes ``request_id`` and
    ``external_req_id`` from each pending request, so we ship just those
    two fields instead of the full Request object.  Concrete typing
    keeps msgspec serialization deterministic across IPC (default,
    PD-disagg, multi-node executor variants) and avoids the
    ``list[Any]`` fallback path.
    """

    request_id: str
    external_req_id: str | None = None


@dataclass
class OmniSchedulerOutput(SchedulerOutput):
    """Scheduler output with omni-specific transfer metadata."""

    finished_requests_needing_kv_transfer: dict[str, dict] = field(default_factory=dict)
    pending_input_registrations: list[OmniChunkRecvHandle] = field(default_factory=list)
    # The upstream scheduler exposes only one speculative depth for the whole
    # batch.  Omni's runtime policy is request-scoped, so carry the ragged
    # depths to the DSpark proposer explicitly.  The scalar inherited from
    # SchedulerOutput remains the maximum depth in this batch and therefore
    # still sizes the rectangular output buffers used by vLLM.
    num_spec_tokens_to_schedule_by_request: dict[str, int] = field(default_factory=dict)
    # Observation-only scheduler round-trip marker shared by every stage.
    # Unlike the speculative fields below, this also covers Talker and
    # Code2Wav so shared-device blocking can be measured without synchronizing
    # CUDA inside the model runner.
    runtime_action_id: int = -1
    runtime_action_start_ns: int = 0
    # Observation-only identifiers for one scheduler -> model runner ->
    # scheduler round trip.  ``verify_k`` describes draft tokens consumed by
    # this call, while ``proposal_k`` describes drafts produced for the next
    # call; both are needed to measure a depth transition without shifting it
    # by one decoding step.
    speculative_action_id: int = -1
    speculative_action_start_ns: int = 0
    speculative_verify_k_by_request: dict[str, int] = field(default_factory=dict)
    speculative_proposal_k_by_request: dict[str, int] = field(default_factory=dict)
