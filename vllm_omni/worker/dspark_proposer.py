"""DSpark proposer compatibility for vLLM-Omni's legacy AR runner."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import torch
from typing_extensions import override
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    next_power_of_2,
)

logger = init_logger(__name__)


@triton.jit
def _copy_and_expand_dspark_ragged_inputs_kernel(
    next_token_ids_ptr,
    target_positions_ptr,
    request_depths_ptr,
    request_offsets_ptr,
    out_input_ids_ptr,
    out_context_positions_ptr,
    out_query_positions_ptr,
    out_context_slot_mapping_ptr,
    out_query_slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    query_start_loc_ptr,
    num_rejected_tokens_ptr,
    parallel_drafting_token_id,
    block_size,
    total_input_tokens,
    BLOCK_SIZE: tl.constexpr,  # noqa: N803
    HAS_NUM_REJECTED: tl.constexpr = False,  # noqa: N803
):
    """Fused DSpark input expansion for request-specific draft depths."""
    req_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    ctx_start = tl.load(query_start_loc_ptr + req_idx)
    ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    request_depth = tl.load(request_depths_ptr + req_idx)
    request_offset = tl.load(request_offsets_ptr + req_idx)
    total_tokens = num_ctx + request_depth

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = j < total_tokens
    is_ctx = j < num_ctx
    is_query = (~is_ctx) & in_bounds
    query_off = j - num_ctx

    ctx_pos_idx = tl.minimum(ctx_start + j, total_input_tokens - 1)
    ctx_pos = tl.load(
        target_positions_ptr + ctx_pos_idx,
        mask=is_ctx,
        other=0,
    )
    if HAS_NUM_REJECTED:
        valid_ctx_end = ctx_end - tl.load(num_rejected_tokens_ptr + req_idx)
    else:
        valid_ctx_end = ctx_end
    last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_pos = last_pos + 1 + query_off
    position = tl.where(is_ctx, ctx_pos, query_pos)

    ctx_out = ctx_start + j
    query_out = request_offset + query_off
    tl.store(out_context_positions_ptr + ctx_out, ctx_pos, mask=is_ctx)
    tl.store(out_query_positions_ptr + query_out, query_pos, mask=is_query)

    block_num = tl.minimum(position // block_size, block_table_stride - 1)
    block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_num,
        mask=in_bounds,
        other=0,
    ).to(tl.int64)
    slot = block_id * block_size + position % block_size
    tl.store(out_context_slot_mapping_ptr + ctx_out, slot, mask=is_ctx)
    tl.store(out_query_slot_mapping_ptr + query_out, slot, mask=is_query)

    anchor_token = tl.load(next_token_ids_ptr + req_idx)
    input_id = tl.where(
        query_off == 0,
        anchor_token,
        parallel_drafting_token_id,
    )
    tl.store(out_input_ids_ptr + query_out, input_id, mask=is_query)


class OmniDSparkProposer(DFlashProposer):
    """Run a Speculators-format DSpark checkpoint in the legacy AR runner.

    vLLM's newer GPU worker has a native ``DSparkSpeculator``.  Omni's AR
    runner still derives from the legacy runner, where ``dspark`` currently
    falls through to ``EagleProposer``.  This adapter keeps the DFlash
    parallel backbone and adds DSpark's anchor layout and sequential Markov
    sampling.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self._omni_dspark = speculative_config.method == "dspark"
        if not self._omni_dspark:
            super().__init__(vllm_config, device, runner)
            return

        # DFlashProposer validates the method in its constructor. DSpark uses
        # the same parallel backbone, so expose that method only while the
        # shared buffers/model are initialized, then restore the public config.
        speculative_config.method = "dflash"
        try:
            super().__init__(vllm_config, device, runner)
        finally:
            speculative_config.method = "dspark"

        # Keep the base propose path in DFlash mode: Qwen3DSparkForCausalLM is
        # a DFlashQwen3ForCausalLM subclass and returns one hidden-state tensor.
        self.method = "dflash"
        self.sample_from_anchor = not bool(getattr(self.draft_model_config.hf_config, "dspark_bonus_anchor", False))
        self._dspark_prev_tokens = torch.zeros(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        self._request_depths: list[int] | None = None
        self._request_depth_offsets: list[int] | None = None
        self._ragged_layout_cache: OrderedDict[tuple[int, ...], dict] = OrderedDict()
        self._ragged_active_indices: list[torch.Tensor] | None = None
        self._ragged_row_indices: list[torch.Tensor] | None = None
        self._ragged_depths_logged = False

    def set_request_depths(self, depths: list[int] | None) -> None:
        """Set the request-major speculative depths for the next proposal.

        vLLM's public proposer API has one scalar K.  DSpark is parallel, so
        it can instead execute a ragged collection of query rows and scatter
        the results back into the rectangular buffer expected upstream.
        """
        self._request_depths = None if depths is None else [int(k) for k in depths]
        self._request_depth_offsets = None
        self._ragged_active_indices = None
        self._ragged_row_indices = None

    def _get_ragged_layout(self, request_depths: list[int]) -> dict:
        """Return small reusable GPU tensors for a request-depth pattern."""
        key = tuple(request_depths)
        cached = self._ragged_layout_cache.get(key)
        if cached is not None:
            self._ragged_layout_cache.move_to_end(key)
            return cached

        offsets = [0]
        for depth in request_depths:
            offsets.append(offsets[-1] + depth)
        active_indices: list[torch.Tensor] = []
        row_indices: list[torch.Tensor] = []
        for step in range(max(request_depths)):
            active_cpu = [req_idx for req_idx, depth in enumerate(request_depths) if depth > step]
            active_indices.append(torch.tensor(active_cpu, dtype=torch.long, device=self.device))
            row_indices.append(
                torch.tensor(
                    [offsets[req_idx] + step for req_idx in active_cpu],
                    dtype=torch.long,
                    device=self.device,
                )
            )

        cached = {
            "offsets": offsets,
            "depths_gpu": torch.tensor(
                request_depths,
                dtype=torch.int32,
                device=self.device,
            ),
            "offsets_gpu": torch.tensor(
                offsets,
                dtype=torch.int32,
                device=self.device,
            ),
            "offsets_cpu": torch.tensor(offsets, dtype=torch.int32),
            "active_indices": active_indices,
            "row_indices": row_indices,
        }
        self._ragged_layout_cache[key] = cached
        # Depth patterns are tiny, but an unbounded cache would retain CUDA
        # allocations for the lifetime of a busy server.
        if len(self._ragged_layout_cache) > 128:
            self._ragged_layout_cache.popitem(last=False)
        return cached

    @override
    def load_model(self, target_model) -> None:
        if self._omni_dspark:
            target_config = target_model.config
            if not hasattr(target_config, "image_token_index"):
                thinker_config = getattr(target_config, "thinker_config", None)
                image_token_id = getattr(thinker_config, "image_token_id", None)
                if image_token_id is None:
                    image_token_id = getattr(target_config, "image_token_id", None)
                if image_token_id is None:
                    raise ValueError("Qwen3-Omni DSpark requires an image token ID on the target or thinker config.")
                # The shared vLLM proposer still uses the older
                # ``image_token_index`` spelling. Qwen3-Omni stores the same
                # value under ``thinker_config.image_token_id``.
                target_config.image_token_index = image_token_id
        super().load_model(target_model)

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        if not self._omni_dspark:
            return super().set_inputs_first_pass(
                target_token_ids,
                next_token_ids,
                target_positions,
                target_hidden_states,
                token_indices_to_sample,
                cad,
                num_rejected_tokens_gpu,
            )

        batch_size = cad.batch_size()
        self._dspark_prev_tokens[:batch_size].copy_(next_token_ids)
        if not self.sample_from_anchor:
            return super().set_inputs_first_pass(
                target_token_ids,
                next_token_ids,
                target_positions,
                target_hidden_states,
                token_indices_to_sample,
                cad,
                num_rejected_tokens_gpu,
            )

        request_depths = self._request_depths
        if request_depths is not None:
            if len(request_depths) < batch_size:
                raise ValueError(f"Missing per-request DSpark depths: depths={len(request_depths)}, batch={batch_size}")
            request_depths = request_depths[:batch_size]
            if any(k < 1 or k > self.num_speculative_tokens for k in request_depths):
                raise ValueError(
                    f"Per-request DSpark depths must be in [1, {self.num_speculative_tokens}]: {request_depths}"
                )
            if len(set(request_depths)) > 1:
                return self._set_inputs_first_pass_ragged(
                    target_token_ids=target_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=target_hidden_states,
                    cad=cad,
                    num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                    request_depths=request_depths,
                )

        # The trained layout is [anchor, mask, ...], and all K query slots
        # predict a draft token. DFlash's default is [bonus, mask x K], where
        # only the masks predict, hence K+1 slots.
        num_context = target_token_ids.shape[0]
        num_query_per_req = self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req
        self._dflash_num_context = num_context
        self._dflash_hidden_states = target_hidden_states

        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        block_size_tokens = min(256, next_power_of_2(max_tokens_per_req))
        num_blocks = (max_tokens_per_req + block_size_tokens - 1) // block_size_tokens
        grid = (batch_size, num_blocks)

        # The draft was trained with ordinary 1D sequence positions. The
        # verifier may supply 3D M-RoPE positions; generated text uses the
        # temporal row as the equivalent 1D position sequence.
        target_positions_1d = target_positions[0] if target_positions.ndim == 2 else target_positions
        unused_dflash_sample_indices = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dflash_inputs_kernel[grid](
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions_1d,
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=unused_dflash_sample_indices,
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=block_size_tokens,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        # Query rows are request-major and every row is sampled in the
        # anchor-first DSpark layout.
        token_indices_to_sample = self.arange[:num_query_total]
        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu
        new_seq_lens_cpu_upper_bound = (
            cad.seq_lens_cpu_upper_bound + num_query_per_req if cad.seq_lens_cpu_upper_bound is not None else None
        )
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=query_slot_mapping,
            causal=self.dflash_causal,
        )
        return num_query_total, token_indices_to_sample, new_cad

    def _set_inputs_first_pass_ragged(
        self,
        *,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        request_depths: list[int],
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        """Build DSpark query metadata for request-specific K values."""
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        layout = self._get_ragged_layout(request_depths)
        offsets = layout["offsets"]
        num_query_total = offsets[-1]
        if not self._ragged_depths_logged:
            logger.info(
                "DSpark ragged request depths enabled: batch=%d query_rows=%d rectangular_rows=%d depths=%s",
                batch_size,
                num_query_total,
                batch_size * self.num_speculative_tokens,
                tuple(request_depths),
            )
            self._ragged_depths_logged = True
        self._request_depth_offsets = offsets
        self._ragged_active_indices = layout["active_indices"]
        self._ragged_row_indices = layout["row_indices"]
        self._dflash_num_context = num_context
        self._dflash_hidden_states = target_hidden_states

        depths = layout["depths_gpu"]
        offsets_gpu = layout["offsets_gpu"]
        target_positions_1d = target_positions[0] if target_positions.ndim == 2 else target_positions
        max_tokens_per_req = cad.max_query_len + max(request_depths)
        block_size_tokens = min(256, next_power_of_2(max_tokens_per_req))
        num_blocks = (max_tokens_per_req + block_size_tokens - 1) // block_size_tokens
        has_num_rejected = num_rejected_tokens_gpu is not None
        _copy_and_expand_dspark_ragged_inputs_kernel[(batch_size, num_blocks)](
            next_token_ids_ptr=self._dspark_prev_tokens,
            target_positions_ptr=target_positions_1d,
            request_depths_ptr=depths,
            request_offsets_ptr=offsets_gpu,
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            total_input_tokens=num_context,
            BLOCK_SIZE=block_size_tokens,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        effective_seq_lens = cad.seq_lens
        if num_rejected_tokens_gpu is not None:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu
        new_query_start_loc = offsets_gpu
        new_query_start_loc_cpu = layout["offsets_cpu"]
        new_seq_lens_cpu_upper_bound = (
            cad.seq_lens_cpu_upper_bound + max(request_depths) if cad.seq_lens_cpu_upper_bound is not None else None
        )
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + depths,
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=max(request_depths),
            max_seq_len=cad.max_seq_len + max(request_depths),
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=self._slot_mapping_buffer[:num_query_total],
            causal=self.dflash_causal,
        )
        token_indices_to_sample = self.arange[:num_query_total]
        return num_query_total, token_indices_to_sample, new_cad

    @override
    def _sample_draft_tokens(self, hidden_states, sampling_metadata):
        if not self._omni_dspark:
            return super()._sample_draft_tokens(hidden_states, sampling_metadata)

        num_steps = self.num_speculative_tokens
        if num_steps == 0:
            return hidden_states.new_empty((0,), dtype=torch.long), None
        request_depths = self._request_depths
        if request_depths is not None and self._request_depth_offsets is not None and len(set(request_depths)) > 1:
            return self._sample_ragged_draft_tokens(
                hidden_states,
                sampling_metadata,
                request_depths,
                self._request_depth_offsets,
            )
        if hidden_states.shape[0] % num_steps:
            raise ValueError(
                "DSpark hidden-state rows must contain complete draft blocks: "
                f"rows={hidden_states.shape[0]}, block={num_steps}"
            )

        num_reqs = hidden_states.shape[0] // num_steps
        base_logits = self.model.compute_draft_logits(hidden_states)
        base_logits = base_logits.view(num_reqs, num_steps, -1)
        prev = self._dspark_prev_tokens[:num_reqs].long()
        draft_tokens: list[torch.Tensor] = []
        draft_probs: list[torch.Tensor] = []

        for step in range(num_steps):
            markov_embed = self.model.markov_embed(prev)
            logits = base_logits[:, step] + self.model.markov_bias(markov_embed)
            sampled, probs = self._sample_from_logits(logits, sampling_metadata)
            sampled = self.model.map_draft_to_target(sampled)
            draft_tokens.append(sampled)
            prev = sampled

            if probs is not None:
                if self.model.draft_id_to_target_id is not None:
                    target_probs = probs.new_zeros((num_reqs, self.vocab_size))
                    draft_ids = torch.arange(probs.shape[-1], device=probs.device, dtype=torch.long)
                    target_ids = self.model.map_draft_to_target(draft_ids)
                    target_probs.index_copy_(1, target_ids, probs)
                    probs = target_probs
                draft_probs.append(probs)

        tokens = torch.stack(draft_tokens, dim=1).reshape(-1)
        if not draft_probs:
            return tokens, None
        probs = torch.stack(draft_probs, dim=1).reshape(num_reqs * num_steps, -1)
        return tokens, probs

    def _sample_ragged_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
        request_depths: list[int],
        offsets: list[int],
    ):
        """Markov-sample ragged DSpark rows and scatter to [B, max_K]."""
        batch_size = len(request_depths)
        max_depth = self.num_speculative_tokens
        if hidden_states.shape[0] != offsets[-1]:
            raise ValueError(f"Ragged DSpark row count mismatch: rows={hidden_states.shape[0]}, expected={offsets[-1]}")

        base_logits = self.model.compute_draft_logits(hidden_states)
        prev = self._dspark_prev_tokens[:batch_size].long().clone()
        tokens = torch.zeros(
            (batch_size, max_depth),
            dtype=torch.long,
            device=hidden_states.device,
        )
        probs_out = None

        active_indices = self._ragged_active_indices
        row_indices = self._ragged_row_indices
        if active_indices is None or row_indices is None:
            raise RuntimeError("Missing cached DSpark ragged sampling layout")

        for step, (active, rows) in enumerate(zip(active_indices, row_indices, strict=True)):
            if active.numel() == 0:
                break
            markov_embed = self.model.markov_embed(prev[active])
            logits = base_logits[rows] + self.model.markov_bias(markov_embed)
            step_sampling_metadata = sampling_metadata
            if sampling_metadata.temperature is not None:
                step_sampling_metadata = replace(
                    sampling_metadata,
                    temperature=sampling_metadata.temperature[active],
                )
            sampled, probs = self._sample_from_logits(
                logits,
                step_sampling_metadata,
            )
            sampled = self.model.map_draft_to_target(sampled)
            tokens[active, step] = sampled
            prev[active] = sampled

            if probs is not None:
                if self.model.draft_id_to_target_id is not None:
                    target_probs = probs.new_zeros((active.numel(), self.vocab_size))
                    draft_ids = torch.arange(
                        probs.shape[-1],
                        device=probs.device,
                        dtype=torch.long,
                    )
                    target_ids = self.model.map_draft_to_target(draft_ids)
                    target_probs.index_copy_(1, target_ids, probs)
                    probs = target_probs
                if probs_out is None:
                    probs_out = probs.new_zeros((batch_size, max_depth, probs.shape[-1]))
                probs_out[active, step] = probs

        flat_tokens = tokens.reshape(-1)
        if probs_out is None:
            return flat_tokens, None
        return flat_tokens, probs_out.reshape(
            batch_size * max_depth,
            probs_out.shape[-1],
        )


def install_legacy_dspark_proposer_patch() -> None:
    """Route legacy-runner DSpark requests through ``OmniDSparkProposer``."""
    from vllm.config.speculative import SpeculativeConfig
    from vllm.v1.worker import gpu_model_runner

    original_use_dflash = SpeculativeConfig.use_dflash
    if getattr(original_use_dflash, "_omni_dspark_patched", False):
        return

    def use_dflash_or_dspark(self) -> bool:
        return original_use_dflash(self) or self.method == "dspark"

    use_dflash_or_dspark._omni_dspark_patched = True
    SpeculativeConfig.use_dflash = use_dflash_or_dspark
    # The legacy runner selects this module global during __init__. The adapter
    # is also a DFlashProposer subclass, so existing isinstance checks remain valid.
    gpu_model_runner.DFlashProposer = OmniDSparkProposer
