# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from vllm_omni.data_entry_keys import MetaStruct, OmniPayloadStruct, unflatten_payload
from vllm_omni.runtime_observability import (
    emit_runtime_event,
    resolve_runtime_request_id,
    runtime_observability_enabled,
)

from ..adapter import construct_next_stage_streaming_input_prompt
from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec
from ..utils.logging import get_connector_logger
from .base import OmniTransferAdapterBase

logger = get_connector_logger(__name__)


@dataclass
class _TalkerTextCreditState:
    """Scheduler-side shadow of the Qwen3-Omni Talker text FIFO."""

    enqueued_end: int = 0
    available: int = 0
    windows: int = 0
    bootstrap_rows: int = 0
    enqueued_rows: int = 0
    consumed_rows: int = 0
    duplicate_rows: int = 0
    max_credit: int = 0
    wait_steps: int = 0
    wait_time_s: float = 0.0
    wait_started_s: float | None = None
    upstream_finished: bool = False
    summary_emitted: bool = False
    observation_request_id: str | None = None


@dataclass
class _Code2WavWorkState:
    """Observable Code2Wav work queued by Talker codec chunks."""

    pending_chunks: deque[tuple[int, int]] = field(default_factory=deque)
    pending_total_frames: int = 0
    pending_new_frames: int = 0
    enqueued_chunks: int = 0
    scheduled_chunks: int = 0
    enqueued_total_frames: int = 0
    enqueued_new_frames: int = 0
    scheduled_total_frames: int = 0
    scheduled_new_frames: int = 0
    max_pending_total_frames: int = 0
    summary_emitted: bool = False
    observation_request_id: str | None = None


class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    """Chunk-level transfer adapter for Omni connector pipelines.

    This class coordinates per-request chunk exchange between adjacent stages,
    and implements asynchronous get/put of chunks via background threads.
    It tracks per-request chunk indices for put/get, and accumulates
    payloads across chunks (concatenating tensors/lists in AR mode). It also
    caches prompt token ids and additional information for scheduler use.

    Scheduler integration is handled via WAITING_FOR_CHUNK transitions:
    requests are moved to waiting for chunk deque while polling, then restored
    to waiting/running queues once a chunk arrives. The requests will finish
    loading chunk util detecting the payload "finished" flag.

    The base class owns background recv/save loops; load/save only enqueue
    work and return immediately.
    """

    def __init__(self, vllm_config: Any):
        model_config = vllm_config.model_config
        self.scheduler_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        active_stream_window = int(getattr(model_config, "active_stream_window", 0) or 0)
        model_max_num_seqs = int(getattr(model_config, "max_num_seqs", self.scheduler_max_num_seqs) or 0)
        if model_max_num_seqs <= 0:
            model_max_num_seqs = self.scheduler_max_num_seqs
        self._active_window = min(active_stream_window, model_max_num_seqs) if active_stream_window > 0 else 0
        if self._active_window > 0:
            logger.info(
                "Bounded active-stream window enabled: K=%d. "
                "Multi-replica deployments require sticky per-stream routing across Stage 1 "
                "replicas (each replica owns an independent active-set; without sticky routing, "
                "a stream can be active on one replica and non-active on another and both will "
                "race to evict it).",
                self._active_window,
            )
        self.connector = self.create_connector(model_config)
        super().__init__(model_config)
        self.model_mode = getattr(model_config, "worker_type", None) or "ar"
        self._talker_text_credit_enabled = (
            bool(getattr(model_config, "async_chunk", False))
            and getattr(model_config, "model_arch", None) == "Qwen3OmniMoeForConditionalGeneration"
            and getattr(model_config, "model_stage", None) == "talker"
        )
        self._talker_text_credits: dict[str, _TalkerTextCreditState] = {}
        if self._talker_text_credit_enabled:
            logger.info("Qwen3-Omni Talker verified-text FIFO credit scheduling enabled.")
        self._code2wav_work_enabled = (
            runtime_observability_enabled()
            and getattr(model_config, "model_arch", None) == "Qwen3OmniMoeForConditionalGeneration"
            and getattr(model_config, "model_stage", None) == "code2wav"
        )
        self._code2wav_work_states: dict[str, _Code2WavWorkState] = {}
        # State specific to Chunk management
        self.custom_process_next_stage_input_func: Callable[..., OmniPayloadStruct | None] | None = None
        custom_process_next_stage_input_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if custom_process_next_stage_input_func:
            module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_next_stage_input_func = getattr(module, func_name)
        # mapping for request id and chunk id
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.segment_finished_requests: set[str] = set()
        self.request_payload = {}
        self.code_prompt_token_ids: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()
        self.requests_origin_status = {}
        self._active_streams: dict[str, Any] = {}
        # Private hold-queue for non-active running requests. Restored to
        # running_queue inside restore_queues(). Avoids calling
        # waiting_queue.prepend_requests mid-step, which trips vllm's
        # per-step LogitsProcessor invariant
        # ("Cannot register new removed request after self.removed has
        #   been read").
        self._held_non_active: deque[Any] = deque()
        self.requests_num_chunks_sent: dict[str, int] = defaultdict(int)
        self._pending_streaming_prefills: dict[str, dict] = {}

    @staticmethod
    def _is_truthy_scalar(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return value.numel() == 1 and bool(value.item())
        return bool(value) if value is not None else False

    def _talker_text_credit_state(self, request_id: str) -> _TalkerTextCreditState | None:
        if not getattr(self, "_talker_text_credit_enabled", False):
            return None
        states = getattr(self, "_talker_text_credits", None)
        if states is None:
            states = {}
            self._talker_text_credits = states
        return states.setdefault(request_id, _TalkerTextCreditState())

    def _register_talker_text_window(self, request_id: str, payload: Mapping[str, Any]) -> None:
        """Turn a verified Thinker text span into Talker scheduling credit."""
        state = self._talker_text_credit_state(request_id)
        if state is None:
            return
        state.observation_request_id = resolve_runtime_request_id(
            self.request_ids_mapping.get(request_id, request_id),
            payload,
        )
        observation_request_id = state.observation_request_id

        now = time.monotonic()
        if state.wait_started_s is not None:
            wait_s = now - state.wait_started_s
            state.wait_time_s += wait_s
            state.wait_started_s = None
            emit_runtime_event(
                "talker_text_wait_end",
                request_id=observation_request_id,
                stage="talker",
                unit="seconds",
                delta=wait_s,
                inventory=state.available,
            )

        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            state.upstream_finished = state.upstream_finished or self._is_truthy_scalar(meta.get("finished"))

        embed = payload.get("embed")
        ids = payload.get("ids")
        if (
            state.bootstrap_rows == 0
            and isinstance(embed, Mapping)
            and embed.get("prefill") is not None
            and isinstance(ids, Mapping)
            and bool(ids.get("output"))
        ):
            # The first accepted text token is consumed by Talker prefill, not
            # by a cached decode step, so it is traced but not granted as credit.
            state.bootstrap_rows = 1
            emit_runtime_event(
                "talker_text_bootstrap",
                request_id=observation_request_id,
                stage="talker",
                unit="text_rows",
                delta=1,
                inventory=state.available,
            )

        if not isinstance(embed, Mapping):
            return
        start = embed.get("decode_token_start")
        end = embed.get("decode_token_end")
        if start is None or end is None:
            return
        start = int(start)
        end = int(end)
        if start < 0 or end < start:
            logger.error(
                "[TalkerCreditTrace] invalid verified text span request=%s span=[%d,%d)",
                request_id,
                start,
                end,
            )
            return

        state.windows += 1
        if start > state.enqueued_end:
            logger.error(
                "[TalkerCreditTrace] gap in verified text spans request=%s expected_start=%d actual_start=%d end=%d",
                request_id,
                state.enqueued_end,
                start,
                end,
            )
            return

        span_rows = end - start
        new_start = max(start, state.enqueued_end)
        new_rows = max(0, end - new_start)
        state.duplicate_rows += span_rows - new_rows
        if new_rows:
            state.enqueued_end = end
            state.enqueued_rows += new_rows
            state.available += new_rows
            state.max_credit = max(state.max_credit, state.available)
            emit_runtime_event(
                "talker_text_enqueue",
                request_id=observation_request_id,
                stage="talker",
                unit="text_rows",
                delta=new_rows,
                inventory=state.available,
                span_start=new_start,
                span_end=end,
            )

    def _has_talker_text_credit(self, request_id: str) -> bool:
        if not getattr(self, "_talker_text_credit_enabled", False):
            return False
        state = getattr(self, "_talker_text_credits", {}).get(request_id)
        return state is not None and state.available > 0

    def _begin_talker_text_wait(self, request_id: str) -> None:
        state = self._talker_text_credit_state(request_id)
        if state is None or state.upstream_finished or state.wait_started_s is not None:
            return
        state.wait_steps += 1
        state.wait_started_s = time.monotonic()

    def _consume_talker_text_credit(self, request: Request, num_scheduled_tokens: int) -> None:
        state = self._talker_text_credit_state(request.request_id)
        if state is None or state.available <= 0:
            return

        # postprocess_scheduler_output runs after vLLM advances
        # num_computed_tokens. Only the part strictly beyond the Talker prompt
        # invokes decode and consumes one text FIFO row per scheduled token.
        scheduled_end = int(request.num_computed_tokens)
        scheduled_start = max(0, scheduled_end - int(num_scheduled_tokens))
        prompt_end = int(request.num_prompt_tokens)
        decode_steps = max(0, scheduled_end - max(scheduled_start, prompt_end))
        if decode_steps <= 0:
            return
        consumed = min(decode_steps, state.available)
        state.available -= consumed
        state.consumed_rows += consumed
        emit_runtime_event(
            "talker_text_consume",
            request_id=state.observation_request_id
            or self.request_ids_mapping.get(request.request_id, request.request_id),
            stage="talker",
            unit="text_rows",
            delta=-consumed,
            inventory=state.available,
            scheduled_decode_steps=decode_steps,
        )
        if consumed != decode_steps:
            logger.error(
                "[TalkerCreditTrace] scheduler over-consumed text credit request=%s requested=%d available=%d",
                request.request_id,
                decode_steps,
                consumed,
            )

    def _log_talker_text_credit_summary(self, request_id: str) -> None:
        state = getattr(self, "_talker_text_credits", {}).get(request_id)
        if state is None or state.summary_emitted:
            return
        if state.wait_started_s is not None:
            state.wait_time_s += time.monotonic() - state.wait_started_s
            state.wait_started_s = None
        state.summary_emitted = True
        logger.info(
            "[TalkerCreditTrace] request=%s windows=%d accepted_rows=%d bootstrap_rows=%d "
            "enqueued_rows=%d consumed_rows=%d duplicate_rows=%d max_credit=%d "
            "scheduler_wait_events=%d scheduler_wait_ms=%.3f remaining_credit=%d upstream_finished=%s",
            request_id,
            state.windows,
            state.bootstrap_rows + state.enqueued_rows,
            state.bootstrap_rows,
            state.enqueued_rows,
            state.consumed_rows,
            state.duplicate_rows,
            state.max_credit,
            state.wait_steps,
            state.wait_time_s * 1000.0,
            state.available,
            state.upstream_finished,
        )
        emit_runtime_event(
            "talker_text_summary",
            request_id=state.observation_request_id or self.request_ids_mapping.get(request_id, request_id),
            stage="talker",
            unit="text_rows",
            inventory=state.available,
            windows=state.windows,
            bootstrap_rows=state.bootstrap_rows,
            enqueued_rows=state.enqueued_rows,
            consumed_rows=state.consumed_rows,
            duplicate_rows=state.duplicate_rows,
            max_credit=state.max_credit,
            wait_events=state.wait_steps,
            wait_time_s=state.wait_time_s,
            upstream_finished=state.upstream_finished,
        )

    @staticmethod
    def _int_scalar(value: Any, default: int = 0) -> int:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return default
            return int(value.item())
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _register_code2wav_chunk(
        self,
        request_id: str,
        external_request_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Register real codec-frame work received by the Code2Wav stage."""
        if not self._code2wav_work_enabled:
            return
        codes = payload.get("codes")
        audio_codes = codes.get("audio") if isinstance(codes, Mapping) else None
        if isinstance(audio_codes, torch.Tensor):
            num_codes = int(audio_codes.numel())
        elif isinstance(audio_codes, (list, tuple)):
            num_codes = len(audio_codes)
        else:
            return
        if num_codes <= 0:
            return
        if num_codes % 16 != 0:
            logger.warning(
                "[Code2WavInventoryTrace] codec count is not divisible by 16 request=%s count=%d",
                request_id,
                num_codes,
            )
        total_frames = (num_codes + 15) // 16
        meta = payload.get("meta")
        left_context_frames = self._int_scalar(meta.get("left_context_size")) if isinstance(meta, Mapping) else 0
        left_context_frames = max(0, min(total_frames, left_context_frames))
        new_frames = total_frames - left_context_frames

        state = self._code2wav_work_states.setdefault(request_id, _Code2WavWorkState())
        state.observation_request_id = resolve_runtime_request_id(
            external_request_id,
            payload,
        )
        state.pending_chunks.append((total_frames, new_frames))
        state.pending_total_frames += total_frames
        state.pending_new_frames += new_frames
        state.enqueued_chunks += 1
        state.enqueued_total_frames += total_frames
        state.enqueued_new_frames += new_frames
        state.max_pending_total_frames = max(
            state.max_pending_total_frames,
            state.pending_total_frames,
        )
        emit_runtime_event(
            "code2wav_codec_enqueue",
            request_id=state.observation_request_id,
            stage="code2wav",
            unit="codec_frames",
            delta=total_frames,
            inventory=state.pending_total_frames,
            new_frames=new_frames,
            left_context_frames=left_context_frames,
            pending_new_frames=state.pending_new_frames,
            pending_chunks=len(state.pending_chunks),
        )

    def _schedule_code2wav_chunk(self, request_id: str) -> None:
        if not self._code2wav_work_enabled:
            return
        state = self._code2wav_work_states.get(request_id)
        if state is None or not state.pending_chunks:
            return
        total_frames, new_frames = state.pending_chunks.popleft()
        state.pending_total_frames = max(0, state.pending_total_frames - total_frames)
        state.pending_new_frames = max(0, state.pending_new_frames - new_frames)
        state.scheduled_chunks += 1
        state.scheduled_total_frames += total_frames
        state.scheduled_new_frames += new_frames
        emit_runtime_event(
            "code2wav_codec_schedule",
            request_id=state.observation_request_id or self.request_ids_mapping.get(request_id, request_id),
            stage="code2wav",
            unit="codec_frames",
            delta=-total_frames,
            inventory=state.pending_total_frames,
            new_frames=new_frames,
            pending_new_frames=state.pending_new_frames,
            pending_chunks=len(state.pending_chunks),
        )

    def _log_code2wav_work_summary(self, request_id: str) -> None:
        state = self._code2wav_work_states.get(request_id)
        if state is None or state.summary_emitted:
            return
        state.summary_emitted = True
        emit_runtime_event(
            "code2wav_codec_summary",
            request_id=state.observation_request_id or self.request_ids_mapping.get(request_id, request_id),
            stage="code2wav",
            unit="codec_frames",
            inventory=state.pending_total_frames,
            pending_new_frames=state.pending_new_frames,
            enqueued_chunks=state.enqueued_chunks,
            scheduled_chunks=state.scheduled_chunks,
            enqueued_total_frames=state.enqueued_total_frames,
            enqueued_new_frames=state.enqueued_new_frames,
            scheduled_total_frames=state.scheduled_total_frames,
            scheduled_new_frames=state.scheduled_new_frames,
            max_pending_total_frames=state.max_pending_total_frames,
        )

    @staticmethod
    def _confirmed_num_computed_tokens(request: Request) -> int:
        # vLLM async scheduling advances num_computed_tokens with output
        # placeholders before the corresponding token is committed. Connector
        # chunk send watermarks must use only committed tokens.
        num_computed = int(getattr(request, "num_computed_tokens", 0))
        num_placeholders = int(getattr(request, "num_output_placeholders", 0) or 0)
        return max(0, num_computed - num_placeholders)

    @classmethod
    def create_connector(cls, model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            connector_config = {}
        elif not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)

    def load_async(self, request: Request):
        """Register a request for asynchronous chunk retrieval.

        This method does not read from the connector directly. It records
        request metadata and enqueues the request id for the background
        receive loop to poll.

        Stage-0 has no upstream producer, so this call is a no-op there.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        self._cancelled_load_reqs.discard(request.request_id)
        self._pending_load_reqs.append(request)
        with self._recv_cond:
            self._recv_cond.notify()

    def save_async(
        self,
        multimodal_output: dict[str, Any] | None = None,
        request: Request | None = None,
        is_segment_finished: bool = False,
    ):
        """Build and enqueue one chunk for asynchronous sending.

        Payload extraction happens in ``_send_single_request`` on the
        background save_loop thread.

        For streaming input request ``is_segment_finished`` marks the end
        of the current realtime input segment. It is intentionally separate
        from ``request.is_finished()``: a resumable `/v1/realtime` session
        can finish one audio segment and later continue with another segment
        under the same external request id. For other requests, it is the same
        as ``request.is_finished()``.

        Args:
            multimodal_output: Per-request multimodal output dictionary
            request: Request object
            is_segment_finished: whether the segment of request is finished
        """
        is_finished = request.is_finished() and not request.resumable

        confirmed_num_computed_tokens = self._confirmed_num_computed_tokens(request)

        # If the request is preempted, skip the already saved chunks.
        if confirmed_num_computed_tokens < self.requests_num_chunks_sent.get(request.external_req_id, 0):
            logger.warning(
                f"Enqueue save_async for request {request.external_req_id}, "
                f"request.num_computed_tokens={request.num_computed_tokens}, "
                f"request.num_output_placeholders={getattr(request, 'num_output_placeholders', 0)}, "
                f"previous_chunks_sent={self.requests_num_chunks_sent.get(request.external_req_id, 0)}"
            )
            return

        self.requests_num_chunks_sent[request.external_req_id] = confirmed_num_computed_tokens
        task = {
            "multimodal_output": multimodal_output,
            "request": request,
            "is_finished": is_finished,
            "is_segment_finished": is_segment_finished,
        }
        self._pending_save_reqs.append(task)
        with self._save_cond:
            self._save_cond.notify()

    def _poll_single_request(self, request: Request):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        req_id = request.request_id
        chunk_id = self.get_req_chunk[req_id]
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        # Use timeout=0 for non-blocking poll
        try:
            result = self.connector.get(
                str(target_stage_id),
                str(stage_id),
                connector_get_key,
            )
        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {connector_get_key}: {e}")
            return False

        if result is None:
            return False
        payload_data, size = result

        if payload_data:
            # Update connector state
            self.get_req_chunk[req_id] += 1

            meta = payload_data.get("meta", {})
            payload_finished = self._is_truthy_scalar(meta.get("finished"))
            payload_segment_finished = self._is_truthy_scalar(meta.get("is_segment_finished"))
            if self.model_mode == "ar":
                self._register_talker_text_window(req_id, payload_data)
                request.additional_information = payload_data
                if chunk_id > 0 and request.resumable:
                    # For new streaming input segment, we should update prompt from payload
                    construct_next_stage_streaming_input_prompt(payload_data, request)

                if payload_finished:
                    self.finished_requests.add(req_id)
                    request.resumable = False
                if payload_segment_finished:
                    self.segment_finished_requests.add(req_id)
            else:
                if payload_finished:
                    self.finished_requests.add(req_id)
                    request.resumable = False
                if payload_segment_finished:
                    self.segment_finished_requests.add(req_id)

                self._register_code2wav_chunk(
                    req_id,
                    external_req_id,
                    payload_data,
                )
                new_ids = payload_data.get("codes", {}).get("audio")
                has_tensor_codes = isinstance(new_ids, torch.Tensor)
                use_tensor_codes = has_tensor_codes and new_ids.ndim >= 2
                if use_tensor_codes:
                    request.prompt_token_ids = [0] if new_ids.numel() > 0 else []
                elif has_tensor_codes:
                    new_ids = new_ids.tolist()
                elif new_ids is None:
                    new_ids = []
                    request.prompt_token_ids = new_ids
                if not use_tensor_codes:
                    request.prompt_token_ids = new_ids
                prev_info = getattr(request, "additional_information", None)
                info = dict(prev_info) if isinstance(prev_info, dict) else {}
                for key, value in payload_data.items():
                    if key == "codes":
                        if use_tensor_codes and isinstance(value, dict):
                            existing_sub = info.get(key)
                            merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                            merged_sub.update(value)
                            info[key] = merged_sub
                        continue
                    if isinstance(value, dict):
                        existing_sub = info.get(key)
                        merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                        for sk, sv in value.items():
                            if key == "meta" and sk == "finished":
                                continue
                            merged_sub[sk] = sv
                        info[key] = merged_sub
                        continue
                    info[key] = value
                request.additional_information = info
                request.num_computed_tokens = 0

                # Empty chunk with more data expected: keep polling.
                has_new_ids = bool(new_ids.numel()) if use_tensor_codes else bool(new_ids)
                if not has_new_ids and not payload_finished:
                    # The base recv loop treats False as "not ready yet" and
                    # requeues the request. Do not mark an empty non-terminal
                    # chunk as ready, otherwise Stage1 can consume before the
                    # first DAC frame arrives.
                    return False

            # Mark as finished for consumption
            self._finished_load_reqs.add(req_id)
            logger.debug(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")
            return True

        return False

    def _send_single_request(self, task: dict):
        raw_mm = task["multimodal_output"]
        multimodal_output = unflatten_payload(raw_mm) if isinstance(raw_mm, Mapping) else raw_mm
        request = task["request"]
        is_finished = task["is_finished"]
        is_segment_finished = task["is_segment_finished"]
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        external_req_id = request.external_req_id
        chunk_id = self.put_req_chunk[external_req_id]
        connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"
        # Process payload in save_loop thread
        payload_data: OmniPayloadStruct | None = None
        if self.custom_process_next_stage_input_func:
            try:
                payload_data = self.custom_process_next_stage_input_func(
                    transfer_manager=self,
                    multimodal_output=multimodal_output,
                    request=request,
                    # Existing processors use is_finished as a flush signal.
                    is_finished=is_segment_finished,
                )

            except Exception as e:
                logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")

        if payload_data is None:
            if not (is_segment_finished or is_finished):
                return
            # Segment/request finish markers must still reach downstream even when
            # the processor has no tensor payload.
            payload_data = OmniPayloadStruct()
        if payload_data.meta is None:
            payload_data.meta = MetaStruct()
        payload_data.meta.finished = torch.tensor(is_finished, dtype=torch.bool)
        payload_data.meta.is_segment_finished = torch.tensor(is_segment_finished, dtype=torch.bool)

        success, size, metadata = self.connector.put(
            from_stage=str(stage_id),
            to_stage=str(next_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )

        if success:
            self.put_req_chunk[external_req_id] += 1
            logger.debug(f"[Stage-{stage_id}] Sent {connector_put_key}")
            # Sender uses struct attr access here; the receive path in
            # `_load_one_request` / `_update_request_payload` reads dict keys.
            # That asymmetry is intentional: `OmniMsgpackDecoder` is type-erased
            # (no target type), so the wire round-trips struct -> dict. If you
            # change the schema, update both ends — see test_wire_round_trip.
            finished_flag = payload_data.meta.finished if payload_data.meta is not None else None
            is_payload_finished = False
            if isinstance(finished_flag, torch.Tensor):
                is_payload_finished = finished_flag.numel() == 1 and bool(finished_flag.item())
            elif finished_flag is not None:
                is_payload_finished = bool(finished_flag)

            # Reclaim per-request async state only after the terminal payload
            # has been sent successfully. This avoids cleanup->save races.
            if is_payload_finished:
                self.cleanup(request.request_id, external_req_id)

        if is_segment_finished:
            self.code_prompt_token_ids.pop(external_req_id, None)
            self.requests_num_chunks_sent.pop(external_req_id, None)
            cached_ic = getattr(self, "_cached_ic", None)
            if cached_ic is not None:
                cached_ic.pop(external_req_id, None)

    def is_done_receiving_chunks(self, request_id: str) -> bool:
        """Return True if the request should stop polling upstream chunks.

        Covers both the whole-request finish marker (``finished_requests``) and
        the per-segment finish marker (``segment_finished_requests``) used while
        waiting for the next streaming input slice.
        """
        return request_id in self.finished_requests or request_id in self.segment_finished_requests

    ########################################################################
    # Cleanup
    ########################################################################

    def cleanup_receiver(self, request_id: str) -> None:
        """Reclaim receiver-side per-request state (keyed by internal id).

        Safe to call from the scheduler even when ``save_async()`` has
        enqueued work that the background thread has not yet processed,
        because it only touches receiver-side dictionaries.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self._log_talker_text_credit_summary(request_id)
        self._log_code2wav_work_summary(request_id)
        getattr(self, "_talker_text_credits", {}).pop(request_id, None)
        self._code2wav_work_states.pop(request_id, None)
        if request_id in self.finished_requests:
            self._evict_finished_active_streams({request_id})
        else:
            self._active_streams.pop(request_id, None)
        self.finished_requests.discard(request_id)
        self.segment_finished_requests.discard(request_id)
        self.get_req_chunk.pop(request_id, None)
        self.requests_with_ready_chunks.discard(request_id)
        self.request_ids_mapping.pop(request_id, None)
        self.requests_origin_status.pop(request_id, None)

        self._cancelled_load_reqs.add(request_id)
        self._finished_load_reqs.discard(request_id)

    def cleanup_sender(self, external_req_id: str) -> None:
        """Reclaim sender-side per-request state (keyed by external id).

        Must only be called after the terminal chunk has actually been
        sent (i.e. from ``_send_single_request``), not before.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self.put_req_chunk.pop(external_req_id, None)
        self.request_payload.pop(external_req_id, None)
        self.code_prompt_token_ids.pop(external_req_id, None)
        self.requests_num_chunks_sent.pop(external_req_id, None)
        self._pending_streaming_prefills.pop(external_req_id, None)
        thinker_talker_states = getattr(self, "_qwen3_thinker_talker_window_states", None)
        if isinstance(thinker_talker_states, dict):
            thinker_talker_states.pop(external_req_id, None)
        talker_codec_states = getattr(self, "_qwen3_talker_codec_trace_states", None)
        if isinstance(talker_codec_states, dict):
            talker_codec_states.pop(external_req_id, None)

        cached_ic = getattr(self, "_cached_ic", None)
        if cached_ic is not None:
            cached_ic.pop(external_req_id, None)

    def cleanup(
        self,
        request_id: str,
        external_req_id: str | None = None,
    ) -> None:
        """Reclaim all per-request state after a request finishes.

        Idempotent: calling with an already-cleaned or unknown id is safe.

        Args:
            request_id: Internal request id (receive / scheduler side key).
            external_req_id: External request id (send / payload side key).
                When *None*, looked up from ``request_ids_mapping``.
        """
        if external_req_id is None:
            external_req_id = self.request_ids_mapping.get(request_id, request_id)

        self.cleanup_receiver(request_id)
        self.cleanup_sender(external_req_id)

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        *,
        scheduler_requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Process pending chunks for waiting and running queues.

        When ``scheduler_requests`` is provided, purges any
        ``waiting_for_chunk_*_requests`` deque entries whose
        ``request_id`` is no longer tracked by it (e.g. after a
        mid-flight abort that ran ``Scheduler._free_request``) before
        processing chunks. Without this purge, ``restore_queues`` would
        later re-inject the freed ``Request`` onto ``running_queue`` and
        the worker's ``_update_states`` would crash with ``KeyError``
        reading ``self.requests[req_id]``. See vllm-project/vllm-omni#3736.

        ``scheduler_requests`` is keyword-only and optional; production
        schedulers always pass their live request map, while legacy
        callers that don't track aborts may omit it to keep the prior
        (unguarded) behaviour.
        """
        if self.connector.stage_id == 0:
            return

        # Purge deque entries whose request was freed mid-flight (abort →
        # Scheduler._free_request) before any chunk processing, so neither
        # the legacy nor the active-stream path can re-inject a zombie
        # Request onto the queues. See vllm-project/vllm-omni#3736.
        if scheduler_requests is not None:
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_waiting_requests, scheduler_requests)
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_running_requests, scheduler_requests)

        if self._active_window <= 0:
            self._process_chunk_queue_legacy(
                waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
            )
            self._process_chunk_queue_legacy(
                running_queue,
                self.waiting_for_chunk_running_requests,
                RequestStatus.RUNNING,
                self._finished_load_reqs,
            )
            while len(running_queue) > self.scheduler_max_num_seqs:
                request = running_queue.pop()
                request.status = RequestStatus.PREEMPTED
                waiting_queue.prepend_requests([request])
            return

        self._promote_active_streams(running_queue)
        self._promote_active_streams(waiting_queue)
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, self._finished_load_reqs
        )
        self._promote_active_streams(waiting_queue)
        self._preempt_non_active_running(waiting_queue, running_queue)

    def _evict_finished_active_streams(self, request_ids: set[str] | None = None) -> None:
        for request_id in list(self._active_streams):
            if request_ids is not None and request_id not in request_ids:
                continue
            if request_id in self.finished_requests:
                self._active_streams.pop(request_id, None)

    def _promote_active_streams(self, queue: Any) -> None:
        if len(self._active_streams) >= self._active_window:
            return
        for request in list(queue):
            if len(self._active_streams) >= self._active_window:
                return
            request_id = request.request_id
            if request_id in self._active_streams or request_id in self.finished_requests:
                continue
            # Iterating the existing queue preserves FIFO admission.
            self._active_streams[request_id] = request

    def _ensure_active_stream(self, request: Request) -> bool:
        if self._active_window <= 0:
            return True
        request_id = request.request_id
        if request_id in self._active_streams:
            self._active_streams[request_id] = request
            return True
        if request_id in self.finished_requests or len(self._active_streams) >= self._active_window:
            return False
        self._active_streams[request_id] = request
        return True

    def _preempt_non_active_running(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        # Hold non-active running requests in a private deque rather than
        # routing them back through waiting_queue. Routing through the
        # vllm RequestQueue mid-step triggers
        #   "Cannot register new removed request after self.removed has
        #    been read"
        # in vllm.v1.sample.logits_processor.state when the persistent
        # batch was already snapshotted. They are returned to
        # running_queue in restore_queues() so the next scheduler tick
        # re-evaluates them through _promote_active_streams.
        index = len(running_queue) - 1
        while index >= 0:
            request = running_queue[index]
            if request.request_id in self._active_streams:
                index -= 1
                continue
            request = running_queue.pop(index)
            self._held_non_active.append(request)
            index -= 1

    def _process_chunk_queue_legacy(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if self._has_talker_text_credit(request.request_id):
                    # A verified Thinker window can feed several consecutive
                    # Talker decode steps. Keep scheduling until its FIFO
                    # credit is exhausted instead of waiting for another chunk.
                    continue
                if self.is_done_receiving_chunks(request.request_id):
                    request.additional_information = None
                    continue
                # Requests that waiting for chunk
                self._begin_talker_text_wait(request.request_id)
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            self.requests_origin_status[request.request_id] = target_status
            waiting_for_chunk_list.append(request)

    def _purge_untracked_chunk_requests(
        self,
        deque_list: deque[Any],
        scheduler_requests: dict[str, Request],
    ) -> None:
        """Drop deque entries whose ``request_id`` is not in
        ``scheduler_requests`` and reclaim their receiver-side state.

        Handles requests that were aborted mid-flight while parked in a
        chunk-transfer deque: ``Scheduler._free_request`` deleted the
        entry from ``scheduler.requests`` but the deque still holds a
        reference to the now-freed ``Request``. Order of survivors is
        preserved.
        """
        if not deque_list:
            return
        for _ in range(len(deque_list)):
            request = deque_list.popleft()
            if request.request_id in scheduler_requests:
                deque_list.append(request)
            else:
                self.cleanup_receiver(request.request_id)

    def restore_queues(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        scheduler_requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.

        Re-runs the zombie purge first to close the race window where an
        abort fires *between* ``process_pending_chunks`` and the
        ``finally``-clause ``restore_queues`` call. Without the second
        purge, ``running_queue.extend(...)`` would still re-inject a
        freed ``Request`` and crash the worker on the next tick.

        ``scheduler_requests`` is optional for back-compat with legacy
        callers (older tests pass only the two queue arguments). When
        provided, it gates both the deque purge and the per-request
        admit checks below; when ``None``, the purge is skipped and
        every parked request is restored unconditionally (the
        pre-purge behavior).
        """
        if scheduler_requests is not None:
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_waiting_requests, scheduler_requests)
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_running_requests, scheduler_requests)
        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            if scheduler_requests is None or request.request_id in scheduler_requests:
                waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            live_running_requests = [
                request
                for request in self.waiting_for_chunk_running_requests
                if scheduler_requests is None or request.request_id in scheduler_requests
            ]
            running_queue.extend(live_running_requests)
        self.waiting_for_chunk_running_requests = deque()

        if self._held_non_active:
            running_queue.extend(self._held_non_active)
            self._held_non_active = deque()

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return

        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)
            cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
            scheduled_counts = getattr(scheduler_output, "num_scheduled_tokens", {})
            if self._talker_text_credit_enabled and cached_reqs:
                for req_id in cached_reqs.req_ids:
                    request = requests.get(req_id) if req_id else None
                    if request is not None:
                        self._consume_talker_text_credit(
                            request,
                            int(scheduled_counts.get(req_id, 1)),
                        )
        scheduled_req_ids = self._scheduled_request_ids(scheduler_output)
        for req_id in scheduled_req_ids:
            self._schedule_code2wav_chunk(req_id)
        self._clear_chunk_ready(scheduler_output)
        if scheduled_req_ids:
            # Terminal chunks must stay active until they are scheduled once.
            self._evict_finished_active_streams(scheduled_req_ids)

    @staticmethod
    def _scheduled_request_ids(scheduler_output: Any) -> set[str]:
        req_ids: set[str] = set()
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                req_id = getattr(req_data, "req_id", None)
                if req_id:
                    req_ids.add(req_id)
        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id:
                    req_ids.add(req_id)
        return req_ids

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info
            if request and additional_info:
                request.additional_information = None

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if not self._ensure_active_stream(request):
                continue
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if self._has_talker_text_credit(request.request_id):
                    # Keep the request runnable while the worker-side Talker
                    # FIFO still has verified text rows.
                    continue
                if self.is_done_receiving_chunks(request.request_id):
                    request.additional_information = None
                    continue
                # Requests that waiting for chunk
                self._begin_talker_text_wait(request.request_id)
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            self.requests_origin_status[request.request_id] = target_status
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)

    def finish_requests(
        self, request_ids: Any, finished_status: RequestStatus, requests: dict[str, Request] | None = None
    ) -> list[tuple[str, int]]:
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = requests.keys()

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = requests.get(req_id) if requests else None
            if request is None or request.is_finished():
                # Invalid request ID.
                continue
            if req_id in self.requests_origin_status:
                request.status = self.requests_origin_status.pop(req_id)

        request_ids = set(request_ids)

        self.waiting_for_chunk_waiting_requests = deque(
            request for request in self.waiting_for_chunk_waiting_requests if request.request_id not in request_ids
        )
        self.waiting_for_chunk_running_requests = deque(
            request for request in self.waiting_for_chunk_running_requests if request.request_id not in request_ids
        )

        for req_id in request_ids:
            self._log_talker_text_credit_summary(req_id)
            self._log_code2wav_work_summary(req_id)
            getattr(self, "_talker_text_credits", {}).pop(req_id, None)
            self._code2wav_work_states.pop(req_id, None)
            self.requests_with_ready_chunks.discard(req_id)
            self.finished_requests.discard(req_id)
            self._finished_load_reqs.discard(req_id)
            self._cancelled_load_reqs.add(req_id)

        return []
