from types import SimpleNamespace

import torch
from vllm.config.speculative import SpeculativeConfig

from vllm_omni.worker.dspark_proposer import (
    OmniDSparkProposer,
    install_legacy_dspark_proposer_patch,
)


def test_dspark_load_model_adapts_qwen3_omni_image_token_name(monkeypatch):
    from vllm.v1.spec_decode.dflash import DFlashProposer

    target = SimpleNamespace(config=SimpleNamespace(thinker_config=SimpleNamespace(image_token_id=151655)))
    proposer = object.__new__(OmniDSparkProposer)
    proposer._omni_dspark = True
    seen = []
    monkeypatch.setattr(
        DFlashProposer,
        "load_model",
        lambda self, model: seen.append(model.config.image_token_index),
    )

    proposer.load_model(target)

    assert target.config.image_token_index == 151655
    assert seen == [151655]


def test_legacy_runner_routes_dspark_through_dflash_slot():
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from vllm.v1.worker import gpu_model_runner

    install_legacy_dspark_proposer_patch()
    fake = SimpleNamespace(method="dspark")
    assert SpeculativeConfig.use_dflash(fake)
    assert gpu_model_runner.DFlashProposer is OmniDSparkProposer
    assert issubclass(OmniDSparkProposer, DFlashProposer)


def test_dspark_sampling_feeds_mapped_token_to_next_markov_step():
    class FakeModel:
        draft_id_to_target_id = torch.tensor([4, 4, 4])

        def __init__(self):
            self.seen_prev = []

        def compute_draft_logits(self, hidden_states):
            return torch.zeros(hidden_states.shape[0], 3)

        def markov_embed(self, prev):
            self.seen_prev.append(prev.clone())
            return prev

        def markov_bias(self, prev):
            logits = torch.zeros(prev.shape[0], 3)
            draft_id = torch.where(prev == 4, 1, 2)
            logits.scatter_(1, draft_id[:, None], 10.0)
            return logits

        def map_draft_to_target(self, draft_ids):
            return draft_ids + 4

    proposer = object.__new__(OmniDSparkProposer)
    proposer._omni_dspark = True
    proposer.num_speculative_tokens = 2
    proposer._dspark_prev_tokens = torch.tensor([4], dtype=torch.int32)
    proposer.model = FakeModel()
    proposer.vocab_size = 8
    proposer._sample_from_logits = lambda logits, _: (logits.argmax(dim=-1), None)

    tokens, probs = proposer._sample_draft_tokens(
        torch.zeros(2, 4),
        SimpleNamespace(),
    )

    assert probs is None
    assert tokens.tolist() == [5, 6]
    assert [x.tolist() for x in proposer.model.seen_prev] == [[4], [5]]
