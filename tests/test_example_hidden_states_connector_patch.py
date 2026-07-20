from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
)

import vllm_omni.patch  # noqa: F401


def test_aborted_unscheduled_hidden_state_request_is_ignored():
    connector = object.__new__(ExampleHiddenStatesConnector)
    connector._request_filenames = {}
    request = SimpleNamespace(request_id="aborted-before-schedule")

    assert connector.request_finished(request, []) == (False, None)
