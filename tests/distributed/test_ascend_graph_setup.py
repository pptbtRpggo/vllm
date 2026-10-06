# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace

from vllm.distributed.ascend_device_delay import prepare_ascend_full_graph


def test_full_graph_registry_initialized_once_without_changing_runner(monkeypatch):
    registered = []
    module = ModuleType("vllm_ascend.compilation.acl_graph")
    module.get_graph_params = lambda: registered[0] if registered else None
    module.set_graph_params = lambda sizes: registered.append(sizes.copy())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True)
            ),
        ),
        model_runner=SimpleNamespace(cudagraph_batch_sizes=[1, 8], use_aclgraph=False),
    )
    prepare_ascend_full_graph(worker)
    prepare_ascend_full_graph(worker)
    assert registered == [[1, 8]]
    assert worker.model_runner.use_aclgraph is False
    worker.vllm_config.model_config.enforce_eager = True
    prepare_ascend_full_graph(worker)
    assert registered == [[1, 8]]
