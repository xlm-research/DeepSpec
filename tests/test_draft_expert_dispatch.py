"""Optional backend selection and the draft/target resource boundary."""

from types import SimpleNamespace
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from deepspec.distributed.draft_expert_dispatch import (
    build_draft_expert_dispatcher,
    close_draft_expert_dispatchers,
)
from deepspec.modeling.glm5_next_parallel import parallelize_glm5_next_model


def _check_backend_consensus(rank, rendezvous):
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous,
        timeout=timedelta(seconds=30),
    )
    try:
        test = DraftExpertDispatchTest()
        topology = test._topology("auto", expert_parallel_group=dist.group.WORLD)
        with patch("deepspec.distributed.deepep_dispatch.require_deepep",
                   side_effect=ImportError("rank 1 lacks DeepEP") if rank else None):
            with warnings.catch_warnings(record=True) as emitted:
                assert build_draft_expert_dispatcher(test._model(), topology=topology) is None
                assert len(emitted) == 1
            topology.expert_dispatch_backend = "deepep"
            try:
                build_draft_expert_dispatcher(test._model(), topology=topology)
            except (ImportError, RuntimeError):
                pass
            else:
                raise AssertionError("Every EP peer must fail explicit DeepEP selection")
    finally:
        dist.destroy_process_group()


class DraftExpertDispatchTest(unittest.TestCase):
    def _topology(self, backend="native", **overrides):
        values = dict(
            expert_dispatch_backend=backend,
            expert_parallel_size=2,
            expert_parallel_group=object(),
            tensor_parallel_size=1,
            context_parallel_size=1,
            tensor_parallel_group=None,
            tensor_parallel_rank=0,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def _model(self):
        return SimpleNamespace(config=SimpleNamespace(
            model_type="glm5_next_text", n_routed_experts=8,
            hidden_size=256, num_experts_per_tok=2,
        ))

    @patch("deepspec.distributed.deepep_dispatch.require_deepep", side_effect=ImportError("missing V2"))
    def test_missing_optional_dependency_keeps_native_and_auto_usable(self, require):
        self.assertIsNone(build_draft_expert_dispatcher(self._model(), topology=self._topology()))
        require.assert_not_called()
        with self.assertWarnsRegex(UserWarning, "missing V2"):
            self.assertIsNone(build_draft_expert_dispatcher(
                self._model(), topology=self._topology("auto"),
            ))
        with self.assertRaisesRegex(ImportError, "missing V2"):
            build_draft_expert_dispatcher(self._model(), topology=self._topology("deepep"))

    @patch("deepspec.distributed.deepep_dispatch.require_deepep")
    def test_auto_falls_back_before_loading_dependency_for_unsupported_layout(self, require):
        with self.assertWarnsRegex(UserWarning, "eager"):
            self.assertIsNone(build_draft_expert_dispatcher(
                self._model(), topology=self._topology("auto", use_compile=True),
            ))
        require.assert_not_called()

    @patch("deepspec.distributed.draft_expert_dispatch.build_draft_expert_dispatcher",
           side_effect=AssertionError("target must not allocate a draft buffer"))
    def test_target_adapter_never_selects_deepep(self, build):
        # EP dispatch selection must stay draft-only even if a caller's
        # topology happens to carry a training backend option.
        model = self._model()
        model.layers = []
        model.embed_tokens = torch.nn.Embedding(8, 4)
        parallelize_glm5_next_model(model, topology=self._topology("deepep"), draft=False)
        build.assert_not_called()

    def test_model_teardown_closes_shared_buffer_once(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        dispatcher = Mock()
        for layer in model:
            layer._deepspec_deepep_dispatcher = dispatcher
        close_draft_expert_dispatchers(model)
        dispatcher.close.assert_called_once_with()
        close_draft_expert_dispatchers(None)
        close_draft_expert_dispatchers(torch.nn.Linear(2, 2))

    def test_heterogeneous_dependencies_choose_one_backend_per_ep_group(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory, "rendezvous").as_uri()
            mp.spawn(_check_backend_consensus, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
