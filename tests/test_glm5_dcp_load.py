import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from safetensors.torch import save_file

from deepspec.modeling.target.glm5_checkpoint import (
    Glm5CheckpointTopology,
    Glm5HuggingFaceLoadPlanner,
    Glm5QuantizedHuggingFaceStorageReader,
    load_glm5_huggingface_checkpoint,
)
from tests.distributed_test_utils import require_torchrun


class _TinyGlmState(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList(
            [nn.Identity(), nn.Identity(), nn.Identity(), nn.Module()]
        )
        layer = self.language_model.layers[3]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(4, 8, bias=False)


class Glm5DcpLoadTest(unittest.TestCase):
    @staticmethod
    def _write_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
        weight_map = {}
        shard_count = len(tensors)
        for shard_index, (name, tensor) in enumerate(tensors.items(), start=1):
            filename = f"model-{shard_index:05}-of-{shard_count:05}.safetensors"
            save_file({name: tensor}, path / filename)
            weight_map[name] = filename
        (path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": weight_map,
                }
            ),
            encoding="utf-8",
        )

    def test_dcp_maps_tp_conv_and_ep_expert_slices(self):
        prefix = "model.language_model.layers.3"
        checkpoint = {
            f"{prefix}.self_attn.q_proj.weight": torch.arange(
                32, dtype=torch.float32
            ).reshape(8, 4),
            f"{prefix}.self_attn.q_conv1d.weight": torch.arange(
                8, dtype=torch.float32
            ).reshape(4, 1, 2)
            + 100,
            f"{prefix}.self_attn.k_conv1d.weight": torch.arange(
                8, dtype=torch.float32
            ).reshape(4, 1, 2)
            + 200,
            f"{prefix}.self_attn.v_conv1d.weight": torch.arange(
                8, dtype=torch.float32
            ).reshape(4, 1, 2)
            + 300,
        }
        for expert in range(4):
            checkpoint[f"{prefix}.mlp.experts.{expert}.gate_proj.weight"] = (
                torch.arange(6, dtype=torch.float32).reshape(2, 3) + 1000 + expert * 10
            )
            checkpoint[f"{prefix}.mlp.experts.{expert}.up_proj.weight"] = (
                torch.arange(6, dtype=torch.float32).reshape(2, 3) + 2000 + expert * 10
            )
            checkpoint[f"{prefix}.mlp.experts.{expert}.down_proj.weight"] = (
                torch.arange(6, dtype=torch.float32).reshape(3, 2) + 3000 + expert * 10
            )

        with tempfile.TemporaryDirectory() as directory:
            self._write_checkpoint(Path(directory), checkpoint)
            state = {
                "language_model.layers.3.self_attn.q_proj.weight": torch.empty(4, 4),
                "language_model.layers.3.self_attn.conv1d.weight": torch.empty(6, 1, 2),
                "language_model.layers.3.mlp.experts.gate_up_proj": torch.empty(
                    2, 4, 3
                ),
                "language_model.layers.3.mlp.experts.down_proj": torch.empty(2, 3, 2),
            }
            dcp.load(
                state,
                storage_reader=dcp.HuggingFaceStorageReader(directory, thread_count=4),
                planner=Glm5HuggingFaceLoadPlanner(
                    Glm5CheckpointTopology(
                        tensor_parallel_rank=1,
                        tensor_parallel_size=2,
                        expert_parallel_rank=1,
                        expert_parallel_size=2,
                    )
                ),
                no_dist=True,
            )

        self.assertTrue(
            torch.equal(
                state["language_model.layers.3.self_attn.q_proj.weight"],
                checkpoint[f"{prefix}.self_attn.q_proj.weight"][4:],
            )
        )
        expected_conv = torch.cat(
            [
                checkpoint[f"{prefix}.self_attn.{projection}_conv1d.weight"][2:]
                for projection in ("q", "k", "v")
            ]
        )
        self.assertTrue(
            torch.equal(
                state["language_model.layers.3.self_attn.conv1d.weight"],
                expected_conv,
            )
        )
        expected_gate_up = torch.stack(
            [
                torch.cat(
                    [
                        checkpoint[f"{prefix}.mlp.experts.{expert}.gate_proj.weight"],
                        checkpoint[f"{prefix}.mlp.experts.{expert}.up_proj.weight"],
                    ]
                )
                for expert in (2, 3)
            ]
        )
        self.assertTrue(
            torch.equal(
                state["language_model.layers.3.mlp.experts.gate_up_proj"],
                expected_gate_up,
            )
        )
        self.assertTrue(
            torch.equal(
                state["language_model.layers.3.mlp.experts.down_proj"],
                torch.stack(
                    [
                        checkpoint[f"{prefix}.mlp.experts.{expert}.down_proj.weight"]
                        for expert in (2, 3)
                    ]
                ),
            )
        )

    def test_quantized_reader_dequantizes_only_the_owned_tp_slice(self):
        checkpoint_fqn = "model.language_model.layers.3.self_attn.q_proj.weight"
        raw_weight = (
            torch.arange(256 * 128, dtype=torch.float32).reshape(256, 128) % 7 - 3
        ).to(torch.float8_e4m3fn)
        scale = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
        checkpoint = {
            checkpoint_fqn: raw_weight,
            checkpoint_fqn.replace(".weight", ".weight_scale_inv"): scale,
        }

        with tempfile.TemporaryDirectory() as directory:
            self._write_checkpoint(Path(directory), checkpoint)
            state = {
                "language_model.layers.3.self_attn.q_proj.weight": torch.empty(
                    128, 128, dtype=torch.bfloat16
                )
            }
            dcp.load(
                state,
                storage_reader=Glm5QuantizedHuggingFaceStorageReader(
                    directory,
                    thread_count=2,
                    target_dtype=torch.bfloat16,
                    block_size=128,
                ),
                planner=Glm5HuggingFaceLoadPlanner(
                    Glm5CheckpointTopology(
                        tensor_parallel_rank=1,
                        tensor_parallel_size=2,
                        expert_parallel_rank=0,
                        expert_parallel_size=1,
                    )
                ),
                no_dist=True,
            )

        expected = (raw_weight[128:].float() * 3.0).to(torch.bfloat16)
        self.assertTrue(
            torch.equal(
                state["language_model.layers.3.self_attn.q_proj.weight"],
                expected,
            )
        )

    def test_vectorized_fp8_dequantization_matches_pytorch_for_unaligned_slice(self):
        full_shape = torch.Size((353, 421))
        row_slice = slice(37, 310)
        col_slice = slice(71, 389)
        slice_info = ((0, 3), (0, 4), row_slice, col_slice)
        weight = (
            torch.arange(273 * 318, dtype=torch.float32).reshape(273, 318) % 9 - 4
        ).to(torch.float8_e4m3fn)
        scales = torch.linspace(0.25, 1.75, 12, dtype=torch.float32).reshape(3, 4)

        pytorch_reader = dcp.QuantizedHuggingFaceStorageReader(
            ".", target_dtype=torch.bfloat16, block_size=128
        )
        glm5_reader = Glm5QuantizedHuggingFaceStorageReader(
            ".", target_dtype=torch.bfloat16, block_size=128
        )
        expected = pytorch_reader._dequantize_tensor(
            weight, scales, full_shape, slice_info
        )
        actual = glm5_reader._dequantize_tensor(
            weight, scales, full_shape, slice_info
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_loader_assigns_materialized_dcp_state_to_meta_model(self):
        checkpoint_fqn = "model.language_model.layers.3.self_attn.q_proj.weight"
        checkpoint_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        with tempfile.TemporaryDirectory() as directory:
            self._write_checkpoint(
                Path(directory),
                {
                    checkpoint_fqn: checkpoint_weight,
                    "model.visual.blocks.0.norm1.weight": torch.ones(4),
                    "lm_head.weight": torch.ones(4, 4),
                },
            )
            initialized_here = not dist.is_initialized()
            if initialized_here:
                dist.init_process_group(
                    "gloo",
                    init_method=f"file://{Path(directory) / 'rendezvous'}",
                    rank=0,
                    world_size=1,
                )
            try:
                with torch.device("meta"):
                    model = _TinyGlmState()
                model.requires_grad_(False)
                output = io.StringIO()
                with (
                    patch.dict(os.environ, {"DEEPSPEC_DCP_LOAD_THREADS": "2"}),
                    redirect_stdout(output),
                    redirect_stderr(output),
                ):
                    load_glm5_huggingface_checkpoint(
                        model=model,
                        checkpoint_dir=directory,
                        config=SimpleNamespace(quantization_config=None),
                        topology=Glm5CheckpointTopology(
                            tensor_parallel_rank=0,
                            tensor_parallel_size=1,
                            expert_parallel_rank=0,
                            expert_parallel_size=1,
                        ),
                    )
            finally:
                if initialized_here:
                    dist.destroy_process_group()

        self.assertNotIn("UNEXPECTED", output.getvalue())
        self.assertIn(
            "visual tower skipped, lm_head handled by draft initializer",
            output.getvalue(),
        )
        loaded = model.language_model.layers[3].self_attn.q_proj.weight
        self.assertFalse(loaded.is_meta)
        self.assertFalse(loaded.requires_grad)
        torch.testing.assert_close(loaded, checkpoint_weight)

    @unittest.skipUnless(
        "LOCAL_RANK" in os.environ and torch.cuda.is_available(),
        "run with four torchrun CUDA workers",
    )
    def test_meta_model_loads_directly_into_tp_fsdp2_shards(self):
        runtime = require_torchrun(self, world_size=4)
        mesh = init_device_mesh(
            "cuda",
            (2, 2),
            mesh_dim_names=("fsdp", "tp"),
        )
        tp_rank = mesh["tp"].get_local_rank()
        fsdp_rank = mesh["fsdp"].get_local_rank()

        with torch.device("meta"):
            model = _TinyGlmState()
        projection = model.language_model.layers[3].self_attn.q_proj
        local_weight = projection.weight.narrow(0, tp_rank * 4, 4).contiguous()
        projection.weight = nn.Parameter(local_weight, requires_grad=False)
        projection.out_features = 4
        fully_shard(model, mesh=mesh["fsdp"], reshard_after_forward=False)

        checkpoint_fqn = "model.language_model.layers.3.self_attn.q_proj.weight"
        checkpoint_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        with tempfile.TemporaryDirectory() as directory:
            self._write_checkpoint(Path(directory), {checkpoint_fqn: checkpoint_weight})
            state = model.state_dict()
            dcp.load(
                state,
                storage_reader=dcp.HuggingFaceStorageReader(directory),
                planner=Glm5HuggingFaceLoadPlanner(
                    Glm5CheckpointTopology(
                        tensor_parallel_rank=tp_rank,
                        tensor_parallel_size=2,
                        expert_parallel_rank=0,
                        expert_parallel_size=1,
                    )
                ),
            )
            model.load_state_dict(state, strict=True, assign=True)

        loaded = model.language_model.layers[3].self_attn.q_proj.weight
        self.assertFalse(loaded.is_meta)
        source_start = tp_rank * 4 + fsdp_rank * 2
        torch.testing.assert_close(
            loaded.to_local(),
            checkpoint_weight[source_start : source_start + 2].to(runtime.device),
        )


if __name__ == "__main__":
    unittest.main()
