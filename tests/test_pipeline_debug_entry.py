"""Debug wrappers must use the frozen v3 verifier when training uses v3."""

import importlib.util
import json
from pathlib import Path


def test_debug_entry_uses_independent_v3_verification(tmp_path, monkeypatch):
    from deepspec.pipeline import execution

    path = (
        Path(__file__).parents[1]
        / "scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_debug_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "status.json").write_text(
        json.dumps({"state": "succeeded", "cleanup_complete": True})
    )
    calls = []
    report = {"verified": True, "independent": True}
    monkeypatch.setattr(
        execution, "verify_run", lambda root: calls.append(root) or report
    )
    assert module.verify_training(tmp_path) == report
    assert calls == [tmp_path]
