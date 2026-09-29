import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from torchspec.utils import wandb as wandb_utils


@pytest.fixture
def wandb_state(monkeypatch, tmp_path):
    args = SimpleNamespace(
        use_wandb=True,
        wandb_mode="online",
        wandb_key=None,
        wandb_host=None,
        wandb_group="experiment",
        wandb_project="test-project",
        wandb_random_suffix=True,
        rank=0,
        wandb_team=None,
        wandb_dir=str(tmp_path / "logs"),
        wandb_run_id="test-run-id",
        sglang_enable_metrics=False,
    )
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(wandb_utils, "_DEFAULT_WANDB_API_KEY", "test-default-key", raising=False)
    monkeypatch.delattr(wandb_utils.wandb.util, "generate_id", raising=False)
    init = MagicMock()
    login = MagicMock()
    monkeypatch.setattr(wandb_utils.wandb, "init", init)
    monkeypatch.setattr(wandb_utils.wandb, "login", login)
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", MagicMock())
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="test-run-id"))
    return args, init, login


@pytest.mark.parametrize("mode", ["offline", "online"])
@pytest.mark.parametrize("random_suffix", [True, False])
def test_primary_initialization_without_legacy_generate_id(wandb_state, mode, random_suffix):
    """W&B initialization must work without the removed wandb.util export."""
    args, init, login = wandb_state
    args.wandb_mode = mode
    args.wandb_random_suffix = random_suffix

    wandb_utils.init_wandb_primary(args)

    init.assert_called_once()
    kwargs = init.call_args.kwargs
    if random_suffix:
        assert re.fullmatch(r"experiment_[a-z0-9]{8}", kwargs["group"])
        assert kwargs["name"] == f"{kwargs['group']}-RANK_0"
    else:
        assert kwargs["group"] == kwargs["name"] == "experiment"
    assert kwargs["settings"].mode == ("offline" if mode == "offline" else "shared")
    assert args.wandb_run_id == "test-run-id"
    if mode == "online":
        login.assert_called_once_with(key="test-default-key", host=None)
    else:
        login.assert_not_called()


@pytest.mark.parametrize("role", ["primary", "secondary"])
@pytest.mark.parametrize(
    ("configured_key", "environment_key", "expected_key"),
    [
        ("test-configured-key", "test-env-key", "test-configured-key"),
        (None, "test-env-key", "test-env-key"),
        (None, None, "test-default-key"),
    ],
)
def test_wandb_key_precedence_and_run_config_redaction(
    wandb_state, monkeypatch, role, configured_key, environment_key, expected_key
):
    args, init, login = wandb_state
    args.wandb_key = configured_key
    args.wandb_host = "https://wandb.example.test"
    if environment_key is not None:
        monkeypatch.setenv("WANDB_API_KEY", environment_key)

    getattr(wandb_utils, f"init_wandb_{role}")(args)

    login.assert_called_once_with(key=expected_key, host=args.wandb_host)
    assert args.wandb_key == configured_key
    assert init.call_args.kwargs["config"]["wandb_key"] is None
    assert expected_key not in repr(init.call_args.kwargs["config"])


@pytest.mark.parametrize("role", ["primary", "secondary"])
@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_wandb_does_not_login_when_offline_or_disabled(wandb_state, role, mode):
    args, _, login = wandb_state
    args.wandb_mode = mode
    args.wandb_key = "test-configured-key"

    getattr(wandb_utils, f"init_wandb_{role}")(args)

    login.assert_not_called()
