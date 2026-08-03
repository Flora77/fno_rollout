import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = (
    REPO_ROOT
    / ".codex"
    / "skills"
    / "sea-surface-sparse-experiments"
    / "scripts"
    / "validate_sparse_config.py"
)
SPEC = importlib.util.spec_from_file_location("sparse_config_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


@pytest.mark.parametrize(
    "name",
    [
        "b1_bilinear_frozen.example.json",
        "p1_partialconv_mae_frozen.example.json",
        "p2_partialconv_mae_joint.example.json",
        "h1_hybrid_pconv1_unet_frozen.example.json",
        "h1_a4/pretrained_seed42.formal.json",
        "h1_a5/random_init_seed42.formal.json",
        "h1_a6_no_skip_partialconv3_unet_frozen.formal.json",
        "b5_gno_frozen.example.json",
        "b5_gino_frozen.example.json",
        "b6_gno_joint_30f.example.json",
        "b6_gino_joint_30f.example.json",
        "pc_d1_mask_aware_pooling.example.json",
        "pc_d2_confidence.example.json",
        "st_d0_sensor_token_grid_query.example.json",
    ],
)
def test_example_configs_fix_mask_and_supervision_contracts(name):
    path = REPO_ROOT / "config" / "sparse_experiments" / name
    config = json.loads(path.read_text(encoding="utf-8"))

    errors, _ = VALIDATOR.validate(config, base_dir=REPO_ROOT)

    assert errors == []


@pytest.mark.parametrize(
    "name",
    [
        "b5_gno_frozen.example.json",
        "b5_gino_frozen.example.json",
        "b6_gno_joint_30f.example.json",
        "b6_gino_joint_30f.example.json",
    ],
)
def test_coordinate_examples_use_nonlinear_periodic_reconstruction(name):
    path = REPO_ROOT / "config" / "sparse_experiments" / name
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["model"]["gno_kernel_mode"] == "channelwise_nonlinear"
    assert config["model"]["hard_observation_consistency"] is True
    if name == "b5_gno_frozen.example.json":
        assert config["model"]["gno_architecture"] == "residual_graph_v3"
        assert config["model"]["operator_hidden_channels"] == 64
        assert config["model"]["gno_kernel_rank"] == 64
        assert config["model"]["gno_kernel_normalization"] == "neighbor_mean"
        assert config["model"]["gno_graph_layers"] == 3
        assert config["model"]["gno_grid_refinement_layers"] == 3
        assert config["model"]["raw_observation_supervision"] is True
        assert config["training"]["loss_weights"]["history_gradient"] == 0.05
    else:
        assert config["model"]["gno_kernel_rank"] == 16
        assert config["training"]["loss_weights"]["history_gradient"] == 0.20
    if config["pipeline"]["reconstructor"] == "gino":
        assert config["model"]["latent_shape"] == [32, 32]
        assert config["model"]["latent_n_modes"] == [16, 16]
        assert config["model"]["latent_positional_embedding"] == "none"


@pytest.mark.parametrize(
    "name",
    [
        "b6_gno_joint_30f.example.json",
        "b6_gino_joint_30f.example.json",
    ],
)
def test_b6_examples_select_only_full_curriculum_rollout(name):
    path = REPO_ROOT / "config" / "sparse_experiments" / name
    config = json.loads(path.read_text(encoding="utf-8"))
    training = config["training"]

    assert training["use_rollout_curriculum"] is True
    assert training["rollout_train_steps"] == [30, 60, 120, 180, 240, 300]
    assert training["max_rollout_steps"] == 300
    assert training["primary_loss"] == "forecast_300_nrmse"


def test_missing_mask_id_is_rejected():
    path = (
        REPO_ROOT
        / "config"
        / "sparse_experiments"
        / "p1_partialconv_mae_frozen.example.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    config["observation"].pop("mask_id")

    errors, _ = VALIDATOR.validate(config, base_dir=REPO_ROOT)

    assert any("mask_id" in error for error in errors)


def test_mask_metadata_mismatch_is_rejected():
    path = (
        REPO_ROOT
        / "config"
        / "sparse_experiments"
        / "p2_partialconv_mae_joint.example.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    config["observation"]["observation_rate"] = 0.1

    errors, _ = VALIDATOR.validate(config, base_dir=REPO_ROOT)

    assert any("observation rate differs" in error for error in errors)


def test_p1_forecast_supervision_requires_fixed_30_frame_rollout():
    path = (
        REPO_ROOT
        / "config"
        / "sparse_experiments"
        / "p1_partialconv_mae_frozen.example.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    config["training"]["forecast_supervision"] = True
    config["training"]["use_rollout_curriculum"] = True
    config["training"]["max_rollout_steps"] = 300

    errors, _ = VALIDATOR.validate(config, base_dir=REPO_ROOT)

    assert any("use_rollout_curriculum=false" in error for error in errors)
    assert any("max_rollout_steps=30" in error for error in errors)


def test_p1_forecast_supervision_accepts_fixed_30_frame_rollout():
    path = (
        REPO_ROOT
        / "config"
        / "sparse_experiments"
        / "p1_partialconv_mae_frozen.example.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    config["training"]["forecast_supervision"] = True
    config["training"]["use_rollout_curriculum"] = False
    config["training"]["max_rollout_steps"] = 30
    config["training"]["loss_weights"]["rollout"] = 1.0

    errors, _ = VALIDATOR.validate(config, base_dir=REPO_ROOT)

    assert errors == []
