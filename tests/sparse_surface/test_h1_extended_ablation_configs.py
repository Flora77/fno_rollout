import copy
import json
from pathlib import Path

import pytest

from neuralop.training.sparse_experiment_runner import (
    SUPPORTED_EXPERIMENTS,
    load_resolved_sparse_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config" / "sparse_experiments"


def _read(relative_path: str) -> dict:
    return json.loads((CONFIG_ROOT / relative_path).read_text(encoding="utf-8"))


def _scientific_payload_without_initialization(config: dict) -> dict:
    payload = copy.deepcopy(config)
    payload.pop("experiment_id")
    payload.pop("experiment_name")
    payload["pipeline"].pop("reconstructor_checkpoint", None)
    return payload


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        flattened = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(item, path))
        return flattened
    return {prefix: value}


@pytest.mark.parametrize("seed", (42, 43, 44))
def test_h1_a4_a5_are_strict_initialization_pairs(seed):
    pretrained = _read(f"h1_a4/pretrained_seed{seed}.formal.json")
    random_init = _read(f"h1_a5/random_init_seed{seed}.formal.json")

    assert pretrained["experiment_id"] == "H1-A4"
    assert random_init["experiment_id"] == "H1-A5"
    assert pretrained["runtime"]["seed"] == random_init["runtime"]["seed"] == seed
    assert pretrained["observation"]["seed"] == random_init["observation"]["seed"] == 42
    assert pretrained["model"] == {
        "hybrid_unet_channels": [12, 24, 40],
        "partialconv_levels": 3,
        "use_skip_connections": True,
    }
    assert random_init["model"] == pretrained["model"]
    assert Path(
        PROJECT_ROOT / pretrained["pipeline"]["reconstructor_checkpoint"]
    ).is_file()
    assert "reconstructor_checkpoint" not in random_init["pipeline"]
    assert (
        _scientific_payload_without_initialization(pretrained)
        == _scientific_payload_without_initialization(random_init)
    )


def test_h1_a6_differs_from_h1_a1_only_by_skip_switch():
    baseline = _read("h1_a1_partialconv_unet_frozen.formal.json")
    no_skip = _read("h1_a6_no_skip_partialconv3_unet_frozen.formal.json")
    for payload in (baseline, no_skip):
        payload.pop("experiment_id")
        payload.pop("experiment_name")

    baseline_flat = _flatten(baseline)
    no_skip_flat = _flatten(no_skip)
    differing = {
        key
        for key in baseline_flat
        if baseline_flat[key] != no_skip_flat.get(key)
    } | {
        key
        for key in no_skip_flat
        if no_skip_flat[key] != baseline_flat.get(key)
    }

    assert differing == {"model.use_skip_connections"}
    assert baseline_flat["model.use_skip_connections"] is True
    assert no_skip_flat["model.use_skip_connections"] is False


@pytest.mark.parametrize(
    ("relative_path", "experiment_id"),
    [
        ("h1_a4/pretrained_seed42.formal.json", "H1-A4"),
        ("h1_a5/random_init_seed42.formal.json", "H1-A5"),
        ("h1_a6_no_skip_partialconv3_unet_frozen.formal.json", "H1-A6"),
    ],
)
def test_h1_extended_ids_resolve_through_shared_runner(
    relative_path, experiment_id, tmp_path
):
    resolved = load_resolved_sparse_config(
        CONFIG_ROOT / relative_path,
        project_root=PROJECT_ROOT,
        run_dir=tmp_path / experiment_id.lower(),
        device="cpu",
    )

    assert experiment_id in SUPPORTED_EXPERIMENTS
    assert resolved["experiment_id"] == experiment_id
    assert resolved["pipeline"]["reconstructor"] == "hybrid_pconv_unet"
    assert resolved["pipeline"]["coupling"] == "frozen"
