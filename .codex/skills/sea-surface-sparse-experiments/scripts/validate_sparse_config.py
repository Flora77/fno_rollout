#!/usr/bin/env python3
"""Validate a JSON definition for a sparse sea-surface experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ALLOWED_MASKS = {"all_valid", "fixed_points", "random_points", "blocks", "stripes"}
ALLOWED_RECONSTRUCTORS = {
    "none",
    "bilinear",
    "kriging",
    "pod",
    "mask_unet",
    "hybrid_pconv_unet",
    "gno",
    "gino",
    "partialconv_mae",
    "vit_mae",
    "vit_mask_aware_pool",
    "vit_mask_aware_confidence",
    "sensor_token_grid_query",
    "tubelet_mae",
    "direct",
}
ALLOWED_COUPLINGS = {"existing", "frozen", "joint", "direct"}


TEMPLATE: dict[str, Any] = {
    "experiment_id": "P1",
    "experiment_name": "p1_partialconv_mae_frozen_fixed_points_r05_seed42",
    "data": {
        "input_steps": 60,
        "output_steps": 30,
        "rollout_steps": 300,
        "height": 64,
        "width": 64,
        "dt": 0.25,
    },
    "observation": {
        "mask_type": "fixed_points",
        "observation_rate": 0.05,
        "manifest_path": "./data/masks/sparse_masks.npz",
        "mask_id": "mask_00000",
        "noise_std_fraction": 0.0,
        "temporal_dropout": 0.0,
        "seed": 42,
    },
    "pipeline": {
        "reconstructor": "partialconv_mae",
        "coupling": "frozen",
        "forecaster": "rfno",
        "rfno_checkpoint": "./checkpoints/<baseline>.pt",
    },
    "training": {
        "pretrain_reconstructor": True,
        "forecast_supervision": False,
        "use_rollout_curriculum": False,
        "max_rollout_steps": 30,
    },
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive_int(section: dict[str, Any], key: str, errors: list[str]) -> None:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{key} must be a positive integer, got {value!r}")


def _manifest_document_path(path: Path) -> Path:
    return path.with_suffix(".json") if path.suffix.lower() == ".npz" else path


def _validate_manifest_selection(
    observation: dict[str, Any], errors: list[str], *, base_dir: Path
) -> None:
    manifest_value = observation.get("manifest_path")
    mask_id = observation.get("mask_id")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        errors.append("observation.manifest_path must be a non-empty path")
        return
    if not isinstance(mask_id, str) or not mask_id.strip():
        errors.append("observation.mask_id must be a non-empty string")
        return

    manifest_path = _manifest_document_path(Path(manifest_value))
    if not manifest_path.is_absolute():
        manifest_path = base_dir / manifest_path
    if not manifest_path.is_file():
        errors.append(f"mask manifest JSON does not exist: {manifest_path}")
        return
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read mask manifest {manifest_path}: {exc}")
        return
    entries = document.get("entries") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        errors.append(f"mask manifest has no entries list: {manifest_path}")
        return
    selected = next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("mask_id") == mask_id
        ),
        None,
    )
    if selected is None:
        errors.append(f"mask_id {mask_id!r} is not present in {manifest_path}")
        return

    if selected.get("mask_type") != observation.get("mask_type"):
        errors.append(
            "selected mask_type differs from observation.mask_type: "
            f"manifest={selected.get('mask_type')!r}, "
            f"config={observation.get('mask_type')!r}"
        )
    requested_rate = selected.get("requested_observation_rate")
    configured_rate = observation.get("observation_rate")
    if _finite_number(requested_rate) and _finite_number(configured_rate):
        if not math.isclose(
            float(requested_rate), float(configured_rate), rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(
                "selected observation rate differs from config: "
                f"manifest={requested_rate}, config={configured_rate}"
            )
    manifest_seed = selected.get("seed")
    configured_seed = observation.get("seed")
    if isinstance(manifest_seed, int) and isinstance(configured_seed, int):
        if manifest_seed != configured_seed:
            errors.append(
                "selected mask seed differs from config: "
                f"manifest={manifest_seed}, config={configured_seed}"
            )
    manifest_dropout = selected.get("temporal_dropout", 0.0)
    configured_dropout = observation.get("temporal_dropout", 0.0)
    if _finite_number(manifest_dropout) and _finite_number(configured_dropout):
        if not math.isclose(
            float(manifest_dropout),
            float(configured_dropout),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(
                "selected temporal dropout differs from config: "
                f"manifest={manifest_dropout}, config={configured_dropout}"
            )


def validate(
    config: dict[str, Any], *, base_dir: Path | None = None
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for key in ("experiment_id", "experiment_name", "data", "observation", "pipeline"):
        if key not in config:
            errors.append(f"missing required top-level key: {key}")

    if errors:
        return errors, warnings

    if not isinstance(config["experiment_id"], str) or not config["experiment_id"].strip():
        errors.append("experiment_id must be a non-empty string")
    if not isinstance(config["experiment_name"], str) or not config["experiment_name"].strip():
        errors.append("experiment_name must be a non-empty string")

    data = config["data"]
    observation = config["observation"]
    pipeline = config["pipeline"]
    if not isinstance(data, dict) or not isinstance(observation, dict) or not isinstance(pipeline, dict):
        return ["data, observation, and pipeline must be JSON objects"], warnings
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)

    for key in ("input_steps", "output_steps", "rollout_steps", "height", "width"):
        _positive_int(data, key, errors)
    if data.get("input_steps") != 60:
        warnings.append("input_steps differs from the established 60-frame baseline")
    if data.get("output_steps") != 30:
        warnings.append("output_steps differs from the established 30-frame chunk")
    if data.get("rollout_steps") != 300:
        warnings.append("rollout_steps differs from the established 300-frame evaluation")

    mask_type = observation.get("mask_type")
    if mask_type not in ALLOWED_MASKS:
        errors.append(f"observation.mask_type must be one of {sorted(ALLOWED_MASKS)}, got {mask_type!r}")
    rate = observation.get("observation_rate")
    if not _finite_number(rate) or not 0.0 < float(rate) <= 1.0:
        errors.append("observation.observation_rate must be in (0, 1]")
    for key in ("noise_std_fraction", "temporal_dropout"):
        value = observation.get(key, 0.0)
        if not _finite_number(value) or not 0.0 <= float(value) < 1.0:
            errors.append(f"observation.{key} must be in [0, 1)")
    if not isinstance(observation.get("seed"), int):
        errors.append("observation.seed must be an integer")
    _validate_manifest_selection(observation, errors, base_dir=base_dir)

    reconstructor = pipeline.get("reconstructor")
    coupling = pipeline.get("coupling")
    if reconstructor not in ALLOWED_RECONSTRUCTORS:
        errors.append(f"pipeline.reconstructor must be one of {sorted(ALLOWED_RECONSTRUCTORS)}")
    if coupling not in ALLOWED_COUPLINGS:
        errors.append(f"pipeline.coupling must be one of {sorted(ALLOWED_COUPLINGS)}")
    if reconstructor == "direct" and coupling != "direct":
        errors.append("direct reconstructor requires pipeline.coupling='direct'")
    if coupling == "direct" and reconstructor != "direct":
        errors.append("pipeline.coupling='direct' requires reconstructor='direct'")
    if coupling == "frozen" and not str(pipeline.get("rfno_checkpoint", "")).strip():
        errors.append("frozen RFNO experiments require pipeline.rfno_checkpoint")
    if mask_type == "all_valid" and _finite_number(rate) and float(rate) != 1.0:
        errors.append("all_valid masks require observation_rate=1.0")
    if reconstructor in {"gno", "gino"} and observation.get("coordinate_mode") is None:
        warnings.append("GNO/GINO experiments should document observation.coordinate_mode")
    if reconstructor == "partialconv_mae" and mask_type == "all_valid":
        warnings.append("PartialConv-MAE with all-valid input does not exercise missing-data behavior")

    training = config.get("training", {})
    if not isinstance(training, dict):
        errors.append("training must be a JSON object")
    elif reconstructor in {
        "partialconv_mae",
        "hybrid_pconv_unet",
        "gno",
        "gino",
        "mask_unet",
        "vit_mae",
        "vit_mask_aware_pool",
        "vit_mask_aware_confidence",
        "sensor_token_grid_query",
        "tubelet_mae",
    }:
        supervision = training.get("forecast_supervision")
        if not isinstance(supervision, bool):
            errors.append(
                "Learned reconstructor requires explicit training.forecast_supervision"
            )
        if config.get("experiment_id") == "P1" and supervision is True:
            if training.get("use_rollout_curriculum") is not False:
                errors.append(
                    "P1 forecast-supervised downstream training requires "
                    "use_rollout_curriculum=false"
                )
            if int(training.get("max_rollout_steps", 0)) != 30:
                errors.append(
                    "P1 forecast-supervised downstream training requires "
                    "max_rollout_steps=30"
                )
        if config.get("experiment_id") == "P2" and supervision is not True:
            errors.append("P2 must use forecast_supervision=true")
        if config.get("experiment_id") in {"V3", "V4"} and supervision is not True:
            errors.append(f"{config.get('experiment_id')} must use forecast_supervision=true")
        if config.get("experiment_id") == "MU-V3":
            if supervision is not True:
                errors.append("MU-V3 must use forecast_supervision=true")
            if training.get("use_rollout_curriculum") is not False:
                errors.append("MU-V3 requires use_rollout_curriculum=false")
            if int(training.get("max_rollout_steps", 0)) != 30:
                errors.append("MU-V3 requires max_rollout_steps=30")
        mae_mask_ratio = float(training.get("mae_mask_ratio", 0.0))
        if not 0.0 <= mae_mask_ratio < 1.0:
            errors.append("training.mae_mask_ratio must be in [0,1)")
        if mae_mask_ratio > 0.0 and supervision is True:
            errors.append("MAE pretraining cannot use forecast supervision")
        weights = training.get("loss_weights", {})
        if not isinstance(weights, dict):
            errors.append("training.loss_weights must be a JSON object")
        elif supervision is False:
            for key in ("rollout", "forecast_gradient", "spectrum"):
                if float(weights.get(key, 0.0)) != 0.0:
                    errors.append(
                        f"reconstruction-only training requires loss_weights.{key}=0"
                    )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, help="JSON experiment configuration")
    parser.add_argument("--write-template", type=Path, help="write a valid template JSON and exit")
    args = parser.parse_args()

    if args.write_template is not None:
        args.write_template.parent.mkdir(parents=True, exist_ok=True)
        args.write_template.write_text(json.dumps(TEMPLATE, indent=2), encoding="utf-8")
        print(f"wrote template: {args.write_template}")
        return 0
    if args.config is None:
        parser.error("provide CONFIG or --write-template PATH")

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read JSON config: {exc}", file=sys.stderr)
        return 2
    if not isinstance(config, dict):
        print("ERROR: top-level JSON value must be an object", file=sys.stderr)
        return 2

    errors, warnings = validate(config, base_dir=Path.cwd())
    for item in warnings:
        print(f"WARNING: {item}")
    for item in errors:
        print(f"ERROR: {item}", file=sys.stderr)
    if errors:
        print(f"validation failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("configuration is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
