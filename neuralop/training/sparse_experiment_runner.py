"""Production-grade shared runtime infrastructure for sparse sea-surface experiments.

This module intentionally contains no multi-epoch training loop.  It freezes the
scientific inputs to a run, constructs leak-safe loaders, provides shared sparse
evaluation, and manages resumable checkpoints for the training entry point.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
import numpy as np
from torch import Tensor, nn
from torch.utils.data import DataLoader

from neuralop.data.datasets.sea_surface_simple import SeaSurfaceSimpleDataset
from neuralop.data.datasets.sparse_sea_surface import (
    SparseMaskManifest,
    SparseSeaSurfaceDataset,
    sparse_model_inputs,
)
from neuralop.evaluation.sparse_surface import SparseSurfaceMetricAccumulator
from neuralop.models.fno_deeponet import (
    FNODeepONet,
    FNODeepONetGridForecaster,
    FNODeepONetReconstructor,
    fno_deeponet_kwargs,
)
from neuralop.models.reconstructors import (
    PartialConvMaskedAutoencoder,
    PeriodicConfidenceMaskAwarePoolingViTReconstructor,
    PeriodicGINOReconstructor,
    PeriodicGNOReconstructor,
    PeriodicHybridPartialConvUNet,
    PeriodicMaskAwarePoolingViTReconstructor,
    PeriodicMaskUNet,
    PeriodicSensorTokenGridQueryReconstructor,
    PeriodicViTMaskedAutoencoder,
    PODSpatialReconstructor,
)
from neuralop.models.sparse_forecast_pipeline import (
    BilinearFrozenRFNOPipeline,
    FNODeepONetDirectSparsePipeline,
    LearnedReconstructionRFNOPipeline,
    PartialConvMAERFNOPipeline,
)
from neuralop.training.sparse_coupled_forecast import (
    SparseCoupledForecastTrainer,
    SparseCoupledTrainingConfig,
)


SUPPORTED_EXPERIMENTS = frozenset(
    {
        "B1", "B3", "B4", "B5", "B5-GNO", "B5-GINO", "B6",
        "H1", "H1-A1", "H1-A2", "H1-A3", "H1-A4", "H1-A5", "H1-A6",
        "P1", "P2", "V0", "V1", "V2", "V3", "V4", "PC-D1", "PC-D2",
        "ST-D0", "MU-V3", "FD-R1", "FD-A1", "FD-L1",
    }
)
DEFAULT_FORECAST_HORIZONS = (30, 60, 120, 180, 240, 300)

# These defaults were added after the locked B4/P2 checkpoints were written.
# They are not consumed by the corresponding model constructors.  Compatibility
# is intentionally restricted to the named field, experiment, and exact default
# value; arbitrary model fields or non-default values remain scientific changes.
_LEGACY_UNUSED_MODEL_DEFAULTS = {
    "B4": {
        "pod_rank": 64,
        "pod_ranks": [8, 16, 32, 64],
        "pod_ridge": 1.0e-6,
    },
    "P2": {
        "mask_unet_channels": [12, 24, 40],
        "pod_rank": 64,
        "pod_ranks": [8, 16, 32, 64],
        "pod_ridge": 1.0e-6,
    },
}


def _deep_merge(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = copy.deepcopy(value)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Union[str, Path], *, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: Union[str, Path], project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary artifact already exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_checkpoint_mapping(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping: {path}")
    return checkpoint


def read_b0_normalization(checkpoint_path: Union[str, Path]) -> Tuple[float, float]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _load_checkpoint_mapping(checkpoint_path)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise KeyError(f"B0 checkpoint lacks config metadata: {checkpoint_path}")
    input_steps = int(checkpoint_config.get("input_steps", 60))
    output_steps = int(checkpoint_config.get("output_steps", 30))
    if input_steps != 60 or output_steps != 30:
        raise ValueError(
            f"B0 checkpoint must use 60-to-30 contract, got {input_steps}-to-{output_steps}"
        )
    missing = [
        key
        for key in ("train_dataset_mean", "train_dataset_std")
        if key not in checkpoint
    ]
    if missing:
        raise KeyError(
            f"B0 checkpoint lacks training normalization metadata {missing}: "
            f"{checkpoint_path}"
        )
    mean = float(checkpoint["train_dataset_mean"])
    std = float(checkpoint["train_dataset_std"])
    if not math.isfinite(mean):
        raise ValueError("B0 train_dataset_mean must be finite")
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("B0 train_dataset_std must be finite and positive")
    return mean, std


def _default_config() -> Dict[str, Any]:
    return {
        "data": {
            "data_root": "./data/bimodal",
            "train_dir": None,
            "val_dir": None,
            "test_dir": None,
            "split_manifest_path": None,
            "variable": "height",
            "input_steps": 60,
            "output_steps": 30,
            "rollout_steps": 300,
            "stride": 4,
            "batch_size": 2,
            "val_batch_size": 2,
            "test_batch_size": 2,
            "num_workers": 0,
            "pin_memory": True,
        },
        "model": {
            "mask_unet_channels": [12, 24, 40],
            "pod_rank": 64,
            "pod_ranks": [8, 16, 32, 64],
            "pod_ridge": 1.0e-6,
            "partialconv_encoder_channels": [24, 48, 72],
            "partialconv_decoder_channels": [32, 24, 16],
            "vit_tokenization": "spatial_patch",
            "vit_patch_size": 4,
            "vit_tubelet_size": 10,
            "vit_encoder_dim": 128,
            "vit_encoder_depth": 4,
            "vit_encoder_heads": 4,
            "vit_decoder_dim": 64,
            "vit_decoder_depth": 2,
            "vit_decoder_heads": 4,
            "vit_mlp_ratio": 4.0,
            "fno_deeponet_branch_width": 64,
            "fno_deeponet_branch_modes": 24,
            "fno_deeponet_branch_layers": 4,
            "fno_deeponet_branch_hidden": 512,
            "fno_deeponet_latent_dim": 256,
            "fno_deeponet_trunk_hidden": 384,
            "fno_deeponet_trunk_layers": 4,
            "fno_deeponet_time_fourier_bands": 24,
            "fno_deeponet_spatial_fourier_bands": 16,
            "fno_deeponet_time_max_frequency": 48.0,
            "fno_deeponet_query_chunk_size": 8192,
            "fno_deeponet_checkpoint_trunk": True,
        },
        "evaluation": {
            "forecast_horizons": list(DEFAULT_FORECAST_HORIZONS),
            "primary_metric": "forecast_300.nrmse",
        },
        "runtime": {
            "run_root": "./runs/sparse",
            "seed": 42,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "hash_data_files": True,
        },
    }


def _validate_experiment_contract(config: Mapping[str, Any]) -> None:
    experiment_id = str(config.get("experiment_id", "")).upper()
    if experiment_id not in SUPPORTED_EXPERIMENTS:
        raise ValueError(
            f"experiment_id must be one of {sorted(SUPPORTED_EXPERIMENTS)}, "
            f"got {experiment_id!r}"
        )
    experiment_name = str(config.get("experiment_name", "")).strip()
    if not experiment_name or any(token in experiment_name for token in ("/", "\\", "..")):
        raise ValueError("experiment_name must be a safe non-empty directory name")

    data = config["data"]
    if int(data["input_steps"]) != 60 or int(data["output_steps"]) != 30:
        raise ValueError("Sparse RFNO experiments require the established 60-to-30 contract")
    if int(data["rollout_steps"]) != 300:
        raise ValueError("Formal sparse RFNO evaluation must use 300-frame rollout")

    observation = config.get("observation")
    pipeline = config.get("pipeline")
    if not isinstance(observation, Mapping) or not isinstance(pipeline, Mapping):
        raise ValueError("observation and pipeline must be configuration objects")
    if not str(observation.get("mask_id", "")).strip():
        raise ValueError("A fixed observation.mask_id is required")
    exact_expected = {
        "B1": ("bilinear", "frozen"),
        "B3": ("pod", "frozen"),
        "B4": ("mask_unet", "frozen"),
        "B5-GNO": ("gno", "frozen"),
        "B5-GINO": ("gino", "frozen"),
        "H1": ("hybrid_pconv_unet", "frozen"),
        "H1-A1": ("hybrid_pconv_unet", "frozen"),
        "H1-A2": ("hybrid_pconv_unet", "frozen"),
        "H1-A3": ("hybrid_pconv_unet", "frozen"),
        "H1-A4": ("hybrid_pconv_unet", "frozen"),
        "H1-A5": ("hybrid_pconv_unet", "frozen"),
        "H1-A6": ("hybrid_pconv_unet", "frozen"),
        "P1": ("partialconv_mae", "frozen"),
        "P2": ("partialconv_mae", "joint"),
        "V0": ("vit_mae", "frozen"),
        "V1": ("vit_mae", "frozen"),
        "V2": ("tubelet_mae", "frozen"),
        "V3": ("vit_mae", "frozen"),
        "V4": ("vit_mae", "joint"),
        "PC-D1": ("vit_mask_aware_pool", "frozen"),
        "PC-D2": ("vit_mask_aware_confidence", "frozen"),
        "ST-D0": ("sensor_token_grid_query", "frozen"),
        "MU-V3": ("mask_unet", "frozen"),
        "FD-R1": ("fno_deeponet", "frozen"),
        "FD-A1": ("fno_deeponet", "joint"),
        "FD-L1": ("fno_deeponet", "direct"),
    }
    actual = (str(pipeline.get("reconstructor")), str(pipeline.get("coupling")))
    if experiment_id in exact_expected and actual != exact_expected[experiment_id]:
        raise ValueError(
            f"{experiment_id} requires reconstructor/coupling="
            f"{exact_expected[experiment_id]}, got {actual}"
        )
    if experiment_id in {"B5", "B6"}:
        expected_coupling = "frozen" if experiment_id == "B5" else "joint"
        if actual[0] not in {"gno", "gino"} or actual[1] != expected_coupling:
            raise ValueError(
                f"{experiment_id} requires gno/gino with coupling="
                f"{expected_coupling}, got {actual}"
            )
        if str(observation.get("mask_type")) != "fixed_points":
            raise ValueError(
                "Current GNO/GINO vertical slice requires a fixed_points layout"
            )
    if experiment_id in {
        "V0", "V1", "V2", "V3", "V4", "PC-D1", "PC-D2"
    }:
        model = config["model"]
        training = config.get("training", {})
        tokenization = str(model.get("vit_tokenization", "spatial_patch"))
        expected_tokenization = {
            "V2": "tubelet",
            "PC-D1": "spatial_mask_aware_pool",
            "PC-D2": "spatial_mask_aware_pool_confidence",
        }.get(experiment_id, "spatial_patch")
        if tokenization != expected_tokenization:
            raise ValueError(
                f"{experiment_id} requires model.vit_tokenization="
                f"{expected_tokenization!r}"
            )
        patch_size = int(model.get("vit_patch_size", 4))
        height = int(data.get("height", 64))
        width = int(data.get("width", 64))
        if patch_size <= 0 or height % patch_size or width % patch_size:
            raise ValueError(
                "model.vit_patch_size must divide the configured spatial dimensions"
            )
        if tokenization == "tubelet":
            tubelet_size = int(model.get("vit_tubelet_size", 10))
            if tubelet_size <= 0 or int(data["input_steps"]) % tubelet_size:
                raise ValueError(
                    "model.vit_tubelet_size must divide the 60 history frames"
                )
        mae_mask_ratio = float(training.get("mae_mask_ratio", 0.0))
        if not 0.0 <= mae_mask_ratio < 1.0:
            raise ValueError("training.mae_mask_ratio must be in [0,1)")
        forecast_supervision = bool(training.get("forecast_supervision", False))
        reconstructor_checkpoint = pipeline.get("reconstructor_checkpoint")
        if mae_mask_ratio > 0.0:
            if forecast_supervision:
                raise ValueError("MAE pretraining cannot use forecast supervision")
            if reconstructor_checkpoint:
                raise ValueError(
                    "MAE pretraining must start without reconstructor_checkpoint"
                )
        if experiment_id == "V0":
            if mae_mask_ratio != 0.0 or reconstructor_checkpoint:
                raise ValueError(
                    "V0 is the random-initialized sparse fine-tuning control"
                )
        if experiment_id in {"PC-D1", "PC-D2"}:
            if (
                bool(training.get("pretrain_reconstructor", False))
                or mae_mask_ratio != 0.0
                or reconstructor_checkpoint
                or forecast_supervision
            ):
                raise ValueError(
                    f"{experiment_id} infrastructure requires direct random initialization, "
                    "pretrain_reconstructor=false, mae_mask_ratio=0, no "
                    "reconstructor_checkpoint, and forecast_supervision=false"
                )
        if experiment_id in {"V1", "V2"} and mae_mask_ratio == 0.0:
            if not reconstructor_checkpoint:
                raise ValueError(
                    f"{experiment_id} sparse fine-tuning requires its MAE "
                    "pretraining checkpoint"
                )
        if experiment_id in {"V3", "V4"}:
            if mae_mask_ratio != 0.0 or not reconstructor_checkpoint:
                raise ValueError(
                    f"{experiment_id} requires a locked V1 reconstructor checkpoint"
                )
            if not forecast_supervision:
                raise ValueError(f"{experiment_id} requires forecast_supervision=true")
            if bool(training.get("use_rollout_curriculum", False)):
                if experiment_id == "V3":
                    raise ValueError("V3 frozen validation starts with fixed 30-frame loss")
            elif int(training.get("max_rollout_steps", 30)) != 30:
                raise ValueError(
                    f"{experiment_id} without curriculum requires max_rollout_steps=30"
                )
    if experiment_id == "ST-D0":
        model = config["model"]
        training = config.get("training", {})
        if (
            int(model.get("sensor_token_dim", 128)) != 128
            or int(model.get("sensor_encoder_depth", 4)) != 4
            or int(model.get("sensor_encoder_heads", 4)) != 4
        ):
            raise ValueError("ST-D0 requires sensor encoder dim=128/depth=4/heads=4")
        if bool(model.get("hard_observation_consistency", False)):
            raise ValueError("ST-D0 main experiment disables hard observation consistency")
        if (
            bool(training.get("pretrain_reconstructor", False))
            or float(training.get("mae_mask_ratio", 0.0)) != 0.0
            or pipeline.get("reconstructor_checkpoint")
            or bool(training.get("forecast_supervision", False))
        ):
            raise ValueError(
                "ST-D0 infrastructure requires direct random initialization, "
                "pretrain_reconstructor=false, mae_mask_ratio=0, no "
                "reconstructor_checkpoint, and forecast_supervision=false"
            )
    if experiment_id == "MU-V3":
        training = config.get("training", {})
        if (
            bool(training.get("pretrain_reconstructor", False))
            or float(training.get("mae_mask_ratio", 0.0)) != 0.0
            or not pipeline.get("reconstructor_checkpoint")
        ):
            raise ValueError(
                "MU-V3 requires pretrain_reconstructor=false, mae_mask_ratio=0, "
                "and a locked Mask U-Net reconstructor checkpoint"
            )
        if not bool(training.get("forecast_supervision", False)):
            raise ValueError("MU-V3 requires forecast_supervision=true")
        if bool(training.get("use_rollout_curriculum", False)):
            raise ValueError("MU-V3 seed gate requires a fixed 30-frame rollout")
        if int(training.get("max_rollout_steps", 30)) != 30:
            raise ValueError("MU-V3 requires max_rollout_steps=30")
    if experiment_id in {"FD-R1", "FD-A1", "FD-L1"}:
        training = config.get("training", {})
        expected_forecaster = {
            "FD-R1": "rfno",
            "FD-A1": "fno_deeponet_ar",
            "FD-L1": "fno_deeponet_direct",
        }[experiment_id]
        if str(pipeline.get("forecaster")) != expected_forecaster:
            raise ValueError(
                f"{experiment_id} requires pipeline.forecaster={expected_forecaster!r}"
            )
        forecast_supervision = bool(training.get("forecast_supervision", False))
        if experiment_id == "FD-R1" and forecast_supervision:
            raise ValueError("FD-R1 trains reconstruction only before frozen-RFNO scoring")
        if experiment_id in {"FD-A1", "FD-L1"} and not forecast_supervision:
            raise ValueError(f"{experiment_id} requires forecast_supervision=true")
        if experiment_id == "FD-A1":
            if not bool(training.get("use_rollout_curriculum", False)):
                raise ValueError("FD-A1 requires curriculum-based autoregressive training")
            if int(training.get("max_rollout_steps", 0)) != 300:
                raise ValueError("FD-A1 curriculum must reach the 300-frame rollout")
        if experiment_id == "FD-L1":
            if bool(training.get("use_rollout_curriculum", False)):
                raise ValueError("FD-L1 is a direct one-shot, non-autoregressive experiment")
            if int(training.get("max_rollout_steps", 0)) != 300:
                raise ValueError("FD-L1 must decode the complete 300-frame forecast")
    if not bool(config["runtime"].get("hash_data_files", False)):
        raise ValueError("Formal sparse experiments require runtime.hash_data_files=true")


def load_resolved_sparse_config(
    config_path: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load, validate and resolve every path and scientific default."""

    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Sparse experiment config not found: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("Sparse experiment config root must be an object")
    resolved = _default_config()
    _deep_merge(resolved, raw)
    project_root_path = (
        Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    )
    _validate_experiment_contract(resolved)

    data = resolved["data"]
    data_root = _resolve_path(data["data_root"], project_root_path)
    data["data_root"] = str(data_root)
    for split, key in (("train", "train_dir"), ("val", "val_dir"), ("test", "test_dir")):
        value = data.get(key)
        data[key] = str(
            _resolve_path(value, project_root_path)
            if value
            else (data_root / split).resolve()
        )
    split_manifest_path = data.get("split_manifest_path")
    if split_manifest_path:
        data["split_manifest_path"] = str(
            _resolve_path(split_manifest_path, project_root_path)
        )

    observation = resolved["observation"]
    manifest_path = _resolve_path(observation["manifest_path"], project_root_path)
    observation["manifest_path"] = str(manifest_path)
    manifest = SparseMaskManifest(manifest_path)
    mask_id = str(observation["mask_id"])
    mask_metadata = manifest.metadata(mask_id)
    if mask_metadata.mask_type != observation.get("mask_type"):
        raise ValueError(
            f"Configured mask_type {observation.get('mask_type')!r} does not match "
            f"manifest {mask_metadata.mask_type!r}"
        )
    requested_rate = mask_metadata.requested_observation_rate
    if requested_rate is not None and not math.isclose(
        float(requested_rate),
        float(observation["observation_rate"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Configured observation rate differs from selected mask metadata")
    configured_seed = observation.get("seed")
    if (
        mask_metadata.seed is not None
        and configured_seed is not None
        and int(mask_metadata.seed) != int(configured_seed)
    ):
        raise ValueError("Configured observation seed differs from selected mask metadata")
    configured_dropout = float(observation.get("temporal_dropout", 0.0))
    if not math.isclose(
        configured_dropout,
        mask_metadata.temporal_dropout,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "Configured temporal_dropout differs from selected mask metadata"
        )
    noise_fraction = float(observation.get("noise_std_fraction", 0.0))
    if noise_fraction < 0.0:
        raise ValueError("observation.noise_std_fraction must be non-negative")
    if noise_fraction != 0.0:
        raise NotImplementedError(
            "Observation noise is not yet implemented in the shared sparse dataset; "
            "refusing to run a mislabeled non-zero-noise experiment"
        )

    training = resolved["training"]
    pretraining_manifest_path = training.get("mae_mask_manifest_path")
    if pretraining_manifest_path:
        pretraining_manifest_path = _resolve_path(
            pretraining_manifest_path, project_root_path
        )
        if not pretraining_manifest_path.is_file():
            raise FileNotFoundError(
                f"MAE mask manifest not found: {pretraining_manifest_path}"
            )
        pretraining_manifest = json.loads(
            pretraining_manifest_path.read_text(encoding="utf-8")
        )
        expected = {
            "mask_id": str(training.get("mae_mask_id", "")),
            "mask_ratio": float(training.get("mae_mask_ratio", 0.0)),
            "patch_size": int(resolved["model"].get("vit_patch_size", 4)),
            "tokenization": str(
                resolved["model"].get("vit_tokenization", "spatial_patch")
            ),
            "training_seed": int(
                training.get("mae_mask_seed", resolved["runtime"].get("seed", 42))
            ),
            "validation_seed": int(
                training.get(
                    "mae_validation_mask_seed",
                    int(resolved["runtime"].get("seed", 42)) + 1729,
                )
            ),
        }
        for field, expected_value in expected.items():
            if pretraining_manifest.get(field) != expected_value:
                raise ValueError(
                    f"MAE mask manifest {field} differs from config: "
                    f"{pretraining_manifest.get(field)!r} != {expected_value!r}"
                )
        if pretraining_manifest.get("mask_semantics") != "1=observed,0=missing":
            raise ValueError("MAE mask manifest uses incompatible mask semantics")
        training["mae_mask_manifest_path"] = str(pretraining_manifest_path)

    pipeline = resolved["pipeline"]
    checkpoint_path = _resolve_path(pipeline["rfno_checkpoint"], project_root_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"B0 checkpoint not found: {checkpoint_path}")
    pipeline["rfno_checkpoint"] = str(checkpoint_path)
    reconstructor_checkpoint = pipeline.get("reconstructor_checkpoint")
    if reconstructor_checkpoint:
        reconstructor_path = _resolve_path(reconstructor_checkpoint, project_root_path)
        if not reconstructor_path.is_file():
            raise FileNotFoundError(
                f"Reconstructor checkpoint not found: {reconstructor_path}"
            )
        pipeline["reconstructor_checkpoint"] = str(reconstructor_path)
    continuation_checkpoint = pipeline.get("continuation_checkpoint")
    if continuation_checkpoint:
        continuation_path = _resolve_path(continuation_checkpoint, project_root_path)
        if not continuation_path.is_file():
            raise FileNotFoundError(
                f"Continuation checkpoint not found: {continuation_path}"
            )
        pipeline["continuation_checkpoint"] = str(continuation_path)
    pod_basis_path = pipeline.get("pod_basis_path")
    if pod_basis_path:
        pipeline["pod_basis_path"] = str(
            _resolve_path(pod_basis_path, project_root_path)
        )

    mean, std = read_b0_normalization(checkpoint_path)
    resolved["normalization"] = {
        "source": "b0_checkpoint_train_only",
        "mean": mean,
        "std": std,
    }

    runtime = resolved["runtime"]
    runtime["run_root"] = str(_resolve_path(runtime["run_root"], project_root_path))
    runtime["device"] = str(device or runtime["device"])
    if run_dir is not None:
        runtime["run_dir"] = str(_resolve_path(run_dir, project_root_path))
    else:
        runtime["run_dir"] = str(
            Path(runtime["run_root"]) / str(resolved["experiment_name"])
        )

    horizons = tuple(int(value) for value in resolved["evaluation"]["forecast_horizons"])
    if not horizons or 30 not in horizons or 300 not in horizons:
        raise ValueError("evaluation.forecast_horizons must include both 30 and 300")
    if min(horizons) <= 0 or max(horizons) > int(data["rollout_steps"]):
        raise ValueError("evaluation.forecast_horizons are outside the rollout range")
    resolved["evaluation"]["forecast_horizons"] = list(sorted(set(horizons)))
    resolved["_meta"] = {
        "source_config": str(config_path),
        "project_root": str(project_root_path),
        "schema_version": 1,
    }
    return resolved


def scientific_config_sha256(config: Mapping[str, Any]) -> str:
    scientific = {
        key: config[key]
        for key in (
            "experiment_id",
            "experiment_name",
            "data",
            "observation",
            "pipeline",
            "model",
            "training",
            "evaluation",
            "normalization",
        )
        if key in config
    }
    experiment_id = str(config.get("experiment_id", "")).upper()
    model = copy.deepcopy(scientific.get("model", {}))
    for key, default_value in _LEGACY_UNUSED_MODEL_DEFAULTS.get(
        experiment_id, {}
    ).items():
        if key in model and model[key] == default_value:
            del model[key]
    if "model" in scientific:
        scientific["model"] = model
    # Device and run location are operational. Seed and content hashing are scientific
    # controls and must distinguish checkpoints that are safe to resume.
    runtime = config.get("runtime", {})
    scientific["runtime"] = {
        "seed": int(runtime["seed"]),
        "hash_data_files": bool(runtime.get("hash_data_files", False)),
    }
    return _sha256_bytes(_canonical_json(scientific).encode("utf-8"))


def _mat_entries(
    directory: Path, *, data_root: Path, include_hash: bool
) -> Sequence[Dict[str, Any]]:
    files = sorted(directory.glob("*.mat"), key=lambda path: path.name)
    if not files:
        raise ValueError(f"No MAT files found in split directory: {directory}")
    entries = []
    for source_id, path in enumerate(files):
        entry: Dict[str, Any] = {
            "source_id": source_id,
            "name": path.name,
            "relative_path": str(path.relative_to(data_root)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
        }
        if include_hash:
            entry["sha256"] = sha256_file(path)
        entries.append(entry)
    return entries


def freeze_data_split(config: Mapping[str, Any]) -> Dict[str, Any]:
    data = config["data"]
    data_root = Path(data["data_root"])
    include_hash = bool(config["runtime"].get("hash_data_files", False))
    splits = {
        split: _mat_entries(
            Path(data[f"{split}_dir"]),
            data_root=data_root,
            include_hash=include_hash,
        )
        for split in ("train", "val", "test")
    }
    names = {split: {entry["name"] for entry in entries} for split, entries in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = names[left] & names[right]
        if overlap:
            raise ValueError(
                f"MAT filename leakage between {left} and {right}: {sorted(overlap)}"
            )
    if include_hash:
        hashes = {
            split: {entry["sha256"] for entry in entries} for split, entries in splits.items()
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = hashes[left] & hashes[right]
            if overlap:
                raise ValueError(f"MAT content leakage between {left} and {right}")
    return {
        "data_root": str(data_root),
        "hash_data_files": include_hash,
        "splits": splits,
        "counts": {split: len(entries) for split, entries in splits.items()},
    }


def load_frozen_data_split(
    manifest_path: Union[str, Path],
    config: Mapping[str, Any],
    *,
    verify_splits: Sequence[str] = ("train", "val"),
) -> Dict[str, Any]:
    """Load a pre-frozen split and verify only explicitly requested MAT files.

    Stored hashes for all splits are checked for cross-split overlap using JSON metadata.
    File bytes are read only for ``verify_splits``; a train/val preparation therefore
    never opens test MAT files.
    """

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen data split not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("Frozen data split root must be an object")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "val", "test"}:
        raise ValueError("Frozen data split must contain train, val and test entries")
    if not bool(manifest.get("hash_data_files", False)):
        raise ValueError("Frozen data split must contain MAT content hashes")

    configured_root = Path(config["data"]["data_root"]).resolve()
    frozen_root = Path(str(manifest.get("data_root", "")))
    if not frozen_root.is_absolute():
        frozen_root = (path.parent / frozen_root).resolve()
    else:
        frozen_root = frozen_root.resolve()
    if frozen_root != configured_root:
        raise ValueError(
            f"Frozen data_root {frozen_root} differs from configured {configured_root}"
        )

    hashes: Dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        entries = splits[split]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Frozen split {split} must contain MAT entries")
        split_hashes = {str(entry.get("sha256", "")) for entry in entries}
        if "" in split_hashes or len(split_hashes) != len(entries):
            raise ValueError(f"Frozen split {split} has missing or duplicate hashes")
        hashes[split] = split_hashes
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if hashes[left] & hashes[right]:
            raise ValueError(f"Frozen MAT content leakage between {left} and {right}")

    requested = tuple(dict.fromkeys(str(split) for split in verify_splits))
    invalid = set(requested) - {"train", "val", "test"}
    if invalid:
        raise ValueError(f"Invalid frozen split verification request: {sorted(invalid)}")
    for split in requested:
        entries = splits[split]
        split_dir = Path(config["data"][f"{split}_dir"])
        current_names = {
            candidate.name for candidate in split_dir.glob("*.mat") if candidate.is_file()
        }
        frozen_names = {str(entry["name"]) for entry in entries}
        if current_names != frozen_names:
            raise ValueError(f"Current {split} MAT list differs from frozen manifest")
        for entry in entries:
            candidate = (frozen_root / str(entry["relative_path"])).resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"Frozen {split} MAT file not found: {candidate}")
            if candidate.stat().st_size != int(entry["size_bytes"]):
                raise ValueError(f"Frozen {split} MAT size changed: {candidate.name}")
            if sha256_file(candidate) != str(entry["sha256"]):
                raise ValueError(f"Frozen {split} MAT hash changed: {candidate.name}")

    result = copy.deepcopy(dict(manifest))
    result["data_root"] = str(frozen_root)
    return result


@dataclass
class SparseDataBundle:
    dense_datasets: Dict[str, SeaSurfaceSimpleDataset]
    sparse_datasets: Dict[str, SparseSeaSurfaceDataset]
    loaders: Dict[str, DataLoader]
    generators: Dict[str, torch.Generator]
    split_manifest: Dict[str, Any]
    normalization_mean: float
    normalization_std: float


def build_sparse_dataloaders(
    config: Mapping[str, Any],
    split_manifest: Optional[Dict[str, Any]] = None,
    *,
    splits: Sequence[str] = ("train", "val"),
    include_targets: bool = True,
) -> SparseDataBundle:
    """Build requested splits from the frozen MAT list using B0 train statistics."""

    data = config["data"]
    normalization = config["normalization"]
    mean = float(normalization["mean"])
    std = float(normalization["std"])
    manifest = split_manifest or freeze_data_split(config)
    dense: Dict[str, SeaSurfaceSimpleDataset] = {}
    sparse: Dict[str, SparseSeaSurfaceDataset] = {}
    loaders: Dict[str, DataLoader] = {}
    generators: Dict[str, torch.Generator] = {}
    seed = int(config["runtime"]["seed"])
    batch_sizes = {
        "train": int(data["batch_size"]),
        "val": int(data["val_batch_size"]),
        "test": int(data["test_batch_size"]),
    }

    requested_splits = tuple(dict.fromkeys(str(split) for split in splits))
    invalid = set(requested_splits) - {"train", "val", "test"}
    if invalid or not requested_splits:
        raise ValueError(f"Invalid requested data splits: {sorted(invalid)}")

    data_root = Path(manifest["data_root"])
    for split in requested_splits:
        entries = manifest["splits"][split]
        frozen_files = [
            str((data_root / entry["relative_path"]).resolve()) for entry in entries
        ]
        dense_dataset = SeaSurfaceSimpleDataset(
            data_dir=str(data[f"{split}_dir"]),
            variable=data.get("variable"),
            input_steps=int(data["input_steps"]),
            output_steps=int(data["rollout_steps"]),
            stride=int(data["stride"]),
            normalize=True,
            mean=mean,
            std=std,
            mat_files=frozen_files,
        )
        if dense_dataset.mean != mean or dense_dataset.std != std:
            raise RuntimeError(f"{split} dataset did not retain B0 train normalization")
        sparse_dataset = SparseSeaSurfaceDataset(
            dense_dataset,
            config["observation"]["manifest_path"],
            mask_ids=[str(config["observation"]["mask_id"])],
            include_target=include_targets,
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            sparse_dataset,
            batch_size=batch_sizes[split],
            shuffle=split == "train",
            num_workers=int(data["num_workers"]),
            pin_memory=bool(data["pin_memory"]),
            generator=generator,
        )
        dense[split] = dense_dataset
        sparse[split] = sparse_dataset
        loaders[split] = loader
        generators[split] = generator
    return SparseDataBundle(dense, sparse, loaders, generators, manifest, mean, std)


def _git_provenance(project_root: Path) -> Dict[str, Any]:
    def run(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=str(project_root),
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    status = run("status", "--porcelain")
    return {
        "revision": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "working_tree_dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def collect_provenance(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    config_sha256: str,
    split_manifest_source: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    checkpoint_path = Path(config["pipeline"]["rfno_checkpoint"])
    manifest = SparseMaskManifest(config["observation"]["manifest_path"])
    cuda_available = torch.cuda.is_available()
    provenance = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "scientific_config_sha256": config_sha256,
        "git": _git_provenance(project_root),
        "files": {
            "source_config": {
                "path": config["_meta"]["source_config"],
                "sha256": sha256_file(config["_meta"]["source_config"]),
            },
            "rfno_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "mask_manifest": {
                "path": str(manifest.manifest_path),
                "sha256": sha256_file(manifest.manifest_path),
            },
            "mask_archive": {
                "path": str(manifest.npz_path),
                "sha256": sha256_file(manifest.npz_path),
            },
        },
        "normalization": dict(config["normalization"]),
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else [],
        },
    }
    pretraining_manifest_path = config.get("training", {}).get(
        "mae_mask_manifest_path"
    )
    if pretraining_manifest_path:
        mae_manifest_path = Path(pretraining_manifest_path)
        provenance["files"]["mae_mask_manifest"] = {
            "path": str(mae_manifest_path),
            "sha256": sha256_file(mae_manifest_path),
        }
    for name in ("reconstructor_checkpoint", "continuation_checkpoint"):
        optional_checkpoint = config["pipeline"].get(name)
        if optional_checkpoint:
            optional_path = Path(optional_checkpoint)
            provenance["files"][name] = {
                "path": str(optional_path),
                "sha256": sha256_file(optional_path),
            }
    if split_manifest_source is not None:
        source_path = Path(split_manifest_source).resolve()
        provenance["files"]["data_split_source"] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        }
    return provenance


def validate_continuation_checkpoint(
    config: Mapping[str, Any], checkpoint_path: Union[str, Path]
) -> Dict[str, Any]:
    """Validate a controlled D1->D2 continuation without weakening normal resume.

    A continuation may change only curriculum/checkpoint-selection controls. Model,
    data, observation, optimizer, scheduler, loss, normalization, and seed contracts
    must remain identical so that restoring optimizer/RNG state is meaningful.
    """

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Continuation checkpoint not found: {checkpoint_path}")
    parent_run_dir = checkpoint_path.parent.parent
    parent_config_path = parent_run_dir / "config.resolved.json"
    parent_split_path = parent_run_dir / "data_split.json"
    if not parent_config_path.is_file() or not parent_split_path.is_file():
        raise FileNotFoundError(
            "Continuation checkpoint must belong to a frozen run containing "
            "config.resolved.json and data_split.json"
        )
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    checkpoint = _load_checkpoint_mapping(checkpoint_path)
    parent_hash = scientific_config_sha256(parent_config)
    if checkpoint.get("scientific_config_sha256") != parent_hash:
        raise ValueError("Continuation checkpoint does not match its parent run config")
    if not bool(checkpoint.get("complete_training_state", False)):
        raise ValueError("Continuation checkpoint lacks complete training state")

    allowed_training_changes = {
        "primary_loss",
        "primary_mode",
        "early_stopping_patience",
        "use_rollout_curriculum",
        "rollout_train_steps",
        "rollout_curriculum_boundaries",
        "max_rollout_steps",
        "abort_gradient_norm",
        "max_peak_memory_mb",
    }

    def contract(value: Mapping[str, Any]) -> Dict[str, Any]:
        training = {
            key: item
            for key, item in value.get("training", {}).items()
            if key not in allowed_training_changes
        }
        pipeline = {
            key: item
            for key, item in value.get("pipeline", {}).items()
            if key != "continuation_checkpoint"
        }
        evaluation = {
            key: item
            for key, item in value.get("evaluation", {}).items()
            if key != "primary_metric"
        }
        runtime = value.get("runtime", {})
        return {
            "experiment_id": str(value.get("experiment_id", "")).upper(),
            "data": value.get("data", {}),
            "observation": value.get("observation", {}),
            "pipeline": pipeline,
            "model": value.get("model", {}),
            "training": training,
            "evaluation": evaluation,
            "normalization": value.get("normalization", {}),
            "runtime": {
                "seed": runtime.get("seed"),
                "hash_data_files": runtime.get("hash_data_files"),
            },
        }

    current_contract = contract(config)
    parent_contract = contract(parent_config)
    differing = [
        key
        for key in current_contract
        if _canonical_json(current_contract[key]) != _canonical_json(parent_contract[key])
    ]
    if differing:
        raise ValueError(
            "Continuation changed protected scientific controls: "
            f"{differing}"
        )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parent_run_dir": str(parent_run_dir),
        "parent_config_sha256": parent_hash,
        "parent_split_path": str(parent_split_path),
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
    }


@dataclass
class SparseExperimentContext:
    config: Dict[str, Any]
    run_dir: Path
    config_sha256: str
    provenance: Dict[str, Any]
    data: SparseDataBundle
    resumed: bool = False


def prepare_sparse_experiment(
    config_path: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    resume_checkpoint: Optional[Union[str, Path]] = None,
    existing_run_dir: Optional[Union[str, Path]] = None,
    loader_splits: Sequence[str] = ("train", "val"),
    frozen_split_manifest: Optional[Union[str, Path]] = None,
    training_mode: bool = False,
) -> SparseExperimentContext:
    project_root_path = (
        Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    )
    resolved = load_resolved_sparse_config(
        config_path, project_root=project_root_path, run_dir=run_dir, device=device
    )
    config_hash = scientific_config_sha256(resolved)
    if resume_checkpoint is not None and existing_run_dir is not None:
        raise ValueError("Specify only one of resume_checkpoint and existing_run_dir")
    resumed = resume_checkpoint is not None or existing_run_dir is not None
    if resumed:
        if resume_checkpoint is not None:
            resume_path = Path(resume_checkpoint).resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            target_run_dir = resume_path.parent.parent
        else:
            target_run_dir = Path(existing_run_dir).resolve()  # type: ignore[arg-type]
            if not target_run_dir.is_dir():
                raise FileNotFoundError(f"Existing run directory not found: {target_run_dir}")
        existing_config_path = target_run_dir / "config.resolved.json"
        if not existing_config_path.is_file():
            raise FileNotFoundError(
                f"Resume run lacks config.resolved.json: {target_run_dir}"
            )
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        existing_hash = scientific_config_sha256(existing_config)
        if config_hash != existing_hash:
            raise ValueError("Resume config differs from the frozen scientific config")
        resolved = existing_config
        config_hash = existing_hash
        if device is not None:
            resolved["runtime"]["device"] = str(device)
    else:
        target_run_dir = Path(resolved["runtime"]["run_dir"])

    if resumed:
        saved_split_path = target_run_dir / "data_split.json"
        if not saved_split_path.is_file():
            raise FileNotFoundError(f"Resume run lacks data_split.json: {target_run_dir}")
        split_manifest = load_frozen_data_split(
            saved_split_path, resolved, verify_splits=loader_splits
        )
        split_manifest_source: Optional[Path] = saved_split_path
    elif frozen_split_manifest is not None or resolved["data"].get("split_manifest_path"):
        configured_manifest = (
            frozen_split_manifest or resolved["data"]["split_manifest_path"]
        )
        split_manifest_source = Path(configured_manifest).resolve()
        split_manifest = load_frozen_data_split(
            split_manifest_source, resolved, verify_splits=loader_splits
        )
    else:
        split_manifest_source = None
        split_manifest = freeze_data_split(resolved)
    include_targets = (
        bool(resolved.get("training", {}).get("forecast_supervision", False))
        if training_mode
        else True
    )
    data = build_sparse_dataloaders(
        resolved,
        split_manifest,
        splits=loader_splits,
        include_targets=include_targets,
    )
    provenance = collect_provenance(
        resolved,
        project_root=project_root_path,
        config_sha256=config_hash,
        split_manifest_source=split_manifest_source,
    )
    if not resumed:
        target_run_dir.mkdir(parents=True, exist_ok=False)
        for child in ("checkpoints", "logs", "evaluation"):
            (target_run_dir / child).mkdir()
        _write_json(target_run_dir / "config.resolved.json", resolved)
        _write_json(target_run_dir / "data_split.json", split_manifest)
        _write_json(target_run_dir / "provenance.json", provenance)
    return SparseExperimentContext(
        config=resolved,
        run_dir=target_run_dir,
        config_sha256=config_hash,
        provenance=provenance,
        data=data,
        resumed=resumed,
    )


def _load_reconstructor_state(reconstructor: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = _load_checkpoint_mapping(checkpoint_path)
    state = checkpoint.get("reconstructor_state_dict")
    if state is None:
        state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError(
            "Reconstructor checkpoint requires reconstructor_state_dict or model_state_dict"
        )
    if any(str(key).startswith("reconstructor.") for key in state):
        state = {
            str(key)[len("reconstructor.") :]: value
            for key, value in state.items()
            if str(key).startswith("reconstructor.")
        }
    reconstructor.load_state_dict(state, strict=True)


def build_sparse_pipeline(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    experiment_id = str(config["experiment_id"]).upper()
    checkpoint = config["pipeline"]["rfno_checkpoint"]
    if experiment_id == "B1":
        return BilinearFrozenRFNOPipeline.from_b0_checkpoint(
            checkpoint, map_location=device
        ).to(device)

    if experiment_id == "B3":
        basis_path = config["pipeline"].get("pod_basis_path")
        if not basis_path:
            raise ValueError("B3 requires pipeline.pod_basis_path")
        reconstructor = PODSpatialReconstructor.from_artifact(
            basis_path,
            rank=int(config["model"]["pod_rank"]),
            ridge=float(config["model"].get("pod_ridge", 1.0e-6)),
        )
        return BilinearFrozenRFNOPipeline.from_b0_checkpoint(
            checkpoint,
            map_location=device,
            reconstructor=reconstructor,
        ).to(device)

    model_config = config["model"]
    if experiment_id == "FD-L1":
        direct_operator = FNODeepONet(
            **fno_deeponet_kwargs(
                model_config,
                input_steps=int(config["data"]["input_steps"]),
                height=int(config["data"].get("height", 64)),
                width=int(config["data"].get("width", 64)),
            )
        )
        mean, std = read_b0_normalization(checkpoint)
        return FNODeepONetDirectSparsePipeline(
            direct_operator,
            forecast_steps=int(config["data"]["rollout_steps"]),
            hard_observation_consistency=bool(
                model_config.get("hard_observation_consistency", False)
            ),
            normalization_mean=mean,
            normalization_std=std,
            checkpoint_path=checkpoint,
        ).to(device)
    if experiment_id in {"B4", "MU-V3"}:
        reconstructor = PeriodicMaskUNet(
            input_steps=int(config["data"]["input_steps"]),
            channels=tuple(model_config["mask_unet_channels"]),
        )
    elif experiment_id in {
        "H1", "H1-A1", "H1-A2", "H1-A3", "H1-A4", "H1-A5", "H1-A6"
    }:
        reconstructor = PeriodicHybridPartialConvUNet(
            input_steps=int(config["data"]["input_steps"]),
            channels=tuple(model_config["hybrid_unet_channels"]),
            partialconv_levels=int(model_config.get("partialconv_levels", 1)),
            use_skip_connections=bool(
                model_config.get("use_skip_connections", True)
            ),
        )
    elif experiment_id in {"P1", "P2"}:
        reconstructor = PartialConvMaskedAutoencoder(
            input_steps=int(config["data"]["input_steps"]),
            encoder_channels=tuple(model_config["partialconv_encoder_channels"]),
            decoder_channels=tuple(model_config["partialconv_decoder_channels"]),
        )
    elif experiment_id in {"V0", "V1", "V2", "V3", "V4"}:
        reconstructor = PeriodicViTMaskedAutoencoder(
            input_steps=int(config["data"]["input_steps"]),
            patch_size=int(model_config.get("vit_patch_size", 4)),
            tokenization=str(
                model_config.get("vit_tokenization", "spatial_patch")
            ),
            tubelet_size=int(model_config.get("vit_tubelet_size", 10)),
            encoder_dim=int(model_config.get("vit_encoder_dim", 128)),
            encoder_depth=int(model_config.get("vit_encoder_depth", 4)),
            encoder_heads=int(model_config.get("vit_encoder_heads", 4)),
            decoder_dim=int(model_config.get("vit_decoder_dim", 64)),
            decoder_depth=int(model_config.get("vit_decoder_depth", 2)),
            decoder_heads=int(model_config.get("vit_decoder_heads", 4)),
            mlp_ratio=float(model_config.get("vit_mlp_ratio", 4.0)),
        )
    elif experiment_id == "PC-D1":
        reconstructor = PeriodicMaskAwarePoolingViTReconstructor(
            input_steps=int(config["data"]["input_steps"]),
            patch_size=int(model_config.get("vit_patch_size", 4)),
            tokenization=str(
                model_config.get(
                    "vit_tokenization", "spatial_mask_aware_pool"
                )
            ),
            tubelet_size=int(model_config.get("vit_tubelet_size", 10)),
            encoder_dim=int(model_config.get("vit_encoder_dim", 128)),
            encoder_depth=int(model_config.get("vit_encoder_depth", 4)),
            encoder_heads=int(model_config.get("vit_encoder_heads", 4)),
            decoder_dim=int(model_config.get("vit_decoder_dim", 64)),
            decoder_depth=int(model_config.get("vit_decoder_depth", 2)),
            decoder_heads=int(model_config.get("vit_decoder_heads", 4)),
            mlp_ratio=float(model_config.get("vit_mlp_ratio", 4.0)),
        )
    elif experiment_id == "PC-D2":
        reconstructor = PeriodicConfidenceMaskAwarePoolingViTReconstructor(
            input_steps=int(config["data"]["input_steps"]),
            patch_size=int(model_config.get("vit_patch_size", 4)),
            tokenization=str(
                model_config.get(
                    "vit_tokenization",
                    "spatial_mask_aware_pool_confidence",
                )
            ),
            tubelet_size=int(model_config.get("vit_tubelet_size", 10)),
            encoder_dim=int(model_config.get("vit_encoder_dim", 128)),
            encoder_depth=int(model_config.get("vit_encoder_depth", 4)),
            encoder_heads=int(model_config.get("vit_encoder_heads", 4)),
            decoder_dim=int(model_config.get("vit_decoder_dim", 64)),
            decoder_depth=int(model_config.get("vit_decoder_depth", 2)),
            decoder_heads=int(model_config.get("vit_decoder_heads", 4)),
            mlp_ratio=float(model_config.get("vit_mlp_ratio", 4.0)),
        )
    elif experiment_id == "ST-D0":
        reconstructor = PeriodicSensorTokenGridQueryReconstructor(
            input_steps=int(config["data"]["input_steps"]),
            dimension=int(model_config.get("sensor_token_dim", 128)),
            encoder_depth=int(model_config.get("sensor_encoder_depth", 4)),
            encoder_heads=int(model_config.get("sensor_encoder_heads", 4)),
            encoder_mlp_ratio=float(
                model_config.get("sensor_encoder_mlp_ratio", 4.0)
            ),
            decoder_depth=int(model_config.get("sensor_decoder_depth", 3)),
            decoder_heads=int(model_config.get("sensor_decoder_heads", 4)),
            decoder_mlp_ratio=float(
                model_config.get("sensor_decoder_mlp_ratio", 2.0)
            ),
            coordinate_bands=int(
                model_config.get("sensor_coordinate_fourier_bands", 8)
            ),
            query_chunk_size=int(
                model_config.get("sensor_query_chunk_size", 512)
            ),
            hard_observation_consistency=bool(
                model_config.get("hard_observation_consistency", False)
            ),
        )
    elif experiment_id in {"FD-R1", "FD-A1"}:
        reconstructor = FNODeepONetReconstructor(
            FNODeepONet(
                **fno_deeponet_kwargs(
                    model_config,
                    input_steps=int(config["data"]["input_steps"]),
                    height=int(config["data"].get("height", 64)),
                    width=int(config["data"].get("width", 64)),
                )
            ),
            hard_observation_consistency=bool(
                model_config.get("hard_observation_consistency", False)
            ),
        )
    elif experiment_id in {"B5", "B5-GNO", "B5-GINO", "B6"}:
        common = {
            "input_steps": int(config["data"]["input_steps"]),
            "hidden_channels": int(model_config.get("operator_hidden_channels", 32)),
            "radius": float(model_config.get("gno_radius", 0.20)),
            "mlp_channels": tuple(model_config.get("gno_mlp_channels", (64, 64))),
            # Resolved configs written before the nonlinear GNO upgrade omit this
            # field and must continue to construct the checkpoint-compatible model.
            "kernel_mode": str(
                model_config.get("gno_kernel_mode", "scalar_legacy")
            ),
            "kernel_rank": int(model_config.get("gno_kernel_rank", 16)),
            "hard_observation_consistency": bool(
                model_config.get("hard_observation_consistency", False)
            ),
        }
        if str(config["pipeline"]["reconstructor"]) == "gno":
            reconstructor = PeriodicGNOReconstructor(
                **common,
                architecture=str(
                    model_config.get("gno_architecture", "single_integral_v2")
                ),
                temporal_channels=int(
                    model_config.get("gno_temporal_channels", 32)
                ),
                temporal_dilations=tuple(
                    model_config.get("gno_temporal_dilations", (1, 2, 4))
                ),
                graph_layers=int(model_config.get("gno_graph_layers", 3)),
                grid_refinement_layers=int(
                    model_config.get("gno_grid_refinement_layers", 3)
                ),
                kernel_normalization=str(
                    model_config.get("gno_kernel_normalization", "l1")
                ),
                query_chunk_size=int(
                    model_config.get("gno_query_chunk_size", 512)
                ),
                raw_observation_supervision=bool(
                    model_config.get("raw_observation_supervision", False)
                ),
                temporal_tokens=int(
                    model_config.get("gno_temporal_tokens", 4)
                ),
                relative_fourier_frequencies=tuple(
                    model_config.get("gno_relative_fourier_frequencies", ())
                ),
                cache_geometry=bool(
                    model_config.get("gno_cache_geometry", False)
                ),
            )
        else:
            reconstructor = PeriodicGINOReconstructor(
                **common,
                latent_shape=tuple(model_config.get("latent_shape", (16, 16))),
                n_modes=tuple(model_config.get("latent_n_modes", (8, 8))),
                fno_layers=int(model_config.get("latent_fno_layers", 3)),
                latent_positional_embedding=str(
                    model_config.get("latent_positional_embedding", "grid")
                ),
            )
    else:
        raise AssertionError(f"Unhandled learned experiment: {experiment_id}")
    reconstructor_checkpoint = config["pipeline"].get("reconstructor_checkpoint")
    if reconstructor_checkpoint:
        _load_reconstructor_state(reconstructor, Path(reconstructor_checkpoint))
    if experiment_id == "FD-A1":
        forecaster_operator = FNODeepONet(
            **fno_deeponet_kwargs(
                model_config,
                input_steps=int(config["data"]["input_steps"]),
                height=int(config["data"].get("height", 64)),
                width=int(config["data"].get("width", 64)),
                prefix="fno_deeponet_forecaster",
            )
        )
        forecaster = FNODeepONetGridForecaster(
            forecaster_operator,
            output_steps=int(config["data"]["output_steps"]),
        )
        mean, std = read_b0_normalization(checkpoint)
        return LearnedReconstructionRFNOPipeline(
            forecaster,
            reconstructor,
            input_steps=int(config["data"]["input_steps"]),
            output_steps=int(config["data"]["output_steps"]),
            normalization_mean=mean,
            normalization_std=std,
            coupling="joint",
            checkpoint_path=checkpoint,
            checkpoint_config={"model_arch": "fno_deeponet_ar"},
        ).to(device)
    return PartialConvMAERFNOPipeline.from_b0_checkpoint(
        checkpoint,
        reconstructor,
        map_location=device,
        coupling=str(config["pipeline"]["coupling"]),
    ).to(device)


def coupled_training_config(config: Mapping[str, Any]) -> SparseCoupledTrainingConfig:
    training = config.get("training", {})
    loss_weights = training.get("loss_weights", {})
    return SparseCoupledTrainingConfig(
        coupling=str(config["pipeline"]["coupling"]),
        reconstructor_lr=float(training.get("reconstructor_lr", 2.0e-3)),
        rfno_lr=float(training.get("rfno_lr", 2.0e-4)),
        weight_decay=float(training.get("weight_decay", 1.0e-4)),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        input_steps=int(config["data"]["input_steps"]),
        one_shot_steps=int(config["data"]["output_steps"]),
        max_rollout_steps=int(training.get("max_rollout_steps", 300)),
        use_rollout_curriculum=bool(training.get("use_rollout_curriculum", True)),
        rollout_train_steps=tuple(
            int(value)
            for value in training.get(
                "rollout_train_steps", DEFAULT_FORECAST_HORIZONS
            )
        ),
        rollout_curriculum_boundaries=tuple(
            float(value)
            for value in training.get(
                "rollout_curriculum_boundaries", (0.0, 0.1, 0.2, 0.3, 0.45, 0.6)
            )
        ),
        rollout_detach_context=bool(training.get("rollout_detach_context", False)),
        forecast_supervision=bool(training.get("forecast_supervision", False)),
        hidden_weight=float(loss_weights.get("hidden_reconstruction", 1.0)),
        observation_weight=float(loss_weights.get("observation_consistency", 0.1)),
        history_gradient_weight=float(loss_weights.get("history_gradient", 0.05)),
        rollout_weight=float(loss_weights.get("rollout", 0.0)),
        forecast_gradient_weight=float(loss_weights.get("forecast_gradient", 0.0)),
        spectrum_weight=float(loss_weights.get("spectrum", 0.0)),
    )


def move_sparse_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def run_sparse_one_batch(
    experiment_id: str,
    pipeline: nn.Module,
    batch: Mapping[str, Any],
    *,
    rollout_steps: int = 30,
    training_config: Optional[SparseCoupledTrainingConfig] = None,
    backward: bool = False,
) -> Dict[str, Any]:
    """Run one shared sparse batch with explicitly filtered model inputs."""

    experiment_id = str(experiment_id).upper()
    if experiment_id not in SUPPORTED_EXPERIMENTS:
        raise ValueError(f"Unsupported sparse experiment: {experiment_id}")
    model_inputs = sparse_model_inputs(batch)
    if experiment_id in {"B1", "B3"}:
        pipeline.eval()
        with torch.no_grad():
            return pipeline.rollout(  # type: ignore[attr-defined]
                model_inputs["x_obs"],
                model_inputs["obs_mask"],
                rollout_steps=rollout_steps,
            )

    if not (
        callable(getattr(pipeline, "rollout", None))
        and callable(getattr(pipeline, "optimizer_param_groups", None))
    ):
        raise TypeError("Learned sparse experiments require the shared trainable pipeline")
    config = training_config or SparseCoupledTrainingConfig(
        coupling=str(pipeline.coupling)
    )
    trainer = SparseCoupledForecastTrainer(pipeline, config)
    if backward:
        optimizer = trainer.build_optimizer()
        result = trainer.train_batch(
            batch, optimizer, active_rollout_steps=rollout_steps
        )
        result["optimizer"] = optimizer
        return result
    return trainer.validate_batch(batch, active_rollout_steps=rollout_steps)


@torch.no_grad()
def evaluate_sparse_loader(
    pipeline: nn.Module,
    loader: DataLoader,
    *,
    normalization_mean: float,
    normalization_std: float,
    rollout_steps: int,
    forecast_horizons: Sequence[int],
    frame_interval: float,
    device: torch.device,
    max_batches: Optional[int] = None,
    source_entries: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Shared per-MAT evaluation path for every supported sparse experiment."""

    pipeline.eval()
    active_horizons = tuple(
        int(value) for value in forecast_horizons if int(value) <= rollout_steps
    )
    accumulator = SparseSurfaceMetricAccumulator(
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        forecast_horizons=active_horizons,
        frame_interval=frame_interval,
    )
    batches = 0
    samples = 0
    started = time.perf_counter()
    for raw_batch in loader:
        batch = move_sparse_batch_to_device(raw_batch, device)
        model_inputs = sparse_model_inputs(batch)
        outputs = pipeline.rollout(  # type: ignore[attr-defined]
            model_inputs["x_obs"],
            model_inputs["obs_mask"],
            rollout_steps=rollout_steps,
        )
        accumulator.update(
            outputs["history_reconstruction"],
            batch["x_full"],
            outputs["forecast"],
            batch["y"],
            batch["obs_mask"],
            source_ids=batch["source_id"],
        )
        batches += 1
        samples += int(batch["x_full"].shape[0])
        if max_batches is not None and batches >= max_batches:
            break
    result = accumulator.compute()
    result["evaluated_batches"] = batches
    result["evaluated_samples"] = samples
    result["rollout_steps"] = int(rollout_steps)
    result["elapsed_seconds"] = time.perf_counter() - started
    if samples:
        result["seconds_per_sample"] = result["elapsed_seconds"] / samples
    if source_entries is not None:
        source_lookup = {
            str(entry["source_id"]): {
                "source_id": int(entry["source_id"]),
                "name": str(entry["name"]),
                "relative_path": str(entry["relative_path"]),
            }
            for entry in source_entries
        }
        for source_id, source_metrics in result["per_source"].items():
            if source_id not in source_lookup:
                raise KeyError(f"Missing frozen MAT metadata for source_id={source_id}")
            source_metrics["source"] = source_lookup[source_id]
    return result


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(tuple(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _clean_model_state_dict(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Drop legacy FNO construction metadata stored as an unexpected state key."""

    if "_metadata" not in state or isinstance(state["_metadata"], Tensor):
        return state
    cleaned = state.copy()  # type: ignore[attr-defined]
    del cleaned["_metadata"]
    return cleaned


class SparseCheckpointManager:
    """Atomic best/last checkpoint management with scientific-config validation."""

    def __init__(self, run_dir: Union[str, Path], config_sha256: str) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config_sha256 = str(config_sha256)
        self.best_metadata_path = self.checkpoint_dir / "best.json"

    def _payload(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        *,
        epoch: int,
        global_step: int,
        metrics: Optional[Mapping[str, float]],
        scheduler: Optional[Any],
        scaler: Optional[Any],
        curriculum_state: Optional[Mapping[str, Any]],
        data_loader_generator: Optional[torch.Generator],
    ) -> Dict[str, Any]:
        complete_training_state = all(
            value is not None
            for value in (
                optimizer,
                scheduler,
                scaler,
                curriculum_state,
                data_loader_generator,
            )
        )
        return {
            "schema_version": 2,
            "scientific_config_sha256": self.config_sha256,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict": _clean_model_state_dict(model.state_dict()),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "scheduler_state_dict": (
                None if scheduler is None else scheduler.state_dict()
            ),
            "scaler_state_dict": None if scaler is None else scaler.state_dict(),
            "curriculum_state": dict(curriculum_state or {}),
            "data_loader_generator_state": (
                None
                if data_loader_generator is None
                else data_loader_generator.get_state()
            ),
            "rng_state": _capture_rng_state(),
            "complete_training_state": complete_training_state,
            "metrics": dict(metrics or {}),
        }

    @staticmethod
    def _require_complete_training_arguments(
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        scaler: Optional[Any],
        curriculum_state: Optional[Mapping[str, Any]],
        data_loader_generator: Optional[torch.Generator],
    ) -> None:
        values = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "curriculum_state": curriculum_state,
            "data_loader_generator": data_loader_generator,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(f"Complete training checkpoint requires: {missing}")

    @staticmethod
    def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        if temporary.exists():
            raise FileExistsError(f"Temporary checkpoint already exists: {temporary}")
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)

    def save_last(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        *,
        epoch: int,
        global_step: int,
        metrics: Optional[Mapping[str, float]] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        curriculum_state: Optional[Mapping[str, Any]] = None,
        data_loader_generator: Optional[torch.Generator] = None,
        require_complete_training_state: bool = False,
    ) -> Path:
        if require_complete_training_state:
            self._require_complete_training_arguments(
                optimizer,
                scheduler,
                scaler,
                curriculum_state,
                data_loader_generator,
            )
        path = self.checkpoint_dir / "last.pt"
        self._atomic_torch_save(
            self._payload(
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                scheduler=scheduler,
                scaler=scaler,
                curriculum_state=curriculum_state,
                data_loader_generator=data_loader_generator,
            ),
            path,
        )
        return path

    def save_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        *,
        metric_name: str,
        metric_value: float,
        epoch: int,
        global_step: int,
        mode: str = "min",
        metrics: Optional[Mapping[str, float]] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        curriculum_state: Optional[Mapping[str, Any]] = None,
        data_loader_generator: Optional[torch.Generator] = None,
        require_complete_training_state: bool = False,
    ) -> Tuple[Path, bool]:
        if mode not in {"min", "max"}:
            raise ValueError("Best-checkpoint mode must be 'min' or 'max'")
        metric_value = float(metric_value)
        if not math.isfinite(metric_value):
            raise ValueError("Best-checkpoint metric must be finite")
        previous = None
        if self.best_metadata_path.is_file():
            previous = json.loads(self.best_metadata_path.read_text(encoding="utf-8"))
            if previous.get("metric_name") != metric_name or previous.get("mode") != mode:
                raise ValueError("Best-checkpoint metric contract changed within one run")
        improved = previous is None or (
            metric_value < float(previous["metric_value"])
            if mode == "min"
            else metric_value > float(previous["metric_value"])
        )
        path = self.checkpoint_dir / "best.pt"
        if improved:
            if require_complete_training_state:
                self._require_complete_training_arguments(
                    optimizer,
                    scheduler,
                    scaler,
                    curriculum_state,
                    data_loader_generator,
                )
            all_metrics = dict(metrics or {})
            all_metrics[metric_name] = metric_value
            self._atomic_torch_save(
                self._payload(
                    model,
                    optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=all_metrics,
                    scheduler=scheduler,
                    scaler=scaler,
                    curriculum_state=curriculum_state,
                    data_loader_generator=data_loader_generator,
                ),
                path,
            )
            _write_json(
                self.best_metadata_path,
                {
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "mode": mode,
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "checkpoint_sha256": sha256_file(path),
                },
                overwrite=True,
            )
        return path, improved

    def restore(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        data_loader_generator: Optional[torch.Generator] = None,
        restore_random_state: bool = True,
        require_training_state: bool = False,
        strict: bool = True,
        allow_config_mismatch: bool = False,
    ) -> Dict[str, Any]:
        checkpoint = _load_checkpoint_mapping(Path(checkpoint_path))
        source_config_sha256 = checkpoint.get("scientific_config_sha256")
        if source_config_sha256 != self.config_sha256 and not allow_config_mismatch:
            raise ValueError("Checkpoint scientific config hash does not match this run")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise KeyError("Checkpoint lacks model_state_dict")
        state = _clean_model_state_dict(state)
        model.load_state_dict(state, strict=strict)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        generator_state = checkpoint.get("data_loader_generator_state")
        if data_loader_generator is not None and generator_state is not None:
            data_loader_generator.set_state(generator_state)
        rng_state = checkpoint.get("rng_state")
        if restore_random_state and isinstance(rng_state, Mapping):
            _restore_rng_state(rng_state)

        restored_components = {
            "model": True,
            "optimizer": optimizer is not None and optimizer_state is not None,
            "scheduler": scheduler is not None and scheduler_state is not None,
            "scaler": scaler is not None and scaler_state is not None,
            "data_loader_generator": (
                data_loader_generator is not None and generator_state is not None
            ),
            "rng": restore_random_state and isinstance(rng_state, Mapping),
            "curriculum": "curriculum_state" in checkpoint,
        }
        if require_training_state:
            required = (
                "optimizer",
                "scheduler",
                "scaler",
                "data_loader_generator",
                "rng",
                "curriculum",
            )
            missing = [name for name in required if not restored_components[name]]
            if not bool(checkpoint.get("complete_training_state", False)):
                missing.append("complete_training_state_marker")
            if missing:
                raise RuntimeError(
                    f"Checkpoint did not restore required training state: {missing}"
                )
        return {
            "epoch": int(checkpoint.get("epoch", 0)),
            "global_step": int(checkpoint.get("global_step", 0)),
            "metrics": dict(checkpoint.get("metrics", {})),
            "curriculum_state": dict(checkpoint.get("curriculum_state", {})),
            "restored_components": restored_components,
            "source_scientific_config_sha256": source_config_sha256,
        }


def summarize_dry_run(
    context: SparseExperimentContext,
    pipeline: nn.Module,
    batch_result: Mapping[str, Any],
) -> Dict[str, Any]:
    output = {
        "experiment_id": context.config["experiment_id"],
        "experiment_name": context.config["experiment_name"],
        "run_dir": str(context.run_dir),
        "scientific_config_sha256": context.config_sha256,
        "split_counts": context.data.split_manifest["counts"],
        "dataset_samples": {
            split: len(dataset)
            for split, dataset in context.data.sparse_datasets.items()
        },
        "normalization": {
            "mean": context.data.normalization_mean,
            "std": context.data.normalization_std,
            "source": "b0_checkpoint_train_only",
            "all_splits_equal": all(
                dense.mean == context.data.normalization_mean
                and dense.std == context.data.normalization_std
                for dense in context.data.dense_datasets.values()
            ),
        },
        "fixed_mask_id": context.config["observation"]["mask_id"],
        "history_shape": list(batch_result["history_reconstruction"].shape),
        "forecast_shape": list(batch_result["forecast"].shape),
        "history_finite": bool(
            torch.isfinite(batch_result["history_reconstruction"]).all()
        ),
        "forecast_finite": bool(torch.isfinite(batch_result["forecast"]).all()),
        "full_training_started": False,
    }
    losses = batch_result.get("losses")
    if isinstance(losses, Mapping):
        output["losses"] = {
            key: float(value.detach().cpu()) if isinstance(value, Tensor) else float(value)
            for key, value in losses.items()
        }
    if hasattr(pipeline, "rfno_gradients_are_none"):
        output["rfno_gradients_none"] = bool(pipeline.rfno_gradients_are_none())
    return output


__all__ = [
    "SparseCheckpointManager",
    "SparseDataBundle",
    "SparseExperimentContext",
    "build_sparse_dataloaders",
    "build_sparse_pipeline",
    "collect_provenance",
    "coupled_training_config",
    "evaluate_sparse_loader",
    "freeze_data_split",
    "load_resolved_sparse_config",
    "move_sparse_batch_to_device",
    "prepare_sparse_experiment",
    "read_b0_normalization",
    "run_sparse_one_batch",
    "scientific_config_sha256",
    "sha256_file",
    "summarize_dry_run",
    "validate_continuation_checkpoint",
]
