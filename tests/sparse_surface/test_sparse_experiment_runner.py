import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch
from torch import nn

import neuralop.training.sparse_experiment_runner as sparse_runner

from neuralop.models.reconstructors import (
    PartialConvMaskedAutoencoder,
    PeriodicMaskUNet,
    PODSpatialReconstructor,
)
from neuralop.models.sparse_forecast_pipeline import (
    BilinearFrozenRFNOPipeline,
    MaskUNetFrozenRFNOPipeline,
    PartialConvMAERFNOPipeline,
)
from neuralop.training.sparse_coupled_forecast import SparseCoupledTrainingConfig
from neuralop.training.sparse_multiepoch import SparseMultiEpochTrainer
from neuralop.training.sparse_experiment_runner import (
    SparseCheckpointManager,
    evaluate_sparse_loader,
    freeze_data_split,
    load_resolved_sparse_config,
    move_sparse_batch_to_device,
    prepare_sparse_experiment,
    run_sparse_one_batch,
    scientific_config_sha256,
    validate_continuation_checkpoint,
)
from scripts.sparse_surface.run_sparse_experiment import (
    _evaluation_output_name,
    main as sparse_experiment_main,
)


class TinyRFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(60, 30, kernel_size=1)

    def forward(self, history):
        return self.projection(history)


class DummyScaler:
    def __init__(self, scale=1.0):
        self.scale = float(scale)

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state):
        self.scale = float(state["scale"])


class MetadataLinear(nn.Linear):
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_metadata"] = {"legacy": True}
        return state


def _write_data_root(root: Path, *, duplicate_name: bool = False) -> None:
    data_root = root / "data" / "bimodal"
    for index, split in enumerate(("train", "val", "test")):
        directory = data_root / split
        directory.mkdir(parents=True)
        name = "shared.mat" if duplicate_name and split != "test" else f"{split}.mat"
        field = np.arange(360 * 4 * 4, dtype=np.float32).reshape(360, 4, 4)
        field = field + index * 10000.0
        sio.savemat(directory / name, {"height": field})


def _write_mask(root: Path) -> Path:
    directory = root / "data" / "masks"
    directory.mkdir(parents=True)
    mask = np.zeros((60, 4, 4), dtype=np.uint8)
    mask[:, ::2, ::2] = 1
    np.savez_compressed(directory / "point_masks.npz", mask_00000=mask)
    document = {
        "version": 1,
        "npz_file": "point_masks.npz",
        "height": 4,
        "width": 4,
        "time_steps": 60,
        "entries": [
            {
                "mask_id": "mask_00000",
                "mask_type": "fixed_points",
                "requested_observation_rate": 0.25,
                "effective_observation_rate": 0.25,
                "replicate": 0,
                "seed": 7,
                "temporal_dropout": 0.0,
                "shape": [60, 4, 4],
                "sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            }
        ],
    }
    path = directory / "point_masks.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _write_checkpoint(root: Path) -> Path:
    path = root / "checkpoints" / "b0.pt"
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "config": {"input_steps": 60, "output_steps": 30},
            "model_state_dict": {},
            "train_dataset_mean": 5.0,
            "train_dataset_std": 2.0,
        },
        path,
    )
    return path


def _write_config(root: Path, experiment_id: str = "B1") -> Path:
    expected = {
        "B1": ("bilinear", "frozen"),
        "B3": ("pod", "frozen"),
        "B4": ("mask_unet", "frozen"),
        "P1": ("partialconv_mae", "frozen"),
        "P2": ("partialconv_mae", "joint"),
    }[experiment_id]
    config = {
        "experiment_id": experiment_id,
        "experiment_name": f"{experiment_id.lower()}_test_fixed_r025_seed7",
        "data": {
            "data_root": "./data/bimodal",
            "input_steps": 60,
            "output_steps": 30,
            "rollout_steps": 300,
            "height": 4,
            "width": 4,
            "stride": 300,
            "batch_size": 1,
            "val_batch_size": 1,
            "test_batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
        },
        "observation": {
            "mask_type": "fixed_points",
            "observation_rate": 0.25,
            "manifest_path": "./data/masks/point_masks.npz",
            "mask_id": "mask_00000",
            "seed": 7,
        },
        "pipeline": {
            "reconstructor": expected[0],
            "coupling": expected[1],
            "forecaster": "rfno",
            "rfno_checkpoint": "./checkpoints/b0.pt",
        },
        "training": {
            "epochs": 2,
            "forecast_supervision": experiment_id == "P2",
            "use_rollout_curriculum": experiment_id == "P2",
            "max_rollout_steps": 300 if experiment_id == "P2" else 30,
            "reconstructor_lr": 1.0e-3,
            "rfno_lr": 1.0e-4,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "amp": False,
            "scheduler": {"name": "constant"},
            "early_stopping_patience": 0,
            "primary_loss": "total",
            "primary_mode": "min",
            "loss_weights": {
                "hidden_reconstruction": 1.0,
                "observation_consistency": 0.1,
                "history_gradient": 0.05,
                "rollout": 1.0 if experiment_id == "P2" else 0.0,
                "forecast_gradient": 0.05 if experiment_id == "P2" else 0.0,
                "spectrum": 0.01 if experiment_id == "P2" else 0.0,
            },
        },
        "runtime": {
            "run_root": "./runs/sparse",
            "seed": 7,
            "device": "cpu",
            "hash_data_files": True,
        },
    }
    path = root / f"{experiment_id.lower()}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _workspace(root: Path, experiment_id: str = "B1") -> Path:
    _write_data_root(root)
    _write_mask(root)
    _write_checkpoint(root)
    return _write_config(root, experiment_id)


def test_prepare_freezes_disjoint_splits_and_b0_normalization(tmp_path):
    config_path = _workspace(tmp_path)

    context = prepare_sparse_experiment(config_path, project_root=tmp_path)

    assert context.data.split_manifest["counts"] == {"train": 1, "val": 1, "test": 1}
    split_names = {
        split: {entry["name"] for entry in entries}
        for split, entries in context.data.split_manifest["splits"].items()
    }
    assert split_names["train"].isdisjoint(split_names["val"])
    assert split_names["train"].isdisjoint(split_names["test"])
    assert split_names["val"].isdisjoint(split_names["test"])
    assert context.data.normalization_mean == 5.0
    assert context.data.normalization_std == 2.0
    assert set(context.data.loaders) == {"train", "val"}
    assert "test" not in context.data.dense_datasets
    assert all(dataset.mean == 5.0 for dataset in context.data.dense_datasets.values())
    assert all(dataset.std == 2.0 for dataset in context.data.dense_datasets.values())

    batch = next(iter(context.data.loaders["val"]))
    assert set(batch) == {
        "x_full",
        "x_obs",
        "obs_mask",
        "y",
        "mask_id",
        "source_id",
    }
    assert batch["mask_id"] == ["mask_00000"]
    assert batch["source_id"].tolist() == [0]
    assert (context.run_dir / "config.resolved.json").is_file()
    assert (context.run_dir / "data_split.json").is_file()
    assert (context.run_dir / "provenance.json").is_file()
    provenance = json.loads(
        (context.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["files"]["rfno_checkpoint"]["sha256"]
    assert provenance["files"]["mask_manifest"]["sha256"]
    assert provenance["files"]["mask_archive"]["sha256"]


def test_reconstruction_training_mode_omits_future_target(tmp_path):
    config_path = _workspace(tmp_path)
    context = prepare_sparse_experiment(
        config_path,
        project_root=tmp_path,
        training_mode=True,
    )

    batch = next(iter(context.data.loaders["train"]))
    assert "y" not in batch
    assert set(batch) == {
        "x_full",
        "x_obs",
        "obs_mask",
        "mask_id",
        "source_id",
    }


def test_forecast_supervised_training_mode_keeps_future_target(tmp_path):
    config_path = _workspace(tmp_path, "P2")
    context = prepare_sparse_experiment(
        config_path,
        project_root=tmp_path,
        training_mode=True,
    )

    batch = next(iter(context.data.loaders["train"]))
    assert "y" in batch


def test_protocol_mismatch_override_is_validation_only(
    tmp_path, monkeypatch
):
    config_path = _workspace(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sparse_experiment.py",
            str(config_path),
            "--project-root",
            str(tmp_path),
            "--dry-run",
            "--allow-validation-protocol-mismatch",
        ],
    )
    with pytest.raises(PermissionError, match="validation-only"):
        sparse_experiment_main()


def test_existing_run_directory_is_never_overwritten(tmp_path):
    config_path = _workspace(tmp_path)
    prepare_sparse_experiment(config_path, project_root=tmp_path)

    with pytest.raises(FileExistsError):
        prepare_sparse_experiment(config_path, project_root=tmp_path)


def test_duplicate_mat_names_across_splits_are_rejected(tmp_path):
    _write_data_root(tmp_path, duplicate_name=True)
    _write_mask(tmp_path)
    _write_checkpoint(tmp_path)
    config_path = _write_config(tmp_path)
    # Resolve through a non-creating helper path by using a temporary run preparation.
    from neuralop.training.sparse_experiment_runner import load_resolved_sparse_config

    resolved = load_resolved_sparse_config(config_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="filename leakage"):
        freeze_data_split(resolved)


def test_duplicate_mat_content_with_different_names_is_rejected(tmp_path):
    config_path = _workspace(tmp_path)
    train_path = next((tmp_path / "data" / "bimodal" / "train").glob("*.mat"))
    val_dir = tmp_path / "data" / "bimodal" / "val"
    next(val_dir.glob("*.mat")).unlink()
    (val_dir / "renamed_duplicate.mat").write_bytes(train_path.read_bytes())

    resolved = load_resolved_sparse_config(config_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="content leakage"):
        freeze_data_split(resolved)


def test_checkpoint_manager_saves_best_last_and_restores(tmp_path):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = DummyScaler(8.0)
    generator = torch.Generator().manual_seed(456)
    manager = SparseCheckpointManager(tmp_path / "run", "config-hash")
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}

    last = manager.save_last(
        model,
        optimizer,
        epoch=2,
        global_step=17,
        metrics={"loss": 0.4},
        scheduler=scheduler,
        scaler=scaler,
        curriculum_state={"active_rollout_steps": 120},
        data_loader_generator=generator,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(3)
    expected_generator = torch.rand(3, generator=generator)
    best, improved = manager.save_best(
        model,
        optimizer,
        metric_name="forecast_300.nrmse",
        metric_value=0.3,
        epoch=2,
        global_step=17,
        mode="min",
    )
    _, improved_again = manager.save_best(
        model,
        optimizer,
        metric_name="forecast_300.nrmse",
        metric_value=0.5,
        epoch=3,
        global_step=20,
        mode="min",
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
    optimizer.step()
    scheduler.step()
    scaler.scale = 1.0
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    generator.manual_seed(999)
    state = manager.restore(
        last,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        data_loader_generator=generator,
        require_training_state=True,
    )

    assert last.is_file() and best.is_file()
    assert improved is True and improved_again is False
    assert state["epoch"] == 2
    assert state["global_step"] == 17
    assert state["metrics"] == {"loss": 0.4}
    assert state["curriculum_state"] == {"active_rollout_steps": 120}
    assert all(state["restored_components"].values())
    assert scaler.scale == 8.0
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)
    torch.testing.assert_close(torch.rand(3, generator=generator), expected_generator)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    with pytest.raises(ValueError, match="config hash"):
        SparseCheckpointManager(tmp_path / "run", "different").restore(last, model)


def test_checkpoint_strips_legacy_metadata_key_and_requires_complete_state(tmp_path):
    manager = SparseCheckpointManager(tmp_path / "run", "config-hash")
    model = MetadataLinear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError, match="Complete training checkpoint requires"):
        manager.save_last(
            model,
            optimizer,
            epoch=1,
            global_step=1,
            require_complete_training_state=True,
        )

    checkpoint = manager.save_last(model, optimizer, epoch=1, global_step=1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "_metadata" not in payload["model_state_dict"]
    manager.restore(checkpoint, model)


def test_scientific_hash_includes_runtime_seed(tmp_path):
    config_path = _workspace(tmp_path)
    resolved = load_resolved_sparse_config(config_path, project_root=tmp_path)
    changed = json.loads(json.dumps(resolved))
    changed["runtime"]["seed"] += 1
    assert scientific_config_sha256(resolved) != scientific_config_sha256(changed)


@pytest.mark.parametrize(
    ("experiment_id", "legacy_unused_defaults", "used_field"),
    [
        (
            "B4",
            {"pod_rank": 64, "pod_ranks": [8, 16, 32, 64], "pod_ridge": 1.0e-6},
            "mask_unet_channels",
        ),
        (
            "P2",
            {
                "mask_unet_channels": [12, 24, 40],
                "pod_rank": 64,
                "pod_ranks": [8, 16, 32, 64],
                "pod_ridge": 1.0e-6,
            },
            "partialconv_encoder_channels",
        ),
    ],
)
def test_scientific_hash_ignores_only_known_unused_model_defaults(
    tmp_path, experiment_id, legacy_unused_defaults, used_field
):
    config_path = _workspace(tmp_path, experiment_id)
    resolved = load_resolved_sparse_config(config_path, project_root=tmp_path)
    historical = json.loads(json.dumps(resolved))
    for key in legacy_unused_defaults:
        historical["model"].pop(key, None)
    assert scientific_config_sha256(resolved) == scientific_config_sha256(historical)

    changed_used = json.loads(json.dumps(resolved))
    changed_used["model"][used_field][0] += 1
    assert scientific_config_sha256(resolved) != scientific_config_sha256(changed_used)

    changed_unused_nondefault = json.loads(json.dumps(resolved))
    first_key = next(iter(legacy_unused_defaults))
    value = changed_unused_nondefault["model"][first_key]
    changed_unused_nondefault["model"][first_key] = (
        value + 1 if isinstance(value, int) else [value[0] + 1, *value[1:]]
    )
    assert scientific_config_sha256(resolved) != scientific_config_sha256(
        changed_unused_nondefault
    )


def test_formal_config_requires_content_hashing(tmp_path):
    config_path = _workspace(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["runtime"]["hash_data_files"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash_data_files=true"):
        load_resolved_sparse_config(config_path, project_root=tmp_path)


def test_nonzero_observation_noise_cannot_be_silently_ignored(tmp_path):
    config_path = _workspace(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["observation"]["noise_std_fraction"] = 0.01
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="noise is not yet implemented"):
        load_resolved_sparse_config(config_path, project_root=tmp_path)


def test_prepare_resume_reuses_frozen_run_without_overwrite(tmp_path):
    config_path = _workspace(tmp_path)
    context = prepare_sparse_experiment(config_path, project_root=tmp_path)
    model = nn.Linear(2, 2)
    checkpoint = SparseCheckpointManager(
        context.run_dir, context.config_sha256
    ).save_last(model, None, epoch=4, global_step=23)

    resumed = prepare_sparse_experiment(
        config_path,
        project_root=tmp_path,
        resume_checkpoint=checkpoint,
    )

    assert resumed.resumed is True
    assert resumed.run_dir == context.run_dir
    assert resumed.config_sha256 == context.config_sha256
    assert resumed.data.split_manifest == context.data.split_manifest

    existing = prepare_sparse_experiment(
        config_path,
        project_root=tmp_path,
        existing_run_dir=context.run_dir,
    )
    assert existing.resumed is True
    assert existing.run_dir == context.run_dir
    assert set(existing.data.loaders) == {"train", "val"}

    test_only = prepare_sparse_experiment(
        config_path,
        project_root=tmp_path,
        existing_run_dir=context.run_dir,
        loader_splits=("test",),
    )
    assert set(test_only.data.loaders) == {"test"}


def test_external_frozen_split_train_val_never_hashes_test_mat(tmp_path, monkeypatch):
    config_path = _workspace(tmp_path)
    frozen = prepare_sparse_experiment(config_path, project_root=tmp_path)
    split_manifest = frozen.run_dir / "data_split.json"
    original_sha256 = sparse_runner.sha256_file
    hashed_mat_splits = []

    def guarded_sha256(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.suffix.lower() == ".mat":
            hashed_mat_splits.append(candidate.parent.name)
            if candidate.parent.name == "test":
                raise AssertionError("train/val preparation opened a test MAT file")
        return original_sha256(path, *args, **kwargs)

    monkeypatch.setattr(sparse_runner, "sha256_file", guarded_sha256)
    context = prepare_sparse_experiment(
        config_path,
        project_root=tmp_path,
        run_dir=tmp_path / "runs" / "sparse" / "train_val_only",
        loader_splits=("train", "val"),
        frozen_split_manifest=split_manifest,
    )

    assert set(context.data.loaders) == {"train", "val"}
    assert set(hashed_mat_splits) == {"train", "val"}


def _tiny_partial_pipeline(coupling: str) -> PartialConvMAERFNOPipeline:
    reconstructor = PartialConvMaskedAutoencoder(
        input_steps=60, encoder_channels=(4,), decoder_channels=(4,)
    )
    return PartialConvMAERFNOPipeline(TinyRFNO(), reconstructor, coupling=coupling)


@pytest.mark.parametrize("experiment_id", ["B1", "B3", "B4", "P1", "P2"])
def test_shared_one_batch_and_evaluation_path(experiment_id, tmp_path):
    config_path = _workspace(tmp_path, experiment_id)
    context = prepare_sparse_experiment(config_path, project_root=tmp_path)
    raw_batch = next(iter(context.data.loaders["val"]))
    batch = move_sparse_batch_to_device(raw_batch, torch.device("cpu"))
    if experiment_id == "B1":
        pipeline = BilinearFrozenRFNOPipeline(TinyRFNO())
        training_config = None
    elif experiment_id == "B3":
        modes = torch.linalg.qr(torch.randn(16, 4)).Q.transpose(0, 1)
        pipeline = BilinearFrozenRFNOPipeline(
            TinyRFNO(),
            reconstructor=PODSpatialReconstructor(
                torch.zeros(16), modes, rank=4, height=4, width=4
            ),
        )
        training_config = None
    else:
        coupling = "frozen" if experiment_id in {"B4", "P1"} else "joint"
        pipeline = (
            MaskUNetFrozenRFNOPipeline(
                TinyRFNO(),
                PeriodicMaskUNet(input_steps=60, channels=(4, 8)),
                coupling="frozen",
            )
            if experiment_id == "B4"
            else _tiny_partial_pipeline(coupling)
        )
        training_config = SparseCoupledTrainingConfig(
            coupling=coupling,
            forecast_supervision=experiment_id == "P2",
            max_rollout_steps=30,
            use_rollout_curriculum=False,
        )

    result = run_sparse_one_batch(
        experiment_id,
        pipeline,
        batch,
        rollout_steps=30,
        training_config=training_config,
        backward=experiment_id == "P2",
    )
    metrics = evaluate_sparse_loader(
        pipeline,
        context.data.loaders["val"],
        normalization_mean=5.0,
        normalization_std=2.0,
        rollout_steps=30,
        forecast_horizons=(30,),
        frame_interval=0.25,
        device=torch.device("cpu"),
        max_batches=1,
        source_entries=context.data.split_manifest["splits"]["val"],
    )

    assert result["history_reconstruction"].shape == (1, 60, 4, 4)
    assert result["forecast"].shape == (1, 30, 4, 4)
    assert torch.isfinite(result["history_reconstruction"]).all()
    assert torch.isfinite(result["forecast"]).all()
    assert metrics["source_count"] == 1
    assert "history_reconstruction" in metrics
    assert "forecast_30" in metrics
    assert metrics["per_source"]["0"]["source"]["name"] == "val.mat"
    if experiment_id == "P2":
        rfno_gradients = [parameter.grad for parameter in pipeline.rfno.parameters()]
        assert any(gradient is not None for gradient in rfno_gradients)
        assert all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in rfno_gradients
        )


def test_partial_evaluation_uses_noncanonical_artifact_name():
    assert (
        _evaluation_output_name(
            "val", rollout_steps=300, formal_rollout_steps=300, max_batches=None
        )
        == "val_metrics.json"
    )
    assert "partial" in _evaluation_output_name(
        "val", rollout_steps=30, formal_rollout_steps=300, max_batches=1
    )


@pytest.mark.parametrize("experiment_id", ["B4", "P1", "P2"])
def test_learned_formal_evaluation_requires_resume(
    experiment_id, tmp_path, monkeypatch
):
    config_path = _workspace(tmp_path, experiment_id)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sparse_experiment.py",
            str(config_path),
            "--project-root",
            str(tmp_path),
            "--evaluate-split",
            "val",
        ],
    )
    with pytest.raises(PermissionError, match="requires an explicit trained checkpoint"):
        sparse_experiment_main()


@pytest.mark.parametrize("experiment_id", ["B4", "P1", "P2"])
def test_learned_test_rejects_training_directory_resume(
    experiment_id, tmp_path, monkeypatch
):
    config_path = _workspace(tmp_path, experiment_id)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sparse_experiment.py",
            str(config_path),
            "--project-root",
            str(tmp_path),
            "--resume",
            str(tmp_path / "checkpoints" / "last.pt"),
            "--evaluate-split",
            "test",
            "--allow-test",
        ],
    )
    with pytest.raises(PermissionError, match="new unique run"):
        sparse_experiment_main()


@pytest.mark.parametrize("experiment_id", ["B4", "P1", "P2"])
def test_learned_test_requires_nonexistent_unique_directory(
    experiment_id, tmp_path, monkeypatch
):
    config_path = _workspace(tmp_path, experiment_id)
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.touch()
    split_manifest = tmp_path / "frozen_split.json"
    split_manifest.touch()
    occupied = tmp_path / "runs" / "final_test" / experiment_id.lower()
    occupied.mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sparse_experiment.py",
            str(config_path),
            "--project-root",
            str(tmp_path),
            "--evaluation-checkpoint",
            str(checkpoint),
            "--run-dir",
            str(occupied),
            "--split-manifest",
            str(split_manifest),
            "--evaluate-split",
            "test",
            "--allow-test",
        ],
    )
    with pytest.raises(FileExistsError, match="already exists"):
        sparse_experiment_main()


def test_p1_multiepoch_saves_complete_best_last_and_resumes(tmp_path):
    config_path = _workspace(tmp_path, "P1")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["primary_loss"] = "reconstruction_missing_nrmse"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    context = prepare_sparse_experiment(config_path, project_root=tmp_path)
    pipeline = _tiny_partial_pipeline("frozen")
    trainer = SparseMultiEpochTrainer(context, pipeline, torch.device("cpu"))

    first = trainer.fit(
        max_epochs_this_run=1,
        max_train_batches=1,
        max_val_batches=1,
        run_kind="smoke",
    )
    last = context.run_dir / "checkpoints" / "last.pt"
    best = context.run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)

    assert first["last_epoch"] == 1
    assert last.is_file() and best.is_file()
    assert checkpoint["complete_training_state"] is True
    assert checkpoint["curriculum_state"]["run_kind"] == "smoke"
    assert checkpoint["curriculum_state"]["best_epoch"] == 1
    assert first["best_epoch"] == 1
    assert first["peak_memory_allocated_bytes"] == 0
    assert "val.reconstruction_missing_nrmse" in checkpoint["metrics"]
    assert all(parameter.grad is None for parameter in pipeline.rfno.parameters())

    resumed_pipeline = _tiny_partial_pipeline("frozen")
    resumed_trainer = SparseMultiEpochTrainer(
        context, resumed_pipeline, torch.device("cpu")
    )
    resumed = resumed_trainer.fit(
        resume_checkpoint=last,
        max_epochs_this_run=1,
        max_train_batches=1,
        max_val_batches=1,
        run_kind="smoke",
    )
    assert resumed["start_epoch"] == 2
    assert resumed["last_epoch"] == 2
    assert len((context.run_dir / "logs" / "epochs.jsonl").read_text().splitlines()) == 2


def test_p1_forecast_supervised_downstream_freezes_rfno_and_resumes(tmp_path):
    config_path = _workspace(tmp_path, "P1")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"].update(
        {
            "forecast_supervision": True,
            "use_rollout_curriculum": False,
            "max_rollout_steps": 30,
            "primary_loss": "forecast_30_nrmse",
        }
    )
    config["training"]["loss_weights"].update(
        {
            "hidden_reconstruction": 0.0,
            "observation_consistency": 0.0,
            "history_gradient": 0.0,
            "rollout": 1.0,
            "forecast_gradient": 0.0,
            "spectrum": 0.0,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    context = prepare_sparse_experiment(config_path, project_root=tmp_path)
    assert set(context.data.loaders) == {"train", "val"}

    pipeline = _tiny_partial_pipeline("frozen")
    rfno_before = {
        name: value.detach().clone() for name, value in pipeline.rfno.state_dict().items()
    }
    reconstructor_before = {
        name: value.detach().clone()
        for name, value in pipeline.reconstructor.state_dict().items()
    }
    trainer = SparseMultiEpochTrainer(context, pipeline, torch.device("cpu"))
    initialized_pipeline = _tiny_partial_pipeline("frozen")
    initialized_pipeline.reconstructor.load_state_dict(
        pipeline.reconstructor.state_dict()
    )
    initialized_trainer = SparseMultiEpochTrainer(
        context, initialized_pipeline, torch.device("cpu")
    )
    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }

    assert trainer.reconstruction_only is False
    assert type(trainer.batch_trainer) is type(initialized_trainer.batch_trainer)
    assert trainer._active_rollout_steps(1) == 30
    assert all(not parameter.requires_grad for parameter in pipeline.rfno.parameters())
    assert not optimizer_ids.intersection(
        id(parameter) for parameter in pipeline.rfno.parameters()
    )

    first = trainer.fit(
        max_epochs_this_run=1,
        max_train_batches=1,
        max_val_batches=1,
        run_kind="smoke",
    )
    last = context.run_dir / "checkpoints" / "last.pt"
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)

    assert first["last_epoch"] == 1
    assert checkpoint["curriculum_state"]["active_rollout_steps"] == 30
    assert "val.forecast_30_nrmse" in checkpoint["metrics"]
    assert all(parameter.grad is None for parameter in pipeline.rfno.parameters())
    assert all(
        torch.equal(rfno_before[name], value)
        for name, value in pipeline.rfno.state_dict().items()
    )
    assert any(
        not torch.equal(reconstructor_before[name], value)
        for name, value in pipeline.reconstructor.state_dict().items()
    )

    resumed_pipeline = _tiny_partial_pipeline("frozen")
    resumed_trainer = SparseMultiEpochTrainer(
        context, resumed_pipeline, torch.device("cpu")
    )
    resumed = resumed_trainer.fit(
        resume_checkpoint=last,
        max_epochs_this_run=1,
        max_train_batches=1,
        max_val_batches=1,
        run_kind="smoke",
    )

    assert resumed["start_epoch"] == 2
    assert resumed["last_epoch"] == 2
    assert all(
        parameter.grad is None for parameter in resumed_pipeline.rfno.parameters()
    )


@pytest.mark.parametrize(
    ("use_curriculum", "max_rollout_steps", "message"),
    [
        (True, 30, "fixed 30-frame rollout"),
        (False, 60, "max_rollout_steps=30"),
    ],
)
def test_p1_forecast_supervised_rejects_nonfixed_rollout(
    tmp_path, use_curriculum, max_rollout_steps, message
):
    config_path = _workspace(tmp_path, "P1")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["forecast_supervision"] = True
    config["training"]["use_rollout_curriculum"] = use_curriculum
    config["training"]["max_rollout_steps"] = max_rollout_steps
    config_path.write_text(json.dumps(config), encoding="utf-8")
    context = prepare_sparse_experiment(config_path, project_root=tmp_path)

    with pytest.raises(ValueError, match=message):
        SparseMultiEpochTrainer(
            context, _tiny_partial_pipeline("frozen"), torch.device("cpu")
        )


def test_b4_multiepoch_uses_reconstruction_only_and_keeps_rfno_frozen(tmp_path):
    config_path = _workspace(tmp_path, "B4")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["primary_loss"] = "reconstruction_missing_nrmse"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    context = prepare_sparse_experiment(config_path, project_root=tmp_path)
    pipeline = MaskUNetFrozenRFNOPipeline(
        TinyRFNO(),
        PeriodicMaskUNet(input_steps=60, channels=(4, 8)),
        coupling="frozen",
    )
    trainer = SparseMultiEpochTrainer(context, pipeline, torch.device("cpu"))

    result = trainer.fit(
        max_epochs_this_run=1,
        max_train_batches=1,
        max_val_batches=1,
        run_kind="smoke",
    )
    checkpoint = torch.load(
        context.run_dir / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert result["last_epoch"] == 1
    assert "val.reconstruction_missing_nrmse" in checkpoint["metrics"]
    assert all(parameter.grad is None for parameter in pipeline.rfno.parameters())


def test_p2_multiepoch_joint_updates_rfno_and_writes_formal_checkpoint(tmp_path):
    config_path = _workspace(tmp_path, "P2")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["epochs"] = 1
    config["training"]["use_rollout_curriculum"] = False
    config["training"]["max_rollout_steps"] = 30
    config_path.write_text(json.dumps(config), encoding="utf-8")
    context = prepare_sparse_experiment(config_path, project_root=tmp_path)
    pipeline = _tiny_partial_pipeline("joint")
    before = {
        name: value.detach().clone() for name, value in pipeline.rfno.state_dict().items()
    }
    trainer = SparseMultiEpochTrainer(context, pipeline, torch.device("cpu"))

    result = trainer.fit(run_kind="formal")
    changed = any(
        not torch.equal(before[name], value)
        for name, value in pipeline.rfno.state_dict().items()
    )
    best = torch.load(
        context.run_dir / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert result["last_epoch"] == 1
    assert changed is True
    assert best["complete_training_state"] is True
    assert best["curriculum_state"]["run_kind"] == "formal"


def test_p2_multiepoch_rejects_amp_before_training(tmp_path):
    config_path = _workspace(tmp_path, "P2")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["amp"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    context = prepare_sparse_experiment(config_path, project_root=tmp_path)

    with pytest.raises(ValueError, match="P2 RFNO training requires"):
        SparseMultiEpochTrainer(
            context, _tiny_partial_pipeline("joint"), torch.device("cpu")
        )


def test_p2_controlled_continuation_restores_state_and_selects_300_metric(tmp_path):
    parent_config_path = _workspace(tmp_path, "P2")
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    parent_config["training"]["use_rollout_curriculum"] = False
    parent_config["training"]["max_rollout_steps"] = 30
    parent_config_path.write_text(json.dumps(parent_config), encoding="utf-8")
    parent_context = prepare_sparse_experiment(
        parent_config_path, project_root=tmp_path
    )
    parent_trainer = SparseMultiEpochTrainer(
        parent_context, _tiny_partial_pipeline("joint"), torch.device("cpu")
    )
    parent_trainer.fit(
        max_epochs_this_run=1,
        max_train_batches=1,
        max_val_batches=1,
        run_kind="smoke",
    )
    parent_best = parent_context.run_dir / "checkpoints" / "best.pt"

    child_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    child_config["experiment_name"] = "p2_child_curriculum_fixed_r025_seed7"
    child_config["pipeline"]["continuation_checkpoint"] = str(parent_best)
    child_config["training"].update(
        {
            "use_rollout_curriculum": True,
            "rollout_train_steps": [30, 300],
            "rollout_curriculum_boundaries": [0.0, 0.5],
            "max_rollout_steps": 300,
            "primary_loss": "forecast_300_nrmse",
        }
    )
    child_path = tmp_path / "p2_child.json"
    child_path.write_text(json.dumps(child_config), encoding="utf-8")
    resolved = load_resolved_sparse_config(child_path, project_root=tmp_path)
    metadata = validate_continuation_checkpoint(resolved, parent_best)
    child_context = prepare_sparse_experiment(
        child_path,
        project_root=tmp_path,
        frozen_split_manifest=Path(metadata["parent_split_path"]),
    )
    child_trainer = SparseMultiEpochTrainer(
        child_context, _tiny_partial_pipeline("joint"), torch.device("cpu")
    )

    result = child_trainer.fit(
        continuation_checkpoint=parent_best,
        max_epochs_this_run=1,
        max_train_batches=1,
        max_val_batches=1,
        run_kind="smoke",
    )
    last = torch.load(
        child_context.run_dir / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    best = torch.load(
        child_context.run_dir / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert result["start_epoch"] == 2
    assert result["last_record"]["active_rollout_steps"] == 300
    assert result["continuation"]["restored_components"] == {
        "model": True,
        "optimizer": True,
        "scheduler": True,
        "scaler": True,
        "data_loader_generator": True,
        "rng": True,
        "curriculum": True,
    }
    assert "val.forecast_300_nrmse" in best["metrics"]
    assert best["curriculum_state"]["best_epoch"] == 2
    assert last["complete_training_state"] is True
    assert last["scientific_config_sha256"] == child_context.config_sha256


def test_p2_continuation_rejects_protected_data_change(tmp_path):
    config_path = _workspace(tmp_path, "P2")
    context = prepare_sparse_experiment(config_path, project_root=tmp_path)
    trainer = SparseMultiEpochTrainer(
        context, _tiny_partial_pipeline("joint"), torch.device("cpu")
    )
    trainer.fit(
        max_epochs_this_run=1,
        max_train_batches=1,
        max_val_batches=1,
        run_kind="smoke",
    )
    checkpoint = context.run_dir / "checkpoints" / "last.pt"
    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["experiment_name"] = "p2_changed_data"
    changed["data"]["stride"] = 1
    changed["pipeline"]["continuation_checkpoint"] = str(checkpoint)
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    resolved = load_resolved_sparse_config(changed_path, project_root=tmp_path)

    with pytest.raises(ValueError, match="protected scientific controls"):
        validate_continuation_checkpoint(resolved, checkpoint)
