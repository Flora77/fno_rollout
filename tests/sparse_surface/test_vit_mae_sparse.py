import copy
import json
from pathlib import Path

import torch
from torch import nn

from neuralop.models.reconstructors import (
    PeriodicConfidenceMaskAwarePoolingViTReconstructor,
    PeriodicMaskAwarePoolingViTReconstructor,
    PeriodicViTMaskedAutoencoder,
    random_token_observation_mask,
)
from neuralop.models.sparse_forecast_pipeline import LearnedReconstructionRFNOPipeline
from neuralop.training.masked_reconstruction import (
    MaskedReconstructionLoss,
    MaskedReconstructionPretrainer,
)
from neuralop.training.sparse_coupled_forecast import (
    SparseCoupledForecastTrainer,
    SparseCoupledTrainingConfig,
)
from neuralop.training.sparse_experiment_runner import (
    SUPPORTED_EXPERIMENTS,
    load_resolved_sparse_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config" / "sparse_experiments"


def _tiny_vit(tokenization="spatial_patch"):
    return PeriodicViTMaskedAutoencoder(
        input_steps=60,
        patch_size=4,
        tokenization=tokenization,
        tubelet_size=10,
        encoder_dim=16,
        encoder_depth=1,
        encoder_heads=4,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=2,
        mlp_ratio=2.0,
    )


def _tiny_mask_aware_vit():
    return PeriodicMaskAwarePoolingViTReconstructor(
        input_steps=60,
        patch_size=4,
        encoder_dim=16,
        encoder_depth=1,
        encoder_heads=4,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=2,
        mlp_ratio=2.0,
    )


def _tiny_confidence_vit():
    return PeriodicConfidenceMaskAwarePoolingViTReconstructor(
        input_steps=60,
        patch_size=4,
        encoder_dim=16,
        encoder_depth=1,
        encoder_heads=4,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=2,
        mlp_ratio=2.0,
    )


class TinyRFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(60, 30, 1)

    def forward(self, history):
        return self.projection(history)


def _batch(batch_size=2, size=8):
    x_full = torch.randn(batch_size, 60, size, size)
    obs_mask = torch.zeros_like(x_full)
    obs_mask[:, :, ::2, ::2] = 1.0
    return {
        "x_full": x_full,
        "x_obs": x_full * obs_mask,
        "obs_mask": obs_mask,
        "y": torch.randn(batch_size, 30, size, size),
    }


def test_spatial_vit_mae_encodes_visible_tokens_only_and_ignores_fill():
    torch.manual_seed(1)
    model = _tiny_vit().eval()
    x_full = torch.randn(2, 60, 8, 8)
    mask = torch.zeros_like(x_full)
    mask[:, :, :4, :4] = 1.0
    zero_fill = x_full * mask
    arbitrary_fill = torch.where(mask.bool(), x_full, torch.randn_like(x_full) * 100.0)

    with torch.no_grad():
        first, aux = model.forward_with_aux(zero_fill, mask)
        second = model(arbitrary_fill, mask)

    assert first.shape == x_full.shape
    assert torch.isfinite(first).all()
    assert aux["tokenization"] == "spatial_patch"
    # An 8x8 field with patch size 4 contains four tokens; exactly one is visible.
    torch.testing.assert_close(aux["visible_token_count"], torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(aux["encoder_sequence_length"], torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_tubelet_mae_reconstructs_history_and_drops_masked_tubelets():
    torch.manual_seed(2)
    model = _tiny_vit("tubelet").eval()
    x_full = torch.randn(1, 60, 8, 8)
    mask = random_token_observation_mask(
        x_full,
        patch_size=4,
        mask_ratio=0.75,
        tokenization="tubelet",
        tubelet_size=10,
    )

    with torch.no_grad():
        reconstruction, aux = model.forward_with_aux(x_full * mask, mask)

    # 6 temporal groups x 2 x 2 spatial patches = 24 tokens; 25% remain visible.
    assert reconstruction.shape == x_full.shape
    assert int(aux["visible_token_count"][0]) == 6
    assert int(aux["encoder_sequence_length"][0]) == 6
    assert torch.isfinite(reconstruction).all()


def test_all_missing_vit_mae_is_finite_without_fabricating_visible_tokens():
    model = _tiny_vit().eval()
    x_obs = torch.randn(1, 60, 8, 8) * 100.0
    mask = torch.zeros_like(x_obs)

    with torch.no_grad():
        reconstruction, aux = model.forward_with_aux(x_obs, mask)

    assert int(aux["visible_token_count"][0]) == 0
    # One learned empty-context token keeps the Transformer numerically defined.
    assert int(aux["encoder_sequence_length"][0]) == 1
    assert torch.isfinite(reconstruction).all()


def test_mask_aware_pooling_uses_observed_count_without_count_confidence():
    model = _tiny_mask_aware_vit().eval()
    x_obs = torch.zeros(1, 60, 8, 8)
    mask = torch.zeros_like(x_obs)
    x_obs[:, :, 0, 0] = 2.0
    x_obs[:, :, 1, 1] = 4.0
    mask[:, :, 0, 0] = 1.0
    mask[:, :, 1, 1] = 1.0
    grid = model._grid(8, 8)

    embedding_inputs, visible, observed_count = model._mask_aware_token_inputs(
        x_obs, mask, grid
    )

    values, validity = embedding_inputs.chunk(2, dim=-1)
    torch.testing.assert_close(
        values[0, 0], torch.full_like(values[0, 0], 3.0)
    )
    torch.testing.assert_close(
        validity[0, 0], torch.ones_like(validity[0, 0])
    )
    torch.testing.assert_close(
        observed_count[0, 0], torch.full_like(observed_count[0, 0], 2.0)
    )
    assert visible.tolist() == [[True, False, False, False]]
    # A different nonzero count with the same mean yields the same token input:
    # the count is diagnostic only and is not a confidence feature.
    one_point_x = torch.zeros_like(x_obs)
    one_point_mask = torch.zeros_like(mask)
    one_point_x[:, :, 0, 0] = 3.0
    one_point_mask[:, :, 0, 0] = 1.0
    one_point_inputs, _, one_point_count = model._mask_aware_token_inputs(
        one_point_x, one_point_mask, grid
    )
    torch.testing.assert_close(embedding_inputs[:, 0], one_point_inputs[:, 0])
    assert torch.all(one_point_count[:, 0] == 1)


def test_mask_aware_pooling_masks_boundaries_and_fill_invariance():
    torch.manual_seed(7)
    model = _tiny_mask_aware_vit().eval()
    x_full = torch.randn(1, 60, 8, 8)
    masks = {
        "all_observed": torch.ones_like(x_full),
        "all_missing": torch.zeros_like(x_full),
        "isolated": torch.zeros_like(x_full),
        "irregular": (torch.rand_like(x_full) > 0.7).to(x_full.dtype),
    }
    masks["isolated"][:, :, 2, 3] = 1.0

    with torch.no_grad():
        for name, mask in masks.items():
            fills = (
                torch.where(mask.bool(), x_full, torch.zeros_like(x_full)),
                torch.where(mask.bool(), x_full, torch.full_like(x_full, 100.0)),
                torch.where(mask.bool(), x_full, torch.randn_like(x_full) * 17.0),
            )
            outputs = [model(value, mask) for value in fills]
            assert outputs[0].shape == (1, 60, 8, 8), name
            assert torch.isfinite(outputs[0]).all(), name
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
            torch.testing.assert_close(outputs[0], outputs[2], rtol=0.0, atol=0.0)

        _, all_missing_aux = model.forward_with_aux(
            torch.zeros_like(x_full), masks["all_missing"]
        )
        _, isolated_aux = model.forward_with_aux(
            x_full * masks["isolated"], masks["isolated"]
        )
    assert int(all_missing_aux["visible_token_count"][0]) == 0
    assert int(isolated_aux["visible_token_count"][0]) == 1


def test_mask_aware_pooling_matches_v0_parameter_budget_and_architecture():
    v0 = PeriodicViTMaskedAutoencoder()
    pc_d1 = PeriodicMaskAwarePoolingViTReconstructor()
    v0_parameters = sum(parameter.numel() for parameter in v0.parameters())
    pc_d1_parameters = sum(parameter.numel() for parameter in pc_d1.parameters())

    assert pc_d1_parameters == v0_parameters
    assert pc_d1.encoder_dim == v0.encoder_dim == 128
    assert pc_d1.decoder_dim == v0.decoder_dim == 64
    assert len(pc_d1.encoder.layers) == len(v0.encoder.layers) == 4
    assert len(pc_d1.decoder.layers) == len(v0.decoder.layers) == 2
    assert pc_d1.patch_size == v0.patch_size == 4


def test_confidence_vit_reports_empty_partial_and_full_patch_confidence():
    model = _tiny_confidence_vit().eval()
    x_obs = torch.randn(1, 60, 8, 8)
    mask = torch.zeros_like(x_obs)
    mask[:, :, :4, :4] = 1.0
    mask[:, :, :2, 4:6] = 1.0

    with torch.no_grad():
        reconstruction, aux = model.forward_with_aux(x_obs * mask, mask)

    expected = torch.tensor([[1.0, 0.25, 0.0, 0.0]])
    torch.testing.assert_close(aux["patch_confidence"], expected)
    assert aux["visible_token_count"].tolist() == [2]
    assert reconstruction.shape == (1, 60, 8, 8)
    assert torch.isfinite(reconstruction).all()


def test_confidence_vit_adds_learned_mapping_without_gating():
    model = _tiny_confidence_vit().eval()
    x_obs = torch.ones(1, 60, 8, 8)
    mask = torch.zeros_like(x_obs)
    mask[:, :, :4, :4] = 1.0
    mask[:, :, :2, 4:6] = 1.0
    grid = model._grid(8, 8)
    embedding_inputs, visible, observed_count = model._mask_aware_token_inputs(
        x_obs * mask, mask, grid
    )
    confidence = model._patch_confidence(observed_count, grid)
    with torch.no_grad():
        model.confidence_embedding.weight.fill_(1.0)
        model.confidence_embedding.bias.zero_()
        base_tokens = model.patch_embedding(embedding_inputs)
        confidence_tokens = base_tokens + model.confidence_embedding(
            confidence.unsqueeze(-1)
        )

    expected_delta = confidence.unsqueeze(-1).expand_as(base_tokens)
    torch.testing.assert_close(confidence_tokens - base_tokens, expected_delta)
    assert visible.tolist() == [[True, True, False, False]]
    assert isinstance(model.confidence_embedding, nn.Linear)


def test_confidence_vit_masks_boundaries_and_is_fill_invariant():
    torch.manual_seed(8)
    model = _tiny_confidence_vit().eval()
    x_full = torch.randn(1, 60, 8, 8)
    masks = {
        "all_observed": torch.ones_like(x_full),
        "all_missing": torch.zeros_like(x_full),
        "isolated": torch.zeros_like(x_full),
        "irregular": (torch.rand_like(x_full) > 0.7).to(x_full.dtype),
    }
    masks["isolated"][:, :, 2, 3] = 1.0

    with torch.no_grad():
        for name, mask in masks.items():
            fills = (
                torch.where(mask.bool(), x_full, torch.zeros_like(x_full)),
                torch.where(mask.bool(), x_full, torch.full_like(x_full, 100.0)),
                torch.where(mask.bool(), x_full, torch.randn_like(x_full) * 17.0),
            )
            grid = model._grid(8, 8)
            token_inputs = [
                model._mask_aware_token_inputs(value, mask, grid)[0]
                for value in fills
            ]
            outputs = [model(value, mask) for value in fills]
            torch.testing.assert_close(
                token_inputs[0], token_inputs[1], rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                token_inputs[0], token_inputs[2], rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
            torch.testing.assert_close(outputs[0], outputs[2], rtol=0.0, atol=0.0)
            assert outputs[0].shape == (1, 60, 8, 8), name
            assert torch.isfinite(outputs[0]).all(), name

        _, all_observed_aux = model.forward_with_aux(
            x_full, masks["all_observed"]
        )
        _, all_missing_aux = model.forward_with_aux(
            torch.zeros_like(x_full), masks["all_missing"]
        )
        _, isolated_aux = model.forward_with_aux(
            x_full * masks["isolated"], masks["isolated"]
        )
    torch.testing.assert_close(
        all_observed_aux["patch_confidence"], torch.ones(1, 4)
    )
    torch.testing.assert_close(
        all_missing_aux["patch_confidence"], torch.zeros(1, 4)
    )
    assert int(all_missing_aux["visible_token_count"][0]) == 0
    assert int(isolated_aux["visible_token_count"][0]) == 1


def test_confidence_vit_preserves_pc_d1_architecture_and_parameter_budget():
    v0 = PeriodicViTMaskedAutoencoder()
    pc_d1 = PeriodicMaskAwarePoolingViTReconstructor()
    pc_d2 = PeriodicConfidenceMaskAwarePoolingViTReconstructor()
    v0_parameters = sum(parameter.numel() for parameter in v0.parameters())
    pc_d1_parameters = sum(parameter.numel() for parameter in pc_d1.parameters())
    pc_d2_parameters = sum(parameter.numel() for parameter in pc_d2.parameters())

    assert pc_d1_parameters == v0_parameters
    assert pc_d2_parameters == v0_parameters + 2 * v0.encoder_dim
    assert abs(pc_d2_parameters - v0_parameters) / v0_parameters <= 0.05
    assert pc_d2.patch_embedding.in_features == pc_d1.patch_embedding.in_features
    assert pc_d2.encoder_dim == pc_d1.encoder_dim
    assert pc_d2.decoder_dim == pc_d1.decoder_dim
    assert len(pc_d2.encoder.layers) == len(pc_d1.encoder.layers)
    assert len(pc_d2.decoder.layers) == len(pc_d1.decoder.layers)


def test_mae_pretraining_uses_exact_patch_mask_and_never_reads_future_y():
    torch.manual_seed(3)
    model = _tiny_vit()
    trainer = MaskedReconstructionPretrainer(
        model,
        MaskedReconstructionLoss(
            hidden_weight=1.0,
            observation_weight=0.0,
            history_gradient_weight=0.0,
        ),
        mae_mask_ratio=0.75,
        mae_patch_size=4,
        mae_tokenization="spatial_patch",
    )
    batch = _batch()

    comparison = copy.deepcopy(trainer)
    first = trainer.forward_batch(batch)
    changed_future = {**batch, "y": torch.full_like(batch["y"], 1.0e9)}
    second = comparison.forward_batch(changed_future)

    assert float(first["supervision_mask"].mean()) == 0.25
    torch.testing.assert_close(first["supervision_mask"], second["supervision_mask"])
    torch.testing.assert_close(first["reconstruction"], second["reconstruction"])
    torch.testing.assert_close(first["losses"]["total"], second["losses"]["total"])


def test_mae_validation_mask_is_repeatable_while_training_mask_changes():
    trainer = MaskedReconstructionPretrainer(
        _tiny_vit(),
        MaskedReconstructionLoss(),
        mae_mask_ratio=0.75,
        mae_patch_size=4,
        evaluation_mask_seed=123,
    )
    batch = _batch(batch_size=8)

    first_val = trainer.validate_batch(batch)["supervision_mask"]
    second_val = trainer.validate_batch(batch)["supervision_mask"]
    torch.testing.assert_close(first_val, second_val)

    trainer.set_mask_mode(training=True)
    first_train = trainer.forward_batch(batch)["supervision_mask"]
    second_train = trainer.forward_batch(batch)["supervision_mask"]
    assert not torch.equal(first_train, second_train)


def test_common_pretraining_mask_sequence_is_architecture_independent_and_resumable():
    class PassThroughReconstructor(nn.Module):
        def forward(self, x_obs, obs_mask):
            return x_obs

    batch = _batch(batch_size=2)
    criterion = MaskedReconstructionLoss(
        hidden_weight=1.0,
        observation_weight=0.1,
        history_gradient_weight=0.05,
    )
    first = MaskedReconstructionPretrainer(
        PassThroughReconstructor(),
        criterion,
        mae_mask_ratio=0.75,
        mae_patch_size=4,
        training_mask_seed=1042,
    )
    torch.rand(137)
    second = MaskedReconstructionPretrainer(
        PassThroughReconstructor(),
        criterion,
        mae_mask_ratio=0.75,
        mae_patch_size=4,
        training_mask_seed=1042,
    )
    first.set_mask_mode(training=True)
    second.set_mask_mode(training=True)

    first_mask = first.forward_batch(batch)["supervision_mask"]
    second_mask = second.forward_batch(batch)["supervision_mask"]
    torch.testing.assert_close(first_mask, second_mask)

    encoded_state = first.training_mask_state()
    assert encoded_state is not None
    expected_next = first.forward_batch(batch)["supervision_mask"]
    resumed = MaskedReconstructionPretrainer(
        PassThroughReconstructor(),
        criterion,
        mae_mask_ratio=0.75,
        mae_patch_size=4,
        training_mask_seed=9999,
    )
    resumed.load_training_mask_state(encoded_state)
    resumed.set_mask_mode(training=True)
    torch.testing.assert_close(
        resumed.forward_batch(batch)["supervision_mask"],
        expected_next,
    )


def test_vit_mae_frozen_and_joint_rfno_gradient_contracts():
    torch.manual_seed(4)
    batch = _batch(batch_size=1)
    for coupling in ("frozen", "joint"):
        pipeline = LearnedReconstructionRFNOPipeline(
            TinyRFNO(), _tiny_vit(), coupling=coupling
        )
        config = SparseCoupledTrainingConfig(
            coupling=coupling,
            forecast_supervision=True,
            max_rollout_steps=30,
            use_rollout_curriculum=False,
            hidden_weight=1.0,
            observation_weight=0.0,
            history_gradient_weight=0.0,
            rollout_weight=1.0,
            forecast_gradient_weight=0.0,
            spectrum_weight=0.0,
        )
        trainer = SparseCoupledForecastTrainer(pipeline, config)
        optimizer = trainer.build_optimizer()
        result = trainer.train_batch(batch, optimizer, active_rollout_steps=30)

        assert result["history_reconstruction"].shape == (1, 60, 8, 8)
        assert result["forecast"].shape == (1, 30, 8, 8)
        reconstructor_grads = [
            parameter.grad
            for parameter in pipeline.reconstructor.parameters()
            if parameter.requires_grad
        ]
        assert any(
            gradient is not None and torch.count_nonzero(gradient)
            for gradient in reconstructor_grads
        )
        assert all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in reconstructor_grads
        )
        rfno_grads = [parameter.grad for parameter in pipeline.rfno.parameters()]
        if coupling == "frozen":
            assert all(not parameter.requires_grad for parameter in pipeline.rfno.parameters())
            assert all(gradient is None for gradient in rfno_grads)
        else:
            assert any(
                gradient is not None and torch.count_nonzero(gradient)
                for gradient in rfno_grads
            )
            assert all(torch.isfinite(gradient).all() for gradient in rfno_grads)


def test_v0_and_v1_downstream_are_strict_initialization_pair():
    v0 = json.loads(
        (CONFIG_ROOT / "v0_vit_random_sparse_finetune.example.json").read_text(
            encoding="utf-8"
        )
    )
    v1 = json.loads(
        (CONFIG_ROOT / "v1_vit_mae_sparse_finetune.example.json").read_text(
            encoding="utf-8"
        )
    )
    assert v0["model"] == v1["model"]
    left, right = copy.deepcopy(v0), copy.deepcopy(v1)
    for payload in (left, right):
        payload.pop("experiment_id")
        payload.pop("experiment_name")
        payload["pipeline"].pop("reconstructor_checkpoint", None)
    assert left == right


def test_v2_a1_changes_only_pretraining_mask_ratio():
    baseline = json.loads(
        (CONFIG_ROOT / "v2_tubelet_mae_pretrain.example.json").read_text(
            encoding="utf-8"
        )
    )
    mask90 = json.loads(
        (CONFIG_ROOT / "v2_a1_tubelet_mae_pretrain_mask90.example.json").read_text(
            encoding="utf-8"
        )
    )
    baseline.pop("experiment_name")
    mask90.pop("experiment_name")
    baseline_ratio = baseline["training"].pop("mae_mask_ratio")
    mask90_ratio = mask90["training"].pop("mae_mask_ratio")
    assert baseline_ratio == 0.75
    assert mask90_ratio == 0.9
    assert baseline == mask90


def test_runnable_vit_pretraining_configs_resolve_without_test_access(tmp_path):
    for filename, experiment_id in (
        ("v0_vit_random_sparse_finetune.example.json", "V0"),
        ("v1_vit_mae_pretrain.example.json", "V1"),
        ("v2_tubelet_mae_pretrain.example.json", "V2"),
        ("pc_d1_mask_aware_pooling.example.json", "PC-D1"),
        ("pc_d2_confidence.example.json", "PC-D2"),
    ):
        resolved = load_resolved_sparse_config(
            CONFIG_ROOT / filename,
            project_root=PROJECT_ROOT,
            run_dir=tmp_path / experiment_id.lower(),
            device="cpu",
        )
        assert experiment_id in SUPPORTED_EXPERIMENTS
        assert resolved["experiment_id"] == experiment_id
        assert resolved["runtime"]["hash_data_files"] is True
