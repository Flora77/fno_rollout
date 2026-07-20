from dataclasses import dataclass
import os
import torch


@dataclass
class SeaSurfaceRolloutConfig:
    # -------------------------
    # data
    # -------------------------
    data_root: str = "./data/bimodal"
    variable: str = "height"
    input_steps: int = 60
    output_steps: int = 30
    stride: int = 4
    normalize: bool = True
    batch_size: int = 16
    val_batch_size: int = 16
    test_batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True

    # -------------------------
    # model
    # -------------------------
    # Options:
    #   fno / tfno
    #   fno_unet_gated_decoder   : FNO + single periodic U-Net residual correction
    #   fno_aunet_gated_decoder  : FNO + single periodic Attention U-Net residual correction
    model_arch: str = "fno_unet_gated_decoder"
    n_modes: tuple = (28, 28)
    hidden_channels: int = 32
    lifting_channels: int = 64
    projection_channels: int = 64
    n_layers: int = 4

    # FNO + single U-Net/AU-Net + gated residual
    # coarse = FNO(x)
    # z = [x, coarse] if use_context else coarse
    # residual = UNet(z) or AttentionUNet(z)
    # pred = coarse + residual_scale * sigmoid(GateNet(z)) * residual
    fno_unet_fno_arch: str = "fno"
    fno_unet_refiner_type: str = "unet"  # "unet" or "aunet"; model_arch has priority
    fno_unet_depth: int = 3
    fno_unet_base_channels: int = 32
    fno_unet_decoder_dropout: float = 0.0
    fno_unet_use_context: bool = True
    fno_unet_use_residual: bool = True
    fno_unet_residual_scale: float = 1.0

    # Gated residual: adaptive correction strength in space and predicted time channel.
    # 0.0 -> initial gate about 0.5; -1.0 -> more conservative initial gate about 0.27.
    fno_unet_use_gated_residual: bool = False
    fno_unet_gate_hidden_channels: int = 32
    fno_unet_gate_bias_init: float = 0.0

    # AU-Net attention gate. None/0 means automatic: min(gate_channels, skip_channels)//2.
    fno_aunet_attention_inter_channels: object = None

    # Periodic padding in all CNN/U-Net/Gate convolution blocks.
    # Options: "periodic"/"circular", "zero", "reflect".
    fno_unet_padding_mode: str = "periodic"

    # -------------------------
    # optimization
    # -------------------------
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    n_epochs: int = 100
    early_stop_patience: int = 100
    grad_clip_norm: float = 1.0

    # Auxiliary slope loss. Keeps local wave slopes/high-wavenumber details from being over-smoothed.
    use_spatial_gradient_loss: bool = True
    spatial_gradient_loss_weight: float = 0.05

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # lr scheduler
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "cosine"     # "step" / "cosine" / "plateau"
    lr_scheduler_step_size: int = 20
    lr_scheduler_gamma: float = 0.5
    lr_scheduler_t_max: int = n_epochs
    lr_scheduler_eta_min: float = 1e-6
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6

    # -------------------------
    # rollout training design
    # -------------------------
    use_long_rollout_curriculum: bool = True
    rollout_train_steps: tuple = (30, 60, 120, 180, 240, 300)
    rollout_curriculum_boundaries: tuple = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6)
    rollout_steps: int = 300
    rollout_stride: int = 4
    rollout_detach_context: bool = False

    use_segment_weighting: bool = True
    segment_weight_type: str = "linear"  # none / linear / power / exp
    segment_weight_min: float = 1.0
    segment_weight_max: float = 2.5
    segment_weight_power: float = 2.0
    normalize_segment_weights: bool = True

    use_within_chunk_temporal_weighting: bool = False
    chunk_time_weight_type: str = "linear"  # none / linear / power / exp
    chunk_time_weight_min: float = 1.0
    chunk_time_weight_max: float = 2.0
    chunk_time_weight_power: float = 2.0
    normalize_chunk_time_weights: bool = True

    # -------------------------
    # evaluation / plotting
    # -------------------------
    dt: float = 0.25
    evaluation_num_full_samples_to_save: int = 3
    evaluation_num_trace_points: int = 5
    spectral_high_k_ratio: float = 0.67
    spectral_band_split_ratios: tuple = (0.33, 0.67, 0.85)  # only for validation spectrum statistics
    plot_num_samples: int = 3
    plot_future_steps: tuple = (59, 149, 299)
    denormalize_for_plot: bool = True

    # -------------------------
    # experiment / io
    # -------------------------
    experiment_name: str = "mode_m28x28_h32_lp64"
    checkpoint_dir: str = "./checkpoints/hparam_sensitivity"
    log_dirname: str = "logs"
    plot_dirname: str = "plots"
    train_summary_csv: str = "ablation_train_summary.csv"
    val_summary_csv: str = "ablation_val_summary.csv"
    test_summary_csv: str = "ablation_test_summary.csv"
    save_val_mat: bool = True
    save_test_mat: bool = True

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.experiment_name}.pt")

    @property
    def plot_dir(self) -> str:
        return os.path.join(self.checkpoint_dir, self.plot_dirname, self.experiment_name)

    @property
    def train_summary_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.train_summary_csv)

    @property
    def val_summary_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.val_summary_csv)

    @property
    def test_summary_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.test_summary_csv)

    @property
    def log_dir(self) -> str:
        return os.path.join(self.checkpoint_dir, self.log_dirname)

    @property
    def val_mat_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_val_rollout.mat")

    @property
    def test_mat_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_test_rollout.mat")
