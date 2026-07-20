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
    input_steps: int = 40
    output_steps: int = 20
    stride: int = 4
    normalize: bool = True
    batch_size: int = 16
    val_batch_size: int = 16
    test_batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True

    # -------------------------
    # coarse predictor: FNO / TFNO
    # -------------------------
    model_arch: str = "fno"  # fno / tfno
    n_modes: tuple = (32, 32)
    hidden_channels: int = 32
    lifting_channels: int = 64
    projection_channels: int = 64
    n_layers: int = 4

    # -------------------------
    # optimization
    # -------------------------
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    n_epochs: int = 100
    early_stop_patience: int = 100
    grad_clip_norm: float = 1.0
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
    rollout_train_steps: tuple = (20, 40, 80, 120, 160, 240)
    rollout_curriculum_boundaries: tuple = (0.0, 0.10, 0.20, 0.30, 0.45, 0.60)
    rollout_steps: int = 240
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
    # coarse-to-fine design
    # -------------------------
    use_coarse_to_fine: bool = True

    # Training stage:
    #   "coarse_only"  : train only FNO coarse predictor, output is upsampled coarse field.
    #   "refiner_only" : load/freeze coarse predictor, train U-Net/CNN refiner.
    #   "joint"        : train coarse predictor and refiner together.
    c2f_train_stage: str = "joint"

    # Coarse predictor predicts a temporally coarser future sequence.
    # Example: rollout_steps=240, coarse_time_factor=4 => coarse_steps=60.
    c2f_coarse_time_factor: int = 4
    c2f_coarse_target_mode: str = "uniform"  # "last_in_group" / "uniform"
    c2f_temporal_upsample_mode: str = "linear"     # "linear" / "nearest"

    # Refiner receives [current 40-frame context, current coarse-base 20-frame chunk]
    # as channels and predicts a residual correction for that chunk.
    c2f_refiner_type: str = "unet"  # "unet" / "cnn"
    c2f_refiner_in_channels: int = 60   # input_steps + output_steps = 40 + 20
    c2f_refiner_out_channels: int = 20  # output_steps
    c2f_refiner_base_channels: int = 64
    c2f_refiner_depth: int = 3
    c2f_refiner_dropout: float = 0.0
    c2f_refiner_residual_scale: float = 0.2

    # Loss weights.
    c2f_coarse_loss_weight: float = 0.5
    c2f_base_loss_weight: float = 0.1  # only used in coarse_only as optional upsampled-base supervision

    # Optional pretrained coarse model path for refiner_only stage.
    # It can be either a c2f checkpoint with "model_state_dict" or a dict containing "coarse_state_dict".
    c2f_pretrained_coarse_path: str = ".\checkpoints\c2f_coarse_only.pt"
    c2f_freeze_coarse_in_refiner_only: bool = True

    # -------------------------
    # evaluation / plotting
    # -------------------------
    dt: float = 0.25
    evaluation_num_full_samples_to_save: int = 3
    evaluation_num_trace_points: int = 5
    spectral_high_k_ratio: float = 0.67
    spectral_band_split_ratios: tuple = (0.33, 0.67, 0.85)
    plot_num_samples: int = 3
    plot_future_steps: tuple = (19, 59, 119)
    denormalize_for_plot: bool = True

    # -------------------------
    # experiment / io
    # -------------------------
    experiment_name: str = "c2f_joint_unet_residual_scale02_losswCoarse05_baseLossw01"
    checkpoint_dir: str = "./checkpoints"
    log_dirname: str = "logs"
    plot_dirname: str = "plots"
    train_summary_csv: str = "ablation_train_summary.csv"
    val_summary_csv: str = "ablation_val_summary.csv"
    test_summary_csv: str = "ablation_test_summary.csv"
    save_val_mat: bool = True
    save_test_mat: bool = True

    @property
    def c2f_coarse_steps(self) -> int:
        factor = max(int(self.c2f_coarse_time_factor), 1)
        return int((int(self.rollout_steps) + factor - 1) // factor)

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
