from dataclasses import dataclass
import os
import torch


@dataclass
class SeaSurfaceRolloutConfig:
    # -------------------------
    # data
    # -------------------------
    data_root: str = "./data/bimodal_1s"
    variable: str = "height"

    # RNO/MSFNO-RNO setting:
    # 40 input frames -> one forward predicts 20 frames -> autoregressive rollout to 240 frames.
    input_steps: int = 16
    output_steps: int = 16
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
    # fno / tfno / msfno
    model_arch: str = "msfno"

    # Base FNO settings used by each MSFNO branch.
    # The paper commonly uses 32 spatial modes on a 64x64 grid; if GPU memory is tight, use (28, 28).
    n_modes: tuple = (32, 32)
    hidden_channels: int = 32
    lifting_channels: int = 64
    projection_channels: int = 64
    n_layers: int = 4

    # -------------------------
    # MSFNO settings
    # -------------------------
    # Paper-style multi-scale branches. Main paper setting for N branches:
    #     c_i = {0.5, 1, 2, 4, ..., 2^(N-2)}
    # For N=4 this is (0.5, 1, 2, 4). Appendix B also tests (1, 2, 4, 8).
    msfno_branch_arch: str = "fno"       # fno / tfno
    msfno_scales: tuple = (0.5, 1.0, 2.0, 4.0)

    # To keep parameter count comparable with a width=32 single FNO, each branch uses width≈16.
    # For a larger paper-style setting, try msfno_scales=(0.5,1,2,4,8,16,32,64) and width_factor=1.0.
    msfno_branch_width_factor: float = 0.5
    msfno_branch_hidden_channels: int = 0       # 0 -> hidden_channels * factor
    msfno_branch_lifting_channels: int = 0      # 0 -> lifting_channels * factor
    msfno_branch_projection_channels: int = 0   # 0 -> projection_channels * factor

    # Complete paper-style scaling adapted to this 2D-FNO code:
    #     branch input z_i = [c_i * eta, c_i * x, c_i * y].
    msfno_scale_input_field: bool = True  # 缩放输入幅值
    msfno_add_scaled_coords: bool = True  # 在输入中添加坐标分量
    msfno_scale_coordinates: bool = True  # 缩放坐标范围
    msfno_coord_range: tuple = (0.0, 1.0) # 坐标缩放范围
    msfno_output_scale: bool = False      # 是否缩放输出
    msfno_branch_positional_embedding: object = None # 是否使用位置编码

    # fusion = conv follows the paper's CNN-filter idea.
    # Alternatives for ablation: weighted_sum / mean /conv
    msfno_fusion: str = "conv"
    msfno_conv_hidden_channels: tuple = (32, 64, 32)
    # msfno_conv_hidden_channels: tuple = (8, 8)
    msfno_conv_kernel_size: tuple = (3, 3, 3)
    msfno_conv_norm: str = "batch"       # batch / instance / none
    msfno_conv_activation: str = "relu"  # relu / gelu / silu / sin / none

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
    lr_scheduler_type: str = "cosine"     # step / cosine / plateau
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
    # rollout_train_steps: tuple = (20, 40, 80, 120, 160, 240)
    # rollout_curriculum_boundaries: tuple = (0.0, 0.10, 0.20, 0.30, 0.45, 0.60)
    # rollout_train_steps = (40, 80, 160, 240, 320, 480)
    # rollout_curriculum_boundaries = (0.0, 0.10, 0.25, 0.40, 0.60, 0.75)
    # rollout_train_steps = (16,)
    # rollout_curriculum_boundaries =(0.0,)  
    rollout_train_steps: tuple = (16, 32, 48, 64, 80, 90)
    rollout_curriculum_boundaries: tuple = (0.0, 0.10, 0.20, 0.30, 0.45, 0.60)
    rollout_steps: int = 90
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
    dt: float = 1.0
    evaluation_num_full_samples_to_save: int = 3
    evaluation_num_trace_points: int = 5
    spectral_high_k_ratio: float = 0.67
    spectral_band_split_ratios: tuple = (0.33, 0.67, 0.85)
    plot_num_samples: int = 3
    plot_future_steps: tuple = (19,49,89)
    denormalize_for_plot: bool = True

    # -------------------------
    # experiment / io
    # -------------------------
    experiment_name: str = "msfno_16_90_rollout"
    checkpoint_dir: str = "./checkpoints"
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
