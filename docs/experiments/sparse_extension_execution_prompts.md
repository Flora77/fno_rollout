# H1、B5、B6 扩展实验执行提示词（节约 token 版）

## 1. 实验定义

| ID | 重构器 | 卷积/传播 | 跳跃连接 | 初始化 | RFNO |
|---|---|---|---|---|---|
| B4（已有） | Mask U-Net | 全普通卷积，显式 mask 通道 | 有 | 重构训练 | 冻结 |
| P1（已有） | PartialConv-MAE | 编码器全 PartialConv | 无 | MAE 预训练 | 冻结 |
| H1 | Hybrid U-Net | 仅首层 PartialConv，后续普通卷积 | 有 | 可预训练 | 冻结 |
| H1-A1 | PartialConv U-Net | 各编码尺度 PartialConv | 有 | 与 H1 相同 | 冻结 |
| H1-A2 | Hybrid no-skip | 仅首层 PartialConv | 无 | 与 H1 相同 | 冻结 |
| H1-A3 | Hybrid random-init | 与 H1 相同 | 有 | 随机 | 冻结 |
| B5-GNO | 显式 GNO | 传感器直接积分到 64x64 | 不适用 | 重构训练 | 冻结 |
| B5-GINO | GINO-style | GNO→16x16 潜网格 FNO→GNO | 不适用 | 重构训练 | 冻结 |
| B6-GNO/GINO | 与对应 B5 相同 | 与对应 B5 相同 | 不适用 | 对应 B5 best | 联合 |

H1 参数量约 85k，与 B4（90k）和 P1（86k）同量级。示例 GINO 采用
24 隐通道，约 91k 参数；显式 GNO 采用 128 隐通道，约 36k 参数，需同时
报告精度和参数效率，不能用任意综合评分掩盖规模差异。

## 2. 第一轮：只做 smoke

```text
使用 $sea-surface-sparse-experiments。

主要类别：learned_reconstruction_frozen。
实验ID：H1、B5、B6。

使用新增示例配置分别执行：配置验证、一个真实batch前向/反向和resume smoke。
不访问test，不启动正式多epoch训练。

检查：输入仅x_obs和obs_mask；填充值不变性；输出(B,60,64,64)；
frozen RFNO grad全None；joint时重构器和RFNO均有有限非零梯度；
无NaN/Inf；记录峰值显存和单batch时间。

节约token：只报告通过/失败、命令、耗时和阻塞项，不打印完整JSON。
```

配置：

- `config/sparse_experiments/h1_hybrid_pconv1_unet_frozen.example.json`
- `config/sparse_experiments/b5_gno_frozen.example.json`
- `config/sparse_experiments/b5_gino_frozen.example.json`
- `config/sparse_experiments/b6_gno_joint_30f.example.json`
- `config/sparse_experiments/b6_gino_joint_30f.example.json`

## 3. H1 主实验

```text
使用 $sea-surface-sparse-experiments。

主要类别：learned_reconstruction_frozen。
实验ID：H1。

基于h1_hybrid_pconv1_unet_frozen.example.json，使用与B4完全相同的
train/val split、mask_00006、B0归一化、seed、epoch、optimizer、损失和
冻结RFNO，正式训练“首层PartialConv+普通卷积U-Net”。不访问test。

以val缺测区重构NRMSE选择best.pt，随后完整评价val-300，输出重构、
F30-F300、Corr、SSP、gradient、per-MAT、训练时间和推理时间。

不得修改B4/P1已有结果；结束后只报告best epoch、关键指标和checkpoint。
```

## 4. H1 结构消融

从 H1 配置复制三个唯一运行配置，除指定字段外完全相同：

- H1-A1：`partialconv_levels=3`，`use_skip_connections=true`；
- H1-A2：`partialconv_levels=1`，`use_skip_connections=false`；
- H1-A3：同 H1 架构，但下游阶段不加载重构预训练 checkpoint。

```text
使用 $sea-surface-sparse-experiments。

主要类别：learned_reconstruction_frozen。
实验ID：H1-ablation。

只使用train/val，不访问test。执行三组严格消融：
1. first-PartialConv vs all-PartialConv，唯一变化partialconv_levels=1→3；
2. skip vs no-skip，唯一变化use_skip_connections=true→false；
3. pretrained vs random-init，采用相同30个下游epoch、数据顺序、优化器、
   30帧forecast supervision和冻结RFNO，唯一变化是否加载H1预训练best.pt。

B4和P1已有正式结果直接纳入结构对照，不重复训练；若其训练预算与本次
配对消融不一致，表中明确标为“外部锚点”，不得用于唯一变量因果结论。

输出module_ablation_summary.csv，分别报告重构缺测区NRMSE、F30-F300、
参数量和耗时；不构造综合评分。节约token，仅报告排序和差值。
```

消融解释规则：

- H1 与 H1-A1：PartialConv 使用深度的贡献；
- H1 与 H1-A2：跳跃连接的贡献；
- H1 pretrained 与 H1-A3：预训练初始化的贡献；
- B4 与 H1：输入端普通卷积+mask 通道和首层 PartialConv 的工程对照；
- P1 与 H1-A1：在均使用多层 PartialConv 时，MAE 无跳跃解码器和 U-Net
  跳跃解码器的结构对照。该比较若预算不同，只作支持性证据。

## 5. B5 固定 RFNO

```text
使用 $sea-surface-sparse-experiments。

主要类别：learned_reconstruction_frozen。
实验ID：B5-GNO、B5-GINO。

分别使用b5_gno_frozen.example.json和b5_gino_frozen.example.json，
在同一train/val、mask_00006、seed42、B0归一化与训练预算下训练重构器。
RFNO始终冻结且不进入optimizer。不访问test。

GNO显式输出64x64历史场；GINO必须执行传感器GNO lifting→16x16潜网格
FNO传播→输出GNO投影。以val缺测区重构NRMSE选best，然后完整val-300。

报告重构、F30-F300、参数量、显存、重构耗时和总推理耗时。只在两者
均完成后比较，不根据结果修改radius或latent_shape。

节约token：结束或异常时汇报摘要。
```

## 6. B6 联合 RFNO

```text
使用 $sea-surface-sparse-experiments。

主要类别：learned_reconstruction_joint。
实验ID：B6-GNO、B6-GINO。

从对应B5 best.pt初始化重构器，从锁定B0 checkpoint初始化RFNO。
先执行30帧联合训练；两者均进入optimizer，RFNO学习率为重构器的0.1倍。
不访问test，不使用teacher forcing。

除coupling=frozen→joint外，B5/B6成对实验保持模型、数据、mask、损失和
seed一致。检查重构器与RFNO均收到有限非零梯度，resume完整恢复。

30帧稳定后才生成30→60→120→180→240→300 curriculum配置；进入
300帧后仅按val forecast_300_nrmse选best。节约token，不逐epoch播报。
```

## 7. 推荐运行顺序

1. 全部配置验证和单batch smoke；
2. H1 seed42 主实验；
3. H1-A1、H1-A2、H1-A3 单种子消融；
4. 只对有希望的 H1/H1-A* 补 seed43、44；
5. B5-GNO 与 B5-GINO seed42；
6. 优先模型进入对应 B6 30帧联合训练；
7. 只有 B6-30 明显优于 B5 才扩展到 300帧；
8. 验证集锁定最终候选后，统一更新最终 test manifest；此前禁止访问 test。

## 8. 直接运行命令模板

```powershell
& "D:\software\miniconda3\envs\fno\python.exe" `
  ".codex\skills\sea-surface-sparse-experiments\scripts\validate_sparse_config.py" `
  "config\sparse_experiments\h1_hybrid_pconv1_unet_frozen.example.json"

& "D:\software\miniconda3\envs\fno\python.exe" `
  "scripts\sparse_surface\train_sparse_experiment.py" `
  "config\sparse_experiments\h1_hybrid_pconv1_unet_frozen.example.json" `
  --split-manifest "data\splits\sea_surface_bimodal_v1.json" `
  --smoke --max-epochs 1 --max-train-batches 1 --max-val-batches 1
```

将配置路径替换为 B5/B6 对应文件即可。正式训练时删除所有 `--smoke` 和
`--max-*` 参数。正式 val-300 需显式传入训练得到的 `best.pt`，并使用新的
唯一评价目录；不得使用 test。
