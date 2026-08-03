# ViT-MAE / Tubelet-MAE 后续实验提示词

以下提示词按顺序使用。每次只执行一个阶段，训练和验证均不得构造 test loader。
已有 test 已经揭示，因此这些新增方法只能用于验证集扩展研究；若未来希望作新的
确认性测试，应先冻结一套此前从未访问的外部/新增 holdout 数据。

## 0. 正式实验前只读审计

```text
使用 $sea-surface-sparse-experiments。

主要类别：evaluation_only。
实验族：V0、V1、V2、V3、V4。

只读审计ViT-MAE/Tubelet-MAE新路径，不修改代码、不启动训练、不访问test。

检查：
- 编码器是否只接收可见token，mask token是否只进入解码器；
- 空间Patch是否把60帧作为通道；
- Tubelet是否为10×4×4且采用时空无关随机遮挡；
- mask始终为1=观测、0=缺测；
- MAE预训练是否只由x_full构造遮挡后的x_obs，y是否完全未使用；
- V0/V1下游配置除初始化外是否完全一致；
- V1/V2严格token化比较是否都使用75%遮挡；
- V2-A1是否仅把遮挡率75%改为90%；
- V3冻结RFNO、V4联合RFNO的梯度和optimizer契约；
- train/val是否不构造test loader。

输出GO或NO-GO、阻塞项和涉及文件。
节约token：只报告阻塞项，不输出完整配置。
```

## 1. 两样本过拟合与真实batch验收

```text
使用 $sea-surface-sparse-experiments。

主要类别：pretraining。
实验ID：V0、V1、V2 infrastructure acceptance。

不启动正式训练、不访问test。

对空间ViT-MAE与Tubelet-MAE分别执行：
1. 两样本过拟合；
2. 一个真实train batch前向和反向；
3. AMP有限值检查；
4. last.pt保存并resume一个batch；
5. 缺测填充值0、100、随机值不变性；
6. 全观测和全缺测边界检查。

验收：
- reconstruction=(B,60,64,64)；
- 空间MAE在75%遮挡下编码64/256个可见token；
- Tubelet-MAE在75%遮挡下编码384/1536个可见token；
- loss明显下降且梯度有限；
- 不读取y，不构造test loader。

节约token：仅报告PASS/FAIL、峰值显存、resume结果和阻塞项。
```

## 2. V1空间MAE短程pilot

```text
使用 $sea-surface-sparse-experiments。

主要类别：pretraining。
实验ID：V1-pilot。

使用config/sparse_experiments/v1_vit_mae_pretrain.example.json，
只运行3到5个真实epoch，不访问test。

固定：
- train/val冻结清单和B0训练归一化；
- patch_size=4，encoder=128/4层/4头，decoder=64/2层/4头；
- 75%随机空间Patch遮挡；
- 编码器只处理可见token；
- y不参与输入或损失；
- 仅缺失Patch像素重构损失；
- 保存last.pt并验证resume。

验收：
- train/val masked reconstruction loss整体下降；
- 无NaN/Inf或显存持续增长；
- 可见token数与配置一致；
- 不构造test loader。

节约token：不逐epoch播报，结束后只报告趋势、显存、checkpoint和是否允许正式预训练。
```

## 3. V1空间MAE正式三seed预训练

```text
使用 $sea-surface-sparse-experiments。

主要类别：pretraining。
实验ID：V1-spatial-MAE。

基于已通过pilot的配置，正式执行seed=[42,43,44]的空间MAE预训练。
不访问test，不修改模型或损失。

共同设置：
- 75%随机空间Patch遮挡；
- 100 epoch上限和相同早停规则；
- 相同train/val、优化器、scheduler、batch size；
- 只改变runtime.seed；
- 按val masked-region NRMSE选best.pt；
- 分别保存best/last、训练曲线、实际epoch、hash、时间和峰值显存。

节约token：不逐epoch轮询，仅在异常或三seed全部结束后报告checkpoint路径、hash和均值±样本标准差。
```

## 4. V0与V1严格初始化消融

```text
使用 $sea-surface-sparse-experiments。

主要类别：learned_reconstruction_frozen。
实验ID：V0-vs-V1-initialization。

执行空间Sparse ViT随机初始化与MAE预训练初始化的三seed严格配对下游实验。
seed=[42,43,44]，不访问test。

共同设置：
- 相同split、mask_00006、数据顺序、B0归一化；
- 相同ViT结构、30个实际epoch、AdamW、lr、scheduler和损失；
- 只训练稀疏历史重构，不使用forecast loss；
- RFNO冻结且不进入optimizer；
- 两组均从epoch1、global_step0和空optimizer状态开始。

唯一差异：
- V0不加载reconstructor checkpoint；
- V1加载对应seed的空间MAE best.pt；
- 禁止用resume或continue-from代替初始化。

完成后用同一val评价器输出缺测区重构NRMSE、Corr、SSP、
gradient NRMSE、per-MAT、学习曲线和效率；
给出V1−V0配对差值及Student-t 95%置信区间。

节约token：只报告显著性、均值±标准差和推荐。
```

## 5. V2时空Tubelet实验与90%遮挡消融

```text
使用 $sea-surface-sparse-experiments。

主要类别：pretraining。
实验ID：V2、V2-A1。

先运行seed42，不访问test。

第一组使用
config/sparse_experiments/v2_tubelet_mae_pretrain.example.json，
采用10×4×4 Tubelet和75%时空无关随机遮挡。

第二组使用
config/sparse_experiments/v2_a1_tubelet_mae_pretrain_mask90.example.json，
唯一变化为mae_mask_ratio：0.75→0.90。

两组保持train/val、模型、epoch、优化器、损失和seed完全相同。
分别完成预训练，再把对应best.pt填入
config/sparse_experiments/v2_tubelet_mae_sparse_finetune.example.json，
使用与V0/V1相同的稀疏下游预算训练和评价。

输出：
- V2-75与V1的严格token化差异；
- V2-90与V2-75的遮挡率差异；
- 重构指标、参数量、显存、训练和推理时间。

门槛：
- 若V2任一版本在seed42的缺测区NRMSE和稳定性均不优于V1，不补seed43/44；
- 若具有竞争力，再补三seed并报告配对95%置信区间。

节约token：仅报告门槛结论和是否扩展seed。
```

## 6. V0–V2重构层验证集决策

```text
使用 $sea-surface-sparse-experiments。

主要类别：evaluation_only。
实验ID：vit-mae-reconstruction-val-scorecard。

只读汇总V0、V1、V2-75、V2-90以及已有B4、H1-A4的val结果。
不训练、不访问test、不构造综合评分。

输出：
- 缺测区重构NRMSE、Corr、SSP、gradient NRMSE；
- 三seed均值和样本标准差；
- V1−V0预训练配对差值及95%CI；
- V2-75−V1 token化配对差值；
- V2-90−V2-75遮挡率配对差值；
- per-MAT最差值、参数量、训练时间、显存和推理时间；
- reconstruction_val_scorecard.csv；
- 是否允许最佳ViT重构器进入V3。

节约token：只报告排名、显著性和推荐。
```

## 7. V3冻结RFNO预测感知训练

```text
使用 $sea-surface-sparse-experiments。

主要类别：learned_reconstruction_frozen。
实验ID：V3。

前提：V1或V2在三seed验证中稳定且具备竞争力；否则停止。
将锁定的最佳ViT重构器checkpoint写入
config/sparse_experiments/v3_vit_mae_frozen_rfno.example.json。

执行30帧forecast-aware训练，不访问test：
- RFNO从B0 checkpoint恢复；
- RFNO requires_grad=false、不进入optimizer、grad始终None；
- 梯度允许穿过RFNO输入进入重构器；
- active_rollout_steps=30，不使用teacher forcing；
- 输入模型的字段仅x_obs和obs_mask；
- 保存完整best/last/resume状态。

训练后用同一评价器完整运行val-300，输出重构、F30–F300、
Corr、SSP、gradient、error-growth、per-MAT和效率。

节约token：结束后只报告V3相对其重构初始化模型的差值和是否允许进入V4。
```

## 8. V4联合RFNO的30帧门槛实验

```text
使用 $sea-surface-sparse-experiments。

主要类别：learned_reconstruction_joint。
实验ID：V4-D1。

从与V3相同的锁定ViT重构器checkpoint和B0 RFNO初始化，
执行30帧联合训练，不访问test。

要求：
- coupling=joint；
- 重构器和RFNO均进入optimizer；
- RFNO学习率为重构器的0.1倍；
- active_rollout_steps=30；
- 不使用teacher forcing；
- 两部分均获得有限非零梯度；
- AMP=false；
- best/last/resume状态完整；
- P1/B4/V3冻结模式回归测试继续通过。

训练后完整评价val-300。
门槛：V4的F30与F300均不劣于V3，且无稳定性或显存异常，才允许长序列训练。

节约token：不逐epoch播报，只报告门槛PASS/FAIL、关键差值和checkpoint。
```

## 9. V4长序列curriculum

```text
使用 $sea-surface-sparse-experiments。

主要类别：learned_reconstruction_joint。
实验ID：V4-D2。

前提：V4-D1门槛PASS；否则停止并保留V3。
从V4-D1 best.pt严格continue-from，执行30→60→120→180→240→300 curriculum。
不访问test。

要求：
- 恢复optimizer、scheduler、scaler、RNG和curriculum状态；
- rollout只使用重构历史和模型自身预测；
- 不用真实未来更新上下文；
- 每阶段记录active_rollout_steps及原始/加权loss；
- 进入300帧后只按val forecast_300_nrmse选择best.pt；
- 同时记录Corr、SSP、gradient误差、error-growth、时间和峰值显存；
- NaN、梯度爆炸或显存异常时安全保存last并停止。

节约token：不轮询日志，仅在异常或自然结束后报告最佳epoch、指标、路径和hash。
```

## 10. 新增方法最终验证集汇总

```text
使用 $sea-surface-sparse-experiments。

主要类别：evaluation_only。
实验ID：vit-mae-extended-val-scorecard。

只读汇总B4、H1-A4、V0、V1、V2、V3和满足门槛时的V4验证集结果。
不训练、不访问test、不构造任意综合评分。

输出：
- 重构与F30–F300 NRMSE、Corr、SSP、gradient NRMSE；
- 三seed均值和样本标准差；
- 所有严格配对因素效应及95%CI；
- per-MAT最差值、参数量、训练时间、显存和推理时间；
- 固定布局排名；
- vit_mae_extended_val_scorecard.csv；
- 论文能够支持和不能支持的结论。

注意：已有test已经访问，本轮不得用既有test重新选择或证明新增方法。
节约token：只报告排名、显著性、推荐和适用边界。
```
