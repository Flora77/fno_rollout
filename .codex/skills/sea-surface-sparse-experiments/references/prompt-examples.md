# Prompt examples

Use the skill explicitly because implicit invocation is disabled.

## 1. Read-only audit

```text
使用 $sea-surface-sparse-experiments。

只读审计当前仓库，为 Stage 1 稀疏观测数据层制定最小实现方案。
本轮不要修改代码，不要启动训练。

给出：涉及文件、样本字段、张量形状、mask语义、泄漏风险、测试清单。
```

## 2. Implement the common sparse wrapper

```text
使用 $sea-surface-sparse-experiments。

只完成 Stage 1：新增 SparseSeaSurfaceDataset wrapper。
保留 SeaSurfaceSimpleDataset 的 {x,y} 行为。
从共享 manifest 加载 mask，返回 x_full、x_obs、obs_mask、y、mask_id。
mask 中 1=观测，0=缺测；不得把 x_full 传入模型。

验收：形状测试、复现性测试、mask语义测试、无未来信息泄漏测试。
不要实现模型，不要启动完整训练。
```

## 3. Implement PartialConv only

```text
使用 $sea-surface-sparse-experiments。

只实现周期边界 PartialConv2d 和单元测试，不实现MAE或RFNO连接。
输入 value 和 mask；mask为1表示有效。
局部无有效值时输出0并保持输出mask为0。

验收：全有效、全缺失、孤立点、随机孔洞；改变缺失位置填充值不得改变输出。
```

## 4. Implement P1

```text
使用 $sea-surface-sparse-experiments。

实现 P1：PartialConv-MAE 重构60帧历史，再送入冻结的现有RFNO预测30帧。
复用已有RFNO checkpoint和归一化，不改其内部实现。
分别返回并记录 history_reconstruction 和 forecast。

验收：RFNO不进入optimizer，backward后RFNO grad全部为None；运行一个batch测试。
不要启动完整训练。
```

## 5. Add P2 joint mode

```text
使用 $sea-surface-sparse-experiments。

在P1管线中增加 coupling=joint，不复制trainer。
先做30帧联合损失，再预留现有300帧curriculum接口。
分别记录hidden reconstruction、observation consistency、rollout、gradient、spectrum损失。

验收：重构器和RFNO都有有限梯度，无NaN；frozen模式回归测试仍通过。
```

## 6. Review without editing

```text
使用 $sea-surface-sparse-experiments。

只审查当前稀疏观测实验diff，不修改文件。
检查完整场/未来场泄漏、mask语义、冻结状态、梯度路径、归一化、rollout teacher forcing、
mask公平性、指标混淆和测试覆盖。按严重程度给出文件与行号。
```

## Prompt formula

Every implementation prompt should state:

1. skill name;
2. one stage and one scientific objective;
3. allowed and forbidden change scope;
4. tensor and mask contracts;
5. freeze/joint policy;
6. executable acceptance tests;
7. whether expensive training is authorized.
