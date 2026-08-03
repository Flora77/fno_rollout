# ViT-MAE 稀疏海面重构实验设计

## 1. 文献来源

空间 Patch MAE 采用 He 等提出的非对称 Masked Autoencoder 思路：编码器只处理可见
Patch，轻量解码器补入 mask token 并重建被遮挡区域。

时空 Tubelet 分支参考 Feichtenhofer 等的 MAE-ST。该工作将视频划分为时空 Patch，
采用与时空位置无关的随机遮挡，并同样只将可见 token 送入编码器。原论文在视频上
发现 90% 时空随机遮挡优于较低遮挡率。为保证本文 V1 与 V2 首轮比较只改变 token
划分方式，V2 主实验先保持与 V1 相同的 75% 遮挡率；90% 作为预先规定的 V2-A1
遮挡率消融，不根据测试集选择。

参考文献：

1. He K, Chen X, Xie S, et al. Masked Autoencoders Are Scalable Vision Learners.
   Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
   2022: 16000-16009. <https://arxiv.org/abs/2111.06377>
2. Feichtenhofer C, Fan H, Li Y, He K. Masked Autoencoders As Spatiotemporal
   Learners. Advances in Neural Information Processing Systems, 2022, 35:
   35946-35958. <https://proceedings.neurips.cc/paper_files/paper/2022/hash/e97d1081481a4017df96b51be31001d3-Abstract-Conference.html>

BibTeX：

```bibtex
@inproceedings{he2022masked,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and
          Doll{\\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and
             Pattern Recognition},
  pages={16000--16009},
  year={2022}
}

@inproceedings{feichtenhofer2022masked,
  title={Masked Autoencoders As Spatiotemporal Learners},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Li, Yanghao and He, Kaiming},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={35946--35958},
  year={2022}
}
```

## 2. 实验映射

| ID | Token 划分 | 初始化/预训练 | 下游阶段 | RFNO |
|---|---|---|---|---|
| V0 | 空间 4×4 Patch，60帧作通道 | 随机初始化 | 5%固定点稀疏重构 | 冻结且不参与损失 |
| V1 | 空间 4×4 Patch，60帧作通道 | 75%空间 MAE | 相同稀疏重构 | 冻结且不参与损失 |
| V2 | 10×4×4 Tubelet | 75%时空随机 MAE | 相同稀疏重构 | 冻结且不参与损失 |
| V3 | V1 锁定模型 | V1 best.pt | 30帧预测感知训练 | 冻结，梯度穿过其输入 |
| V4 | V1 锁定模型 | V1 best.pt | 30帧联合训练，满足门槛后扩展 | 联合，较小学习率 |

## 3. 公平性约束

- V0 与 V1 下游训练的唯一科学差异是重构器初始化。
- V1 与 V2 首轮比较保持遮挡率、训练预算、优化器、下游数据和损失相同，仅改变
  空间 Patch 与时空 Tubelet token 化。
- MAE 预训练只读取历史 `x_full`，先经随机遮挡算子生成 `x_obs` 和 `obs_mask`；
  完整历史张量不直接进入模型，未来 `y` 不参与预训练。
- 稀疏微调和预测阶段的模型输入始终只有 `x_obs` 与 `obs_mask`。
- V3 中 RFNO 参数不进入优化器且梯度始终为 `None`；V4 中重构器与 RFNO 均应
  获得有限非零梯度。
- 训练和验证不构造 test loader；最终方法确定前不得访问 test。

## 4. 首轮模型规模

- `patch_size=4`
- 空间模型：每个样本 16×16=256 个 token
- Tubelet 模型：`tubelet_size=10`，每个样本 6×16×16=1536 个 token
- encoder：dim=128，depth=4，heads=4
- decoder：dim=64，depth=2，heads=4
- 空间 MAE 遮挡率：75%
- Tubelet 主比较遮挡率：75%；V2-A1 预设消融为90%
- 二维周期 Fourier 空间位置编码；Tubelet 额外包含时间组 Fourier 编码

该配置用于首轮小模型验证。只有在两样本过拟合、真实 batch、AMP、resume 和显存
检查通过后，才允许执行三随机种子正式训练。
