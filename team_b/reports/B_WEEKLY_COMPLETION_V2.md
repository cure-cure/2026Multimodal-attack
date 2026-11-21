# B 图像模块 v2：阈值、分类和固定划分交付

## CIDER 阈值更正

验证集最大化 balanced accuracy/F1 后，严格 `>` 阈值为 `0.0022773519`；旧阈值为 `0.0056802332`。test 未参与选阈值。

| test 指标 | 旧阈值 | v2 阈值 |
|---|---:|---:|
| accuracy | 0.7000 | 0.9055 |
| balanced_accuracy | 0.7000 | 0.9055 |
| precision | 1.0000 | 0.9421 |
| recall | 0.4000 | 0.8641 |
| f1 | 0.5714 | 0.9014 |
| fpr | 0.0000 | 0.0531 |
| specificity | 1.0000 | 0.9469 |
| auroc | 0.9679 | 0.9679 |
| average_precision | 0.9720 | 0.9720 |

## 高斯/频域特征的安全与不安全分类

模型只使用四个图像特征：base_train 拟合，validation 选概率阈值，fusion_train 冻结迁移评估。它预测的是 HoliSafe 整体安全标签，不是图像对抗标签。

| frozen fusion_train 指标 | 数值 |
|---|---:|
| samples | 808 |
| accuracy | 0.8243 |
| balanced_accuracy | 0.5007 |
| precision | 0.8331 |
| recall | 0.9866 |
| f1 | 0.9034 |
| fpr | 0.9852 |
| auroc | 0.4831 |
| average_precision | 0.8356 |

### 两类特征差异（全部 3,415 条有标签样本）

| 特征 | safe mean | unsafe mean | unsafe-safe | Cohen's d | AUROC |
|---|---:|---:|---:|---:|---:|
| gaussian_residual_mae | 0.018089 | 0.017955 | -0.000134 | -0.0138 | 0.4863 |
| gaussian_residual_rmse | 0.037286 | 0.036617 | -0.000669 | -0.0406 | 0.4802 |
| high_frequency_ratio | 0.047827 | 0.046101 | -0.001727 | -0.0357 | 0.4772 |
| spectral_centroid | 0.053776 | 0.052278 | -0.001497 | -0.0520 | 0.4856 |

结论：这些低层视觉特征对整体安全标签的区分很弱，必须作为辅助证据，不能单独替代文字与融合模块。
