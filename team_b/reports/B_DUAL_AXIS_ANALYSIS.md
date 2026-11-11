# Task B dual-axis label analysis

The two truth definitions are analyzed separately and are never pooled.

## Image-adversarial axis (CIDER official data)

Threshold `0.00568023` with strict `>`; observed clean pass rate 95.12% on 410 clean pairs.

| split | n | AUROC | AP | accuracy | recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| validation | 328 | 0.9378 | 0.9375 | 0.7104 | 0.4268 | 0.0061 |
| test | 1280 | 0.9679 | 0.9720 | 0.7000 | 0.4000 | 0.0000 |

Important limitation: the four adversarial images in the official validation and test folders have identical bytes; only the HarmBench requests differ. The test row is therefore request-holdout transfer performance, not image-holdout generalization. Clean images do not overlap. Do not combine these splits into one D handoff manifest or describe 1,280 pairs as 1,280 independent images.

## Overall-risk axis (HoliSafe)

Exploratory comparison on 3415 labeled pairs (2838 unsafe, 577 safe).

| signal | safe mean | unsafe mean | raw AUROC |
|---|---:|---:|---:|
| cider_statistic | 0.001328 | 0.001559 | 0.5395 |
| gaussian_residual_mae | 0.018089 | 0.017955 | 0.4863 |
| gaussian_residual_rmse | 0.037286 | 0.036617 | 0.4802 |
| high_frequency_ratio | 0.047827 | 0.046101 | 0.4772 |
| spectral_centroid | 0.053776 | 0.052278 | 0.4856 |

HoliSafe results are broad safety associations, not adversarial-image accuracy.
