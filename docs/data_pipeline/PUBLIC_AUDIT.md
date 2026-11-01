# Public data audit and HoliSafe course package

Audit run: 2026-11-01, offline, using local downloads only. The audit tool does not download data or infer labels.

## HoliSafe-Bench

- Pinned dataset revision: `b6892a0efadd275026bc903953158a9943d76028`.
- 4,031 source records, 1,796 referenced image files; all image paths exist and decode. Download metadata checks passed; no incomplete files were found.
- Seven exact duplicate image-content + query records were collapsed, leaving 4,024 inputs. All duplicate groups had the same final-input-risk label.
- The 1,796 image files contain 1,787 unique byte/pixel hashes. Five paths (18 source records) have conflicting `image_safe` / `image_safety_label` annotations across records; those image-safety values are preserved as source data and not mapped to `image_adv_label`.
- Official `type` counts before exact-input deduplication: SSS 682, SUU 899, USU 971, UUU 761, SSU 718. In the 4,024-input canonical cohort, overall-safe=680 and overall-unsafe=3,344.
- The course split is derived from the source benchmark's official test data; it is not the official benchmark protocol. Related image/request groups are co-allocated using exact pixel identity, dHash distance <=4, exact normalized query text, and token-set Jaccard >=0.85 for queries with at least six distinct tokens. These rules are conservative grouping heuristics, not human semantic review.
- 992 connected groups were assigned with approximate 50/20/15/15 ratios and stratification by overall risk, the five official `type` combinations, and seven categories:

| Split | Samples | Overall safe | Overall unsafe |
| --- | ---: | ---: | ---: |
| base_train | 2,002 | 341 | 1,661 |
| fusion_train | 808 | 135 | 673 |
| val | 605 | 101 | 504 |
| test | 609 | 103 | 506 |

- Mapped fields: `harmful_content_label` from official query-safety character; `overall_risk_label` from official final-input-safety character. `image_adv_label`, `jailbreak_intent_label`, and `prompt_injection_label` remain unknown and blank.

## JailBreakV-28K

- Pinned dataset revision: `f949ca582fff13d396ac8fce59596afafb2b78d3`. The full CSV has 28,000 unique IDs; the 280-row mini CSV is an exact subset, not additional data.
- CSV references 11,751 distinct images, but only 300 local images exist and decode; just 360 full-table rows currently have images. The mini CSV has only 3 rows with images. The other 27,640 full rows are unusable as image-text samples until the official image package is obtained.
- This source is excluded from the current course package. It does not provide per-sample MLLM jailbreak-success labels.

## Delivery checks

- `work/qc_report.txt`: 0 errors, 0 warnings.
- Per-split type/category counts and connected-group counts are in `split_statistics.csv`.
- The team ZIP has 4,024 inputs, 1,787 bundled images, relative image paths, and source restrictions. Test labels and test provenance are kept in `output/A_private_DO_NOT_SHARE/`.
- ZIP CRC check passed. Every input image path resolves in the archive. No test IDs were found in shared label CSVs or `provenance_train_validation.csv`; test manifests have no `label` field.
- This cohort supports a course-derived overall multimodal safety-risk experiment. It does not by itself support claims about jailbreak-intent, prompt-injection, or adversarial-image detection. No model predictions or performance metrics are included.

## Remaining integration item

D's current fit interface accepts `validation`, while the group plan assigns `fusion_train` for fitting and `val` for threshold selection. Keep the split names and adapt D's fit interface before consuming the fusion manifest. Do not rename `fusion_train` to bypass the mismatch.
