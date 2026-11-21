# Task B publication-scope review

Status: **v2/week5 derived artifacts synced by explicit repository-update request; broader publication scope remains pending group-lead approval**.

Update 2026-11-22: `artifacts/v2/` and `artifacts/week5/` are intentionally included with their code, metrics, predictions, cached OCR/features/evidence and reproducibility records. Model weights, original restricted images and private dataset packages remain excluded.

No presence in Git history or in this working tree should be interpreted as permission to publish course data or derived per-sample content. Before the next public release, the group lead must decide whether each category may stay public, be shared only inside the team, or be removed from the public branch (including history if required).

| category | examples currently tracked | default until approved |
|---|---|---|
| per-sample OCR and text regions | `artifacts/ocr*.jsonl`, `artifacts/ocr_cache*/` | team-only |
| per-sample HoliSafe evidence/features | `artifacts/image_evidence*.jsonl`, `artifacts/features*.jsonl` | team-only |
| campus-derived outputs | files containing `campus12` | team-only |
| per-sample CIDER scores | `artifacts/cider*.jsonl` | team-only; aggregated reports may be public |
| grouped source mappings | `artifacts/manifest.inputs.grouped.jsonl` | team-only |
| aggregate metrics and source provenance | `team_b/reports/`, `team_b/provenance/` | proposed public |
| implementation and tests | `team_b/*.py`, `team_b/scripts/`, `team_b/tests/` | proposed public |

The root `.gitignore` now blocks accidental addition of new copies in the pending categories. It does not remove already tracked files or rewrite Git history. After the group lead decides, record the decision, approver and date here, then perform any untracking/history cleanup as a separate reviewed step.
