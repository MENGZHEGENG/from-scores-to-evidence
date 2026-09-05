# Reproduction Plan

This repository exposes the public code path for reproducing the core analyses after the required upstream inputs are prepared under their original licenses. The demo path uses synthetic scores only, so it checks software behavior rather than reproducing paper numbers. Paper-ready tables and figures are not bundled; scripts regenerate local outputs from the prepared inputs.

## Inputs

- `evidence_scores.csv`: one entry per utterance or segment with bona fide labels, synthesis-family labels, passive detector scores, conditional keyed-probe scores, retrieval scores, and speaker-profile margins.
- `retrieval_full_scores.csv`: optional full retrieval-score file for checking how the matched evidence-card subset differs from the larger retrieval score set.
- `support_features.csv` and `query_features.csv`: optional feature-level inputs for recomputing retrieval/profile scores and top-k overlap checks.
- `stress_check_scores.csv`: one entry per utterance or segment for each auxiliary stress check and score name.
- `ssl_attack_scores.csv`: one entry per utterance, detector, and signal-processing variant with bonafide labels and detector scores.

The expected schemas are listed in `docs/input_schemas.md` and enforced by `scripts/validate_inputs.py`. The ASVspoof 5 Track 1 development split policy is recorded in `configs/asvspoof5_track1_dev_splits.yaml` and applied by `scripts/apply_asvspoof5_splits.py`.

## Core Result Map

After preparing all three score-level input files in one directory, the complete public pipeline is:

```bash
python scripts/run_all.py --input-dir prepared_scores --out runs/main
```

The table below lists the same stages separately for targeted reruns and inspection.

| result | public command | primary output |
|---|---|---|
| ASVspoof 5 held-out-family assignment | `python scripts/apply_asvspoof5_splits.py --scores evidence_scores.csv --out runs/main/evidence_scores_with_folds.csv` | `evidence_scores_with_folds.csv` |
| retrieval/profile score construction | `python scripts/run_retrieval_profile.py --support support_features.csv --query query_features.csv --out runs/main/retrieval_profile.csv --audit runs/main/retrieval_profile_audit.json --k 10` | `retrieval_profile.csv`, `retrieval_profile_audit.json` |
| retrieval subset sensitivity | `python scripts/run_retrieval_selection_check.py --full-scores retrieval_full_scores.csv --matched-scores runs/main/evidence_scores_with_folds.csv --out runs/main` | `retrieval_selection_check.csv`, `retrieval_score_retention.csv`, `retrieval_selection_check.json` |
| matched evidence-card comparison | `python scripts/run_matched_evidence.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main` | `matched_evidence_metrics.csv`, `family_eer.csv`, `paired_bootstrap_deltas.csv`, `evidence_cards.csv` |
| calibration ablation with passive-shape, no-watermark passive + retrieval, one-conflict, squared-gap, and nonlinear controls | `python scripts/run_calibration_ablation.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main` | `calibration_ablation.csv` |
| calibration split-stability check | `python scripts/run_calibration_split_stability.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main` | `calibration_split_stability.csv`, `calibration_split_stability_runs.csv` |
| WavLM and self-VC stress checks | `python scripts/run_stress_checks.py --scores stress_check_scores.csv --out runs/main` | `stress_checks.csv` |
| active-selection check | `python scripts/run_active_selection.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main` | `active_selection.csv` |
| manual-review and selective-risk triage | `python scripts/run_evidence_card_triage.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main` | `evidence_card_triage.csv`, `selective_risk.csv` |
| SSL attack-stress summary | `python scripts/run_attack_stress.py --scores ssl_attack_scores.csv --out runs/main` | `attack_stress.csv` |
| editable figures | `python scripts/make_figures.py --results runs/main --out runs/main/figures` | `*.svg`, `*.pdf` |

## Recommended Checks

```bash
python -m pytest -q
python scripts/run_all.py --demo --out runs/demo
python scripts/run_retrieval_selection_check.py \
  --full-scores retrieval_full_scores.csv \
  --matched-scores runs/main/evidence_scores_with_folds.csv \
  --out runs/main
python scripts/run_calibration_split_stability.py \
  --scores runs/main/evidence_scores_with_folds.csv \
  --out runs/main
python scripts/validate_inputs.py \
  --evidence-scores runs/demo/inputs/evidence_scores.csv \
  --attack-scores runs/demo/inputs/ssl_attack_scores.csv \
  --stress-check-scores runs/demo/inputs/stress_check_scores.csv
python scripts/validate_repo.py --history
```

Use fixed sample identifiers, fixed family labels, fixed score orientation, and fixed random seeds when preparing real inputs. The evidence card is a calibration layer over score-level evidence, not a watermark generator or a replacement for the upstream detector. Keep raw speech, trained weights, and large precomputed score dumps outside this repository unless their licenses permit redistribution. See `docs/data_sources.md` for upstream data locations.

For shared per-example outputs, hash utterance, speaker, and neighbor identifiers and drop local audio paths with `scripts/redact_outputs_for_release.py`. Keep any salt used for stable hashing outside shared files.
