# Reproduction Plan

This file maps the main reported results to public commands.

## 1. Prepare score-level inputs

- Build the CSV files described in `docs/input_schemas.md`.
- Use `configs/asvspoof5_track1_dev_splits.yaml` for the matched ASVspoof 5 split policy.
- Validate the prepared files with `python scripts/validate_inputs.py`.

## 2. Run the main pipeline

```bash
python scripts/run_all.py --input-dir prepared_scores --out runs/main
```

This writes the matched comparison, calibration ablation, split-stability checks, selective-review summaries, stress summaries, and figures.

## 3. Re-run individual components when needed

- Matched comparison: `python scripts/run_matched_evidence.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main`
- Calibration ablation: `python scripts/run_calibration_ablation.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main`
- Split stability: `python scripts/run_calibration_split_stability.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main`
- Selective review: `python scripts/run_evidence_card_triage.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main`
- Retrieval subset sensitivity: `python scripts/run_retrieval_selection_check.py --full-scores path/to/retrieval_full_scores.csv --matched-scores runs/main/evidence_scores_with_folds.csv --out runs/main`
- SSL attack stress: `python scripts/run_attack_stress.py --scores ssl_attack_scores.csv --out runs/main`
- Figures: `python scripts/make_figures.py --results runs/main --out runs/main/figures`

## 4. Public-tree check

```bash
python scripts/validate_repo.py --clean --history --strict-local
```

Run this before pushing to make sure manuscript files, generated outputs, cache directories, and banned wording are not present in the public repository.
