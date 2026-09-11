# From Scores to Evidence

Reproducibility code for **From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection**.

This repository contains the code, configuration, and documentation needed to rerun the score-level analyses behind the paper. It is intentionally focused on reproducibility: the manuscript source, built PDFs, and other submission files are kept out of this repository.

## Repository layout

- `speech_evidence_cards/` — core Python implementation for calibration, retrieval scoring, cue aggregation, redaction, and plotting.
- `scripts/` — command-line entry points for the main experiments, demo run, and figure generation.
- `configs/` — split policy and default settings.
- `docs/` — upstream data notes, input schemas, and a result-to-command map.
- `tests/` — smoke tests for the public code path.

## What is not tracked here

This repository does not include:

- manuscript source or submission files;
- built paper PDFs, figure PDFs, or table exports;
- raw speech audio;
- pretrained model checkpoints;
- complete score dumps;
- machine-specific launch scripts or local cluster commands.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Demo run

The demo path uses small synthetic inputs so you can verify the public code path before preparing full inputs.

```bash
python scripts/run_all.py --demo --out runs/demo
python scripts/validate_inputs.py \
  --evidence-scores runs/demo/inputs/evidence_scores.csv \
  --attack-scores runs/demo/inputs/ssl_attack_scores.csv \
  --stress-check-scores runs/demo/inputs/stress_check_scores.csv
python scripts/validate_repo.py
```

If your environment is not ready to render figures yet, add `--skip-figures` to the demo command.

## Main reproduction path

Prepare a directory `prepared_scores/` with the score-level CSV files described in `docs/input_schemas.md`, then run:

```bash
python scripts/run_all.py --input-dir prepared_scores --out runs/main
```

This command runs the matched multi-cue comparison, calibration controls, split-stability checks, selective-review analysis, stress summaries, and figure generation.

## Stage-by-stage commands

### Matched comparison

```bash
python scripts/apply_asvspoof5_splits.py \
  --scores path/to/evidence_scores.csv \
  --out runs/main/evidence_scores_with_folds.csv
python scripts/validate_inputs.py --evidence-scores runs/main/evidence_scores_with_folds.csv
python scripts/run_matched_evidence.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main
```

### Calibration ablation

```bash
python scripts/run_calibration_ablation.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main
```

### Calibration split stability

```bash
python scripts/run_calibration_split_stability.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main
```

### Selective review / triage

```bash
python scripts/run_evidence_card_triage.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main
```

### Retrieval-profile scoring

```bash
python scripts/run_retrieval_profile.py \
  --support path/to/support_features.csv \
  --query path/to/query_features.csv \
  --out runs/main/retrieval_profile.csv \
  --audit runs/main/retrieval_profile_audit.json \
  --k 10
```

### Retrieval subset sensitivity

```bash
python scripts/run_retrieval_selection_check.py \
  --full-scores path/to/retrieval_full_scores.csv \
  --matched-scores runs/main/evidence_scores_with_folds.csv \
  --out runs/main
```

### Stress checks

```bash
python scripts/run_stress_checks.py --scores path/to/stress_check_scores.csv --out runs/main
python scripts/run_attack_stress.py --scores path/to/ssl_attack_scores.csv --out runs/main
```

### Figures

```bash
python scripts/make_figures.py --results runs/main --out runs/main/figures
```

## Public repo check

Before pushing changes to the public repository, run:

```bash
python scripts/validate_repo.py --clean --history --strict-local
```

This check looks for tracked manuscript files, generated outputs, local caches, and banned project-stage wording.

## Documentation

- `docs/data_sources.md` — upstream data sources and access notes.
- `docs/input_schemas.md` — required columns for each score-level input file.
- `docs/reproduction_plan.md` — map from paper results to public commands and outputs.

## Citation

Please cite the paper and repository. The paper citation is available directly as BibTeX:

```bibtex
@article{geng2026scores,
  title   = {From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection},
  author  = {Geng, Mengzhe and Lu, Yujia and Littell, Patrick and Kunz, Manuela and Chen, Xie},
  journal = {arXiv preprint arXiv:2609.08899},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.08899},
  doi     = {10.48550/arXiv.2609.08899}
}
```

See `CITATION.md` for the paper and repository links.
