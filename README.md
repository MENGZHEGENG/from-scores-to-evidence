# From Scores to Evidence: Cue-Preserving Calibration for Speech Deepfake Detection

This repository contains the public implementation for From Scores to Evidence, a cue-preserving calibration pipeline for speech deepfake detection and review. It provides scripts, configuration, and method code for reproducing the core analyses once the required speech corpora, detector scores, retrieval features, and keyed-probe scores are prepared under their original licenses. Paper tables and editable vector figures are regenerated locally by the commands below rather than committed as static outputs.

## What is included

- `speech_evidence_cards/`: reusable Python implementation for metrics, evidence-card construction, retrieval scoring, watermark probing, calibration, and plotting.
- `scripts/`: command-line entry points for the main experiments and editable visualizations.
- `scripts/redact_outputs_for_release.py`: helper for hashing example identifiers and removing local audio paths before sharing derived CSV outputs.
- `configs/main.yaml`: default schema and experiment settings.
- `configs/asvspoof5_track1_dev_splits.yaml`: split policy for the ASVspoof 5 Track 1 development matched evaluation.
- `docs/data_sources.md`: upstream data locations and licensing notes.
- `docs/reproduction_plan.md`: mapping from core results to public commands and expected outputs.
- `tests/`: smoke tests for the public release package.
- `CITATION.md`: citation status; final citation metadata is intentionally withheld until the author list and public preprint record are approved.
- `LICENSE`: current reuse status; replace with the approved open-source license before public release.

## Release Scope

This repository does not include:

- Raw speech audio.
- Model checkpoints or large pretrained weights.
- Precomputed complete-result tables.
- Paper-ready figure or table files.
- Machine-specific launch files or queue-system submission scripts.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_all.py --demo --out runs/demo
python scripts/validate_inputs.py \
  --evidence-scores runs/demo/inputs/evidence_scores.csv \
  --attack-scores runs/demo/inputs/ssl_attack_scores.csv \
  --stress-check-scores runs/demo/inputs/stress_check_scores.csv
python scripts/validate_repo.py
```

The demo command creates a small synthetic input set under `runs/demo/inputs/`, runs the same public scripts used for the main analysis, and writes outputs under `runs/demo/`. The demo checks software behavior; it is not a substitute for the paper experiments, and generated outputs are left uncommitted by default.
If your environment does not have a working Matplotlib backend yet, add `--skip-figures` to check the metric pipeline first.

Before changing the repository visibility, also run:

```bash
python scripts/validate_repo.py --clean --history --strict-local
```

This stricter check removes ignored generated outputs and cache directories, then scans previous Git commits and the local checkout before public release.

## Reproducing the main experiments

Prepare input CSV files with the schemas below, using `docs/data_sources.md` and `configs/asvspoof5_track1_dev_splits.yaml` to keep upstream data access and split construction explicit. Then run the scripts in order.

For a prepared directory containing `evidence_scores.csv`, `ssl_attack_scores.csv`, and `stress_check_scores.csv`, the end-to-end public command is:

```bash
python scripts/run_all.py --input-dir prepared_scores --out runs/main
```

This command runs the matched multi-cue evaluation, calibration controls, calibration split-stability check, selective-risk analysis, stress summaries, and editable figure generation. The separate commands below expose the same stages when a result needs to be rerun or audited independently.
If `retrieval_full_scores.csv` is also present in the prepared input directory, `run_all.py` additionally writes the retrieval subset-sensitivity analysis used by the appendix.

### Matched multi-cue comparison

Expected file: `evidence_scores.csv`

Required columns:

| column | meaning |
|---|---|
| `sample_id` | stable utterance or segment identifier |
| `label` | `1` for the positive class and `0` for the negative class; in the paper this is bona fide versus spoof |
| `family` | held-out synthesis family or evaluation group |
| `passive_score` | passive detector bonafide score; larger means more bonafide |
| `watermark_score` | conditional keyed-probe score; larger means more mark-consistent under the chosen key |
| `retrieval_score` | neighbor-label evidence score; larger means more bonafide |
| `profile_margin` | speaker-profile consistency margin; larger means more consistent |

Commands:

```bash
python scripts/apply_asvspoof5_splits.py \
  --scores path/to/evidence_scores.csv \
  --out runs/main/evidence_scores_with_folds.csv
python scripts/validate_inputs.py --evidence-scores runs/main/evidence_scores_with_folds.csv
python scripts/run_matched_evidence.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main
python scripts/run_calibration_ablation.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main
python scripts/run_calibration_split_stability.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main
python scripts/run_active_selection.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main
python scripts/run_evidence_card_triage.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main
```

The split command adds `held_out_family` and `split_role` columns. Spoof examples keep their A09--A16 family as the held-out fold; bona fide examples are assigned to the same eight folds by a stable hash of `sample_id`. The keyed-probe score is an evidence field, not proof that an original file carried a mark.

The calibration output contains eleven leave-family-out settings: base stream fusion, passive-only shape controls, retrieval/profile controls that omit watermark fields, passive + retrieval/profile controls with and without passive shape terms, two one-conflict controls, a squared-gap disagreement control, a nonlinear control with score-interaction products, and the evidence-card model with both absolute disagreement fields. The split-stability command repeats the main learned calibrators under alternate bona fide fold assignments while keeping spoof folds fixed. The triage command writes two files: `evidence_card_triage.csv`, which ranks fixed-threshold card errors under small review budgets, and `selective_risk.csv`, which applies each system's own EER threshold before deferring decisions closest to that threshold in score-rank space.

Before sharing per-example outputs, redact identifiers and local paths:

```bash
python scripts/redact_outputs_for_release.py \
  --input runs/main/evidence_cards.csv \
  --output runs/share/evidence_cards_redacted.csv \
  --salt "$PRIVATE_REDACTION_SALT" \
  --summary runs/share/redaction_summary.json
```

The salt is optional for a one-off public file, but a non-public fixed salt keeps hashed identifiers consistent across several files without exposing original utterance, speaker, neighbor, or path values.

### Retrieval/profile scoring

Expected files: `support_features.csv` and `query_features.csv`

Each file must include `sample_id`, `label`, `family`, and one or more numeric `feat_*` columns. Optional `speaker_id`, `source_path`, and `source` columns are carried into the top-k overlap audit.

Command:

```bash
python scripts/run_retrieval_profile.py \
  --support path/to/support_features.csv \
  --query path/to/query_features.csv \
  --out runs/main/retrieval_profile.csv \
  --audit runs/main/retrieval_profile_audit.json \
  --k 10
```

For spoof queries, support examples from the same synthesis family are removed before nearest-neighbor search. The output includes retrieval scores, nearest-neighbor metadata, top-k neighbor identifiers, and an overlap audit for held-out family, sample, speaker, and audio-path reuse.

### Retrieval Subset Sensitivity

Expected files: a full retrieval-score CSV and the matched evidence-card CSV. Both files must include `sample_id`, `label`, `family`, and `retrieval_score`, with larger retrieval scores indicating bona fide evidence.

Command:

```bash
python scripts/run_retrieval_selection_check.py \
  --full-scores path/to/retrieval_full_scores.csv \
  --matched-scores runs/main/evidence_scores_with_folds.csv \
  --out runs/main
```

The check compares the matched subset with same-size random draws from the full retrieval-score file. It summarizes family-level EER shifts, retained-versus-omitted spoof-score shifts, and KS distances so the matched analysis is separated from full-score prevalence.

### Stress checks

Expected file: `stress_check_scores.csv`

Required columns:

| column | meaning |
|---|---|
| `check` | check group, such as `WavLM check` or `Self-VC` |
| `score_name` | score being evaluated within that group |
| `sample_id` | stable utterance or segment identifier |
| `label` | `1` for the positive target of that score and `0` for the negative target |
| `score` | score oriented so larger values support the positive target |

Command:

```bash
python scripts/validate_inputs.py --stress-check-scores path/to/stress_check_scores.csv
python scripts/run_stress_checks.py --scores path/to/stress_check_scores.csv --out runs/main
```

### SSL attack-stress experiment

Expected file: `ssl_attack_scores.csv`

Required columns:

| column | meaning |
|---|---|
| `sample_id` | stable utterance or segment identifier |
| `label` | `1` for bonafide and `0` for spoof |
| `model` | frozen SSL detector name |
| `attack_family` | signal-processing family, e.g. noise or high-pass |
| `attack_variant` | concrete variant, e.g. 0 dB or 2.5 kHz |
| `score` | model bonafide score |

Command:

```bash
python scripts/validate_inputs.py --attack-scores path/to/ssl_attack_scores.csv
python scripts/run_attack_stress.py --scores path/to/ssl_attack_scores.csv --out runs/main
```

### Figures

```bash
python scripts/make_figures.py --results runs/main --out runs/main/figures
```

The figure script writes editable SVG and PDF files to the requested output directory. These generated files are for local inspection or inclusion in a manuscript build and are left out of version control by default.

## Notes on reproducibility

The scripts assume that upstream datasets and pretrained models are acquired separately from their original providers. Keep input preparation deterministic: fixed sample IDs, fixed family labels, fixed score orientation, and fixed random seeds for bootstrap and demo runs.
