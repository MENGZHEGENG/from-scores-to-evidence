# Data Sources

This repository does not redistribute speech audio, trained weights, or large score tables. Prepare those inputs from the original providers, then convert the resulting scores to the CSV schemas in `docs/input_schemas.md`.

## Main Matched Evaluation

- ASVspoof 5: use the official ASVspoof site and the ASVspoof 5 Zenodo record for the Track 1 development data and challenge documentation.
- Split policy: use `configs/asvspoof5_track1_dev_splits.yaml` with `scripts/apply_asvspoof5_splits.py`; the held-out synthesis families are A09 through A16, with bona fide examples assigned to the same eight folds by a stable `sample_id` hash.

## Auxiliary Stress Checks

- ASVspoof 2021 DF: use the official ASVspoof 2021 page and the ASVspoof challenge baseline/evaluation repository for keys, metadata, and evaluation conventions.
- In-The-Wild: use the provider dataset page for the published real-world audio deepfake benchmark.
- WaveFake: use the WaveFake project repository and Zenodo release for generated-audio data and accompanying code.

## Expected Local Inputs

After obtaining data under the applicable licenses, create only the score-level files consumed by the scripts:

- `evidence_scores.csv` for matched evidence-card evaluation.
- `stress_check_scores.csv` for passive-detector and watermark-probe stress checks.
- `ssl_attack_scores.csv` for signal-processing stress summaries.

Run `python scripts/validate_inputs.py` before any experiment command.
