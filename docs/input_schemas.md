# Input Schemas

The public scripts use score-level inputs so that licensed audio and model weights do not need to be redistributed. Keep input preparation deterministic: stable sample identifiers, fixed family labels, fixed score orientation, and fixed random seeds for any upstream model scoring.

## Evidence Scores

Use `evidence_scores.csv` for the matched-comparison, calibration, and active-selection scripts. Required columns:

| column | type | requirement |
|---|---|---|
| `sample_id` | string | non-empty and unique within the file |
| `label` | integer | `1` for the positive class and `0` for the negative class; in the paper this is bona fide versus spoof, and both classes must be present |
| `family` | string | non-empty held-out synthesis family or evaluation group; at least two groups are required |
| `passive_score` | float | finite passive-detector bonafide score |
| `watermark_score` | float | finite conditional keyed-probe score |
| `retrieval_score` | float | finite neighbor-label evidence score |
| `profile_margin` | float | finite speaker-profile consistency margin |

All four evidence scores must be oriented so that larger values support the corresponding positive interpretation. For `watermark_score`, that interpretation is mark consistency under the chosen key, not proof that an original file carried a mark.
The triage script also uses an optional `nearest_distance` column when available; larger values should mean the nearest retrieved neighbor is farther away.

For the ASVspoof 5 Track 1 development setting, run `scripts/apply_asvspoof5_splits.py` after preparing this file. The command preserves the original columns and adds `held_out_family` plus `split_role`; downstream scripts ignore extra columns they do not need.

## Retrieval Feature Scores

Use `support_features.csv` and `query_features.csv` with `scripts/run_retrieval_profile.py` when the retrieval score is recomputed rather than provided. Required columns:

| column | type | requirement |
|---|---|---|
| `sample_id` | string | non-empty utterance or segment identifier |
| `label` | integer or string | `1`/`bonafide` for bona fide and `0`/`spoof` for spoof |
| `family` | string | synthesis family or `bonafide` group label |
| `feat_*` | float | one or more finite feature columns shared by support and query files |

Optional columns `speaker_id`, `source_path`, and `source` are copied into the output. When present, the audit states whether any top-k neighbor shares a query sample identifier, held-out spoof family, speaker identifier, or audio path.

If per-example outputs will be shared, run `scripts/redact_outputs_for_release.py` first. The helper hashes utterance, neighbor, and speaker identifiers and removes local audio-path columns while keeping labels, groups, scores, and metrics reproducible.

## Retrieval Subset-Selection Scores

Use `retrieval_full_scores.csv` with `scripts/run_retrieval_selection_check.py` when auditing how a matched evidence-card subset differs from the larger retrieval-score file. The matched file may be `evidence_scores.csv` or `runs/main/evidence_scores_with_folds.csv`.

Required columns for both files:

| column | type | requirement |
|---|---|---|
| `sample_id` | string | non-empty utterance or segment identifier; matched identifiers must appear in the full file |
| `label` | integer | `1` for bona fide and `0` for spoof |
| `family` | string | synthesis family or evaluation group |
| `retrieval_score` | float | finite neighbor-label evidence score; larger means more bona fide |

The command writes a family-level EER comparison, an overall retained-versus-omitted score table, and a compact JSON audit. It does not change labels or resample the matched subset; random draws are used only as a matched-size reference.

## Stress Check Scores

Use `stress_check_scores.csv` for auxiliary checks such as stronger passive streams or self-voice-conversion keyed probes. Required columns:

| column | type | requirement |
|---|---|---|
| `check` | string | non-empty check group name |
| `score_name` | string | non-empty score name within the group |
| `sample_id` | string | non-empty utterance or segment identifier |
| `label` | integer | `1` for the positive target of that score and `0` for the negative target; both classes must be present per group |
| `score` | float | finite score oriented so larger values support the positive target |

Entries must be unique for each `(check, score_name, sample_id)` tuple. Each `(check, score_name)` group must contain both labels so that EER, minDCF, and calibration summaries are defined.

## SSL Attack Scores

Use `ssl_attack_scores.csv` for the signal-processing stress script. Required columns:

| column | type | requirement |
|---|---|---|
| `sample_id` | string | non-empty utterance or segment identifier |
| `label` | integer | `1` for bonafide, `0` for spoof; both classes must be present |
| `model` | string | non-empty frozen detector name |
| `attack_family` | string | non-empty signal-processing family, such as noise or high-pass |
| `attack_variant` | string | non-empty concrete variant, such as 0 dB or 2.5 kHz |
| `score` | float | finite bonafide score |

Entries must be unique for each `(sample_id, model, attack_family, attack_variant)` tuple. Each model/family/variant group must contain both labels so that EER, minDCF, and calibration summaries are defined.

## Preflight Check

Run the input validator before launching the experiment scripts:

```bash
python scripts/validate_inputs.py --evidence-scores path/to/evidence_scores.csv --attack-scores path/to/ssl_attack_scores.csv
```

Use `--json` when integrating the check into a larger workflow.
