# Input Schemas

The public code path uses score-level CSV files so that licensed audio and large models stay outside the repository.

## `evidence_scores.csv`

Required columns:

| column | type | meaning |
|---|---|---|
| `sample_id` | string | stable utterance or segment identifier |
| `label` | integer | `1` for bona fide and `0` for spoof |
| `family` | string | held-out synthesis family or evaluation group |
| `passive_score` | float | passive bona fide score |
| `watermark_score` | float | keyed-probe consistency score |
| `retrieval_score` | float | retrieval support score |
| `profile_margin` | float | speaker-profile consistency margin |

All scores must be oriented so that larger values support the positive interpretation of that field.

## `support_features.csv` and `query_features.csv`

Required columns:

| column | type | meaning |
|---|---|---|
| `sample_id` | string | stable identifier |
| `label` | integer or string | bona fide or spoof label |
| `family` | string | synthesis family or bona fide group |
| `feat_*` | float | one or more numeric feature columns |

Optional columns such as `speaker_id`, `source`, and `source_path` are carried into the retrieval audit when present.

## `retrieval_full_scores.csv`

Required columns:

| column | type | meaning |
|---|---|---|
| `sample_id` | string | stable identifier |
| `label` | integer | bona fide/spoof label |
| `family` | string | evaluation family |
| `retrieval_score` | float | retrieval support score |

This file is used for the matched-subset sensitivity check.

## `stress_check_scores.csv`

Required columns:

| column | type | meaning |
|---|---|---|
| `check` | string | stress-check group |
| `score_name` | string | score within that group |
| `sample_id` | string | stable identifier |
| `label` | integer | positive-target label |
| `score` | float | score oriented toward the positive target |

## `ssl_attack_scores.csv`

Required columns:

| column | type | meaning |
|---|---|---|
| `sample_id` | string | stable identifier |
| `label` | integer | `1` for bona fide and `0` for spoof |
| `model` | string | detector name |
| `attack_family` | string | attack group |
| `attack_variant` | string | concrete attack setting |
| `score` | float | bona fide score |
