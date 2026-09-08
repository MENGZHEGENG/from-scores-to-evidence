# Data Sources

This repository works from score-level inputs. It does not redistribute speech audio, pretrained weights, or full private result tables.

## Main evaluation inputs

Use the original providers for:

- **ASVspoof 5 Track 1 development data** and the accompanying challenge documentation;
- any pretrained detector used to generate the passive scores;
- any keyed-probe pipeline used to generate conditional watermark scores;
- any retrieval or speaker-profile features used for the auxiliary cues.

The split policy used in the paper is documented in `configs/asvspoof5_track1_dev_splits.yaml`.

## Auxiliary stress inputs

Use the original providers for any passive-detector stress benchmarks and signal-processing attack evaluations referenced in the paper.

## Local preparation rule

After obtaining the upstream data under the original licenses, prepare only the score-level CSV files consumed by the scripts in this repository. Validate those files with `python scripts/validate_inputs.py` before running experiments.
