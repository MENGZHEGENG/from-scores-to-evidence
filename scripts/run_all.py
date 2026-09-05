#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speech_evidence_cards.experiments import (
    make_demo_scores,
    run_active_selection,
    run_attack_stress,
    run_calibration,
    run_calibration_split_stability,
    run_evidence_card_triage,
    run_matched_evidence,
    run_stress_checks,
)
from speech_evidence_cards.row_selection import run_row_selection_check


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public evidence-card reproduction pipeline.")
    parser.add_argument("--demo", action="store_true", help="Generate deterministic synthetic inputs before running.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing evidence_scores.csv, ssl_attack_scores.csv, and stress_check_scores.csv. In --demo mode, defaults to <out>/inputs.",
    )
    parser.add_argument(
        "--full-retrieval-scores",
        type=Path,
        default=None,
        help="Optional full retrieval-score CSV for row-selection sensitivity checks.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/demo"), help="Directory for result CSVs and figures.")
    parser.add_argument("--skip-figures", action="store_true", help="Run metrics only and skip SVG/PDF figure generation.")
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir is not None else args.out / "inputs"
    if args.demo:
        input_dir.mkdir(parents=True, exist_ok=True)
        evidence, attacks, stress = make_demo_scores()
        evidence.to_csv(input_dir / "evidence_scores.csv", index=False)
        attacks.to_csv(input_dir / "ssl_attack_scores.csv", index=False)
        stress.to_csv(input_dir / "stress_check_scores.csv", index=False)

    evidence_scores = pd.read_csv(input_dir / "evidence_scores.csv")
    attack_scores = pd.read_csv(input_dir / "ssl_attack_scores.csv")
    stress_scores = pd.read_csv(input_dir / "stress_check_scores.csv")
    args.out.mkdir(parents=True, exist_ok=True)
    run_matched_evidence(evidence_scores, args.out)
    run_calibration(evidence_scores, args.out)
    run_calibration_split_stability(evidence_scores, args.out)
    run_active_selection(evidence_scores, args.out)
    run_evidence_card_triage(evidence_scores, args.out)
    run_attack_stress(attack_scores, args.out)
    run_stress_checks(stress_scores, args.out)
    full_retrieval_scores = args.full_retrieval_scores or input_dir / "retrieval_full_scores.csv"
    if full_retrieval_scores.exists():
        run_row_selection_check(pd.read_csv(full_retrieval_scores), evidence_scores, args.out)
    if not args.skip_figures:
        from speech_evidence_cards.figures import make_all

        make_all(args.out, args.out / "figures")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
