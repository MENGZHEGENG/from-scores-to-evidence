#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speech_evidence_cards.row_selection import run_row_selection_check


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a matched evidence-card subset against the full retrieval-score file."
    )
    parser.add_argument("--full-scores", type=Path, required=True, help="CSV with full retrieval-score rows")
    parser.add_argument("--matched-scores", type=Path, required=True, help="CSV with matched evidence-card rows")
    parser.add_argument("--out", type=Path, required=True, help="Directory for check outputs")
    parser.add_argument("--score-column", default="retrieval_score", help="Score column to compare")
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Matched-size random draws per family")
    parser.add_argument("--seed", type=int, default=20260830, help="Random seed for bootstrap draws")
    parser.add_argument("--json", action="store_true", help="Print output paths as JSON")
    args = parser.parse_args()

    outputs = run_row_selection_check(
        pd.read_csv(args.full_scores),
        pd.read_csv(args.matched_scores),
        args.out,
        score_column=args.score_column,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(outputs, indent=2))
    else:
        for name, path in outputs.items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
