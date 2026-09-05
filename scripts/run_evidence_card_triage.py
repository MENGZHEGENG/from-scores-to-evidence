#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speech_evidence_cards.experiments import run_evidence_card_triage


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evidence-card manual-review triage metrics.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/main"))
    parser.add_argument("--budgets", type=float, nargs="*", default=[0.05, 0.10, 0.20])
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    args = parser.parse_args()
    outputs = run_evidence_card_triage(
        pd.read_csv(args.scores),
        args.out,
        budgets=tuple(args.budgets),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
