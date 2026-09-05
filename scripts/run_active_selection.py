#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speech_evidence_cards.experiments import run_active_selection


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evidence-card active-selection checks.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/main"))
    parser.add_argument("--budgets", type=int, nargs="*", default=[100, 500, 1000, 2000])
    args = parser.parse_args()
    outputs = run_active_selection(pd.read_csv(args.scores), args.out, budgets=args.budgets)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
