#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speech_evidence_cards.experiments import DEFAULT_STABILITY_SEEDS, run_calibration_split_stability


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeat calibration with alternate bona fide fold assignments."
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/main"))
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args()
    outputs = run_calibration_split_stability(
        pd.read_csv(args.scores),
        args.out,
        seeds=tuple(args.seeds) if args.seeds else DEFAULT_STABILITY_SEEDS,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
