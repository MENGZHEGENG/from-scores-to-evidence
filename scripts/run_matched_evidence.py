#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speech_evidence_cards.experiments import run_matched_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched held-out-family evidence-card metrics.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/main"))
    args = parser.parse_args()
    outputs = run_matched_evidence(pd.read_csv(args.scores), args.out)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
