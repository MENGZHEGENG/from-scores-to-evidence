#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speech_evidence_cards.experiments import make_demo_scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic demo inputs for the public evidence-card scripts.")
    parser.add_argument("--out", type=Path, default=Path("runs/demo/inputs"))
    parser.add_argument("--n", type=int, default=2400)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evidence, attacks, stress = make_demo_scores(args.n)
    evidence.to_csv(args.out / "evidence_scores.csv", index=False)
    attacks.to_csv(args.out / "ssl_attack_scores.csv", index=False)
    stress.to_csv(args.out / "stress_check_scores.csv", index=False)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
