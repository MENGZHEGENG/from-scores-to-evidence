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

from speech_evidence_cards.splits import (
    assign_asvspoof5_track1_folds,
    load_split_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Add deterministic ASVspoof 5 held-out-family columns to an evidence score file.")
    parser.add_argument("--scores", type=Path, required=True, help="Path to evidence_scores.csv")
    parser.add_argument("--out", type=Path, required=True, help="Destination for the score file with split columns")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "asvspoof5_track1_dev_splits.yaml",
        help="Split-policy YAML file",
    )
    parser.add_argument(
        "--allow-other-families",
        action="store_true",
        help="Permit spoof-family labels outside the configured held-out set",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    args = parser.parse_args()

    policy = load_split_policy(args.config)
    scores = pd.read_csv(args.scores)
    assigned = assign_asvspoof5_track1_folds(
        scores,
        policy=policy,
        allow_other_families=args.allow_other_families,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(args.out, index=False)

    counts = assigned["held_out_family"].value_counts().sort_index().to_dict()
    payload = {
        "input": str(args.scores),
        "output": str(args.out),
        "rows": len(assigned),
        "held_out_families": list(policy.held_out_families),
        "assigned_counts": {str(key): int(value) for key, value in counts.items()},
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"wrote {args.out} ({payload['rows']} rows)")
        for family, count in payload["assigned_counts"].items():
            print(f"{family}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
