#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speech_evidence_cards.inputs import (
    read_csv,
    validate_attack_scores,
    validate_evidence_scores,
    validate_stress_check_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate score-level input files before running experiments.")
    parser.add_argument("--evidence-scores", type=Path, help="Path to evidence_scores.csv")
    parser.add_argument("--attack-scores", type=Path, help="Path to ssl_attack_scores.csv")
    parser.add_argument("--stress-check-scores", type=Path, help="Path to stress_check_scores.csv")
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation output.")
    args = parser.parse_args()
    if args.evidence_scores is None and args.attack_scores is None and args.stress_check_scores is None:
        parser.error("provide --evidence-scores, --attack-scores, --stress-check-scores, or a combination")

    reports = []
    if args.evidence_scores is not None:
        reports.append(validate_evidence_scores(read_csv(args.evidence_scores)))
    if args.attack_scores is not None:
        reports.append(validate_attack_scores(read_csv(args.attack_scores)))
    if args.stress_check_scores is not None:
        reports.append(validate_stress_check_scores(read_csv(args.stress_check_scores)))

    if args.json:
        print(json.dumps({"ready": all(report.ready for report in reports), "reports": [report.to_dict() for report in reports]}, indent=2))
    else:
        for report in reports:
            if report.ready:
                print(f"{report.name}: validation passed ({report.row_count} rows)")
            else:
                print(f"{report.name}: validation failed ({report.row_count} rows)")
                for issue in report.issues:
                    print(f"- {issue}")
    return 0 if all(report.ready for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
