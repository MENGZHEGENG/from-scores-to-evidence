#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speech_evidence_cards.redaction import redact_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hash identifiers and remove local paths from shareable CSV outputs.")
    parser.add_argument("--input", required=True, type=Path, help="CSV produced by the public experiment scripts")
    parser.add_argument("--output", required=True, type=Path, help="redacted CSV destination")
    parser.add_argument(
        "--salt",
        default="",
        help="optional private salt for stable hashes across files; keep the salt outside shared material",
    )
    parser.add_argument("--summary", "--report", dest="report", type=Path, help="optional JSON summary destination")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = redact_csv(args.input, args.output, salt=args.salt)
    payload = report.to_dict()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
