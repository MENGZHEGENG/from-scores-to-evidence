#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from speech_evidence_cards.retrieval import (
    audit_retrieval_neighbors,
    retrieval_profile_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build held-out-family retrieval/profile scores from feature CSV files.")
    parser.add_argument("--support", type=Path, required=True, help="CSV file with support rows and feat_* columns")
    parser.add_argument("--query", type=Path, required=True, help="CSV file with query rows and feat_* columns")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path for retrieval/profile rows")
    parser.add_argument("--audit", type=Path, help="Optional JSON path for top-k overlap checks")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    support = pd.read_csv(args.support)
    query = pd.read_csv(args.query)
    output = retrieval_profile_scores(support, query, k=args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    audit = audit_retrieval_neighbors(output)
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output), "ready": audit.ready, "issues": list(audit.issues)}, indent=2))
    return 0 if audit.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
