#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speech_evidence_cards.figures import make_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Build editable PDF/SVG figures from experiment outputs.")
    parser.add_argument("--results", type=Path, default=Path("runs/main"))
    parser.add_argument("--out", type=Path, default=Path("runs/main/figures"))
    args = parser.parse_args()
    make_all(args.results, args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
