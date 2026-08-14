"""Survivorship-bias check on the frozen CSI300 membership file.

Why this exists
---------------
``verify_v2_data.py`` asserts "csi300 has 300 names on date D". That check
passes just as happily on a file that back-fills TODAY's constituents across
all of history — which is exactly the survivorship bias it is supposed to
rule out. A back-filled file also has 300 names on every date.

What actually distinguishes a point-in-time file:

1. Names whose membership ENDED before the snapshot end date. A back-filled
   file has none: every row would run to the last trading day.
2. Low overlap between the 2008 cross-section and the 2025 one. CSI300 is
   rebalanced twice a year, so ~17 years apart the overlap should be a
   minority of the index, not most of it.
3. Membership intervals that start after the index inception — i.e. names
   that joined later — and names appearing in several disjoint intervals
   (left the index, came back).

Usage::

    conda run -n quant python check_survivorship.py \
        --provider-uri /root/.qlib/qlib_data/cn_data_20260810

Exit code is non-zero if the file looks back-filled, so this can gate a
protocol freeze.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Set, Tuple

#: Below this 2008-vs-2025 overlap the file is clearly point-in-time.
#: CSI300 turns over roughly 10% of its names per semi-annual review, so after
#: ~34 reviews a genuine overlap lands well under half the index.
MAX_PLAUSIBLE_OVERLAP = 0.50


def parse_instruments(path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """symbol -> [(start, end), ...] from a qlib instruments file."""
    spans: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{lineno}: expected 'symbol start end', got {line!r}")
            symbol, start, end = parts[0], parts[1], parts[2]
            spans[symbol].append((start, end))
    return dict(spans)


def members_on(spans: Dict[str, List[Tuple[str, str]]], day: str) -> Set[str]:
    return {sym for sym, ivals in spans.items()
            if any(s <= day <= e for s, e in ivals)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider-uri", required=True)
    ap.add_argument("--market", default="csi300")
    ap.add_argument("--early", default="2008-06-02", help="early cross-section date")
    ap.add_argument("--late", default="2025-06-03", help="late cross-section date")
    args = ap.parse_args()

    root = Path(args.provider_uri).expanduser()
    inst_path = root / "instruments" / f"{args.market}.txt"
    if not inst_path.exists():
        print(f"FAIL: {inst_path} not found", file=sys.stderr)
        return 2

    spans = parse_instruments(inst_path)
    calendar = (root / "calendars" / "day.txt").read_text(encoding="utf-8").split()
    snapshot_end = calendar[-1].strip()

    total_rows = sum(len(v) for v in spans.values())
    print(f"file        : {inst_path}")
    print(f"snapshot end: {snapshot_end}")
    print(f"symbols     : {len(spans)}   intervals: {total_rows}")
    print()

    problems: List[str] = []

    # --- 1. names that left the index -----------------------------------
    ever_left = {
        sym for sym, ivals in spans.items()
        if max(e for _, e in ivals) < snapshot_end
    }
    print(f"[1] names whose membership ended before {snapshot_end}: {len(ever_left)}")
    if ever_left:
        sample = sorted(ever_left)[:8]
        print(f"    e.g. {', '.join(sample)}")
    else:
        problems.append(
            "no symbol ever leaves the index — the file looks back-filled "
            "from the current constituents"
        )

    # --- 2. cross-section overlap ---------------------------------------
    early = members_on(spans, args.early)
    late = members_on(spans, args.late)
    if not early or not late:
        problems.append(
            f"empty cross-section (early={len(early)}, late={len(late)}); "
            "check the dates against the calendar range"
        )
        overlap_ratio = float("nan")
    else:
        overlap = early & late
        overlap_ratio = len(overlap) / len(early)
        print(f"[2] {args.early}: {len(early)} names   "
              f"{args.late}: {len(late)} names")
        print(f"    overlap: {len(overlap)}  ({overlap_ratio:.1%} of the early set)")
        if overlap_ratio > MAX_PLAUSIBLE_OVERLAP:
            problems.append(
                f"{overlap_ratio:.1%} of the {args.early} constituents are still "
                f"in the index on {args.late}; a point-in-time CSI300 turns over "
                f"far more than that in 17 years"
            )

    # --- 3. structure ----------------------------------------------------
    rejoined = {sym: ivals for sym, ivals in spans.items() if len(ivals) > 1}
    first_day = calendar[0].strip()
    joined_later = {
        sym for sym, ivals in spans.items()
        if min(s for s, _ in ivals) > first_day
    }
    print(f"[3] symbols with >1 membership interval (left & rejoined): {len(rejoined)}")
    print(f"    symbols joining after {first_day}: {len(joined_later)}")
    if not rejoined:
        problems.append("no symbol has more than one membership interval")

    # --- verdict ---------------------------------------------------------
    print()
    if problems:
        print("VERDICT: SUSPECT — survivorship bias likely")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERDICT: point-in-time membership looks genuine")
    print("  constituents enter and leave; historical cross-sections differ "
          "substantially from today's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
