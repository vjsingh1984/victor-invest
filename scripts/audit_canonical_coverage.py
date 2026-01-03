#!/usr/bin/env python3
"""
Audit canonical key coverage across cached SEC CompanyFacts payloads.

Usage:
    python3 scripts/audit_canonical_coverage.py \
        --sector-map data/sector_industry_ticker_map.txt \
        --facts-root data/sec_cache/facts/raw \
        --output coverage_report.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.canonical_key_mapper import get_canonical_mapper  # noqa: E402


def has_raw_support(
    canonical_key: str,
    mapper,
    sector: Optional[str],
    us_gaap: Dict,
    memo: Dict[Tuple[str, Optional[str]], bool],
    seen: Optional[set] = None,
) -> bool:
    """
    Determine if a canonical key can be satisfied directly from raw tags (or via derived dependencies).
    """
    cache_key = (canonical_key, sector)
    if cache_key in memo:
        return memo[cache_key]

    if canonical_key == "current_price":
        memo[cache_key] = True
        return True

    mapping = mapper.mappings.get(canonical_key)
    if not mapping:
        memo[cache_key] = False
        return False

    tags = mapper.get_tags(canonical_key, sector)
    if any(tag in us_gaap for tag in tags):
        memo[cache_key] = True
        return True

    derived_meta = mapping.get("derived", {})
    if not derived_meta.get("enabled"):
        memo[cache_key] = False
        return False

    required_fields = derived_meta.get("required_fields") or []
    if not required_fields:
        memo[cache_key] = False
        return False

    seen = seen or set()
    if canonical_key in seen:
        memo[cache_key] = False
        return False

    next_seen = seen | {canonical_key}
    result = all(has_raw_support(dep, mapper, sector, us_gaap, memo, next_seen) for dep in required_fields)
    memo[cache_key] = result
    return result


def parse_sector_map(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """Parse the pipe-delimited sector map exported from the database."""
    mapping: Dict[str, Dict[str, Optional[str]]] = {}
    if not path.exists():
        return mapping

    with path.open() as handle:
        for line in handle:
            if "|" not in line:
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 5:
                continue
            ticker = parts[0]
            if not ticker or ticker.lower() == "ticker":
                continue

            sec_sector, alt_sector, sec_industry, alt_industry = parts[1:5]
            sector_value = sec_sector or alt_sector or "Unknown"
            industry_value = sec_industry or alt_industry or None

            mapping[ticker.upper()] = {
                "sector": sector_value.strip(),
                "industry": industry_value.strip() if industry_value else None,
            }

    return mapping


def latest_snapshot(symbol_dir: Path) -> Optional[Path]:
    """Return the newest gz snapshot inside a symbol directory."""
    files = sorted(symbol_dir.glob("*.json.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_us_gaap(snapshot: Path) -> Optional[Dict]:
    """Load the us-gaap node from a cached raw snapshot."""
    try:
        with gzip.open(snapshot, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("facts", {}).get("us-gaap")


def audit_coverage(facts_root: Path, sector_map: Dict[str, Dict[str, Optional[str]]], limit: Optional[int]):
    mapper = get_canonical_mapper()
    coverage = defaultdict(lambda: defaultdict(lambda: {"present": 0, "total": 0}))
    symbol_counter = 0

    for symbol_dir in sorted(facts_root.iterdir()):
        if limit and symbol_counter >= limit:
            break
        if not symbol_dir.is_dir():
            continue

        snapshot = latest_snapshot(symbol_dir)
        if not snapshot:
            continue

        us_gaap = load_us_gaap(snapshot)
        if not us_gaap:
            continue

        symbol = symbol_dir.name.upper()
        sector = sector_map.get(symbol, {}).get("sector", "Unknown")
        mapper_sector = sector if sector != "Unknown" else None

        memo = {}
        for canonical_key, mapping in mapper.mappings.items():
            if not mapping:
                continue

            # Skip keys that have neither tags nor derivations configured
            has_defined_tags = bool(mapping.get("global_fallback") or mapping.get("sector_specific"))
            has_derived_logic = mapping.get("derived", {}).get("enabled", False)
            if not has_defined_tags and not has_derived_logic:
                continue

            bucket = coverage[canonical_key][sector]
            bucket["total"] += 1

            if has_raw_support(canonical_key, mapper, mapper_sector, us_gaap, memo):
                bucket["present"] += 1

        symbol_counter += 1

    return coverage, symbol_counter


def print_summary(coverage: Dict, total_symbols: int) -> None:
    print(f"Audited {total_symbols} symbols\n")
    header = f"{'Canonical Key':40s} {'Coverage':>12s}   Top Gaps"
    print(header)
    print("-" * len(header))

    for canonical_key in sorted(coverage.keys()):
        sector_stats = coverage[canonical_key]
        overall_present = sum(stats["present"] for stats in sector_stats.values())
        overall_total = sum(stats["total"] for stats in sector_stats.values())
        if overall_total == 0:
            continue

        pct = (overall_present / overall_total) * 100
        sector_gaps = []
        for sector, stats in sector_stats.items():
            total = stats["total"]
            if total == 0:
                continue
            sector_pct = (stats["present"] / total) * 100
            if sector_pct < 80:
                sector_gaps.append(f"{sector}:{sector_pct:.0f}%")

        gap_str = ", ".join(sorted(sector_gaps)[:3]) if sector_gaps else ""
        print(f"{canonical_key:40s} {overall_present:4d}/{overall_total:<4d} ({pct:5.1f}%)   {gap_str}")


def serialize_report(coverage: Dict, output_path: Path, facts_root: Path, total_symbols: int) -> None:
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "facts_root": str(facts_root),
        "symbols_audited": total_symbols,
        "coverage": {},
    }

    for canonical_key, sector_stats in coverage.items():
        output["coverage"][canonical_key] = {}
        for sector, stats in sector_stats.items():
            total = stats["total"]
            present = stats["present"]
            pct = (present / total) * 100 if total else 0
            output["coverage"][canonical_key][sector] = {
                "present": present,
                "total": total,
                "coverage_pct": pct,
            }

    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Audit canonical key coverage using cached SEC raw facts.")
    parser.add_argument(
        "--facts-root", type=Path, default=Path("data/sec_cache/facts/raw"), help="Root directory for raw facts cache."
    )
    parser.add_argument(
        "--sector-map",
        type=Path,
        default=Path("data/sector_industry_ticker_map.txt"),
        help="Pipe-delimited sector map file.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of symbols to scan.")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON coverage report.")
    args = parser.parse_args()

    if not args.facts_root.exists():
        print(f"Facts root not found: {args.facts_root}")
        sys.exit(1)

    sector_map = parse_sector_map(args.sector_map)
    coverage, total_symbols = audit_coverage(args.facts_root, sector_map, args.limit)
    print_summary(coverage, total_symbols)

    if args.output:
        serialize_report(coverage, args.output, args.facts_root, total_symbols)
        print(f"\nSaved coverage report to {args.output}")


if __name__ == "__main__":
    main()
