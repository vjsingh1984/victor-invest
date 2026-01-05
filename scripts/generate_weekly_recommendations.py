#!/usr/bin/env python3
"""
Generate Weekly Stock Recommendations from RL Predictions

This script queries the valuation_outcomes database and generates
a structured recommendations report with top LONG and SHORT opportunities.

Usage:
    python3 scripts/generate_weekly_recommendations.py
    python3 scripts/generate_weekly_recommendations.py --days 14  # Custom lookback
    python3 scripts/generate_weekly_recommendations.py --top 20   # More recommendations
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import text
from investigator.infrastructure.database.db import get_database_engine


def get_predictions(engine, days: int = 7) -> List[Dict[str, Any]]:
    """Fetch recent predictions from database."""
    query = text("""
    SELECT
        symbol,
        analysis_date,
        blended_fair_value,
        current_price,
        predicted_upside_pct,
        tier_classification,
        position_type,
        model_weights,
        context_features,
        reward_30d,
        reward_90d
    FROM valuation_outcomes
    WHERE analysis_date >= CURRENT_DATE - INTERVAL :days
      AND predicted_upside_pct IS NOT NULL
      AND blended_fair_value IS NOT NULL
      AND current_price IS NOT NULL
    ORDER BY analysis_date DESC, predicted_upside_pct DESC
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"days": f"{days} days"})
        rows = result.fetchall()

    return [dict(row._mapping) for row in rows]


def categorize_predictions(predictions: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize predictions into actionable buckets."""
    longs = []
    shorts = []
    skips = []

    for p in predictions:
        upside = float(p.get("predicted_upside_pct", 0) or 0)
        position = p.get("position_type", "").upper()

        if position == "LONG" and upside > 15:
            longs.append(p)
        elif position == "SHORT" and upside < -20:
            shorts.append(p)
        else:
            skips.append(p)

    # Sort by upside/downside
    longs.sort(key=lambda x: -float(x.get("predicted_upside_pct", 0) or 0))
    shorts.sort(key=lambda x: float(x.get("predicted_upside_pct", 0) or 0))

    return {
        "longs": longs,
        "shorts": shorts,
        "skips": skips,
    }


def get_tier_performance(engine) -> Dict[str, Dict]:
    """Get historical performance by tier."""
    query = text("""
    SELECT
        tier_classification,
        COUNT(*) as total_predictions,
        AVG(reward_90d) as avg_reward_90d,
        COUNT(CASE WHEN reward_90d > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate
    FROM valuation_outcomes
    WHERE reward_90d IS NOT NULL
    GROUP BY tier_classification
    HAVING COUNT(*) >= 10
    ORDER BY avg_reward_90d DESC
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    return {
        row[0]: {
            "predictions": int(row[1]),
            "avg_reward": float(row[2]) if row[2] else 0,
            "win_rate": float(row[3]) if row[3] else 50,
        }
        for row in rows
    }


def format_recommendation(pred: Dict, tier_perf: Dict) -> Dict:
    """Format a single recommendation with metadata."""
    symbol = pred.get("symbol", "")
    upside = float(pred.get("predicted_upside_pct", 0) or 0)
    fair_value = float(pred.get("blended_fair_value", 0) or 0)
    price = float(pred.get("current_price", 0) or 0)
    tier = pred.get("tier_classification", "unknown")

    # Get tier performance
    tier_info = tier_perf.get(tier, {})

    # Extract sector from context
    context = pred.get("context_features") or {}
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except:
            context = {}

    return {
        "symbol": symbol,
        "analysis_date": str(pred.get("analysis_date", "")),
        "upside_pct": round(upside, 1),
        "fair_value": round(fair_value, 2),
        "current_price": round(price, 2),
        "tier": tier,
        "sector": context.get("sector", "Unknown"),
        "tier_win_rate": round(tier_info.get("win_rate", 50), 1),
        "tier_avg_reward": round(tier_info.get("avg_reward", 0), 4),
    }


def generate_report(
    predictions: List[Dict],
    tier_perf: Dict,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Generate the full recommendations report."""
    categorized = categorize_predictions(predictions)

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_predictions": len(predictions),
            "long_opportunities": len(categorized["longs"]),
            "short_opportunities": len(categorized["shorts"]),
            "skipped": len(categorized["skips"]),
        },
        "top_longs": [
            format_recommendation(p, tier_perf)
            for p in categorized["longs"][:top_n]
        ],
        "top_shorts": [
            format_recommendation(p, tier_perf)
            for p in categorized["shorts"][:top_n]
        ],
        "best_tiers": [
            {
                "tier": tier,
                "predictions": info["predictions"],
                "avg_reward": round(info["avg_reward"], 4),
                "win_rate": round(info["win_rate"], 1),
            }
            for tier, info in sorted(
                tier_perf.items(),
                key=lambda x: x[1]["avg_reward"],
                reverse=True
            )[:5]
        ],
    }

    return report


def print_report(report: Dict) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("WEEKLY RL RECOMMENDATIONS")
    print("=" * 60)
    print(f"Generated: {report['generated_at']}")
    print(f"Total Predictions: {report['summary']['total_predictions']}")
    print(f"Long Opportunities: {report['summary']['long_opportunities']}")
    print(f"Short Opportunities: {report['summary']['short_opportunities']}")

    if report["top_longs"]:
        print("\n" + "-" * 40)
        print("TOP LONG RECOMMENDATIONS")
        print("-" * 40)
        for i, rec in enumerate(report["top_longs"][:5], 1):
            print(f"{i}. {rec['symbol']:6s} | +{rec['upside_pct']:5.1f}% | "
                  f"${rec['current_price']:.2f} → ${rec['fair_value']:.2f} | "
                  f"{rec['tier']}")

    if report["top_shorts"]:
        print("\n" + "-" * 40)
        print("TOP SHORT RECOMMENDATIONS")
        print("-" * 40)
        for i, rec in enumerate(report["top_shorts"][:5], 1):
            print(f"{i}. {rec['symbol']:6s} | {rec['upside_pct']:5.1f}% | "
                  f"${rec['current_price']:.2f} → ${rec['fair_value']:.2f} | "
                  f"{rec['tier']}")

    if report["best_tiers"]:
        print("\n" + "-" * 40)
        print("BEST PERFORMING TIERS")
        print("-" * 40)
        for tier in report["best_tiers"]:
            print(f"  {tier['tier']}: {tier['avg_reward']:+.4f} reward, "
                  f"{tier['win_rate']:.0f}% win rate")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate weekly stock recommendations from RL predictions"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top recommendations per category (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: logs/weekly/recommendations_YYYYMMDD.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    args = parser.parse_args()

    # Initialize database
    engine = get_database_engine()

    # Fetch data
    predictions = get_predictions(engine, days=args.days)
    tier_perf = get_tier_performance(engine)

    if not predictions:
        print(f"No predictions found in the last {args.days} days")
        return

    # Generate report
    report = generate_report(predictions, tier_perf, top_n=args.top)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("logs/weekly")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"recommendations_{datetime.now():%Y%m%d}.json"

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    if not args.quiet:
        print_report(report)
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
