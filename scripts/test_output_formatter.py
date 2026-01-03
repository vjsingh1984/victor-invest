#!/usr/bin/env python3
"""
Test Output Formatter - Demonstrate Size Reduction

Usage:
    python3 test_output_formatter.py results/CL_20251108_203055.json
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from investigator.application import OutputDetailLevel, format_analysis_output


def test_formatter(input_file: str):
    """Test formatter with different detail levels"""

    # Load original results
    print(f"Loading {input_file}...")
    with open(input_file, "r") as f:
        original_data = json.load(f)

    original_size = len(json.dumps(original_data))
    print(f"\nOriginal size: {original_size:,} bytes ({original_size/1024:.1f} KB)")

    # Test VERBOSE (should be same as original)
    verbose_data = format_analysis_output(original_data, OutputDetailLevel.VERBOSE)
    verbose_size = len(json.dumps(verbose_data))
    print(f"\nVERBOSE detail level:")
    print(f"  Size: {verbose_size:,} bytes ({verbose_size/1024:.1f} KB)")
    print(f"  Reduction: 0% (same as original)")

    # Test STANDARD (default - removes duplicates and metadata)
    standard_data = format_analysis_output(original_data, OutputDetailLevel.STANDARD)
    standard_size = len(json.dumps(standard_data))
    reduction_pct = (original_size - standard_size) / original_size * 100
    print(f"\nSTANDARD detail level (default):")
    print(f"  Size: {standard_size:,} bytes ({standard_size/1024:.1f} KB)")
    print(f"  Reduction: {reduction_pct:.1f}%")
    print(f"  Removed: {original_size - standard_size:,} bytes")

    # Test MINIMAL (executive summary only)
    minimal_data = format_analysis_output(original_data, OutputDetailLevel.MINIMAL)
    minimal_size = len(json.dumps(minimal_data))
    minimal_reduction_pct = (original_size - minimal_size) / original_size * 100
    print(f"\nMINIMAL detail level (summary only):")
    print(f"  Size: {minimal_size:,} bytes ({minimal_size/1024:.1f} KB)")
    print(f"  Reduction: {minimal_reduction_pct:.1f}%")
    print(f"  Removed: {original_size - minimal_size:,} bytes")

    # Show structure of STANDARD output
    print(f"\n{'='*60}")
    print("STANDARD output structure (investor decision-making):")
    print(f"{'='*60}")
    print("Top-level keys:", list(standard_data.keys()))

    if "fundamental" in standard_data:
        print("\nFundamental section keys:", list(standard_data["fundamental"].keys()))

    if "technical" in standard_data:
        print("Technical section keys:", list(standard_data["technical"].keys()))

    if "synthesis" in standard_data:
        print("Synthesis section keys:", list(standard_data["synthesis"].keys()))

    # Show structure of MINIMAL output
    print(f"\n{'='*60}")
    print("MINIMAL output structure (executive summary):")
    print(f"{'='*60}")
    print(json.dumps(minimal_data, indent=2)[:500] + "...")

    # Save formatted outputs for inspection
    output_dir = Path(input_file).parent
    base_name = Path(input_file).stem

    standard_file = output_dir / f"{base_name}_standard.json"
    minimal_file = output_dir / f"{base_name}_minimal.json"

    with open(standard_file, "w") as f:
        json.dump(standard_data, f, indent=2, default=str)

    with open(minimal_file, "w") as f:
        json.dump(minimal_data, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("Formatted outputs saved:")
    print(f"  STANDARD: {standard_file}")
    print(f"  MINIMAL: {minimal_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_output_formatter.py <results_file.json>")
        print("\nExample:")
        print("  python3 test_output_formatter.py results/CL_20251108_203055.json")
        sys.exit(1)

    input_file = sys.argv[1]

    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    test_formatter(input_file)
