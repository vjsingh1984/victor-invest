#!/usr/bin/env python3
"""
Simple script to analyze ABNB without complex dependencies
"""

import subprocess
import sys
import json
from datetime import datetime


def run_analysis(symbol="ABNB"):
    """Run comprehensive analysis for a symbol"""

    print(f"\n{'='*60}")
    print(f"Starting Comprehensive Analysis for {symbol}")
    print(f"Time: {datetime.now()}")
    print(f"{'='*60}\n")

    # Step 1: Run SEC fundamental analysis
    print("Step 1: Running SEC Fundamental Analysis...")
    try:
        result = subprocess.run(
            ["python", "sec_fundamental.py", "--symbol", symbol], capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print("✅ SEC analysis completed")
        else:
            print(f"⚠️ SEC analysis had issues: {result.stderr[:200]}")
    except Exception as e:
        print(f"❌ SEC analysis failed: {e}")

    print()

    # Step 2: Run technical analysis
    print("Step 2: Running Technical Analysis...")
    try:
        result = subprocess.run(
            ["python", "yahoo_technical.py", "--symbol", symbol], capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("✅ Technical analysis completed")
        else:
            print(f"⚠️ Technical analysis had issues: {result.stderr[:200]}")
    except Exception as e:
        print(f"❌ Technical analysis failed: {e}")

    print()

    # Step 3: Run synthesis
    print("Step 3: Running Investment Synthesis...")
    try:
        result = subprocess.run(
            ["python", "synthesizer.py", "--symbol", symbol, "--synthesis-mode", "comprehensive", "--report"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print("✅ Synthesis completed")
            print("\n" + "=" * 60)
            print("ANALYSIS SUMMARY:")
            print("=" * 60)
            # Try to extract summary from output
            output_lines = result.stdout.split("\n")
            for i, line in enumerate(output_lines):
                if "INVESTMENT RECOMMENDATION" in line.upper():
                    # Print next 10 lines as summary
                    for j in range(i, min(i + 10, len(output_lines))):
                        print(output_lines[j])
                    break
        else:
            print(f"⚠️ Synthesis had issues: {result.stderr[:200]}")
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")

    print()
    print(f"{'='*60}")
    print(f"Analysis Complete for {symbol}")
    print(f"Check the 'reports' directory for detailed PDF report")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ABNB"
    run_analysis(symbol)
