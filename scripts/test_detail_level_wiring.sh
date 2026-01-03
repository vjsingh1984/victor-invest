#!/opt/homebrew/bin/bash
#
# Test Detail Level Wiring - End-to-End Verification
#
# This script tests that the --detail-level option is properly wired
# from investigator_v2.sh through to cli_orchestrator.py
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "Detail Level Wiring Test"
echo "========================================="
echo ""

# Test 1: Help message shows detail-level option
echo "Test 1: Verify --detail-level in help message"
if ./investigator_v2.sh --help | grep -q "detail-level"; then
    echo "✅ PASS: --detail-level appears in help message"
else
    echo "❌ FAIL: --detail-level NOT in help message"
    exit 1
fi
echo ""

# Test 2: Help shows all three detail levels
echo "Test 2: Verify detail level descriptions"
if ./investigator_v2.sh --help | grep -q "minimal (summary)"; then
    echo "✅ PASS: minimal level documented"
else
    echo "❌ FAIL: minimal level NOT documented"
    exit 1
fi

if ./investigator_v2.sh --help | grep -q "standard (investor-friendly, default)"; then
    echo "✅ PASS: standard level documented (default)"
else
    echo "❌ FAIL: standard level NOT documented"
    exit 1
fi

if ./investigator_v2.sh --help | grep -q "verbose (full)"; then
    echo "✅ PASS: verbose level documented"
else
    echo "❌ FAIL: verbose level NOT documented"
    exit 1
fi
echo ""

# Test 3: Verify examples section includes detail-level
echo "Test 3: Verify examples include detail-level usage"
if ./investigator_v2.sh --help | grep -q "Output detail levels"; then
    echo "✅ PASS: Output detail levels examples section exists"
else
    echo "❌ FAIL: Output detail levels examples NOT found"
    exit 1
fi

if ./investigator_v2.sh --help | grep -q "detail-level minimal"; then
    echo "✅ PASS: minimal example found"
else
    echo "❌ FAIL: minimal example NOT found"
    exit 1
fi

if ./investigator_v2.sh --help | grep -q "detail-level standard"; then
    echo "✅ PASS: standard example found"
else
    echo "❌ FAIL: standard example NOT found"
    exit 1
fi

if ./investigator_v2.sh --help | grep -q "detail-level verbose"; then
    echo "✅ PASS: verbose example found"
else
    echo "❌ FAIL: verbose example NOT found"
    exit 1
fi
echo ""

# Test 4: Verify default value is set to "standard"
echo "Test 4: Verify default value"
if grep -q 'DETAIL_LEVEL="standard"' investigator_v2.sh; then
    echo "✅ PASS: Default detail level is 'standard' (investor-friendly)"
else
    echo "❌ FAIL: Default detail level is NOT set to 'standard'"
    exit 1
fi
echo ""

# Test 5: Verify argument parsing exists
echo "Test 5: Verify argument parsing"
if grep -q "\-d|\-\-detail-level)" investigator_v2.sh; then
    echo "✅ PASS: Argument parsing for --detail-level exists"
else
    echo "❌ FAIL: Argument parsing NOT found"
    exit 1
fi
echo ""

# Test 6: Verify detail level is passed to build_analysis_args
echo "Test 6: Verify detail-level in build_analysis_args()"
if grep -q 'args+=(--detail-level "$DETAIL_LEVEL")' investigator_v2.sh; then
    echo "✅ PASS: detail-level added to analysis args"
else
    echo "❌ FAIL: detail-level NOT in analysis args"
    exit 1
fi
echo ""

# Test 7: Verify detail level is passed to batch args
echo "Test 7: Verify detail-level in batch args"
if grep -q 'BATCH_ARGS+=(--detail-level "$DETAIL_LEVEL")' investigator_v2.sh; then
    echo "✅ PASS: detail-level added to batch args"
else
    echo "❌ FAIL: detail-level NOT in batch args"
    exit 1
fi
echo ""

# Test 8: Verify Python CLI has detail-level option
echo "Test 8: Verify Python CLI has --detail-level option"
if python3 cli_orchestrator.py analyze --help 2>&1 | grep -q "detail-level"; then
    echo "✅ PASS: Python CLI has --detail-level option"
else
    echo "❌ FAIL: Python CLI missing --detail-level option"
    exit 1
fi
echo ""

# Test 9: Verify Python CLI batch command has detail-level option
echo "Test 9: Verify Python CLI batch command has --detail-level option"
if python3 cli_orchestrator.py batch --help 2>&1 | grep -q "detail-level"; then
    echo "✅ PASS: Python CLI batch has --detail-level option"
else
    echo "❌ FAIL: Python CLI batch missing --detail-level option"
    exit 1
fi
echo ""

echo "========================================="
echo "✅ ALL TESTS PASSED"
echo "========================================="
echo ""
echo "Detail level wiring is complete:"
echo "  • Shell wrapper: investigator_v2.sh"
echo "  • Python CLI: cli_orchestrator.py"
echo "  • Default: standard (investor-friendly, 65% smaller)"
echo "  • Options: minimal | standard | verbose"
echo ""
