#!/bin/bash
# InvestiGator - Missing Package Installation Script
# This script installs TA-Lib and markdown for enhanced report generation

echo "📦 Installing missing packages for InvestiGator..."
echo ""

# Activate conda environment
echo "🔧 Activating conda environment: investment_ai_env"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate investment_ai_env

# Install markdown via conda
echo ""
echo "📥 Installing markdown package..."
conda install -c conda-forge markdown -y

# Install TA-Lib (requires special handling on macOS)
echo ""
echo "📥 Installing TA-Lib (Technical Analysis Library)..."

# Check if Homebrew is available
if command -v brew &> /dev/null; then
    echo "   Installing TA-Lib C library via Homebrew..."
    brew install ta-lib
    
    echo "   Installing TA-Lib Python wrapper..."
    pip install TA-Lib
else
    echo "⚠️  Homebrew not found. Installing via conda..."
    conda install -c conda-forge ta-lib -y
fi

# Verify installations
echo ""
echo "✅ Verifying installations..."
python -c "import talib; import markdown; print('✅ All packages installed successfully!')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation complete! You can now run InvestiGator with full features."
else
    echo ""
    echo "❌ Some packages failed to install. Please check the errors above."
    echo "   You can try manual installation:"
    echo "   - For markdown: pip install markdown"
    echo "   - For TA-Lib: brew install ta-lib && pip install TA-Lib"
fi
