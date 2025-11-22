#!/bin/bash
# Cleanup script for obsolete/duplicate files
# This script moves files to _obsolete/ instead of deleting them

echo "=========================================="
echo "Cleaning up obsolete/duplicate files"
echo "=========================================="

# Create _obsolete directory
mkdir -p _obsolete
mkdir -p _obsolete/tests
mkdir -p _obsolete/docs

echo ""
echo "1. Moving duplicate __init__ copy.py..."
if [ -f "src/stock_options/__init__ copy.py" ]; then
    mv "src/stock_options/__init__ copy.py" _obsolete/
    echo "   ✓ Moved: src/stock_options/__init__ copy.py"
else
    echo "   - Not found (already cleaned?)"
fi

echo ""
echo "2. Moving duplicate indicators.py (keeping the one in utils/)..."
if [ -f "src/stock_options/indicators.py" ]; then
    mv "src/stock_options/indicators.py" _obsolete/
    echo "   ✓ Moved: src/stock_options/indicators.py"
    echo "   → The correct one is in: src/stock_options/utils/indicators.py"
else
    echo "   - Not found (already cleaned?)"
fi

echo ""
echo "3. Moving old test files to _obsolete/tests/..."
if [ -f "test_flatten_default.py" ]; then
    mv "test_flatten_default.py" _obsolete/tests/
    echo "   ✓ Moved: test_flatten_default.py"
else
    echo "   - Not found"
fi

if [ -f "test_incremental_indicators.py" ]; then
    mv "test_incremental_indicators.py" _obsolete/tests/
    echo "   ✓ Moved: test_incremental_indicators.py"
else
    echo "   - Not found"
fi

echo ""
echo "4. Checking for duplicate documentation..."
if [ -f "docs/EPISODE_MANAGEMENT_AND_COMPARABILITY.md" ]; then
    # Check if it's a duplicate of EPISODIOS_Y_COMPARABILIDAD.md
    echo "   Found: docs/EPISODE_MANAGEMENT_AND_COMPARABILITY.md"
    echo "   Moving to _obsolete/docs/ (we have EPISODIOS_Y_COMPARABILIDAD.md)"
    mv "docs/EPISODE_MANAGEMENT_AND_COMPARABILITY.md" _obsolete/docs/
    echo "   ✓ Moved to _obsolete/docs/"
else
    echo "   - Not found (already cleaned?)"
fi

echo ""
echo "5. Moving old notebooks to _obsolete/..."
mkdir -p _obsolete/notebooks

for notebook in short.ipynb testing_gym_env.ipynb testing_gym_env_PHIA.ipynb; do
    if [ -f "$notebook" ]; then
        mv "$notebook" _obsolete/notebooks/
        echo "   ✓ Moved: $notebook"
    fi
done

echo ""
echo "=========================================="
echo "Cleanup Summary"
echo "=========================================="
echo ""
echo "Moved files to: _obsolete/"
echo ""
echo "You can:"
echo "  - Review files in _obsolete/"
echo "  - Delete _obsolete/ if you're sure you don't need them"
echo "  - Keep _obsolete/ as backup"
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Current directory structure:"
ls -la _obsolete/
