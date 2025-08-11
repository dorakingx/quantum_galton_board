#!/bin/bash

echo "🚀 Quick Start: Gaussian Distribution"
echo "======================================"

# Run Gaussian experiment
python -m qgb gaussian \
    --layers 12 \
    --shots 20000 \
    --mode tree \
    --plot \
    --seed 123

echo ""
echo "✅ Gaussian experiment completed!"
echo "📊 Check the outputs/ directory for results and plots."
