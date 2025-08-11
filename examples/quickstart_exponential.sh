#!/bin/bash

echo "🚀 Quick Start: Exponential Distribution"
echo "========================================"

# Run exponential experiment
python -m qgb exponential \
    --layers 10 \
    --lambda 0.35 \
    --shots 20000 \
    --mode tree \
    --noisy \
    --optimize \
    --plot \
    --seed 123

echo ""
echo "✅ Exponential experiment completed!"
echo "📊 Check the outputs/ directory for results and plots."
