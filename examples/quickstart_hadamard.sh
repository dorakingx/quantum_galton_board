#!/bin/bash

echo "🚀 Quick Start: Hadamard Quantum Walk"
echo "====================================="

# Run Hadamard quantum walk experiment
python -m qgb hadamard \
    --steps 12 \
    --shots 20000 \
    --mode tree \
    --noisy \
    --optimize \
    --plot \
    --seed 123

echo ""
echo "✅ Hadamard quantum walk experiment completed!"
echo "📊 Check the outputs/ directory for results and plots."
