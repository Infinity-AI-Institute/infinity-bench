#!/bin/bash
# Quick setup script for infinity-benchmark

echo "Installing infinity-benchmark dependencies..."
pip install -r requirements.txt

echo "Setup complete! You can now run:"
echo "  python run_mlp_baseline.py"
echo "  python run_mlp_baseline.py --out results.json"
