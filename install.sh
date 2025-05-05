#!/bin/bash
# Installation script for Synth

# Display banner
echo "======================================================"
echo "  Synth: Production-Ready Synthetic Data Generator    "
echo "======================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $python_version"

# Set up virtual environment
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Create example schemas
echo "Creating example schemas..."
python main.py schema create-example tabular --output tabular_schema.json
python main.py schema create-example timeseries --output timeseries_schema.json

# Make scripts executable
chmod +x main.py
chmod +x examples/ecommerce_example.py

echo ""
echo "Installation complete! Here are some ways to get started:"
echo ""
echo "1. Run the e-commerce example:"
echo "   ./examples/ecommerce_example.py"
echo ""
echo "2. Generate tabular data with CLI:"
echo "   ./main.py run --schema tabular_schema.json --type tabular"
echo ""
echo "3. Start the API server:"
echo "   ./main.py api"
echo ""
echo "4. Run with continuous data generation and drift:"
echo "   ./main.py run --schema timeseries_schema.json --type timeseries --continuous --drift"
echo ""
echo "For more information, see the README.md file."
echo ""