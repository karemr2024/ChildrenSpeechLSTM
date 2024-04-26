#!/bin/bash

# Check if a directory for the virtual environment exists
if [ ! -d "venv" ]; then
    # Create the virtual environment in a directory named "venv"
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete. The virtual environment is ready and isolated."
