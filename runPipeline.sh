#!/bin/bash

# Path to your Python virtual environment activation script
ENV_PATH="/Users/emre/Desktop/ChildrenSpeechLSTM_NonGit/.venv/bin/activate"

# Activate Python environment
if [ -f "$ENV_PATH" ]; then
    echo "Activating Python virtual environment..."
    source "$ENV_PATH"
else
    echo "Virtual environment activation script not found at $ENV_PATH."
    echo "Continuing without activating any environment."
fi

# Ensure the current directory is where the scripts are
cd "$(dirname "$0")"

# Run feature extraction
echo "Running feature extraction..."
python featureExtraction.py
echo "Feature extraction completed."

# Run data preparation
echo "Running data preparation..."
python dataPrep_test.py
echo "Data preparation completed."

# Run model training and evaluation
echo "Running model training and evaluation..."
python speech_model_test.py
echo "Model training and evaluation completed."

# Deactivate Python environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating Python virtual environment..."
    deactivate
fi

echo "All processes completed."
