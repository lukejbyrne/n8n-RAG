#!/bin/bash

# Check if a virtual environment is currently active and deactivate it
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Deactivating the current virtual environment..."
    deactivate
fi

# Define the virtual environment name
VENV_DIR="venv"

# Remove existing virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf $VENV_DIR
fi

# Create a new virtual environment
echo "Creating virtual environment..."
python3 -m venv $VENV_DIR

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping dependency installation."
fi

echo "Setup complete. Virtual environment activated."
