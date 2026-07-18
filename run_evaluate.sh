#!/bin/bash

# Activate the virtual environment
source .venv_linux/bin/activate

# Dynamically find and inject all NVIDIA runtime pip packages into LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(find $PWD/.venv_linux/lib/python3.10/site-packages/nvidia -type d -name lib | tr "\n" ":")/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "Running evaluation script with GPU acceleration..."
python evaluate.py
