#!/bin/bash

# Setup script for OpenVLA LIBERO environment
# This script installs all dependencies in the correct order

set -e  # Exit on any error

echo "Setting up OpenVLA LIBERO environment..."

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv_openvla_libero
source venv_openvla_libero/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (important for CUDA compatibility)
echo "Installing PyTorch..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Install TensorFlow and let it handle protobuf dependencies
echo "Installing TensorFlow..."
pip install tensorflow==2.15.0

# Install core dependencies from minimal requirements
echo "Installing core dependencies..."
pip install -r requirements_minimal.txt

# Install LIBERO from source
echo "Installing LIBERO..."
if [ ! -d "LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
cd LIBERO
pip install -e .
cd ..

# Fix the import issue in run_libero_eval.py
echo "Fixing import issue..."
sed -i 's/from libero.libero import benchmark/from libero import benchmark/' run_libero_eval.py

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from libero import benchmark; print('LIBERO imported successfully')"
python -c "import robosuite; print('Robosuite imported successfully')"
python -c "import mujoco; print('MuJoCo imported successfully')"

echo "Setup complete! Activate the environment with:"
echo "source venv_openvla_libero/bin/activate"
