#!/bin/bash

# Alternative setup script that installs packages step by step
# This approach handles dependency conflicts better

set -e  # Exit on any error

echo "Setting up OpenVLA LIBERO environment (step-by-step approach)..."

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv_openvla_libero_alt
source venv_openvla_libero_alt/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages step by step
echo "Installing PyTorch..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

echo "Installing TensorFlow (this will handle protobuf automatically)..."
pip install tensorflow==2.15.0

echo "Installing transformers and related..."
pip install transformers==4.40.1 accelerate==1.8.1

echo "Installing diffusers and peft..."
pip install diffusers==0.33.1 peft==0.11.1

echo "Installing robotics packages..."
pip install robosuite==1.4.1 mujoco==3.3.3 gym==0.26.2

echo "Installing computer vision..."
pip install opencv-python pillow imageio

echo "Installing scientific computing..."
pip install "numpy>=1.23.5,<2.0.0" scipy matplotlib

echo "Installing utilities..."
pip install draccus==0.8.0 wandb tqdm einops easydict huggingface-hub PyYAML h5py requests jsonlines rich nltk sentencepiece

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
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "PyTorch import failed"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" || echo "TensorFlow import failed"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')" || echo "NumPy import failed"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" || echo "OpenCV import failed"
python -c "from libero import benchmark; print('LIBERO imported successfully')" || echo "LIBERO import failed"
python -c "import robosuite; print('Robosuite imported successfully')" || echo "Robosuite import failed"
python -c "import mujoco; print('MuJoCo imported successfully')" || echo "MuJoCo import failed"

echo "Setup complete! Activate the environment with:"
echo "source venv_openvla_libero_alt/bin/activate"
