#!/bin/bash

set -e

echo "=================================================="
echo "Installing system dependencies"
echo "=================================================="

sudo apt update
sudo apt install -y git python3 python3-pip python3-venv

echo "=================================================="
echo "Cloning CardioXplain repository"
echo "=================================================="

if [ ! -d "CardioXplain" ]; then
    git clone --branch phase2 https://github.com/HiranCser/CardioXplain.git
else
    echo "Repository already exists. Pulling latest changes..."
    cd CardioXplain
    git checkout phase2
    git pull
    cd ..
fi

echo "=================================================="
echo "Entering project directory"
echo "=================================================="

cd CardioXplain

echo "=================================================="
echo "Creating virtual environment"
echo "=================================================="

if [ ! -d "venv" ]; then
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

echo "=================================================="
echo "Activating virtual environment"
echo "=================================================="

source venv/bin/activate

echo "=================================================="
echo "Upgrading pip"
echo "=================================================="

pip install --upgrade pip

echo "=================================================="
echo "Installing project requirements"
echo "=================================================="

pip install -r requirements.txt

echo "=================================================="
echo "Installing CUDA 11.8 compatible PyTorch"
echo "=================================================="

pip uninstall -y torch torchvision torchaudio || true

pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

echo "=================================================="
echo "Verifying PyTorch installation"
echo "=================================================="

python -c "import torch; print('Torch Version:', torch.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

echo "=================================================="
echo "Setup completed successfully!"
echo "=================================================="

echo ""
echo "To activate environment later:"
echo "cd CardioXplain"
echo "source venv/bin/activate"