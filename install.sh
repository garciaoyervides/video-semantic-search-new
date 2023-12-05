#!/bin/bash
echo "Contextual Search App Installer will install a Python virtual environment"
CWD=$(pwd)
CUDA=$(nvidia-smi | grep "CUDA Version")
python -m venv "$CWD"/venv || exit $1
source $CWD/venv/bin/activate

if [[ $CUDA == *"CUDA Version: 12"* ]]; then
    echo "CUDA 12.* detected"
    pip3 install torch torchvision torchaudio
elif [[ $CUDA == *"CUDA Version: 11"* ]]; then
    echo "CUDA 11.* detected"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No compatible CUDA detected"
    exit $1
fi

pip install -r requirements.txt

echo "Virtual  Environment successfully installed, now you can start the servers."