#!/bin/bash
echo "Contextual Search App Installer will install a Python virtual environment"
CWD=$(pwd)
CUDA=$(nvcc --version | grep release)
python -m venv "$CWD"/venv || exit $1

if [[ $CUDA == *"12"* ]]; then
    echo "CUDA 12.* detected"
    $CWD/venv/bin/python pip3 install torch torchvision torchaudio
elif [[ $CUDA == *"11"* ]]; then
    echo "CUDA 11.* detected"
    $CWD/venv/bin/python pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No compatible CUDA detected"
    exit $1
fi

$CWD/venv/bin/python pip install -r requirements.txt

echo "Virtual  Environment successfully installed, now you can start the servers."