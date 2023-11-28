#!/bin/bash
CWD=$(pwd)
eval "$(conda shell.bash hook)"
conda activate "$CWD"/venvs/video-search
chroma run --path ./db