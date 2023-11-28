#!/bin/bash
CWD=$(pwd)
eval "$(conda shell.bash hook)"
conda activate "$CWD"/venvs/video-search
cd frontend
python -m streamlit run main.py --server.maxUploadSize=4000