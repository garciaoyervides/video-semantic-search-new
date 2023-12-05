#!/bin/bash
CWD=$(pwd)
cd frontend
$CWD/venv/bin/python -m streamlit run main.py --server.maxUploadSize=4000