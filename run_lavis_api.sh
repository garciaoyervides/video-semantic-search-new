#!/bin/bash
CWD=$(pwd)
eval "$(conda shell.bash hook)"|| exit $1
conda activate "$CWD"/venvs/video-search|| exit $1
cd lavis-api || exit $1
python -m flask run --no-debugger --no-reload --host="0.0.0.0"  -p 5002