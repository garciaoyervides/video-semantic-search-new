#!/bin/bash
CWD=$(pwd)
eval "$(conda shell.bash hook)"|| exit $1
conda activate "$CWD"/venvs/video-search|| exit $1
cd whisper-x-api || exit $1
python -m flask run --no-debugger --no-reload -p 5001