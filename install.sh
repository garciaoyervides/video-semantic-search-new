#!/bin/bash
echo "Contextual Search App Installer will install a Python virtual environment"
CWD=$(pwd)
conda create --prefix "$CWD"/venvs/video-search --file "$CWD"/VIDEO_SEARCH_VENV.txt -y || exit $1
eval "$(conda shell.bash hook)" || exit $1
conda activate "$CWD"/venvs/video-search || exit $1
pip install streamlit ftfy regex tqdm scikit-learn moviepy yellowbrick chromadb transformers==4.31.0 openai|| exit $1
pip install git+https://github.com/openai/CLIP.git || exit $1
pip install git+https://github.com/TatsuyaShirakawa/KTS.git || exit $1
#pip install git+https://github.com/garciaoyervides/LAVIS.git || exit $1
#pip install git+https://github.com/m-bain/whisperx.git || exit $1
#pip install git+https://github.com/openai/whisper.git || exit $1
#pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt || exit $1
conda deactivate
echo "App successfully installed, now you can start the servers."