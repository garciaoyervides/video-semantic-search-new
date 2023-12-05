# video-semantic-search
Video semantic search tool

Requirements: Python >= 3.9, conda, CUDA 11.* or 12.*
It depends on an external server (of your choosing) running Whisper or Vicuna.

## Install
OPTION 1: Run install.sh
```
sh install.sh
```
This will create a Python virtual environment and install requirements.

OPTION 2: Create a python virtual environment at:
```
video-semantic-search-new/venv
```
Install Pytorch.
Install requirements.txt with pip.

## Configure
Add the .env variables as follows:

frontend
```
API_ENDPOINT=http://127.0.0.1:5000
LAVIS_API_ENDPOINT=http://127.0.0.1:5002
```

search-api
```
WHISPER_API_ENDPOINT="your whisper api endpoint"
VICUNA_API_ENDPOINT="your vicuna api endpoint"
LAVIS_API_ENDPOINT=http://127.0.0.1:5002
```
## Run
Then run all servers:
```
./run_frontend_api.sh
./run_search_api.sh
./run_lavis_api.sh
```

search_api server can run on GPU, modest VRAM usage.
Warning: lavis-api server requires A LOT of RAM (around 16GB).

lavis-api and Whisper are required only when processing videos. Vicuna is required when processing video and for the image description function.
The search function only requires the search-api server.