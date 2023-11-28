# video-semantic-search
Video semantic search tool

Requirements: Python >= 3.9, conda, CUDA 12.1

Only tested on Ubuntu linux-64 with CUDA 12.1, and WSL Ubuntu with CUDA 12.1

Run install.sh
```
sh install.sh
```
This will create a Python virtual environment using conda.

Then run all servers:
```
sh run_frontend_api.sh
sh run_search_api.sh
sh run_whisper_x_api.sh
sh run_lavis_api.sh
```

search_api server runs on GPU, modest VRAM usage.
whisper_x server runs on GPU, modest VRAM usage.
Warning: lavis_api server runs on CPU, requires A LOT of RAM (around 16GB).

whisper_x and lavis_api servers are required only when processing videos. The search function only requires the search_api server.