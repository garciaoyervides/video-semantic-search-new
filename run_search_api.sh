#!/bin/bash
CWD=$(pwd)
cd search-api
$CWD/venv/bin/python -m flask run --no-debugger --no-reload  --host="0.0.0.0" -p 5000