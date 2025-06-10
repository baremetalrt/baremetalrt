#!/bin/bash
# Activate the Python virtual environment and launch the FastAPI OpenAI-compatible backend in WSL2

# Activate venv (edit path if your venv is elsewhere)
source ~/baremetalrt-venv/bin/activate

# Change to project directory (edit if your path is different)
cd ~/baremetalrt

# Start the API
uvicorn api.openai_api:app --host 0.0.0.0 --port 8000
