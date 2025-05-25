@echo off
REM === BareMetalRT Backend Startup Script ===
REM This script ensures you always start the backend from the project root,
REM using the correct virtual environment and settings.

REM Change directory to project root (parent of 'scripts')
cd /d %~dp0\..

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Start the FastAPI backend server
python -m uvicorn api.openai_api:app --host 0.0.0.0 --port 8000

REM Keep window open so you can see errors or logs
pause
