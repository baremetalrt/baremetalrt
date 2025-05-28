@echo off
REM === BareMetalRT Backend Startup Script ===
REM Prompts user to select a model, launches the model script, then starts the backend.

REM Change directory to project root (parent of 'scripts')
cd /d %~dp0\..

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM List available models
setlocal enabledelayedexpansion
set MODELS[0]=llama2_7b_chat_8int.py
set MODELS[1]=llama2_13b_chat_4bit.py
set MODELS[2]=llama2_7b_chat_fp16.py
set MODELS[3]=mistral_7b_instruct_8bit.py
set MODELS[4]=mixtral_8x7b_instruct_4bit.py
set MODELS[5]=petals_llama2_70b_chat.py

set NMODELS=6

:show_menu
@echo.
@echo Select a model to launch:
for /L %%i in (0,1,%NMODELS%) do (
    if defined MODELS[%%i] echo   %%i: !MODELS[%%i]!
)
set /p CHOICE=Enter model number (0-%NMODELS%-1): 

REM Validate input
set VALID=0
for /L %%i in (0,1,%NMODELS%) do (
    if "!CHOICE!"=="%%i" set VALID=1
)
if %VALID%==0 (
    echo Invalid selection. Exiting.
    pause
    exit /b 1
)

REM Launch the selected model script
start "Model" python scripts\!MODELS[%CHOICE%]!

REM Start the FastAPI backend server
start "Backend" python -m uvicorn api.openai_api:app --host 0.0.0.0 --port 8000

REM Keep window open so you can see errors or logs
pause
endlocal
