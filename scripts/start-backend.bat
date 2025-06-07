@echo off
REM === BareMetalRT Backend Startup Script ===
REM Prompts user to select a model, launches the model script, then starts the backend.

REM Change directory to project root (parent of 'scripts')
cd /d %~dp0\..

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM List available models
setlocal enabledelayedexpansion
set MODELS[0]=llama3_1_8b
set MODELS[1]=llama2_7b_chat_8int
set NMODELS=2

:show_menu
@echo.
@echo Select a model to set as 'online':
for /L %%i in (0,1,%NMODELS%-1) do (
    if defined MODELS[%%i] echo   %%i: !MODELS[%%i]!
)
set /p CHOICE=Enter model number (0-%NMODELS%-1): 

REM Validate input
set VALID=0
for /L %%i in (0,1,%NMODELS%-1) do (
    if "!CHOICE!"=="%%i" set VALID=1
)
if %VALID%==0 (
    echo Invalid selection. Exiting.
    pause
    exit /b 1
)

REM Build JSON with selected model online, others offline
setlocal EnableDelayedExpansion
set JSON={
for /L %%i in (0,1,%NMODELS%-1) do (
    set STATUS=offline
    if %%i==%CHOICE% set STATUS=online
    set MODEL=!MODELS[%%i]!
    set JSON=!JSON!"!MODEL!": "!STATUS!"
    if not %%i==%NMODELS%-1 set JSON=!JSON!, 
)
set JSON=!JSON!}

REM Write to api/model_status.json
echo !JSON! > api\model_status.json
echo Set !MODELS[%CHOICE%]! as online in api\model_status.json.
endlocal

REM Start the FastAPI backend server
start "Backend" python -m uvicorn api.openai_api:app --host 0.0.0.0 --port 8000

REM Keep window open so you can see errors or logs
pause
endlocal
