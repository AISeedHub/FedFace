
@echo off
setlocal enabledelayedexpansion

rem Set use case variable
set "USE_CASE=face_detection"

rem Move current directory to parent directory
cd /d "%~dp0\.."

echo 🌸 FedFlower - Face Classification Central Training
echo ==================================================

rem Run central training 5 times
for /L %%i in (1,1,5) do (
    echo.
    echo === Training Round %%i of 5 ===
    echo Starting central training round %%i...
    uv run python src/use_cases/!USE_CASE!/central_run.py
    echo Central training round %%i completed.
    echo.
)

echo All 5 training rounds completed.

pause