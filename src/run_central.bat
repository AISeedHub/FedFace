
@echo off
setlocal enabledelayedexpansion

rem Set use case variable
set "USE_CASE=face_detection"

rem Move current directory to parent directory
cd /d "%~dp0\.."

echo 🌸 FedFlower - Face Classification Central Training
echo ==================================================

rem Run central training
echo Starting central training...
uv run python src/use_cases/%USE_CASE%/central_run.py

echo Central training completed.

pause