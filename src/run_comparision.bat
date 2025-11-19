
@echo off
setlocal enabledelayedexpansion

rem Set use case variable
set "USE_CASE=face_detection"

rem Move current directory to parent directory
cd /d "%~dp0\.."

rem Run 5 times
for /L %%i in (1,1,5) do (
    echo.
    echo === Testing Round %%i of 5 ===
    echo Starting round %%i...
    uv run python src/use_cases/!USE_CASE!/test_comparison.py
    echo Round %%i completed.
    echo.
)

echo All 5 testing rounds completed.

pause