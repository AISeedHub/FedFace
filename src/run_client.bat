@echo off

rem Set the use case
set USE_CASE=face_detection

rem Read client_id from command line argument
if "%1"=="" (
    echo No client_id provided. Using default client_id=0
    set CLIENT_ID=0
) else (
    set CLIENT_ID=%1
    echo Using client_id=%CLIENT_ID%
)

rem Move current directory to parent directory
cd /d "%~dp0\.."

rem Start the client
echo Starting the client service ...
uv run python src/use_cases/%USE_CASE%/main_client.py --client-id %CLIENT_ID%

pause