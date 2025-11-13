@echo off

rem Set the use case
set USE_CASE=face_detection

rem Move current directory to parent directory
cd /d "%~dp0\.."

rem Prepare the dataset
rem Print out the notification
echo Distributing data ...
uv run python src/use_cases/%USE_CASE%/utils/distribute_data.py
echo Data distributed.

rem Open File Explorer to show distributed data
echo Opening File Explorer to show distributed data...
set DATA_PATH=src\use_cases\%USE_CASE%\distributed_data
if exist "%DATA_PATH%" (
    start explorer "%CD%\%DATA_PATH%"
) else (
    echo Warning: Directory %DATA_PATH% does not exist.
    echo Please check if the data distribution was successful.
)

rem Ask user confirmation to continue
echo.
echo Please review the distributed data in the opened folder.

:ask_continue
set /p choice="Do you want to continue to start the server? (YES/NO): "

rem Convert to uppercase for comparison
for %%i in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do call set choice=%%choice:%%i=%%i%%

if /i "%choice%"=="YES" (
    echo Continuing to start the server...
    goto start_server
) else if /i "%choice%"=="Y" (
    echo Continuing to start the server...
    goto start_server
) else if /i "%choice%"=="NO" (
    echo Operation cancelled by user.
    pause
    exit /b 0
) else if /i "%choice%"=="N" (
    echo Operation cancelled by user.
    pause
    exit /b 0
) else (
    echo Please answer YES or NO.
    goto ask_continue
)

:start_server
rem Start the server
echo Starting the server ...
uv run python src/use_cases/%USE_CASE%/main_server.py

pause