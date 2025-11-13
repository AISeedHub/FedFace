#!/bin/bash

# shellcheck disable=SC2034
USE_CASE="face_detection"

# move current directory to parent directory
cd "$(dirname "$0")/.."

# Prepare the dataset
# print out the notification
echo "Distributing data ..."
uv run python src/use_cases/$USE_CASE/utils/distribute_data.py
echo "Data distributed."

# Open File Explorer to show distributed data
echo "Opening File Explorer to show distributed data..."
DATA_PATH="src/use_cases/$USE_CASE/distributed_data"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # For Windows (Git Bash, MSYS2, Cygwin)
    explorer.exe "$(cygpath -w "$DATA_PATH")" 2>/dev/null || cmd.exe /c "start explorer \"$PWD\\$DATA_PATH\""
elif command -v wslpath &> /dev/null; then
    # For WSL
    explorer.exe "$(wslpath -w "$PWD/$DATA_PATH")"
else
    echo "Warning: Could not open File Explorer. Please manually check: $DATA_PATH"
fi

# Ask user confirmation to continue
echo ""
echo "Please review the distributed data in the opened folder."
while true; do
    read -p "Do you want to continue to start the server? (YES/NO): " yn
    case $yn in
        [Yy]* | YES | yes | Yes )
            echo "Continuing to start the server..."
            break
            ;;
        [Nn]* | NO | no | No )
            echo "Operation cancelled by user."
            exit 0
            ;;
        * )
            echo "Please answer YES or NO."
            ;;
    esac
done

# Start the server
echo "Starting the server ..."
uv run python src/use_cases/$USE_CASE/main_server.py