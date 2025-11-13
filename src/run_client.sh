#!/bin/bash

# shellcheck disable=SC2034
USE_CASE="face_detection"

# read client_id from command line argument
if [ $# -eq 0 ]; then
    echo "No client_id provided. Using default client_id=0"
    CLIENT_ID=0
else
    CLIENT_ID=$1
    echo "Using client_id=$CLIENT_ID"
fi

# move current directory to parent directory
cd "$(dirname "$0")/.."


# Start the client
echo "Starting the client service ..."
uv run python src/use_cases/$USE_CASE/main_client.py --client-id $CLIENT_ID