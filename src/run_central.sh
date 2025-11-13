#!/bin/bash

# shellcheck disable=SC2034
USE_CASE="face_detection"

# move current directory to parent directory
cd "$(dirname "$0")/.."

echo "🌸 FedFlower - Face Classification Central Training"
echo "=" * 50

# Run central training
echo "Starting central training..."
uv run python src/use_cases/$USE_CASE/central_run.py

echo "Central training completed."