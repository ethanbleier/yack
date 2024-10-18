#!/bin/bash

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
fi

source "$PROJECT_ROOT/venv/bin/activate"

pip install -r "$PROJECT_ROOT/requirements.txt"

echo "Starting Face and Hand Tracker..."

python "$PROJECT_ROOT/src/main.py"

deactivate
