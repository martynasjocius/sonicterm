#!/bin/bash

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

if ! python -c "import pygame, numpy, rich, watchdog, psutil" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

if [ ! -d "scenes" ]; then
  echo "Creating scenes/ directory..."
  mkdir scenes
fi

if [ $# -eq 0 ]; then
  echo "Running default scene with TUI (default mode)..."
  python sonicterm.py scenes/default.json
else
  echo "Running SonicTerm with arguments: $@"
  python sonicterm.py "$@"
fi
