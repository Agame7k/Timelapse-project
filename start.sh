#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check for environment file
if [ ! -f "cred.env" ]; then
    echo "Error: cred.env not found! Please create it using the template."
    exit 1
fi

# Create required directories
mkdir -p timelapse_photos_primary/camera0
mkdir -p timelapse_photos_backup/camera0

# Start the bot
echo "Starting Timelapse Bot..."
python3 Timelapse.py