#!/bin/bash

# YouTube Daily Runner - Runs at login, but only executes if >36 hours since last run
SCRIPT_DIR="/Users/vuk/projects/youtube_daily"
LAST_RUN_FILE="$SCRIPT_DIR/.last_run"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
PYTHON_SCRIPT="$SCRIPT_DIR/fetch_youtube_videos.py"

# Get current timestamp
CURRENT_TIME=$(date +%s)

# Check if last run file exists
if [ -f "$LAST_RUN_FILE" ]; then
    LAST_RUN=$(cat "$LAST_RUN_FILE")
    TIME_DIFF=$((CURRENT_TIME - LAST_RUN))
    HOURS_PASSED=$((TIME_DIFF / 3600))

    echo "==================================" >> "$SCRIPT_DIR/run_log.txt"
    echo "Checked at: $(date)" >> "$SCRIPT_DIR/run_log.txt"
    echo "Last run: $(date -r $LAST_RUN '+%Y-%m-%d %H:%M:%S')" >> "$SCRIPT_DIR/run_log.txt"
    echo "Hours since last run: $HOURS_PASSED" >> "$SCRIPT_DIR/run_log.txt"

    # Only run if more than 36 hours (129600 seconds) have passed
    if [ $TIME_DIFF -lt 129600 ]; then
        echo "Script ran $HOURS_PASSED hours ago. Need 36 hours. Skipping." >> "$SCRIPT_DIR/run_log.txt"
        exit 0
    fi
else
    echo "==================================" >> "$SCRIPT_DIR/run_log.txt"
    echo "No previous run detected. Running for the first time." >> "$SCRIPT_DIR/run_log.txt"
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Run the YouTube fetch script
echo "Running YouTube fetch at $(date '+%Y-%m-%d %H:%M:%S')..." >> "$SCRIPT_DIR/run_log.txt"

"$VENV_PYTHON" "$PYTHON_SCRIPT" >> "$SCRIPT_DIR/run_log.txt" 2>&1

# Record this run time
echo $CURRENT_TIME > "$LAST_RUN_FILE"
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')" >> "$SCRIPT_DIR/run_log.txt"
