# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YouTube Daily Video Tracker - An automated Python script that monitors YouTube channels for new long-form videos, fetches transcripts, generates AI summaries using a local Llama model, and sends email reports.

## Core Architecture

### Single-File Application
- **fetch_youtube_videos.py**: Main script containing all functionality
  - YouTube API integration for channel/video data retrieval
  - Transcript fetching using youtube-transcript-api
  - AI-powered summarization via local Llama 2 model (llama-cpp-python)
  - Email report generation with HTML formatting and file attachments
  - PrettyTable output formatting for console display

### Key Data Flow
1. Channel handles (@username) → YouTube API → Channel IDs
2. Channel IDs → Uploads playlist → Recent video IDs
3. Video IDs → Filter by duration (exclude Shorts <61s) and date
4. Video IDs → Transcript API → Raw transcript text
5. Transcript → Local Llama model → AI summary
6. All data → PrettyTable console output + Text file + HTML email

### Configuration System
- Environment variables managed via `.env` file:
  - `YOUTUBE_API_KEY`: Required for YouTube Data API v3
  - `EMAIL_ENABLED`: Toggle email functionality
  - `EMAIL_HOST`, `EMAIL_PORT`, `EMAIL_USERNAME`, `EMAIL_PASSWORD`, `EMAIL_TO`: SMTP configuration
- In-script constants (lines 48-52):
  - `FETCH_TRANSCRIPTS`: Enable/disable transcript fetching
  - `DAYS_FILTER`: Time window for video retrieval (default: 1 day)
  - `MAX_VIDEOS_PER_CHANNEL`: Limit videos processed per channel (default: 3)
  - `LLAMA_MODEL_PATH`: Path to local GGUF model file

### Channel Management
Three channel lists defined in script (lines 64-213):
- `CHANNELS_BAK`: Original full list (archived)
- `CHANNELS`: Currently active subset (17 channels focused on health, personal development, finance)
- `CHANNELS_FULL`: Complete list minus removed channels

## Running the Application

### Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # If exists, otherwise create .env
# Edit .env and add your YOUTUBE_API_KEY
```

### Execution
```bash
# Run with default settings (1 day lookback)
python fetch_youtube_videos.py

# Output files generated:
# - YOUTUBE_VUK_MMDDYYYY.txt (saved in project root)
# - Email sent if EMAIL_ENABLED=true
```

### Dependencies
- **google-api-python-client**: YouTube Data API v3 access
- **youtube-transcript-api**: Extract video transcripts (rate limited by IP)
- **llama-cpp-python**: Local LLM inference (requires GGUF model file)
- **prettytable**: Console table formatting
- **python-dotenv**: Environment variable management

## Important Implementation Details

### Llama Model Integration
- Model loaded once at startup (line 56) to avoid repeated initialization overhead
- Only loaded when `FETCH_TRANSCRIPTS=True`
- Default path: `/Users/vuk/projects/david_fast_api_backup/david_fast_api/llama_models/llama-2-7b-chat-hf-q4_k_m.gguf`
- Context window: 8192 tokens (n_ctx parameter)
- Summary generation uses dynamic length calculation (~20% of transcript, 500-5000 words)

### Transcript Fetching Rate Limits
- YouTube Transcript API is IP-based rate limited
- Script includes 4-second delay between requests (line 590)
- Error handling for: IP blocks (429), disabled transcripts, no transcript available
- The `get_video_transcript()` function returns tuple: (transcript_text, error_type)

### Video Filtering Logic
- Filters OUT YouTube Shorts: videos must have 'H' (hours) or 'M' (minutes) in duration, or >61 seconds
- Date filtering: only videos published within `DAYS_FILTER` days
- Duration parsing uses ISO 8601 format from YouTube API (line 506-511)

### Email Report Format
- HTML body with clickable video links
- Preserves newlines in summaries as `<br>` tags for readability
- Attaches full text report file (YOUTUBE_VUK_MMDDYYYY.txt)
- Supports multiple comma-separated recipients via `EMAIL_TO`

### Summary Formatting Requirements
The Llama prompt (lines 281-318) enforces strict formatting rules:
- Complete itemized lists with each item on new line
- All actionable items highlighted (stocks, foods, supplements, strategies)
- No truncation mid-sentence or mid-list
- Maximum 4096 tokens for summary generation

## Output File Naming
Pattern: `YOUTUBE_VUK_MMDDYYYY.txt` where MMDDYYYY is current date (e.g., `YOUTUBE_VUK_01242026.txt`)

## Common Modifications

### Adjusting Time Window
Change `DAYS_FILTER` constant (line 49) to desired number of days.

### Adding/Removing Channels
Edit the `CHANNELS` list (lines 127-152). Use format `@username` for channel handles.

### Changing Model Path
Update `LLAMA_MODEL_PATH` constant (line 51) to point to your GGUF model file.

### Disabling Email
Set `EMAIL_ENABLED=False` in `.env` file (or `EMAIL_ENABLED=false`).

### Summary Length Adjustment
Modify the calculation on lines 272-274 in `summarize_transcript()`. Current: 20% of transcript length (0.2 multiplier).

## Security Notes
- `.env` file contains sensitive credentials (API keys, email passwords) - gitignored
- `cookies.txt` is gitignored (browser cookies for authenticated requests)
- All output .txt files are gitignored except `requirements.txt`
