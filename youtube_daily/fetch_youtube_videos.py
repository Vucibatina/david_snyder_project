#!/usr/bin/env python3
"""
Batch process to fetch YouTube videos from multiple channels.
Filters out YouTube Shorts and only shows regular long-form videos.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from http.cookiejar import MozillaCookieJar
from io import StringIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, IpBlocked
from prettytable import PrettyTable, ALL
from llama_cpp import Llama

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv('YOUTUBE_API_KEY')

if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in environment variables. Please create a .env file with your API key.")

# Email Configuration
EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USERNAME = os.getenv('EMAIL_USERNAME', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
EMAIL_TO = os.getenv('EMAIL_TO', 'vucibatina@hotmail.com,vucibatina@gmail.com')

# Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Configuration
FETCH_TRANSCRIPTS = True  # Set to True to fetch transcripts (may hit IP limits)
DAYS_FILTER = 3  # Only fetch transcripts for videos newer than this many days
MAX_VIDEOS_PER_CHANNEL = 3  # Maximum number of videos to process per channel
LLAMA_MODEL_PATH = "/Users/vuk/projects/david_fast_api_backup/david_fast_api/llama_models/llama-2-7b-chat-hf-q4_k_m.gguf"

# Initialize Llama model (load once at startup) - only if transcripts are enabled
if FETCH_TRANSCRIPTS:
    print("Loading Llama model...")
    llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=8192, n_threads=4)  # Increased context window for longer summaries
    print("Llama model loaded!")
else:
    llm = None
    print("Transcript fetching disabled - Llama model not loaded")

# List of YouTube channel IDs or handles
# Format: @username for handles
CHANNELS_BAK = [
    '@MarkMoss',
    '@LiamOttley',
    '@HeresyFinancial',
    '@RaoulPal',
    '@OrionTaraban',
    '@DavidBayer',
    '@ClearValueTax',
    '@DavidSnyderNLP',
    '@DejanBeric',
    '@BenAzadi',
    '@Kaggle',
    '@TomBilyeu',
    '@JessicaOs',
    '@ViktorJeremic',
    '@EFL',
    '@TheDotPot',
    '@TealSwan',
    '@MilitarySummary',
    '@SimplyBitcoin',
    '@XOFruit',
    '@TuckerCarlson',
    '@KatieClarke',
    '@Fireship',
    '@THEGRIM',
    '@EntrepreneursinCars',
    '@Drberg',
    '@JOSEMORALEJO',
    '@SpotlightTV',
    '@PowerfulJRE',
    '@JulianGoldieSEO',
    '@BlicTV',
    '@OliviaAlexa',
    '@HelmCast',
    '@Pivokosa',
    '@WeebUnion',
    '@UFDTech',
    '@CaseyZander',
    '@AIEngineer',
    '@DrTraceyMarks',
    '@HusseinNasser',
    '@motivationaldoc',
    '@RenaMalikMD',
    '@ReverseAgingRevolution',
    '@TheAIAdvantage',
    '@PeakProsperity',
    '@AIAnytime',
    '@AaronDoughty',
    '@TAGMEDIATV',
    '@HasanAboulHasan',
    '@TheDiaryOfACEO',
    '@dr.jamesdinicolantonio2215',
    '@ArthurMello',
    '@ScottRitter',
    '@TripodProduction',
    '@RuhiCenetDocumentaries',
    '@TinyTechnicalTutorials',
    '@SystemDesignFightClub',
    '@RobMulla',
    '@AIFoundations',
]

# Health & Wellness + Personal Development & Business + Finance channels
CHANNELS = [
    # Health & Wellness (7 channels)
    '@BenAzadi',
    '@Drberg',
    '@DrTraceyMarks',
    '@motivationaldoc',
    '@RenaMalikMD',
    '@ReverseAgingRevolution',
    '@dr.jamesdinicolantonio2215',

    # Personal Development & Business (6 channels)
    '@TomBilyeu',
    '@TheDiaryOfACEO',

    # Finance & Economics (3 channels)
    '@1MarkMoss',
    '@HeresyFinancial',
    '@RaoulPalTJM',

    # NLP & Psychology (1 channel)
    '@DavidSnyderNLP',

    '@DavidOndrej',
    '@wimhof1',
    '@BryanJohnson',
    '@DrGundry',
    '@DoctorMike',
    '@BobbyParrish',
    '@GlucoseRevolution',
    '@diabe_tech',
]

# Full channel list (EFL and TheDotPot removed) - Uncomment when ready
CHANNELS_FULL = [
    '@MarkMoss',
    '@LiamOttley',
    '@HeresyFinancial',
    '@RaoulPal',
    '@OrionTaraban',
    '@DavidBayer',
    '@ClearValueTax',
    '@DavidSnyderNLP',
    '@DejanBeric',
    '@BenAzadi',
    '@Kaggle',
    '@TomBilyeu',
    '@JessicaOs',
    '@ViktorJeremic',
    '@TealSwan',
    '@MilitarySummary',
    '@SimplyBitcoin',
    '@XOFruit',
    '@TuckerCarlson',
    '@KatieClarke',
    '@Fireship',
    '@THEGRIM',
    '@EntrepreneursinCars',
    '@Drberg',
    '@JOSEMORALEJO',
    '@SpotlightTV',
    '@PowerfulJRE',
    '@JulianGoldieSEO',
    '@BlicTV',
    '@OliviaAlexa',
    '@HelmCast',
    '@Pivokosa',
    '@WeebUnion',
    '@UFDTech',
    '@CaseyZander',
    '@AIEngineer',
    '@DrTraceyMarks',
    '@HusseinNasser',
    '@motivationaldoc',
    '@RenaMalikMD',
    '@ReverseAgingRevolution',
    '@TheAIAdvantage',
    '@PeakProsperity',
    '@AIAnytime',
    '@AaronDoughty',
    '@TAGMEDIATV',
    '@HasanAboulHasan',
    '@TheDiaryOfACEO',
    '@dr.jamesdinicolantonio2215',
    '@ArthurMello',
    '@ScottRitter',
    '@TripodProduction',
    '@RuhiCenetDocumentaries',
    '@TinyTechnicalTutorials',
    '@SystemDesignFightClub',
    '@RobMulla',
    '@AIFoundations',
]


def get_video_transcript(video_id):
    """
    Fetch the transcript for a YouTube video.

    Args:
        video_id: YouTube video ID

    Returns:
        Tuple of (transcript_text, error_type) where error_type is None if successful,
        or a string describing the error ('ip_blocked', 'no_transcript', 'disabled', 'error')
    """
    try:
        # Create API instance and get list of available transcripts
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Try to find English transcript (manual or auto-generated)
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If English not found, try to find any manually created transcript
            transcript = transcript_list.find_generated_transcript(['en'])

        # Fetch the actual transcript data
        transcript_data = transcript.fetch()

        # Combine all transcript segments into one text
        transcript_text = ' '.join([entry.text for entry in transcript_data])
        return (transcript_text, None)
    except TranscriptsDisabled:
        return (None, 'disabled')
    except NoTranscriptFound:
        return (None, 'no_transcript')
    except Exception as e:
        # Check if it's an IP block or other error
        error_msg = str(e)
        error_msg_lower = error_msg.lower()
        if 'too many requests' in error_msg_lower or '429' in error_msg_lower:
            return (None, 'ip_blocked')
        return (None, f'error: {error_msg[:50]}')


def save_transcript(video_id, channel_title, published_at, title, transcript_text):
    """Save transcript to data/<video_id>.txt with metadata header."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{video_id}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"{channel_title}\n")
        f.write(f"{published_at}\n")
        f.write(f"{title}\n")
        f.write(f"{transcript_text}\n")


def summarize_transcript(transcript_text, max_words=None):
    """
    Summarize a video transcript using local Llama model.

    Args:
        transcript_text: Full transcript text
        max_words: Maximum words for summary (if None, calculated as ~20% of transcript length)

    Returns:
        Summary string or error message
    """
    try:
        # Calculate dynamic summary length based on transcript length (~20% = 1/5th)
        if max_words is None:
            transcript_word_count = len(transcript_text.split())
            # Aim for 20% (1/5th) of original length, minimum 500, maximum 5000 words
            max_words = min(5000, max(500, int(transcript_word_count * 0.2)))

        # Truncate transcript if too long (to fit in context window)
        max_transcript_chars = 8000
        if len(transcript_text) > max_transcript_chars:
            transcript_text = transcript_text[:max_transcript_chars] + "..."

        prompt = f"""Summarize the following YouTube video transcript using bullet points. The transcript is approximately {max_words * 5} characters — scale the number and depth of bullet points to match: longer transcripts require more bullets and more detail per bullet.

CRITICAL FORMATTING RULES - YOU MUST FOLLOW THESE EXACTLY:

1. OPENING: DO NOT start with "The speaker discusses", "The video discusses", "In this video", or any similar filler phrase. Jump straight into the content and topic itself.

2. BULLET POINTS (MANDATORY FORMAT):
   - Write the entire summary as bullet points using "•"
   - Each bullet point should be a self-contained insight, fact, or recommendation
   - Longer transcripts = more bullet points and more detailed bullets
   - Sub-bullets (  ◦) for nested details, examples, or elaborations

3. ITEMIZATION (ABSOLUTELY MANDATORY): When the speaker mentions ANY numbered points, laws, steps, rules, principles, tips, strategies, or lists:
   - YOU MUST INCLUDE THE COMPLETE LIST - DO NOT CUT IT SHORT
   - Put EACH item on a NEW LINE
   - Use clear numbering: 1), 2), 3), etc.
   - NEVER truncate lists - always show the complete list

4. PRACTICALITY (HIGH PRIORITY): Extract and highlight ALL actionable items:
   - Specific stocks, cryptocurrencies, or assets to buy/sell
   - Trading strategies with entry/exit points
   - Foods, supplements, or products to consume/avoid
   - Step-by-step instructions (format as numbered list)
   - Tools, resources, or techniques mentioned
   - Specific recommendations or advice
   - Complete dosages, amounts, or measurements

5. COMPLETENESS: Do NOT cut off mid-sentence. Complete all thoughts and lists fully.

Transcript:
{transcript_text}

Summary:"""

        response = llm(
            prompt,
            max_tokens=4096,  # Increased to accommodate longer, complete summaries
            temperature=0.7,
            stop=["Transcript:", "\n\n\n\n"],
            echo=False
        )

        summary = response['choices'][0]['text'].strip()
        return summary if summary else "[Unable to generate summary]"

    except Exception as e:
        return f"[Error summarizing: {str(e)[:50]}]"


def send_email_report(report_file_path, days_filter, all_videos_data):
    """
    Send the YouTube video summary report via email with file attachment.

    Args:
        report_file_path: Path to the report file to attach
        days_filter: Number of days covered in the report
        all_videos_data: List of tuples (channel_title, video_dict) for all videos

    Returns:
        Boolean indicating success or failure
    """
    if not EMAIL_ENABLED:
        print("Email sending is disabled. Set EMAIL_ENABLED=True in .env to enable.")
        return False

    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        print("Email credentials not configured. Please set EMAIL_USERNAME and EMAIL_PASSWORD in .env file.")
        return False

    # Parse email recipients (comma-separated)
    email_recipients = [email.strip() for email in EMAIL_TO.split(',')]

    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_USERNAME
        msg['To'] = ', '.join(email_recipients)
        msg['Subject'] = f"YouTube Video Report - Past {days_filter} Days"

        # Build HTML email body with video information
        html_body = f"""<html>
<head></head>
<body>
<p>YouTube Video Summary Report for the past {days_filter} days:</p>
<br>
"""

        for channel_title, video in all_videos_data:
            youtube_url = f"https://www.youtube.com/watch?v={video['video_id']}"
            video_title = video['title']
            summary = video.get('summary', '[Transcripts disabled]')
            # Convert newlines to <br> tags for HTML formatting
            summary_html = summary.replace('\n', '<br>\n')

            html_body += f"""<p><strong>{channel_title}</strong><br>
<a href="{youtube_url}">{video_title}</a><br>
{summary_html}</p>

"""

        html_body += f"""<br>
<p><em>Full report attached: {os.path.basename(report_file_path)}</em></p>
</body>
</html>"""

        # Attach HTML body
        msg.attach(MIMEText(html_body, 'html'))

        # Attach the report file
        with open(report_file_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(report_file_path)}'
            )
            msg.attach(part)

        # Connect to SMTP server and send email
        print(f"\nSending email report to {len(email_recipients)} recipient(s): {', '.join(email_recipients)}...")
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()  # Enable TLS encryption
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"✓ Email sent successfully to: {', '.join(email_recipients)}")
        return True

    except Exception as e:
        print(f"✗ Failed to send email: {str(e)}")
        return False


def get_channel_id(channel_identifier):
    """
    Convert a channel handle (@username) or return channel ID if already provided.

    Args:
        channel_identifier: Channel ID (UC...) or handle (@username)

    Returns:
        Channel ID string
    """
    if channel_identifier.startswith('@'):
        try:
            # Search for channel by handle
            request = youtube.search().list(
                part='snippet',
                q=channel_identifier,
                type='channel',
                maxResults=1
            )
            response = request.execute()

            if response['items']:
                return response['items'][0]['snippet']['channelId']
            else:
                return None
        except HttpError as e:
            return None
    else:
        return channel_identifier


def get_channel_videos(channel_id, max_results=50, days_filter=3):
    """
    Fetch videos from a YouTube channel, filtering out Shorts and old videos.

    Args:
        channel_id: YouTube channel ID
        max_results: Maximum number of videos to fetch (default 50)
        days_filter: Only include videos from the past N days (default 3)

    Returns:
        Tuple of (channel_title, list of recent video dictionaries)
    """
    videos = []
    channel_title = None
    cutoff_date = datetime.now() - timedelta(days=days_filter)

    try:
        # Get the channel's uploads playlist ID
        channel_response = youtube.channels().list(
            part='contentDetails,snippet',
            id=channel_id
        ).execute()

        if not channel_response['items']:
            return (None, [])

        channel_title = channel_response['items'][0]['snippet']['title']
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # Fetch videos from the uploads playlist
        next_page_token = None

        while len(videos) < max_results:
            playlist_response = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=uploads_playlist_id,
                maxResults=min(50, max_results - len(videos)),
                pageToken=next_page_token
            ).execute()

            video_ids = [item['contentDetails']['videoId'] for item in playlist_response['items']]

            # Get video details to filter out Shorts
            videos_response = youtube.videos().list(
                part='snippet,contentDetails',
                id=','.join(video_ids)
            ).execute()

            for video in videos_response['items']:
                # Skip videos without duration (deleted, private, or live streams)
                if 'duration' not in video.get('contentDetails', {}):
                    continue

                # Parse duration to filter out Shorts (typically under 60 seconds)
                duration = video['contentDetails']['duration']

                # Convert ISO 8601 duration to seconds
                is_long_video = 'H' in duration or 'M' in duration or (
                    'S' in duration and int(duration.split('PT')[1].split('S')[0]) > 61
                )

                if is_long_video:
                    title = video['snippet']['title']
                    published_at = video['snippet']['publishedAt']

                    # Parse and format the date
                    date_obj = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')

                    # Only include videos from the past N days
                    if date_obj >= cutoff_date:
                        formatted_date = date_obj.strftime('%Y-%b-%d')

                        videos.append({
                            'title': title,
                            'published_at': formatted_date,
                            'video_id': video['id'],
                            'date_obj': date_obj
                        })

            next_page_token = playlist_response.get('nextPageToken')

            if not next_page_token:
                break

    except HttpError as e:
        pass

    return (channel_title, videos)


def main():
    """Main function to process all channels and display videos from past 3 days.

    Returns:
        List of tuples (channel_title, video_dict) for all videos found
    """

    print(f"Fetching videos from the past {DAYS_FILTER} days...\n")

    channels_with_videos = 0
    total_videos = 0
    all_videos_data = []  # Collect all videos with channel info for email

    for channel_identifier in CHANNELS:
        channel_id = get_channel_id(channel_identifier)

        if channel_id:
            channel_title, videos = get_channel_videos(channel_id, days_filter=DAYS_FILTER)

            # Only display channels that have videos in the past N days
            if videos and channel_title:
                channels_with_videos += 1

                # Sort videos by date (most recent first)
                sorted_videos = sorted(videos, key=lambda x: x['date_obj'], reverse=True)

                # Limit to maximum number of videos per channel
                if len(sorted_videos) > MAX_VIDEOS_PER_CHANNEL:
                    print(f"Note: {channel_title} has {len(sorted_videos)} videos, limiting to {MAX_VIDEOS_PER_CHANNEL} most recent")
                    sorted_videos = sorted_videos[:MAX_VIDEOS_PER_CHANNEL]

                total_videos += len(sorted_videos)

                # Print channel name as header
                print("=" * 120)
                print(f"  {channel_title}")
                print("=" * 120)

                # Fetch transcripts and summaries for each video
                if FETCH_TRANSCRIPTS:
                    print(f"Fetching transcripts and generating summaries for {len(sorted_videos)} videos...")
                    for video in sorted_videos:
                        transcript, error = get_video_transcript(video['video_id'])
                        if transcript:
                            save_transcript(video['video_id'], channel_title, video['published_at'], video['title'], transcript)
                            print(f"  Summarizing: {video['title'][:60]}...")
                            video['summary'] = summarize_transcript(transcript)
                        else:
                            video['summary'] = f"[No transcript: {error}]"
                        time.sleep(4)  # Rate limiting - increased delay to avoid IP blocking
                    print()

                # Collect videos for email body
                for video in sorted_videos:
                    all_videos_data.append((channel_title, video))

                # Create table for this channel's videos with line separation between rows
                table = PrettyTable()
                table.field_names = ["YouTube Link", "Date", "Title", "Summary"]
                table.align["YouTube Link"] = "l"
                table.align["Date"] = "c"
                table.align["Title"] = "l"
                table.align["Summary"] = "l"
                table.max_width["YouTube Link"] = 45
                table.max_width["Title"] = 40
                table.max_width["Summary"] = 200  # Increased from 70 to allow full summaries with complete lists
                table.hrules = ALL  # Add horizontal rules between all rows for clear separation

                # Add rows to table
                for video in sorted_videos:
                    youtube_url = f"https://www.youtube.com/watch?v={video['video_id']}"
                    table.add_row([
                        youtube_url,
                        video['published_at'],
                        video['title'],
                        video.get('summary', '[Transcripts disabled]')
                    ])

                print(table)
                print()  # Empty line between channels

    # Summary
    print("=" * 80)
    print(f"Summary: Found {total_videos} videos from {channels_with_videos} channels in the past {DAYS_FILTER} days")
    print("=" * 80)

    return all_videos_data


if __name__ == '__main__':
    # Capture output to send via email
    output_capture = StringIO()

    # Redirect stdout to capture all print statements
    original_stdout = sys.stdout
    sys.stdout = output_capture

    try:
        # Run the main function and get video data
        all_videos_data = main()
    finally:
        # Restore original stdout
        sys.stdout = original_stdout

    # Get the captured output
    report_content = output_capture.getvalue()

    # Print to console
    print(report_content)

    # Generate filename with date format: YOUTUBE_VUK_[MMDDYYYY].txt
    today = datetime.now()
    filename = f"YOUTUBE_VUK_{today.strftime('%m%d%Y')}.txt"

    # Save report to file
    print(f"\nSaving report to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"✓ Report saved successfully to {filename}")

    # Send email report with file attachment and formatted body
    send_email_report(filename, DAYS_FILTER, all_videos_data)
