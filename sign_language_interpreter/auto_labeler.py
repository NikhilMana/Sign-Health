import os
import re
import ffmpeg
from pathlib import Path
import string
import sys
import hashlib

def sanitize_folder_name(text):
    """
    Sanitize text to create valid folder names.
    Replaces spaces with underscores and removes invalid characters.
    """
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^a-z0-9_]', '', text)
    return text[:100] if text else 'unknown'
fdgdsrgsdfgfdsgdfsg
def parse_srt_file(srt_file_path):
    """
    Parse the .srt file and return a list of dictionaries.
    Each dictionary contains 'start_time', 'end_time', and 'text' for each subtitle entry.
    """
    print(f"Parsing subtitle file: {srt_file_path}")
    
    subtitle_data = []
    
    try:
        with open(srt_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        print("Warning: UTF-8 decoding failed, trying latin-1")
        with open(srt_file_path, 'r', encoding='latin-1') as file:
            content = file.read()
    
    subtitle_blocks = content.strip().split('\n\n')
    
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        
        if len(lines) >= 3:
            timing_line = lines[1]
            timing_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
            match = re.match(timing_pattern, timing_line)
            
            if match:
                start_time = match.group(1)
                end_time = match.group(2)
                text = ' '.join(lines[2:]).strip()
                
                if text:
                    subtitle_data.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': text,
                        'folder_name': sanitize_folder_name(text)
                    })
    
    print(f"Parsed {len(subtitle_data)} subtitle entries")
    return subtitle_data

def time_to_seconds(time_str):
    """
    Convert SRT time format (HH:MM:SS,mmm) to seconds.
    """
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0

def create_video_clip(input_video, start_time, end_time, output_path):
    """
    Use ffmpeg to trim the input video and create a new clip.
    """
    try:
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        duration = end_seconds - start_seconds
        
        if duration <= 0:
            print(f"Invalid duration: {duration}s")
            return False
        
        (
            ffmpeg
            .input(input_video, ss=start_seconds, t=duration)
            .output(output_path, vcodec='copy', acodec='copy', avoid_negative_ts='make_zero')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main(input_video=None, input_subtitle=None, output_directory='videos'):
    """
    Main function that orchestrates the automatic video labeling process.
    """
    print("Automatic Video Labeler")
    print("=" * 30)
    
    if not input_video or not input_subtitle:
        input_video = input("Enter video path (or press Enter for 'downloaded_videos/input_video.mp4'): ").strip()
        if not input_video:
            input_video = 'downloaded_videos/input_video.mp4'
        
        input_subtitle = input("Enter subtitle path (or press Enter for 'downloaded_videos/input_video.en.srt'): ").strip()
        if not input_subtitle:
            input_subtitle = 'downloaded_videos/input_video.en.srt'
    
    if not os.path.exists(input_video):
        print(f"Error: Input video file '{input_video}' not found!")
        return
    
    if not os.path.exists(input_subtitle):
        print(f"Error: Input subtitle file '{input_subtitle}' not found!")
        return
    
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    subtitle_data = parse_srt_file(input_subtitle)
    
    if not subtitle_data:
        print("No subtitle data found!")
        return
    
    print(f"\nProcessing {len(subtitle_data)} subtitle entries...")
    
    success_count = 0
    fail_count = 0
    
    for i, entry in enumerate(subtitle_data, 1):
        folder_name = entry['folder_name']
        text = entry['text']
        
        print(f"\n[{i}/{len(subtitle_data)}] {text[:50]}...")
        
        label_directory = Path(output_directory) / folder_name
        label_directory.mkdir(parents=True, exist_ok=True)
        
        file_count = len(list(label_directory.glob("*.mp4")))
        output_path = label_directory / f"{file_count}.mp4"
        
        if create_video_clip(input_video, entry['start_time'], entry['end_time'], str(output_path)):
            success_count += 1
            print(f"✓ Saved to {folder_name}/{file_count}.mp4")
        else:
            fail_count += 1
            print(f"✗ Failed")
    
    print(f"\n{'='*30}")
    print(f"Completed: {success_count} success, {fail_count} failed")
    print(f"Output directory: '{output_directory}'")



if __name__ == "__main__":
    ffmpeg_bin_path = r"C:\ffmpeg\bin"
    if os.path.isdir(ffmpeg_bin_path) and ffmpeg_bin_path not in os.environ['PATH']:
        os.environ['PATH'] = ffmpeg_bin_path + os.pathsep + os.environ['PATH']

    try:
        ffmpeg.probe('dummy')
    except FileNotFoundError:
        print("Error: ffmpeg not found in PATH")
        print("Download from: https://ffmpeg.org/download.html")
        sys.exit(1)
    except ffmpeg.Error:
        pass

    main()
