import os
import re
from pathlib import Path
import string

def parse_srt_file(srt_file_path):
    """
    Parse the .srt file and return a list of dictionaries.
    Each dictionary contains 'start_time', 'end_time', and 'text' for each subtitle entry.
    The text is cleaned: converted to lowercase and stripped of punctuation.
    """
    print(f"Parsing subtitle file: {srt_file_path}")
    
    subtitle_data = []
    
    try:
        with open(srt_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(srt_file_path, 'r', encoding='latin-1') as file:
            content = file.read()
    
    # Split content by double newlines to get individual subtitle blocks
    subtitle_blocks = content.strip().split('\n\n')
    
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        
        if len(lines) >= 3:
            # Skip the first line (subtitle number)
            # Second line contains timing information
            timing_line = lines[1]
            
            # Parse timing information (format: 00:00:01,000 --> 00:00:03,000)
            timing_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
            match = re.match(timing_pattern, timing_line)
            
            if match:
                start_time = match.group(1)
                end_time = match.group(2)
                
                # Combine all text lines (lines 2 and onwards)
                text_lines = lines[2:]
                text = ' '.join(text_lines)
                
                # Clean the text: convert to lowercase and strip punctuation
                text = text.lower()
                text = text.translate(str.maketrans('', '', string.punctuation))
                text = text.strip()
                
                # Only add if text is not empty
                if text:
                    subtitle_data.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': text
                    })
    
    print(f"Parsed {len(subtitle_data)} subtitle entries")
    return subtitle_data

def main():
    """
    Test the auto labeler parsing functionality.
    """
    print("Testing Auto Labeler - SRT Parsing")
    print("=" * 40)
    
    # Define input paths
    input_subtitle = 'input_video.en.srt'  # Input subtitle file path
    
    # Check if input file exists
    if not os.path.exists(input_subtitle):
        print(f"Error: Input subtitle file '{input_subtitle}' not found!")
        return
    
    # Parse the subtitle file
    subtitle_data = parse_srt_file(input_subtitle)
    
    if not subtitle_data:
        print("No subtitle data found!")
        return
    
    print(f"\nFound {len(subtitle_data)} subtitle entries:")
    print("-" * 50)
    
    # Show first 10 entries as examples
    for i, entry in enumerate(subtitle_data[:10]):
        start_time = entry['start_time']
        end_time = entry['end_time']
        text = entry['text']
        
        print(f"{i+1:2d}. Time: {start_time} --> {end_time}")
        print(f"    Text: \"{text}\"")
        print()
    
    if len(subtitle_data) > 10:
        print(f"... and {len(subtitle_data) - 10} more entries")
    
    # Show unique labels that would be created
    unique_labels = set(entry['text'] for entry in subtitle_data)
    print(f"\nUnique labels that would be created: {len(unique_labels)}")
    print("-" * 50)
    
    for i, label in enumerate(sorted(unique_labels)[:20], 1):
        print(f"{i:2d}. {label}")
    
    if len(unique_labels) > 20:
        print(f"... and {len(unique_labels) - 20} more unique labels")
    
    print(f"\nTotal video clips that would be created: {len(subtitle_data)}")
    print(f"Unique label directories: {len(unique_labels)}")

if __name__ == "__main__":
    main()
