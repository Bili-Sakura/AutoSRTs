from datetime import timedelta

# Function to save the transcription result to an SRT file
def save_to_srt(chunks, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:

        for idx, chunk in enumerate(chunks):
            start = chunk['timestamp'][0] 
            end = chunk['timestamp'][1] 
            start_time = timedelta(seconds=start)
            end_time = timedelta(seconds=end)      
            text = chunk['text']

            # Write in SRT format
            f.write(f"{idx + 1}\n")
            f.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            f.write(f"{text}\n\n")

# Function to save transcribed text to a TXT file
def save_to_txt(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

# Format time to SRT standard (hh:mm:ss,mmm)
def format_srt_time(td):
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

