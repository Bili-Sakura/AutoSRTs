import yaml
from tqdm import tqdm
from transformers import pipeline
from utils import save_to_srt
import re
from datetime import timedelta

def format_srt_time(td):
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    
def calculate_time_ratio(segment, segment_word_count, words_used):
    """
    Calculate the adjusted time based on the number of words used in the ASR segment.
    """
    start_time, end_time = segment['timestamp']
    duration = end_time - start_time
    time_per_word = duration / segment_word_count
    return words_used * time_per_word

def split_long_sentence(sentence, word_count, max_words_per_sentence):
    """
    Split a long sentence into subparts if it exceeds the maximum allowed words.
    
    Args:
    - sentence (str): The sentence to be split.
    - word_count (int): Total number of words in the sentence.
    - max_words_per_sentence (int): Maximum allowed words per sentence.
    
    Returns:
    - List of sentence subparts.
    """
    words = sentence.split()
    if word_count <= max_words_per_sentence:
        return [sentence]
    
    # Split sentence into subparts of max_words_per_sentence words
    subparts = [' '.join(words[i:i + max_words_per_sentence]) for i in range(0, len(words), max_words_per_sentence)]
    return subparts

def resentence(segments, max_length=160):
    """
    Concatenate segments into sentences based on punctuation and maximum length.
    Restart sentence when a sentence-ending punctuation is found ('.', '!', '?'),
    or when accumulated length exceeds max_length. If exceeding max_length, try
    to split at ','.
    """
    restructured_segments = []
    current_sentence_segments = []
    current_sentence_start_time = None
    current_sentence_text = ''
    accumulated_length = 0  # Length of the current sentence in characters

    # Define sentence-ending punctuation
    sentence_endings = re.compile(r'[.!?]')
    
    for segment in segments:
        text = segment['text'].rstrip()  # Remove trailing spaces
        start_time, end_time = segment['timestamp']
        
        # Remove leading spaces to avoid extra spaces when concatenating
        text = text.lstrip()
        
        # If starting a new sentence, set the start time
        if not current_sentence_segments:
            current_sentence_start_time = start_time
        
        current_sentence_segments.append(segment)
        # Add a space if needed
        if current_sentence_text:
            current_sentence_text += ' ' + text
        else:
            current_sentence_text = text
        
        accumulated_length = len(current_sentence_text)
        
        # Check if segment contains sentence-ending punctuation
        end_punctuation_found = sentence_endings.search(text)
        
        # Check if accumulated length exceeds max_length
        length_exceeded = accumulated_length >= max_length
        
        if end_punctuation_found or length_exceeded:
            split_indices = []
            sentence_text = current_sentence_text

            # If length exceeded, try to split at last comma before max_length
            if length_exceeded:
                last_comma_index = current_sentence_text.rfind(',', 0, max_length)
                if last_comma_index != -1:
                    split_index = last_comma_index + 1  # Include the comma
                    sentence_text = current_sentence_text[:split_index].strip()
                    remaining_text = current_sentence_text[split_index:].strip()
                else:
                    # No comma found, split at current segment
                    sentence_text = current_sentence_text.strip()
                    remaining_text = ''
            else:
                remaining_text = ''

            # Determine end time for the current sentence
            cumulative_length = 0
            for i, seg in enumerate(current_sentence_segments):
                seg_text = seg['text'].strip()
                cumulative_length += len(seg_text) + (1 if i > 0 else 0)  # Add space if not first word
                if cumulative_length >= len(sentence_text):
                    end_time = seg['timestamp'][1]
                    break
            else:
                end_time = current_sentence_segments[-1]['timestamp'][1]
            
            # Append the sentence
            restructured_segments.append({
                'timestamp': (current_sentence_start_time, end_time),
                'text': sentence_text
            })
            
            # Prepare for next sentence
            if remaining_text:
                current_sentence_segments = current_sentence_segments[i+1:]  # Segments after the split
                current_sentence_start_time = current_sentence_segments[0]['timestamp'][0]
                current_sentence_text = remaining_text
                accumulated_length = len(current_sentence_text)
            else:
                current_sentence_segments = []
                current_sentence_start_time = None
                current_sentence_text = ''
                accumulated_length = 0
        # Else, continue accumulating segments
        
    # Handle any remaining text
    if current_sentence_segments:
        end_time = current_sentence_segments[-1]['timestamp'][1]
        restructured_segments.append({
            'timestamp': (current_sentence_start_time, end_time),
            'text': current_sentence_text
        })
        
    return restructured_segments






def summarize(text,configs):
    model_config = configs['translation_model']

    pipe = pipeline(
        "text-generation", 
        model=model_config['model_path'], 
        torch_dtype=model_config['torch_dtype'],
        device_map=model_config['device_map'],
    )
    pipe.tokenizer.padding_side = "left"
    prompt=f"summarize it and give result only:{text}"
    summary=pipe(prompt, max_new_tokens=512,temperature=0.001)
    print(f"summary:{summary}")
    return summary

def translate_chunks_and_save_to_srt(formatted_chunks, prompt_template, summary, configs):

    # Extract configuration parameters for readability
    model_config = configs['translation_model']

    # Initialize the text-generation pipeline
    pipe = pipeline(
        "text-generation", 
        model=model_config['model_path'], 
        torch_dtype=model_config['torch_dtype'],
        device_map=model_config['device_map'],
    )
    pipe.tokenizer.padding_side = "left"

    # Prepare prompts for each chunk
    prompts = []
    for chunk in formatted_chunks:
        print(chunk["text"])
        formatted_prompt = prompt_template.format(
            source_language=model_config['source_language'],
            target_language=model_config['target_language'],
            summary=summary,
            user_defined_rules=model_config['user_defined_rules'],
            source_subtitle=chunk["text"]
        )
        prompts.append(formatted_prompt)

    # Open the SRT files before starting the translation loop
    translated_srt_path = configs['translated_srt']
    output_srt_path = configs['output_srt']
    bilingual_srt_path = configs['bilingual_srt']  # New bilingual SRT path
    with open(translated_srt_path, 'w', encoding='utf-8') as translated_srt_file, \
         open(output_srt_path, 'w', encoding='utf-8') as output_srt_file, \
         open(bilingual_srt_path, 'w', encoding='utf-8') as bilingual_srt_file:

        srt_counter = 1  # SRT entry numbering

        # Translate each chunk in batches and save to SRT files incrementally
        for idx in tqdm(range(0, len(prompts), model_config['batch_size']), desc="Translation"):
            batch_prompts = prompts[idx:idx + model_config['batch_size']]
            batch_results = pipe(
                batch_prompts, 
                max_new_tokens=model_config['max_length'], 
                batch_size=model_config['batch_size'],
                temperature=0.001
            )
            for i, result in enumerate(batch_results):
                # Extract the translated text from the model's output
                match = re.search(r'"text":\s*"([^"]+)"', result[0]["generated_text"])
                if match:
                    translated_text = match.group(1)
                    print(f"translated_text: {translated_text}")
                else:
                    print("No match found")
                    translated_text = ''  # Handle cases where no match is found

                # Get the corresponding chunk and timestamps
                chunk_idx = idx + i
                chunk = formatted_chunks[chunk_idx]
                timestamp = chunk['timestamp']
                start = timestamp[0]
                end = timestamp[1]
                start_time = timedelta(seconds=start)
                end_time = timedelta(seconds=end)

                # Write the translated chunk to the translated SRT file
                translated_srt_file.write(f"{srt_counter}\n")
                translated_srt_file.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
                translated_srt_file.write(f"{translated_text}\n\n")

                # Write the original chunk to the output SRT file
                output_srt_file.write(f"{srt_counter}\n")
                output_srt_file.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
                output_srt_file.write(f"{chunk['text']}\n\n")

                # Write the bilingual chunk to the bilingual SRT file
                bilingual_srt_file.write(f"{srt_counter}\n")
                bilingual_srt_file.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
                bilingual_srt_file.write(f"{chunk['text']}\n")  # Original text
                bilingual_srt_file.write(f"{translated_text}\n\n")  # Translated text

                srt_counter += 1  # Increment the SRT numbering

    print("Translated subtitles saved.")
    print("Subtitles saved.")
    print("Bilingual subtitles saved.")

