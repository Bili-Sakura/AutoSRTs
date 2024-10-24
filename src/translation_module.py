import yaml
from tqdm import tqdm
from transformers import pipeline
from utils import save_to_srt
import re

def calculate_time_ratio(segment, segment_word_count, words_used):
    """
    Calculate the adjusted time based on the number of words used in the ASR segment.
    """
    start_time, end_time = segment['timestamp']
    duration = end_time - start_time
    time_per_word = duration / segment_word_count
    return words_used * time_per_word

def resentence(segments):
    """
    Concatenate segments into sentences based on punctuation found within the segments.
    """
    restructured_segments = []
    sentence_segments = []
    sentence_start_time = None

    # Define sentence-ending punctuation
    sentence_endings = re.compile(r'[.!?。！？]$')  # Matches if text ends with punctuation

    for segment in segments:
        text = segment['text']
        start_time, end_time = segment['timestamp']

        # If starting a new sentence, set the start time
        if not sentence_segments:
            sentence_start_time = start_time

        sentence_segments.append(segment)

        # Check if the current segment ends with sentence-ending punctuation
        if sentence_endings.search(text):
            # Concatenate the texts of the segments to form the sentence
            sentence_text = ' '.join(s['text'] for s in sentence_segments).strip()

            # The sentence ends here; create a new segment
            restructured_segments.append({
                'timestamp': (sentence_start_time, end_time),
                'text': sentence_text
            })

            # Reset for the next sentence
            sentence_segments = []
            sentence_start_time = None

    # Handle any remaining segments that didn't end with punctuation
    if sentence_segments:
        sentence_text = ' '.join(s['text'] for s in sentence_segments).strip()
        end_time = sentence_segments[-1]['timestamp'][1]
        restructured_segments.append({
            'timestamp': (sentence_start_time, end_time),
            'text': sentence_text
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

def translate_chunks_and_save_to_srt(formatted_chunks,prompt_template,summary,configs):

    # Extract configuration parameters for readability
    model_config = configs['translation_model']

    pipe = pipeline(
        "text-generation", 
        model=model_config['model_path'], 
        torch_dtype=model_config['torch_dtype'],
        device_map=model_config['device_map'],
    )
    pipe.tokenizer.padding_side = "left"
    prompts=[]
    for chunk in formatted_chunks:
        formatted_prompt = prompt_template.format(
            source_language=model_config['source_language'],
            target_language=model_config['target_language'],
            summary=summary,
            user_defined_rules=model_config['user_defined_rules'],
            source_subtitle=chunk["text"]
        )
        prompts.append(formatted_prompt)

    # Translate each chunk in batch and save to an SRT file
    translated_chunks = []
    for idx in tqdm(range(0, len(prompts), model_config['batch_size']), desc="Translation"):
        batch_prompts = prompts[idx:idx + model_config['batch_size']]
        batch_results = pipe(
            batch_prompts, 
        max_new_tokens=model_config['max_length'], 
        batch_size=model_config['batch_size'],
        temperature=0.001
        )
        for result in batch_results:
            match = re.search(r'"text":\s*"([^"]+)"', result[0]["generated_text"])
            if match:
                translated_text = match.group(1)
                print(f"translated_text:{translated_text}")
            else:
                print("No match found")
            translated_chunks.append({
                'timestamp': formatted_chunks[len(translated_chunks)]['timestamp'],
                'text': translated_text
            })
    
    # Save the translated chunks to an SRT file
    save_to_srt(translated_chunks, configs['translated_srt'])
    print(f"Translated subtitles saved")
    save_to_srt(formatted_chunks, configs['output_srt'])
    print(f"Subtitles saved")
