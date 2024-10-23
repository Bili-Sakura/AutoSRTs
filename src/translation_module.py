import yaml
from tqdm import tqdm
from transformers import pipeline
from utils import save_to_srt
import re




def resentence(text, chunks):
    # Step 1: Split the full subtitle text into sentences using '.'
    sentences = text.split('.')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]  # Clean up and add back '.'

    re_chunks = []
    sentence_idx = 0  # To track which sentence we're processing
    
    # Step 2: Iterate over each chunk
    for chunk in chunks:
        subtitle_seg = chunk['text']  # Original chunk text (could contain multiple sentences)
        start_time = chunk['timestamp'][0]  # Start time of the chunk
        end_time = chunk['timestamp'][1]    # End time of the chunk
        chunk_duration = end_time - start_time  # Duration of this chunk
        
        # Step 3: Estimate how to split time proportionally for each sentence in this chunk
        # Count total words in the chunk to distribute time proportionally
        words_in_chunk = subtitle_seg.split(' ')
        total_words = len(words_in_chunk)
        
        # Initialize current_time to start of chunk
        current_time = start_time
        
        # Continue while there are sentences left to process and we are still in this chunk
        while sentence_idx < len(sentences):
            sentence = sentences[sentence_idx]  # Get the current sentence
            
            # Count words in the sentence to allocate proportional time
            words_in_sentence = sentence.split(' ')
            sentence_word_count = len(words_in_sentence)
            
            # Calculate the proportional time for this sentence based on word count
            sentence_duration = chunk_duration * (sentence_word_count / total_words)
            
            # Create a new re_chunk for this sentence
            re_chunk = {
                'text': sentence,
                'timestamp': [current_time, current_time + sentence_duration]
            }
            re_chunks.append(re_chunk)
            
            # Update current_time for the next sentence
            current_time += sentence_duration
            sentence_idx += 1  # Move to the next sentence
            
            # If the current sentence used up all the time in this chunk, break out of the loop
            if current_time >= end_time:
                break

    return re_chunks




def summarize(text,configs):
    model_config = configs['translation_model']

    pipe = pipeline(
        "text-generation", 
        model=model_config['model_path'], 
        # torch_dtype=model_config['torch_dtype'],
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
        model=model_config['model_id'], 
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
        batch_results = pipe(batch_prompts, max_new_tokens=512, batch_size=model_config['batch_size'],temperature=0.001)
        for result in batch_results:
            match = re.search(r'"text":\s*"([^"]+)"', result[0]["generated_text"])
            if match:
                translated_text = match.group(1)
                print(translated_text)
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
