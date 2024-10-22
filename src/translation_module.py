import yaml
from tqdm import tqdm
from transformers import pipeline
from utils import save_to_srt
import re
# Load configuration from config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def translate_chunks_and_save_to_srt(chunks, output_srt, config_path="config.yaml"):
    # Load the configuration
    config = load_config(config_path)

    # Extract configuration parameters for readability
    model_config = config['translation_model']

    pipe = pipeline(
        "text-generation", 
        model=model_config['model_id'], 
        torch_dtype=model_config['torch_dtype'],
        device_map=model_config['device_map'],
    )
    pipe.tokenizer.padding_side = "left"

    # Prepare the batch of prompts
    prompts = [f"You are a professional tranlator to assist me to translate the following english subtitle into chinese: '{chunk['text']}'. Output your translation result in json, with key as 'text'. " for chunk in chunks]

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
                'timestamp': chunks[len(translated_chunks)]['timestamp'],
                'text': translated_text
            })
    
    # Save the translated chunks to an SRT file
    save_to_srt(translated_chunks, output_srt)
    print(f"Translated subtitles saved to {output_srt}")