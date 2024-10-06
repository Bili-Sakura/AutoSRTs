import yaml
from tqdm import tqdm
from transformers import pipeline
from utils import save_to_srt

# Load configuration from config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def translate_text(text, config_path="config.yaml"):
    # Load the configuration
    config = load_config(config_path)

    # Extract configuration parameters for readability
    model_config = config['translation_model']


    translation_pipe = pipeline(
        "translation", 
        model=model_config['model_id'], 
        torch_dtype=model_config['torch_dtype'],
        device_map=model_config['device_map'],
        )

    # Perform translation
    result = translation_pipe(text, max_length=model_config.get('max_length', 256))

    # Return the translated text
    return result[0]['translation_text']


def translate_chunks_and_save_to_srt(chunks, output_srt, config_path="config.yaml"):
    # Translate each chunk and save to an SRT file
    translated_chunks = []
    for idx in tqdm(range(len(chunks)), desc="Translation"):
        translated_text = translate_text(chunks[idx]['text'], config_path)
        translated_chunks.append({
            'timestamp': chunks[idx]['timestamp'],
            'text': translated_text
        })
    
    # Save the translated chunks to an SRT file

    save_to_srt(translated_chunks, output_srt)
    print(f"Translated subtitles saved to {output_srt}")

