import yaml
from src import translate_chunks_and_save_to_srt,transcribe_audio
from utils import save_to_srt, save_to_txt

# Load configuration from config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

    

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    audio_file = config['main']['audio_file']
    output_srt = config['main']['output_srt']
    output_txt = config['main']['output_txt']
    translated_srt = config['main']['translated_srt']

    try:
        # Transcribe audio file
        text, chunks = transcribe_audio(audio_file, config_path="config.yaml")
        
        # Save the result to an SRT file
        save_to_srt(chunks, output_srt)
        print(f"Subtitle saved to {output_srt}")

        # Save the transcribed text to a TXT file
        save_to_txt(text, output_txt)
        print(f"Transcription text saved to {output_txt}")

        # Translate subtitles and save to translated SRT
        translate_chunks_and_save_to_srt(chunks, translated_srt, config_path="config.yaml")
        print(f"Translation saved to {translated_srt}")

    except Exception as e:
        print(f"An error occurred: {e}")

    