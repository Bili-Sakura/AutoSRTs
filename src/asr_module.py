import tqdm
import yaml
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_torch_sdpa_available

print(is_torch_sdpa_available())


# Load configuration from config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def transcribe_audio(audio_file, config_path="config.yaml"):
    # Load the configuration
    config = load_config(config_path)

    # Extract configuration parameters for readability
    model_config = config['model']
    pipeline_config = config['pipeline']
    source_language = pipeline_config.get('source_language', None)

    # Load the pre-trained Whisper model from Hugging Face
    model_id = model_config['model_id']
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=model_config['torch_dtype'],
        device_map=model_config['device_map'],
        use_safetensors=model_config['use_safetensors'],
        attn_implementation="sdpa"
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # Create the ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
        generate_kwargs = {"language": source_language},
        chunk_length_s=30,
        batch_size=200,  
    )

    # Perform transcription on the audio file with parameters from config
    result = pipe(audio_file)

    # Return the transcribed text and chunk information
    return result["text"], result["chunks"]