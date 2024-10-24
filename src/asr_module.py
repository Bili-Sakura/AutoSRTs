import tqdm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_torch_sdpa_available

def transcribe_audio(configs):

    model_config = configs['asr_model']

    # Load the pre-trained Whisper model from Hugging Face
    model_path = model_config['model_path']

    processor = AutoProcessor.from_pretrained(model_path)
    try:
        if model_config['flash_attention']:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                torch_dtype=model_config['torch_dtype'],
                device_map=model_config['device_map'],
                # low_cpu_mem_usage=True, 
                attn_implementation="flash_attention_2"
            )
        elif is_torch_sdpa_available():
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                device_map=model_config['device_map'],
                # low_cpu_mem_usage=True, 
                attn_implementation="sdpa"
            )
        else:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                device_map=model_config['device_map'],
            )
    except Exception as e:
        print(f"An error occurred while loading the model: {e}. Please check the model path and configuration.")
        raise


    # Create the ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps='word',
        batch_size=model_config['batch_size'],  
    )
    
    # Perform transcription on the audio file with parameters from config
    audio_file = configs["path_to_audio_file"]
    result = pipe([audio_file])

    del pipe
    torch.cuda.empty_cache()  # Clears the GPU memory cache


    # Return the transcribed text and chunk information
    return result[0]["text"], result[0]["chunks"]