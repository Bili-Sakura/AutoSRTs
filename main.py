import yaml
from src import translate_chunks_and_save_to_srt,transcribe_audio,summarize,resentence
from utils import save_to_srt, save_to_txt
   

if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", 'r') as file:
        configs= yaml.safe_load(file)

    # Step1: Transcribe the audio file
    text, chunks = transcribe_audio(configs)

    # Step2: Reorganize chunks to form complete sentence with timestamp under chunked asr mode.
    formatted_chunks=resentence(text,chunks)

    # Step3: Define Translation Prompt Template

    prompt_template = """
    You are a translation expert. 
    Your task is to translate the given sentence in {source_language} into {target_language} one by one.
    Here is the topic summary of the whole subtitle: {summary}
    Here are some rules you should obey: {user_defined_rules}
    Your translation should be formatted in JSON, with the key as 'text'.
    Here is an example where we translate a subtitle from English into Chinese.
    Source subtitle: "Hello, welcome to my channel."
    Your answer should be:
    ```json
    "text": "你好，欢迎来到我的频道。"
    ```
    Now, here is the source subtitle you are going to translate: {source_subtitle}
    """

    # Step4: Summarize the topic
    summary=summarize(text,configs)
    

    # Step5: Do translation and save results in srt files

    translate_chunks_and_save_to_srt(formatted_chunks,prompt_template,summary,configs)




    