# ASR and Translation Subtitles Project

This project generates subtitles from audio using Automatic Speech Recognition (ASR) with Whisper and translates them using the LLMs.

<!-- ## Project Structure
- `src/`: Contains the main source code for ASR, translation, and subtitle generation.
- `utils/`: Utility functions for the project.
- `tests/`: Unit tests for the project.
- `docs/`: Documentation for the project.

## Setup
Install the dependencies:
```
pip install -r requirements.txt
```

Run the main script:
```
python src/main.py example_audio.mp3 example_audio.srt zh
``` -->

## To-Do List

- Timestamp (under chunked mode) occasionally to be not correct, check and fix if possible.
- Sentence divisions (under chunked mode) are not good, try to refined it with LLMs.
- Translation results are bad given with small model.
- GUI required (considier to use streamlit for deployment).
- Show the process with log files.
- Inference accelaration (batch inference and so on).