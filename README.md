# ASR and Translation Subtitles Project

This project generates subtitles from audio using Automatic Speech Recognition (ASR) with Whisper and translates them using the LLMs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/pics/overview-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/pics/overview.svg">
  <img alt="Project Overview" src="./assets/pics/overview.svg">
</picture>

## Features

- **Automatic Speech Recognition (ASR):** Transcribes audio using the Whisper model, providing accurate text outputs for English.
- **Chunk Restructuring:** Automatically restructures ASR sub-segments into complete sentences, handling cases where sentences may be split or prematurely cut.
- **Proportional Timestamp Adjustment:** Ensures each restructured sentence is accurately timestamped based on the length of the words in the transcription.
- **Bilingual Subtitles:** Translates the transcribed sentences into the target language using Large Language Models (LLMs), handling technical and trending content with up-to-date world knowledge.
- **SRT Output:** Generates subtitles in SRT format, with each line containing both the original transcription and its translated counterpart.
- **Customizable Configurations:** Users can easily configure language pairs, model size, and processing parameters to suit their needs.
- **Efficient Chunk Processing:** Supports chunked processing for long audio files, with improvements in inference speed and sentence refinement.
- **Planned GUI:** A user-friendly interface, potentially built with Streamlit, is planned to make the process accessible without using the command line.

## Framework

### 1. **Input: Audio File**

- The user provides an audio file (e.g., `.wav`, `.mp3`) that serves as the input to the pipeline.

### 2. **Preprocessing**

- The audio is split into chunks if necessary, especially for longer files.
- Optional noise reduction or audio enhancement can be applied to improve transcription accuracy.

### 3. **ASR Module (Whisper Model)**

- The Whisper ASR model transcribes each chunk of audio into English text.
- The raw transcription may contain incomplete sentences due to chunk boundaries.

### 4. **Post-processing of ASR Output**

- **Restructuring:** The transcribed segments are restructured to form coherent, full sentences, even if they were split across chunks.
- **Timestamp Adjustment:** The duration of each sentence is recalculated based on the number of words in the original transcription, ensuring accurate timestamps.

### 5. **Translation using Large Language Models (LLMs)**

- The restructured transcription is translated into the target language using advanced LLMs, which can handle technical jargon and trending topics that smaller models might miss.
- This step ensures high-quality translation, particularly for technical content, such as new machine learning concepts.

### 6. **Post-processing of Translated Output**

- The translated sentences are aligned with the original transcription and timestamps.
- Both original and translated texts are formatted into bilingual SRT format, ensuring that each subtitle line contains both languages.

### 7. **Output: Bilingual SRT Subtitle**

- The final output is an SRT file that contains the original transcription and its corresponding translation, both aligned with accurate timestamps.

## To-Do List

- [ ] Timestamp (under chunked mode) occasionally to be not correct, check and fix if possible. 
- [x] Sentence divisions (under chunked mode) are not good, rematch them.
- [ ] Translation results compared with different LLMs.
- [ ] GUI required (considier to use `streamlit` for deployment).
- [ ] Show the process with log files.
- [x] Inference accelaration (batch inference and so on).
