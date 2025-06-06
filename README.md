# Video Subtitle Tool

A Python tool that can extract subtitles from videos using OpenAI's Whisper and translate existing subtitle files using Google Translate API.

## Features

1. Extract subtitles from video files using OpenAI's Whisper
   - Real-time progress monitoring during transcription
   - Support for custom model directory
2. Translate existing SRT subtitle files to different languages

## Requirements

- Python 3.6 or higher
- FFmpeg installed on your system
- Required packages (install using `pip install -r requirements.txt`):
  - googletrans==3.1.0a0
  - openai-whisper
  - torch
  - ffmpeg-python

## Installation

1. Install FFmpeg:
   - On macOS: `brew install ffmpeg`
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Extract Subtitles from Video

```bash
python srt_translator.py extract video_file.mp4 output.srt [--model MODEL_SIZE] [--model-dir MODEL_DIRECTORY]
```

Arguments:
- `video_file.mp4`: Path to the input video file
- `output.srt`: Path where the SRT file will be saved
- `--model`: (Optional) Whisper model size to use. Choices: tiny, base, small, medium, large (default: base)
- `--model-dir`: (Optional) Custom directory to store/load Whisper models

Examples:
```bash
# Use default model directory
python srt_translator.py extract movie.mp4 subtitles.srt --model base

# Use custom model directory
python srt_translator.py extract movie.mp4 subtitles.srt --model base --model-dir /path/to/models
```

The transcription process will show real-time progress, including:
- Percentage complete
- Number of segments processed
- Elapsed time

### Translate Existing Subtitles

```bash
python srt_translator.py translate input.srt output.srt
```

Arguments:
- `input.srt`: Path to the input SRT file
- `output.srt`: Path where the translated SRT file will be saved

Example:
```bash
python srt_translator.py translate subtitles.srt translated_subtitles.srt
```

## Model Storage

By default, Whisper models are stored in:
- macOS/Linux: `~/.cache/whisper/`
- Windows: `C:\Users\<username>\.cache\whisper\`

You can specify a custom directory using the `--model-dir` option. The directory should have write permissions as the models will be downloaded there if they don't exist.

## Whisper Model Sizes

- `tiny`: ~75MB, fastest, least accurate
- `base`: ~1GB, good balance of speed and accuracy
- `small`: ~2GB, better accuracy, slower
- `medium`: ~5GB, high accuracy, slower
- `large`: ~10GB, best accuracy, slowest

Choose the model size based on your needs for accuracy vs. processing speed and available disk space.

## Supported Translation Languages

Some common language codes:
- 'en' - English
- 'es' - Spanish
- 'fr' - French
- 'de' - German
- 'it' - Italian
- 'pt' - Portuguese
- 'ru' - Russian
- 'ja' - Japanese
- 'ko' - Korean
- 'zh-cn' - Chinese (Simplified)

For a complete list of supported language codes, refer to the [Google Translate documentation](https://cloud.google.com/translate/docs/languages).