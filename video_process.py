import argparse
from googletrans import Translator
import re
import whisper
import ffmpeg
import os
from datetime import timedelta
import time
import json
import openrouter
from openrouter import OpenRouter

def summarize_content(text, model="deepseek/deepseek-chat-v3-0324:free"):
    """
    Summarize content using OpenRouter API with Claude 3 Opus model.
    
    Args:
        text (str): Text content to summarize
        model (str): Model to use for summarization
        
    Returns:
        str: Summarized text
    """
    try:
        client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
        user_prompt = """
        这是一份讲授宏观经济学课程的录音文稿，请整理这份文稿，满足以下几个条件：
        1. 如果不是简体中文，转换成简体中文。
        2. 保持原来的语句和表达，仅仅去掉无意义的重复词语。
        3. 添加标点符号。
        4. 切分段落。
        5. 如果碰到公式，使用纯文本输出。
        请注意，最大限度忠于原文。
        """
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that processes content concisely."},
                {"role": "user", "content": f"{user_prompt}:\n\n{text}"}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return text  # Return original text if summarization fails

def process_subtitles_for_summary(srt_blocks):
    """
    Process SRT blocks and create a summary using OpenRouter.
    
    Args:
        srt_blocks (list): List of SrtBlock objects
        
    Returns:
        str: Summarized content
    """
    # Combine all subtitle text
    full_text = " ".join([block.text for block in srt_blocks])
    
    # Get summary using OpenRouter
    summary = summarize_content(full_text)
    
    return summary


class SrtBlock:
    def __init__(self, number, start_time, end_time, text, ts_start, ts_end):
        self.number = number
        self.start_time = start_time
        self.end_time = end_time
        self.ts_start = ts_start # in seconds
        self.ts_end = ts_end # in seconds
        self.text = text

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format."""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def extract_audio_from_video(video_path, audio_track=0):
    print(f"Extracting audio from video: {video_path}")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    try:
        # Create a temporary audio file
        # Create a temporary audio file with the same name prefix as the video file
        audio_file = f"{base_name}.mp3"
        track_map = f'0:a:{audio_track}'    # map='0:a:1' selects the second audio track
        # Use ffmpeg to extract audio from the video
        (
            ffmpeg
            .input(video_path)
            .output(audio_file, acodec='libmp3lame', map=track_map, ar='16k')  
            .global_args('-y')  # Overwrite output file if it exists
            .global_args('-loglevel', 'error')  # Reduce ffmpeg output
            .run(capture_stdout=True, capture_stderr=True)
        )
            
        print(f"Audio extracted successfully to: {audio_file}")
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        raise Exception("Failed to extract audio from video file")
    return audio_file

def extract_subtitles_from_audio(audio_file, model_size="base", model_dir=None, load_from_json=False, dump_to_json=False, src_lang="ja", output_srt=True, initial_prompt=""):
    """
    Extract subtitles from a video file using Whisper and save as SRT.
    
    Args:
        video_path (str): Path to the input video file
        output_srt_path (str): Path where the SRT file will be saved
        model_size (str): Size of the Whisper model to use (tiny, base, small, medium, large)
        model_dir (str): Custom directory to store/load Whisper models
    """
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    json_output_path = f"{base_name}_transcription.json"
    if load_from_json:
        # load the json file into result
        print(f"Loading existing transcription from: {json_output_path}")
        with open(json_output_path, 'r', encoding='utf-8') as json_file:
            result = json.load(json_file)
    else:
        # Set custom model directory if provided
        if model_dir:
            os.environ["WHISPER_MODEL_DIR"] = model_dir
            print(f"Using custom model directory: {model_dir}")
        
        print(f"Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_file, verbose=True, language=src_lang, initial_prompt=initial_prompt)

    #########################################################
    # load audio and pad/trim it to fit 30 seconds
    #audio = whisper.load_audio(audio_file)
    #audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    #mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # detect the spoken language
    #_, probs = model.detect_language(mel)
    #print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    #options = whisper.DecodingOptions(
    #    task="transcribe",
    #    language="ja", #max(probs, key=probs.get),  # Use detected language
    #    temperature=0.0,  # Use greedy decoding
    #    fp16=True,  # Use fp16 for faster inference
    #    without_timestamps=False,  # Include timestamps in the output
    #    suppress_tokens="-1"  # Suppress non-speech tokens
    #)
    #result = whisper.decode(model, mel, options)
    #########################################################


    # Dump the transcription result to a JSON file
    
    if dump_to_json:
        print(f"Saving transcription results to JSON: {json_output_path}")
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
    
        print(f"Transcription saved to: {json_output_path}")

    original_srt_blocks = []
    if output_srt:
        for i, segment in enumerate(result['segments'], start=1):
            # Format timestamps
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            text = segment['text']
            srt_block = SrtBlock(i, start_time, end_time, text, segment['start'], segment['end'])
            original_srt_blocks.append(srt_block)
    else:
        for i, segment in enumerate(result['segments'], start=1):
            original_srt_blocks.append(segment['text'])
    return original_srt_blocks

def find_time_points(srt_blocks):
    time_points = []
    srt_block_num = len(srt_blocks)
    i = 0
    slide_window_size = 3
    while i < srt_block_num-slide_window_size:
        window_blocks = srt_blocks[max(0, i): min(srt_block_num, i+slide_window_size)]
        aggregated_text = " ".join([block.text.replace(",", "").strip() for block in window_blocks])
        if aggregated_text.lower().startswith("students book page"):
            # Use regex to extract page numbers from text like "Students book Page 2"
            import re
            page_match = re.search(r'students\s+book\s+page\s+(\d+)', aggregated_text.lower())
            if page_match:
                page_number = int(page_match.group(1))
                if len(time_points) == 0 or f"page_{page_number}" != time_points[-1][1]:
                    time_points.append((window_blocks[0].ts_start, f"page_{page_number}"))
                    i += slide_window_size
                else:
                    i += 1
            else:
                print(f"No page number found in {aggregated_text}")
                raise Exception(f"No page number found in {aggregated_text}")
        else:
            i += 1

    return time_points

def split_audio_file(audio_file, time_points):
    """
    Split an audio file into multiple segments based on given time points.
    
    Args:
        audio_file (str): Path to the audio file to split
        time_points (list): List of time points in seconds where to split the audio
    
    Returns:
        list: List of paths to the generated audio segments
    """
    from pydub import AudioSegment
    import os
    
    # Ensure time points are sorted
    time_points = sorted(time_points, key=lambda x: x[0])
    # Get the base name and extension of the audio file
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    extension = os.path.splitext(audio_file)[1]
    
    # Load the audio file
    print(f"Loading audio file: {audio_file}")
    audio = AudioSegment.from_file(audio_file)
    
    # Add the end of the audio as the final time point
    duration_ms = len(audio)
    time_points.append((duration_ms / 1000, ""))  # Convert ms to seconds
    
    # Create segments
    output_files = []
    start_time = 0
    last_title = "0"
    for i, (ts, title) in enumerate(time_points):
        # Convert seconds to milliseconds
        start_ms = int(start_time * 1000)
        end_ms = int(ts * 1000)
        
        # Extract segment
        segment = audio[start_ms:end_ms]
        
        # Create output filename
        output_path = f"{base_name}_{i+1}_{last_title}{extension}"
        last_title = title
        # Export segment
        print(f"Exporting segment {i+1}: {start_time:.2f}s to {ts:.2f}s")
        segment.export(output_path, format=extension.replace('.', ''))
        # Add to output list
        output_files.append(output_path)
        
        # Set start time for next segment
        start_time = ts

    print(f"Split {audio_file} into {len(output_files)} segments")
    return output_files


def translate_srt_blocks(srt_blocks, target_lang="zh-cn", src_lang="ja"):
    aggregated_text = []
    translated_srt_blocks = []
    aggregation_num = 10
    total_segments = len(srt_blocks)
    translator = Translator()

    for i, srt_block in enumerate(srt_blocks):
        aggregated_text.append(srt_block.text)
        if i % aggregation_num == aggregation_num-1:
            text_to_translate = "\n\n\n".join(aggregated_text)
            translated_text = translator.translate(text_to_translate, src=src_lang ,dest=target_lang).text
            translated_aggregated_text = translated_text.split("\n\n\n")
            print(f"Translated {i-aggregation_num+1}-{i} / {total_segments} segments")
            assert len(translated_aggregated_text) == aggregation_num
            for j in range(i-aggregation_num+1, i+1):
                translated_srt_block = srt_blocks[j].deepcopy()
                translated_srt_block.text = translated_aggregated_text[j-i+aggregation_num-1]
                translated_srt_blocks.append(translated_srt_block)
            aggregated_text = []

    if len(aggregated_text) > 0:
        text_to_translate = "\n\n\n".join(aggregated_text)
        translated_text = translator.translate(text_to_translate, src=src_lang ,dest=target_lang).text
        translated_aggregated_text = translated_text.split("\n\n\n")
        print(f"Translated {len(translated_srt_blocks)}-{len(srt_blocks)} / {total_segments} segments")
        for j in range(len(translated_srt_blocks), len(srt_blocks)):
            translated_srt_block = srt_blocks[j].deepcopy()
            translated_srt_block.text = translated_aggregated_text[j-len(translated_srt_blocks)]
            translated_srt_blocks.append(translated_srt_block)
    return translated_srt_blocks

def save_subtitle_file(srt_blocks, output_srt_path, output_srt=True, summary=False):
    print("Converting to SRT format...")
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for str_block in srt_blocks:
            if output_srt:
                # Format timestamps
                start_time = str_block.start_time
                end_time = str_block.end_time
                f.write(f"{str_block.number}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{str_block.text.strip()}\n\n")
            else:
                if not summary:
                    f.write(f"{str_block.strip()}\n")
        if (not output_srt) and summary:
            summary_text = "\n".join([block.text.strip() for block in srt_blocks])
            summary_text = summarize_content(summary_text)
            f.write(summary_text)
    print(f"Subtitles extracted and saved to: {output_srt_path}")
    return

def translate_srt(input_file, output_file, target_lang="zh-cn", src_lang="ja"):
    # Read the input SRT file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into subtitle blocks
    subtitle_blocks_text = content.strip().split('\n\n')

    srt_blocks = []
    for block_text in subtitle_blocks_text:
        lines = block_text.split('\n')
        assert len(lines) >= 3
        number = lines[0]
        timestamp = lines[1].split("-->")
        start_time = timestamp[0].strip()
        end_time = timestamp[1].strip()
        # Translate the text lines
        text_lines = "\n".join(lines[2:])
        srt_blocks.append(SrtBlock(number, start_time, end_time, text_lines))
    
    translated_srt_blocks = translate_srt_blocks(srt_blocks, target_lang, src_lang)
    save_subtitle_file(translated_srt_blocks, output_file)
    return

def text_to_speech(text, output_file, language='en', slow=False, tld='com'):
    """
    Convert text to speech using Google Text-to-Speech (gTTS) and save as an audio file.
    
    Args:
        text (str): The text to convert to speech
        output_file (str): Path to save the output audio file
        language (str): Language code for the speech (default: 'en')
            Common language codes:
            - 'en': English
            - 'ja': Japanese
            - 'zh-cn': Chinese (Simplified)
            - 'es': Spanish
            - 'fr': French
            - 'de': German
            - 'it': Italian
            - 'pt': Portuguese
            - 'ru': Russian
            - 'ko': Korean
        slow (bool): Whether to speak slowly (default: False)
        tld (str): Top-level domain for Google Translate (default: 'com')
            Options: 'com', 'co.uk', 'co.in', etc.
    
    Returns:
        str: Path to the generated audio file, or None if conversion failed
    
    Example:
        >>> text_to_speech("Hello, world!", "hello.mp3", language="en")
        >>> text_to_speech("こんにちは", "hello.mp3", language="ja")
        >>> text_to_speech("你好", "hello.mp3", language="zh-cn", slow=True)
    """
    try:
        from gtts import gTTS
        import os
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create a gTTS object with specified parameters
        tts = gTTS(
            text=text,
            lang=language,
            slow=slow,
            tld=tld
        )
        
        # Save the audio file
        tts.save(output_file)
        
        print(f"Text-to-speech conversion completed:")
        print(f"  - Output file: {output_file}")
        print(f"  - Language: {language}")
        print(f"  - Speed: {'Slow' if slow else 'Normal'}")
        print(f"  - TLD: {tld}")
        
        return output_file
    except Exception as e:
        print(f"Error in text-to-speech conversion: {str(e)}")
        return None

def batch_text_to_speech(texts, output_dir, language='en', slow=False, tld='com', prefix='tts_'):
    """
    Convert multiple text segments to speech files.
    
    Args:
        texts (list): List of text segments to convert
        output_dir (str): Directory to save the output audio files
        language (str): Language code for the speech (default: 'en')
        slow (bool): Whether to speak slowly (default: False)
        tld (str): Top-level domain for Google Translate (default: 'com')
        prefix (str): Prefix for output filenames (default: 'tts_')
    
    Returns:
        list: List of paths to the generated audio files
    
    Example:
        >>> texts = ["Hello", "World", "How are you?"]
        >>> batch_text_to_speech(texts, "output", language="en")
    """
    try:
        import os
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_files = []
        for i, text in enumerate(texts, 1):
            output_file = os.path.join(output_dir, f"{prefix}{i:03d}.mp3")
            result = text_to_speech(text, output_file, language, slow, tld)
            if result:
                output_files.append(result)
        
        print(f"\nBatch conversion completed:")
        print(f"  - Total files: {len(output_files)}")
        print(f"  - Output directory: {output_dir}")
        
        return output_files
    except Exception as e:
        print(f"Error in batch text-to-speech conversion: {str(e)}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Video subtitle extraction and translation tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Parser for translation
    translate_parser = subparsers.add_parser('translate', help='Translate an existing SRT file')
    translate_parser.add_argument('input_file', help='Input SRT file path')
    translate_parser.add_argument('output_file', help='Output SRT file path')
    
    # Extract audio from video
    extract_audio_parser = subparsers.add_parser('extract_audio', help='Extract audio from a video file')
    extract_audio_parser.add_argument('video_file', help='Input video file path')

    # Parser for video subtitle extraction
    extract_parser = subparsers.add_parser('extract_srt', help='Extract subtitles from a video file')
    extract_parser.add_argument('video_file', help='Input video file path')
    extract_parser.add_argument('output_srt', help='Output SRT file path')
    extract_parser.add_argument('--src-lang', default='ja', help='Source language for transcription (default: ja)')
    extract_parser.add_argument('--track', default='0', help='Audio track number to extract (default: 0)')
    extract_parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3', 'large-v3-turbo', 'turbo'],
                              help='Whisper model size (default: base)')
    extract_parser.add_argument('--model-dir', help='Custom directory to store/load Whisper models')
    extract_parser.add_argument('--initial-prompt', default="", help='Initial prompt for Whisper transcription (default: empty)')
    # Parse for video subtitle extraction and translation
    extract_parser = subparsers.add_parser('translate_srt', help='Extract subtitles from a video file')
    extract_parser.add_argument('video_file', help='Input video file path')
    extract_parser.add_argument('output_srt', help='Output SRT file path')
    extract_parser.add_argument('--track', default='0', help='Audio track number to extract (default: 0)')
    extract_parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3', 'large-v3-turbo', 'turbo'],
                              help='Whisper model size (default: base)')
    extract_parser.add_argument('--model-dir', help='Custom directory to store/load Whisper models')
    extract_parser.add_argument('--src-lang', default='ja', help='Source language for transcription (default: ja)')
    extract_parser.add_argument('--target-lang', default='zh-cn', help='Target language for translation (default: zh-cn)')
    extract_parser.add_argument('--initial-prompt', default="", help='Initial prompt for Whisper transcription (default: empty)')
    # Parser for audio split
    extract_parser = subparsers.add_parser('split', help='Split audio file')
    extract_parser.add_argument('audio_file', help='Input audio file path')
    extract_parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3', 'large-v3-turbo', 'turbo'],
                              help='Whisper model size (default: base)')
    extract_parser.add_argument('--model-dir', help='Custom directory to store/load Whisper models')
    extract_parser.add_argument('--initial-prompt', default="", help='Initial prompt for Whisper transcription (default: empty)')
    args = parser.parse_args()
    
    try:
        if args.command == 'translate':
            translate_srt(args.input_file, args.output_file, target_lang="zh-cn", src_lang="ja")
            print(f"Translation completed successfully!")
            print(f"Translated file saved as: {args.output_file}")
        elif args.command == 'extract_audio':
            extract_audio_from_video(args.video_file, audio_track=0)
            print(f"Audio extracted successfully!")
            print(f"Audio file saved.")
        elif args.command == 'extract_srt':
            audio_file = extract_audio_from_video(args.video_file, audio_track=int(args.track))
            output_srt = args.output_srt.endswith('.srt')
            original_srt_blocks = extract_subtitles_from_audio(audio_file, args.model, args.model_dir, load_from_json=False, dump_to_json=False, src_lang=args.src_lang, output_srt=output_srt, initial_prompt=args.initial_prompt if hasattr(args, 'initial_prompt') else "")
            save_subtitle_file(original_srt_blocks, args.output_srt, output_srt=output_srt)
        elif args.command == 'translate_srt':
            audio_file = extract_audio_from_video(args.video_file, audio_track=int(args.track))
            original_srt_blocks = extract_subtitles_from_audio(audio_file, args.model, args.model_dir, load_from_json=False, dump_to_json=False, src_lang= args.src_lang, output_srt=True, initial_prompt=args.initial_prompt if hasattr(args, 'initial_prompt') else "")
            save_subtitle_file(original_srt_blocks, args.output_srt)
            translated_srt_blocks = translate_srt_blocks(original_srt_blocks, target_lang=args.target_lang , src_lang= args.src_lang)
            save_subtitle_file(translated_srt_blocks, args.output_srt.replace(".srt", f"_{args.target_lang}.srt"))        
        elif args.command == 'split':
            subtitle_blocks = extract_subtitles_from_audio(args.audio_file, args.model, args.model_dir, load_from_json=False, dump_to_json=True, src_lang="en", output_srt=True, initial_prompt=args.initial_prompt if hasattr(args, 'initial_prompt') else "")
            time_points = find_time_points(subtitle_blocks)
            split_audio_file(args.audio_file, time_points)
        else:
            parser.print_help()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main_temp():
    dir_path = '/Users/dongbingfeng/Library/Mobile Documents/com~apple~CloudDocs/temp'
    for f in os.listdir(dir_path):
        if f.endswith('.mp4'):
            print(f"Processing video file: {f}")
            video_file = os.path.join(dir_path, f)
            audio_file = extract_audio_from_video(video_file, audio_track=0)
            output_srt = os.path.join(dir_path, f.replace('.mp4', '.txt'))
            original_srt_blocks = extract_subtitles_from_audio(audio_file, "large-v3", load_from_json=False, dump_to_json=False, src_lang="Chinese", output_srt=False, initial_prompt="普通话，宏观经济学")
            save_subtitle_file(original_srt_blocks, output_srt, output_srt=False)
def main_temp2():
    dir_path = '/Users/dongbingfeng/Library/Mobile Documents/com~apple~CloudDocs/temp/152.理解中国经济的六层思维.txt'
    with open(dir_path, 'r', encoding='utf-8') as f:
        text = f.read()
    summary = summarize_content(text)
    print(summary)

if __name__ == '__main__':
    main()
    #main_temp2()