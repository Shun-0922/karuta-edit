import os
import subprocess
from faster_whisper import WhisperModel

def extract_audio(input_video: str, output_audio: str = "audio5.wav"):
    """MP4などの動画から音声を抽出してWAVに変換"""
    cmd = [
        "ffmpeg",
        "-y",  # 既存ファイルを上書き
        "-i", input_video,
        "-ac", "1",       # モノラル
        "-ar", "16000",   # 16kHz
        output_audio
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_audio




if __name__ == "__main__":
    for i in range(6, 13):
        input_video = f"mp4s/test{i}.mp4"  # あなたの動画ファイル名に変更
        print(f"🎬 Extracting audio from {input_video}...")
        audio_path = extract_audio(input_video, output_audio=f"wavs/audio{i}.wav")
