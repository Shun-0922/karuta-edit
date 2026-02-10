import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time
import subprocess
import json



from utils import return_top_scores


def get_audio_sample_rate(input_video: str) -> int | None:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "json",
        input_video,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    data = json.loads(result.stdout)
    streams = data.get("streams", [])

    if not streams:
        return None  # 音声ストリームがない

    return int(streams[0]["sample_rate"])



def extract_audio(input_video: str, output_audio: str):
    """MP4などの動画から音声を抽出してWAVに変換"""
    cmd = [
        "ffmpeg",
        "-y",  # 既存ファイルを上書き
        "-i", input_video,
        "-ac", "1",       # モノラル
        "-ar", f"{get_audio_sample_rate(input_video)}",
        output_audio
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_audio



def simplify_waveform(input_path: str, output_path: str):
    waveform, sample_rate = sf.read(input_path)
    # モノラル化
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    waveform = np.abs(waveform)
    
    frame_size = sample_rate // 10
    num_frames = len(waveform) // frame_size
    waveform = waveform[:num_frames * frame_size]
    waveform = waveform.reshape(num_frames, frame_size)
    waveform = np.max(waveform, axis=1)

    waveform = waveform.copy()
    np.save(output_path, waveform)
    print(f"Saved simplified waveform to {output_path}")




def cut_and_concat_mp4(
    input_video: str,
    segments: list[tuple[float, float]],
    output_video: str,
):
    tmp_dir = "offline_app/tmp_offline"
    os.makedirs(tmp_dir, exist_ok=True)
    segment_files = []

    for i, (start, end) in tqdm(enumerate(segments), total=len(segments)):
        seg_file_name = f"seg_{i}.mp4"
        seg_file_pass = os.path.join(tmp_dir, seg_file_name)
        duration = end - start

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", input_video,
            "-t", str(duration),
            #"-c", "copy",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "ultrafast",
            seg_file_pass,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        segment_files.append(seg_file_name)

    # --- concat 用ファイル作成 ---
    concat_list = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for seg in segment_files:
            f.write(f"file '{seg}'\n")

    # --- 結合 ---
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        output_video,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Error in ffmpeg concat:", result.stderr.decode())

    for seg in segment_files:
        os.remove(os.path.join(tmp_dir, seg))
    os.remove(concat_list)
    os.rmdir(tmp_dir)






def main():
    file_name = "test2"
    before = 2.0
    after = 3.0


    start_time = time.time()

    input_file = f"offline_app/{file_name}.mp4"
    extract_audio(input_file, output_audio=f"offline_app/{file_name}.wav")
    simplify_waveform(f"offline_app/{file_name}.wav", output_path=f"offline_app/simplified_{file_name}.npy")
    waveform = np.load(f"offline_app/simplified_{file_name}.npy")

    print("loaded simplified waveform. Time: ", time.time() - start_time)

    _, score_dict = return_top_scores(waveform)

    print("calculated scores. Time: ", time.time() - start_time)

    sorted_scores = sorted(score_dict.items(), key=lambda x: x[0])

    for idx, score in sorted_scores:
        print(f"Time: {idx/10} s, Score: {score/100} pts")

    segments = []
    for idx, score in sorted_scores:
        center_time = idx / 10.0
        start = max(0.1, center_time - before)
        end = min(center_time + after, len(waveform) / 10.0 - 0.1)
        segments.append((start, end))


    cut_and_concat_mp4(
        input_video=f"offline_app/{file_name}.mp4",
        segments=segments,
        output_video=f"offline_app/processed_{file_name}.mp4",
    )

    print("Processed video. Time: ", time.time() - start_time)

    os.remove(f"offline_app/{file_name}.wav")
    os.remove(f"offline_app/simplified_{file_name}.npy")





if __name__ == "__main__":
    main()
    