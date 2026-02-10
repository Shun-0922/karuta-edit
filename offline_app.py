import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time
import json
import ffmpeg



from utils import return_top_scores


def get_audio_sample_rate(input_video: str) -> int | None:
    probe = ffmpeg.probe(input_video)
    audio_streams = [
        s for s in probe["streams"] if s["codec_type"] == "audio"
    ]
    if not audio_streams:
        return None
    return int(audio_streams[0]["sample_rate"])



def extract_audio(input_video: str, output_audio: str):
    """MP4などの動画から音声を抽出してWAVに変換"""
    probe = ffmpeg.probe(input_video)
    audio_streams = [
        s for s in probe["streams"] if s["codec_type"] == "audio"
    ]

    if not audio_streams:
        raise RuntimeError("No audio stream found in video")

    sample_rate = int(audio_streams[0]["sample_rate"])

    (
        ffmpeg
        .input(input_video)
        .output(
            output_audio,
            ac=1,              # mono
            ar=sample_rate     # original sample rate
        )
        .overwrite_output()
        .run(quiet=True)
    )

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
    segment_paths = []

    for i, (start, end) in enumerate(segments):
        seg_path = os.path.join(tmp_dir, f"seg_{i}.mp4")
        duration = end - start

        (
            ffmpeg
            .input(input_video, ss=start, t=duration)
            .output(
                seg_path,
                vcodec="libx264",
                acodec="aac",
                preset="ultrafast"
            )
            .overwrite_output()
            .run(quiet=True)
        )

        segment_paths.append(seg_path)

    # --- concat ---
    concat_file = os.path.join(tmp_dir, "concat.txt")
    with open(concat_file, "w") as f:
        for p in segment_paths:
            file_name = p.split("/")[-1]
            f.write(f"file '{file_name}'\n")

    (
        ffmpeg
        .input(concat_file, format="concat", safe=0)
        .output(output_video, c="copy")
        .overwrite_output()
        .run()
    )

    # cleanup
    for p in segment_paths:
        os.remove(p)
    os.remove(concat_file)
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
    