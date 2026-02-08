import os
import soundfile as sf
import numpy as np


def get_simplified_waveform(input_path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    for i, files in enumerate(os.listdir(input_path)):
        if not files.endswith(".wav"):
            continue
        wav_path = os.path.join(input_path, files)
        waveform, sample_rate = sf.read(wav_path)  # waveform: [samples] または [samples, channels]
        # モノラル化
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        waveform = np.abs(waveform)
        
        #1600frame ごとにさいだい値を取る
        frame_size = 1600
        num_frames = len(waveform) // frame_size
        waveform = waveform[:num_frames * frame_size]
        waveform = waveform.reshape(num_frames, frame_size)
        waveform = np.max(waveform, axis=1)

        waveform = waveform.copy()
        output_file = os.path.join(output_path, f"simplified_{files[:-4]}.npy")
        np.save(output_file, waveform)
        print(f"Saved simplified waveform to {output_file}")
        


if __name__ == "__main__":
    get_simplified_waveform("test/wavs", output_path="test/simplified_waveforms")
