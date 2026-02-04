import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm






def return_max_waveforms(silence_lengths, waveform, new_sample_rate):
    max_waveforms = []
    for silence_length in silence_lengths:

        max_waveform = waveform.copy()
        # 1sごとに、no_spike_waveformとwaveformの差分の総和を計算
        for j in range(len(waveform) // new_sample_rate):
            start = j * new_sample_rate
            end = (j + 1) * new_sample_rate
            if end > len(waveform) - 1:
                continue
            max = 1
            max_start = (j - silence_length + 1) * new_sample_rate
            max_end = (j + 1) * new_sample_rate
            if max_start < 0 or max_end > len(waveform) - 1:
                continue
            max = np.max(waveform[max_start:max_end])
            max_waveform[start:end] = max
        max_waveforms.append(max_waveform)
    return max_waveforms
    




def return_interval_probability(waveform, max_waveforms, new_sample_rate):
    interval_probability = np.zeros_like(waveform)

    noise_scores = np.zeros_like(waveform)
    small_volume_scores = np.zeros_like(waveform)
    larger_before_scores = np.zeros_like(waveform)
    larger_after_scores = np.zeros_like(waveform)
    decrease_scores = np.zeros_like(waveform)

    for i in tqdm(range(len(waveform))):
        if i < 8*new_sample_rate or i > len(waveform) - 3.5*new_sample_rate:
            continue
        noise_score = 0
        noise_score_eval_point = i - new_sample_rate
        for max_waveform in max_waveforms:
            noise_score += np.mean(max_waveform > max_waveform[noise_score_eval_point])*20
        noise_scores[i] = noise_score*noise_score

        small_volume_score = 0
        minmax = 1
        for j in range(new_sample_rate - 1):
            max_start = i - (new_sample_rate - 1) + 1 + j
            max_end = i + 1 + j
            minmax = np.max(waveform[max_start:max_end]) if np.max(waveform[max_start:max_end]) < minmax else minmax
        small_volume_score = np.mean(waveform > minmax)*100
        small_volume_scores[i] = small_volume_score*small_volume_score

        larger_before_score = np.sum(waveform[i - 6*new_sample_rate:i - new_sample_rate] > waveform[i])
        larger_before_scores[i] = larger_before_score

        larger_after_score = 2*np.sum(waveform[i + new_sample_rate:i + 7*new_sample_rate//2] > waveform[i])
        larger_after_scores[i] = larger_after_score

        decrease_score = 0
        for j in range(i - 2*new_sample_rate, i - new_sample_rate):
            for k in range(j + 1, i - new_sample_rate):
                if waveform[j] > waveform[k]:
                    decrease_score += 1
        decrease_scores[i] = decrease_score

    print(noise_scores[100])
    interval_probability = noise_scores*small_volume_scores*larger_before_scores*larger_after_scores
        
            



    return interval_probability, noise_scores, small_volume_scores, larger_before_scores, larger_after_scores, decrease_scores




def plot_wav_volume(path, output_path="waveform.png"):
    # 音声読み込み

    fig, ax = plt.subplots(5,1, figsize = (800, 30), sharex=True, sharey=True)

    max_duration = 0

    for i, files in enumerate(os.listdir(path)):
        if not files.endswith(".wav"):
            continue
        wav_path = os.path.join(path, files)

        waveform, sample_rate = sf.read(wav_path)  # waveform: [samples] または [samples, channels]
        # モノラル化
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        waveform = np.abs(waveform)
            

        #1600frame ごとにさいだい値を取る
        frame_size = 1600
        new_sample_rate = sample_rate // frame_size
        num_frames = len(waveform) // frame_size
        waveform = waveform[:num_frames * frame_size]
        waveform = waveform.reshape(num_frames, frame_size)
        waveform = np.max(waveform, axis=1)

        waveform = waveform.copy()




        silence_lengths = [4,5,6,7,8]
        max_waveforms = return_max_waveforms(silence_lengths, waveform, new_sample_rate)
    






        

        
            




        interval_probability, noise_scores, small_volume_scores, larger_before_scores, larger_after_scores, decrease_scores = return_interval_probability(waveform, max_waveforms, new_sample_rate)

        
        for j in tqdm(range(len(waveform))):
            ax[i].axvspan(xmin=j / new_sample_rate, xmax=(j + 1) / new_sample_rate, ymin=0, ymax=0.17, color="tab:red", alpha=noise_scores[j]/10000)
            ax[i].axvspan(xmin=j / new_sample_rate, xmax=(j + 1) / new_sample_rate, ymin=0.17, ymax=0.33, color="tab:red", alpha=small_volume_scores[j]/10000)
            ax[i].axvspan(xmin=j / new_sample_rate, xmax=(j + 1) / new_sample_rate, ymin=0.33, ymax=0.5, color="tab:red", alpha=larger_before_scores[j]/50)
            ax[i].axvspan(xmin=j / new_sample_rate, xmax=(j + 1) / new_sample_rate, ymin=0.5, ymax=0.67, color="tab:red", alpha=larger_after_scores[j]/50)
            ax[i].axvspan(xmin=j / new_sample_rate, xmax=(j + 1) / new_sample_rate, ymin=0.67, ymax=0.84, color="tab:red", alpha=decrease_scores[j]/50)
            ax[i].axvspan(xmin=j / new_sample_rate, xmax=(j + 1) / new_sample_rate, ymin=0.84, ymax=1.0, color="tab:red", alpha=interval_probability[j]/np.max(interval_probability))

                



        

        
    
    

    

        print(waveform.shape)

        # 時間軸
        duration = len(waveform) / new_sample_rate
        time = np.linspace(0, duration, len(waveform))

        # プロット（ax形式）
        
        ax[i].plot(time, waveform, color='blue')
        for max_waveform in max_waveforms:
            ax[i].plot(time, max_waveform, color='orange', linewidth=5)
        
        if duration >= max_duration:
            max_duration = duration
            ax[i].set_xticks(np.arange(0, duration + 1, 5.0))
        ax[i].grid(True)

        mean_amplitude = np.mean(waveform)
        ax[i].axhline(y=mean_amplitude, color='g', linestyle='--', label='Mean Amplitude')




    # 保存
    fig.savefig(output_path)
    print(f"Saved waveform figure to {output_path}")

# --- 使用例 ---
if __name__ == "__main__":
    plot_wav_volume("wavs", output_path="waveform.png")
