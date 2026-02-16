import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from utils import return_top_scores
from offline_app import extract_audio, simplify_waveform, cut_and_concat_mp4

st.set_page_config(page_title="かるた動画自動編集アプリ", layout="wide")


url = "https://docs.google.com/presentation/d/1gG8EdmBDSkv82v8wLjVtbLoWbaBhAx5MWzBW1FoKmxg/edit?usp=sharing"
st.write(f'[使い方・仕組み]({url})')




slider_values = st.slider('区間を指定：', -10.0, 10.0, (-1.5, 3.0), step = 0.5)
if slider_values[0] >= 0:
    st.warning('下限は負の値にしてください。')
if slider_values[1] <= 0:
    st.warning('上限は正の値にしてください。')
if slider_values[0] == slider_values[1]:
    st.snow()
if slider_values[0] == 0.0 and slider_values[1] == 0.0:
    st.balloons()
if slider_values[0] < 0 and slider_values[1] > 0:
    st.success(f'上の句の開始時点の{-slider_values[0]}秒前から、下の句の終了時点の{slider_values[1]}秒後までを残します。')


if 'state' not in st.session_state:
    st.session_state.state = 1
    for key in list(st.session_state.keys()):
        if key != "state":
            del st.session_state[key]





# file uploader
uploaded_file = st.file_uploader("動画をアップロード：", type=["mp4", "mov"], key="file_uploader")


if uploaded_file is not None and st.session_state.state == 1:
    st.status('動画を分析中...しばらくお待ちください。')
    tmpdirname = tempfile.mkdtemp()
    suffix = os.path.splitext(uploaded_file.name)[1]
    if suffix.lower() not in [".mp4", ".mov"]:
        st.error("対応している動画形式はMP4またはMOVのみです。")
        st.session_state.state = 1
        st.rerun()
    input_video_path = os.path.join(tmpdirname, f"input{suffix}")

    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    audio_path = os.path.join(tmpdirname, "audio.wav")
    extract_audio(input_video_path, audio_path)

    simplified_waveform_path = os.path.join(tmpdirname, "simplified.npy")
    simplify_waveform(audio_path, simplified_waveform_path)

    waveform = np.load(simplified_waveform_path)
    _, score_dict = return_top_scores(waveform)

    st.session_state.update({
        "tmpdir": tmpdirname,
        "input_video": input_video_path,
        "waveform": waveform,
        "sorted_scores": sorted(score_dict.items(), key=lambda x: x[0]),
        "state": 2,
    })
    st.rerun()



if st.session_state.state == 2:
    sorted_scores = st.session_state.sorted_scores
    before = abs(slider_values[0])
    after = abs(slider_values[1])
    length_times_10 = len(sorted_scores)*int(before*10 + after*10)
    st.success(f'動画の分析が完了しました。{length_times_10//600}分{length_times_10%600//10}秒の動画に短縮されます。')
    score_data = {
        "Time (s)": [f"{str(int((idx - before*10)//36000))}時間{str(int((idx - before*10)%36000//600))}分{str((idx - before*10)%600/10)}秒 ~ {str(int((idx + after*10)//36000))}時間{str(int((idx + after*10)%36000//600))}分{str((idx + after*10)%600/10)}秒" for idx, _ in sorted_scores],
        "Score": [score / 100.0 for _, score in sorted_scores]
    }
    df = pd.DataFrame(score_data)
    #st.dataframe(df)

    if st.button("動画を編集する"):
        st.session_state.state = 3
        st.rerun()


if st.session_state.state == 3:
    st.status('動画を編集中...しばらくお待ちください。')
    progress = st.progress(0)

    segments = []
    before = abs(slider_values[0])
    after = abs(slider_values[1])

    waveform = st.session_state.waveform
    sorted_scores = st.session_state.sorted_scores

    for idx, _ in sorted_scores:
        center = idx / 10.0
        start = max(0.1, center - before)
        end = min(center + after, len(waveform) / 10.0 - 0.1)
        if segments and segments[-1][1] >= start:
            segments[-1] = (segments[-1][0], end)
        else:
            segments.append((start, end))

    output_video = os.path.join(st.session_state.tmpdir, "processed.mp4")

    cut_and_concat_mp4(
        input_video=st.session_state.input_video,
        segments=segments,
        output_video=output_video,
        progress_callback=lambda p: progress.progress(p),
    )

    with open(output_video, "rb") as f:
        st.session_state.processed_video = f.read()

    st.session_state.state = 4
    st.rerun()




if st.session_state.state == 4:
    st.success('動画の編集が完了しました')
    st.download_button(
        "ダウンロード",
        data=st.session_state.processed_video,
        file_name="processed_video.mp4",
        mime="video/mp4",
        on_click=lambda: st.session_state.clear()
    )