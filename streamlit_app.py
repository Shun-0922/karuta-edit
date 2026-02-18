import streamlit as st
import numpy as np
import tempfile
import os
from utils import return_top_scores
from offline_app import (
    extract_audio,
    simplify_waveform,
    cut_and_concat_mp4,
    extract_preview_clip,
)

st.set_page_config(page_title="かるた動画自動編集アプリ", layout="wide")


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """秒数を MM:SS.s 形式に変換する。"""
    seconds = max(0.0, seconds)
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m:02d}:{s:04.1f}"


def compute_segments(
    sorted_scores: list[tuple[int, float]],
    enabled_set: set[int],
    before: float,
    after: float,
    waveform_len: int,
) -> list[tuple[float, float]]:
    """有効なスコアのみからマージ済みセグメントを構築する。"""
    segments: list[tuple[float, float]] = []
    for idx, _ in sorted_scores:
        if idx not in enabled_set:
            continue
        center = idx / 10.0
        start = max(0.1, center - before)
        end = min(center + after, waveform_len / 10.0 - 0.1)
        if segments and segments[-1][1] >= start:
            segments[-1] = (segments[-1][0], end)
        else:
            segments.append((start, end))
    return segments


def estimate_duration(segments: list[tuple[float, float]]) -> float:
    """セグメント合計秒数を計算する。"""
    return sum(end - start for start, end in segments)


# ---------------------------------------------------------------------------
# 共通 UI
# ---------------------------------------------------------------------------

url = "https://docs.google.com/presentation/d/1gG8EdmBDSkv82v8wLjVtbLoWbaBhAx5MWzBW1FoKmxg/edit?usp=sharing"
st.write(f'[使い方・仕組み]({url})')

slider_values = st.slider('区間を指定：', -10.0, 10.0, (-1.5, 3.0), step=0.5)
if slider_values[0] >= 0:
    st.warning('下限は負の値にしてください。')
if slider_values[1] <= 0:
    st.warning('上限は正の値にしてください。')
if slider_values[0] == slider_values[1]:
    st.snow()
if slider_values[0] == 0.0 and slider_values[1] == 0.0:
    st.balloons()
if slider_values[0] < 0 and slider_values[1] > 0:
    st.success(
        f'上の句の開始時点の{-slider_values[0]}秒前から、'
        f'下の句の終了時点の{slider_values[1]}秒後までを残します。'
    )

if 'state' not in st.session_state:
    st.session_state.state = 1
    for key in list(st.session_state.keys()):
        if key != "state":
            del st.session_state[key]

# file uploader
uploaded_file = st.file_uploader(
    "動画をアップロード：", type=["mp4", "mov"], key="file_uploader"
)


# ---------------------------------------------------------------------------
# State 1: 分析
# ---------------------------------------------------------------------------

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
    sorted_scores = sorted(score_dict.items(), key=lambda x: x[0])

    st.session_state.update({
        "tmpdir": tmpdirname,
        "input_video": input_video_path,
        "waveform": waveform,
        "sorted_scores": sorted_scores,
        "segment_enabled": {idx: True for idx, _ in sorted_scores},
        "preview_clips": {},
        "preview_playing": None,
        "state": 2,
    })
    st.rerun()


# ---------------------------------------------------------------------------
# State 2: セグメント選択 & プレビュー
# ---------------------------------------------------------------------------

if st.session_state.state == 2:
    sorted_scores = st.session_state.sorted_scores
    before = abs(slider_values[0])
    after = abs(slider_values[1])
    waveform = st.session_state.waveform

    # 有効セットと推定時間
    enabled_set = {
        idx for idx, v in st.session_state.segment_enabled.items() if v
    }
    enabled_count = len(enabled_set)
    total_count = len(sorted_scores)

    segments = compute_segments(
        sorted_scores, enabled_set, before, after, len(waveform)
    )
    est_sec = estimate_duration(segments)
    est_min = int(est_sec) // 60
    est_sec_remainder = est_sec - est_min * 60

    # Step 1: st.success を分離 + st.metric 追加
    st.success('動画の分析が完了しました。')
    col_m1, col_m2, _ = st.columns([1, 1, 2])
    with col_m1:
        st.metric("選択区間", f"{enabled_count}/{total_count}")
    with col_m2:
        st.metric("推定出力", f"{est_min}分{est_sec_remainder:.0f}秒")

    # Step 2: 全選択 / 全解除 / 反転 ボタン
    col_all, col_none, col_invert, _ = st.columns([1, 1, 1, 5])
    with col_all:
        if st.button("全選択"):
            for idx, _ in sorted_scores:
                st.session_state.segment_enabled[idx] = True
            st.rerun()
    with col_none:
        if st.button("全解除"):
            for idx, _ in sorted_scores:
                st.session_state.segment_enabled[idx] = False
            st.rerun()
    with col_invert:
        if st.button("反転"):
            for idx, _ in sorted_scores:
                st.session_state.segment_enabled[idx] = not st.session_state.segment_enabled.get(idx, True)
            st.rerun()

    # Step 3: サイドバイサイドレイアウト
    col_player, col_list = st.columns([2, 3])

    # 左カラム: プレビュープレーヤー
    preview_playing = st.session_state.get("preview_playing")
    with col_player:
        if preview_playing is not None:
            cache_key = (preview_playing, before, after)
            clip_path = st.session_state.preview_clips.get(cache_key)
            if clip_path and os.path.exists(clip_path):
                seg_num = next(
                    (i + 1 for i, (idx, _) in enumerate(sorted_scores) if idx == preview_playing),
                    None,
                )
                center = preview_playing / 10.0
                st.caption(
                    f"プレビュー: #{seg_num}  "
                    f"[{format_time(max(0.1, center - before))} - "
                    f"{format_time(min(center + after, len(waveform) / 10.0 - 0.1))}]"
                )
                st.video(clip_path, autoplay=True)
        else:
            st.info("▶ 再生ボタンを押してプレビュー")

    # 右カラム: セグメントリスト (スクロール可能コンテナ)
    with col_list:
        container = st.container(height=500)
        with container:
            for i, (idx, _score) in enumerate(sorted_scores):
                center = idx / 10.0
                seg_start = max(0.1, center - before)
                seg_end = min(center + after, len(waveform) / 10.0 - 0.1)

                col_cb, col_info, col_play = st.columns([0.5, 3, 1])
                with col_cb:
                    checked = st.checkbox(
                        f"seg_{idx}",
                        value=st.session_state.segment_enabled.get(idx, True),
                        key=f"cb_{idx}",
                        label_visibility="collapsed",
                    )
                    if checked != st.session_state.segment_enabled.get(idx, True):
                        st.session_state.segment_enabled[idx] = checked
                        st.rerun()
                with col_info:
                    st.markdown(
                        f"**#{i + 1}** &nbsp; "
                        f"[{format_time(seg_start)} - {format_time(seg_end)}]"
                    )
                with col_play:
                    is_playing = preview_playing == idx
                    btn_label = "■ 停止" if is_playing else "▶ 再生"
                    if st.button(btn_label, key=f"play_{idx}"):
                        if is_playing:
                            # 停止: プレビューをクリアして案内に戻す
                            st.session_state.preview_playing = None
                            st.rerun()
                        else:
                            # Step 4: st.spinner でプレビュー生成中フィードバック
                            cache_key = (idx, before, after)
                            if cache_key not in st.session_state.preview_clips:
                                with st.spinner("プレビュー生成中..."):
                                    clip_dir = os.path.join(
                                        st.session_state.tmpdir, "previews"
                                    )
                                    os.makedirs(clip_dir, exist_ok=True)
                                    clip_path = os.path.join(
                                        clip_dir, f"preview_{idx}.mp4"
                                    )
                                    extract_preview_clip(
                                        input_video=st.session_state.input_video,
                                        center_sec=center,
                                        before_sec=before,
                                        after_sec=after,
                                        output_path=clip_path,
                                    )
                                    st.session_state.preview_clips[cache_key] = clip_path
                            st.session_state.preview_playing = idx
                            st.rerun()

    # Step 5: 0件警告の改善 & 編集ボタン
    if enabled_count == 0:
        st.warning("区間が選択されていません。「全選択」ボタンで復帰できます。")

    if st.button("動画を編集する", disabled=(enabled_count == 0)):
        st.session_state.state = 3
        st.rerun()


# ---------------------------------------------------------------------------
# State 3: 動画編集
# ---------------------------------------------------------------------------

if st.session_state.state == 3:
    before = abs(slider_values[0])
    after = abs(slider_values[1])
    waveform = st.session_state.waveform
    sorted_scores = st.session_state.sorted_scores

    enabled_set = {
        idx for idx, v in st.session_state.segment_enabled.items() if v
    }
    segments = compute_segments(
        sorted_scores, enabled_set, before, after, len(waveform)
    )
    est_sec = estimate_duration(segments)
    est_min = int(est_sec) // 60
    est_sec_remainder = est_sec - est_min * 60
    output_video = os.path.join(st.session_state.tmpdir, "processed.mp4")

    # State 2 と同数の要素を描画してから処理開始することで、
    # ブロッキング中に旧 State 2 の UI が残るのを防ぐ
    st.info('動画を編集中...しばらくお待ちください。')
    col_m1, col_m2, _ = st.columns([1, 1, 2])
    with col_m1:
        st.metric("対象区間", f"{len(segments)}")
    with col_m2:
        st.metric("推定出力", f"{est_min}分{est_sec_remainder:.0f}秒")
    progress = st.progress(0)
    st.container(height=500)
    st.empty()
    st.empty()

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


# ---------------------------------------------------------------------------
# State 4: ダウンロード
# ---------------------------------------------------------------------------

if st.session_state.state == 4:
    st.success('動画の編集が完了しました')
    st.download_button(
        "ダウンロード",
        data=st.session_state.processed_video,
        file_name="processed_video.mp4",
        mime="video/mp4",
        on_click=lambda: st.session_state.clear(),
    )
