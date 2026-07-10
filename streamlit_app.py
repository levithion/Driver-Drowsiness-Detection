import tempfile

import imageio
import streamlit as st

from model_utils import predict_image


st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 960px;
        }
        .hero {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 2rem;
            color: #e2e8f0;
            box-shadow: 0 24px 80px rgba(15, 23, 42, 0.35);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.4rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin-bottom: 0;
            color: #cbd5e1;
            font-size: 1.02rem;
        }
        .status-card {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 1rem 1.25rem;
            color: #e2e8f0;
        }
        .label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #94a3b8;
            margin-bottom: 0.25rem;
        }
        .value {
            font-size: 1.4rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Driver Drowsiness Detection</h1>
        <p>Upload a frame or use your camera, then run inference against the trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

left, right = st.columns([1.2, 0.8], gap="large")


def analyze_video(video_bytes, sample_every_n_frames=15, max_frames=120):
    frame_results = []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_video:
        temp_video.write(video_bytes)
        temp_video.flush()

        reader = imageio.get_reader(temp_video.name, format="ffmpeg")

        for frame_index, frame in enumerate(reader):
            if frame_index >= max_frames:
                break

            if frame_index % sample_every_n_frames != 0:
                continue

            image = frame[:, :, :3]
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp_image:
                imageio.imwrite(temp_image.name, image)
                with open(temp_image.name, "rb") as image_file:
                    label, confidence = predict_image(image_file.read())
                    frame_results.append((frame_index, label, confidence))

        reader.close()

    if not frame_results:
        raise ValueError("No usable frames were extracted from the video.")

    drowsy_count = sum(1 for _, label, _ in frame_results if label == "Drowsy")
    alert_count = len(frame_results) - drowsy_count
    average_confidence = sum(confidence for _, _, confidence in frame_results) / len(frame_results)
    final_label = "Drowsy" if drowsy_count > alert_count else "Alert"

    return final_label, average_confidence, frame_results

with left:
    source_mode = st.radio(
        "Input source",
        ["Upload image", "Upload video", "Use camera"],
        horizontal=True,
    )

    uploaded_file = None
    camera_image = None
    video_file = None

    if source_mode == "Upload image":
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
        )
    elif source_mode == "Upload video":
        video_file = st.file_uploader(
            "Choose a video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
        )
    else:
        camera_image = st.camera_input("Capture a frame")

    image_source = uploaded_file if uploaded_file is not None else camera_image

    run_prediction = st.button("Run Detection", type="primary", use_container_width=True)

with right:
    st.markdown(
        """
        <div class="status-card">
            <div class="label">Model</div>
            <div class="value">Keras .keras classifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <div class="status-card">
            <div class="label">Input size</div>
            <div class="value">224 × 224</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if image_source is not None:
    st.image(image_source, caption="Selected frame", use_container_width=True)

if video_file is not None:
    st.video(video_file)

if run_prediction:
    if source_mode == "Upload video":
        if video_file is None:
            st.error("Upload a video first.")
        else:
            with st.spinner("Analyzing video frames..."):
                try:
                    final_label, average_confidence, frame_results = analyze_video(video_file.getvalue())

                    result_col, conf_col = st.columns(2)
                    with result_col:
                        st.metric("Prediction", final_label)
                    with conf_col:
                        st.metric("Average confidence", f"{average_confidence:.2%}")

                    st.caption(f"Analyzed {len(frame_results)} sampled frames from the video.")

                    if final_label == "Drowsy":
                        st.error("High drowsiness risk detected in the video. Please take a break.")
                    else:
                        st.success("Driver appears alert across the sampled frames.")

                    with st.expander("Frame-by-frame results"):
                        for frame_index, label, confidence in frame_results:
                            st.write(f"Frame {frame_index}: {label} ({confidence:.2%})")
                except Exception as exc:
                    st.exception(exc)
    elif image_source is None:
        st.error("Upload or capture an image first.")
    else:
        image_bytes = image_source.getvalue()

        with st.spinner("Analyzing frame..."):
            try:
                label, confidence = predict_image(image_bytes)

                result_col, conf_col = st.columns(2)
                with result_col:
                    st.metric("Prediction", label)
                with conf_col:
                    st.metric("Confidence", f"{confidence:.2%}")

                if label == "Drowsy":
                    st.error("High drowsiness risk detected. Please take a break.")
                else:
                    st.success("Driver appears alert.")
            except Exception as exc:
                st.exception(exc)