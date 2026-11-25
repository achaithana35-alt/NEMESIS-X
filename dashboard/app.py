
"""NEMESIS-X Dashboard."""

import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.pipeline import AnomalyDetectionPipeline
from src.utils import load_config
from src.video_io import VideoSource


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Video Anomaly Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2196F3, #FF5722);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .alert-critical {
        background: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #F44336;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin: 0.25rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------

def main():

    st.markdown(
        '<p class="main-header">🎯 NEMESIS-X</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "*Multi-object tracking + temporal anomaly classification "
        "powered by YOLOv8, ByteTrack & VideoMAE*"
    )

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    video_file = None

    with st.sidebar:

        st.header("⚙️ Configuration")

        source_type = st.selectbox(
            "Video Source",
            ["Webcam", "Video File", "RTSP Stream", "Demo Mode"],
            index=1,
        )

        if source_type == "Video File":

            video_file = st.file_uploader(
                "Upload Video",
                type=["mp4", "avi", "mkv"],
            )

        elif source_type == "RTSP Stream":

            rtsp_url = st.text_input(
                "RTSP URL",
                placeholder="rtsp://camera-address:port/path",
            )

        elif source_type == "Webcam":

            cam_index = st.number_input(
                "Camera Index",
                0,
                10,
                0,
            )

        st.divider()

        st.subheader("Detection Settings")

        conf_threshold = st.slider(
            "Confidence Threshold",
            0.1,
            0.9,
            0.25,
            0.05,
        )

        anomaly_threshold = st.slider(
            "Anomaly Threshold",
            0.3,
            0.9,
            0.65,
            0.05,
        )

        st.divider()

        st.subheader("Display Options")

        show_heatmap = st.checkbox(
            "Show Anomaly Heatmap",
            True,
        )

        show_tracks = st.checkbox(
            "Show Tracking Trails",
            True,
        )

        show_scores = st.checkbox(
            "Show Anomaly Scores",
            True,
        )

        st.divider()

        st.subheader("Model Info")

        st.info(
            "**Detector:** YOLOv8s\n\n"
            "**Tracker:** ByteTrack\n\n"
            "**Classifier:** VideoMAE-Base\n\n"
            "**Dataset:** UCF-Crime"
        )

        st.divider()

        # START BUTTON

        live_running = st.session_state.get("live_running", False)
        if source_type == "Webcam":
            live_source = int(cam_index)
        elif source_type == "RTSP Stream":
            live_source = rtsp_url.strip()
        else:
            live_source = None
        if live_running and (
            source_type not in ("Webcam", "RTSP Stream")
            or st.session_state.live_source != live_source
        ):
            _stop_live_pipeline()
            live_running = False
            st.info("Video source changed. Start the live stream again to reconnect.")

        if source_type in ("Webcam", "RTSP Stream") and live_running:
            st.button(
                "Stop Webcam" if source_type == "Webcam" else "Stop Live Stream",
                type="primary",
                use_container_width=True,
                key="webcam_stop" if source_type == "Webcam" else "rtsp_stop",
                on_click=_stop_live_from_button,
            )
            start_clicked = False
        elif source_type in ("Webcam", "RTSP Stream"):
            start_clicked = st.button(
                "Start Webcam" if source_type == "Webcam" else "Start Live Stream",
                type="primary",
                use_container_width=True,
                key="webcam_start" if source_type == "Webcam" else "rtsp_start",
            )
        else:
            start_clicked = st.button(
                "▶ Start Analysis",
                type="primary",
                use_container_width=True,
                key="analysis_start",
            )

    # -----------------------------------------------------
    # MAIN DASHBOARD LAYOUT
    # -----------------------------------------------------
    if (
        source_type not in ("Webcam", "RTSP Stream")
        and st.session_state.get("live_running")
    ):
        st.session_state.webcam_stop_stats = _stop_live_pipeline()

    if source_type in ("Webcam", "RTSP Stream") and live_running:
        _render_live_pipeline()
    else:
        col_video, col_metrics = st.columns([3, 1])

        with col_video:
            st.subheader("📹 Live Feed")
            video_placeholder = st.empty()

        with col_metrics:
            st.subheader("📊 Metrics")
            fps_metric = st.empty()
            tracks_metric = st.empty()
            alerts_metric = st.empty()
            latency_metric = st.empty()

    st.divider()

    col_timeline, col_distribution = st.columns([2, 1])
    with col_timeline:
        st.subheader("📈 Anomaly Score Timeline")
        timeline_placeholder = st.empty()

    with col_distribution:
        st.subheader("🎯 Score Distribution")
        dist_placeholder = st.empty()

    st.divider()
    st.subheader("🚨 Alert Log")
    alert_placeholder = st.empty()

    if source_type == "Demo Mode":
        _run_demo_mode(
            video_placeholder,
            fps_metric,
            tracks_metric,
            alerts_metric,
            latency_metric,
            timeline_placeholder,
            dist_placeholder,
            alert_placeholder,
            anomaly_threshold,
        )

    elif source_type == "Video File":
        if start_clicked:
            if video_file is None:
                st.warning("Please upload a video first.")
            else:
                project_dir = Path(__file__).resolve().parent.parent
                video_path = (
                    project_dir
                    / f"uploaded_video{Path(video_file.name).suffix.lower()}"
                )
                output_path = project_dir / "outputs" / "output.mp4"
                video_path.write_bytes(video_file.getbuffer())

                st.success("Video uploaded successfully!")
                st.info("Starting anomaly detection pipeline...")

                with st.spinner("Processing video. Please wait..."):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "src.pipeline",
                            "--source",
                            str(video_path),
                        ],
                        cwd=str(project_dir),
                        capture_output=True,
                        text=True,
                    )

                if result.returncode == 0:
                    st.success("Video processing completed!")
                    if output_path.exists():
                        st.subheader("Processed Video")
                        video_placeholder.video(output_path.read_bytes())
                        st.success("Processed video displayed successfully!")
                    else:
                        st.warning(
                            "Pipeline finished, but output.mp4 was not found."
                        )

                    with st.expander("Pipeline Output"):
                        st.text(result.stdout)

                    stats_path = project_dir / "outputs" / "pipeline_stats.json"
                    if stats_path.exists():
                        st.subheader("Pipeline Statistics")
                        try:
                            with stats_path.open("r", encoding="utf-8") as file:
                                stats = json.load(file)
                            st.json(stats)
                        except (OSError, json.JSONDecodeError) as error:
                            st.warning(f"Could not read statistics: {error}")
                else:
                    st.error("Pipeline failed!")
                    st.subheader("Error Details")
                    st.code(result.stderr or result.stdout)
        else:
            st.info(
                "Upload a video and click Start Analysis "
                "to begin processing."
            )

    elif source_type in ("Webcam", "RTSP Stream"):
        if start_clicked:
            if source_type == "RTSP Stream" and not rtsp_url.strip().lower().startswith(
                "rtsp://"
            ):
                st.warning("Enter a valid RTSP URL beginning with rtsp://.")
            else:
                try:
                    _start_live_pipeline(
                        source=live_source,
                        source_type=source_type,
                        conf_threshold=conf_threshold,
                        anomaly_threshold=anomaly_threshold,
                        show_heatmap=show_heatmap,
                        show_tracks=show_tracks,
                        show_scores=show_scores,
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not start {source_type.lower()}: {error}")

        if not st.session_state.get("live_running"):
            if source_type == "Webcam":
                st.info(
                    "Start the webcam to process live frames through the anomaly "
                    "detection pipeline. The camera must be connected to the machine "
                    "running this Streamlit app."
                )
            else:
                st.info(
                    "Enter your camera's RTSP URL and click Start Live Stream. "
                    "The stream must be reachable from the machine running Streamlit."
                )

    stopped_stats = st.session_state.pop("webcam_stop_stats", None)
    if stopped_stats is not None:
        st.success(
            f"Live stream stopped after {stopped_stats['total_frames']} processed frames."
        )
        st.subheader("Pipeline Statistics")
        st.json(stopped_stats)

    elif source_type in ("Webcam", "RTSP Stream"):
        stats_path = (
            Path(__file__).resolve().parent.parent
            / "outputs"
            / "pipeline_stats.json"
        )
        if stats_path.exists():
            try:
                with stats_path.open("r", encoding="utf-8") as file:
                    stats = json.load(file)
                st.subheader("Last saved pipeline statistics")
                st.json(stats)
            except (OSError, json.JSONDecodeError) as error:
                st.warning(f"Could not read saved statistics: {error}")


def _stop_live_from_button():
    st.session_state.webcam_stop_stats = _stop_live_pipeline()


def _start_live_pipeline(
    source,
    source_type,
    conf_threshold,
    anomaly_threshold,
    show_heatmap,
    show_tracks,
    show_scores,
):
    project_dir = Path(__file__).resolve().parent.parent
    config = load_config(project_dir / "configs" / "default.yaml")
    config.setdefault("pipeline", {})["show_display"] = False
    config["pipeline"]["save_output"] = False
    config.setdefault("detector", {})["conf_threshold"] = conf_threshold
    config.setdefault("anomaly", {})["anomaly_threshold"] = anomaly_threshold
    config.setdefault("visualization", {}).update(
        {
            "show_heatmap": show_heatmap,
            "show_tracks": show_tracks,
            "show_scores": show_scores,
        }
    )

    pipeline = AnomalyDetectionPipeline(config)
    video_config = config.get("video", {})
    video_source = VideoSource(
        source=source,
        width=video_config.get("width", 1280),
        height=video_config.get("height", 720),
        fps=video_config.get("fps", 30),
        buffer_size=2,
    )
    try:
        video_source.start()
    except Exception:
        video_source.stop()
        raise

    st.session_state.webcam_pipeline = pipeline
    st.session_state.webcam_source = video_source
    st.session_state.live_source = source
    st.session_state.live_source_type = source_type
    st.session_state.live_running = True


def _stop_live_pipeline():
    video_source = st.session_state.get("webcam_source")
    if video_source is not None:
        video_source.stop()

    pipeline = st.session_state.get("webcam_pipeline")
    stats = pipeline.generate_stats() if pipeline is not None else {
        "total_frames": 0
    }

    for key in (
        "webcam_source",
        "webcam_pipeline",
        "live_source",
        "live_source_type",
    ):
        st.session_state.pop(key, None)
    st.session_state.live_running = False
    return stats


@st.fragment(run_every=0.1)
def _render_live_pipeline():
    col_video, col_metrics = st.columns([3, 1])
    with col_video:
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()
    with col_metrics:
        st.subheader("📊 Metrics")
        fps_metric = st.empty()
        tracks_metric = st.empty()
        alerts_metric = st.empty()
        latency_metric = st.empty()

    pipeline = st.session_state.get("webcam_pipeline")
    video_source = st.session_state.get("webcam_source")
    if pipeline is None or video_source is None:
        return

    source_type = st.session_state.get("live_source_type", "live video")
    frame = video_source.read()
    if frame is None:
        st.caption(f"Waiting for a frame from the {source_type.lower()}...")
        return

    try:
        annotated = pipeline.process_frame(frame)
    except Exception as error:
        _stop_live_pipeline()
        st.error(f"{source_type} pipeline failed: {error}")
        return

    video_placeholder.image(
        annotated,
        channels="BGR",
        width="stretch",
        alt=f"Live {source_type.lower()} processed by the anomaly detection pipeline",
    )
    width, height = video_source.resolution
    st.caption(
        f"{source_type} capture: {width}x{height} at {video_source.actual_fps:.1f} FPS"
    )

    stats = pipeline.get_live_stats()
    fps_metric.metric("FPS", f"{stats['avg_fps']:.1f}")
    tracks_metric.metric("Active Tracks", stats["active_tracks"])
    alerts_metric.metric("Total Alerts", stats["total_alerts"])
    latency_metric.metric("Frames Processed", stats["total_frames"])


# ---------------------------------------------------------
# DEMO MODE FUNCTION
# ---------------------------------------------------------

def _run_demo_mode(
    video_placeholder,
    fps_metric,
    tracks_metric,
    alerts_metric,
    latency_metric,
    timeline_placeholder,
    dist_placeholder,
    alert_placeholder,
    anomaly_threshold,
):

    np.random.seed(42)

    timeline_scores = []
    alerts = []

    for i in range(200):

        score = 0.2 + 0.1 * np.sin(i / 20)

        if 60 < i < 90 or 140 < i < 160:

            score = min(
                score + np.random.uniform(0.3, 0.6),
                1.0,
            )

        else:

            score = max(
                score + np.random.normal(0, 0.05),
                0.0,
            )

        timeline_scores.append(score)

        demo_frame = _generate_demo_frame(
            i,
            score,
            anomaly_threshold,
        )

        video_placeholder.image(
            demo_frame,
            channels="BGR",
            use_container_width=True,
        )

        fps_val = 28 + np.random.uniform(-2, 4)

        num_tracks = np.random.randint(2, 8)

        det_latency = 8 + np.random.uniform(-1, 3)

        fps_metric.metric("FPS", f"{fps_val:.1f}")

        tracks_metric.metric(
            "Active Tracks",
            num_tracks,
        )

        latency_metric.metric(
            "Detection Latency",
            f"{det_latency:.1f} ms",
        )

        if score >= anomaly_threshold:

            alerts.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "frame": i + 1,
                    "score": score,
                    "type": np.random.choice(
                        [
                            "Fighting",
                            "Robbery",
                            "Vandalism",
                            "Assault",
                        ]
                    ),
                }
            )

        alerts_metric.metric(
            "Total Alerts",
            len(alerts),
        )

        if i % 5 == 0:

            timeline_placeholder.plotly_chart(
                _create_timeline_chart(
                    timeline_scores,
                    anomaly_threshold,
                ),
                use_container_width=True,
            )

        if i % 10 == 0 and len(timeline_scores) > 10:

            dist_placeholder.plotly_chart(
                _create_distribution_chart(
                    timeline_scores,
                    anomaly_threshold,
                ),
                use_container_width=True,
            )

        if alerts:

            alert_html = ""

            for alert in reversed(alerts[-10:]):

                alert_html += (
                    f'<div class="alert-critical">'
                    f'<strong>{alert["time"]}</strong> | '
                    f'Frame #{alert["frame"]} | '
                    f'{alert["type"]} | '
                    f'Score: {alert["score"]:.3f}'
                    f'</div>'
                )

            alert_placeholder.markdown(
                alert_html,
                unsafe_allow_html=True,
            )

        time.sleep(0.05)


# ---------------------------------------------------------
# GENERATE DEMO FRAME
# ---------------------------------------------------------

def _generate_demo_frame(
    frame_idx,
    anomaly_score,
    threshold,
):

    frame = np.zeros(
        (480, 854, 3),
        dtype=np.uint8,
    )

    for y in range(480):

        intensity = int(20 + 10 * (y / 480))

        frame[y, :] = [
            intensity,
            intensity + 5,
            intensity + 10,
        ]

    np.random.seed(frame_idx)

    num_objects = np.random.randint(2, 5)

    for obj_idx in range(num_objects):

        x1 = np.random.randint(50, 600)
        y1 = np.random.randint(50, 350)

        w = np.random.randint(60, 150)
        h = np.random.randint(80, 200)

        obj_score = min(
            anomaly_score * np.random.uniform(0.5, 1.2),
            1.0,
        )

        if obj_score >= threshold:

            color = (0, 0, 255)

        else:

            color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x1 + w, y1 + h),
            color,
            2,
        )

        label = f"ID:{obj_idx + 1} Score:{obj_score:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    cv2.putText(
        frame,
        f"FPS: {28 + np.random.uniform(-2, 4):.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    if anomaly_score >= threshold:

        cv2.rectangle(
            frame,
            (0, 440),
            (854, 480),
            (0, 0, 200),
            -1,
        )

        cv2.putText(
            frame,
            f"ANOMALY DETECTED - Score: {anomaly_score:.3f}",
            (10, 465),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return frame


# ---------------------------------------------------------
# TIMELINE CHART
# ---------------------------------------------------------

def _create_timeline_chart(scores, threshold):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=scores,
            mode="lines",
            name="Anomaly Score",
            line=dict(
                color="#2196F3",
                width=2,
            ),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.1)",
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#F44336",
        annotation_text=f"Threshold ({threshold})",
    )

    fig.update_layout(
        height=300,
        margin=dict(
            l=40,
            r=20,
            t=20,
            b=40,
        ),
        yaxis=dict(
            range=[0, 1],
            title="Score",
        ),
        xaxis=dict(
            title="Frame",
        ),
        template="plotly_dark",
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------
# SCORE DISTRIBUTION CHART
# ---------------------------------------------------------

def _create_distribution_chart(scores, threshold):

    scores_arr = np.array(scores)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=scores_arr,
            nbinsx=30,
            marker_color="#2196F3",
            opacity=0.7,
            name="Score Distribution",
        )
    )

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#F44336",
        annotation_text="Threshold",
    )

    fig.update_layout(
        height=300,
        margin=dict(
            l=40,
            r=20,
            t=20,
            b=40,
        ),
        xaxis=dict(
            title="Score",
            range=[0, 1],
        ),
        yaxis=dict(
            title="Count",
        ),
        template="plotly_dark",
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":

    import json

    main()
