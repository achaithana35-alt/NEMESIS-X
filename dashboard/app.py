
"""NEMESIS-X Dashboard."""

import time
import json
import tempfile
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.pipeline import AnomalyDetectionPipeline
from src.utils import load_config
from src.video_io import VideoSource
from src.video_analysis import VideoAnalyzer


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
        "*Open-vocabulary video evidence from object detection, tracking, "
        "temporal actions, motion, and optional language prompts*"
    )

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    video_file = None
    local_video_path = None
    semantic_prompts = []
    use_semantic_model = False
    sample_interval_seconds = 1.0
    project_dir = Path(__file__).resolve().parent.parent

    with st.sidebar:

        st.header("⚙️ Configuration")

        source_type = st.selectbox(
            "Video Source",
            ["Webcam", "Video File", "RTSP Stream", "Demo Mode"],
            index=1,
        )

        if source_type == "Video File":
            local_videos = _list_project_videos(project_dir)
            if local_videos:
                file_source = st.radio(
                    "Video file source",
                    ["Project video", "Upload video"],
                    key="video_file_source",
                )
            else:
                file_source = "Upload video"

            if file_source == "Project video":
                selected_video = st.selectbox(
                    "Choose a project video",
                    local_videos,
                    format_func=lambda path: path.relative_to(project_dir).as_posix(),
                    key="project_video",
                )
                local_video_path = selected_video
            else:
                video_file = st.file_uploader(
                    "Upload Video",
                    type=["mp4", "avi", "mkv", "mov"],
                    key="video_upload",
                )

            sample_interval_seconds = st.slider(
                "Analyze one frame every (seconds)",
                min_value=0.25,
                max_value=3.0,
                value=1.0,
                step=0.25,
                key="analysis_sample_interval",
            )
            st.caption(
                "The table is sampled; creating the playable annotated MP4 "
                "runs detection on every video frame and may take longer."
            )
            use_semantic_model = st.checkbox(
                "Enable CLIP open-vocabulary matching",
                value=False,
                key="enable_clip_analysis",
                help=(
                    "Ranks your event descriptions against sampled frames. "
                    "The first use downloads model weights; inference stays local."
                ),
            )
            if use_semantic_model:
                st.caption(
                    "CLIP scores are relative to the descriptions below, not "
                    "probabilities that an event occurred."
                )
                semantic_prompts = [
                    prompt.strip()
                    for prompt in st.text_area(
                        "Event descriptions (one per line)",
                        value=(
                            "ordinary traffic moving through an intersection\n"
                            "a vehicle stopping suddenly or colliding\n"
                            "a pedestrian running through traffic\n"
                            "people gathering around an incident\n"
                            "a person falling or lying on the ground\n"
                            "smoke or fire in a public area\n"
                            "people fighting\n"
                            "an unattended object in a public place"
                        ),
                        height=160,
                        key="event_descriptions",
                    ).splitlines()
                    if prompt.strip()
                ]

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

        st.subheader("Detection settings")
        conf_threshold = st.slider(
            "Detection confidence",
            0.1,
            0.9,
            0.25,
            0.05,
        )

        if source_type == "Demo Mode":
            anomaly_threshold = st.slider(
                "Demo anomaly threshold",
                0.3,
                0.9,
                0.65,
                0.05,
            )
        if source_type in ("Webcam", "RTSP Stream"):
            show_tracks = st.checkbox("Show tracking trails", True)

        st.divider()

        st.subheader("Model Info")

        if source_type == "Video File":
            st.info(
                "**Detector:** YOLOv8s (COCO objects)\n\n"
                "**Tracker:** ByteTrack\n\n"
                "**Semantic model:** CLIP ViT-B/32 (optional)"
            )
        else:
            st.info(
                "**Detector:** YOLOv8s\n\n"
                "**Tracker:** ByteTrack\n\n"
                "**Action model:** VideoMAE Kinetics-400\n\n"
                "Action scores are not anomaly probabilities. First use may "
                "download model weights."
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
                width="stretch",
                key="webcam_stop" if source_type == "Webcam" else "rtsp_stop",
                on_click=_stop_live_from_button,
            )
            start_clicked = False
        elif source_type in ("Webcam", "RTSP Stream"):
            start_clicked = st.button(
                "Start Webcam" if source_type == "Webcam" else "Start Live Stream",
                type="primary",
                width="stretch",
                key="webcam_start" if source_type == "Webcam" else "rtsp_start",
            )
        else:
            start_clicked = st.button(
                "▶ Start Analysis",
                type="primary",
                width="stretch",
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

    if source_type == "Demo Mode":
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
            st.subheader("📈 Demo anomaly score timeline")
            timeline_placeholder = st.empty()

        with col_distribution:
            st.subheader("🎯 Demo score distribution")
            dist_placeholder = st.empty()

        st.divider()
        st.subheader("🚨 Demo alert log")
        alert_placeholder = st.empty()

    elif source_type in ("Webcam", "RTSP Stream") and live_running:
        _render_live_pipeline()

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
        analysis_video_name = (
            local_video_path.name
            if file_source == "Project video" and local_video_path is not None
            else video_file.name
            if file_source == "Upload video" and video_file is not None
            else None
        )

        if start_clicked:
            if (
                file_source == "Project video"
                and local_video_path is None
            ) or (
                file_source == "Upload video"
                and video_file is None
            ):
                st.warning("Choose a project video or upload a video first.")
            elif use_semantic_model and not semantic_prompts:
                st.warning("Add at least one event description for CLIP matching.")
            else:
                temporary_path = None
                video_path = (
                    local_video_path
                    if file_source == "Project video"
                    else None
                )
                if file_source == "Upload video" and video_file is not None:
                    with tempfile.NamedTemporaryFile(
                        suffix=Path(video_file.name).suffix,
                        delete=False,
                    ) as temporary_file:
                        temporary_file.write(video_file.getbuffer())
                        temporary_path = Path(temporary_file.name)
                    video_path = temporary_path

                try:
                    config = load_config(project_dir / "configs" / "default.yaml")
                    config.setdefault("detector", {})[
                        "conf_threshold"
                    ] = conf_threshold
                    analyzer = VideoAnalyzer(
                        config,
                        use_semantic_model=use_semantic_model,
                    )
                    progress = st.progress(
                        0,
                        text="Preparing full-video analysis...",
                    )
                    with st.status(
                        "Analyzing every frame and rendering the annotated MP4...",
                        expanded=True,
                    ) as status:
                        result = analyzer.analyze(
                            video_path,
                            sample_interval_seconds=sample_interval_seconds,
                            semantic_prompts=semantic_prompts,
                            progress_callback=lambda current, total, timestamp: progress.progress(
                                min(current / max(total, 1), 1.0),
                                text=(
                                    f"Annotated frame {current}/{total} "
                                    f"({timestamp:.1f}s)"
                                ),
                            ),
                        )
                        status.update(
                            label="Video analysis complete",
                            state="complete",
                        )
                    progress.empty()
                    result["video_name"] = analysis_video_name
                    st.session_state.video_analysis = result
                except (OSError, RuntimeError, ValueError, ImportError) as error:
                    st.error(f"Could not analyze the video: {error}")
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)

        saved_analysis = st.session_state.get("video_analysis")
        if (
            saved_analysis is not None
            and saved_analysis.get("video_name") == analysis_video_name
        ):
            if saved_analysis.get("analysis_version") == 2:
                _render_video_analysis(saved_analysis)
            elif not start_clicked:
                st.info(
                    "This saved result predates the browser-compatible annotated "
                    "video. Run the analysis again to generate an H.264 MP4."
                )
        elif not start_clicked:
            st.info(
                "Choose a project video or upload one, then click Start Analysis. "
                "The dashboard samples frames and reports detection, tracking, "
                "motion, and optional semantic evidence."
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
                        show_tracks=show_tracks,
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


def _list_project_videos(project_dir: Path) -> list[Path]:
    video_extensions = {".avi", ".mkv", ".mov", ".mp4"}
    return sorted(
        (
            path
            for path in project_dir.iterdir()
            if path.is_file() and path.suffix.lower() in video_extensions
        ),
        key=lambda path: path.name.lower(),
    )


def _render_video_analysis(result: dict) -> None:
    rows = result["rows"]
    if not rows:
        st.warning("No video frames could be decoded for analysis.")
        return

    metadata = result["metadata"]
    st.subheader("Video overview")
    metrics = st.columns(4)
    metrics[0].metric("Duration", f"{metadata['duration_seconds']:.1f} s")
    metrics[1].metric(
        "Frame size",
        f"{metadata['width']} × {metadata['height']}",
    )
    metrics[2].metric("Frames annotated", metadata["processed_frames"])
    metrics[3].metric("Unique tracks", result["unique_tracks"])
    st.caption(
        f"Frame-level table summarizes {metadata['sampled_frames']} samples "
        f"at {metadata['sampling_interval_seconds']:.2f}-second intervals."
    )

    total_detections = sum(row["object_count"] for row in rows)
    class_totals = Counter()
    for row in rows:
        class_totals.update(row["object_counts"])
    if class_totals:
        st.write(
            "Most observed objects in sampled frames: "
            + ", ".join(
                f"{label} ({count})"
                for label, count in class_totals.most_common(5)
            )
        )
    st.caption(
        f"{total_detections} object detections across sampled frames. "
        "Detection counts are observations, not unique objects."
    )

    st.subheader("Likely event setting")
    st.info(result["scene_context"])

    frame_table = pd.DataFrame(rows)
    display_columns = [
        "frame",
        "timestamp_s",
        "object_count",
        "tracked_objects",
        "unique_tracks_seen",
        "people",
        "nearby_person_pairs",
        "max_track_speed_px_s",
        "mean_detection_confidence",
        "object_classes",
    ]
    if any(row["top_prompt_match"] for row in rows):
        display_columns.extend(["top_prompt_match", "prompt_match_score"])

    st.subheader("Frame-level analysis")
    st.dataframe(
        frame_table[display_columns],
        width="stretch",
        hide_index=True,
    )

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.subheader("Objects and people over time")
        st.line_chart(
            frame_table.set_index("timestamp_s")[["object_count", "people"]],
        )
    with chart_columns[1]:
        st.subheader("Motion and nearby people")
        st.line_chart(
            frame_table.set_index("timestamp_s")[
                ["max_track_speed_px_s", "nearby_person_pairs"]
            ],
        )

    if any(row["top_prompt_match"] for row in rows):
        st.subheader("Semantic prompt matches")
        score_rows = []
        for row in rows:
            score_rows.append(
                {
                    "timestamp_s": row["timestamp_s"],
                    **row["prompt_scores"],
                }
            )
        score_table = pd.DataFrame(score_rows).set_index("timestamp_s")
        st.line_chart(
            score_table,
        )
        st.info(
            "CLIP ranks only the descriptions you supplied. These relative "
            "similarity scores are not calibrated event probabilities or proof "
            "that an event occurred."
        )

    preview_frames = result["preview_frames"]
    st.subheader("Annotated frame evidence")
    st.video(
        result["annotated_video"],
        format="video/mp4",
        alt="Full-length video with per-frame object detections and tracking IDs",
    )
    st.download_button(
        "Download annotated video (MP4)",
        data=result["annotated_video"],
        file_name=f"{Path(result['video_name']).stem}_annotated.mp4",
        mime="video/mp4",
        width="stretch",
    )
    st.caption(
        "Annotations are generated for every frame. Audio is not copied into "
        "the annotated output."
    )

    if preview_frames:
        st.subheader("Sampled annotated frame")
        preview_index = st.select_slider(
            "Choose a sampled frame",
            options=list(range(len(preview_frames))),
            format_func=lambda index: (
                f"{preview_frames[index]['timestamp_s']:.2f} seconds"
            ),
        )
        selected_preview = preview_frames[preview_index]
        st.image(
            selected_preview["image"],
            channels="BGR",
            width="stretch",
            alt=f"Annotated video frame at {selected_preview['timestamp_s']:.2f} seconds",
        )

    csv_bytes = frame_table[display_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download frame analysis as CSV",
        data=csv_bytes,
        file_name=f"{Path(result['video_name']).stem}_frame_analysis.csv",
        mime="text/csv",
        width="stretch",
    )

    st.caption(
        "Motion speed is measured in pixels per second and nearby-person counts "
        "use image-space distance. Neither is a real-world distance or an "
        "anomaly verdict."
    )


def _stop_live_from_button():
    st.session_state.webcam_stop_stats = _stop_live_pipeline()


def _start_live_pipeline(
    source,
    source_type,
    conf_threshold,
    show_tracks,
):
    project_dir = Path(__file__).resolve().parent.parent
    config = load_config(project_dir / "configs" / "default.yaml")
    config.setdefault("pipeline", {})["show_display"] = False
    config["pipeline"]["save_output"] = False
    config.setdefault("detector", {})["conf_threshold"] = conf_threshold
    config.setdefault("visualization", {})["show_tracks"] = show_tracks

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
    alerts_metric.metric("Action predictions", stats["action_predictions"])
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
            width="stretch",
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
                width="stretch",
            )

        if i % 10 == 0 and len(timeline_scores) > 10:

            dist_placeholder.plotly_chart(
                _create_distribution_chart(
                    timeline_scores,
                    anomaly_threshold,
                ),
                width="stretch",
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
    main()
