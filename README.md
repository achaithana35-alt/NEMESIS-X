<h1 align="center">Real-Time Video Anomaly Detection</h1>

<p align="center">
  <strong>Multi-object tracking and temporal anomaly classification for surveillance video streams</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white" alt="YOLOv8"></a>
  <a href="#"><img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/UCF--Crime_AUC-85.0%25-brightgreen" alt="AUC"></a>
  <a href="#"><img src="https://img.shields.io/badge/Datasets-4_Benchmarks-orange" alt="Datasets"></a>
</p>

---

## Architecture

```mermaid
graph TD
    subgraph Input
        A[Video I/O Async<br/>Webcam, RTSP, Files]
    end

    subgraph Detection & Tracking
        B[YOLOv8 Detection<br/>FP16, 640x640]
        C[ByteTrack<br/>ID Persistence & Trajectories]
    end

    subgraph Classification
        D[VideoMAE<br/>16-frame Temporal Analysis]
        E[MIL Aggregation<br/>Per-track Scoring]
    end

    subgraph Output
        F[Grad-CAM Heatmaps<br/>Visual Explainability]
        G[Streamlit Dashboard<br/>Alerts & Timelines]
    end

    A -->|Frames| B
    B -->|Bounding Boxes| C
    C -->|Object Tracks| D
    D -->|Features| E
    E -->|Anomaly Scores| F
    F -->|Rendered Frames| G
```

## Key Features

| Feature | Description |
|---------|-------------|
| **YOLOv8 Detection** | Real-time person/vehicle detection optimized for FP16 inference. |
| **ByteTrack Tracking** | Multi-object identity persistence across frames with velocity estimation. |
| **VideoMAE Classification** | Temporal anomaly scoring using Video Masked Autoencoders. |
| **85% ROC-AUC** | Tested on the UCF-Crime benchmark across 13 anomaly categories. |
| **30+ FPS** | Sustains real-time performance through ONNX optimization (40% latency reduction). |
| **Grad-CAM Heatmaps** | Adds visual explainability to show where anomalies occur in the frame. |
| **Streamlit Dashboard** | Live monitoring UI featuring an anomaly timeline and alert log. |
| **4 Benchmark Datasets** | Support for downloading and preparing UCF-Crime, ShanghaiTech, CUHK Avenue, and XD-Violence. |
| **Docker Support** | Containerized deployment with NVIDIA CUDA support. |

## Performance

### Inference Latency Benchmark

| Backend | Avg Latency | P95 | FPS | Speedup | Latency Reduction |
|---------|------------|-----|-----|---------|-------------------|
| PyTorch FP32 (baseline) | 18.42ms | 21.35ms | 54.3 | 1.00x | - |
| PyTorch FP16 | 12.15ms | 14.20ms | 82.3 | 1.52x | 34.0% |
| **ONNX Runtime (CUDA)** | **10.85ms** | **12.40ms** | **92.2** | **1.70x** | **41.1%** |
| TensorRT FP16 | 7.20ms | 8.50ms | 138.9 | 2.56x | 60.9% |

### UCF-Crime Evaluation

| Metric | Score |
|--------|-------|
| **Frame-level ROC-AUC** | **85.0%** |
| Average Precision | 76.4% |
| F1-Score | 70.0% |
| False Alarm Rate | 21.6% |
| Equal Error Rate | 23.1% |

### Multi-Dataset Support

| Dataset | Videos | Anomaly Types | Domain |
|---------|--------|---------------|--------|
| **UCF-Crime** | 1,900 | 13 (Arson, Robbery, Assault, etc.) | Surveillance / Security |
| **ShanghaiTech** | 437 | Campus anomalies (biking, loitering) | Campus Safety |
| **CUHK Avenue** | 37 | Pedestrian anomalies (running, throwing) | Pedestrian Monitoring |
| **XD-Violence** | 4,754 | 6 (Fighting, Riot, Explosion, etc.) | Multimodal Violence |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Pari2003/Real-Time-Video-Anomaly-Detection.git
cd Real-Time-Video-Anomaly-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run on Webcam

```bash
python -m src.pipeline --source 0
```

### Run on Video File

```bash
python -m src.pipeline --source path/to/surveillance_video.mp4
```

### Run on RTSP Stream

```bash
python -m src.pipeline --source "rtsp://camera.local/stream1"
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Choose **Webcam**, select the camera index, and click **Start Webcam** to view
frames processed by the pipeline. The webcam must be connected to the machine
running Streamlit; click **Stop Webcam** to release it.

To process a camera stream in the dashboard, choose **RTSP Stream**, enter a
reachable `rtsp://` URL, and click **Start Live Stream**. Click **Stop Live
Stream** to disconnect.

### Run with Docker

```bash
docker build -t anomaly-detector .
docker run --gpus all -p 8501:8501 anomaly-detector
```

## Project Structure

```
├── src/
│   ├── detector.py           # YOLOv8 object detection wrapper
│   ├── tracker.py            # ByteTrack multi-object tracker
│   ├── anomaly_classifier.py # VideoMAE anomaly classification + MIL
│   ├── pipeline.py           # End-to-end pipeline orchestrator
│   ├── video_io.py           # Async video I/O (webcam, RTSP, files)
│   ├── heatmap.py            # Grad-CAM heatmaps & visualization
│   └── utils.py              # FPS counter, config, latency tracking
├── optimization/
│   ├── export_onnx.py        # YOLOv8 -> ONNX export
│   ├── benchmark.py          # Latency benchmark suite
│   └── quantize.py           # FP16/INT8 quantization
├── evaluation/
│   ├── evaluate_ucf_crime.py # UCF-Crime benchmark evaluation
│   ├── compute_metrics.py    # ROC-AUC, AP, EER computation
│   └── plot_results.py       # Publication-quality plots
├── dashboard/
│   └── app.py                # Streamlit real-time monitoring UI
├── configs/
│   ├── default.yaml          # Master configuration
│   ├── bytetrack.yaml        # Tracker configuration
│   └── inference.yaml        # Optimization settings
├── tests/                    # Unit tests
├── results/                  # Benchmark tables, ROC curves, metrics
├── Dockerfile                # Docker deployment
└── .github/workflows/ci.yml  # CI/CD pipeline
```

## Configuration

All pipeline parameters are managed via YAML configs in `configs/`:

```yaml
# configs/default.yaml
detector:
  model: "yolov8s.pt"
  conf_threshold: 0.25
  half: true            # FP16 inference

anomaly:
  model_name: "MCG-NJU/videomae-base"
  clip_length: 16       # Frames per classification window
  anomaly_threshold: 0.65

tracker:
  track_buffer: 30      # Lost track retention (frames)
  match_threshold: 0.8  # ByteTrack IoU threshold
```

## Optimization

### ONNX Export

```bash
python optimization/export_onnx.py --model yolov8s.pt --half --output weights/yolov8s.onnx
```

### Run Benchmarks

```bash
python optimization/benchmark.py --pytorch-model yolov8s.pt --onnx-model weights/yolov8s.onnx
```

### Quantization

```bash
# FP16 quantization
python optimization/quantize.py --input weights/yolov8s.onnx --precision fp16

# Dynamic INT8 quantization
python optimization/quantize.py --input weights/yolov8s.onnx --precision int8-dynamic
```

## Evaluation

### Run UCF-Crime Evaluation

```bash
python evaluation/evaluate_ucf_crime.py --dataset-path data/UCF-Crime/ --output-dir results/
```

### Generate Plots

```bash
python evaluation/plot_results.py --results results/metrics.json
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **YOLOv8 over YOLOv5** | Native ByteTrack integration, better mAP/latency tradeoff |
| **ByteTrack over DeepSORT** | Handles low-confidence detections, no Re-ID network overhead |
| **VideoMAE over I3D** | Superior temporal representation, pre-trained on video data |
| **MIL over supervised** | UCF-Crime only has video-level labels, MIL handles this naturally |
| **ONNX over TensorRT** | Portable across hardware; TensorRT supported as optional upgrade |
| **Sliding window** | Enables continuous scoring without waiting for clip boundaries |

## References

- [YOLOv8](https://github.com/ultralytics/ultralytics) - Ultralytics Object Detection
- [ByteTrack](https://arxiv.org/abs/2110.06864) - Multi-Object Tracking by Associating Every Detection Box
- [VideoMAE](https://arxiv.org/abs/2203.12602) - Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training
- [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) - Real-World Anomaly Detection in Surveillance Videos

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
