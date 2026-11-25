# Inference Latency Benchmark Results

**Input Size**: 640×640 | **Iterations**: 200 | **GPU**: NVIDIA RTX 3060

| Backend | Avg (ms) | P95 (ms) | P99 (ms) | FPS | Speedup | Latency Reduction |
|---------|----------|----------|----------|-----|---------|-------------------|
| PyTorch (FP32) | 18.42 | 21.35 | 23.10 | 54.3 | 1.00x | 0.0% |
| PyTorch (FP16) | 12.15 | 14.20 | 15.80 | 82.3 | 1.52x | 34.0% |
| ONNX (CUDA) | 10.85 | 12.40 | 13.90 | 92.2 | 1.70x | 41.1% |
| TensorRT (FP16) | 7.20 | 8.50 | 9.30 | 138.9 | 2.56x | 60.9% |

> **Key Takeaway**: ONNX + FP16 optimization achieves ~41% latency
> reduction over the PyTorch FP32 baseline, enabling sustained 30+ FPS
> real-time inference on surveillance video streams. TensorRT further
> pushes this to 60%+ reduction for edge deployment scenarios.
