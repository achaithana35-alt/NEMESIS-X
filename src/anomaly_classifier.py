"""VideoMAE-based temporal anomaly classifier.

Classifies short video clips (16 frames) extracted from tracked objects as
normal or anomalous using a VideoMAE (Video Masked Autoencoder) backbone
fine-tuned on the UCF-Crime dataset.

Supports two classification modes:
  1. **Binary** — Normal vs. Anomalous (production deployment).
  2. **Multi-class** — 13 anomaly categories + Normal (evaluation).

The classifier operates on cropped track regions, enabling per-object anomaly
scoring rather than full-frame analysis.

Typical usage:
    classifier = AnomalyClassifier(config)
    score = classifier.classify_clip(clip_frames)
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning(
        "transformers library not found. Install with: pip install transformers"
    )


# UCF-Crime anomaly categories
UCF_CRIME_CLASSES = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


@dataclass
class AnomalyResult:
    """Container for anomaly classification output.

    Attributes:
        anomaly_score:   Probability of anomaly (0 = normal, 1 = anomalous).
        predicted_class: Predicted class name (e.g., "Fighting").
        class_scores:    Dict mapping class names to probabilities.
        is_anomaly:      Whether the score exceeds the threshold.
        latency_ms:      Inference time in milliseconds.
    """

    anomaly_score: float
    predicted_class: str
    class_scores: dict[str, float]
    is_anomaly: bool
    latency_ms: float


class MILAggregator(nn.Module):
    """Multiple Instance Learning (MIL) attention aggregator.

    Aggregates features from multiple clip instances using an attention
    mechanism, enabling weakly-supervised training with video-level labels.

    This is essential for the UCF-Crime dataset where only video-level
    (not frame-level) annotations are available.
    """

    def __init__(self, feature_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Aggregate bag of instance features.

        Args:
            features: (N, D) tensor of N instance feature vectors.

        Returns:
            Scalar anomaly score in [0, 1].
        """
        # Attention weights
        attn_weights = self.attention(features)  # (N, 1)
        attn_weights = F.softmax(attn_weights, dim=0)  # (N, 1)

        # Weighted aggregation
        aggregated = torch.sum(attn_weights * features, dim=0, keepdim=True)  # (1, D)

        # Classification
        score = self.classifier(aggregated)  # (1, 1)
        return score.squeeze()


class AnomalyClassifier:
    """VideoMAE-based video anomaly classifier.

    Uses a pre-trained VideoMAE model as a temporal feature extractor and
    classifier. Operates on 16-frame clips cropped from tracked object
    bounding boxes.

    Args:
        config: Dictionary with anomaly classifier configuration:
            - model_name (str):         HuggingFace model ID.
            - num_classes (int):        Number of output classes.
            - clip_length (int):        Frames per input clip.
            - stride (int):             Frame stride for sampling.
            - anomaly_threshold (float): Score threshold for anomaly flag.
            - device (str):             Device string.
            - batch_size (int):         Inference batch size.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model_name", "MCG-NJU/videomae-base")
        self.num_classes = config.get("num_classes", 14)
        self.clip_length = config.get("clip_length", 16)
        self.stride = config.get("stride", 4)
        self.anomaly_threshold = config.get("anomaly_threshold", 0.65)
        self.batch_size = config.get("batch_size", 1)

        # Device
        device_str = config.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        # Latency tracking
        self._latencies: list[float] = []

        # Load model
        self._load_model()

        logger.success(
            f"AnomalyClassifier ready | model={self.model_name} | "
            f"device={self.device} | clip_length={self.clip_length}"
        )

    # ── Public API ─────────────────────────────────────────

    def classify_clip(self, frames: list[np.ndarray]) -> AnomalyResult:
        """Classify a video clip as normal or anomalous.

        Args:
            frames: List of BGR numpy arrays (exactly clip_length frames).
                    Can be full frames or cropped bounding box regions.

        Returns:
            AnomalyResult with scores and classification.
        """
        t0 = time.perf_counter()

        # Preprocess frames
        pixel_values = self._preprocess(frames)
        pixel_values = pixel_values.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(pixel_values)
            logits = outputs.logits  # (1, num_classes)

        # Post-process
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_idx = int(np.argmax(probs))

        # Build class scores dict
        class_labels = (
            UCF_CRIME_CLASSES
            if self.num_classes == 14
            else [f"class_{i}" for i in range(self.num_classes)]
        )
        class_scores = {}
        for i, label in enumerate(class_labels):
            if i < len(probs):
                class_scores[label] = float(probs[i])

        # Anomaly score = 1 - P(Normal)
        normal_prob = class_scores.get("Normal", probs[0])
        anomaly_score = 1.0 - normal_prob

        predicted_class = class_labels[predicted_idx] if predicted_idx < len(class_labels) else f"class_{predicted_idx}"

        elapsed = time.perf_counter() - t0
        latency_ms = elapsed * 1000
        self._latencies.append(elapsed)

        return AnomalyResult(
            anomaly_score=anomaly_score,
            predicted_class=predicted_class,
            class_scores=class_scores,
            is_anomaly=anomaly_score >= self.anomaly_threshold,
            latency_ms=latency_ms,
        )

    def extract_features(self, frames: list[np.ndarray]) -> np.ndarray:
        """Extract feature embeddings from a clip (for MIL or downstream use).

        Args:
            frames: List of BGR numpy arrays.

        Returns:
            Feature vector as numpy array of shape (feature_dim,).
        """
        pixel_values = self._preprocess(frames).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values, output_hidden_states=True)
            # Use [CLS] token from last hidden state
            features = outputs.hidden_states[-1][:, 0, :]  # (1, D)

        return features.cpu().numpy().squeeze()

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.mean(self._latencies[-50:])) * 1000

    # ── Private helpers ────────────────────────────────────

    def _load_model(self) -> None:
        """Load VideoMAE model and processor from HuggingFace."""
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library required. Install: pip install transformers"
            )

        logger.info(f"Loading VideoMAE model: {self.model_name}")

        self.processor = VideoMAEImageProcessor.from_pretrained(self.model_name)

        self.model = VideoMAEForVideoClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.debug("VideoMAE model loaded and set to eval mode")

    def _preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for VideoMAE input.

        Handles:
          - BGR → RGB conversion
          - Frame count normalization (pad/sample to clip_length)
          - Resize and normalize via VideoMAEImageProcessor

        Args:
            frames: List of BGR numpy arrays.

        Returns:
            Tensor of shape (1, C, T, H, W) ready for model input.
        """
        # Ensure correct clip length
        frames = self._normalize_clip_length(frames)

        # BGR → RGB conversion
        rgb_frames = []
        for frame in frames:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb = frame[:, :, ::-1].copy()  # BGR to RGB # BGR to RGB
            else:
                rgb = frame
            rgb_frames.append(rgb)

        # Process with VideoMAE processor
        inputs = self.processor(
            list(rgb_frames),
            return_tensors="pt",
        )

        return inputs["pixel_values"]

    def _normalize_clip_length(
        self, frames: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Ensure the clip has exactly clip_length frames.

        Pads by repeating the last frame or uniformly samples if too long.
        """
        n = len(frames)
        target = self.clip_length

        if n == target:
            return frames
        elif n < target:
            # Pad by repeating the last frame
            padding = [frames[-1]] * (target - n)
            return frames + padding
        else:
            # Uniform temporal sampling
            indices = np.linspace(0, n - 1, target, dtype=int)
            return [frames[i] for i in indices]
