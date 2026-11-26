"""VideoMAE action classification for short temporal clips.

The pretrained Kinetics-400 checkpoint reports action labels. It is not
trained to classify UCF-Crime anomalies, so its outputs must not be presented
as anomaly probabilities.

Typical usage:
    classifier = VideoMAEActionClassifier(config)
    action = classifier.classify_clip(clip_frames)
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


@dataclass
class ActionResult:
    """Container for a temporal action classification result.

    Attributes:
        predicted_action: Highest-scoring Kinetics-400 action label.
        confidence:       Softmax score relative to the model's action labels.
        class_scores:     Dict mapping action labels to softmax scores.
        latency_ms:       Inference time in milliseconds.
    """

    predicted_action: str
    confidence: float
    class_scores: dict[str, float]
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


class VideoMAEActionClassifier:
    """VideoMAE Kinetics-400 action classifier.

    Args:
        config: Dictionary with anomaly classifier configuration:
            - model_name (str):         HuggingFace model ID.
            - clip_length (int):        Frames per input clip.
            - stride (int):             Frame stride for sampling.
            - device (str):             Device string.
            - batch_size (int):         Inference batch size.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get(
            "model_name",
            "MCG-NJU/videomae-base-finetuned-kinetics",
        )
        self.clip_length = config.get("clip_length", 16)
        self.stride = config.get("stride", 4)
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
            f"VideoMAE action classifier ready | model={self.model_name} | "
            f"device={self.device} | clip_length={self.clip_length}"
        )

    # ── Public API ─────────────────────────────────────────

    def classify_clip(self, frames: list[np.ndarray]) -> ActionResult:
        """Return the highest-scoring Kinetics-400 action for a clip.

        Args:
            frames: List of BGR numpy arrays (exactly clip_length frames).
                    Can be full frames or cropped bounding box regions.

        Returns:
            ActionResult with action labels and relative model scores.
        """
        t0 = time.perf_counter()

        # Preprocess frames
        pixel_values = self._preprocess(frames)
        pixel_values = pixel_values.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(pixel_values)
            logits = outputs.logits  # (1, number of action labels)

        # Post-process
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_idx = int(np.argmax(probs))

        configured_labels = self.model.config.id2label
        class_labels = [
            configured_labels.get(
                index,
                configured_labels.get(str(index), f"class_{index}"),
            )
            for index in range(len(probs))
        ]
        class_scores = {
            label: float(probability)
            for label, probability in zip(class_labels, probs)
        }

        elapsed = time.perf_counter() - t0
        latency_ms = elapsed * 1000
        self._latencies.append(elapsed)

        return ActionResult(
            predicted_action=class_labels[predicted_idx],
            confidence=float(probs[predicted_idx]),
            class_scores=class_scores,
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

        self.model = VideoMAEForVideoClassification.from_pretrained(self.model_name)
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
        if n == 0:
            raise ValueError("Cannot classify an empty video clip.")
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


AnomalyClassifier = VideoMAEActionClassifier
