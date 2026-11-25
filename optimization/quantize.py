"""Model quantization utilities.

Provides FP16 and INT8 quantization for ONNX models to further reduce
inference latency and model size.

Usage:
    python optimization/quantize.py --input weights/yolov8s.onnx --precision fp16
"""

import argparse
from pathlib import Path

import numpy as np
from loguru import logger


def quantize_fp16(
    input_path: str,
    output_path: str | None = None,
) -> str:
    """Convert ONNX model to FP16 precision.

    Args:
        input_path:  Path to the input ONNX model.
        output_path: Path for the quantized model. Auto-generated if None.

    Returns:
        Path to the quantized model.
    """
    try:
        from onnxconverter_common import float16
        import onnx
    except ImportError:
        logger.error("Install: pip install onnxconverter-common onnx")
        raise

    if output_path is None:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_fp16.onnx")

    logger.info(f"Quantizing to FP16: {input_path} → {output_path}")

    model = onnx.load(input_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, output_path)

    # Report size reduction
    orig_size = Path(input_path).stat().st_size / (1024 * 1024)
    quant_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100

    logger.success(
        f"FP16 quantization complete | "
        f"{orig_size:.1f}MB → {quant_size:.1f}MB ({reduction:.1f}% reduction)"
    )
    return output_path


def quantize_int8_dynamic(
    input_path: str,
    output_path: str | None = None,
) -> str:
    """Apply dynamic INT8 quantization to an ONNX model.

    Dynamic quantization computes quantization parameters at runtime,
    requiring no calibration dataset.

    Args:
        input_path:  Path to the input ONNX model.
        output_path: Path for the quantized model.

    Returns:
        Path to the quantized model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error("Install: pip install onnxruntime")
        raise

    if output_path is None:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_int8.onnx")

    logger.info(f"Dynamic INT8 quantization: {input_path} → {output_path}")

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    # Report size reduction
    orig_size = Path(input_path).stat().st_size / (1024 * 1024)
    quant_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100

    logger.success(
        f"INT8 quantization complete | "
        f"{orig_size:.1f}MB → {quant_size:.1f}MB ({reduction:.1f}% reduction)"
    )
    return output_path


def quantize_int8_static(
    input_path: str,
    calibration_data_path: str,
    output_path: str | None = None,
    num_calibration_batches: int = 100,
) -> str:
    """Apply static INT8 quantization with calibration data.

    Static quantization uses representative data to compute optimal
    quantization parameters, yielding better accuracy than dynamic.

    Args:
        input_path:              Path to the input ONNX model.
        calibration_data_path:   Path to calibration images directory.
        output_path:             Path for the quantized model.
        num_calibration_batches: Number of calibration batches.

    Returns:
        Path to the quantized model.
    """
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError:
        logger.error("Install: pip install onnxruntime")
        raise

    if output_path is None:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_int8_static.onnx")

    logger.info(f"Static INT8 quantization: {input_path} → {output_path}")
    logger.info(f"Calibration data: {calibration_data_path}")

    class ImageCalibrationReader(CalibrationDataReader):
        """Reads calibration images for static quantization."""

        def __init__(self, data_path: str, img_size: int = 640, max_batches: int = 100):
            self.data_path = Path(data_path)
            self.img_size = img_size
            self.max_batches = max_batches

            # Gather image files
            extensions = {".jpg", ".jpeg", ".png", ".bmp"}
            self.images = [
                p for p in self.data_path.rglob("*")
                if p.suffix.lower() in extensions
            ][:max_batches]

            self.current = 0
            logger.info(f"Calibration reader: {len(self.images)} images")

        def get_next(self):
            import cv2
            if self.current >= len(self.images):
                return None

            img = cv2.imread(str(self.images[self.current]))
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            img = np.expand_dims(img, axis=0)  # Add batch dim

            self.current += 1
            return {"images": img}

    calibration_reader = ImageCalibrationReader(
        data_path=calibration_data_path,
        max_batches=num_calibration_batches,
    )

    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    logger.success(f"Static INT8 quantization complete: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Model Quantization")
    parser.add_argument("--input", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "int8-dynamic", "int8-static"],
        default="fp16",
        help="Quantization precision",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default="data/calibration/",
        help="Calibration data path (for static INT8)",
    )
    parser.add_argument("--num-batches", type=int, default=100, help="Calibration batches")

    args = parser.parse_args()

    if args.precision == "fp16":
        quantize_fp16(args.input, args.output)
    elif args.precision == "int8-dynamic":
        quantize_int8_dynamic(args.input, args.output)
    elif args.precision == "int8-static":
        quantize_int8_static(
            args.input,
            calibration_data_path=args.calibration_data,
            output_path=args.output,
            num_calibration_batches=args.num_batches,
        )


if __name__ == "__main__":
    main()
