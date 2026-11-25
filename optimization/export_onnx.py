"""Export YOLOv8 model to ONNX format for optimized inference.

Provides automated export with graph simplification and optional
dynamic/static shape configuration. The ONNX model can then be used
with ONNX Runtime or further converted to TensorRT for maximum throughput.

Usage:
    python optimization/export_onnx.py --model yolov8s.pt --output weights/yolov8s.onnx
"""

import argparse
from pathlib import Path

from loguru import logger


def export_to_onnx(
    model_path: str = "yolov8s.pt",
    output_path: str | None = None,
    img_size: int = 640,
    half: bool = True,
    simplify: bool = True,
    dynamic: bool = False,
    opset: int = 17,
) -> str:
    """Export a YOLOv8 model to ONNX format.

    Args:
        model_path:  Path to the PyTorch model (.pt file).
        output_path: Path for the exported ONNX file. If None, auto-generated.
        img_size:    Input image size.
        half:        Export with FP16 weights.
        simplify:    Run onnx-simplifier to optimize the graph.
        dynamic:     Enable dynamic input shapes.
        opset:       ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    from ultralytics import YOLO

    logger.info(f"Exporting {model_path} to ONNX")
    logger.info(f"  Image size: {img_size}")
    logger.info(f"  Half (FP16): {half}")
    logger.info(f"  Simplify: {simplify}")
    logger.info(f"  Dynamic shapes: {dynamic}")
    logger.info(f"  Opset: {opset}")

    model = YOLO(model_path)

    # Export
    export_path = model.export(
        format="onnx",
        imgsz=img_size,
        half=half,
        simplify=simplify,
        dynamic=dynamic,
        opset=opset,
    )

    # Move to desired output path if specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_file = Path(export_path)
        if export_file.exists() and str(export_file) != str(output_path):
            import shutil
            shutil.move(str(export_file), str(output_path))
            export_path = str(output_path)

    logger.success(f"ONNX model exported to: {export_path}")

    # Verify the exported model
    _verify_onnx(export_path)

    return export_path


def _verify_onnx(onnx_path: str) -> None:
    """Verify the exported ONNX model is valid."""
    try:
        import onnx

        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Print model info
        graph = model.graph
        logger.info("ONNX model verified successfully")
        logger.info(f"  Inputs:  {[i.name for i in graph.input]}")
        logger.info(f"  Outputs: {[o.name for o in graph.output]}")

        # File size
        file_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        logger.info(f"  File size: {file_size:.1f} MB")

    except ImportError:
        logger.warning("onnx package not installed, skipping verification")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="YOLOv8 model path")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--half", action="store_true", default=True, help="FP16 export")
    parser.add_argument("--no-simplify", action="store_true", help="Skip graph simplification")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")

    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        img_size=args.img_size,
        half=args.half,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
