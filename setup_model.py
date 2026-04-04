"""
setup_model.py — Download and optionally export the YOLO model.

Run this once before using the system. Downloads yolo11n.pt from
Ultralytics GitHub releases into the local models/ directory.

Usage:
    python setup_model.py                    # Download .pt model only
    python setup_model.py --export onnx      # Download + export to ONNX
    python setup_model.py --export openvino  # Download + export to OpenVINO

The models/ directory is .gitignore'd — weight files stay out of VCS.
For air-gapped deployment: run this once on a machine with internet,
then copy the models/ folder to the target device.
"""

import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Download and optionally export the YOLO model."
    )
    parser.add_argument(
        "--export",
        type=str,
        choices=["onnx", "openvino"],
        default=None,
        help="Export format: 'onnx' or 'openvino'. Default: no export (keep .pt).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m.pt",
        help="Model name to download (default: yolo11m.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to store model files (default: models/).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    model_name = args.model
    output_path = os.path.join(output_dir, model_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if model already exists
    if os.path.exists(output_path) and args.export is None:
        print(f"[INFO] Model already exists at '{output_path}'. Nothing to do.")
        print(f"[TIP]  To re-download, delete '{output_path}' and run again.")
        return

    # Download model via Ultralytics API
    print(f"[INFO] Downloading '{model_name}' via Ultralytics...")
    print(f"[INFO] This requires an internet connection (one-time download).")

    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        print(f"[INFO] Model downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        sys.exit(1)

    # Move the downloaded file to our models/ directory
    # Ultralytics downloads to the current directory or its cache
    if os.path.exists(model_name) and not os.path.exists(output_path):
        shutil.move(model_name, output_path)
        print(f"[INFO] Moved model to '{output_path}'")
    elif not os.path.exists(output_path):
        # Try to copy from ultralytics cache
        # The model object knows its path
        model_file = str(model.ckpt_path) if hasattr(model, 'ckpt_path') else model_name
        if os.path.exists(model_file):
            shutil.copy2(model_file, output_path)
            print(f"[INFO] Copied model to '{output_path}'")
        else:
            print(f"[WARN] Could not locate downloaded file. "
                  f"You may need to manually copy it to '{output_path}'.")

    # Optional export
    if args.export:
        print(f"\n[INFO] Exporting model to {args.export.upper()} format...")

        # Reload from our local path
        model = YOLO(output_path)

        if args.export == "onnx":
            export_path = model.export(format="onnx")
            print(f"[INFO] ONNX model exported to: {export_path}")
            print(f"[TIP]  Update config.yaml → detector.model_path to the ONNX path.")
            print(f"[TIP]  ONNX gives ~1.5–2× CPU speedup over PyTorch.")

        elif args.export == "openvino":
            export_path = model.export(format="openvino")
            print(f"[INFO] OpenVINO model exported to: {export_path}")
            print(f"[TIP]  Update config.yaml → detector.model_path to the OpenVINO directory.")
            print(f"[TIP]  OpenVINO gives ~2–4× CPU speedup on Intel processors.")

    print("\n[DONE] Model setup complete.")
    print(f"[NEXT] Run: python main.py --config config.yaml --video sample.mp4")


if __name__ == "__main__":
    main()
