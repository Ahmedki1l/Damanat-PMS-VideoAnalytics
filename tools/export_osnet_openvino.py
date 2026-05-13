"""
tools/export_osnet_openvino.py — Export OSNet-AIN to OpenVINO IR (FP32 / INT8).

Exports the ``osnet_ain_x1_0`` backbone used by ``src/reid_matcher/reid_matcher.py``
to an OpenVINO Intermediate Representation so the runtime can drop from
~1 s/image to ~40 ms/image on CPU. This is task **T-A1** of Phase 1 / WS-A.

Pipeline
--------

1. Build the torchreid OSNet-AIN model with pretrained ImageNet/MSMT weights
   (mirrors the load path at ``src/reid_matcher/reid_matcher.py``).
2. ``torch.onnx.export`` → ``model.onnx`` with a dynamic batch dimension,
   input name ``images``, output name ``features``.
3. ``openvino.convert_model`` → FP32 IR (``model.xml`` + ``model.bin``).
4. (Optional) ``nncf.quantize`` against a calibration set in
   ``--calibration-dir`` → INT8 IR. Skipped when ``--skip-quantisation`` is
   set or the directory contains fewer than ``--min-calibration-images``
   images (default 32).

Data dependency D-1
-------------------

INT8 quantisation needs ~300 representative facility crops in
``tests/data/calibration_crops/``. If the directory does not exist or has
too few images the script logs a warning and ships the FP32 IR only — the
runtime path will still load it, the per-image latency target relaxes from
≤40 ms to ≤80 ms. Use ``tools/build_calibration_set.py`` to sample crops
from ``vehicle_images/`` for human review.

Usage
-----

    python tools/export_osnet_openvino.py
    python tools/export_osnet_openvino.py --input-size 192x96
    python tools/export_osnet_openvino.py --skip-quantisation
    python tools/export_osnet_openvino.py \
        --calibration-dir tests/data/calibration_crops \
        --output-dir models/osnet_openvino_int8

Outputs (in ``--output-dir``)
-----------------------------

    model.xml          # OpenVINO IR (INT8 if quantised, FP32 otherwise)
    model.bin          # OpenVINO IR weights
    model.onnx         # Intermediate ONNX (kept for debugging / re-export)
    metadata.yaml      # Backbone, input size, normalisation, quantisation
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("export_osnet_openvino")


DEFAULT_OUTPUT_DIR = Path("models/osnet_openvino_int8")
DEFAULT_CALIBRATION_DIR = Path("tests/data/calibration_crops")
DEFAULT_MIN_CALIBRATION_IMAGES = 32
DEFAULT_INPUT_SIZE = "192x96"   # H x W

# Normalisation parameters — must match ``VehicleReIDMatcher._preprocess``.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# CLI / arg parsing
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the exporter."""
    parser = argparse.ArgumentParser(
        prog="export_osnet_openvino",
        description=(
            "Export the OSNet-AIN ReID backbone (torchreid) to OpenVINO IR "
            "(FP32 by default; INT8 when a calibration set is supplied). "
            "Data dependency D-1: INT8 needs ~300 facility crops in "
            "tests/data/calibration_crops/. See models/osnet_openvino_int8/"
            "README.md for details."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-size",
        type=str,
        default=DEFAULT_INPUT_SIZE,
        help="Input resolution as HxW (e.g. 192x96 or 256x128).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write model.xml, model.bin, model.onnx, metadata.yaml.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help=(
            "Directory of representative vehicle crops for INT8 calibration. "
            "Ignored when --skip-quantisation is set."
        ),
    )
    parser.add_argument(
        "--min-calibration-images",
        type=int,
        default=DEFAULT_MIN_CALIBRATION_IMAGES,
        help=(
            "If --calibration-dir has fewer images than this the script falls "
            "back to the FP32-only path."
        ),
    )
    parser.add_argument(
        "--skip-quantisation",
        action="store_true",
        help=(
            "Skip INT8 quantisation and ship only the FP32 IR. Use this when "
            "the facility has not yet supplied calibration crops (D-1)."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="osnet_ain_x1_0",
        help="torchreid backbone name (must match the runtime path).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version to use during torch.onnx.export.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=300,
        help="Number of calibration images sampled by NNCF for activations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def parse_input_size(spec: str) -> Tuple[int, int]:
    """Parse "HxW" → (H, W)."""
    spec = spec.strip().lower()
    if "x" not in spec:
        raise ValueError(f"--input-size must be HxW, got {spec!r}")
    h_str, w_str = spec.split("x", 1)
    h = int(h_str)
    w = int(w_str)
    if h <= 0 or w <= 0:
        raise ValueError(f"--input-size must be positive, got {h}x{w}")
    return (h, w)


# --------------------------------------------------------------------------- #
# Model build / export
# --------------------------------------------------------------------------- #


def build_torchreid_osnet(model_name: str):
    """Build the OSNet-AIN model with pretrained weights.

    Mirrors the construction path used by ``VehicleReIDMatcher.__init__``.
    """
    import torch  # local import — keeps the CLI usable without torch installed
    import torchreid

    logger.info("Building torchreid model %s (pretrained=True)...", model_name)
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=1000,   # matches the runtime path
        pretrained=True,
    )
    model.eval()
    return model


def _torchreid_weights_hash(model_name: str) -> Optional[str]:
    """Return a sha256 of the cached torchreid pretrained weights, if found."""
    cache_dir = Path.home() / ".cache" / "torch" / "checkpoints"
    if not cache_dir.exists():
        return None
    for f in cache_dir.glob(f"{model_name}*"):
        try:
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
            return f"{f.name}:{h.hexdigest()[:16]}"
        except OSError:
            continue
    return None


def export_to_onnx(
    model,
    output_path: Path,
    input_size: Tuple[int, int],
    opset: int,
) -> None:
    """Trace the torch model to ONNX with a dynamic batch dim."""
    import torch

    h, w = input_size
    dummy = torch.randn(1, 3, h, w, dtype=torch.float32)

    logger.info("Exporting ONNX to %s (input 1x3x%dx%d)...", output_path, h, w)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={
            "images": {0: "batch"},
            "features": {0: "batch"},
        },
    )


def convert_onnx_to_ir(onnx_path: Path, ir_xml_path: Path):
    """Convert ONNX to OpenVINO IR via ``openvino.convert_model``."""
    import openvino as ov

    logger.info("Converting ONNX %s -> OpenVINO IR...", onnx_path)
    model = ov.convert_model(str(onnx_path))
    logger.info("Saving FP32 IR to %s", ir_xml_path)
    # ``compress_to_fp16=False`` keeps the master copy FP32; the IR is
    # quantised separately below.
    ov.save_model(model, str(ir_xml_path), compress_to_fp16=False)
    return model


# --------------------------------------------------------------------------- #
# Calibration data loader
# --------------------------------------------------------------------------- #


def _list_calibration_images(calibration_dir: Path) -> List[Path]:
    if not calibration_dir.exists() or not calibration_dir.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in calibration_dir.iterdir() if p.suffix.lower() in exts)


def _letterbox_preprocess(
    bgr: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Preprocess a BGR crop into a CHW float32 tensor (NCHW with batch 1).

    Must match ``OpenVINOReIDBackend._preprocess`` so the quantisation
    statistics are representative of the runtime distribution.
    """
    target_h, target_w = target_size
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 127, dtype=np.uint8)
    dw, dh = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw, :] = resized
    chw = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(NORM_STD, dtype=np.float32).reshape(3, 1, 1)
    chw = (chw - mean) / std
    return chw[np.newaxis, ...].astype(np.float32)


def build_calibration_dataset(
    calibration_dir: Path,
    input_size: Tuple[int, int],
    max_images: int,
):
    """Return an ``nncf.Dataset`` wrapping calibration crops.

    Returns ``None`` when no usable images are found.
    """
    import nncf

    image_paths = _list_calibration_images(calibration_dir)
    if not image_paths:
        return None, []

    image_paths = image_paths[:max_images]
    tensors: List[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("Skipping unreadable calibration image: %s", p)
            continue
        tensors.append(_letterbox_preprocess(img, input_size))
    if not tensors:
        return None, []

    def _yield():
        for t in tensors:
            yield t

    return nncf.Dataset(list(_yield())), image_paths


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def write_metadata(
    output_dir: Path,
    *,
    backbone: str,
    input_size: Tuple[int, int],
    quantisation: str,
    quant_count: int,
    weights_hash: Optional[str],
    onnx_opset: int,
    extra: Optional[dict] = None,
) -> None:
    """Write ``metadata.yaml`` capturing reproducibility info.

    We write YAML manually (without a yaml dep) so the script remains
    runnable without pyyaml installed.
    """
    h, w = input_size
    timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"backbone: {backbone}",
        f"input_size: [{h}, {w}]   # [height, width]",
        f"normalisation:",
        f"  mean: {NORM_MEAN}",
        f"  std: {NORM_STD}",
        f"quantisation: {quantisation}",
        f"calibration_image_count: {quant_count}",
        f"source_weights_hash: {weights_hash or 'unknown'}",
        f"onnx_opset: {onnx_opset}",
        f"export_timestamp: {timestamp}",
        f"input_name: images",
        f"output_name: features",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {json.dumps(v)}")

    metadata_path = output_dir / "metadata.yaml"
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote metadata to %s", metadata_path)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_size = parse_input_size(args.input_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "model.onnx"
    ir_xml = args.output_dir / "model.xml"

    # 1. Build torchreid model -------------------------------------------------
    t0 = time.perf_counter()
    model = build_torchreid_osnet(args.model_name)
    logger.info("Built torchreid model in %.2fs", time.perf_counter() - t0)

    # 2. Export to ONNX --------------------------------------------------------
    t0 = time.perf_counter()
    export_to_onnx(model, onnx_path, input_size, opset=args.opset)
    logger.info("ONNX export in %.2fs", time.perf_counter() - t0)

    # 3. Convert ONNX -> OpenVINO IR (FP32) ------------------------------------
    t0 = time.perf_counter()
    ov_model = convert_onnx_to_ir(onnx_path, ir_xml)
    logger.info("OV conversion in %.2fs", time.perf_counter() - t0)

    quantisation_status = "fp32"
    calibration_image_count = 0

    # 4. Optional INT8 quantisation -------------------------------------------
    if args.skip_quantisation:
        logger.info("Skipping INT8 quantisation (--skip-quantisation set).")
    else:
        try:
            import nncf  # noqa: F401 - dep check, used inside helper
        except ImportError:
            logger.warning(
                "nncf not installed - skipping INT8 quantisation. Install "
                "'nncf' (already in requirements.txt) for the fast path."
            )
        else:
            dataset, used_paths = build_calibration_dataset(
                args.calibration_dir, input_size, max_images=args.subset_size
            )
            if dataset is None or len(used_paths) < args.min_calibration_images:
                count = len(used_paths) if used_paths is not None else 0
                logger.warning(
                    "Calibration directory %s has only %d image(s) (need >= %d). "
                    "Skipping INT8; emitting FP32 IR only. See "
                    "models/osnet_openvino_int8/README.md (D-1).",
                    args.calibration_dir, count, args.min_calibration_images,
                )
            else:
                import nncf as _nncf
                logger.info(
                    "Quantising IR with NNCF (%d calibration images)...",
                    len(used_paths),
                )
                t0 = time.perf_counter()
                ov_quantised = _nncf.quantize(
                    ov_model,
                    dataset,
                    subset_size=min(args.subset_size, len(used_paths)),
                    preset=_nncf.QuantizationPreset.PERFORMANCE,
                    fast_bias_correction=True,
                )
                import openvino as ov
                ov.save_model(ov_quantised, str(ir_xml), compress_to_fp16=False)
                quantisation_status = "int8"
                calibration_image_count = len(used_paths)
                logger.info(
                    "INT8 quantisation done in %.2fs (subset_size=%d).",
                    time.perf_counter() - t0, calibration_image_count,
                )

    # 5. Write metadata --------------------------------------------------------
    write_metadata(
        args.output_dir,
        backbone=args.model_name,
        input_size=input_size,
        quantisation=quantisation_status,
        quant_count=calibration_image_count,
        weights_hash=_torchreid_weights_hash(args.model_name),
        onnx_opset=args.opset,
        extra={"calibration_dir": str(args.calibration_dir)},
    )

    logger.info(
        "Export complete. Output dir: %s (quantisation=%s)",
        args.output_dir, quantisation_status,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
