# Calibration crops for ReID INT8 quantisation (D-1)

This directory holds the vehicle-crop calibration set used by
`tools/export_osnet_openvino.py` to apply post-training INT8 quantisation
to the OSNet-AIN OpenVINO IR shipped at `models/osnet_openvino_int8/`.

Image files (`*.jpg`/`*.png`/`*.bmp`) inside this directory are **not
tracked by git** — `.gitignore` excludes them so the repository stays
small. The directory itself (and this README) are tracked so the path
exists on a fresh clone.

## How to populate

Auto-sample from `vehicle_images/`:

```
python tools/build_calibration_set.py --target 300
```

The sampler round-robins across camera buckets so no single camera
dominates the calibration statistics. Re-running is idempotent unless
`--clean` is passed.

## Manual review checklist

Before running the exporter, ensure the calibration set contains:

* **≥ 300 crops** drawn from a representative time-of-day / weather mix.
* **Multiple camera angles** — at minimum CAM-03 (gate), CAM-07 (park
  entry), CAM-11 (floor) so the activation distribution covers all three
  zones the runtime sees.
* **A diverse color/type mix** — see `tools/prepare_color_dataset.py`
  (WS-B) for the canonical 11-class palette.
* **No motion-blurred / heavily occluded crops** — `cv2.imread` succeeds
  but the resulting feature is noisy and the calibrator will hard-code
  the noise into the activation scales.

Discard anything below 32 × 32 pixels — the sampler already filters these
but a manual pass is cheap insurance.

## What happens if the directory is empty

`tools/export_osnet_openvino.py` falls back to the FP32 IR and logs a
warning. The runtime still loads it; the latency target relaxes from
≤ 40 ms / image to ≤ 80 ms / image. See `models/osnet_openvino_int8/README.md`
for details.
