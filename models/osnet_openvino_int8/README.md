# OSNet OpenVINO IR (`models/osnet_openvino_int8/`)

This directory holds the OpenVINO Intermediate Representation (IR) of the
`osnet_ain_x1_0` ReID backbone used by `src/reid_matcher/reid_matcher.py`.

The runtime loads `model.xml` (+ `model.bin`) when both
`MatchingConfig.use_openvino_reid` is true and `model.xml` exists.
Otherwise it falls back to the legacy torchreid path (~1 s/image on CPU).

## Layout

| File | Purpose |
| --- | --- |
| `model.xml` | OpenVINO IR graph (FP32 weights or INT8 quantised) |
| `model.bin` | OpenVINO IR weights |
| `model.onnx` | Source ONNX (kept for re-export / debugging) |
| `metadata.yaml` | Backbone, input size, normalisation, quantisation, hash |

## How it gets built

```
python tools/export_osnet_openvino.py \
    --input-size 192x96 \
    --output-dir models/osnet_openvino_int8 \
    --calibration-dir tests/data/calibration_crops
```

The `--input-size` (default `192x96`) and normalisation parameters
(`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) **must** match
the values baked into `MatchingConfig.reid_input_size` and
`OpenVINOReIDBackend._preprocess`. If you change one, change all three.

## Data dependency D-1

INT8 quantisation needs ~300 representative facility crops in
`tests/data/calibration_crops/`. The repository ships an empty directory.
Two ways to populate it:

1. **Auto-sample** from `vehicle_images/`:

   ```
   python tools/build_calibration_set.py --target 300
   ```

   The script round-robins across camera buckets so no single camera
   dominates the calibration statistics.

2. **Manual** — drop labelled crops directly into
   `tests/data/calibration_crops/`. Any `.jpg`/`.png`/`.bmp` ≥ 32 × 32
   pixels is accepted.

If the directory has fewer than `--min-calibration-images` (default 32)
when the exporter runs, INT8 quantisation is skipped and only the FP32 IR
is produced. The runtime still loads it; the latency target relaxes from
≤ 40 ms / image to ≤ 80 ms / image until INT8 is shipped.

## Verification

After running the exporter, the following must succeed:

```
python -c "from src.reid_matcher import VehicleReIDMatcher; \
    m = VehicleReIDMatcher(); print(m.backend)"
# -> openvino  (or 'torchreid' on the fallback path)

python -m pytest tests/test_reid_cpu_latency.py -v
# -> median latency below the configured threshold
```

## Known limitations

* INT8 calibration shifts the cosine distribution slightly (typically
  `1 - cos_sim < 0.02` against torchreid). Phase 2 / T2.3 calibrates the
  acceptance thresholds against the new distribution; until then the
  Phase-0 thresholds are kept.
* `torch.onnx.export` requires `torch` and `torchreid` to be installed on
  the export host. The runtime does NOT require either of them — only
  `openvino` and `numpy`.
