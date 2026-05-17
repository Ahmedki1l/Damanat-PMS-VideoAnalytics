# Plate OCR models

This directory is the canonical location for offline / air-gapped copies of
the PaddleOCR-mobile weights consumed by `src/ocr/plate_ocr.py`
(`PaddlePlateOCR`). In a normal install PaddleOCR downloads the weights
to `~/.paddlex/official_models/` on first use — this README documents what
those weights are so deployments without internet access can mirror them
here and point the plugin at this directory via `model_dir=...`.

## Models used by `PaddlePlateOCR`

The plugin defaults to the PaddleOCR v5 mobile pipeline. Three models are
pulled at first use:

| Component                 | Model name                  | Disk footprint | Notes |
|---------------------------|-----------------------------|----------------|-------|
| Text detection            | `PP-OCRv5_mobile_det`       | ~ 5 MB         | DB head, mobile backbone |
| Text recognition (Latin)  | `en_PP-OCRv5_mobile_rec`    | ~ 8 MB         | English-aware recogniser; covers Latin alphanumerics — the canonical script on Saudi / UAE plates |
| Textline orientation      | `PP-LCNet_x1_0_textline_ori`| ~ 7 MB         | Only loaded when `use_angle_cls=True` (default); skip with `--no-angle-cls` |

**Total mobile footprint: ~ 12-20 MB on disk.** The plan mentions a
`~50 MB` envelope; the v5 mobile pipeline is well within that. The
server variants (`PP-OCRv5_server_det`, ~ 80 MB) are explicitly **not**
used — `PaddlePlateOCR` pins the detection model name to the mobile
variant in code so a stray `~/.paddlex` cache cannot regress the runtime.

## Versioning

* PaddleOCR runtime: `paddleocr>=2.7.0` (the plugin supports both the
  2.x and 3.x Python APIs and auto-detects which one is installed at
  construction time).
* PaddlePaddle runtime: `paddlepaddle>=2.6.0` (CPU build). On Windows
  paddlepaddle 3.x + oneDNN tripped a runtime `NotImpl` in our
  development environment; the plugin disables oneDNN by default via
  `enable_mkldnn=False` and `FLAGS_use_mkldnn=0`. Linux production
  builds can flip this on for a ~30 % speedup once oneDNN compatibility
  is verified on the target paddlepaddle build.

## Download instructions

### Online install (default)

Nothing to do — `PaddlePlateOCR()` downloads the weights to
`~/.paddlex/official_models/` on first call:

```python
from src.ocr.plate_ocr import PaddlePlateOCR
ocr = PaddlePlateOCR()       # downloads on first read()
text, conf = ocr.read(crop)
```

### Air-gapped / offline install

1. On a machine with internet access, run the plugin once so PaddleOCR
   populates `~/.paddlex/official_models/`:

   ```bash
   python -c "from src.ocr.plate_ocr import PaddlePlateOCR; \
              import numpy as np; \
              PaddlePlateOCR().read(np.zeros((100,400,3), dtype=np.uint8))"
   ```

2. Copy the three model subdirectories from
   `~/.paddlex/official_models/` into this `models/plate_ocr/` folder:

   ```
   models/plate_ocr/
     PP-OCRv5_mobile_det/
     en_PP-OCRv5_mobile_rec/
     PP-LCNet_x1_0_textline_ori/    # optional, only if you use angle classification
   ```

3. Point the plugin at the mirrored directory either via the
   `model_dir=` constructor argument or by setting
   `MatchingConfig.plate_ocr_model` to this path in `config.yaml`:

   ```yaml
   matching:
     plate_ocr_model: models/plate_ocr
   ```

## Regional accuracy notes

* The default `lang='en'` recogniser handles the Latin row of Saudi /
  UAE plates (which is the row driven by the country's emirate code +
  digits). The Arabic row beneath is usually unreadable from
  floor-camera angles and is intentionally **not** the OCR target.
* If a deployment needs Arabic-script readings, construct the plugin
  with `lang='arabic'`; PaddleOCR will load the matching mobile
  recogniser (`arabic_PP-OCRv5_mobile_rec` or similar depending on
  version) on first use. The detection model stays language-agnostic.

## Re-training

Phase 1 / Wave 1 ships only the off-the-shelf mobile weights. A
plate-specific CRNN fine-tuned on facility data is listed in the plan
under the `risks` section as a future swap; the plugin's `PlateOCR`
abstract base keeps that swap a single-file change (new class, register
on `MatchingConfig.plate_ocr_model`). When that work lands its weights
should also be checked into this directory.
