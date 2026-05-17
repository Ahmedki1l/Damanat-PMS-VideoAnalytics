# Client Deployment Guide

This is the one-page install + run reference for shipping Damanat PMS
Video Analytics to a client site. The repository is self-contained:
every model the runtime needs is committed into `models/`. No model
downloads, no training runs, no internet access required for the engine
to boot and process its first frame.

The only thing the client must supply is the camera credentials in
`config.yaml`.

---

## Prerequisites

- **OS:** Windows 10/11 (production target) or Ubuntu 22.04+
- **CPU:** Intel x86_64 with AVX2 (AVX-512 preferred for OpenVINO speed-ups)
- **RAM:** 8 GB minimum, 16 GB recommended (14-camera deployment)
- **Disk:** ~5 GB free (Python venv + repo)
- **Python:** 3.10–3.12 (3.12 tested in production; 3.11 also works)
- **Database:** SQL Server 2019+ reachable at the URL in `config.yaml`
- **Network:** RTSP access to each camera; **no outbound internet required at runtime**

---

## Install steps

```powershell
# 1. Clone the repo (or unzip the delivered archive into a working dir)
git clone <repo-url> Damanat-PMS-VideoAnalytics
cd Damanat-PMS-VideoAnalytics

# 2. Create a CPU-only virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies (~5 minutes, downloads ~600 MB of wheels)
pip install -r requirements.txt

# 4. Copy the config template and fill in camera credentials + DB URL
copy config.example.yaml config.yaml
# Edit config.yaml — replace every YOUR_USER / YOUR_PASSWORD pair and the
# DATABASE_URL line. Do not commit this file.

# 5. (One-off) Draw slot polygons for any camera missing one in slots/
#    — see "Slot polygons" section below if `slots/b2_cam10.json` etc.
#    is missing.

# 6. Run
python main.py
```

That's the whole flow. The engine logs `Loading YOLO model from
'models/yolo11s_openvino_model'` and `Loading IR from
.../osnet_facility_carla_int8_256x128/model.xml` on startup — those are
the only model loads on the runtime path.

---

## What ships in this bundle

| Component | Path | Why it's in git |
|---|---|---|
| Vehicle detector — production | `models/yolo11m_openvino_model/` | YOLO11m OpenVINO IR (~77 MB), ~70-120 ms/frame on CPU. Better recall than `s` on oblique angles. |
| Vehicle detector — rollback | `models/yolo11s_openvino_model/` | YOLO11s OpenVINO IR (~37 MB), ~30-50 ms/frame on CPU. Faster, slightly lower recall. |
| ReID — production | `models/osnet_facility_carla_int8_256x128/` | OSNet-AIN INT8, CARLA-pretrained → facility-finetuned (3.5 MB) |
| ReID — rollback (un-finetuned) | `models/osnet_openvino_int8_256x128/` | Drop-in replacement if the fine-tuned IR regresses |
| ReID — rollback (ImageNet-init finetuned) | `models/osnet_facility_int8_256x128/` | Earlier two-stage variant for A/B comparison |
| Colour classifier | `models/color_classifier_openvino/` | 11-class colour head, OpenVINO IR + labels.json |
| Type classifier | `models/type_classifier_openvino/` | 6-class body-type head, OpenVINO IR + labels.json |
| Slot polygons | `slots/*.json` | Camera-specific slot boundaries (one file per camera) |
| Source code | `src/`, `tools/`, `tests/` | The whole pipeline |

Total committed model size: ~50 MB. Total install footprint (including
the CPU PyTorch wheel chain) is ~3.5 GB.

---

## Configuration knobs the client may want to touch

These are in `config.yaml` after copying from `config.example.yaml`:

| Key | Default | When to change |
|---|---|---|
| `cameras[].ip / user / password` | placeholder | **Always.** Each camera's RTSP creds. |
| `database.DATABASE_URL` | placeholder | **Always.** The PMS database connection string. |
| `processing.target_fps_per_camera` | 1 | Raise to 2 if CPU has headroom and you want faster slot-state convergence. |
| `processing.stream_channel` | 102 (sub) | Set to 101 if your NVR's main stream is what's available. |
| `detector.confidence` | 0.25 | Lower → more detections, more false positives. |
| `output.show_video` | false | Set to true for a debug live view on one camera. |
| `matching.voting_enabled` | true (3-frame 2-of-3) | Disable for instant slot binding (less stable, more flicker). |
| `matching.reid_openvino_model_dir` | `osnet_facility_carla_int8_256x128` | Rollback path — see table above. |

---

## Plate OCR is disabled in this build

`matching.plate_ocr_model: ""` routes to `NoopPlateOCR`. The cascade
ensemble rule (`≥2 of {ReID, colour, type}`) still works fine without it.
To re-enable:

```yaml
matching:
  plate_ocr_model: "paddleocr-mobile"
```

…and uncomment the two `paddleocr` + `paddlepaddle` lines in
`requirements.txt`, then `pip install paddleocr paddlepaddle` (~500 MB).
PaddleOCR will auto-download its mobile det+rec weights (~10 MB) to
`~/.paddlex/official_models/` on the first call — needs internet for
that one-time fetch.

---

## Slot polygons

Each camera needs a `slots/<floor>_<camera>.json` file describing the
parking slot boundaries. The repo ships polygons for every camera EXCEPT
`b2_cam10.json` (referenced in config but the file was not recovered).

If a polygon file is missing on first run the engine logs a warning and
skips slot binding for that camera (detection + ReID still run). To
create one:

```powershell
python draw_slots.py --camera CAM-10 --output slots/b2_cam10.json
```

…then re-launch the engine.

---

## Verifying a healthy run

After `python main.py` boots, look for these lines in the log within
~10 seconds:

```
[INFO] Loading YOLO model from 'models/yolo11m_openvino_model'...
[REID/OV] Loading IR from .../osnet_facility_carla_int8_256x128/model.xml
[REID/OV] Backend ready (quantisation=int8, ...).
[REID] Active backend: openvino (preprocessing=ON, input=256x128)
[MatchDecision] color classifier loaded from models/color_classifier_openvino/model.xml
[MatchDecision] type classifier loaded from models/type_classifier_openvino/model.xml
```

Then expect periodic `[INFO] CAM-XX: detected N vehicles` and
`[MATCH_EVENT] verdict=confirmed plate=XXX-YYYY` lines as traffic
flows through.

---

## Rolling back the ReID model

If the new IR misbehaves in production:

```yaml
# In config.yaml under matching:
reid_openvino_model_dir: "models/osnet_openvino_int8_256x128"  # un-finetuned baseline
```

…and restart the engine. The previous IR is still on disk; no
re-deployment needed. The threshold block in `config.yaml` was
calibrated for the new IR's cosine distribution, so after rollback also
consider raising `b1_zone` back to `0.55` and `reid_solo_confirm` to
`0.70` to match the un-finetuned distribution.

---

## Retraining the ReID on new facility data (optional)

The pipeline for periodic retrains is documented at the end of the
`README.md` under "Retraining ReID on new facility data (Top-20 loop)".
Briefly: collect more vehicle crops into a `<PLATE>/` folder structure,
then run:

```powershell
python tools/curate_facility_dataset.py
python tools/split_facility_dataset.py
python tools/finetune_osnet_top20.py --epochs 40 --patience 8 \
    --init-from models/osnet_carla_pretrain_<date>.pt
python tools/export_osnet_openvino.py \
    --checkpoint models/osnet_facility_finetune_<date>.pt \
    --output-dir models/osnet_facility_carla_int8_256x128 \
    --input-size 256x128 \
    --calibration-dir data/facility_top20/eval --subset-size 200
python tests/test_facility_match_accuracy.py
```

Then update `config.yaml` if the new bench beats the old one, and
restart the engine. **The client never has to run any of this** unless
they want to bring new identities online; the shipped model already
handles the 20 verified identities from the audit.

---

## What's NOT in the bundle and why

| Item | Why excluded |
|---|---|
| `config.yaml` | Contains live RTSP credentials — operator must copy from `config.example.yaml` and fill in. |
| Training datasets (`data/facility_top20/`, `data/external/`) | ~3 GB, not needed at runtime — only for future retraining. |
| Training intermediate checkpoints (`models/*.pt`) | `.pt` files are gitignored. The committed OpenVINO IRs are what the engine actually loads. |
| `vehicle_images/` / `snapshots/` | Runtime output, generated on the client machine. |
| PaddleOCR weights | OCR is disabled in this build; weights would only be needed if you re-enable it. |

---

## Support contact

For installation issues, model regressions, or retraining help, contact
the Damanat PMS engineering team. Include the engine log from the
faulty run, the contents of `data/facility_top20/eval_report_squish_carla.json`
(if present), and the output of `pip freeze` so we can reproduce.
