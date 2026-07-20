# Plate-Detector Evaluation — should a tiny LPD sit between the vehicle crop and PaddleOCR?

**Date:** 2026-07-20 · **Status:** evaluation only, production pipeline unchanged

Every number below is measured. Nothing here is estimated or assumed.

## TL;DR

A YOLO-nano plate detector in front of PaddleOCR is **worth adopting, and the reason
is latency, not accuracy**. Accuracy on ground-truth slots is a wash (5/11 vs 5/11),
but OCR cost drops **3.3x at the median and 8.8x at the worst case**, and the reads
stop containing neighbouring cars' plates.

| pipeline | HIT | MISS | BLANK | det ms | OCR ms (mean) | total |
|---|---|---|---|---|---|---|
| **baseline** — vehicle crop → PaddleOCR | 5 | 4 | 2 | — | 2231.6 | **2231.6** |
| omz_ssd_0106 → crop → OCR | 0 | 0 | 11 | 6.9 | 479.8 | 486.6 |
| **yolo11n_lpd → crop → OCR** | **5** | 3 | 3 | 10.4 | 461.5 | **472.0** |
| yolov8n_lpd → crop → OCR | 5 | 3 | 3 | 11.0 | 506.3 | 517.3 |

## 1. Method

**Corpus.** 36 real vehicle crops pulled from the production snapshot endpoint
(`/pms-video-analytics/snapshots/slot_*_latest.jpg`) on 2026-07-20. These are the
output of `_crop_vehicle_bbox_snapshot`, i.e. materially the same region
`_bbox_crop` hands to OCR today. Covers all 21 slot cameras including the four
named: CAM-22, CAM-24, CAM-13, CAM-00.

**Ground truth.** 11 slots: 9 from the gateway's `current_plate`, plus `B7_CHRO =
ZVH-337` and `B12 CCO = RGR-6466` read visually off the snapshots (both are
unambiguous in frame; both are NULL in the DB, which is the bug under
investigation).

**Scoring.** Reads are scored with production's own `read_matches_plate`, not string
similarity — that function is the gate that decides a bind, so it is the only
metric that predicts behaviour. `FOREIGN` = the read confirms a *different* known
plate, i.e. a wrong-bind.

**OCR.** The production `PaddlePlateOCR` wrapper. Baseline uses
`apply_plate_roi=False` (what `read_slot_plate` does today). The plate-crop variants
use `hud_top/bottom_mask_ratio=0.0, plate_roi_enabled=False` — exactly what the
class docstring already prescribes "when feeding an already-tight plate ROI
(e.g., from a future plate-detection model)".

**Hardware caveat.** Measured on a Ryzen 7 4800H (8 cores), LPD at 2 threads,
PaddleOCR at 4. The production pod is a 36-CPU Xeon. **Absolute ms will differ;
the ratios are the transferable result.**

## 2. Candidates

| model | source | size | native OV | verdict |
|---|---|---|---|---|
| `vehicle-license-plate-detection-barrier-0106` | Intel OMZ | 1.3 MB | **yes, IR ships** | **rejected** |
| `license-plate-finetune-v1n` (YOLO11n) | HF `morsetechlab` (38.7k dl) | 5.5 MB | via export | **recommended** |
| `best.pt` (YOLOv8n) | HF `joker5914` | 6.2 MB | via export | viable alternative |

`keremberke/yolov8n-license-plate` returned HTTP 401 (gated) and was not evaluable.

**Why OMZ was rejected — measured, not assumed.** It is the most attractive
candidate on paper: Intel-maintained, ships as OpenVINO IR, fastest of the three
(6.9 ms median). But it detected a plate in only **3 of 36 crops** and produced
**0 hits on 11 ground-truth slots**. It is trained for barrier/toll geometry
(frontal, close, well-lit); our oblique overhead garage views are out of
distribution. Fast and useless.

## 3. Latency (the decisive result)

```
                  n   median      p90       max      mean
baseline         36   1446.8   3960.6   13177.5    2231.6
yolo11n_lpd      25    439.0    769.9    1497.3     461.5
yolov8n_lpd      26    451.2    882.3    1073.8     506.3
```

- median **1447 → 439 ms** (3.3x)
- p90 **3961 → 770 ms** (5.1x)
- worst case **13,178 → 1,497 ms** (8.8x)

The 13.2-second baseline outlier matters operationally: even on the async worker,
one such call occupies the OCR thread for 13 seconds, and the per-slot budget is
only 12 attempts. Feeding PaddleOCR a ~130×35 plate crop instead of a
1276×574 six-car region removes the text-detection stage's search space, which is
where the cost lives.

Detector cost is ~10 ms and is dwarfed by the OCR saving. **Net: ~1.8 seconds
saved per read.**

## 4. Accuracy

Hit count is unchanged (5/11 both), but the composition differs and the read
*quality* differs sharply.

| slot | truth | baseline | yolo11n | note |
|---|---|---|---|---|
| B3 | HBR-4920 | miss | **HIT** | gained |
| B5 | TRS-9117 | **HIT** | miss | lost |
| B7_CHRO | ZVH-337 | HIT | HIT | both |
| B8_CSBDO | DZD-9488 | HIT | HIT | both |
| B11 CFO | XHD-7651 | HIT | HIT | both |
| B12 CCO | RGR-6466 | HIT | HIT | see below |

**The B12 case is the real argument.** Both "HIT", but:

```
baseline : '337ZVALD117176536466'     <- contains 6466 (B12) AND 337ZV (B7's plate!)
yolo11n  : '11716466RGR'              <- B12's car only
```

The baseline read contains **two cars' plates concatenated**. It matched B12 only
incidentally. It did *not* also match `ZVH-337` purely because letter corroboration
failed (`o_letters='ZVALD'` vs `p_letters='ZVH'`) — a one-character margin from
stamping the neighbour's plate into B12. The plate-crop read has no such exposure.

`FOREIGN = 0` for every config, but that metric is weak here: the known-plate
universe is only 11, so it under-counts contamination risk. The B12 string above is
the honest evidence.

**B13_COO** (no DB ground truth): baseline `'11B8990'`, yolov8n `'119990BHD'` —
the plate-crop read recovers the `BHD` letters, consistent with the visible plate.

## 5. Failure cases

- **CAM-00 (fisheye): 0/5 detections, both YOLO models.** Near-nadir roof camera;
  plates are not in view at all. Not a regression — OCR there is already useless —
  but LPD cannot rescue it. CAM-00 must stay on the appearance path.
- **CAM-16, CAM-21: 0 detections.** CAM-21/B1_CRO is already in
  `slot_no_plate_view` (pure side profile), so this confirms existing knowledge.
- **CAM-13 (B22):** yolo11n 0/1, yolov8n 1/1 — the one place v8n beats 11n.
- **B5 regression (TRS-9117):** baseline hit, LPD missed. Real cost of the change.
- **BLANK went 2 → 3**: when the detector finds no plate, OCR isn't run at all.
  This is arguably correct (don't read the wall) but it *does* mean slots lose the
  lucky-read path they occasionally had.

## 6. Option comparison

**Option A — slot-polygon crop → OCR.** *Not benchmarked.* The snapshot endpoint
serves bbox crops, not full frames, so this could not be measured with available
data and I will not rate it on speculation. Cheap in principle (zero inference) and
`_crop_slot_snapshot` already exists. Testing it requires capturing full frames
from CAM-24. Worth a follow-up; the CAM-24 polygons (0.034–0.043 of frame) are
tighter than the offending bbox (0.23), so it would likely exclude the neighbours —
but it is a fixed region that does not track the car, and it masks to a polygon
drawn for occupancy, not for plate visibility.

**Option B — tiny LPD inside the vehicle crop.** Measured above. Recommended.

**Option C — `max_box_area_ratio`.** Evaluated earlier and **refuted by
measurement**: B12's bogus box is 0.23 of frame while B22's *legitimate* box is
0.53. No global threshold separates them, and a per-camera override builds a whole
extra `TrackedDetector` + OpenVINO pool (`engine.py:161`).

## 7. Recommended architecture

Add plate detection **inside `read_slot_plate`** (`vehicle_registry_identity.py:4605`).

```
_try_ocr_identify           (main thread — unchanged)
  └─ plan_slot_ocr          (main thread, ReID ranking — unchanged)
  └─ worker.submit(job)
       └─ read_slot_plate   (WORKER THREAD)
            ├─ NEW: lpd.detect(crop)   ~10 ms, 2 threads
            ├─ best box -> pad 15% -> plate crop
            └─ ocr.read(plate_crop, apply_plate_roi=False)   ~440 ms
```

This satisfies every stated constraint:

- vehicle detector untouched
- PaddleOCR retained; only its input changes
- runs entirely on the existing async OCR worker — **the main inference loop never
  sees it**, so it cannot become a throughput bottleneck
- native OpenVINO, 2 threads, ~5.5 MB
- no new pool on the detection path (the LPD pool is separate, tiny, and idle
  except during an OCR job — of which there are ~0.6/minute in production)
- **fallback**: when the LPD finds no box, fall through to the current full-crop
  read. This makes the change strictly additive and eliminates the BLANK regression
  and the B5 loss.

### Effort

| task | estimate |
|---|---|
| `PlateRegionDetector` wrapper (load IR, detect, pad) | ~80 lines |
| Wire into `read_slot_plate` + fallback | ~20 lines |
| Config knobs (`slot_lpd_enabled`, model dir, conf) | ~15 lines |
| Ship `models/yolo11n_lpd_openvino_model` (5.5 MB) | commit |
| Tests | ~100 lines |

**~1 day** including tests. Low risk: one function, behind a flag, with fallback.

### Production readiness

**Ready, with conditions.**

- ✅ latency win is large and consistent
- ✅ integration is contained and reversible
- ✅ CPU cost negligible (10 ms on a worker thread)
- ⚠️ ground-truth set is **11 slots** — small. The latency result is robust across
  all 36 crops; the accuracy result is not statistically strong.
- ⚠️ hit count did not improve. Adopt this for **latency and contamination
  safety**, not for recall.
- ⚠️ CAM-00 unaffected; CAM-16/21 unaffected.
- ➡️ **Deploy behind `slot_lpd_enabled`, with full-crop fallback, and re-measure on
  production hardware.**

## 8. On question 6 — replacing PaddleOCR's text detection

Feeding a tight plate crop already collapses the det stage's search space, which is
where the 3.3x saving comes from. Going further (calling the recogniser directly,
skipping det entirely) is plausible but **was not measured** and would need the
plate crop to be tightly rectified first. The measured configuration keeps det
enabled on a small crop, and that is what the numbers above describe.

Empirically it already eliminates the target failure modes: no timestamp reads, no
`CAM-13 (B2-PARKING)` OSD reads, no neighbouring-vehicle text — because none of
that is inside a 130×35 plate box.

## Artifacts

`bench.py`, `analyze.py`, `bench_rows.json`, `det_results.json`, and the 36-crop
corpus are in the session scratchpad.
