# OSNet Facility Fine-tune (Phase 3 / T3.1)

This directory is the output target for `tools/finetune_osnet_facility.py`.

Until you run the fine-tune, the directory contains only this README. The
fine-tune script writes:

| File | Description |
|---|---|
| `best.pth` | Best triplet-accuracy checkpoint produced by the training loop. |
| `finetune_report.json` | Mining yield + status (`ok`, `skipped`, `partial`). |

The re-exported OpenVINO IR lands in a sibling directory
(`models/osnet_finetuned_int8/`) and is created by the same script when
training completes.

## When to retrain

Run the fine-tune when **any** of the following hold:

1. WS-A INT8 quantisation drift (`tools/calibrate_thresholds.py` reports a
   median cosine offset >= 0.03 from the snapshot baseline).
2. New camera angles or lighting (e.g. an additional floor camera has been
   commissioned) — even a 3-pp accuracy regression on the bench is enough.
3. Quarterly cadence as part of the production hygiene SLO.
4. After a major facility re-arrangement (different gate, repainted slot
   markings) where the visual context has materially changed.

## Data requirements

The script mines from two complementary sources:

### Logs (`--match-log-glob`)

Production `MATCH_EVENT` audit lines emitted by `reid_match_perf` (see
`src/vehicle_registry/vehicle_registry_identity.py:~1288`). Each row gives
one confirmed (gate snapshot, slot snapshot) pair. The mining pipeline also
extracts `ocr_contradiction` and `reattach_below` rejection lines to seed
the hard-negative bucket.

Recommended yield: ~2000 MATCH_EVENT rows for a usable fine-tune.

### Filenames (`--vehicle-images-dir`)

Persisted vehicle crops under `vehicle_images/` follow the naming
convention `<plate>_<camera>_<YYYYMMDD>_<HHMMSS>.jpg`. The mining loop
groups files by plate and emits cross-camera pairs as positives. This is
the fallback when the audit log is incomplete (e.g. a fresh deployment).

Recommended yield: ~500 distinct plates with crops from >= 2 cameras.

### Database (`--db-url`)

Optional. When supplied, the script joins each mined plate against
`dbo.parking_sessions` and drops plates that never reached a `parked` or
`exited` state. This removes the long tail of false-positive log rows
from the training set.

## Expected accuracy gains

| Scenario | Baseline (pretrained) | After fine-tune | Notes |
|---|---|---|---|
| Same-camera matches | 0.92 | 0.94-0.96 | Limited gain — the pretrained model already handles single-view well. |
| Cross-camera (gate -> floor) | 0.78 | 0.84-0.88 | Biggest win. Cross-view invariance is where facility lighting matters. |
| OCR-marginal band (0.40-0.55) | 0.55 | 0.65-0.72 | Hard-negative mining is what moves this number. |

Numbers assume >= 1500 positive pairs and >= 500 negatives. With less data
the gain shrinks proportionally and you may see no improvement.

## Workflow

1. Confirm `tools/calibrate_thresholds.py` shows cosine drift OR the
   end-to-end bench in `tests/test_matching_accuracy.py` reports < 0.90.
2. (Optional) Audit the mining output without training:

   ```
   python tools/finetune_osnet_facility.py --dry-run
   ```

   Inspect `models/osnet_finetuned/finetune_report.json` and confirm the
   positive / negative ratio is at least 3:1.
3. Run the full pipeline:

   ```
   python tools/finetune_osnet_facility.py --epochs 15 --batch-size 32 \
       --learning-rate 1e-4 --apply-export
   ```

   `--apply-export` rewrites `MatchingConfig.reid_openvino_model_dir`
   default in `src/config.py` to point at the new IR. Remove the flag if
   you want to A/B the new artifact via `config.yaml` instead.
4. Re-run `tools/calibrate_thresholds.py` and the E2E bench to confirm
   the accuracy regression test passes.

## Caveats

- **Plate label leakage:** the training pairs are stratified by plate so
  the validation set never sees a plate that appears in training. This
  prevents over-estimating val accuracy but does NOT prevent the
  training set from over-fitting to "easy" plates that appear in many
  cameras. Audit `finetune_report.json` for the plate-yield distribution
  before trusting val numbers > 0.95.
- **Triplet margin (`--margin 0.3`):** lower margin → easier loss → faster
  but shallower improvements. Bump to 0.4 once you have >= 5000 positives.
- **Hard-negative sourcing:** OCR-contradiction rejections are the gold
  source; the script falls back to cross-plate synthetic negatives when
  the rejection log is sparse, which produces an easier loss landscape.
  Phase 2 `match_voter_perf` rejections are NOT yet mined — a future
  revision should join this log too for richer negatives.
