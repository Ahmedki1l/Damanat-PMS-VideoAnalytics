# Matching Pipeline — Defect Report

**Scope:** vehicle↔plate matching path — detector, tracker, ReID, match decision, identity binding, snapshot/gallery capture.

**Method:** source audit of the real tree (`src/`, `tools/`, `tests/`, `models/`, `config.yaml`), plus three empirical checks that source reading alone could not settle:
- ran the detector IR to determine its true inference resolution;
- read the **live `config` DB row**, which is the actual runtime source of truth (see D8b);
- read `models/PS_carMatching.pt`'s provenance block and measured the aspect ratio of all 332 curated crops.

Every claim is cited to `file:line`. Claims of *absence* were established by repo-wide search.

> **Revision note.** Sections D6, D15 and D24 were revised after the live DB row, the `PS_carMatching.pt` checkpoint, and the sibling training project (`D:\Work\Spectech\Find-Tuning for ps\find tune\`) came to light. Each carries an inline note explaining what changed. **D15 changed materially and twice** — an identity-disjoint benchmark does exist, and the number the team has been quoting is the contaminated one. D24 is now largely closed.

**Summary of severity:**

| Sev | Count | Theme |
|---|---|---|
| Critical | 3 | A wrong plate can bind to a car and become permanent |
| High | 6 | Config says one thing, runtime does another |
| Medium | 10 | Unmeasured quality, disabled guards, silent divergence |
| Low | 7 | Stale docs, dead config, misleading artifacts |

Two findings dominate:

1. **ReID never selects which car receives the ANPR plate.** At the gate that decision is made by arrival order (D1). Every guard in the matching cascade is downstream of it.
2. **`config.yaml` is not the running configuration.** The `config` DB table overrides the detector, tracker, state-machine and assigner blocks — and currently disagrees with the YAML on four values, including a state-machine debounce of 1 frame where the YAML claims 5 (D8b). The `matching:` block, by contrast, *is* YAML-authoritative. Nothing in the file marks which is which.

---

## Critical

### D1 — CAM-23 Park_Entry binds the plate by FIFO, with no appearance check and no multi-car abstain

**Impact.** When two cars are inside the `Park_Entry` polygon within the 10 s bind window, the pending ANPR plate is attached to whichever car the tracker happened to list first. The wrong car is then carried through the rest of the pipeline as that plate.

**Root cause.** `_process_park_entry_zone` (`src/core/engine/engine_tracking.py:604`) iterates **every** in-zone detection (`:614`) and opens a `ParkEntryCandidate` per track. It never calls `_select_primary_zone_detection` — the single-best-car selector that *does* exist and *is* used by the CAM-03 confirmation zone (defined `engine_tracking.py:561`, used at `:701`).

The bind itself, `bind_next_pending_anpr_to_candidate` (`src/vehicle_registry/vehicle_registry_identity.py:764`), is explicitly a FIFO rule (docstring `:770`): it walks `_pending_event_order`, takes the first `direction == "entry"`, `status == "pending"` event whose age ≤ `PENDING_ANPR_BIND_TTL_SECONDS` (10 s, `vehicle_registry.py:55`), and binds it. **There is no feature vector, cosine score, colour, or geometry comparison anywhere in that function.** The first candidate to call it wins; the bind flips `event.status = "provisional"` (`:814`), so the second car finds nothing to bind and silently gets no plate.

The 10 s TTL is the *only* mitigation, and it was added for a different failure (the night-gate stale-plate swap), not for tailgating.

Note there are **three** FIFO call sites, not one: `engine_tracking.py:647`, `src/api.py:497`, `src/api.py:568`.

**Fix.** Two changes, both in `_process_park_entry_zone`:

1. Restrict candidate processing to the primary in-zone car, exactly as the CAM-03 path does:

```python
primary = self._select_primary_zone_detection(frame, detections, zone)
for detection in detections:
    if detection.track_id == -1:
        continue
    if primary is not None and detection.track_id != primary.track_id:
        continue
    ...
```

2. Abstain when the zone is ambiguous. Before calling `bind_next_pending_anpr_to_candidate`, count in-zone tracks; if more than one is present while a fresh pending entry exists, skip the bind for this frame and log it. A car that stays in the zone alone on a later frame will bind correctly; a genuine tailgate will simply never auto-bind, which is the safe outcome.

The right long-term fix is to make the gate use `bind_anpr_event_to_candidate` (`vehicle_registry_identity.py:829`), which binds by `event_id` and has no FIFO fallback. It is already wired — but only on the ANPR-image API path (`src/api.py:483`), which requires the integrator to POST the vehicle frame alongside the event. Confirm with the ANPR vendor whether that image is being sent; if it is, route CAM-23 through the identity-safe binder and delete the FIFO path.

**Verification.** Replay footage with two cars entering within 10 s and assert the plate lands on the car whose bbox has the highest zone-overlap at the moment the ANPR event timestamps.

---

### D2 — A mis-bind at the gate permanently poisons the plate's ReID gallery

**Impact.** The wrong car's crop is written to `gallery/<plate>/` as a **matchable** reference. Every future match for that plate is then measured partly against a foreign car, so the error is self-reinforcing and survives restart (the gallery is durable, retained 5 days).

**Root cause.** `seed_gallery_from_park_entry` (`src/vehicle_registry/vehicle_registry_identity.py:523`) reads `candidate.snapshot_image` (`:563`) and calls `store.save_ref(..., gate_only=False)` at `:579` — with **no `gallery_min_identity_similarity` check**.

That contamination guard genuinely exists, but in a *different function*. `_seed_plate_gallery_reference` computes `id_sim` against established refs and refuses the write below the floor (`:488-508`), then calls `save_ref` at `:510`. `seed_gallery_from_park_entry` never calls it. The in-memory enrichment in the same function gates only on `gallery_dedup_cosine`, not on identity.

The function's own docstring admits the consequence (`:536-538`): *"the plate binding here is still provisional (FIFO) — a mis-bind injects a wrong-plate CAM-23 crop; the shortened pending-bind TTL is the mitigation."*

This is worse than a bootstrap-only hole: it applies **even after** the plate's gallery already has established references.

**Fix.** Apply the same floor before the `:579` write, mirroring `:496-508`:

```python
floor = float(getattr(self._matching_config, "gallery_min_identity_similarity", 0.0))
established = store.load_vectors(plate)          # excludes gate_only refs
if floor > 0.0 and established:
    id_sim = max(compute_similarity(feature, ev) for ev in established)
    if id_sim < floor:
        logger.warning("[gallery] Rejected foreign Park_Entry seed for %s (id_sim=%.3f < %.2f)", plate, id_sim, floor)
        return False
store.save_ref(plate, crop, feature, quality=quality, camera_id=camera_id, gate_only=False)
```

Better still, refactor so **all three** `save_ref` call sites (`:510`, `:579`, `:2089`) go through one guarded admission helper. Three independent write paths with three different guard levels is the actual structural defect.

Note the guard fails open on an empty gallery, which is correct — the first crop has nothing to compare against. That is precisely why D1 must be fixed too: the identity floor cannot protect the *first* reference.

---

### D3 — A car lingering in the zone keeps an `open` candidate and can grab the next car's plate

**Impact.** A car queued or stopped inside `Park_Entry` (waiting for a barrier, for instance) will consume the plate of the car that arrives *behind* it.

**Root cause.** `bind_next_pending_anpr_to_candidate` requires `candidate.status == "open"` (`vehicle_registry_identity.py:777`). A lingering car that never bound (no pending plate existed when it arrived) keeps that status indefinitely, because the expiry sweep is keyed on `last_seen_at`:

```python
if candidate.status in ("open", "provisional"):
    age = (now - candidate.last_seen_at).total_seconds()
    if age > self.CANDIDATE_EXPIRY_SECONDS:      # 30 s, vehicle_registry.py:63
```
(`src/vehicle_registry/vehicle_registry_core.py:541-545`)

and `last_seen_at` is refreshed on **every frame** the car is in the zone (`vehicle_registry_core.py:686, 703, 749, 800`, via `update_park_entry_candidate_snapshot`). So `CANDIDATE_EXPIRY_SECONDS` never fires for a stationary in-zone car.

Nor does leaving the zone help. The exit path only deletes the *track→candidate map entry*, never closing the candidate itself:

```python
left_zone = last_track_ids - currently_inside
for track_id in left_zone:
    if track_id in self._park_entry_track_to_candidate:
        del self._park_entry_track_to_candidate[track_id]
```
(`src/core/engine/engine_tracking.py:681-684`)

So while the car remains in the zone, `bind_next_pending_anpr_to_candidate` is called for it on **every frame** (`:647`), and its `status` is still `"open"`. The moment a new ANPR event arrives, that lingering candidate binds it — **potentially before the car that actually triggered the read has even entered the zone.**

The realistic trigger: a car enters while no pending event exists (the ANPR missed it, or the plate arrived >10 s late), then queues at the barrier. Its candidate stays `open` indefinitely. The next car's plate is read — and the queued car takes it.

**Fix.** Expire the *bind eligibility* on a different clock from the *liveness* clock. `ParkEntryCandidate` **already carries `entered_at`**, set once at creation (`vehicle_registry_core.py:594`) and never refreshed — unlike `last_seen_at`. So this is a one-liner in `bind_next_pending_anpr_to_candidate`, using a field that already exists:

```python
if (now - candidate.entered_at).total_seconds() > self.PENDING_ANPR_BIND_TTL_SECONDS:
    return None   # this candidate has been sitting in the zone; it did not just trigger the ANPR
```

Rationale: an ANPR read fires as a car *crosses* the gate. A candidate that has existed longer than the bind TTL demonstrably did not just cross it. This is a strictly better anti-swap signal than `last_seen_at`, and it composes with D1's primary-car selection.

---

## High — config says one thing, runtime does another

### D4 — `global_default: 0.71` and `reattach_default: 0.68` never execute; the live bars are 0.55 and 0.52

**Impact.** Two of the three matching bars run **0.13–0.16 looser** than `config.yaml` advertises. Tuning those YAML values has literally zero runtime effect. The 2026-07-07 recalibration (which moved the envelope onto PS_carMatching's cosine distribution) was therefore only half-applied.

**Root cause.** Hardcoded Python signature defaults shadow the config:

- `match_global_session(..., similarity_threshold: float = 0.55, ...)` — `vehicle_registry_identity.py:2326`
- `reattach_track_to_confirmed_session(..., similarity_threshold: float = 0.52, ...)` — `vehicle_registry_identity.py:2738`

The only production callers omit the argument (`engine_tracking.py:1075` and `:1025`), so the defaults apply. Downstream, `decide_global` and `decide_reattach` only fall back to config when the value is `None`:

```python
base = float(similarity_threshold) if similarity_threshold is not None else cfg.global_default
```
(`src/matching/match_decision.py:282`, mirrored at `:336`)

Since `0.55 is not None`, `cfg.global_default` is dead on the no-plate branch. Same for `cfg.reattach_default` on the same-camera branch. The `has_plate` branch *does* honour YAML (`:288-293`), which is why this went unnoticed.

**Fix.** Change the two signature defaults to `None` so the existing config-fallback path activates:

```python
def match_global_session(self, ..., similarity_threshold: Optional[float] = None, ...)
def reattach_track_to_confirmed_session(self, ..., similarity_threshold: Optional[float] = None, ...)
```

This is a one-line change per function and requires no call-site edits. **Do not ship it blind** — it tightens the global bar from 0.55 → 0.71 and reattach from 0.52 → 0.68, which will reduce false binds but also reduce recall. Validate end-to-end first; the pipeline's abstain margin, 2-of-3 voting and ensemble already suppress false positives beyond the pairwise bench, so the right operating point may sit below 0.71.

### D5 — `b1_cross_camera: 0.43` is dead config

**Impact.** Cross-camera B1 confirmations use the same-view bars (`b1_anpr` 0.63 / `b1_zone` 0.71) instead of the intended lenient 0.43, so genuine cross-camera confirmations are rejected. Adjusting the knob does nothing.

**Root cause.** The key is parsed (`src/config.py:177` dataclass, `:621` loader) but **no code reads it**. `decide_b1` selects only between `cfg.b1_anpr` and `cfg.b1_zone` (`match_decision.py:204-209`). It even accepts a `cross_camera: bool = False` parameter (`:184`) that is never referenced in the function body.

**Fix.** Either implement it or delete it. To implement, mirror `decide_reattach`'s pattern:

```python
if is_anpr_candidate:
    threshold, threshold_reason = cfg.b1_anpr, "b1_anpr"
else:
    threshold, threshold_reason = cfg.b1_zone, "b1_zone"
if cross_camera:
    threshold, threshold_reason = min(threshold, cfg.b1_cross_camera), "b1_cross_camera"
```

Then confirm the caller (`confirm_at_b1_entrance`) actually passes `cross_camera=`. If no caller ever will, remove the key from `config.yaml`, `config.example.yaml` and `config.py` rather than leaving a knob that lies.

### D6 — The detector runs at 320×320; `config.yaml`'s `imgsz: 640` is dead and its comment is inverted

> **Revised after reading the live `config` DB row.** An earlier draft of this report claimed Ultralytics *silently overrides* 640→320. That is true only for a YAML-only run. In production the DB row already holds `imgsz = 320`, so no override occurs. The conclusion (the detector runs at 320) is unchanged; the mechanism is not.

**Impact.** The detector sees a 1280×720 frame downscaled ~4× linearly. A car 150 px wide in the frame is ~37 px at the network. This caps recall on small/distant vehicles on oblique floor cameras. Separately, `config.yaml` misleads every reader about what the detector actually does.

**Root cause — two independent paths that both land on 320:**

1. **In production**, `sync_app_config_from_db` overwrites `detector.imgsz` from the DB, which holds `320` (verified against the live `config` table). The YAML value is never consulted.
2. **Even if it were**, the OpenVINO IR is statically shaped — `<data shape="1,3,320,320" .../>` (`models/yolo11m_320_int8_openvino_model/yolo11m_320_int8.xml:5`), `metadata.yaml` records `imgsz: 320`, `args.dynamic: false`. Ultralytics overwrites the requested size from export metadata for a non-dynamic model. Confirmed empirically: passing `imgsz=640` yields `predictor.imgsz == [320,320]` and `pre_transform -> (320, 320, 3)`.

So `imgsz: 640` in `config.yaml` cannot take effect by either route. Worse, its comment is inverted: `imgsz: 640  # Lower resolution for CPU efficiency` — 640 is *higher* than the real 320.

**Fix.** The runtime is already consistent (DB 320 = IR 320). The defect is documentation and dead config:

- Set `imgsz: 320` in `config.yaml`, fix the comment, and add a note that detector fields are DB-owned (D8b). No behaviour change; the config simply stops lying.
- If you *want* 640, changing YAML is not enough: update the DB row **and** re-export the IR at 640 (or with `dynamic=True`), re-run INT8 calibration, re-benchmark. Expect a large latency increase — the 320 export exists for CPU throughput.

Because there is no detector accuracy benchmark (D14), you cannot currently quantify what 320 costs you in recall. Fix D14 first, then make this call with data.

### D7 — `use_lab_clahe: true` is a no-op on the production backend

**Impact.** Operators believe ReID crops are luminance-normalised under varying garage lighting. They are not.

**Root cause.** `VehicleReIDMatcher.extract_feature` calls the backend with no keyword:

```python
return _ensure_f32(self._backend.extract(image))     # src/reid_matcher/reid_matcher.py:376
```

and `OpenVINOReIDBackend.extract` declares `normalise_luminance: Optional[bool] = False` (`reid_openvino_backend.py:204`), so the CLAHE branch at `:175` never runs. The config value is parsed (`config.py:303`, `:694`) and read exactly once — into a logging dict:

```python
flags = {"CLAHE": cfg.use_lab_clahe, ...}     # vehicle_registry_identity.py:3249
```

(The torchreid fallback backend *does* apply CLAHE via its own `ReIDPreprocessingConfig`, but that is not the production path. The detector-side CLAHE at `config.py:102` is a separate, genuinely-enabled knob.)

**Fix — and a warning.** Do **not** simply thread the flag through. `PS_carMatching` was trained and INT8-calibrated on non-CLAHE crops; enabling CLAHE only at inference introduces a train/inference skew of exactly the kind that already cost this project 0.16 rank-1 once (the letterbox/squish incident documented at `tools/export_osnet_openvino.py:378`).

Two coherent options:

1. **Remove the flag** (`config.yaml`, `config.example.yaml`, `config.py`) and delete the dead `normalise_luminance` parameter from the OpenVINO backend. Cheapest, and matches current behaviour.
2. **Adopt CLAHE properly**: add it to the training transform in `tools/finetune_osnet_top20.py` *and* the calibration preprocess in `tools/export_osnet_openvino.py`, retrain, re-export, re-benchmark, then wire `extract_feature(image, normalise_luminance=cfg.use_lab_clahe)`.

Option 1 unless you have evidence lighting variance is hurting you.

### D8 — Commit `fbb60c2` did not do what its message says

**Impact.** The commit claims *"Zone vehicle crops (entrance, Park_Entry) are now masked to the zone's ROI polygon"*. Park_Entry crops are **not** masked. Anyone reading the log believes the gate crop is clean; it is not — and that crop is the one seeded into the gallery (D2).

**Root cause.** The diff changed exactly one call site, in `_process_confirmation_zone` (CAM-03):

```diff
-                crop = self._crop_detection(
+                crop = self._crop_detection_to_zone(
                     frame,
                     detection,
+                    zone,
                     padding_ratio=0.12,
+                    mask_outside_zone=True,
                 )
```
(`src/core/engine/engine_tracking.py:722`)

`_process_park_entry_zone` still calls `self._crop_detection(frame, detection)` at `:636` — raw bbox, `padding_ratio=0.0`, no mask.

**Fix.** Apply the same change at `:636`:

```python
crop = self._crop_detection_to_zone(frame, detection, zone, padding_ratio=0.0, mask_outside_zone=True)
```

Two caveats worth understanding before you do:

- The mask blacks out pixels outside the **zone polygon**, not outside the **car silhouette**. A second car sitting *inside* the polygon is fully retained. It removes the "white car parked outside the entrance" case; it does not solve tailgating (that is D1).
- A masked crop has large black regions where the car extends beyond the polygon. `PS_carMatching` was calibrated on unmasked `Cropped_Vehicles`, so masked crops are mildly out-of-distribution. Measure the cosine-margin impact on the eval set before rolling out; if it degrades, mask only for the *snapshot artifact* and keep the ReID embedding on the unmasked crop.

### D8b — The `config` DB table is the real source of truth, and it disagrees with `config.yaml` on four live values

**Impact.** `config.yaml` does **not** describe the running system. Four fields differ, and one of them — the state-machine debounce — changes occupancy behaviour substantially. Anyone debugging from the YAML is reasoning about a system that does not exist.

**Root cause.** `main.py:271` calls `sync_app_config_from_db`, whose docstring states plainly: *"This makes the database the source of truth for runtime settings."* (`src/services/config_service.py:79-82`). It overwrites the detector / processing / tracker / state-machine / assigner / output fields from the DB row (`:92-120`). YAML only seeds the table when it is empty (`ensure_config_initialized`).

I read the live `config` row. It diverges from `config.yaml` as follows:

| Field | `config.yaml` | **live DB row** | Consequence |
|---|---|---|---|
| `detector.confidence` | 0.25 | **0.35** | A materially higher detection bar than the YAML comment ("Low bar") describes |
| `detector.imgsz` | 640 | **320** | Matches the IR; the YAML value is dead (D6) |
| `state_machine.confirm_enter_frames` | 5 | **1** | A slot is marked occupied after a **single** frame |
| `state_machine.confirm_leave_frames` | 8 | **3** | And vacated after 3 |
| `assigner.overlap_threshold` | 0.2 | **0.3** | Stricter slot-polygon overlap |

`model_path`, `classes`, `tracker_type`, `target_fps`, `stream_channel` and the slot reference resolution do agree.

The `confirm_enter_frames = 1` value is the one to look at: the debounce that `config.yaml` documents as a 5-frame confirmation is, in production, no debounce at all. A single spurious detection marks a slot occupied. At ~5 fps this is a meaningful false-occupancy source, and it interacts badly with D10 (a passing truck is a parkable vehicle).

Note also that the `matching:` block has **no columns** in the `Config` model (`src/model/config_run.py:14-90`) — I checked every column. So all matching thresholds, gallery settings and voting config are pure YAML and are never DB-overridden. **This asymmetry is the single most confusing thing about this codebase's configuration**, and is why D4/D5/D6 are easy to misdiagnose: some of `config.yaml` is authoritative, some of it is decorative, and nothing marks which is which.

**Fix.**

1. Annotate `config.yaml` inline: mark every DB-owned block `# DB-OWNED — edit the config table, not this file` and every YAML-owned block `# YAML-authoritative`.
2. Log the **effective** config after `sync_app_config_from_db` — at minimum `model_path`, `imgsz`, `confidence`, `confirm_enter/leave_frames`, `overlap_threshold`. Right now nothing prints what actually took effect.
3. Reconcile the four divergences deliberately: decide whether `confirm_enter_frames = 1` is intentional (it may have been set to chase a latency complaint) and either update the DB or the YAML so they agree.
4. Consider a `--config-source={yaml,db}` flag to force YAML for debugging.

Related first-boot hazard: `DetectorConfig.device` defaults to `"auto"` (`config.py:77`), but `DeviceEnum` defines only `cpu` and `cuda` (`src/schemas/config_run.py:10-12`). Seeding an empty table writes `"auto"` into an `Enum(DeviceEnum)` column. Add `AUTO = "auto"` to the enum, or resolve `auto` → `cpu`/`cuda` before seeding. (The live row holds `CPU`, so this has not bitten yet.)

---

## Medium

### D9 — Nothing in the system measures occlusion; `view_quality` is blind to a neighbouring car

**Impact.** In a garage with cars parked shoulder-to-shoulder, an oblique axis-aligned bbox contains background and slivers of the adjacent vehicle. That crop passes every quality gate and is embedded as if it were clean. This is the root cause of "the bbox includes the car next to it".

**Root cause.** `_bbox_view_quality` (`src/core/engine/engine_tracking.py:418-448`) is:

```python
edge = (ix * iy) / (bw * bh)                                  # frame-border truncation
size = (bh - _VQ_MIN_H) / (_VQ_GOOD_H - _VQ_MIN_H)            # apparent height, 45→90 px
return edge * size
```

It captures truncation and apparent size. It has no occlusion term. A repo-wide search confirms **no detection-vs-detection IoU exists anywhere** — every `overlap` in the codebase is detection-vs-*zone* (`_zone_overlap_ratio` `:575`, `_detection_overlaps_zone` `:712`). NMS (`iou=0.7`, Ultralytics default, never overridden) suppresses duplicate boxes of the *same* car; it does not tighten the retained box or reject contaminated ones.

**Fix.** Add a neighbour-occlusion factor to `view_quality` and let the existing `gallery_min_view_quality = 0.9` gate do the rest:

```python
def _neighbour_clearance(self, detection, detections) -> float:
    """1.0 when no other vehicle box overlaps this one; →0 as a neighbour covers it."""
    box = _to_shapely(detection.bbox)
    worst = 0.0
    for other in detections:
        if other.track_id == detection.track_id:
            continue
        inter = box.intersection(_to_shapely(other.bbox)).area
        worst = max(worst, inter / max(1.0, box.area))
    return max(0.0, 1.0 - worst)
```

then `return edge * size * clearance`. Because `gallery_min_view_quality` is already 0.9, even ~10% neighbour coverage will now exclude the crop from the gallery. Tune on real frames — start by *logging* the clearance distribution for a day before enforcing it, so you learn what fraction of current gallery refs are contaminated.

This is the highest-leverage quality fix in the report: it improves ReID input quality globally, not just at the gate.

### D10 — `class_id` is written and never read; buses and trucks are treated as cars

**Impact.** Classes 5 (bus) and 7 (truck) are admitted deliberately, because SUVs and vans are often misclassified as truck. But a *genuine* bus or truck is then indistinguishable from a car and can claim a parking slot.

**Root cause.** `Detection.class_id` is populated at `src/detection/tracker.py:221` (and `detector.py:190`). A repo-wide search for readers of `.class_id` returns **zero results**. Slot/zone logic uses only `bottom_center` geometry. The registry's `vehicle_type` comes from the separate MobileNetV3 type classifier, not from YOLO.

**Fix.** The type classifier already distinguishes `bus` (`models/type_classifier_openvino/labels.json`, 6 classes, test acc 0.907). Either gate slot occupancy on the classifier's verdict, or add a size sanity check (a bus bbox will exceed any slot polygon's area by a wide margin). The cheapest correct move is to keep `[2,5,7]` for detection recall and reject occupancy when the type classifier says `bus` with high confidence.

### D11 — The first camera to start gets `track_buffer=30`; every other camera gets 60

**Impact.** Asymmetric lost-track survival. Camera #1 drops occluded tracks roughly twice as fast as the rest, causing more ID switches and more orphan-reattach work on that one feed.

**Root cause.** `_new_tracker` sets `cfg.track_buffer = 60` (`src/detection/tracker.py:114`). But on the very first `detect_and_track` call the predictor does not yet exist, so Ultralytics builds `trackers[0]` itself from the stock `bytetrack.yaml` (`track_buffer: 30`), and the code then *adopts that instance* for the first camera:

```python
if camera_id not in self._camera_trackers and predictor is not None and getattr(predictor, "trackers", None):
    self._camera_trackers[camera_id] = predictor.trackers[0]
```
(`src/detection/tracker.py:183-189`)

**Fix.** Overwrite the adopted tracker rather than inheriting it:

```python
if camera_id not in self._camera_trackers:
    self._camera_trackers[camera_id] = self._new_tracker()
```

placed *before* the first `model.track()` call, so the swap at `:167` has something to install. Alternatively, force-set `predictor.trackers[0].max_time_lost` after bootstrap.

Related: `_new_tracker` passes `frame_rate=30` (`:125`) while its own comment reasons in ~1.3 fps. ByteTrack computes `max_time_lost = frame_rate/30 * track_buffer`, so the effective buffer is 60 frames — which at the real ~5 fps is ~12 s, not the ~46 s the comment claims. Correct the comment or pass the real frame rate.

### D12 — Two safety guards are disabled in production

**Impact.** An ANPR misread that produces two different plate strings for one physical car has **no** safety net: both become confirmed identities and each can independently bind a slot.

**Root cause.** Both are off by config value, and both short-circuit before doing anything:

- `anpr_min_accept_confidence: 0.0` — the hold at `src/api.py:747-754` requires `min_conf > 0.0`, so it never fires. (It is also a genuine no-op until the ANPR server starts sending a per-read confidence.)
- `identity_reconcile_min_similarity: 0.0` — `_reconcile_duplicate_identity` early-returns `[]` when the floor is ≤ 0 (`vehicle_registry_identity.py:3640-3642`).

`_claim_plate_globally` only dedupes by *plate string*, so it cannot catch two different misread strings for one car.

**Fix.** These are correctly built and deliberately parked, so the fix is operational, not structural:

- `anpr_min_accept_confidence` — ask the ANPR vendor to include per-read confidence, then set ~0.8. Until then it cannot be enabled; leave a comment saying so (it already has one).
- `identity_reconcile_min_similarity` — the config comment recommends ~0.75. Validate on multi-car footage first, because enabling it *closes real sessions*. Suggested rollout: run in shadow mode (log what it *would* merge) for a week, review the merges, then enable.

### D13 — `MatchingConfig` dataclass defaults are far weaker than the YAML, and silently disable the anti-swap guard

**Impact.** Any construction path that does not load `config.yaml` — a test, a tool, a missing `matching:` block — reverts to an envelope calibrated for a superseded model, including **`single_candidate_min_reid = 0.0`**, which re-opens the exact night-gate blind-bind swap that guard was written to close.

**Root cause.** `src/config.py:158-368` defaults vs `config.yaml`:

| Key | code default | production YAML |
|---|---|---|
| `single_candidate_min_reid` | **0.0 (off)** | 0.35 |
| `gallery_min_identity_similarity` | **0.0 (off)** | 0.35 |
| `b1_anpr` | 0.47 | 0.63 |
| `b1_zone` | 0.55 | 0.71 |
| `reid_solo_confirm` | 0.70 | 0.86 |
| `gallery_max_refs_per_car` | 10 | 20 |

The two safety floors defaulting to `0.0` is the dangerous part: absence of config silently means *absence of protection*.

**Fix.** Make the dataclass defaults equal the calibrated production values, so a missing config is safe rather than permissive. Safety floors in particular (`single_candidate_min_reid`, `gallery_min_identity_similarity`) should default to their enabled values, and any intentional disable should be explicit in YAML.

### D14 — There is no detector benchmark and no tracker benchmark

**Impact.** Detector recall/precision and tracker ID-switch rate — the two numbers that determine whether "the bbox is on the right car" — have never been measured. Any claim about detector or tracker performance for this facility is currently unsupported.

**Root cause (evidence of absence).** `tools/bench_yolo.py` measures throughput only (inferences/s, fps/camera) — its own docstring says so. A case-insensitive repo-wide search for `MOTA`, `IDF1`, `HOTA`, `id_switch`, `fragmentation`, `motmetrics`, `trackeval` returns no files. Every `eval_report*.json` in the tree is a ReID/plate-matching eval (keys `n_plates`, `threshold`, `fine_tuned`), not a detection eval.

**Fix.**

- **Detector:** label ~200 frames sampled across floors, cameras and times of day (the oblique B1/B2 views are the hard case). Run `mAP@50`, and — more importantly for this system — **recall at the operating confidence 0.25**, broken down by bbox height bucket. That directly answers whether 320×320 (D6) is costing you distant cars.
- **Tracker:** annotate track IDs on a few multi-minute clips containing slot transitions and cross-aisle passes. Report **IDF1** and **ID-switch count**, not MOTA — for this application, identity continuity is the whole game. The `track_buffer` question (D11) cannot be settled without it.

Both belong in `tools/` alongside `calibrate_thresholds.py`, and both should be wired into `tests/` as regression floors.

### D15 — The headline ReID number is measured on an eval set that is 65% contaminated by training identities

> **Revised twice.** Draft 1 claimed the eval set was *fully* closed-set — that describes the top-20 split which produced the `osnet_facility_*` bakeoff candidates, not PS_carMatching. Draft 2 corrected this to 65% overlap and asserted that **no open-set benchmark exists anywhere**. That assertion was also wrong: an identity-disjoint benchmark exists in the sibling training project (D24), and its number is already recorded in the shipped checkpoint. The corrected picture is below.

**Impact.** The number quoted internally, `rank1 0.7975 / mAP 0.7981`, is the **contaminated** one — 13 of its 20 evaluation identities were in PS_carMatching's training set (56% of eval images). The **uncontaminated** number is `rank1 0.7368 / mAP 0.505`, and it has been sitting in `PS_carMatching.pt` and `models/PS_carMatching/eval_report.json` all along. **We have been quoting the wrong figure.**

**The two benchmarks, side by side:**

| | `eval_report_ps_carmatching.json` | `PS_carMatching/eval_report.json` |
|---|---|---|
| Source split | `data/facility_top20` | `find tune/reid_data/splits.json` |
| Identities | 20 (13 seen in training) | **20, all held out** |
| Identity-disjoint? | **No** — 65% overlap | **Yes** — verified, pid ranges `0–60` vs `100000–100019`, zero intersection |
| Gallery size | 79 images | **2,022 images** |
| Query size | (pairwise, 1095 pairs) | 20 images |
| rank-1 | 0.7975 | **0.7368** |
| mAP | 0.7981 | **0.505** |

The `find tune` split is exactly what an honest ReID benchmark looks like: train on 61 vehicles, evaluate on 20 vehicles the model has never seen, against a 2,022-image gallery. `dataset_stats.txt` even names the held-out vehicles.

**Read the mAP gap carefully.** 0.798 → 0.505 is a 0.29 drop, but **two effects are confounded**: identity contamination *and* gallery size (79 vs 2,022 images — mAP falls as the gallery grows, regardless of model quality). Do not attribute the whole gap to contamination. What is safe to say is that the open-set/large-gallery figure is the one that resembles production, and it is **substantially lower** than the number in circulation.

**Caveat on the honest number itself:** the query set is only **20 images, one per vehicle**. `rank1 0.7368` is 14 correct out of 19 scored queries — a 95% confidence interval of roughly 0.49–0.91. It is directionally right and far better than the contaminated figure, but it is a small sample and should not be quoted to four decimal places.

Remaining problems with the *facility_top20* bench, unchanged:

- **14 of 20 plates use `random_fallback` splitting** (only 6 use `temporal`, `split_report.json`), so train and eval crops for those cars can be adjacent frames of the same drive-through.
- 8 plates contribute a single eval image, so per-identity rank-1 is 0 or 1 for nearly half the set.
- No **end-to-end multi-camera handoff** bench exists in either project. That gap is real and stands.

**Root cause.** The checkpoint's own metadata records `num_classes = 61` and the full `id_names` list. Cross-referencing it against the eval plates:

- **Seen in training (13/20):** `HGD-2926, HUD-9444, LNV-94, NDD-4141, NJS-7894, NXR-2727, RDJ-9640, SDD-6707, XBD-5588, XRD-6663, ZDR-8501, ZRS-6511, ZVH-337`
- **Genuinely held out (7/20):** `AGA-6649, EEB-80, ERD-7800, HBR-4920, HVA-77, LRS-9439, RTB-2016`
- By image count: 44/79 (56%) seen, 35/79 (44%) unseen.

Mitigating fact worth noting: the single largest eval contributor, `HBR-4920` (19 images, 24% of the eval set), is **unseen**. So the micro-average is less contaminated than the identity count alone implies.

Remaining problems with the bench, unchanged:

- **14 of 20 plates use `random_fallback` splitting** (only 6 use `temporal`, `split_report.json`), so train and eval crops for those cars can be adjacent frames of the same drive-through.
- 8 plates contribute a single eval image, so per-identity rank-1 is 0 or 1 for nearly half the set.
- No **identity-disjoint**, open-set, or end-to-end multi-camera-handoff bench exists anywhere in the repo. `models/PS_carMatching/eval_report.json` explicitly disclaims being the end-to-end bench.

**Fix.** The benchmark already exists; the job is to adopt it, not to build it.

1. **Change the number we quote.** `rank1 0.737 / mAP 0.505` (identity-disjoint, 2,022-image gallery) is the headline. Retire `0.7975 / 0.7981` or label it explicitly as a contaminated closed-set diagnostic.
2. **Promote the split into this repo** (D24 step 1), so `tests/test_facility_match_accuracy.py` can regress against the honest number instead of the contaminated one (D16).
3. **Grow the query set.** 20 queries is too few to detect a regression. `Cropped_Vehicles/` holds 13,764 images across 140 identities; the 20 held-out vehicles have 2,022 gallery images between them. Promote a proper share of those to query and re-score — this costs compute only.
4. Fix the `facility_top20` bench regardless: force **temporal** splitting (drop `random_fallback`), and report per-identity rank-1 rather than a micro-average one car dominates.

For threshold work, keep using `threshold_calibration.json` (precision 0.933 / recall 0.737 at 0.71) — a *pairwise* number, unaffected by gallery size. But note it was calibrated on the contaminated `facility_top20` pairs, so **it should be recomputed on the held-out vehicles** before the next threshold change.

### D16 — The accuracy test asserts no accuracy floor, and pins a stale threshold

**Impact.** `tests/test_facility_match_accuracy.py` cannot catch a model regression. A model that scores rank-1 0.10 passes.

**Root cause.** The only assertions are (`:634-635`):

```python
assert report["baseline"] is not None
assert 0.0 <= report["baseline"]["rank1"] <= 1.0
```

An `_acceptance_verdict` (≥3 of 4 metrics beat baseline) *is* computed and written into the report, but never asserted. Separately, `DEFAULT_THRESHOLD = 0.53  # b1_zone` (`:72`) is the **old** `b1_zone`; production is 0.71. Every `eval_report_*.json` therefore reports pair precision/recall at an off-operating-point threshold (0.393 / 0.853), badly understating production precision (0.933 / 0.737).

**Fix.** Assert the acceptance verdict and a hard rank-1/mAP floor for the production IR. Read `DEFAULT_THRESHOLD` from `config.yaml`'s `matching.b1_zone` rather than hardcoding it, so the reports track the deployed operating point automatically.

### D17 — Voting only suppresses single-frame flicker, not systematic mismatches

**Impact.** `voting_min_agree: 2` of `voting_window_frames: 3` is described as protection against wrong matches. It is protection against *noisy* matches. A stable track that consistently matches the same wrong plate wins 2-of-3 and commits.

**Root cause.** `MatchVoter` is keyed per `(camera_id, track_id)` and tallies the winning plate key across the window (`src/matching/match_voter.py:205-221`). It re-tallies decisions; it never re-examines appearance. By construction it cannot detect a consistent error.

**Fix.** No code change required — this is a documentation and expectations defect. Record in the config comment that voting removes flicker only, and that systematic mis-binding is addressed by D1/D2/D9. Do not count voting as a mitigation when reasoning about gate mis-binds.

### D18 — Cross-camera bars of 0.43 sit inside the negative-pair distribution

**Impact.** `global_cross_camera` and `reattach_cross_camera` are both 0.43. The measured **negative**-pair cosine distribution has mean 0.404 and p90 0.550 (`threshold_calibration.json`). A bar of 0.43 is roughly the *mean of the wrong-car distribution*.

**Root cause.** This is a deliberate trade documented in `config.yaml`: cross-floor handovers are inherently low-similarity, and the candidate pool is already bounded to cars that departed an adjacent area within the transit window. The pool bound is real (`vehicle_registry_identity.py:2802-2806` same-floor/handoff gate, plus the colour veto).

But the only appearance filters on that bounded pool are the 0.43 cosine and the colour veto. Two same-colour cars on the same floor can cross-bind.

**Fix.** Do not raise it blindly — that would strand plates on the origin floor, which is why it is low. Instead:

1. Collect real cross-floor footage (B1 → RAMP-DN → B2) and calibrate `*_cross_camera` on it, exactly as `threshold_calibration.json` did for same-view. The config comment already says this is pending.
2. Until then, tighten the *pool* rather than the *threshold*: require the transit-time window to be respected strictly, and add the body-type classifier as a second veto (it is already loaded and scores 0.907 test accuracy — a sedan cannot become an SUV across a ramp).

---

## Low — documentation, dead config, misleading artifacts

### D19 — `reid_openvino_backend.py` module docstring contradicts the code

`:20-22` states *"BGR → RGB letterbox to (H, W) with 127-grey padding"*. `_preprocess` at `:183` does `cv2.resize(rgb, (target_w, target_h))` — an aspect-destroying squish. The *method* docstring (`:164-173`) correctly documents the 2026-05-15 switch away from letterbox; the module docstring was never updated. **Fix:** update `:20-22`. This is exactly the kind of stale note that caused the original 0.16 rank-1 regression.

### D20 — `MatchVoter` docstrings describe defaults that are not the defaults

`:12-13` says *"Default config (3-of-5 …)"* and `:89` says *"Defaults are `False / 5 / 3` — feature is off by default."* Production is enabled, 2-of-3. The code reads config live, so runtime is correct. **Fix:** update both docstrings.

### D21 — `config.example.yaml` is dangerously stale

Every matching threshold is ~0.10–0.18 below production (`b1_zone: 0.53` vs `0.71`, `reid_solo_confirm: 0.68` vs `0.86`), and it omits roughly ten keys production relies on (`gallery_*`, `lock_confidence`, `single_candidate_min_reid`, `global_match_margin`, `identity_reconcile_*`, `multishot_ref_top_k`, `anpr_min_accept_confidence`). A new deployment started from the example ships a mis-calibrated system. **Fix:** regenerate it from `config.yaml` with secrets stripped, and add a CI check that the key sets match.

### D22 — Dead config keys

Parsed, never read: `reid_solo_confirm_threshold` (`config.py:333`, `:711` — the code uses `reid_solo_confirm`), `legacy_color_fallback` (`:249`, `:629`), `processing.per_camera_tracker` (`:67`, `:513`, already labelled DEPRECATED in the example file). `processing.mode` is read only to print a status label (`engine.py:254`) — no code branches on it. **Fix:** delete them; each one is a knob that invites a wasted tuning session.

### D23 — `tools/finetune_osnet_facility.py` is a scaffold, and its preprocessing contradicts inference

Three separate problems in the script the README advertises as the production fine-tune path:

- It **letterboxes** with a 127-grey canvas (`:817-823`), while inference squishes. Training with this script and serving through the OpenVINO backend reproduces the known 0.16 rank-1 regression.
- It never splices the fine-tuned checkpoint into the exported model — it only writes the checkpoint *path* into metadata, and is documented as a future limitation (`:1040`).
- Its docstring claims *"pretrained MSMT weights"* (`:14`), but `torchreid`'s `pretrained=True` for `osnet_ain_x1_0` loads ImageNet weights — confirmed by every `source_weights_hash: osnet_ain_x1_0_imagenet.pth` in the model metadata.

The real artifacts came from `tools/finetune_osnet_top20.py` (which squishes correctly, `:142`). **Fix:** delete `finetune_osnet_facility.py` or mark it clearly experimental, and correct the README so nobody runs it expecting a production model.

### D24 — `PS_carMatching`'s provenance is recoverable from the checkpoint, but its training script and data are not in the repo

> **Revised.** An earlier draft called the model "not reproducible" and its hyperparameters "unrecorded". `models/PS_carMatching.pt` in fact carries a provenance block. Most of the mystery is resolved; what remains is narrower.

The checkpoint records:

| Field | Value |
|---|---|
| `arch` | `osnet_ain_x1_0` |
| `num_classes` | **61** identities (not the 20 of the top-20 split) |
| `input_hw` | `[256, 128]` |
| `epoch` | 20 (of 60; epoch-20 selected) |
| `init_from` | `osnet_facility_finetune_20260515.pt` |
| `source` | `find tune/scripts/03_train.py` (warm-started) |
| `rank1` / `map` | 0.7368 / 0.505 |

Two things fall out of this:

- **The lineage question is answered.** `init_from` points at the May facility fine-tune, which its own log shows was CARLA-initialised. So the full chain is **ImageNet → CARLA (VeRi-CARLA) → facility top-20 (May) → 61-identity warm-start retrain (June)**. PS_carMatching is transitively CARLA-initialised.
- **The "two conflicting eval reports" is not a conflict.** The checkpoint's self-reported `rank1 0.7368 / map 0.505` matches `models/PS_carMatching/eval_report.json` exactly. That report is the *checkpoint's own eval on its own 61-identity split*. `eval_report_ps_carmatching.json` (0.7975/0.7981) is the separate facility_top20 bench. Different protocols, both valid, never to be mixed. Of the two, the 61-identity number is on the larger and probably harder split.

**The training project has since been located** at `D:\Work\Spectech\Find-Tuning for ps\find tune\` (a sibling directory, outside this repo). It contains the full pipeline: `scripts/00_check_env.py` … `01_crop_vehicles.py`, `02_build_reid_dataset.py`, `03_train.py`, `04_export_onnx_int8.py`, `eval_compare.py`, plus `Cropped_Vehicles/` (140 identity folders, 13,764 images), `reid_data/splits.json`, `reid_data/dataset_stats.txt`, and `runs/osnet_ain_facility/`.

**Chain of custody is now fully verified:**

```
runs/osnet_ain_facility/osnet_ain_x1_0_facility.pth   sha256[:16] = 77560afc590de23d
models/PS_carMatching/metadata.yaml source_weights_hash = 77560afc590de23d   ✓ match
PS_carMatching.pt['id_names']  ==  splits.json train vehicles (61)           ✓ exact
```

So the shipped IR provably derives from that checkpoint, trained on that split. The model is reproducible — just not *from this repository*.

**Fix.** Bring the provenance into the repo so it cannot be lost again:

1. Vendor `scripts/` and `reid_data/splits.json` + `dataset_stats.txt` into `tools/reid_finetune/` (small text; no binaries).
2. Add to `models/PS_carMatching/metadata.yaml`: `id_names` (the 61 training vehicles), `num_classes: 61`, `epoch: 20`, `init_from`, `source_script`, and the `splits.json` hash. Then the training identity list is discoverable without loading a `.pt`.
3. Record a dataset manifest (relative paths + per-file hashes) rather than the 13,764 images themselves.
4. Keep `Cropped_Vehicles/` where it is, but pin its location and a content hash in the metadata.

This is now cheap, and — see D15 — it also surfaces an **identity-disjoint benchmark that we did not know existed**.

### D25 — The ReID input is a person aspect ratio, and it costs horizontal detail on cars

Not a correctness bug — training and inference agree on the squish, so the model works. But the geometry is inherited from person-ReID, and the cost is measurable rather than theoretical. I measured all 332 curated facility crops (`data/facility_top20/{train,eval}`):

| Quantity | Value |
|---|---|
| Crops that are landscape (W > H) | **294 / 332 = 89%** |
| Median crop | 902 × 649 px (aspect **1.40**) |
| p10 / p90 aspect | 0.96 / 1.87 |
| ReID tensor | 128 W × 256 H (aspect **0.50**) |
| Shape distortion | 0.50 / 1.40 = **0.36** |

The median car is squeezed to **36% of its true width-for-height**. Concretely, mapping a 902×649 crop onto 128×256 resamples horizontally by 0.14× and vertically by 0.39× — **horizontal detail is discarded ~2.8× more aggressively than vertical**. (Both axes downsample; the earlier framing of "pixels invented vertically" was wrong for the median crop and holds only for crops shorter than 256 px.)

That asymmetry runs against the grain of the signal. For vehicles, the discriminative structure — grille, headlight spacing, window line, body length, badge position — is predominantly *horizontal*. The network is being handed the axis it needs most, at the resolution it needs least.

`256×128` is simply torchreid's default for standing humans (`DEFAULT_INPUT_HW=(256,128)`, `tools/finetune_osnet_top20.py`), carried over unexamined.

**Fix (opportunity, not defect).** Fine-tune a landscape variant (`128 W × 256 H` → i.e. `input_hw=[128, 256]`) or a square `224×224`, and compare against the current model on the **identity-disjoint** eval from D15. Keep everything else fixed. Two cautions:

- OSNet ends in a global average pool, so it accepts other input shapes without architectural change — but the IR is exported with a static `256×128` spatial shape, so this requires a re-export, not just a reshape.
- Do **not** evaluate by feeding landscape inputs to the *current* weights. That reproduces the letterbox/squish mismatch that already cost 0.16 rank-1 (`tools/export_osnet_openvino.py:378`). It must be a retrain.

This is the most likely single source of remaining ReID headroom — but D15's honest benchmark has to exist first, or you will not be able to tell whether a change helped.

---

## Recommended order of work

0. **D8b (the four DB/YAML divergences)** — thirty minutes of work, and until it is done every other number in this report is being read against the wrong config. Specifically, confirm whether `confirm_enter_frames = 1` is intentional. Free, and it de-risks everything below.
1. **D15 + D14** — build an identity-disjoint ReID eval (the training identity list is now recoverable from `PS_carMatching.pt['id_names']`) and a detector recall benchmark. *Nothing else on this list can be safely validated without them.* The cheapest possible first step: re-score the existing eval on the 7 genuinely held-out plates.
2. **D1 + D3 + D2** — fix the gate. Primary-car selection, bind-eligibility on `entered_at` (a one-liner against an existing field), guarded gallery admission. These are the only defects that produce a **permanent, self-reinforcing** wrong identity.
3. **D9** — add the neighbour-clearance term to `view_quality`. Highest-leverage global quality fix; log the distribution before enforcing.
4. **D13** — make unsafe defaults safe, so a missing config cannot disable an anti-swap floor.
5. **D4 + D5 + D6 + D7** — reconcile the remaining config with runtime. Each is small; together they restore the ability to reason about the system from its config file.
6. **D8, D10, D11, D12, D16–D18** — as capacity allows.
7. **D19–D25** — documentation and hygiene; cheap, and D19/D23 have already caused one real regression. D24 is now nearly free: commit the training script and copy `id_names` into `metadata.yaml`.

---

## Appendix — what is working well

Worth stating, because the defect list is long and unbalanced on its own:

- The **decision cascade** (`match_decision.py`) is carefully built. The K-of-N rule's `reid_floor` + `high_entropy_agree` clause (`:461-476`) correctly prevents low-entropy modalities (colour, body type) from confirming an identity on their own — a subtle failure mode most implementations get wrong.
- **Abstain-on-ambiguity** (`global_match_margin`, `:2603`) is the right instinct: refusing to guess between two similar cars is better than a coin flip, because a wrong plate is unrecoverable.
- **`gate_reference_only`** cleanly prevents the wide ANPR gate shot from false-matching a car already parked inside.
- The **rank-5 OCR disambiguation** (`:2540`) uses a high-entropy signal exactly where ReID is weakest — among visually similar top candidates.
- **`_claim_plate_globally`**, **upgrade-only plate replacement**, the **temporal entry-anchor**, and the **locked-slot freeze** form a coherent, layered anti-swap design.

The architecture is sound. The defects are concentrated at one seam — the gate, where identity is *assigned* by time rather than by appearance — and in the drift between what the config says and what the code does.
