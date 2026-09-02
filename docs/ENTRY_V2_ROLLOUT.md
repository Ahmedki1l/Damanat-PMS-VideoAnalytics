# Entry V2 validation and rollout

This document is the deployment contract for the authoritative gate-entry
pipeline shared by PMS-AI and Video Analytics (VA). It changes neither Gateway
nor the database schema. Cameras continue to post only to PMS-AI.

## Safety invariants

- An ANPR read alone never opens a parking session in authoritative mode.
- A session is opened only after a trusted gate attempt, configured physical
  entry evidence, a unique ReID assignment, and an OCR decision agree. Physical
  evidence may be the CAM-23 Hikvision line event, VA's local CAM-23
  `Park_Entry` RTSP transition, or the explicitly configured downstream CAM-03
  `B1_Entrence` fallback.
- CAM-23 OCR is primary. If CAM-23 reports no plate, an unreadable plate, or a
  below-threshold plate, the cached OCR from the ANPR vehicle crop is used
  automatically. A reliable CAM-23 read or multi-text conflict never falls back
  to the ANPR result.
- A corrected plate needs exact evidence. Fuzzy text, arrival order, FIFO, and
  "last reading wins" are not identity rules.
- The physical crossing that confirms the journey supplies the entry time. The
  trusted ANPR camera remains the gate-source provenance.
- Full request images are decoded and discarded in RAM. The exact plate ROI
  produced by LPD and passed to OCR is retained as an operator-requested
  diagnostic under `<snapshot_base_dir>/entry_plate_crops`; its filename
  records camera, source role, frame index, OCR confidence, and read state.
  These diagnostics are not decision inputs, database records, or public
  snapshots. A later parked-car crop remains the separate strict-gallery
  persistence path described below.
- There is no attempt/crossing business TTL. Bounded capacity returns `503`
  instead of evicting evidence. False or retreated evidence is removed through
  the authenticated cancellation endpoint.
- Network, database-lock, inference, and request-size timeouts remain operational
  safeguards. They do not decide whether a vehicle entered.

## End-to-end flow

1. Hikvision posts the ANPR multipart event to PMS-AI.
2. PMS-AI authenticates the source network, parses the event, resolves the
   camera identity/direction, extracts only bounded vehicle crops, and forwards
   an idempotent multipart attempt to `POST /api/v2/entry-attempts` on VA.
   It sends `X-Entry-V2-Mode`; VA rejects a mode mismatch before parsing the
   multipart body or running inference.
   Missing/malformed camera timestamps use an aware PMS receive instant with
   explicit provenance. Stable raw/image IDs make an identical camera retry
   idempotent even though its fallback receive time changes. This fallback is
   allowed for non-destructive entry evidence only; a destructive exit without a
   trustworthy camera time returns camera-facing HTTP 503 instead of being
   ordered against a newer visit using delivery time. PMS accepts source images
   only inside its independent 30 MP decode envelope, then downscales every
   outbound crop to VA's exact 12 MP / 8192-pixel-per-side envelope.
3. VA extracts OSNet ReID embeddings. It also runs the local OpenVINO plate
   detector and PaddleOCR on the plate ROI from the cached ANPR vehicle crop.
   Every image in one multipart burst must pass the same-car consistency gate;
   mixed/tailgating bursts fail closed. After OCR scores an LPD crop, VA saves
   that crop atomically to `entry_plate_crops` for offline diagnosis. A save
   failure is logged but never changes the entry decision. The original request
   bytes and full vehicle frame are then released.
4. CAM-23 may produce either or both independent physical signals:
   - Hikvision posts the line event to PMS-AI. PMS-AI requires an active vehicle
     event and a real inward direction, or an explicitly calibrated one-way
     line, then forwards `POST /api/v2/entry-crossings` to VA. When a verified
     inward CAM-23/CAM-03 event has a usable JPEG but its firmware omits the
     target rectangle, PMS forwards a bounded, downscaled full-frame fallback
     in RAM instead of silently losing the crossing. This path must be field
     calibrated as a single-vehicle view: one image produces one ReID embedding,
     so software cannot prove that a full frame containing multiple cars belongs
     to the line target. Authoritative rollout is no-go for that fallback until
     captured field payloads prove the view isolates one vehicle; otherwise fix
     the Hikvision target rectangle/crop configuration first.
   - VA observes the RTSP `Park_Entry` polygon directly. After the zone was seen
     empty, exactly one stable tracked vehicle crossing it produces an in-process
     primary crossing with bounded JPEG crops. No HTTP loopback is used and the
     source vehicle frames are not retained; only any smaller LPD ROI later
     passed to OCR is written to the diagnostic folder described above.
5. VA applies the same transition state machine to CAM-03's downstream
   `B1_Entrence` polygon and submits it as fallback evidence. If no blocking
   primary evidence has arrived, CAM-03 may confirm the journey. Arrival at
   CAM-03 is the fallback trigger; there is no timer that can prove a delayed
   CAM-23 event will never arrive.
6. VA extracts the crossing vehicle ReID embedding and plate OCR. ReID must pass
   the absolute threshold plus row and column uniqueness margins. Simultaneous
   or mixed vehicles fail closed. Equivalent Hikvision and VA-local CAM-23
   notifications that are present before resolution form one physical-evidence
   family for assignment. OCR from every present member remains visible so a
   conflict abstains; see the late-producer limitation below.
7. VA applies the OCR policy:
   - readable CAM-23 text matching a reported hypothesis confirms it;
   - no, unreadable, or low-confidence CAM-23 text may use the cached ANPR-crop OCR;
   - when CAM-03 is the physical fallback, readable CAM-03 OCR is used first,
     then cached ANPR-crop OCR only if CAM-03 OCR is absent, unreadable, or weak;
   - readable disagreement abstains unless exact independent correction evidence
     satisfies the configured evidence and camera counts.
8. VA sends a metadata-only authenticated decision to
   `POST /api/v1/internal/entry-confirmations` on PMS-AI.
9. PMS-AI serializes confirmation and exit work with the same normalized-plate
   SQL Server application lock, then uses the existing `EntryExitLog` and
   `ParkingSession` models to create the entry atomically. No new table is used.
10. After PMS-AI returns a semantic acknowledgement, VA seeds the validated
   image-free identity into the live registry. In a zoned deployment, CAM-23
   must map unambiguously to an initial area.
11. PMS-AI reconciles `zone_occupancy.current_count` from existing open parking
    sessions in the same entry/exit workflow. Line events only wake this
    reconciliation; they never increment or decrement an aggregate.

If an exit has already happened, PMS-AI returns terminal
`result=stale_after_exit`; VA compacts the evidence without publishing a live
inside-garage identity. If PMS committed but its HTTP response was lost, an
identical retry is idempotent. A closed session or later exit is still returned
as `stale_after_exit`, not as a live duplicate.

If a validated new crossing finds an older open PMS session for the same plate,
PMS closes the old stay at the new crossing timestamp using the audit marker
`SYSTEM-REENTRY-RECONCILE` and opens the new stay atomically. It does not invent
an exit-camera log. Conversely, an older callback arriving after an equal/newer
open stay returns terminal `result=superseded_by_newer_entry`; VA compacts that
evidence without publishing the stale identity.

This ordering policy depends on NTP-aligned ANPR, CAM-23, PMS, and VA clocks. A
physically old exit whose timestamp is skewed past the next crossing can be
interpreted as belonging to the newer stay; clock synchronization is therefore
an authoritative-rollout prerequisite, not an optional observability check.

## Plate correction and pending evidence

Two ANPR reads that uniquely ReID as the same car merge into one attempt group.
If the first plate was wrong and the later exact plate agrees with CAM-23, the
wrong hypothesis is marked superseded and only the corrected plate is committed.

An unrelated earlier attempt is never deleted because a later vehicle confirms.
It remains pending until it receives valid evidence or an operator/system sends
`POST /api/v2/entry-cancellations` with its attempt, group, or crossing ID.
The same authenticated endpoint can remove one proven-unmatchable exit marker by
its exact normalized plate and offset-aware camera timestamp:

```json
{
  "exit_plate": "ABC-1234",
  "exit_captured_at": "2026-07-22T12:15:00+00:00",
  "reason": "operator verified legacy or false exit"
}
```

Both exit fields are mandatory together. The operation is RAM-only and
idempotent: `removed_pending_exits` is `1` for the removal and `0` on an exact
retry. Copy the exact plate/timestamp pair from the retained-exit log; never
approximate a timestamp or cancel by plate alone.

Stable camera retry identifiers deduplicate exactly. The explicitly independent
Hikvision and `va_local_zone` CAM-23 producers may report the same physical car;
strict same-car evidence collapses only that declared producer pair for ReID
column assignment. Generic distinct event IDs remain distinct evidence, and OCR
from every event is retained. Delayed or out-of-order crossings are held in
bounded, RAM-only provisional metadata/embedding state after successful PMS
delivery. A source-earlier ANPR attempt can release a genuine re-entry crossing;
otherwise that crossing cannot match the next car.

The selected no-time ordering policy is an **ANPR-exit-bounded open journey**.
After PMS accepts a confirmation, VA retains a compact finalized journey until
PMS forwards that plate's authenticated, offset-aware ANPR exit timestamp. A
strict Hikvision/`va_local_zone` twin from the confirming CAM-23 stage and
producer-pair window is consumed immediately. Other same-car CAM-23/CAM-03
observations remain bounded provisional evidence while the journey is open,
even when exact OCR agrees. This is intentional: a genuine re-entry crossing
can be delivered before its physically earlier exit webhook.

When the exit arrives, VA classifies retained evidence by source time. A
crossing captured at or before the exit is compacted as part of the closed
journey; one captured after the exit is released to normal unique ReID/OCR
assignment. The result is independent of HTTP arrival order and uses no
business timer. A late old event delivered after closure is still deduplicated
when its source timestamp, topology, OCR, and ReID agree. If the exit bridge is
missing, provisional capacity eventually returns 503 rather than evicting or
guessing; the exit route itself is not subject to V2 ingress capacity.

One exit closes at most one same-plate open journey: the uniquely latest
source-ordered eligible journey. If two candidates share the same latest entry
timestamp, VA closes neither, retains the exit boundary, and exposes the
ambiguity for reconciliation. Exit boundaries that race ahead of callback or
live-registry publication are retained. When multiple same-plate confirmations
are already reserved, VA waits for all eligible callbacks to settle and then
closes the uniquely latest source-ordered journey; callback completion order is
not an identity rule. An exact retry is idempotent. `/api/health` exposes
journey load, pending exits, and the cumulative ambiguous-exit count. Any
unmatched exit degrades health; exact operator cancellation decrements that
count immediately, and full lifecycle capacity is unhealthy and returns
retryable HTTP 503.

Reliable OCR that contradicts a committed journey is never swallowed as a
look-alike duplicate: VA logs a warning, increments
`late_ocr_conflict_count`, and leaves that crossing eligible for its own
source-causal attempt. The already committed first decision is not
retroactively rewritten; a late contradiction is therefore an operational
alarm and a labelled-shadow review item.

The local transition detector is precision-first: it must initially observe the
zone empty for two processed frames, then the same single tracked vehicle for two
processed frames. Short detector omissions are UNKNOWN, not proof that a waiting
vehicle left. After a visit, it re-arms promptly only after seeing that same track
outside the polygon for two consecutive processed frames; tracker-loss fallback
requires eight consecutive empty processed frames, and a reappearance resets the
count. Startup with a vehicle already inside, mixed vehicles, missing tracking,
or unusable crops never fabricates a crossing. These are observation-state
transitions, not business timeouts.

The local bridge does not infer a travel vector from one polygon; it labels the
transition with the configured synthetic direction. Before authoritative use,
field validation must prove `Park_Entry` and `B1_Entrence` are downstream,
physically one-way regions where a reverse/retreat movement cannot produce the
same empty-to-inside transition. Otherwise use directional line-side or two-zone
geometry instead of enabling the local policy.

Duplicate aliases seen in one captured frame count once. Local OCR/ReID/callback
work runs off the camera post-processing thread; one transient ingest failure is
retried with the identical crossing ID and bytes. Any worker failure remains
latched in local-zone health metrics even if that retry succeeds.

## Service-local configuration

`PMS_API_URL` intentionally means the peer service in each container, so its
value is different on the two services. Use plain URLs, not Markdown link text.

PMS-AI example:

```dotenv
PMS_API_URL=http://pms-video-analytics:8000
ENTRY_V2_MODE=shadow
ENTRY_V2_SERVICE_KEY=<same-random-secret-on-both-services>

# Required before authoritative mode. Resolve this from the effective connection
# peer seen in PMS logs; use verified exact /32 peers where possible.
CAMERA_EVENT_ALLOWED_SOURCE_CIDRS=<verified-effective-peer>/32

# Alias only a unique, immutable, entry-only identity (prefer deviceSerial).
# Never alias UNKNOWN-<shared-proxy-or-NAT-IP> to CAM-ENTRY: that peer may carry
# both entry and exit camera events.
ENTRY_V2_CAMERA_ALIASES=<verified-entry-device-serial>=CAM-ENTRY

ENTRY_CONFIRM_CAMERAS=CAM-23,CAM-03
ENTRY_CONFIRM_DIRECTIONS=CAM-23:ramp-entry,CAM-03:B-entry

# Prefer the raw Hikvision direction. Add only a measured one-way line when the
# camera omits direction and that line cannot represent outward traffic.
ENTRY_V2_ONE_WAY_LINES=CAM-23:<calibrated-line-id>

ENTRY_V2_MAX_IMAGE_BYTES=4194304
ENTRY_V2_MAX_SOURCE_IMAGE_BYTES=16777216
ENTRY_V2_MAX_IMAGES=4
ENTRY_V2_MAX_DECODED_PIXELS=12000000
ENTRY_V2_MAX_IMAGE_DIMENSION=8192
ENTRY_V2_MAX_SOURCE_DECODED_PIXELS=30000000
ENTRY_V2_APPLOCK_TIMEOUT_MS=1000

# Existing CAM-03 inward calibration used when its Hikvision event has no bbox.
OCCUPANCY_ENTRANCE_ZONES=1
OCCUPANCY_EXIT_ZONES=2
FORWARD_DIRECTION_FIELD=B-to-A
```

When PMS uses `shadow` or `authoritative`, startup validates the peer as a
credential-free absolute HTTP(S) URL, requires a non-empty service key, and
rejects zero/non-finite transport timeouts, non-positive image limits, or a
crop padding ratio outside `0..0.5`. A malformed effective value therefore
prevents startup instead of turning every camera crop into a silent rejection.

VA example:

```dotenv
PMS_API_URL=http://pms-ai:8080
ENTRY_V2_MODE=shadow
ENTRY_V2_SERVICE_KEY=<same-random-secret-on-both-services>
VA_PROCESS_COUNT=1
VA_SINGLE_PROCESS=1

ENTRY_V2_PRIMARY_CAMERAS=CAM-23
# Keep the real Hikvision values and add the canonical local-zone values.
ENTRY_V2_PRIMARY_LINES=<calibrated-hikvision-line-id>,Park_Entry
ENTRY_V2_PRIMARY_DIRECTIONS=<calibrated-raw-hikvision-inward-direction>,ramp-entry

# Downstream physical fallback. PMS forwards Hikvision's raw CAM-03 line and
# direction when the camera supplies them; VA's local one-way zone uses the
# canonical values. Allowlist BOTH representations. The current PMS calibration
# is raw line `1`, raw inward direction `B-to-A`; replace those two values if the
# captured production payload proves different. The polygon must be calibrated
# as one-way inward. This is separate from cached ANPR-crop OCR fallback.
ENTRY_V2_FALLBACK_CAMERAS=CAM-03
ENTRY_V2_FALLBACK_LINES=1,B1_Entrence
ENTRY_V2_FALLBACK_DIRECTIONS=B-to-A,b-entry

ENTRY_V2_MAX_PENDING_ATTEMPTS=256
ENTRY_V2_MAX_PENDING_CROSSINGS=256
ENTRY_V2_MAX_PENDING_CALLBACKS=128
ENTRY_V2_MAX_CONCURRENT_INGEST_REQUESTS=2
ENTRY_V2_RECEIPT_CAPACITY=4096
ENTRY_V2_JOURNEY_CAPACITY=4096
ENTRY_V2_MAX_IMAGES=4
ENTRY_V2_MAX_IMAGE_BYTES=4194304
ENTRY_V2_MAX_DECODED_IMAGE_PIXELS=12000000
ENTRY_V2_MAX_DECODED_IMAGE_DIMENSION=8192
ENTRY_V2_MAX_METADATA_BYTES=16384

# 0.75 is unreachable — the highest ANPR->ramp score ever observed is 0.689, so
# nothing confirms. The score is a floor against unusable crops; the margins do
# the discriminating. Measurement block is in the Dockerfile.
ENTRY_V2_REID_MIN_SCORE=0.20
ENTRY_V2_REID_ROW_MARGIN=0.08
ENTRY_V2_REID_COLUMN_MARGIN=0.08
ENTRY_V2_MERGE_MIN_SCORE=0.82
ENTRY_V2_MERGE_MARGIN=0.08
ENTRY_V2_EVENT_CONSISTENCY_MIN_SCORE=0.82
# Maximum source-timestamp separation for Hikvision and va_local_zone reports
# to represent one physical crossing. This is not evidence expiry.
ENTRY_V2_PRODUCER_PAIR_MAX_SKEW_SECONDS=5.0
# Producer-pair dedup requires stricter ReID than ordinary entry matching.
ENTRY_V2_PRODUCER_PAIR_MIN_REID_SCORE=0.95
ENTRY_V2_OCR_MIN_CONFIDENCE=0.75
ENTRY_V2_CORRECTION_MIN_EVIDENCE=2
ENTRY_V2_CORRECTION_MIN_CAMERAS=2

# Stricter than live session admission. These values authorize a future parked
# gallery reference only; they do not change the normal entry decision.
ENTRY_V2_GALLERY_ANPR_MIN_CONFIDENCE=0.90
ENTRY_V2_GALLERY_REID_MIN_SCORE=0.85
ENTRY_V2_GALLERY_REID_ROW_MARGIN=0.12
ENTRY_V2_GALLERY_REID_COLUMN_MARGIN=0.12
ENTRY_V2_GALLERY_OCR_MIN_CONFIDENCE=0.90

ENTRY_V2_CALLBACK_TIMEOUT_SECONDS=5.0
ENTRY_V2_CALLBACK_MAX_ATTEMPTS=1
ENTRY_V2_CALLBACK_INITIAL_BACKOFF_SECONDS=0.2
ENTRY_V2_CALLBACK_MAX_BACKOFF_SECONDS=2.0
ENTRY_V2_CALLBACK_RETRY_INTERVAL_SECONDS=5.0

ENTRY_V2_LPD_MODEL_DIR=models/yolo11n_lpd_openvino_model
ENTRY_V2_LPD_CONFIDENCE=0.30
ENTRY_V2_LPD_IOU=0.45
ENTRY_V2_LPD_THREADS=2
ENTRY_V2_OCR_MODEL_DIR=
```

`ENTRY_V2_RECEIPT_CAPACITY` must be at least
`ENTRY_V2_MAX_CONCURRENT_INGEST_REQUESTS`; VA rejects an unsafe configuration.
`ENTRY_V2_JOURNEY_CAPACITY` independently bounds open/retained journey state
and unmatched exit boundaries. It defaults to `4096`; protected open journeys
are never evicted, so exhausted capacity rejects new attempts or exit markers
with HTTP 503 until an authoritative exit closes a journey or an operator uses
the authenticated exact-key cancellation above to remove a verified false,
legacy, or otherwise unmatchable pending record. In-flight evidence already
materialized as a group or journey counts once, even while its callback/receipt
is completing.
`ENTRY_V2_PRODUCER_PAIR_MAX_SKEW_SECONDS` must be finite and greater than zero.
It limits only cross-producer family deduplication by source timestamp; it does
not expire attempts, crossings, or provisional evidence.
`ENTRY_V2_PRODUCER_PAIR_MIN_REID_SCORE` must be finite, within `-1..1`, and at
least the greatest of `0.90`, `ENTRY_V2_EVENT_CONSISTENCY_MIN_SCORE`, and
`ENTRY_V2_MERGE_MIN_SCORE`.
When either canonical RTSP zone is enabled, `ENTRY_V2_MAX_IMAGES` must be at
least `2`, because a local crossing requires two stable processed frames.
That path also fails closed unless the coordinator can be proven to be
co-located with every camera whose local zone is enabled, by one of two routes:

- `VA_SINGLE_PROCESS` explicitly one of `1`, `true`, `yes`, `on`
  (case-insensitive) — one process feeds every camera, so co-location is
  trivially true; or
- `VA_ENTRY_HOST=1` together with a `VA_GROUP_CAMERAS` list containing those
  cameras. Both are set by `supervisor.py` on the single group it launches with
  `--api`, and cleared on every other group.

Neither route present produces
`entry_v2_local_zone_requires_single_process_or_gate_group`.

**Do not set `VA_SINGLE_PROCESS=1` merely to clear that error.** It is an engine
switch, not an attestation: `main.py` also reads it to force `VA_INFER=async`
and to call `engine.run_single_process()`. Under the multi-process supervisor
that gives every group its own async inference queue — the BUILD 4 topology that
was reverted on measured throughput. The second route exists precisely so the
configuration check can be satisfied without changing the engine.
The callback retry interval is an operational delivery cadence, not an evidence
expiry. Transient failures stay in a bounded RAM queue. Permanent authentication
or callback-contract failures are not retried: VA stops admission and reports
unhealthy. Any callback backlog reports degraded health; a full queue reports
unhealthy/HTTP 503.
HTTP 408, 425, 429, and 5xx callback responses remain retryable and must not
turn a transient proxy/server condition into permanent coordinator shutdown.

## Strict durable gallery

The gallery is VA-local and changes neither Gateway nor any database table. A
crop is written only after both evidence stages pass:

1. After PMS has acknowledged the authoritative session, the exact-plate path
   requires ANPR confidence at least `0.90` and matching CAM-23 primary OCR (or
   the explicitly configured CAM-03 physical fallback) at least `0.90`. The
   corrected-plate exception does not trust the wrong/missing ANPR confidence;
   it requires both the configured number of independent OCR evidence IDs and
   distinct cameras to agree on the correction. Those witnesses may combine
   cached OCR of the ANPR vehicle crop with selected CAM-23/CAM-03 crossing OCR;
   the raw camera ANPR text under review is never counted. Both paths require
   ReID score at least `0.85`, row margin at least `0.12`, and column margin at
   least `0.12`.
2. The exact parked crop must independently OCR-match the canonical active plate
   at least `0.90` and be its unique ReID rank-1 result with score at least `0.70`
   and runner-up margin at least `0.15`. Plate detection is mandatory and full-car
   OCR fallback is disabled. Immediately before the disk commit, VA reranks the
   exact extracted crop against the current live candidate set; a vacate, rebind,
   newly stronger candidate, or new OCR ambiguity causes an abstention. The exact
   10%-padded crop must be clear of every other vehicle detection, including
   untracked detections, and must also pass geometry, minimum area, sharpness,
   body-colour, ground-truth similarity, and duplicate checks.

These values are hard safety floors when strict persistence is enabled. YAML may
raise them but configuration loading rejects lower values or unsafe LPD/rank
modes. The admitted parked crop is the only image retained as ReID/gallery
history. Full ANPR and crossing request images remain transient; their LPD
plate ROIs are the separate diagnostic-only exception described above.

Abstained, shadow, pre-ACK, cached-ANPR-only, OCR-consensus-only, ReID-only,
transit, ambiguous, low-confidence, and generic tracking paths cannot write.
Every saved reference records its scalar entry and parked evidence in
`meta.json`; each file and metadata replacement is atomic, and an evidence ID
makes retries idempotent. Strict reload verifies both crop and embedding digests.
Each vector carries its own ReID model tag; mixed/model-upgraded histories are
re-embedded from their verified crops instead of comparing incompatible vectors.
Pre-existing records without verified admission metadata are excluded from ReID
immediately and moved out of the active tree to `gallery_quarantine/` on the
first verified save for that plate, preserving them for operator audit. That
move is a restart-recoverable two-phase operation: `meta.json` first removes the
records from the active set and records their exact pending source/destination
names, then the files move and the audit records become `complete`. An
interrupted or failed move remains fail-closed as `pending` and is retried on the
next strict save.

Use these `matching` values in VA's effective `config.yaml`:

```yaml
matching:
  plate_ocr_model: "enabled"
  slot_lpd_enabled: true
  slot_lpd_fallback_enabled: false
  slot_lpd_model_dir: "models/yolo11n_lpd_openvino_model"
  slot_lpd_confidence: 0.30
  slot_lpd_iou: 0.45
  slot_lpd_num_threads: 2

  gallery_persist_enabled: true
  gallery_max_refs_per_car: 10
  gallery_retention_days: 0.0
  gallery_strict_admission_enabled: true
  gallery_parked_ocr_min_confidence: 0.90
  gallery_parked_reid_min_score: 0.70
  gallery_parked_reid_min_margin: 0.15
  gallery_parked_min_neighbour_clearance: 0.90
  gallery_parked_require_rank_one: true  # mandatory; false fails config loading
  gallery_require_slot_authority: true
  gallery_min_view_quality: 0.90
  gallery_min_sharpness: 40.0
  gallery_min_crop_area: 12000.0
  gallery_accumulate_min_gt_similarity: 0.45

output:
  snapshot_base_dir: "vehicle_images"
```

Do not bake the production `config.yaml` or an environment file into the image.
Mount the effective configuration read-only at `/app/config.yaml` and inject
environment variables through the deployment platform.

With the Dockerfile's `/app` working directory, mount durable storage at
`/app/vehicle_images` (or set `output.snapshot_base_dir`/`SNAPSHOT_PATH` to the
mounted path). Without that volume, correct files still disappear when the
container is replaced. `gallery_retention_days: 0.0` disables automatic gallery
expiry, so verified history survives future return visits until an explicit
plate reset. Monitor disk capacity and change this only when the facility has an
approved retention policy.

Entry V2 creates `entry_plate_crops/` beneath that same resolved image root on
the first successful LPD crop. No additional path variable is required. Each
JPEG filename contains `ocr-confidence-<0.0000>` and `readable` or `unreadable`.
The general snapshot HTTP route explicitly refuses this directory. There is no
automatic retention policy for these diagnostics, so production must monitor
and manage its disk usage.

In V2 modes, legacy VA ingress is authenticated before its body is read and
capped by the ASGI transport guard. PMS uses the same service key. The legacy
multipart upload and `/api/line-crossing` routes return HTTP 410 before body
parsing. Shadow retains the authenticated JSON legacy entry path because the
legacy flow remains authoritative and must keep VA's live identity populated;
V2 full-frame and coordination evidence stays RAM-only; OCR plate-ROI
diagnostics follow the explicit filesystem exception above. In authoritative
mode the same JSON route returns HTTP 410 for every non-exit direction before
registry or image work.
Only the bounded JSON exit bridge remains once V2 owns entry admission.
PMS authenticates that bridge with the same service key. In active V2 modes,
the exit bridge remains available when entry admission itself is unhealthy, so
already-admitted vehicles can still clear VA identity. An invalid/unknown V2
mode never falls back to unauthenticated legacy ingress; guarded routes return
an unavailable response until configuration is corrected.
Transport/5xx and boundary/configuration responses (401, 403, 404, 405, 408,
425, 429) keep the exact plate, direction, and source timestamp in the existing
metadata-only exit delivery spool; deterministic payload errors (400, 413, 422)
are dropped. Active-V2 exit records never age out: their source timestamp makes
delayed replay safe even after a newer re-entry, while expiring them could leave
stale VA identity with no remaining recovery path.
If PMS committed an exit but crashed before the first forward/spool action, the
camera's duplicate retry recreates that idempotent exit forward from the retry's
exact source timestamp while skipping duplicate PMS/session/alert mutations.
For an accepted active-V2 exit, VA echoes that already-validated offset-aware
source instant in `ANPREventResponse.timestamp`; it does not echo the registry's
clock-normalized internal timestamp. PMS treats a missing, naive, or different
timestamp in an otherwise HTTP-2xx response as a semantic delivery failure and
keeps the metadata-only record for retry.
Unexpected authoritative camera-event dispatch or DB commit failures return
HTTP 503 plus `Retry-After: 1`; shadow retains the legacy HTTP-200 behavior.
The same applies when both live exit delivery and its metadata-only spool write
fail (for example, disk full or permissions): authoritative returns 503 so the
duplicate camera retry can recreate the idempotent forward; shadow keeps the
legacy acknowledgement contract.

## Runtime and restart constraints

- Authoritative Entry V2 requires exactly one VA process because attempts,
  crossings, callback reservations, and image-free embeddings are coordinated
  in process. Startup fails closed when `VA_PROCESS_COUNT > 1`.
- Motion `shadow` or `enforce` requires each worker to use the single-process
  engine via explicit `VA_SINGLE_PROCESS=1`. A supervisor deployment may run
  multiple independent camera-group workers when that variable is inherited by
  every child; each child owns its cameras' motion state. This does **not** make
  authoritative Entry V2 multi-process-safe, which still requires
  `VA_PROCESS_COUNT=1` and one replica.
- Active Entry V2 with either canonical local zone also requires an explicitly
  true `VA_SINGLE_PROCESS`; otherwise configuration fails closed before local
  zone evidence can be admitted.
- `VA_PROCESS_COUNT=1` must be explicitly present and valid in authoritative
  mode. This is only a process-local attestation: deployment must also enforce
  one Kubernetes replica/pod, because two replicas can each claim `1`.
- Pending evidence is RAM-only and is lost on a VA restart. A restart after PMS
  committed but before VA published the live identity also loses the required
  embeddings. Under the current no-file/no-new-table constraint this boundary
  cannot be made crash-durable. Drain/cancel before planned restarts and keep
  authoritative production **no-go across unplanned restarts** unless the owner
  accepts a manual reconciliation procedure or authorizes durable compact
  (non-image) evidence/PMS replay.
- Already-admitted verified parked gallery references are restart-durable only
  when `snapshot_base_dir` is backed by the persistent volume described above.
- In a zoned VA configuration, the canonical CAM-23 camera entry must have an
  `area:` assignment. Missing or ambiguous mappings fail identity publication
  and leave the idempotent callback pending for retry.
- A gallery-index failure rolls back the new live registry identity. PMS callback
  idempotency allows the VA retry to complete the handoff safely.

## Motion scheduling and slot hysteresis

Motion is per camera and only schedules YOLO. It never writes occupancy or
declares a vehicle absent. Each camera has an independent frame-difference
baseline, stream epoch, active hold, transition latch, and quiet sentinel.
Stale, skipped, failed, and reconnect frames produce `UNKNOWN`, never `ABSENT`.
Entry V2 primary/fallback cameras and configured entrance zones bypass motion
gating automatically.

`VA_MOTION_SCHEDULER_MODE=legacy|shadow|enforce` overrides
`motion_scheduler.mode`, and `VA_SLOT_STATE_MODE=legacy|shadow|time` overrides
`state_machine.mode`. Record the effective deployment values during rollout;
the environment takes precedence over YAML.

When `motion_scheduler.mode=enforce` and `state_machine.mode=time`, every global
or per-camera `sentinel_interval_seconds` must be less than or equal to
`max_known_gap_seconds`; configuration loading fails otherwise.

Roll out `motion_scheduler.mode` from `legacy` to `shadow`, calibrate per camera,
then use `enforce`. Roll out `state_machine.mode` from `legacy` to `shadow`, then
to `time`. A `LEAVING` slot remains publicly occupied until timed absence is
confirmed. The transition debounce fields are `enter_cancel_seconds=1.0`,
`enter_cancel_min_observations=2`, `leave_start_seconds=1.0`, and
`leave_start_min_observations=2`. See `config.example.yaml` for all fields.

## Go/no-go rollout gates

1. Keep both services `off` while deploying the code and secrets.
2. Set both services to `shadow`; never mix modes.
3. Calibrate the real CAM-23 line and inward direction from captured Hikvision
   payloads. PMS forwards the raw Hikvision direction when present, so VA's
   allowlist must use that exact raw value (for example `B-to-A`), not the
   legacy semantic marker `ramp-entry`. Use the semantic configured direction
   only for a proven one-way line whose camera omits direction. Do not leave the
   authoritative VA direction allowlist empty.
   Apply the same check to CAM-03: in shadow, verify that a PMS-forwarded raw
   crossing (`line_id=1`, `direction=B-to-A` for the current calibration) and a
   VA local-zone crossing (`line_id=B1_Entrence`, `direction=b-entry`) are both
   accepted under the fallback role. Any HTTP 422 is a rollout stop until the
   allowlists match captured traffic.
4. Verify effective camera source peers and every firmware camera alias. Do not
   use the observed shared peer `10.1.20.60` as an entry alias unless packet/payload
   evidence proves it is immutable and entry-only. Prefer unique device serials.
5. Run at least 24 hours of shadow traffic and label every would-confirm,
   abstention, correction, duplicate, and stale-after-exit case. The precision
   gate is zero wrong confirmations in the labelled sample.
6. Measure ReID row/column margins and OCR confidence on facility data; do not
   copy defaults into authoritative production without calibration.
7. Confirm one stable source ID per physical crossing and exercise explicit
   cancellation for a false/retreated vehicle.
8. Load-test bounded capacity, per-camera fairness, callback retry, source-body
   limits, and the decoded-pixel cap on the target Xeon.
9. Run a two-connection SQL Server concurrency test for exit versus confirmation.
   The invariant is: terminal stale/no session, or one paired closed session;
   never a later exit plus an open session.
10. Deploy PMS timestamp propagation/auth first; verify live and spooled exits
    contain offset-aware `captured_at`. Then deploy VA's strict source-time/auth
    enforcement. VA-first rejects older PMS exits with HTTP 422.
11. Exercise callback lost-ACK recovery with real containers and explicitly
    accept or resolve the RAM-only restart limitation above.
12. Verify the **effective container environment**, not only the Jenkins file.
    In particular, the PMS value must be the literal plain URL
    `PMS_API_URL=http://pms-video-analytics:8000`, never
    `[http://...](http://...)`; VA must use `http://pms-ai:8080`.
13. Run the full VA suite inside the production dependency image (Ultralytics,
    Torch, OpenVINO, Paddle) and the full PMS suite against SQL Server.
14. Enable authoritative mode on both services in one controlled change, with a
    rollback to `shadow` prepared. Monitor `503` backpressure; do not convert it
    to HTTP 200 at a proxy or camera integration layer.

Authoritative production remains a no-go until all gates above pass. Shadow mode
is the safe next deployment step.
