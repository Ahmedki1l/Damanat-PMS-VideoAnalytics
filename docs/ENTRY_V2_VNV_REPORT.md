# Entry V2 Verification and Validation Report

**Review date:** 2026-07-22

**Overall verdict:** **PARTIAL — locally verified; authoritative production NO-GO**

**Reviewed artifact:** the Entry V2 and motion/state-machine changes in PMS-AI and Video Analytics (VA)

The cross-repository Entry V2 design is substantially implemented and its
critical model-independent behavior passes locally. The selected post-confirm
ordering policy is implemented as an authenticated,
source-time **ANPR-exit-bounded open journey** and does not use a business timer
to decide whether retained CAM-23/CAM-03 evidence belongs to the completed stay
or a re-entry. The web runtime now uses FastAPI lifespan management and a
bounded FastAPI/Starlette/python-multipart compatibility window; the exact
resolved stack has its own clean-environment regression run. Authoritative PMS
also refuses an enabled CAM-23 whose line and direction filters are both empty,
with a parser-level fail-safe before every image path. Full production
acceptance is still not justified. Shadow/off
compatibility intentionally retains the legacy PMS multipart-snapshot write,
VA still loses process-local entry evidence across an unplanned restart, and
the live-camera, real SQL Server, production model/image, accuracy, and target
Xeon gates have not been executed.

**2026-07-22 requirement amendment:** the owner explicitly authorized one
image-persistence exception: a parked crop may enter VA's durable gallery only
after stricter post-ACK entry evidence plus crop-local parked OCR and rank-1 ReID
evidence pass. Authoritative Entry V2 request pixels remain transient; R08
records the separate legacy off/shadow qualification. The strict gallery is
local to VA and adds no Gateway or database-schema change.

This report uses one overall V&V verdict. Stage-specific gates are included only to make the next action unambiguous:

| Gate | Result | Meaning |
|---|---|---|
| Local implementation conformance | **PASS with qualifications** | Core authoritative behavior traces to code and passing local tests. |
| Shadow activation | **PARTIAL** | Logic is locally testable, but effective deployment, field calibration, and the intentional legacy snapshot compatibility exception must be controlled and disclosed. |
| Authoritative production | **NO-GO** | Restart durability and L3/L4 production validation remain open. |

## 1. Scope and baseline

### 1.1 Reviewed revisions

| Repository | Branch and baseline | Worktree state relevant to this review |
|---|---|---|
| PMS-AI | `develop` at `2066150755fc` | Entry V2, exit reliability, occupancy, configuration, docs, and tests are modified/untracked locally. |
| Video Analytics | `feat/plate-recognition-accuracy` at `ec5fa78708bf` | Entry V2, motion scheduling, slot hysteresis, registry handoff, docs, and tests are modified/untracked locally. |
| API Gateway | `main` at `1832eb755834` | Clean; no changes. |

This report records the implementation and test evidence present in the two
application worktrees at the review date.

### 1.2 Approved requirement baseline

The latest explicit user decisions take precedence over the earlier provisional plan. In particular:

- Cameras send events only to PMS-AI; PMS-AI and VA may communicate directly.
- API Gateway must not change.
- Existing database tables/models must be reused; no table or migration may be added.
- ANPR alone must never open an authoritative session.
- Physical entry must be validated by a configured inward crossing, strict ReID, and OCR.
- CAM-23 OCR is primary. OCR on the cached ANPR vehicle crop is fallback only when primary OCR is missing, unreadable, or below threshold.
- A wrong and later-correct ANPR hypothesis may merge only when strict evidence says they are the same car. Unrelated earlier attempts must not be deleted when a later car confirms.
- No business timeout, FIFO, arrival order, or last-reading-wins rule may decide entry. A vehicle may wait at the barrier for minutes.
- After confirmation, sequential same-car CAM-23/CAM-03 evidence follows the
  selected ANPR-exit-bounded open-journey policy. PMS's authenticated,
  offset-aware ANPR exit timestamp is the ordering boundary; missing exits
  backpressure at bounded capacity instead of causing eviction or a guess.
- Full entry request images remain transient. LPD plate ROIs passed to OCR may
  persist only in the private diagnostic folder requested by the operator; a
  later parked crop may persist separately through the strict, auditable
  gallery-admission exception described above.
- Motion is per camera and schedules inference only. Motion must never declare occupancy or vacancy.
- Gate-open/barrier state is out of scope.

The detailed executable contract is `docs/ENTRY_V2_ROLLOUT.md`. The original attachment that proposed provisional database rows and persisted buffers is superseded wherever it conflicts with the decisions above.

### 1.3 Evidence levels

| Level | Evidence |
|---|---|
| L1 | Static source/diff inspection. |
| L2 | Local automated unit/component/integration tests using fakes, SQLite, mocked SQL Server behavior, or import-only ML dependency stubs. |
| L3 | Real service containers with production model dependencies and SQL Server. |
| L4 | Live facility cameras, labelled traffic, target Xeon load, and operational soak. |

Only L1 and L2 are currently available. An acceptance criterion that needs L3 or L4 is marked **UNVERIFIED**, even when its implementation exists.

## 2. Bidirectional traceability matrix

Status meanings: **PASS** = sufficient evidence at the stated level; **PARTIAL** = some required evidence or scope is missing; **FAIL** = observed behavior contradicts the criterion; **UNVERIFIED** = implementation exists but required production evidence has not been collected.

| ID | Acceptance criterion | Implementation trace | Verification evidence | Status |
|---|---|---|---|---|
| R01 | Cameras ingress only through PMS-AI; PMS-AI forwards bounded internal evidence to VA. | PMS `app/routers/events.py::receive_camera_event`; `app/services/entry_v2_forwarder.py::forward_entry_v2_event`; VA `src/entry/router.py`. | PMS camera source/body-limit tests; VA authenticated/transport-guard contract tests. | **PASS L2**; effective production camera destinations and peers remain L4-unverified. |
| R02 | No API Gateway or database schema/table change. | Existing `EntryExitLog`, `ParkingSession`, and occupancy models are reused. | Gateway worktree is clean; no model, migration, Alembic, or SQL schema files changed. | **PASS L1** |
| R03 | ANPR alone cannot create a session; only a confirmed physical/identity decision may do so. | PMS authoritative dispatch bypasses legacy entry writes; `app/routers/entry_confirmations.py::confirm_entry`; `app/services/entry_confirmation_service.py::apply_confirmed_entry`. | Confirmation create/replay/abstain tests; authoritative legacy-bypass tests. | **PASS L2** |
| R04 | A trusted inward crossing, strict one-to-one ReID, and OCR must agree before confirmation. CAM-23 Hikvision or local `Park_Entry` is primary; strict local CAM-03 `B1_Entrence` polygon membership is the downstream fallback. | VA `src/entry/local_zone.py`, `src/core/engine/engine_tracking.py`, `src/entry/settings.py`, `src/entry/coordinator.py`, and `src/entry/decision.py::find_unique_match`. | Local-zone transition/integration, strict CAM-03 membership, host-grab timestamp provenance, direction/line contracts, absolute score, row-margin, column-margin, concurrent-assignment, and mixed-burst tests. | **PASS L2** for configured logic; one-way polygon geometry and facility thresholds are L4-unverified. |
| R05 | CAM-23 OCR is primary; ANPR-crop OCR is used only for absent/unreadable/low-confidence primary OCR. Reliable disagreement must abstain. | VA `src/entry/analyzer.py`; `src/entry/decision.py::resolve_plate`, `_resolve_primary`, `_resolve_fallback`, `_select_ocr`. | Primary exact-confirm, cached-fallback, low-confidence replacement, readable-conflict, and primary-before-fallback tests. | **PASS L2** |
| R06 | A later exact reading may correct the same car; wrong hypotheses are superseded; an unrelated earlier car remains pending. Source-causal stage chronology must prevent a much-later primary from stealing an earlier fallback. | VA attempt grouping, causal projection/partition, unique assignment, correction evidence, and compaction in `src/entry/coordinator.py` and `src/entry/decision.py`. PMS commits only `canonical_plate`. | Same-car merge/correction, late-correct supersession, unrelated-earlier-preservation, producer-pair, future-attempt, and adversarial stage-order tests. | **PASS L2** |
| R07 | No business TTL/FIFO/last-wins; long barrier waits remain eligible; capacity backpressures and false attempts are explicitly cancelled. Sequential post-confirm evidence is bounded by the authoritative ANPR exit source time. | VA `src/entry/coordinator.py::record_exit`, `_apply_exit_boundary_locked`, `_finalized_evidence_match_kind`, strict producer-pair deduplication, provisional crossing handling, exact pending-exit `cancel_pending`, and single-count capacity reservations; PMS source-time exit forwarding and non-ageing active-V2 exit spool. | Long-wait entry-time, no-eviction capacity, materialized-inflight single-count, strict producer-pair, pre-exit compaction, re-entry-before-exit-delivery release, late conflict, source-time, spool, and authenticated exact-key cancellation tests. | **PASS L2** for the selected ANPR-exit-bounded open-journey policy; effective exit delivery, NTP, and capacity under production failure remain L3/L4-unverified. |
| R08 | Full entry request images remain RAM-only. Only the LPD plate ROI actually passed to OCR may be written to the private `<snapshot_base_dir>/entry_plate_crops` diagnostic folder; it must not affect entry decisions, enter the database, or be exposed by the public snapshot route. | PMS transient vehicle cropping in `app/services/event_parser.py`; VA capability-checked in-memory multipart parsing, compact `FrameEvidence`, diagnostic persistence in `src/entry/analyzer.py`, resolved image-root wiring in `src/entry/runtime.py`, and the private-folder block in `src/api.py`. Unknown Starlette spool contracts fail closed. PMS `_save_legacy_multipart_image` remains for off/shadow compatibility. | Authoritative envelope/no-rollover/coordinator-no-bytes tests plus readable/unreadable diagnostic crop, no-LPD, filename/path safety, disk-failure continuation, runtime image-root, and private-route tests. | **PASS L2** for the V2 diagnostic exception; off/shadow still intentionally preserves additional legacy multipart snapshots and therefore does not satisfy a literal all-modes full-image no-file rule. Persistent-volume and retention operations remain L3-unverified. |
| R09 | Confirmation/session work is atomic, idempotent, source-time ordered, and serialized against exits without new tables. | PMS `app/services/entry_state_lock.py`; `entry_confirmation_service.py`; `parking_session_service.py`; `entry_exit_service.py`. | Duplicate/lost-ACK, rollback, stale-after-exit, superseded-entry, re-entry, and mocked app-lock tests. | **PASS L2**; real two-connection SQL Server contention is **UNVERIFIED L3**. |
| R10 | Exit delivery preserves the exact aware camera timestamp, closes the matching VA open journey, and survives transient delivery failures without repeating PMS mutations. | PMS `entry_exit_service.py` and `app/utils/core_backend_client.py`; VA `src/api.py`, `EntryCoordinator.record_exit`, reserved-callback exit reconciliation, and registry exit ordering. | Missing-source-time fail-closed, exact aware VA response echo despite a naive internal registry clock, duplicate replay, spool, non-age-out, unique-latest multi-callback race, journey closure/release, delayed-exit, and entry/publication race tests. | **PASS L2**; real process-kill/container recovery is **UNVERIFIED L3**. |
| R11 | Occupancy counts derive from existing open sessions; line events never apply deltas. | PMS `app/services/occupancy_service.py::reconcile_zone_counts_from_open_sessions`. | Session-derived journey, drift reconciliation, duplicate, rollback, and non-negative-count tests. | **PASS L2** |
| R12 | Motion detection is independent per camera and schedules YOLO only; stale/skipped/reconnect observations are UNKNOWN. Enforce mode must not sample below the requested inference target on a non-bypass camera. | VA `src/core/motion_scheduler.py`, `src/core/fair_latest_output.py`, `src/camera_manager.py`, and `src/core/engine/engine_single_process.py`. | Per-camera baseline, fairness, sentinel, transition latch, entry bypass, target-rate boundary/override validation, stale-frame, reconnect-epoch, and source-freshness tests. | **PARTIAL**: strong component evidence, but no production-model scheduler → detector → slot-observation integration test. |
| R13 | Slot state resists flicker: UNKNOWN is not absence, timed/count hysteresis is required, and LEAVING remains occupied. | VA `src/models/state_machine.py` and the engine UNKNOWN bridge. | UNKNOWN, single-miss, known-gap, timed hysteresis, shadow comparison, and LEAVING occupancy tests. | **PASS L2**; live-camera behavior is **UNVERIFIED L4**. |
| R14 | Active modes authenticate service boundaries, cap work before inference, fail closed on bad configuration, expose degraded/unhealthy state, and use one VA process. | PMS/VA configuration validators, request guards, semantic ACK checks, VA health/coordinator state, local-zone queue/timestamp health, FastAPI lifespan ownership, bounded web dependencies, and runtime checks. | Key/mode mismatch, malformed URL/numeric values, CAM-23 empty/mismatched physical filters, body/image limits, callback backlog, permanent failure, local queue saturation/retry/recovery, source-timestamp rejection, lifespan cleanup, resolved modern web-stack, process-count/local-zone single-process, and exit-bridge availability tests. | **PASS L2**; effective container values and one-replica topology are **UNVERIFIED L3**. |
| R15 | CPU-only model integration and bounded throughput are viable on the target 16-core Xeon. | Existing OSNet ReID, OpenVINO plate detector, PaddleOCR adapter, per-camera motion scheduler, and bounded queues. | Fake/model-independent adapters and capacity/fairness tests only. | **UNVERIFIED L3/L4**; full production-image suite and target-Xeon load test have not run. |
| R16 | An unplanned restart must not silently leave a confirmed PMS session without usable VA identity, or the risk must have an accepted reconciliation procedure. | Process-local attempts, crossings, callbacks, receipts, and embeddings; documented drain/cancel constraint. | Static review and rollout documentation. | **FAIL as an implied production reliability need** unless the owner explicitly accepts manual reconciliation or authorizes durable non-image evidence/replay. |
| R17 | Gate-open/barrier control must not be added. | No barrier-field parsing or decision path. | Repository search/diff review. | **PASS L1** |
| R18 | Every durable ReID gallery crop must have high-confidence post-ACK ANPR/CAM-23-or-CAM-03/ReID authorization plus crop-local parked OCR and unique rank-1 ReID proof; history must survive restart. | VA `src/entry/identity.py`; `src/vehicle_registry/vehicle_registry_identity.py`; `src/vehicle_registry/gallery_store.py`; `src/core/engine/engine_runtime.py`. | Entry authorization; mandatory strict-mode configuration; exact padded-crop isolation; fresh commit-time reranking; vacate/rebind/stronger-candidate race rejection; threshold boundaries; crop quality; idempotency; quarantine/corruption recovery; restart warm-start; and ReID artifact/preprocessing fingerprint invalidation tests. | **PASS L2**; live-camera precision and persistent-volume deployment are **UNVERIFIED L3/L4**. |

### Backward trace and orphan review

All material new production behavior traces to an explicit or implied safety/operability requirement. Exit spooling, authentication, health gating, request caps, and source-time ordering are not gold-plating; they protect entry/exit correctness under retries and partial failures.

The following known gaps remain:

- `EntryCancellationRequest.reason` is required by the VA contract but is currently discarded rather than logged or surfaced.
- PMS accepts decision evidence such as `ocr_text`, `ocr_evidence_ids`, `superseded_plates`, and `reid_margin`, but does not persist or fully log it. Not persisting it is consistent with the no-new-table constraint; the missing audit exposure is an observability gap.
- Legacy multipart snapshot persistence remains intentionally enabled for
  off/shadow compatibility. Authoritative Entry V2 does not use it, but shadow
  must not be represented as satisfying a literal all-modes no-entry-image-file
  rule.
- Entry V2 now retains only the LPD plate ROI used for OCR in the private
  `entry_plate_crops` diagnostics directory. This is an explicit owner-requested
  exception, not gallery evidence; disk retention/monitoring remains an
  operational requirement.
- Motion mode and per-camera motion metrics are present in performance logs but not exposed in `/api/health`; this limits rollout observability but does not invalidate the core decision logic.

## 3. Verification results

### 3.1 Passing mechanical evidence

| Check | Result |
|---|---|
| PMS full local suite | **347 passed**; includes authoritative CAM-23 empty/configured/mismatch and explicit vehicle-crop bypass coverage |
| VA named Entry V2 suite | **309 passed** on the existing compatible stack |
| VA Entry V2 plus affected API/integration subset | **328 passed** on the existing compatible stack |
| VA plate-crop diagnostic Entry V2/API regression slice | **336 passed** on the existing compatible stack |
| VA clean resolved web stack: FastAPI 0.139.2, Starlette 1.3.1, python-multipart 0.0.32, sse-starlette 3.4.6 | **102 passed** across affected API/integration tests |
| VA Starlette multipart compatibility probes | **PASS** on 0.40.0, 0.45.3, 0.46.0, 0.47.2, and 1.0.0; unknown spool contracts fail closed |
| VA broader changed-feature integration slice before this compatibility follow-up | **565 passed**; Ultralytics and Torch were import-stubbed because the local VA environment lacks them |
| Reported local gate policy | **All passing**; no `xfail` allowance is used for sequential evidence |
| Python compilation of PMS app/tests and VA changed critical modules/tests | **PASS** |
| `git diff --check` in PMS-AI, VA, and Gateway | **PASS** |
| Ruff on changed PMS production modules | **PASS** |
| Ruff on changed VA Entry V2, local-zone, motion scheduler, engine, and focused test modules | **PASS** |
| Gateway and database-schema scope inspection | **PASS** |

The broader 565-test integrated VA slice covered:

- `test_anpr_identity_and_gallery.py`
- `test_anpr_source_timestamp.py`
- `test_async_slot_ocr.py`
- `test_camera_freshness.py`
- `test_color_veto_and_cam23_seed.py`
- `test_db_slot_storage.py`
- `test_entry_v2_adversarial_p0.py`
- `test_entry_v2_analyzer.py`
- `test_entry_v2_callback.py`
- `test_entry_v2_contracts.py`
- `test_entry_v2_decisions.py`
- `test_entry_v2_domain.py`
- `test_entry_v2_exit_lifecycle.py`
- `test_entry_v2_gallery_authorization.py`
- `test_entry_v2_health.py`
- `test_entry_v2_late_crossing_dedup.py`
- `test_entry_v2_local_zone_bridge.py`
- `test_entry_v2_local_zone_coordinator_integration.py`
- `test_entry_v2_producer_pair_skew_settings.py`
- `test_entry_v2_registry_handoff.py`
- `test_health_status.py`
- `test_motion_scheduler.py`
- `test_occupancy_before_identity.py`
- `test_pms_to_va_entry_e2e.py`
- `test_reid_only_slots.py`
- `test_slot_observation_hysteresis.py`
- `test_strict_gallery_admission.py`

### 3.2 Verification not completed

- The full VA collection cannot run in the local host environment because
  Torch, Ultralytics, OpenVINO, PaddleOCR, and the rest of the production ML
  stack are absent. The 565-test gate used import-only stubs for Ultralytics and
  Torch; it did not execute or validate production model inference.
- Production images and model artifacts have not been exercised against real
  ANPR/CAM-23/CAM-03 frames, so OCR/ReID/detector accuracy remains unverified.
- PMS tests use SQLite and mocked SQL Server calls. They do not prove real `sp_getapplock` behavior under two independent SQL Server connections.
- No engine-level test currently drives motion scheduling through the real
  detector and into slot state transitions as one production-model path.
- No process-kill test covers the narrow PMS-commit/VA-publication or PMS-exit-commit/first-forward boundaries in real containers.
- Live camera geometry/direction, camera/PMS/VA NTP alignment, persistent-volume
  behavior, and 16-core Xeon latency/fairness/capacity remain unverified.

## 4. Validation against the originating problems

### 4.1 Vehicles that entered but had no session

The new path makes the session rule explicit and retryable: a session is
created only after physical crossing, ReID, and OCR confirmation, and
duplicate/lost acknowledgements are idempotent. The local-zone bridge now fails
closed on invalid source timestamps, reports queue saturation through health,
and retries retained evidence while the same visit remains in the zone. This
validates the design intent locally.

It does **not** yet prove that every real vehicle will receive a session. Bad CAM-23/CAM-03 geometry, a wrong asserted direction, weak facility ReID margins, OCR thresholds, mixed-camera identity, model latency, or capacity pressure can still cause abstention/backpressure. Recall and precision must be measured on labelled facility traffic before this problem is considered solved.

### 4.2 Wrong ANPR reading followed by a correct reading

The implementation does not trust timing or the last OCR read. It merges hypotheses only when strict ReID says they belong to the same vehicle, uses CAM-23 OCR first, permits the cached ANPR-crop OCR only under the approved fallback conditions, and requires exact independent evidence for correction. The wrong hypothesis is superseded only after the corrected match is approved. The local decision matrix passes.

Camera recognition-area and trigger-line configuration remains important input quality work; software evidence rules cannot make an invalid crop trustworthy.

### 4.3 Vehicle waiting at the barrier for minutes

Authoritative V2 has no evidence expiry and uses the physical crossing time as
entry time, so a long wait does not invalidate the attempt. After confirmation,
the selected open-journey policy retains bounded same-car evidence until PMS
delivers the authenticated ANPR exit source timestamp. Evidence captured on or
before that exit is compacted with the old stay; later evidence is released for
re-entry assignment. A missing exit eventually produces bounded `503`
backpressure instead of expiry, FIFO, or a guess. False/retreated evidence still
requires authenticated cancellation. These semantics pass locally.

Shadow still preserves the legacy workflow and therefore is not behaviorally identical to authoritative V2. Shadow labels must be based on V2 would-confirm/abstain output, not legacy session behavior.

### 4.4 Flickering slot state and CPU use

Motion gating is correctly separated from occupancy semantics: it only
schedules YOLO. UNKNOWN observations, timed/count hysteresis, reconnect epochs,
quiet sentinels, and LEAVING-as-occupied address flicker and stale-frame risks.
Entry cameras and configured entrance zones bypass the motion gate so entry
validation is not starved. Enforce mode now rejects a non-bypass camera whose
motion `analysis_fps` is below `processing.target_fps_per_camera`, preventing
motion sampling from silently lowering the requested detector opportunity rate.

Component behavior passes. Real reduction in CPU load, detector latency, fairness across all cameras, and facility flicker rates remain unvalidated on the target Xeon.

## 5. Definition of Done gate

| DoD item | Result |
|---|---|
| Latest approved requirements reconstructed and precedence recorded | **PASS** |
| Forward requirement → code → test traceability | **PASS** |
| Backward code/test → requirement trace and orphan review | **PASS** |
| Core authoritative behavior locally verified | **PASS with qualifications** |
| No Gateway or DB schema/table changes | **PASS** |
| Rollout/configuration contract documented | **PASS** |
| Entry request pixels are transient in every planned mode | **PARTIAL**: authoritative Entry V2 complies; off/shadow legacy multipart compatibility still writes snapshots, separate from the approved strict parked-gallery exception |
| Strict gallery admits only doubly verified crops and reloads them after restart | **PASS L2**; live precision/volume mount remain **UNVERIFIED** |
| Sequential CAM-23/CAM-03 completion/order policy selected and capacity-safe | **PASS L2** for the ANPR-exit-bounded open journey; production exit delivery, NTP, and load remain **UNVERIFIED** |
| `Park_Entry` and `B1_Entrence` proven physically one-way | **UNVERIFIED L4** |
| Full production model-container suite | **UNVERIFIED** |
| Real PMS↔VA container and SQL Server concurrency validation | **UNVERIFIED** |
| Real camera identity, line/direction, stable-ID, and NTP validation | **UNVERIFIED** |
| 24-hour labelled shadow precision/recall evidence | **UNVERIFIED** |
| Target-Xeon capacity, latency, fairness, and decoded-image load test | **UNVERIFIED** |
| Restart-loss resolved or explicitly accepted with reconciliation runbook | **FAIL**; owner acceptance or a durable reconciliation design is still pending |
| Production deployment and monitored acceptance | **NOT STARTED** |

The Definition of Done is therefore **not met**.

## 6. Required closure actions

### Before shadow

1. Treat off/shadow multipart snapshot persistence as an explicit legacy
   compatibility exception with bounded retention and access controls; do not
   claim literal all-modes no-file compliance. If the owner requires the
   no-entry-image-file rule during shadow too, shadow remains blocked until that
   legacy write is removed and regression-tested.
2. Verify the effective container environments, not only the Jenkins `.env` file:
   - PMS `PMS_API_URL=http://pms-video-analytics:8000` as a literal plain URL;
   - VA `PMS_API_URL=http://pms-ai:8080`;
   - identical non-empty `ENTRY_V2_SERVICE_KEY`;
   - identical `ENTRY_V2_MODE=shadow`;
   - explicit `VA_PROCESS_COUNT=1`, `VA_SINGLE_PROCESS=1`, and one deployed VA replica.
3. Resolve effective camera source peers and immutable entry/exit identities. Do not alias a shared proxy/NAT peer as entry-only without proof.
4. Calibrate the real CAM-23 Hikvision line/raw inward direction and prove the VA `Park_Entry` and CAM-03 `B1_Entrence` polygons are physically downstream and one-way. If reverse traffic can enter either polygon, replace the synthetic direction with directional geometry.
5. Run the full VA collection and focused cross-service gates inside the production dependency images with the actual detector, OCR, ReID artifacts, and representative facility frames.

### During shadow

6. Label at least 24 hours of attempts, crossings, would-confirm decisions, abstentions, corrections, duplicates, pre-/post-exit retained evidence, stale-after-exit cases, and missed legitimate entries. The precision gate is zero wrong confirmations in the labelled sample; recall must also be reported rather than hidden by abstentions.
7. Calibrate ReID absolute/row/column margins, merge/event-consistency/producer-pair scores, OCR confidence, and correction evidence from facility data.
8. Confirm stable source event IDs and exact offset-aware exit source timestamps; exercise authenticated cancellation for false or retreated vehicles.
9. Load-test concurrent ingress, local-zone queue saturation/recovery, delayed and missing ANPR exits, repeated CAM-23 → CAM-03 journeys, callback retry, image/body/decompression limits, per-camera fairness, production model latency, and motion savings on the target Xeon.
10. Add/run an integrated scheduler → production inference → UNKNOWN/state-machine regression.

### Before authoritative production

11. Run the two-connection SQL Server confirmation-versus-exit race test.
12. Exercise lost-ACK, VA outage, callback backlog, disk-full exit-spool behavior, and real process-kill recovery with containers.
13. Enforce NTP alignment and verify exact offset-aware camera timestamps in live and spooled exits, including exit delivery racing ahead of or behind re-entry evidence.
14. Resolve the RAM-only restart boundary, or obtain explicit owner acceptance of a tested manual reconciliation/runbook. Without this, authoritative operation across unplanned restarts remains no-go.
15. Change both services to authoritative in one controlled operation with rollback to shadow, then monitor semantic failures, abstentions, open journeys, provisional capacity, callback backlog, and `503` backpressure without rewriting failures to HTTP 200.

## 7. Sign-off

**V&V decision: PARTIAL / authoritative production NO-GO.** The selected logic
is suitable for continued controlled validation after the before-shadow items
are closed. It is not approved for authoritative production and must not be
represented as fully done.
