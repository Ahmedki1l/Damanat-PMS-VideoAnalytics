# Design Review — decoupling OCR (and ReID) from the real-time pipeline

**Date:** 2026-07-20 · **Status:** review only, nothing implemented
**Scope:** the 5 questions raised, answered against the current codebase.

## Executive summary

**The proposed architecture is ~70% already implemented** (`src/core/engine/async_slot_ocr.py`,
186 lines). OCR already runs on a background thread with a bounded, coalescing
queue and generation-token validation. The residual main-thread work is
*milliseconds*, not seconds.

**And the measured data says OCR is not the bottleneck.** Production logs show
**9 OCR reads in 15 minutes** (~0.6/min) while the pipeline sits at 0.3 fps/camera
because the OpenVINO *detection* pool delivers 7.3 inf/s. `[PERF] consumer:
busy=38–70% … (not consumer-bound)` says the frame-consumer thread — which is
where all the remaining OCR fold-back work lives — is idle over half the time.

**Recommendation: do not build this now.** Completing the split would add a
process boundary, IPC, and a second failure domain to remove work that is already
off the hot path and costs ~0.6 calls per minute. Spend the effort on the
inference ceiling, and land the plate detector — which needs **no** architectural
change, because it slots into a function that already runs on the worker thread.

---

## Q1 — Is full decoupling technically feasible?

**Yes, and most of it already ships.** Here is what actually runs today:

```
MAIN THREAD (per processed frame)                    WORKER THREAD ("slot-ocr")
─────────────────────────────────                    ──────────────────────────
_process_detections_and_events
  ├─ ROI → assign → occupancy → EMIT   ← never blocked by OCR
  └─ _update_slot_identity
       └─ _try_ocr_identify
            ├─ plan_slot_ocr()                       
            │    ├─ reid_matcher.extract_feature()   ~15 ms
            │    └─ reid_rank() → top-5 shortlist
            ├─ worker.submit(SlotOcrJob)  ──────────►  _run()
            └─ return immediately                        └─ read_slot_plate()
                                                              PaddleOCR  2–8 s
  _drain_slot_ocr_results()  ◄──────────────────────  _results.append()
       └─ _apply_async_ocr_result()
            ├─ 5 re-validation guards
            ├─ confirm_slot_ocr()   colour/type classifiers + decision log
            └─ _bind_ocr_identified_plate() → DB commit
```

The heavy part — the 2–8 s PaddleOCR read — is **already off the camera loop**.
`slot_ocr_async` defaults `True` (`config.py:561`) and is on in production.

What remains on the main thread, and why:

| stage | cost | why it's there |
|---|---|---|
| `plan_slot_ocr` | ~15 ms | ReID embed + rank; shares OpenVINO with the tracking loop |
| `confirm_slot_ocr` | ~10–30 ms | colour/type classifiers, same constraint |
| `_bind_ocr_identified_plate` | ~5 ms | DB commit; ordering matters |

So the delta between "today" and "your proposal" is **~30–50 ms per completed OCR
job, at 0.6 jobs/minute**. That is ~0.03% of one core.

### The one stale comment

`async_slot_ocr.py:12-15` says the ReID matcher is *"not safe to call from a second
thread"*. That is **no longer accurate** for the OpenVINO backend:

```python
# reid_openvino_backend.py:133-134
self._request = self._compiled.create_infer_request()
self._lock = threading.Lock()
# :219
with self._lock:
    output = self._request.infer({"images": nchw})
```

It is thread-**safe**. But it is a *single* infer request under a mutex, so it is
also thread-**serialized** — calling it from a worker would not run in parallel
with the main loop, it would block it. Thread-safe ≠ concurrent. This distinction
governs Q3.

---

## Q2 — Advantages and disadvantages

Assessed as *"what would completing the split add over today"*, since the async
worker already exists.

| dimension | today (thread worker) | full split (separate process) |
|---|---|---|
| **Latency** | main loop never waits; fold-back ≤1 frame | identical; IPC adds 1–5 ms |
| **Throughput** | OCR costs ~0.03% of a core on main | ~0 on main. **Gain is noise.** |
| **Memory** | crop refs in a 64-slot dict, coalesced | +1 process: model weights, OpenVINO runtime, Paddle. **~1–2 GB.** |
| **Synchronization** | one `Condition`, ~15 lines | queue serialization + backpressure + heartbeats |
| **Race conditions** | **solved** — 5 guards + generation token | same guards still needed, now across a process boundary |
| **DB consistency** | single writer, main thread | **regression risk**: a second writer to `current_plate` re-opens exactly the divergence fixed in `96c96f5` today |
| **Scalability** | 1 thread; PaddleOCR is one locked engine anyway | N processes *would* scale OCR — but demand is 0.6/min |
| **Failure recovery** | worker exception caught per-job; wedged read bounded by `max_pending=64` | needs supervision, restart, orphan-job reaping |

The decisive rows are **memory** and **DB consistency**. A second process pays
1–2 GB and a second `current_plate` writer to eliminate ~30 ms/minute of work.

### What today already gets right

`AsyncSlotOcr.submit()` — bounded (`max_pending=64`), coalescing (a newer crop
replaces a queued one), and one-in-flight-per-slot. A wedged read cannot grow
memory without bound. The worker catches every exception per job, so one bad crop
cannot kill the thread.

`_apply_async_ocr_result()` re-validates against **current** state before binding:

```python
if not self._ocr_armed.get(slot_id):                     return  # vacated
if self._ocr_generation.get(slot_id) != job.token:       return  # re-armed, new car
if self.vehicle_registry.get_slot_plate(slot_id):        return  # named meanwhile
if state_machine.state != SlotState.OCCUPIED:            return
if state_machine.plate_number:                           return
```

That is the correct pattern for late-arriving async results, and it is the answer
to "how do we avoid conflicting updates" — **already implemented**.

---

## Q3 — Integration with ReID

**Should ReID stay in the main pipeline? For now, yes** — but not for the reason
the code comments give.

The blocker is not thread-safety (it is safe). It is that the backend holds **one
infer request behind one mutex**. Moving ReID to a worker would serialize against
the main loop's own ReID calls, so you'd pay thread-handoff cost for zero
concurrency. Making it genuinely parallel means multiple infer requests — i.e. an
`AsyncInferQueue` like the detector's, which is a real piece of work and *adds a
second pool competing for the same 36 CPUs already saturated by detection*.

**Same queue or separate?** Same. The existing worker already multiplexes two
producers via `SlotOcrJob.kind` (`"slot"` and `"track"`), sharing one queue
deliberately: *"PaddleOCR is a single engine with an internal lock, so two queues
would only let the two paths queue behind each other twice."* The identical
argument applies to ReID's single infer request. Separate queues buy nothing while
the downstream resource is singular.

**Who consumes whom?** Neither — and this is important. They are not a pipeline,
they are **two independent witnesses that must agree**:

```
                 crop
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  ReID shortlist      OCR text read
   (top-5 plates)     ('RRY337ZVH')
        └─────────┬─────────┘
                  ▼
           confirm_plate()
      binds only if EXACTLY ONE
        shortlisted car matches
```

Making OCR emit events ReID consumes (or vice versa) would collapse two
independent signals into a chain, and the independence *is* the safety property —
it is what keeps `current_plate` "correct or NULL".

**OCR finishing after ReID already predicted?** Already handled, and the
precedence is deliberate: a ReID-solo bind stays **PROVISIONAL** (never locked,
never taught to the gallery) precisely so a later OCR read can overrule it
(`engine_runtime.py:1682-1684`). OCR wins because it read the characters.

---

## Q4 — Can both share one snapshot?

**They already do.** In `plan_slot_ocr` (`vehicle_registry_identity.py:4560-4603`),
the *same* `crop_bgr` is used for the ReID embedding and then carried into the OCR
job:

```python
qvec = self.reid_matcher.extract_feature(crop_bgr)      # ReID
kept, rejected = self.reid_rank(qvec, ...)              # shortlist
return SlotOcrPlan(candidates=..., ...)                 # → job.crop = same array
```

and the same array is reused at fold-back for `confirm_slot_ocr(job.plan, job.crop,
…)` and for the gallery write in `_bind_ocr_identified_plate`. One crop, one
allocation, three consumers. The comment at 4560 explains the ordering: *"RANK
FIRST, then read… Scoring costs one embedding (~15 ms) against OCR's ~80–470 ms, so
it is nearly free."*

Nothing to build here.

---

## Q5 — Is there a better architecture?

Yes — **the one you already have, plus two targeted changes.** Ranked by measured
value:

### 1. Fix the inference ceiling (not an OCR problem)

7.3 inf/s ÷ 26 cameras = 0.28 fps/camera. This is the single cause of: slow slot
flips, fragmented tracks, no gallery accumulation, and therefore B7's
`plates_inside()` exclusion that discards a *correct* plate read. Every identity
symptom traces back here. No OCR architecture change improves it.

### 2. Plate detector inside `read_slot_plate` — **zero architectural change**

```
WORKER THREAD (already exists)
  read_slot_plate(crop, allow_retry)
    ├─ NEW: lpd.detect(crop)          ~10 ms
    ├─ best box → pad → plate crop
    └─ ocr.read(plate_crop)           ~440 ms   (was ~2230 ms)
```

Measured (36 live crops, `reports/plate_detector_evaluation.md`): OCR median
1447 → 439 ms, worst case 13,178 → 1,497 ms. It lands entirely inside a function
that is *already* on the worker thread. This is the change your proposal was
reaching for, and it needs none of the proposal's machinery.

### 3. Only if 1 and 2 land and OCR is still the constraint

Then revisit — with the deciding metric being `[PERF] consumer busy%`. Today it is
38–70% and explicitly "not consumer-bound". **If it approaches 100% and slot flips
lag, the fold-back work has become the bottleneck and the split is justified.**
That is a falsifiable trigger, and it is not met.

### Comparison

| | your proposal | recommended |
|---|---|---|
| main loop blocked by OCR | no | no *(already true)* |
| new process / IPC | yes | no |
| extra memory | ~1–2 GB | ~5 MB (LPD weights) |
| second DB writer | yes (risk) | no |
| OCR latency | unchanged | **3.3× better (measured)** |
| addresses 0.3 fps | no | no — item 1 does |
| effort | ~1–2 weeks | ~1 day |

---

## Recommendation

**Do not proceed with the process-level split.** It solves a problem the
measurements do not show: OCR is already asynchronous, runs 0.6×/minute, and the
consumer thread is half idle. The cost — a second process, IPC, a second writer to
`current_plate` days after fixing exactly that class of bug — is not repaid.

**Proceed instead with:**

1. the inference/throughput work already in flight (`VA_INFER_NIREQ=4` measurement pending);
2. the plate detector in `read_slot_plate`, behind `slot_lpd_enabled` with full-crop fallback;
3. a one-line correction to the stale thread-safety comment in `async_slot_ocr.py:12-15`.

**Revisit this design if** `[PERF] consumer busy%` climbs toward 100% while slot
flips lag — that is the measurement that would make the split correct.
