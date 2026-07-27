# Why B7 / B12 / B13 have no plate — root cause across PMS-AI and Video Analytics

**Date:** 2026-07-20 · **Scope:** `Damanat-PMS-AI` (entry pipeline) → `Damanat-PMS-VideoAnalytics` (slot identity)

## Executive summary

The slot cameras read these plates **correctly**. The reads are discarded because
PMS-AI never opened a parking session for the cars, and VA can only *confirm* a
plate against cars it believes are inside — it cannot *recognise* one.

The entry sessions are missing because of a **feedback loop the burst design did
not anticipate**:

> The barrier only opens once ANPR reads the plate correctly.
> The ANPR emits its wrong reads first and the correct one last.
> So a hard-to-read car waits at the barrier for tens of seconds.
> But the entry burst is capped at **20 s measured from the FIRST (wrong) read**,
> and is **dropped entirely** if the ramp crossing hasn't confirmed it by then.
> The crossing cannot happen until the barrier opens — which happens *after* the
> cap has already killed the burst.

A car that reads cleanly on the first try is fine. A car that needs retries is
structurally guaranteed to lose its entry record.

---

## 1. Evidence

### The cars are parked but have no open session

| car | slot | last recorded entry | status | registered? |
|---|---|---|---|---|
| `RGR-6466` | B12 CCO | **2026-07-16 09:31** | closed | vehicle_id 297, employee |
| `BHD-9990` | B13 COO | **2026-07-16 08:28** | closed | vehicle_id 434, employee |
| `ZVH-337` | B7 CHRO | 2026-07-19 09:09 | closed | vehicle_id 294, employee |

`RGR-6466` entered every working day (07-13, 14, 15, 16) then stopped — while the
car sits in B12 right now.

### The gate itself is not dead

Entries per day: 07-13 → 31, 07-14 → 33, 07-15 → 31, **07-16 → 29**, 07-17 → 1,
07-18 → 3 *(weekend)*, 07-19 → 21, **07-20 → 28**.

So ANPR works. **13 regular cars stopped being recorded after 16 July**:

```
ABR-8000  BHD-9990  EEB-80    ERS-7949  HVA-77   LLJ-9005  LXA-3111
RDJ-9640  RGR-6466  RLR-2714  RTB-2016  RZG-136  ZRS-6511
```

### Misreads are minting phantom vehicles

New "cars" since 19 July include `36663XN` (vehicle_id 1526) and `77842SJ`
(vehicle_id 1525) — owner `Unknown`, `is_employee=false`, **1 record each**, both
`status=open`. `77842SJ` is a misread of `DJS-7842`. These phantoms then pollute
VA's candidate set and one of them (`36663XN`) is currently bound to slot **B2**.

### VA's candidate universe is nearly empty

At boot VA knows **12 plates total** (5 from `parking_sessions WHERE status='open'`,
7 from the on-disk gallery), two of which are phantoms. For a garage with 15+
occupied slots. `RGR-6466`, `ZVH-337`, `BHD-9990` appear **zero times** in the log.

### Meanwhile the OCR is working perfectly

After the plate-region detector landed, the slot camera reads B12 as:

```
'1116466RGRJ3'  '11116466RGRJ3'  '12116466RGR'  '11116466RGRJJJ'  ...  (8+ times)
```

and `read_matches_plate('1116466RGRJ3', 'RGR-6466')` → **True**.

Rejected with: `confirms none of ReID's top-5 [...]`, and **zero consensus votes**,
because `RGR-6466` is in neither candidate source.

---

## 2. Root cause — the mechanism

`app/services/entry_exit_service.py::flush_due_entry_bursts`

```python
age  = (now - buf["first_event_time"]).total_seconds()   # from the FIRST read
idle = (now - buf["last_read_at"]).total_seconds()
eligible = buf["force_flush"] or idle > window or age > max_age   # 2.5s / 20s

if require_confirm and not buf["confirmed"]:
    if age > max_age:
        to_drop.append(_entry_bursts.pop(bid))   # ← the entire burst dies
    continue
```

While the driver waits at the barrier the ANPR keeps firing, so `last_read_at`
keeps refreshing and `idle` never exceeds 2.5 s — the burst stays open. But `age`
climbs past 20 s and the burst is **dropped, including the eventually-correct read**.

`app/config.py:343-348` states the assumption explicitly:

> *"Real-world read→crossing travel time is ~8s at this site … 20s covers the
> travel gap plus a slow driver."*

That models **read → drive → cross**. The real sequence for a hard-to-read plate is
**wrong read → retries (N seconds) → correct read → barrier opens → drive → cross.**
The cap is measured from the wrong read at the start of a process whose length is
determined by how long the ANPR takes to succeed.

### The two failure modes

1. **Crossing arrives after 20 s** → burst dropped → no `EntryExitLog`, no open
   session → VA never hears of the car → slot stays NULL. *(B7 / B12 / B13)*
2. **Crossing arrives inside 20 s but the correct read comes later** → burst flushes
   with an early **wrong** read as "winner" → a phantom vehicle is created and bound.
   *(`36663XN`, `77842SJ`)*

Note the winner rule is `max(pic_num, event_time)` — "last read wins" is correct
*only if the burst lives long enough to contain the last read*.

### Why it started around 16 July

The mechanism has always been there (the cap was raised 8 s → 20 s in `2ddd368`
for this same class of problem). Nothing in the code changed after 07-12. What
changed is that these particular cars' gate dwell crossed the 20 s threshold —
plate wear, lighting, camera focus, or ANPR model drift. **The design has a cliff,
and in mid-July this group of cars walked off it.** They will not come back on
their own.

---

## 3. Fix plan — PMS-AI

### P1 — Stop timing the cap from the first read *(config only, deploy today)*

```bash
ANPR_BURST_MAX_SECONDS=120     # was 20
```

`Settings` is pydantic `BaseSettings` with `env_file=".env"` (`app/config.py:426`),
so this needs no code change. Per-car boundaries are still enforced by the Hikvision
`picNum` reset (`_buffer_entry_read`), and the 30 s dedup still prevents doubles.

This alone should restore entries for all 13 cars. It is a mitigation, not the fix.

### P2 — Make the confirm deadline idle-based, not age-based

Continued reads are *evidence the car is still at the gate*, not staleness. The
drop decision should hang off `last_read_at`:

```python
ANPR_BURST_CONFIRM_GRACE_SECONDS = 30.0   # since the LAST read
...
if require_confirm and not buf["confirmed"]:
    if idle > confirm_grace or age > absolute_ceiling:   # ceiling ~180s, safety only
        to_drop.append(...)
```

This models reality: a burst dies 30 s after the ANPR *stops trying*, not 20 s
after it started.

### P3 — Never let a malformed read become the winner, or a vehicle

Two guards at flush time:

* **Format gate** — Saudi plates are `^[A-Z]{2,4}-?[0-9]{2,4}$`. `36663XN` and
  `77842SJ` fail it. Prefer the newest *well-formed* read; only fall back to a
  malformed one if none exists.
* **Registry gate** — do not auto-create a `vehicle` row for a plate that is both
  unknown and malformed. Log it as an unidentified entry instead. This stops the
  phantom-vehicle pollution at source (1525, 1526 are today's).

### P4 — Reconcile the digit run against known vehicles

The digits survive OCR far better than the letters (this is already the basis of
VA's `read_matches_plate`). At flush, if the winning read's digit run uniquely
matches exactly one registered vehicle, prefer that vehicle. `77842SJ` → digits
`77842` → `DJS-7842`. Ambiguous → abstain.

---

## 4. Fix plan — Video Analytics

### P5 — Let a strong slot OCR read open an identity *(the real decoupling)*

Today `confirm_plate` can only pick from `plates_inside()`. That makes VA
**completely dependent** on PMS-AI's entry pipeline — a single point of failure
this incident demonstrates.

Proposal, tightly gated so it cannot invent cars:

```
bind on OCR alone IFF
   ≥ N agreeing reads (reuse slot_ocr_consensus_min_agreement = 3)
   AND the read has a ≥4-digit run (_STRONG_DIGITS, ~1-in-10,000)
   AND it resolves to EXACTLY ONE row in the `vehicles` table
   AND that plate is not locked to another slot
→ open/attach a session, bind the slot
```

`RGR-6466` satisfies every clause right now: 8+ agreeing reads, 4-digit run, one
registered vehicle (id 297), not locked elsewhere. It would have bound hours ago.

This preserves the "correct or NULL" contract — the vehicles table is the
authority, so VA still cannot fabricate an identity, it can only *recognise a
registered car it can plainly see*.

### P6 — Teach the gallery on an OCR bind

Once OCR names a parked car, capture its parked-pose crop as a gallery reference.
That is the missing input that leaves ReID unable to shortlist these cars, and it
is what breaks the documented OCR/ReID deadlock permanently.

### P7 — VA confirms the physical entry (the "harmony" you asked about)

VA already watches CAM-23 with a `Park_Entry` zone, but the log shows
`[PARK_ENTRY][DIAG] CAM-23 frame: registry=True detections=0` — it sees the camera
and detects nothing, because the pipeline is at 0.2–0.3 fps.

Once throughput is fixed, VA can act as a **second, independent confirmer** of the
crossing. PMS-AI currently exposes only `/bind-slot` and `/unbind-slot` to VA
(`app/routers/parking_sessions_internal.py`); this needs a new
`POST /internal/entry/confirm-crossing`. Then a burst can be confirmed by CAM-23
line-detection **or** CAM-03 occupancy **or** VA's own vehicle detection — removing
the single dependency that drops entries today.

**Sequencing note:** P7 depends on VA throughput. At 0.3 fps VA cannot reliably see
a car cross a line. Fix inference first.

---

## 5. Recommended order

| # | change | repo | effort | effect |
|---|---|---|---|---|
| 1 | `ANPR_BURST_MAX_SECONDS=120` | PMS-AI | env var | restores the 13 missing cars |
| 2 | format + registry gate at flush | PMS-AI | ~1 day | stops phantom vehicles |
| 3 | idle-based confirm deadline | PMS-AI | ~1 day | removes the cliff properly |
| 4 | digit-run reconciliation | PMS-AI | ~1 day | recovers misreads |
| 5 | OCR-originated identity (gated) | VA | ~2 days | VA stops depending on the gate |
| 6 | gallery learning on OCR bind | VA | ~1 day | breaks the ReID deadlock |
| 7 | VA as crossing confirmer | VA + PMS-AI | ~3 days | removes the single confirmer |

**Do #1 today.** It is one environment variable and it addresses the immediate
outage. Everything else is durability.

## 6. Cleanup needed regardless

* slot **B2** is bound to `36663XN` — a phantom. Clear it.
* vehicle rows **1525** (`77842SJ`) and **1526** (`36663XN`) are junk; merge or delete.
* the 13 cars have open *physical* occupancy but closed sessions — their sessions
  need reopening or the occupancy report stays wrong.
