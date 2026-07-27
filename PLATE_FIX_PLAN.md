# Plate-Number Remediation Plan — Video Analytics (System 2)

> Companion to `Plate_Number_Wiring_Review.pdf`. This repo **produces the plate
> from OCR** and pushes it into the shared DB and to PMS-AI. Do this repo
> **after PMS-AI Steps 1-3** are merged, because it must adopt PMS-AI's
> normalization spec and consume the new bind-slot behavior.

## Cross-project sequencing (read once)

1. PMS-AI — canonical normalization + bind returns `vehicle_id` + bind opens a
   session. *(prerequisite for this repo)*
2. **Video Analytics (this file)** — adopt the identical normalization, fix
   silent bind/unbind failures, stop double-writing the plate.
3. API Gateway — timezone, slot-identity column, `vehicle_id` joins, client regen.

Blocking decisions (owned with PMS-AI):

- **D1 — Plate normalization spec.** Copy PMS-AI's `normalize_plate()` and its
  test-vector table **verbatim**. Both repos must pass the same table.
- **D3 — Slot plate columns.** Keep **both** `parking_slots.current_plate` and
  `slot_status.plate_number` populated. `slot_status.plate_number` is wired into
  other tables and read by the frontend, so it stays — VA writes the plate to
  both columns and keeps them in sync. See Step 3.

---

## Step 1 — Adopt the canonical normalization (fixes I-1, this repo's half)

**Problem:** OCR output is written/POSTed raw, so it may not match PMS-AI's
ANPR plate string.

**Do:**

- [ ] Add `src/ocr/plate_normalize.py` containing PMS-AI's `normalize_plate()`
      **verbatim**, plus a copy of the shared test-vector table under
      `tests/`.
- [ ] Normalize the OCR result at the single point it becomes a "confirmed
      plate", before it is stored or sent:
  - [ ] `src/core/engine/engine_runtime.py` — where OCR resolves a plate
        (~line 2094) and where `db_slot.current_plate` is set (line 2369).
  - [ ] `src/services/slot_status_service.py` — `log_vehicle_event` and
        `update_current_slot_plate` (normalize `plate` on entry, lines 45,
        123-177).
- [ ] Ensure the plate sent to `pms_api_client.bind_slot_session` /
      `unbind_slot_session` is the normalized value.

**Acceptance:** for a set of test images, the plate VA writes to
`current_plate` equals the plate PMS-AI stores for the same car (identical
strings), verified against the shared vector table.

---

## Step 2 — Stop swallowing bind/unbind failures (fixes I-2)

**Problem:** `pms_api_client._post` catches only `URLError` and `print()`s it
(`pms_api_client.py:22-26`). A 404/409 is an `HTTPError` that is not even
logged, so a failed slot↔plate link is invisible.

**Do:**

- [ ] Replace `print` with the project logger
      (`from src.utils.logger import get_logger`).
- [ ] Check the HTTP status. On 2xx: done. On 4xx/5xx: log at WARNING with the
      response body.
- [ ] Add a **retry/queue** for transient failures and for 404 specifically
      (the ANPR session may open moments later, and PMS-AI Step 3 may now open
      one on bind):
  - [ ] Persist failed binds to a small local outbox (table or JSONL) with
        `plate, slot_id, payload, attempts, next_retry_at`.
  - [ ] A background task retries the outbox with backoff; drop after N
        attempts and log an ERROR.
- [ ] Consume the new response: read `vehicle_id` from the bind response and
      store it alongside the slot binding (so VA can later report the numeric
      id, not just the plate).

**Acceptance:** killing PMS-AI mid-run produces WARNING logs and outbox rows,
not silent `print`s; restarting PMS-AI drains the outbox and the sessions bind.

---

## Step 3 — Keep both slot plate columns in sync (fixes I-4, VA half)

**Problem:** VA writes the plate to **both** `parking_slots.current_plate`
(identity) and `slot_status.plate_number` (occupancy), and today they can
diverge (one may hold `''` while the other holds a plate). We are **keeping both
columns** — `slot_status.plate_number` is wired into other tables and read by
the frontend — so the fix is not removal but keeping them consistent and
documented.

**Do (per D3):**

- [ ] Continue writing the **normalized** plate to **both**
      `parking_slots.current_plate` and `slot_status.plate_number` on every
      identity update in `slot_status_service.log_vehicle_event` /
      `update_current_slot_plate`.
- [ ] Never let them diverge: any path that sets or clears one must set/clear the
      other in the **same commit**. Write the same normalized value to both —
      no `''` in one while the other holds a plate.
- [ ] Keep `reset_all_slot_plates` / `clear_slot_plate_binding` clearing both so
      no stale identity lingers (already the case — preserve it).
- [ ] Document that both columns are authoritative mirrors so downstream
      consumers (the Gateway, dependent tables, and the frontend) can read
      either safely.

**Acceptance:** for every occupied slot,
`parking_slots.current_plate == slot_status.plate_number` (both normalized) —
never one empty while the other is set.

---

## Verification (whole repo)

- [ ] `pytest` green, including the copied normalization vector test.
- [ ] End-to-end: a car parks → OCR reads plate → `current_plate` set →
      bind succeeds with `vehicle_id` returned → the same plate string is
      visible via the Gateway's occupancy slot detail.
- [ ] Failure path: PMS-AI down → outbox fills, WARNING logged; PMS-AI up →
      outbox drains automatically.
