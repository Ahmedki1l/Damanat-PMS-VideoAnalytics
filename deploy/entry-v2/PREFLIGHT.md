# Entry V2 — Kubernetes rollout

`README.md` in this directory documents the off → shadow → authoritative ladder
and its reasoning still holds, but its commands are Docker Compose. **Production
is Kubernetes.** These are the mechanics that replace them.

> **A repo-only change ships nothing.** The deployed ConfigMap is not this
> repository. Every setting below has to be set on the cluster.

---

## The rollout is a loop, not a line

```text
Implement
   ↓
Shadow mode  ────────────────────┐
   ↓                             │
Real-traffic validation (2–3 d)  │
   ↓                             │
Review: logs, Re-ID scores,      │
        margins, mismatches      │
   ↓                             │
Fix + tune  ─────────────────────┘   (as many times as it takes)
   ↓
Approval
   ↓
Authoritative
```

There is no partial promotion and no "authoritative for the good cases": the
mode is one flag on two services.

---

## Step 0 — Preflight

Three facts must hold on the cluster first, because each one invalidates the run
in a way you cannot detect from the results.

```bash
kubectl get deploy pms-video-analytics -o jsonpath='{.spec.replicas}{"\n"}'

kubectl get deploy pms-video-analytics \
  -o jsonpath='{range .spec.template.spec.volumes[*]}{.name}{" "}{.persistentVolumeClaim.claimName}{"\n"}{end}'
```

| Check | Why it invalidates the run |
|---|---|
| **`replicas` must be 1** | Identities and observations live in RAM. Two replicas behind one Service means an ANPR read lands on pod A and its CAM-23 observation on pod B — neither can ever match them, manufacturing the exact dropped-entry class this work removes. `configuration_errors()` already rejects `va_process_count != 1` at AUTHORITATIVE; the deployment has to agree. |
| **`/app/vehicle_images` must be PVC-backed** | The second command prints the claim name. An empty result means `emptyDir` or no mount, and the decision log dies with the pod — a crash on hour 30 of 72 costs the whole window. |
| **The log directory must be writable** | `DecisionLog` degrades silently if it is not, and the whole review depends on it. Step 1 verifies this explicitly. |

There is no alert-suppression check here, and deliberately so: **this pipeline
raises no alerts.** Every outcome is a record in the decision log, which is why
there is nothing to suppress and why the log directory check replaced it.

---

## Step 1 — Deploy with the mode still `off`

`ENV ENTRY_V2_MODE=off` is baked into the Dockerfile, so the deploy alone
changes no behaviour. This step exists to prove the image boots and the log
directory is present and writable **before** shadow is switched on — if the
directory only appeared once the pipeline was live, the check could not be made
while it was still useful.

```bash
kubectl rollout status deploy/pms-video-analytics
kubectl exec deploy/pms-video-analytics -- \
  printenv ENTRY_V2_MODE ENTRY_V2_DECISION_LOG_DIR
kubectl exec deploy/pms-video-analytics -- \
  ls -la /app/vehicle_images/entry_v2_shadow
kubectl exec deploy/pms-video-analytics -- \
  curl -fsS http://127.0.0.1:8000/api/health
kubectl logs deploy/pms-video-analytics --tail=200 | grep -i "entryv2\|entry_v2"
```

You should now see `logger.info` lines at all. If the logs are still only
`print()` output, `configure_logging()` did not run and nothing later is
reviewable.

---

## Step 2 — Shadow

Update the ConfigMap for **both** services. VA and PMS-AI cross-check the mode
on every request via the `X-Entry-V2-Mode` header, and a mismatch is a 503.

```dotenv
ENTRY_V2_MODE=shadow
ENTRY_V2_DECISION_LOG_DIR=/app/vehicle_images/entry_v2_shadow
```

```bash
kubectl edit configmap <va-configmap>       # and the pms-ai one
kubectl rollout restart deploy/pms-video-analytics
kubectl rollout restart deploy/pms-ai
kubectl rollout status deploy/pms-video-analytics
```

**A ConfigMap edit does not restart pods.** The rollout restart is required, not
optional.

Then, within ten minutes of real traffic, confirm records are landing. **Pod
health is not evidence:**

```bash
kubectl exec deploy/pms-video-analytics -- \
  sh -c 'wc -l /app/vehicle_images/entry_v2_shadow/*.jsonl'
```

If that is empty after real entries, **stop.** Either logging did not take
effect, the directory is not writable, or `build_entry_coordinator` degraded to
`DisabledEvidenceProcessor` on a configuration error — which is safe, but
records nothing, and every later conclusion would be drawn from an empty file.

Copy the file off at the end of each day, so a pod restart cannot cost the run:

```bash
POD=$(kubectl get pod -l app=pms-video-analytics -o name | head -1)
kubectl cp "${POD#pod/}:/app/vehicle_images/entry_v2_shadow" ./shadow-$(date +%F)
```

### Optional, and separately gated

Neither of these turns itself on, and both are inert until measured:

```dotenv
# Only after scripts/setup/probe_hik_camera_events.py confirms REAL indexCodes.
# An unknown code answers 200/code=0/empty and would sweep forever finding
# nothing, exactly as HIK_EXIT_RESOURCE_IDS=453 did.
HIK_RAMP_RESOURCE_IDS=

# Only after scripts/setup/probe_hik_images.py measures where the composited
# panel actually sits. A guessed rectangle rejects real plates.
ENTRY_V2_OVERLAY_EXCLUDE_REGIONS=
```

---

## Step 3 — The review

```bash
python tools/summarize_entry_shadow.py ./shadow-2026-08-29
```

Exit code 0 means the **hard stops** are clean. That is necessary and **not
sufficient** — the tool says so itself. The sections it marks "needs eyes" have
to be read by a person:

- every `ambiguous` record: was it genuinely two plausible cars?
- every `unreadable` record: should it have been confirmable?
- every FIFO/Re-ID disagreement: was Re-ID right each time?
- every colour veto: did any of them remove the *correct* identity?
- the Re-ID score and margin distributions: do the shipped thresholds sit on a
  plateau, or on a cliff edge?

## Step 3a — The loop

If anything fails, **fix and tune while still in shadow**, redeploy, and run the
window again. Repeat Step 2 → 3 as many times as it takes.

---

## Step 4 — Authoritative

Only on explicit approval of a clean review.

```dotenv
ENTRY_V2_MODE=authoritative
```

Same ConfigMap edit on both services, same rollout restart. **Leave
`ENTRY_V2_DECISION_LOG_DIR` set** — the log keeps accumulating and becomes the
calibration corpus. Recording does not stop when the mode changes; that is the
point. Keep archiving it off the pod on a schedule.

---

## Rollback

```bash
kubectl rollout undo deploy/pms-video-analytics    # and pms-ai
```

Or set both services back to `shadow` (or `off`) and restart. Rolling back the
mode is instant and needs no image change — which is why mode lives in config
rather than in the image.

**Copy the shadow log off the pod *before* rolling back.** The records from a
failed run are the evidence for why it failed, and a rollback that reschedules
the pod can take them with it. Never delete the gallery PVC.
