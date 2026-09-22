# Named-slot intrusion decisions

Named reserved bays publish occupancy independently of identification. Intrusion
checks follow identification outcomes, not elapsed time:

| Outcome | Alert |
| --- | --- |
| Identification pending | None; bay remains occupied |
| Identified authorized vehicle | None |
| Identified unauthorized vehicle | Critical `vehicle_intrusion` with plate |
| Identification exhausted with no remaining configured retry path | Critical `vehicle_intrusion` without plate, for security review |
| Vehicle leaves while pending | Clear the ownership check and exhaustion marker |

The immediate and deferred intrusion paths both use critical severity. The old
`reserved_slot_identity_timeout_s` YAML key is ignored, and the named-slot engine path no longer
emits `reserved_slot_unidentified`. Legacy category/history support remains.

## Completion and retries

OCR exhaustion is recorded after a final completed synchronous read or a validated
asynchronous result, not when a job is submitted. Rejected submissions do not spend
the attempt budget. An unavailable OCR plan is not a completed identification
failure; appearance retries may still run.

When `slot_reid_solo_enabled` is true and `slot_reid_retry_interval_s` is positive,
appearance retries continue after OCR exhaustion. The ownership check stays pending
for as long as those retries remain enabled. Slots configured in
`slot_no_plate_view` skip OCR; they remain pending with appearance retries enabled,
or reach the plate-less failure outcome when appearance retries are disabled.
A named bay on an explicitly identity-disabled camera/floor also reaches that
plate-less outcome: no identification work is configured for that camera.
No new retry limit or time deadline is introduced.

Pending checks and exhaustion markers belong to an occupancy generation. A departed
or replacement vehicle cannot inherit a completed read from the previous occupant.
Startup resumes pending ownership for occupied named slots whose identification is
armed. Unverified restored plates cannot authorize a completed identification failure.

## Local verification

The focused alert suites pass 48 tests plus four subtests. The expanded runtime,
async OCR, ReID-only and entry-contract run passes 135 tests plus four subtests.
The expanded run uses the existing alert-test import stubs to avoid loading the
unavailable YOLO stack; it verifies control flow, not model accuracy or live camera
behavior. The timeout and immediate-severity regressions both fail against the
unchanged baseline. Live camera/database and deployed acceptance remain pending.
