# Durable entry recovery

Local implementation; activation and target-stack acceptance are pending.

`ENTRY_V2_DURABILITY_ENABLED` defaults to false. When enabled, `ENTRY_V2_DURABILITY_STATE_DIR` is required. It must be a persistent, writable directory owned by one VA process. The journal is `entry_v2_durability.sqlite3` with SQLite WAL and synchronous FULL; a lifetime POSIX file lock rejects another process using the same journal. This is a single-host journal, not a distributed queue. Camera, inference and HTTP concurrency inside that owner are not limited to one request.

The journal binds its entry mode. Do not reuse a shadow journal as an authoritative journal: startup rejects the mismatch to prevent historical shadow decisions creating sessions. Preserve the database and WAL together during backup/recovery; never delete the journal to clear a backlog.

HTTP attempts and local-zone crossings are stored with their exact request, image evidence and fingerprint before durable admission succeeds. Storage failure fails admission. Accepted work survives a coordinator restart and is recovered in bounded batches; matching expiry runs from periodic maintenance independently of a new car arriving. Matching deadlines still apply. Expired, invalid and permanently unresolved work retains an explicit outcome rather than silently appearing as a successfully created session.

Before contacting PMS, VA commits the immutable callback, identity proof, consumed evidence and candidate finalized journey together. A successful callback atomically resolves its receipts and promotes the journey. If PMS committed but the response or VA process was lost, recovery retries the same decision. Restart recovery revalidates open identities with PMS before publishing them. An already-exited session cannot be republished as a live identity; known exit boundaries are retained. Finalized journey restoration protects against late CAM-23/CAM-03 rematching of the same entry.

Pair activation with PMS `ENTRY_V2_CONFIRMATION_RECEIPTS_ENABLED` and its additive receipt table: the unique decision receipt returns the original log/session IDs after a lost response, including `stale_after_exit` when the session has ended. The PMS camera write-ahead spool must also be enabled and persisted. VA durability alone does not make an unreceived camera event recoverable.

No automatic journal purge is implemented. Images and audit records require monitored disk space and an agreed retention policy. Disk exhaustion can refuse new durable admission; camera/PMS retry behavior and durable mounts must be verified operationally. The model still requires sufficient physical evidence to validate an entry; this feature does not turn an ANPR approach without a crossing into a parking session.

## Local evidence and release gates

The combined entry suite covers durable admission failure, pending recovery, callback loss/retry, post-commit crash recovery, late-camera deduplication, superseding exits, mode isolation, single-process ownership and periodic capacity recovery. Focused tests pass locally; these use local SQLite and injected failures, not a production SQL Server or a killed Kubernetes pod.

Before activation, verify mounted storage survives pod replacement, the exact deployed revisions and flags, SQL Server receipt migration/locks/collation, one owner per journal/spool, backlog and quarantine visibility, and the controlled camera-to-session journey. Exercise response loss after SQL commit, VA restart before and after acknowledgement, SQL outage while intake continues, stale callback after exit, and disk-full refusal. Confirm one session and retained entry image per validated entry. No deployment or production migration has been performed by this change.
