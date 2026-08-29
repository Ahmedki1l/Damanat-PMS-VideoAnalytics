"""Stage 1 — the Entry V2 decision log: wiring, retention, and record shape.

The Entry V2 pipeline raises no alerts, so this file IS the operational surface:
if a record is not written, or is written with the wrong field names, nobody can
see what the pipeline decided and the shadow review has nothing to review.
"""
import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from src.entry import decision_record
from src.entry.decision import ReIDMatchEvaluation
from src.entry.domain import (
    CrossingInput,
    CrossingRecord,
    CrossingRole,
    EntryMode,
)
from src.entry.runtime import (
    ENTRY_DECISION_LOG_PREFIX,
    build_decision_log,
    _prepare_decision_log_dir,
)
from src.entry.settings import EntrySettings
from src.matching.decision_log import DecisionLog, prune_old_logs


def _crossing(
    crossing_id="cr-1",
    camera_id="CAM-23",
    role=CrossingRole.PRIMARY,
    captured_at=None,
):
    request = CrossingInput(
        crossing_id=crossing_id,
        source_event_id="evt-1",
        camera_id=camera_id,
        captured_at=captured_at or datetime(2026, 8, 29, 10, 0, tzinfo=timezone.utc),
        line_id="Park_Entry",
        direction="ramp-entry",
        role=role,
        metadata={"source": "hikvision"},
    )
    return CrossingRecord(request=request, evidence=())


def _evaluation(accepted=True):
    return ReIDMatchEvaluation(
        group_id="grp-abc",
        score=0.8123456,
        row_runner=0.41,
        row_margin=0.4023456,
        column_runner=0.0,
        column_margin=0.8123456,
        reason="accepted" if accepted else "row_margin_below_minimum",
        match=object() if accepted else None,
    )


def _settings(**overrides) -> EntrySettings:
    return dataclasses.replace(EntrySettings(), **overrides)


class _TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="entryv2-log-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Record shape — the data contract the calibration tool reads
# --------------------------------------------------------------------------- #
class RecordShapeTests(unittest.TestCase):
    def test_reid_record_carries_scores_and_the_thresholds_they_were_held_to(self):
        settings = _settings(
            reid_min_score=0.75, reid_row_margin=0.08, reid_column_margin=0.08
        )
        record = decision_record.build_record(
            stage="reid_evaluation",
            result=decision_record.RESULT_CONFIRMED,
            reason="accepted",
            crossing=_crossing(),
            evaluation=_evaluation(),
            settings=settings,
        )

        self.assertEqual("entry_decision", record["event"])
        self.assertEqual("reid_evaluation", record["stage"])
        self.assertEqual("confirmed", record["result"])

        reid = record["reid"]
        self.assertEqual("grp-abc", reid["argmax"])
        self.assertAlmostEqual(0.812346, reid["score"])
        self.assertAlmostEqual(0.402346, reid["row_margin"])
        # A record read six weeks later must say what bar it was held to.
        self.assertEqual(0.75, reid["min_score"])
        self.assertEqual(0.08, reid["min_row_margin"])
        self.assertEqual(0.08, reid["min_column_margin"])
        self.assertTrue(reid["accepted"])

    def test_observation_block_names_the_witness_not_a_plate_source(self):
        record = decision_record.build_record(
            stage="reid_evaluation",
            result=decision_record.RESULT_ABSTAINED,
            reason="row_margin_below_minimum",
            crossing=_crossing(camera_id="CAM-03", role=CrossingRole.FALLBACK),
            evaluation=_evaluation(accepted=False),
            settings=_settings(),
        )
        observation = record["observation"]
        self.assertEqual("cam03", observation["witness"])
        self.assertEqual("CAM-03", observation["camera"])
        self.assertEqual("fallback", observation["role"])
        self.assertEqual("hikvision", observation["source"])
        self.assertEqual("2026-08-29T10:00:00+00:00", observation["captured_at"])

    def test_cam23_maps_to_the_cam23_witness(self):
        record = decision_record.build_record(
            stage="s", result=decision_record.RESULT_ABSTAINED, reason="r",
            crossing=_crossing(camera_id="CAM-23"),
        )
        self.assertEqual("cam23", record["observation"]["witness"])

    def test_an_unrecognised_camera_is_reported_rather_than_bucketed(self):
        record = decision_record.build_record(
            stage="s", result=decision_record.RESULT_ABSTAINED, reason="r",
            crossing=_crossing(camera_id="CAM-99"),
        )
        self.assertEqual("cam99", record["observation"]["witness"])

    def test_blocks_from_later_stages_are_absent_not_null(self):
        # A consumer must be able to tell "did not happen" from "not measured".
        record = decision_record.build_record(
            stage="reid_evaluation",
            result=decision_record.RESULT_ABSTAINED,
            reason="score_below_minimum",
            crossing=_crossing(),
            evaluation=_evaluation(accepted=False),
            settings=_settings(),
        )
        for absent in ("identity", "witnesses", "colour", "fifo", "hik", "plate"):
            self.assertNotIn(absent, record)

    def test_ramp_camera_ocr_is_recorded_as_diagnostic_only(self):
        record = decision_record.build_record(
            stage="reid_evaluation",
            result=decision_record.RESULT_ABSTAINED,
            reason="r",
            crossing=_crossing(),
            observed_plate_text="ABC123",
            observed_plate_confidence=0.4211,
        )
        # Present for later analysis...
        self.assertEqual("ABC123", record["observed_plate_text"])
        self.assertAlmostEqual(0.4211, record["observed_plate_confidence"])
        # ...but NEVER as a plate source. CAM-23/CAM-03 do not read plates.
        self.assertNotIn("plate", record)

    def test_record_is_json_serialisable(self):
        record = decision_record.build_record(
            stage="reid_evaluation",
            result=decision_record.RESULT_CONFIRMED,
            reason="accepted",
            crossing=_crossing(),
            evaluation=_evaluation(),
            settings=_settings(),
        )
        self.assertIsInstance(json.dumps(record), str)

    def test_result_vocabulary_is_closed(self):
        self.assertEqual(
            {
                "confirmed",
                "ambiguous",
                "abstained",
                "unreadable",
                "expired",
                "hik_degraded",
            },
            set(decision_record.RESULTS),
        )


# --------------------------------------------------------------------------- #
# Writer behaviour
# --------------------------------------------------------------------------- #
class WriterTests(_TempDirTest):
    def _enabled_log(self, **kwargs):
        """Build a writer with the pytest guard lifted for this construction."""
        env = dict(os.environ)
        env.pop("PYTEST_CURRENT_TEST", None)
        with mock.patch.dict(os.environ, env, clear=True):
            log = DecisionLog(directory=self.tmp, worker="api", **kwargs)
        self.addCleanup(log.close)
        return log

    def test_writer_is_disabled_under_pytest_whatever_the_caller_asked(self):
        # The guard that stops synthetic fixtures poisoning a real corpus.
        log = DecisionLog(directory=self.tmp, worker="api", enabled=True)
        self.addCleanup(log.close)
        self.assertFalse(log.enabled)
        log.emit({"event": "entry_decision"})
        self.assertEqual([], os.listdir(self.tmp))

    def test_prefix_produces_the_entry_filename(self):
        log = self._enabled_log(prefix=ENTRY_DECISION_LOG_PREFIX)
        log.emit({"event": "entry_decision", "result": "confirmed"})
        log.close()

        files = os.listdir(self.tmp)
        self.assertEqual(1, len(files))
        today = datetime.now().strftime("%Y-%m-%d")
        self.assertEqual(f"entry_decisions_api_{today}.jsonl", files[0])

    def test_default_prefix_is_unchanged_for_the_existing_caller(self):
        log = self._enabled_log()
        log.emit({"event": "slot_identity_attempt"})
        log.close()
        self.assertTrue(os.listdir(self.tmp)[0].startswith("decisions_api_"))

    def test_emitted_record_round_trips_with_the_envelope(self):
        log = self._enabled_log(prefix=ENTRY_DECISION_LOG_PREFIX)
        log.emit(
            decision_record.build_record(
                stage="reid_evaluation",
                result=decision_record.RESULT_CONFIRMED,
                reason="accepted",
                crossing=_crossing(),
                evaluation=_evaluation(),
                settings=_settings(),
            )
        )
        log.close()

        path = os.path.join(self.tmp, os.listdir(self.tmp)[0])
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("api", row["worker"])
        self.assertIn("ts", row)
        self.assertIn("v", row)
        self.assertEqual("cr-1", row["observation"]["id"])
        self.assertEqual("grp-abc", row["reid"]["argmax"])

    def test_emit_drops_rather_than_blocks_when_the_queue_is_full(self):
        # The coordinator emits under its lock; blocking here would stall every
        # camera behind a disk write.
        log = self._enabled_log(prefix=ENTRY_DECISION_LOG_PREFIX, max_queue=1)
        log._thread = None  # stop the drain so the queue genuinely fills
        for _ in range(50):
            log.emit({"event": "entry_decision"})
        self.assertGreater(log._dropped, 0)


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
class RetentionTests(_TempDirTest):
    def _touch(self, name):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{}\n")
        return path

    def test_files_older_than_the_retention_are_removed(self):
        old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
        recent = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        self._touch(f"entry_decisions_api_{old}.jsonl")
        self._touch(f"entry_decisions_api_{recent}.jsonl")

        removed = prune_old_logs(self.tmp, 30, ENTRY_DECISION_LOG_PREFIX)

        self.assertEqual(1, removed)
        self.assertEqual(
            [f"entry_decisions_api_{recent}.jsonl"], os.listdir(self.tmp)
        )

    def test_retention_never_touches_the_slot_identity_corpus(self):
        old = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        self._touch(f"decisions_api_{old}.jsonl")          # other corpus
        self._touch(f"entry_decisions_api_{old}.jsonl")    # ours

        prune_old_logs(self.tmp, 30, ENTRY_DECISION_LOG_PREFIX)

        self.assertEqual([f"decisions_api_{old}.jsonl"], os.listdir(self.tmp))

    def test_retention_uses_the_filename_day_not_mtime(self):
        # A file copied off the pod and back would look new by mtime and never
        # expire.
        old = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        path = self._touch(f"entry_decisions_api_{old}.jsonl")
        os.utime(path, None)  # mtime = now

        self.assertEqual(1, prune_old_logs(self.tmp, 30, ENTRY_DECISION_LOG_PREFIX))

    def test_unparsable_and_foreign_names_are_left_alone(self):
        self._touch("entry_decisions_api_not-a-date.jsonl")
        self._touch("entry_decisions_api_2020-01-01.txt")
        self._touch("README.md")

        self.assertEqual(0, prune_old_logs(self.tmp, 30, ENTRY_DECISION_LOG_PREFIX))
        self.assertEqual(3, len(os.listdir(self.tmp)))

    def test_missing_directory_is_not_an_error(self):
        self.assertEqual(
            0, prune_old_logs(os.path.join(self.tmp, "nope"), 30, "entry_decisions")
        )

    def test_zero_retention_prunes_nothing(self):
        old = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        self._touch(f"entry_decisions_api_{old}.jsonl")
        self.assertEqual(0, prune_old_logs(self.tmp, 0, ENTRY_DECISION_LOG_PREFIX))


# --------------------------------------------------------------------------- #
# Factory wiring
# --------------------------------------------------------------------------- #
class FactoryWiringTests(_TempDirTest):
    def test_no_directory_configured_means_no_writer_and_no_side_effects(self):
        self.assertIsNone(build_decision_log(_settings(decision_log_dir="")))

    def test_directory_is_created_even_when_entry_v2_is_off(self):
        # Deployment step 1 ships with mode=off and verifies the directory
        # exists before shadow is ever enabled.
        target = os.path.join(self.tmp, "entry_v2_shadow")
        _prepare_decision_log_dir(
            _settings(mode=EntryMode.OFF, decision_log_dir=target)
        )
        self.assertTrue(os.path.isdir(target))

    def test_preparation_prunes_expired_files(self):
        old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
        with open(
            os.path.join(self.tmp, f"entry_decisions_api_{old}.jsonl"),
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write("{}\n")

        _prepare_decision_log_dir(
            _settings(decision_log_dir=self.tmp, decision_log_retention_days=30)
        )
        self.assertEqual([], os.listdir(self.tmp))

    def test_an_unusable_directory_does_not_raise(self):
        # A log volume that cannot be prepared is a degraded review, not a
        # reason to keep the cameras down.
        with mock.patch("os.makedirs", side_effect=OSError("read-only")):
            _prepare_decision_log_dir(_settings(decision_log_dir=self.tmp))

    def test_worker_name_comes_from_va_group(self):
        with mock.patch.dict(os.environ, {"VA_GROUP": "gate"}, clear=False):
            log = build_decision_log(_settings(decision_log_dir=self.tmp))
        self.addCleanup(log.close)
        self.assertEqual("gate", log._worker)
        self.assertEqual(ENTRY_DECISION_LOG_PREFIX, log._prefix)


# --------------------------------------------------------------------------- #
# Settings validation
# --------------------------------------------------------------------------- #
class SettingsTests(unittest.TestCase):
    def _errors(self, **overrides):
        base = dict(
            mode=EntryMode.SHADOW,
            primary_lines=frozenset({"PARK_ENTRY_X"}),
            pms_base_url="http://pms:8000",
            service_key="k" * 8,
        )
        base.update(overrides)
        return _settings(**base).configuration_errors()

    def test_defaults_are_off_and_thirty_days(self):
        settings = EntrySettings()
        self.assertEqual("", settings.decision_log_dir)
        self.assertEqual(30, settings.decision_log_retention_days)

    def test_env_is_read(self):
        with mock.patch.dict(
            os.environ,
            {
                "ENTRY_V2_DECISION_LOG_DIR": "/app/vehicle_images/entry_v2_shadow",
                "ENTRY_V2_DECISION_LOG_RETENTION_DAYS": "7",
            },
            clear=False,
        ):
            settings = EntrySettings.from_env()
        self.assertEqual(
            "/app/vehicle_images/entry_v2_shadow", settings.decision_log_dir
        )
        self.assertEqual(7, settings.decision_log_retention_days)

    def test_malformed_retention_fails_configuration_when_the_log_is_on(self):
        # _env_int yields 0 for a malformed value, and 0 would silently mean
        # "never prune" on the volume that also holds vehicle imagery.
        errors = self._errors(decision_log_dir="/tmp/x", decision_log_retention_days=0)
        self.assertIn("ENTRY_V2_DECISION_LOG_RETENTION_DAYS", errors)

    def test_retention_is_not_checked_when_the_log_is_off(self):
        errors = self._errors(decision_log_dir="", decision_log_retention_days=0)
        self.assertNotIn("ENTRY_V2_DECISION_LOG_RETENTION_DAYS", errors)

    def test_malformed_queue_max_fails_configuration(self):
        errors = self._errors(decision_log_dir="/tmp/x", decision_log_queue_max=0)
        self.assertIn("ENTRY_V2_DECISION_LOG_QUEUE_MAX", errors)


# --------------------------------------------------------------------------- #
# Coordinator emission
# --------------------------------------------------------------------------- #
class _CollectingLog:
    """Stands in for DecisionLog; records what the coordinator queued."""

    def __init__(self, explode=False):
        self.records = []
        self._explode = explode

    def emit(self, record):
        if self._explode:
            raise RuntimeError("disk on fire")
        self.records.append(record)


class _StubProcessor:
    def analyze(self, **kwargs):  # pragma: no cover - never called here
        raise AssertionError("stage 1 does not run inference")


class _StubSink:
    def deliver(self, payload):  # pragma: no cover - never called here
        raise AssertionError("stage 1 delivers nothing")


class CoordinatorEmissionTests(unittest.TestCase):
    def _coordinator(self, log):
        from src.entry.coordinator import EntryCoordinator

        return EntryCoordinator(
            _settings(mode=EntryMode.SHADOW),
            _StubProcessor(),
            _StubSink(),
            decision_log=log,
        )

    def test_reid_evaluation_emits_one_record_beside_its_text_line(self):
        log = _CollectingLog()
        coordinator = self._coordinator(log)

        coordinator._log_reid_evaluation_locked(_crossing(), _evaluation())

        self.assertEqual(1, len(log.records))
        record = log.records[0]
        self.assertEqual("reid_evaluation", record["stage"])
        self.assertEqual("confirmed", record["result"])
        self.assertEqual("accepted", record["reason"])
        self.assertEqual("grp-abc", record["reid"]["argmax"])
        self.assertEqual("cam23", record["observation"]["witness"])

    def test_a_margin_failure_records_ambiguous_not_abstained(self):
        # Refined in stage 3: clearing the absolute score but failing a margin
        # means two plausible cars, which is a different problem from one weak
        # look at a car. The review has to be able to tell them apart.
        log = _CollectingLog()
        coordinator = self._coordinator(log)

        coordinator._log_reid_evaluation_locked(_crossing(), _evaluation(accepted=False))

        record = log.records[0]
        self.assertEqual("ambiguous", record["result"])
        self.assertEqual("row_margin_below_minimum", record["reason"])
        self.assertFalse(record["reid"]["accepted"])

    def test_a_score_failure_records_abstained(self):
        log = _CollectingLog()
        coordinator = self._coordinator(log)
        evaluation = dataclasses.replace(
            _evaluation(accepted=False),
            score=0.10,
            reason="score_below_minimum",
        )

        coordinator._log_reid_evaluation_locked(_crossing(), evaluation)

        record = log.records[0]
        self.assertEqual("abstained", record["result"])
        self.assertEqual("score_below_minimum", record["reason"])

    def test_the_existing_fingerprint_dedup_still_suppresses_a_repeat(self):
        log = _CollectingLog()
        coordinator = self._coordinator(log)
        crossing, evaluation = _crossing(), _evaluation()

        coordinator._log_reid_evaluation_locked(crossing, evaluation)
        coordinator._log_reid_evaluation_locked(crossing, evaluation)

        self.assertEqual(1, len(log.records))

    def test_a_changed_evaluation_is_recorded_again(self):
        log = _CollectingLog()
        coordinator = self._coordinator(log)
        crossing = _crossing()

        coordinator._log_reid_evaluation_locked(crossing, _evaluation())
        coordinator._log_reid_evaluation_locked(crossing, _evaluation(accepted=False))

        self.assertEqual(2, len(log.records))

    def test_no_log_configured_is_a_silent_no_op(self):
        coordinator = self._coordinator(None)
        coordinator._log_reid_evaluation_locked(_crossing(), _evaluation())

    def test_a_failing_log_can_never_break_the_pipeline(self):
        # The log exists to observe the pipeline; it must not be able to stop it.
        coordinator = self._coordinator(_CollectingLog(explode=True))
        coordinator._log_reid_evaluation_locked(_crossing(), _evaluation())


if __name__ == "__main__":
    unittest.main()


class HikDirectionTests(unittest.TestCase):
    """HikCentral is a PULL source: we call it, it never calls us.

    Pinned here because stages 5 and 6 are where it could drift. The `hik`
    block records a query OUR service issued in response to OUR event, and the
    schema has to make that unambiguous to whoever reads the corpus later.
    """

    def test_the_hik_block_records_a_call_we_made(self):
        record = decision_record.build_record(
            stage="hik_enrich",
            result=decision_record.RESULT_ABSTAINED,
            reason="identity_enriched",
            hik={
                "trigger": "anpr_identity",
                "queried": True,
                "window": ["2026-08-29T12:00:00+00:00", "2026-08-29T12:00:35+00:00"],
                "records": 3,
                "images": 2,
                "reid_matched": ["guid-1"],
                "unmatched": ["guid-2", "guid-3"],
                "api_error": None,
            },
        )
        hik = record["hik"]
        # The trigger names OUR event, never a HikCentral one.
        self.assertEqual("anpr_identity", hik["trigger"])
        self.assertTrue(hik["queried"])
        # We asked for a window; HikCentral did not choose one for us.
        self.assertEqual(2, len(hik["window"]))
        # WE decided which returned records belong to this car.
        self.assertEqual(["guid-1"], hik["reid_matched"])
        self.assertEqual(2, len(hik["unmatched"]))

    def test_the_two_triggers_are_both_our_own_events(self):
        for trigger in ("anpr_identity", "missing_anpr_recovery"):
            record = decision_record.build_record(
                stage="hik_enrich",
                result=decision_record.RESULT_ABSTAINED,
                reason="r",
                hik={"trigger": trigger, "queried": True},
            )
            self.assertEqual(trigger, record["hik"]["trigger"])

    def test_an_unreachable_platform_is_a_degraded_query_not_a_lost_event(self):
        # There was no event of theirs to lose. The entry proceeds on ANPR and
        # camera evidence (rule 18A).
        record = decision_record.build_record(
            stage="hik_enrich",
            result=decision_record.RESULT_HIK_DEGRADED,
            reason="hik_api_unavailable",
            hik={
                "trigger": "anpr_identity",
                "queried": True,
                "records": 0,
                "api_error": "timeout",
            },
        )
        self.assertEqual("hik_degraded", record["result"])
        self.assertEqual("timeout", record["hik"]["api_error"])

    def test_not_calling_is_recorded_as_our_choice(self):
        record = decision_record.build_record(
            stage="hik_enrich",
            result=decision_record.RESULT_ABSTAINED,
            reason="hik_disabled",
            hik={"trigger": "anpr_identity", "queried": False},
        )
        # queried:false means WE chose not to call, never that HikCentral
        # chose not to tell us.
        self.assertFalse(record["hik"]["queried"])
