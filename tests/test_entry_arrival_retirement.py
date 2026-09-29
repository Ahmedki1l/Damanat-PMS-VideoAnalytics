"""A confirmed later lane arrival permanently disqualifies older pending visits."""
from dataclasses import replace
from datetime import timedelta

from src.entry.coordinator import EntryCoordinator
from src.entry.callback import DeliveryResult
from src.entry.durability import EntryDurabilityStore
from src.entry.domain import RecordStatus
from tests.test_entry_v3_gallery_and_retirement import (
    NOW, _Processor, _Sink, attempt, crossing, frame, settings,
)


class CommittedSink(_Sink):
    def deliver(self, payload):
        self.payloads.append(dict(payload))
        return DeliveryResult(True, 1, publish_identity=True, session_committed=True)


def build(store=None):
    evidence = {
        'old': [frame('old', 'ANPR-ENTRY', (1., 0.))],
        'new': [frame('new', 'ANPR-ENTRY', (0., 1.))],
        'cross': [frame('cross', 'CAM-23', (0., 1.), role='primary')],
        'late': [frame('late', 'ANPR-ENTRY', (1., 0.))],
        'return': [frame('return', 'ANPR-ENTRY', (1., 0.))],
        'other': [frame('other', 'OTHER', (1., 0.))],
    }
    return EntryCoordinator(settings(), _Processor(evidence), CommittedSink(), durability_store=store)


def confirm_new(coord):
    coord.ingest_attempt(attempt('new', 'NEW-123', NOW + timedelta(seconds=10)), [b'new'])
    result = coord.ingest_crossing(crossing('cross'), [b'cross'])
    assert result.decision_status == 'confirmed'


def test_newer_confirmation_retires_old_identity_but_not_other_lane():
    coord = build()
    old = coord.ingest_attempt(attempt('old', 'OLD-123'), [b'old'])
    other = coord.ingest_attempt(replace(attempt('other', 'OLD-123'), camera_id='OTHER'), [b'other'])
    assert old.group_id != other.group_id
    confirm_new(coord)
    assert coord._groups[old.group_id].status == RecordStatus.RESOLVED
    assert coord._groups[other.group_id].status == RecordStatus.PENDING
    retry = coord.ingest_attempt(attempt('old', 'OLD-123'), [b'old'])
    assert not retry.accepted
    assert retry.receipt_status == 'expired'


def test_late_old_capture_new_id_retired_but_new_visit_allowed():
    coord = build()
    confirm_new(coord)
    late = coord.ingest_attempt(attempt('late', 'OLD-123'), [b'late'])
    assert not late.accepted
    assert late.receipt_status == 'expired'
    returned = coord.ingest_attempt(attempt('return', 'OLD-123', NOW + timedelta(seconds=40)), [b'return'])
    assert returned.accepted
    assert coord._groups[returned.group_id].status == RecordStatus.PENDING


def test_retirement_survives_restart_and_keeps_evidence(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    coord = build(store)
    coord.ingest_attempt(attempt('old', 'OLD-123'), [b'old'])
    confirm_new(coord)
    assert not store.is_pending('attempt', 'old')
    store.close()
    store = EntryDurabilityStore(str(tmp_path))
    restarted = build(store)
    restarted.recover_durable_inputs()
    assert not restarted._groups
    late = restarted.ingest_attempt(attempt('late', 'OLD-123'), [b'late'])
    assert not late.accepted
    assert not store.is_pending('attempt', 'late')
    with store._connect() as connection:
        row = connection.execute("SELECT images_json FROM entry_receipts WHERE resource_id='old'").fetchone()
    assert row and row[0] != '[]'
    store.close()


def test_equal_capture_time_is_not_retired():
    coord = build()
    old = coord.ingest_attempt(attempt('old', 'OLD-123', NOW + timedelta(seconds=10)), [b'old'])
    confirm_new(coord)
    assert coord._groups[old.group_id].status == RecordStatus.PENDING


def test_repeated_later_read_does_not_rescue_older_pending_car():
    coord = build()
    old = coord.ingest_attempt(attempt('old', 'OLD-123'), [b'old'])
    repeat = coord.ingest_attempt(attempt('return', 'OLD-123', NOW + timedelta(seconds=20)), [b'return'])
    assert repeat.group_id == old.group_id
    confirm_new(coord)
    assert coord._groups[old.group_id].status == RecordStatus.RESOLVED


def test_retired_car_cannot_claim_subsequent_crossing():
    coord = build()
    coord.ingest_attempt(attempt('old', 'OLD-123'), [b'old'])
    confirm_new(coord)
    coord._processor.evidence_by_event['late-cross'] = [
        frame('late-cross', 'CAM-23', (1., 0.), role='primary')
    ]
    result = coord.ingest_crossing(crossing('late-cross', NOW + timedelta(seconds=60)), [b'late-cross'])
    assert result.decision_status != 'confirmed'
    assert [p['canonical_plate'] for p in coord._sink.payloads] == ['NEW-123']


def test_same_plate_new_visit_can_confirm_after_retirement():
    coord = build()
    coord.ingest_attempt(attempt('old', 'OLD-123'), [b'old'])
    confirm_new(coord)
    coord.ingest_attempt(attempt('return', 'OLD-123', NOW + timedelta(seconds=40)), [b'return'])
    coord._processor.evidence_by_event['return-cross'] = [
        frame('return-cross', 'CAM-23', (1., 0.), role='primary')
    ]
    result = coord.ingest_crossing(crossing('return-cross', NOW + timedelta(seconds=60)), [b'return-cross'])
    assert result.decision_status == 'confirmed'
    assert [p['canonical_plate'] for p in coord._sink.payloads] == ['NEW-123', 'OLD-123']


def test_hik_replay_from_another_camera_is_not_retired():
    coord = build()
    confirm_new(coord)
    request = replace(attempt('other', 'NEW-123', hik=True), camera_id='OTHER')
    result = coord.ingest_attempt(request, [b'other'])
    assert result.accepted
    assert coord._groups[result.group_id].status == RecordStatus.PENDING


def test_journey_camera_scope_survives_serialization():
    coord = build()
    confirm_new(coord)
    journey = next(iter(coord._finalized_journeys.values()))
    record = coord._durability_finalized_record(journey)
    restored = coord._durability_finalized_journey_from_record(record)
    assert restored.arrival_camera_ids == ('ANPRENTRY',)
    assert restored == journey
    record.pop('arrival_camera_ids')
    assert coord._durability_finalized_journey_from_record(record).arrival_camera_ids == ()


def test_restored_journey_retires_only_same_camera_hik_replay(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    coord = build(store)
    confirm_new(coord)
    store.close()
    store = EntryDurabilityStore(str(tmp_path))
    restarted = build(store)
    restarted.recover_durable_inputs()
    # After its gate arrival but before its confirmed crossing: this tests the
    # finalized-journey rule, not the older-arrival boundary.
    captured = NOW + timedelta(seconds=15)
    same = restarted.ingest_attempt(attempt('late', 'NEW-123', captured, hik=True), [b'late'])
    other = restarted.ingest_attempt(replace(attempt('other', 'NEW-123', captured, hik=True), camera_id='OTHER'), [b'other'])
    assert restarted._groups[same.group_id].status == RecordStatus.RESOLVED
    assert restarted._groups[other.group_id].status == RecordStatus.PENDING
    store.close()
