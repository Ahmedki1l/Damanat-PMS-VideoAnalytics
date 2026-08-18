"""Re-filing a car's gallery under a corrected plate.

PMS-AI corrects a stay whose ENTRY plate was misread — the exit read is the only
evidence that can catch that. If the gallery does not move with it, the crops
stay filed under the misread and VA re-mints it, so the correction survives in
PMS-AI's tables and dies here.

What must never happen is a reference disappearing. The dangerous case is the
one that looks like an edge case and is not: the corrected plate ALREADY has a
gallery, because the car has parked here before.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src.vehicle_registry.gallery_store import VehicleGalleryStore, safe_plate


def _store(tmp_path, max_refs: int = 10) -> VehicleGalleryStore:
    return VehicleGalleryStore(str(tmp_path), model_tag="test-tag", max_refs=max_refs)


def _plant(store: VehicleGalleryStore, plate: str, count: int, tag: str) -> list:
    """Write `count` references for `plate` straight to disk.

    Deliberately not through `add_reference`: this exercises rename, and the
    admission rules that guard writes would only get in the way of setting up
    a folder that is already on disk.
    """
    folder = os.path.join(str(store._root), safe_plate(plate))
    os.makedirs(folder, exist_ok=True)
    refs = []
    for index in range(count):
        stem = f"crop_{tag}_{index}"
        np.save(os.path.join(folder, stem + ".npy"), np.ones(4, dtype=np.float32))
        with open(os.path.join(folder, stem + ".jpg"), "wb") as fh:
            fh.write(b"jpeg")
        refs.append({
            "crop": stem + ".jpg",
            "vec": stem + ".npy",
            "model_tag": "test-tag",
            "quality": 0.5 + index / 100.0,
            "camera": f"CAM-{index % 3:02d}",
            "ts": f"2026-08-16T09:0{index % 10}:00",
        })
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"plate": plate, "model_tag": "test-tag", "refs": refs}, fh)
    return refs


def _meta(store: VehicleGalleryStore, plate: str) -> dict:
    path = os.path.join(str(store._root), safe_plate(plate), "meta.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _files(store: VehicleGalleryStore, plate: str) -> set:
    folder = os.path.join(str(store._root), safe_plate(plate))
    return set(os.listdir(folder)) if os.path.isdir(folder) else set()


def test_rename_moves_the_folder_when_the_target_is_free(tmp_path):
    store = _store(tmp_path)
    _plant(store, "AAA-2538", 3, "old")

    assert store.rename("AAA-2538", "KXR-2538") is True

    assert not os.path.isdir(os.path.join(str(store._root), safe_plate("AAA-2538")))
    meta = _meta(store, "KXR-2538")
    assert meta["plate"] == "KXR-2538", (
        "meta carries the plate; the folder name is safe_plate-mangled and not "
        "reversible, so all_plates() would keep reporting the misread"
    )
    assert len(meta["refs"]) == 3
    assert store.has("KXR-2538")


def test_rename_of_a_plate_with_no_gallery_reports_it(tmp_path):
    """Normal, not an error: most corrections are for cars whose stay produced
    no gallery at all. The caller must be able to tell that apart from a failure.
    """
    store = _store(tmp_path)
    assert store.rename("AAA-2538", "KXR-2538") is False


def test_merge_keeps_references_from_both_stays(tmp_path):
    """The car has been here before, so the corrected plate already has a folder.

    Dropping either set would be silent data loss: the misread folder holds THIS
    stay's crops, the target folder holds previous stays', and they are one car.
    """
    store = _store(tmp_path, max_refs=10)
    _plant(store, "KXR-2538", 2, "prev")
    _plant(store, "AAA-2538", 3, "this")

    assert store.rename("AAA-2538", "KXR-2538") is True

    meta = _meta(store, "KXR-2538")
    assert len(meta["refs"]) == 5, "both stays' references must survive"
    names = {ref["vec"] for ref in meta["refs"]}
    assert any("prev" in n for n in names) and any("this" in n for n in names)
    on_disk = _files(store, "KXR-2538")
    for ref in meta["refs"]:
        assert ref["vec"] in on_disk and ref["crop"] in on_disk, (
            "a meta record pointing at a file that is not there reads as a "
            "corrupt gallery"
        )
    assert not os.path.isdir(os.path.join(str(store._root), safe_plate("AAA-2538")))


def test_merge_prunes_to_max_refs_and_removes_the_dropped_files(tmp_path):
    """Merging must not let a gallery grow past its cap, and a pruned record
    must take its files with it — otherwise every correction leaks crops."""
    store = _store(tmp_path, max_refs=4)
    _plant(store, "KXR-2538", 3, "prev")
    _plant(store, "AAA-2538", 3, "this")

    assert store.rename("AAA-2538", "KXR-2538") is True

    meta = _meta(store, "KXR-2538")
    assert len(meta["refs"]) == 4
    kept = {ref["vec"] for ref in meta["refs"]} | {ref["crop"] for ref in meta["refs"]}
    leaked = {f for f in _files(store, "KXR-2538") if f != "meta.json"} - kept
    assert not leaked, f"pruned references left their files behind: {leaked}"


def test_rename_is_idempotent(tmp_path):
    """PMS-AI calls this fire-and-forget and cannot tell a lost reply from a
    failed rename, so a replay must be harmless."""
    store = _store(tmp_path)
    _plant(store, "AAA-2538", 2, "old")

    assert store.rename("AAA-2538", "KXR-2538") is True
    assert store.rename("AAA-2538", "KXR-2538") is False
    assert len(_meta(store, "KXR-2538")["refs"]) == 2


def test_two_plates_that_collapse_to_one_folder_only_rewrite_meta(tmp_path):
    """`safe_plate` mangles anything outside [A-Z0-9_-], so two different plate
    strings can share a folder. Moving it onto itself would delete it."""
    store = _store(tmp_path)
    _plant(store, "KXR 2538", 2, "old")

    assert store.rename("KXR 2538", "KXR/2538") is True

    assert _meta(store, "KXR/2538")["plate"] == "KXR/2538"
    assert len(_meta(store, "KXR/2538")["refs"]) == 2


def test_a_disk_failure_is_reported_not_raised(tmp_path, monkeypatch):
    """A correction must not fail because a disk did — PMS-AI's rewrite is
    already committed by the time this is called."""
    store = _store(tmp_path)
    _plant(store, "AAA-2538", 2, "old")

    def boom(*args, **kwargs):
        raise OSError("disk is on fire")

    monkeypatch.setattr(os, "replace", boom)

    assert store.rename("AAA-2538", "KXR-2538") is False
    assert store.has("AAA-2538"), "a failed rename must leave the refs reachable"
