"""The durable per-plate gallery, as a Re-ID evidence source for entry.

WHY THIS EXISTS. Until this module, `src/entry` never opened the gallery — not
one reference to it in the whole package. Every ramp crossing was scored against
the embeddings of THIS visit's ANPR attempts, which are two frontal gate crops.
That is a cross-view comparison with two reference vectors, the hardest match the
system can be asked to make, and it is why the measured accepted-score band tops
out at 0.819 with a median of 0.66.

Meanwhile the store the crops were already in is built for exactly this query.
`VehicleGalleryStore.load_vectors` filters `if not r.get("gate")` — gate/ANPR
crops are kept for provenance but EXCLUDED from matching, and what remains is
CAM-23, CAM-03 and parked views, pruned by `select_diverse_indices` for
viewpoint coverage so "a query from a different camera/angle" stays matchable.
On 2026-09-06 every one of the 27 confirmed entries already had such a gallery,
23 of them with a full twenty references, and none of them were consulted.

So the plate names the car (ANPR), and the ramp cameras describe it. Re-ID's
job at the ramp is to ask whether the car in front of CAM-23 is the car the gate
just named — and the strongest available answer to that is this car's own
previous CAM-23 views, not one frontal crop from thirty seconds ago.

TWO THINGS THIS MODULE IS CAREFUL ABOUT.

*Model contract.* `load_vectors`' own docstring warns that "a caller that scores
vectors directly, without that re-embed, must pass ``current_tag_only=True`` or
it will silently compare across contracts", and names a real production folder
(BHD-9990) holding 4 refs under one model tag and 16 under another. We are that
caller, so we pass it. A cosine distance between two models' vectors is noise
that would look exactly like a confident match.

*Sampling bias.* `max_similarity` over more vectors is biased upward: a car with
twenty references beats a car with two partly by having more chances to score
high, and the row margin assumes its candidates are comparable. Every candidate
is therefore capped to the SAME number of references, and the cap is applied by
`select_diverse_indices` rather than by quality order — taking the top-N sharpest
collapses the set onto one viewpoint, which is the failure the store's own
pruning exists to avoid.

Fails open in every direction. No store, no folder, an unreadable meta.json, a
folder written under a model that is no longer running — all return no
references, and the caller scores exactly what it scored before this module
existed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence, Tuple


logger = logging.getLogger(__name__)

Embedding = Tuple[float, ...]


@dataclass(frozen=True)
class GalleryLookup:
    """What the durable gallery could contribute for one plate.

    The counts are not diagnostics for their own sake: they are what tells a
    shadow review whether a weak score means "this car looks different today"
    or "we had nothing to compare it against". `dropped_model_tag` in
    particular separates an empty gallery from a full one written under a
    retired model.
    """

    vectors: Tuple[Embedding, ...] = ()
    cameras: Tuple[str, ...] = ()
    # References on disk under the CURRENT model contract, before the cap.
    available: int = 0
    # References skipped because they were embedded under another model.
    dropped_model_tag: int = 0
    # The folder name the references actually came from, when one was found.
    resolved_plate: str = ""

    @property
    def used(self) -> int:
        return len(self.vectors)

    def as_record(self) -> dict:
        """The `gallery` block of a decision-log record."""
        return {
            "used": self.used,
            "available": self.available,
            "dropped_model_tag": self.dropped_model_tag,
            "cameras": sorted({camera for camera in self.cameras if camera}),
            "resolved_plate": self.resolved_plate,
        }


EMPTY_LOOKUP = GalleryLookup()


class GalleryReferences(Protocol):
    def lookup(self, plate: str) -> GalleryLookup: ...


class NullGalleryReferences:
    """The default. Entry behaves exactly as it did before this module."""

    def lookup(self, plate: str) -> GalleryLookup:
        del plate
        return EMPTY_LOOKUP


class RegistryGalleryReferences:
    """Read-only view of the registry's on-disk gallery, capped and tag-checked.

    Read-only is deliberate for the shadow window. Teaching the gallery from a
    confirmed entry is a separate change with its own admission policy; this one
    only stops the matcher from ignoring what is already there, so a bad week
    can be rolled back by unsetting one flag with nothing to clean up on disk.
    """

    def __init__(self, registry, *, max_refs: int = 8):
        self._registry = registry
        self._max_refs = max(1, int(max_refs))
        # plate_key -> the folder name that plate is actually filed under.
        # `safe_plate` keeps hyphens, so the folder for HGD-2926 is "HGD-2926"
        # while the identity key is "HGD2926"; resolving by key alone would
        # miss every folder. Cached because the fallback is a directory scan.
        self._folder_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------ #

    @staticmethod
    def _key(value: object) -> str:
        return "".join(
            character
            for character in str(value or "").upper()
            if character.isalnum()
        )

    def _store(self):
        try:
            return getattr(self._registry, "gallery_store", None)
        except Exception:  # pragma: no cover - defensive: registry may lazy-init
            return None

    def _resolve_folder(self, store, plate: str) -> str:
        """The plate string this car's gallery is actually filed under.

        The common case is the string we were handed: PMS-AI forwards the same
        plate to the legacy path that created the folder. The scan is the
        fallback for a punctuation difference between what ANPR reported and
        what the folder was named.
        """
        key = self._key(plate)
        if not key:
            return ""
        cached = self._folder_cache.get(key)
        if cached is not None:
            return cached
        resolved = ""
        try:
            if store.has(plate):
                resolved = plate
            else:
                for candidate in store.all_plates():
                    if self._key(candidate) == key:
                        resolved = candidate
                        break
        except Exception as exc:  # pragma: no cover - disk/permission failure
            logger.debug("[EntryV2][gallery] folder lookup failed for %s: %r", plate, exc)
            return ""
        self._folder_cache[key] = resolved
        return resolved

    def lookup(self, plate: str) -> GalleryLookup:
        store = self._store()
        if store is None or not plate:
            return EMPTY_LOOKUP
        folder = self._resolve_folder(store, plate)
        if not folder:
            return EMPTY_LOOKUP

        try:
            # current_tag_only=True is REQUIRED here, not a tuning choice: we
            # score these vectors directly and never re-embed the crops.
            current, _, current_cameras = store.load_vectors(
                folder, current_tag_only=True
            )
            everything, _, _ = store.load_vectors(folder)
        except Exception as exc:  # pragma: no cover - unreadable meta/crops
            logger.debug("[EntryV2][gallery] load failed for %s: %r", folder, exc)
            return EMPTY_LOOKUP

        dropped = max(0, len(everything) - len(current))
        if not current:
            return GalleryLookup(
                dropped_model_tag=dropped,
                resolved_plate=folder,
            )

        indices = self._cap_indices(current)
        vectors = tuple(
            tuple(float(value) for value in current[index]) for index in indices
        )
        cameras = tuple(
            str(current_cameras[index]) if index < len(current_cameras) else ""
            for index in indices
        )
        return GalleryLookup(
            vectors=vectors,
            cameras=cameras,
            available=len(current),
            dropped_model_tag=dropped,
            resolved_plate=folder,
        )

    def _cap_indices(self, vectors: Sequence) -> Sequence[int]:
        """Which references to keep, capped identically for every candidate.

        Diversity rather than quality order: the store prunes for viewpoint
        coverage precisely because sharpness order collapses a gallery onto one
        dominant view, and a capped set chosen that way would reintroduce the
        cross-view weakness this module exists to remove.
        """
        if len(vectors) <= self._max_refs:
            return range(len(vectors))
        try:
            from src.vehicle_registry.gallery_store import select_diverse_indices

            chosen = select_diverse_indices(list(vectors), self._max_refs)
            if chosen:
                return sorted(chosen)
        except Exception as exc:  # pragma: no cover - selection is best-effort
            logger.debug("[EntryV2][gallery] diverse selection failed: %r", exc)
        # load_vectors returns primary-first, so the head is the safe fallback.
        return range(self._max_refs)
