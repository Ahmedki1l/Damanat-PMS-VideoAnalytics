"""Purge full-frame (non-car) references from the per-plate gallery.

WHY. src/api.py parks the RAW ANPR gate frame on a Park_Entry candidate on
purpose — it is the durable gate photo. The CAM-03 confirmation's "Fix 4"
fallback then asked for "the latest Park_Entry candidate for this plate" without
filtering by camera, so whenever CAM-23 had not seeded (the common case) it got
that ANPR candidate and filed a 2688x1552 frame of road-and-sky as a MATCHABLE
reference. Measured 2026-07-15: 62 such refs across all 38 plate folders.

They are not merely useless, they are harmful:
  * a full frame embeds the SCENE, not the car — 0.41 cosine against its own
    car's crops (below the match bar), so it never helps;
  * 0.33 against OTHER cars' gate frames, because every gate frame shares a
    background — that is similarity with no identity in it;
  * multishot scores them ~1900 on sharpness (a wide scene is texture-rich)
    while a real crop is 999, so `_prune_meta_inplace` — which sorts by quality
    within each camera bucket — keeps the FRAME and evicts the good crop once a
    folder passes gallery_max_refs_per_car.

The leak itself is fixed in code (camera filter on
latest_park_entry_candidate_for_plate, geometry gate in
seed_gallery_from_park_entry, full-frame bar in is_plausible_car_crop). This
removes the refs already written, which no code change can undo.

Safe by design: DRY-RUN unless you pass --apply. Only refs whose crop is at/over
the full-frame bar are touched; meta.json is rewritten without them and the
orphaned .jpg/.npy are deleted. A folder is never emptied — if every ref in it
would go, the folder is REPORTED AND SKIPPED, because a car with no references
is invisible to reid_rank (it skips vectorless sessions) and can then never be
identified at all, which is worse than a bad reference.

Run on the RUN MACHINE, from the project root:

    python tools/purge_fullframe_refs.py                    # dry run (default)
    python tools/purge_fullframe_refs.py --apply            # actually delete
    python tools/purge_fullframe_refs.py --gallery DIR      # non-default path
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vehicle_registry.vehicle_registry_identity import is_plausible_car_crop


def _default_gallery_dir() -> str:
    """Mirror the engine's gallery location so the tool needs no arguments.

    VehicleGalleryStore roots itself at ``<output.snapshot_base_dir>/gallery``
    (gallery_store.py), and snapshot_base_dir honours $SNAPSHOT_PATH.
    """
    base = "vehicle_images"
    try:
        from src.config import load_config

        base = load_config("config.yaml").output.snapshot_base_dir or base
    except Exception:
        base = os.environ.get("SNAPSHOT_PATH", base)
    return os.path.join(base, "gallery")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", default=_default_gallery_dir(),
                    help="per-plate gallery root (default: the engine's)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; omit for a dry run")
    args = ap.parse_args()

    import cv2  # imported here so --help works without OpenCV

    root = args.gallery
    if not os.path.isdir(root):
        print(f"[ERR] gallery dir not found: {root}")
        return 2

    print(f"[gallery] {root}")
    print(f"[mode] {'APPLY — deleting' if args.apply else 'DRY RUN — nothing will be written'}\n")

    tot_refs = tot_bad = tot_folders = 0
    skipped_folders = []

    for plate in sorted(os.listdir(root)):
        d = os.path.join(root, plate)
        meta_path = os.path.join(d, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"  {plate:12s} !! unreadable meta.json ({exc!r}) — skipped")
            continue

        refs = meta.get("refs", [])
        tot_refs += len(refs)
        keep, drop = [], []
        for r in refs:
            crop_path = os.path.join(d, r.get("crop", ""))
            img = cv2.imread(crop_path) if r.get("crop") else None
            # A ref whose crop is gone is left alone: absence is not evidence of
            # a full frame, and load_vectors tolerates it.
            (drop if img is not None and not is_plausible_car_crop(img) else keep).append(r)

        if not drop:
            continue
        tot_folders += 1
        tot_bad += len(drop)

        dims = []
        for r in drop:
            img = cv2.imread(os.path.join(d, r["crop"]))
            dims.append(f"{img.shape[1]}x{img.shape[0]}@q{r.get('quality', 0):.0f}")

        if not keep:
            skipped_folders.append(plate)
            print(f"  {plate:12s} SKIPPED — all {len(drop)} refs are full frames; "
                  f"purging would leave the car unidentifiable. Re-seed it instead.")
            continue

        print(f"  {plate:12s} drop {len(drop)}/{len(refs)}  keep {len(keep)}   [{', '.join(dims)}]")

        if not args.apply:
            continue
        for r in drop:
            for key in ("crop", "vec"):
                fn = r.get(key)
                if not fn:
                    continue
                try:
                    os.remove(os.path.join(d, fn))
                except OSError:
                    pass
        meta["refs"] = keep
        tmp = meta_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            os.replace(tmp, meta_path)  # atomic: a torn meta.json loses the car
        except OSError as exc:
            print(f"  {plate:12s} !! meta rewrite FAILED ({exc!r})")

    print(f"\n[summary] {tot_bad} full-frame ref(s) across {tot_folders} folder(s); "
          f"{tot_refs} refs scanned.")
    if skipped_folders:
        print(f"[summary] {len(skipped_folders)} folder(s) skipped (would be emptied): "
              f"{', '.join(skipped_folders)}")
    if not args.apply:
        print("[summary] DRY RUN — re-run with --apply to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
