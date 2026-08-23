"""
tools/dedup_frames.py — Drop near-duplicate frames before auto-labelling.

WHY THIS EXISTS
---------------
A fixed-interval snapshot dump off a static camera is almost entirely
redundant. Measured on the SPECTECH CAM-03 dump (7221 frames, 5-second
interval, 09:42-19:43 on 2026-08-17):

    5845 / 7220 consecutive pairs are dhash-IDENTICAL (81%)
    7125 / 7220 are within 2 bits of 64                 (99%)
    distinct scenes at dhash>3:  88   (1.2% of the frames)

So those 7221 files carry roughly 100-200 frames of actual information. Feeding
all of them to `finetune_yolo_facility.py` does three bad things:

  1. burns ~40x the teacher-labelling GPU time for no new supervision;
  2. overfits the student to one parked arrangement — the same white sedan in
     the same bay is 40% of the "dataset";
  3. POISONS THE VAL SPLIT. `finetune_yolo_facility.py` splits by random
     `--val-frac`, so near-identical twins land in train AND val. Val recall
     then reads ~0.99 and means nothing, because the model is being tested on
     frames it memorised. This is the failure that makes a bad fine-tune look
     finished.

(3) is the reason to run this BEFORE the teacher, not after.

WHAT IT DOES
------------
Greedy sequential dedup on a perceptual hash: keep frame 0, then keep the next
frame whose hash differs from the last KEPT frame by more than --threshold bits.
Sequential (not all-pairs) because these are time-ordered and the thing we are
removing is temporal stasis.

Hash size matters. A 64-bit (8x8) dhash is blind to a single distant car in a
1280x720 frame. Default here is 33x32 -> 1024 bits, which does see it; the
tradeoff is that JPEG/sensor noise on a genuinely unchanging scene sits around
11/1024 bits, so the threshold has to clear that noise floor. Defaults below are
calibrated on CAM-03; --report prints the distance distribution so you can
re-check them on a camera that looks different.

Usage
-----
    # look before you copy — prints the distribution and what each threshold keeps
    python tools/dedup_frames.py --src "C:/.../SPECTECH/CAM-03" --report

    # one camera -> a deduped pool
    python tools/dedup_frames.py \
        --src "C:/.../SPECTECH/CAM-03" --out data/detector_pool --threshold 40

    # all 26 cameras into one pooled training set (keeps camera in the filename)
    python tools/dedup_frames.py \
        --src "C:/.../SPECTECH/*" --out data/detector_pool --threshold 40

    # STRAIGHT FROM THE TARS — never lands the 1.6 GB folder on disk.
    # This is the mode to use when the full dump is 50 GB and your laptop isn't.
    python tools/dedup_frames.py \
        --tar "C:/.../SPECTECH/*.tar" --out data/detector_pool --threshold 40

    python tools/dedup_frames.py --selftest

NOTE ON TAR ORDER
-----------------
The SPECTECH tars are NOT stored in chronological order (verified on CAM-03.tar:
member 0 is 17:12:23, member 1 is 16:52:08, member 2 is 15:13:58). Sequential
dedup on that order would compare 17:12 against 16:52 and call two unrelated
scenes "a change", keeping far too much. So --tar hashes every member in
whatever order the archive gives, then SORTS BY FILENAME before the greedy pass.
That costs a second read of the archive to extract the keepers, which is still
far cheaper than landing the whole thing.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

HASH_W = 33  # -> (HASH_W-1) * HASH_H bits
HASH_H = 32
IMG_EXT = (".jpg", ".jpeg", ".png")


def dhash_bits(path: Path, w: int = HASH_W, h: int = HASH_H) -> np.ndarray:
    """Horizontal-gradient perceptual hash, as a flat int8 bit vector."""
    im = Image.open(path).convert("L").resize((w, h), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.int8)


def hash_folder(files: List[Path]) -> np.ndarray:
    rows = []
    for i, f in enumerate(files):
        if i % 500 == 0 and i:
            print(f"    hashed {i}/{len(files)}", file=sys.stderr)
        rows.append(dhash_bits(f))
    return np.array(rows)


def greedy_keep(U: np.ndarray, threshold: int) -> List[int]:
    """Keep a frame when it differs from the last KEPT frame by > threshold bits."""
    if len(U) == 0:
        return []
    keep = [0]
    last = U[0]
    for i in range(1, len(U)):
        if int(np.abs(U[i] - last).sum()) > threshold:
            keep.append(i)
            last = U[i]
    return keep


def report(U: np.ndarray, nbits: int) -> None:
    cons = np.abs(np.diff(U, axis=0)).sum(1)
    print(f"  consecutive distance over {nbits} bits:")
    for p in (50, 75, 90, 95, 99):
        print(f"    p{p:<3d} {np.percentile(cons, p):6.0f}")
    print(f"    NOTE: the p50 above is your noise floor - a threshold at or")
    print(f"          below it keeps pure JPEG noise as 'new scenes'.")
    print("  frames kept by threshold:")
    for thr in (10, 20, 40, 80, 160):
        k = len(greedy_keep(U, thr))
        print(f"    >{thr:<4d} {k:6d}  ({100.0 * k / len(U):5.1f}%)")


def dhash_bits_buf(buf: bytes, w: int = HASH_W, h: int = HASH_H) -> np.ndarray:
    """Same hash as dhash_bits, from an in-memory JPEG (no temp file)."""
    im = Image.open(io.BytesIO(buf)).convert("L").resize((w, h), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.int8)


def dedup_tar(tar_path: Path, out: Optional[Path], threshold: int,
              do_report: bool) -> Tuple[int, int]:
    """Two-pass dedup straight out of an archive. Returns (n_members, n_kept).

    Pass 1 hashes every member; the results are sorted by filename (see the
    NOTE at the top — archive order is not chronological) and run through the
    greedy filter. Pass 2 re-opens the archive and extracts only the winners,
    so peak disk usage is the OUTPUT size, not the archive's contents.
    """
    names: List[str] = []
    rows: List[np.ndarray] = []
    with tarfile.open(tar_path, "r|*") as t:   # streaming mode, no seeking
        for m in t:
            if not m.isfile() or Path(m.name).suffix.lower() not in IMG_EXT:
                continue
            f = t.extractfile(m)
            if f is None:
                continue
            names.append(m.name)
            rows.append(dhash_bits_buf(f.read()))
            if len(names) % 500 == 0:
                print(f"    hashed {len(names)}", file=sys.stderr)

    if not names:
        return 0, 0

    order = sorted(range(len(names)), key=lambda i: Path(names[i]).name)
    U = np.array([rows[i] for i in order])

    if do_report:
        report(U, (HASH_W - 1) * HASH_H)
        return len(names), 0

    keep_local = greedy_keep(U, threshold)
    wanted = {names[order[i]] for i in keep_local}
    print(f"  keeping {len(wanted)} / {len(names)} "
          f"({100.0 * len(wanted) / len(names):.1f}%)")

    out.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(tar_path, "r|*") as t:   # pass 2: extract the winners
        for m in t:
            if m.name in wanted:
                f = t.extractfile(m)
                if f is None:
                    continue
                (out / Path(m.name).name).write_bytes(f.read())
                written += 1
    return len(names), written


def collect(src_globs: List[str]) -> List[Tuple[str, List[Path]]]:
    """Expand globs into (group_name, sorted files). One group per directory."""
    groups: List[Tuple[str, List[Path]]] = []
    for pattern in src_globs:
        for d in sorted(glob.glob(pattern)):
            p = Path(d)
            if not p.is_dir():
                continue
            files = sorted(
                q for q in p.iterdir()
                if q.suffix.lower() in IMG_EXT
            )
            if files:
                groups.append((p.name, files))
    return groups


def selftest() -> int:
    U = np.zeros((10, 100), dtype=np.int8)
    assert greedy_keep(U, 5) == [0], "identical frames must collapse to one"
    U2 = U.copy()
    U2[5:, :50] = 1  # a hard change at index 5
    assert greedy_keep(U2, 5) == [0, 5], "a real change must be kept"
    U3 = U.copy()
    U3[3:, :3] = 1  # a change smaller than the threshold
    assert greedy_keep(U3, 5) == [0], "sub-threshold drift must be dropped"
    assert len(dhash_bits.__doc__ or "") > 0
    print("selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", default=[],
                    help="Directory or glob of directories. Repeatable.")
    ap.add_argument("--tar", action="append", default=[],
                    help="Tar archive or glob of archives, deduped WITHOUT "
                         "extracting them first. Repeatable. Use this when the "
                         "full dump does not fit on disk.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Destination pool. Omit with --report to only measure.")
    ap.add_argument("--threshold", type=int, default=40,
                    help="Bits of %d-bit hash that must change. Default 40 "
                         "(calibrated on CAM-03; noise floor is ~11)."
                         % ((HASH_W - 1) * HASH_H))
    ap.add_argument("--report", action="store_true",
                    help="Print the distance distribution and exit without copying.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Write kept/dropped counts per camera as JSON.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.src and not args.tar:
        ap.error("one of --src or --tar is required")
    if not args.report and args.out is None:
        ap.error("--out is required unless --report")

    nbits = (HASH_W - 1) * HASH_H
    summary = {}
    total_in = total_out = 0

    tars = [Path(p) for pattern in args.tar for p in sorted(glob.glob(pattern))]
    for tp in tars:
        print(f"[{tp.name}] streaming from archive")
        n_in, n_out = dedup_tar(tp, args.out, args.threshold, args.report)
        total_in += n_in
        total_out += n_out
        summary[tp.name] = {"frames": n_in, "kept": n_out}

    groups = collect(args.src)
    if not groups and not tars:
        print("nothing matched --src / --tar", file=sys.stderr)
        return 1

    for name, files in groups:
        print(f"[{name}] {len(files)} frames")
        U = hash_folder(files)
        if args.report:
            report(U, nbits)
            summary[name] = {"frames": len(files)}
            continue

        keep = greedy_keep(U, args.threshold)
        total_in += len(files)
        total_out += len(keep)
        print(f"  keeping {len(keep)} / {len(files)} "
              f"({100.0 * len(keep) / len(files):.1f}%)")

        args.out.mkdir(parents=True, exist_ok=True)
        for i in keep:
            src = files[i]
            # Source filenames already carry the camera; keep them as-is so
            # provenance survives pooling all 26 cameras into one directory.
            shutil.copy2(src, args.out / src.name)
        summary[name] = {"frames": len(files), "kept": len(keep)}

    if not args.report:
        print(f"\nTOTAL {total_out} / {total_in} frames kept "
              f"({100.0 * total_out / max(total_in, 1):.1f}%) -> {args.out}")
    if args.manifest:
        args.manifest.write_text(json.dumps(summary, indent=2))
        print(f"manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
