"""Measure where the plate detector actually looks, so the overlay guard can
be configured from data instead of a guess.

Hikvision composites its own plate/OSD panel into the corner of a frame. That
panel is sharp, high-contrast and perfectly rectangular; a real plate twenty
metres away is none of those things. So the detector's TOP-scoring box on a
HikCentral image is often the panel, and OCR on it reads Hikvision's own answer
back to us — an echo, not independent verification.

ENTRY_V2_OVERLAY_EXCLUDE_REGIONS stops that, and it is empty by default because
a guessed rectangle rejects real plates, which is the worse failure. This tool
turns real images into a real number.

    # on PMS-AI, in the pod
    python scripts/setup/probe_hik_images.py --hours 6 --limit 20 --out ./hik-samples
    # then here
    python tools/measure_overlay_regions.py ./hik-samples

WHAT IT PRINTS. Every detected box as a NORMALISED centre (0..1), so geometry is
independent of frame size, plus a clustering of where the top-scoring boxes
land. A tight cluster in one corner across many different cars is the panel:
real plates move around the frame with the car, a composited overlay does not.

WHAT IT WILL NOT DO. Emit a region for you to paste in unread. It proposes one
and shows the evidence; you decide, because getting this wrong silently costs
real entries.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--model-dir", default="models/yolo11n_lpd_openvino_model",
        help="LPD OpenVINO model directory",
    )
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument(
        "--grid", type=int, default=5,
        help="cells per axis when clustering box centres (default 5)",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 2

    images = sorted(
        p for p in args.directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"no images under {args.directory}", file=sys.stderr)
        return 2

    import cv2
    from src.ocr.plate_region_detector import OpenVINOPlateRegionDetector

    detector = OpenVINOPlateRegionDetector(
        args.model_dir, confidence=args.confidence
    )

    print("=" * 74)
    print(f"OVERLAY GEOMETRY   {len(images)} image(s)")
    print("=" * 74)

    top_cells: Counter = Counter()
    all_cells: Counter = Counter()
    rows = []
    no_box = 0

    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        boxes = detector.detect(frame)
        if not boxes:
            no_box += 1
            rows.append((path.name, width, height, None, None, None))
            continue
        for rank, (x1, y1, x2, y2, score) in enumerate(boxes):
            cx = (x1 + x2) / 2.0 / width
            cy = (y1 + y2) / 2.0 / height
            cell = (
                min(args.grid - 1, int(cx * args.grid)),
                min(args.grid - 1, int(cy * args.grid)),
            )
            all_cells[cell] += 1
            if rank == 0:
                top_cells[cell] += 1
                rows.append((path.name, width, height, cx, cy, score))

    print("\nPER IMAGE  (top-scoring box, normalised centre)")
    for name, width, height, cx, cy, score in rows[:40]:
        if cx is None:
            print(f"  {name[:38]:<38} {width}x{height}   no box")
        else:
            print(
                f"  {name[:38]:<38} {width}x{height}   "
                f"centre=({cx:.3f}, {cy:.3f})  score={score:.2f}"
            )
    if len(rows) > 40:
        print(f"  ... {len(rows) - 40} more")
    if no_box:
        print(f"\n  {no_box} image(s) produced no box at all.")

    print(f"\nWHERE TOP-SCORING BOXES LAND  ({args.grid}x{args.grid} grid)")
    _grid(top_cells, args.grid)

    if not top_cells:
        print("\nNothing detected. Nothing to conclude.")
        return 1

    (gx, gy), count = top_cells.most_common(1)[0]
    share = count / sum(top_cells.values())
    step = 1.0 / args.grid
    region = (gx * step, gy * step, (gx + 1) * step, (gy + 1) * step)

    print("\nREADING IT")
    print(
        f"  The busiest cell holds {count}/{sum(top_cells.values())} "
        f"top-scoring boxes ({share:.0%})."
    )
    if share < 0.5:
        print(
            "  That is NOT a tight cluster. Top boxes are spread across the\n"
            "  frame, which is what real plates do - they move with the car.\n"
            "  There is no evidence of a composited panel dominating here, so\n"
            "  LEAVE ENTRY_V2_OVERLAY_EXCLUDE_REGIONS EMPTY. Configuring a\n"
            "  region on this evidence would reject real plates."
        )
        return 0

    print(
        f"  {share:.0%} of top boxes land in one cell across {len(rows)} different\n"
        "  cars. Real plates do not do that; a composited overlay does. Open a\n"
        "  few of these images and confirm by eye that the panel is there."
    )
    print("\n  If confirmed, a starting region for that cell is:")
    print(
        "      ENTRY_V2_OVERLAY_EXCLUDE_REGIONS="
        f"{region[0]:.3g},{region[1]:.3g},{region[2]:.3g},{region[3]:.3g}"
    )
    print(
        "\n  Tighten it to the panel rather than the whole cell. The guard\n"
        "  rejects a box whose CENTRE falls inside, and falls through to the\n"
        "  next candidate - so an over-wide region silently costs real plates."
    )
    return 0


def _grid(cells: Counter, size: int) -> None:
    total = sum(cells.values()) or 1
    print("       " + "".join(f"  x{i}  " for i in range(size)))
    for y in range(size):
        line = f"   y{y} "
        for x in range(size):
            count = cells.get((x, y), 0)
            line += f" {count:>3}  " if count else "   .  "
        print(line)
    print(f"       (counts of top-scoring boxes; {total} total)")


if __name__ == "__main__":
    raise SystemExit(main())
