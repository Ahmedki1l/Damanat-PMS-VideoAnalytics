"""Exercise the gallery write path directly and surface any swallowed error.

The live seed (`_seed_plate_gallery_reference`) swallows exceptions at debug
level, so a failing `save_ref` leaves no trace and no folder. This reproduces
the exact write with a synthetic crop + vector, printing the resolved absolute
root, the enabled flag, and the FULL traceback if it fails.

Run on the RUN MACHINE, from the project root with the venv active:

    .venv/Scripts/python.exe tools/test_gallery_write.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import load_config


def main() -> None:
    cfg = load_config("config.yaml")
    m = cfg.matching
    base = getattr(cfg.output, "snapshot_base_dir", "vehicle_images") or "vehicle_images"

    print(f"gallery_persist_enabled = {getattr(m, 'gallery_persist_enabled', None)}")
    print(f"snapshot_base_dir       = {base}")
    root = os.path.abspath(os.path.join(base, "gallery"))
    print(f"gallery root (abs)      = {root}")
    print(f"base dir exists         = {os.path.isdir(base)}")
    print(f"base dir writable       = {os.access(base if os.path.isdir(base) else '.', os.W_OK)}")

    from src.vehicle_registry.gallery_store import VehicleGalleryStore

    model_dir = (getattr(m, "reid_openvino_model_dir", "") or "").rstrip("/\\")
    tag = f"test:{os.path.basename(model_dir) or 'default'}"
    store = VehicleGalleryStore(base, tag, getattr(m, "gallery_max_refs_per_car", 10))

    # Synthetic but valid BGR crop + L2-normed vector, mirroring a real seed.
    crop = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    vec = np.random.rand(512).astype(np.float32)
    vec /= np.linalg.norm(vec)

    plate = "TEST-0000"
    print(f"\nAttempting save_ref(plate={plate!r}, gate_only=True) ...")
    try:
        fn = store.save_ref(
            plate, crop, vec, quality=998.0, camera_id="CAM-03", gate_only=True
        )
        print(f"  save_ref returned: {fn!r}")
    except Exception:
        print("  save_ref RAISED:")
        traceback.print_exc()
        return

    plate_dir = os.path.join(root, "TEST-0000")
    if os.path.isdir(plate_dir):
        print(f"\n[OK] folder created: {plate_dir}")
        print(f"     contents: {os.listdir(plate_dir)}")
        print("\nThe write path WORKS. If real seeds still make no folder, the")
        print("failure is upstream: crop/feature is None at seed time, or the")
        print("confirm path isn't reached. Set logging to DEBUG and watch for")
        print("'[gallery] seed for ... failed' / '[gallery] save_ref failed'.")
    else:
        print(f"\n[!] save_ref returned but folder is ABSENT: {plate_dir}")
        print("    cv2.imwrite likely returned False (check disk/codec/path).")


if __name__ == "__main__":
    main()