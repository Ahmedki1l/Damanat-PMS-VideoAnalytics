"""
tools/live_camera_test.py — Run several detector models against a LIVE camera and
report which ones see the vehicle and whether the detection is STABLE frame to
frame (does it drop on the next frame?). Run this on the box that can reach the
cameras + DB (the production server), not a laptop.

It grabs N consecutive frames from one camera's RTSP stream and, for each model,
runs detection at a production-like confidence, printing a per-frame hit pattern
(1 = >=1 vehicle detected, 0 = none), the hit rate, and the longest run of
consecutive misses — which is exactly "will it drop it next frame".

Camera lookup: by default it reads the DB `cameras` table (same DATABASE_URL as
config.yaml) to build the RTSP URL. Or pass --rtsp to skip the DB entirely.

Usage
-----
    # All candidate models on CAM-23, 60 frames, main stream:
    python tools/live_camera_test.py --camera CAM-23 --frames 60 --channel 101 \
        --model "deployed 11m=models/yolo11m_320_int8_openvino_model" \
        --model "26n@320=models/yolo26n_ft_320_int8_openvino_model" \
        --model "26n@416=models/yolo26n_ft_416_int8_openvino_model@416" \
        --model "stock 11m=yolo11m.pt" --model "stock 26n=yolo26n.pt"

    # Explicit RTSP (no DB):
    python tools/live_camera_test.py --rtsp "rtsp://user:pass@10.1.13.x:554/Streaming/Channels/101" ...

Notes
-----
* Stock COCO models predict class car/bus/truck (2,5,7); fine-tuned single-class
  models predict vehicle (0). The tool auto-detects from model.names and counts
  vehicles either way, so detection counts are comparable.
* --conf defaults to 0.25 (near production 0.35); lower it to probe whether the
  car is seen weakly vs not at all.
* --save-dir writes annotated frames so you can eyeball what each model boxed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

VEHICLE_COCO = [2, 5, 7]


def ensure_cuda_nms() -> None:
    try:
        import torch, torchvision
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        b = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda")
        torchvision.ops.nms(b, torch.tensor([0.5], device="cuda"), 0.5)
        return
    except NotImplementedError:
        _orig = torchvision.ops.nms
        torchvision.ops.nms = lambda x, s, t: _orig(x.detach().cpu(), s.detach().cpu(), t).to(x.device)


def rtsp_from_db(camera_id: str, channel: int) -> Optional[str]:
    """Look up the camera in the DB `cameras` table and build its RTSP URL."""
    import yaml
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))

    def find(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "DATABASE_URL":
                    return v
                r = find(v)
                if r:
                    return r
        return None

    url = find(cfg)
    from sqlalchemy import create_engine, text
    eng = create_engine(url)
    with eng.connect() as c:
        rows = [dict(r._mapping) for r in c.execute(text("SELECT * FROM cameras"))]
    if not rows:
        return None
    print(f"[db] {len(rows)} cameras; columns: {list(rows[0].keys())}")

    def g(d, *names):
        for n in names:
            for k, v in d.items():
                if k.lower() == n:
                    return v
        return None

    target = None
    for d in rows:
        idv = str(g(d, "camera_id", "id", "name", "cam_id") or "")
        if camera_id.lower() in idv.lower() or idv.lower() in camera_id.lower():
            target = d
            break
    if target is None:
        print(f"[db] {camera_id} not found. Camera ids: "
              f"{[str(g(d,'camera_id','id','name','cam_id')) for d in rows]}")
        return None
    ip = g(target, "ip", "ip_address", "host")
    user = g(target, "username", "user")
    pwd = g(target, "password", "pass")
    port = g(target, "rtsp_port", "port") or 554
    print(f"[db] {camera_id} -> ip={ip} user={user} port={port}")
    return f"rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{channel}"


def vehicle_classes(model) -> List[int]:
    names = getattr(model, "names", {}) or {}
    if len(names) == 1:
        return [0]
    present = [c for c in VEHICLE_COCO if c in names]
    return present or [0]


def parse_model_spec(spec: str) -> Tuple[str, str, Optional[int]]:
    imgsz = None
    if "@" in spec.rsplit("=", 1)[-1]:
        head, sz = spec.rsplit("@", 1)
        if sz.isdigit():
            spec, imgsz = head, int(sz)
    label, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
    return label, path, imgsz


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="live_camera_test",
                                description="Test detector models on a live camera.")
    p.add_argument("--camera", default="CAM-23", help="Camera id to look up in the DB.")
    p.add_argument("--rtsp", default=None, help="Explicit RTSP URL (skips DB lookup).")
    p.add_argument("--channel", type=int, default=101, help="RTSP channel (101 main, 102 sub).")
    p.add_argument("--model", action="append", default=[], dest="models",
                   help="label=path[@imgsz]. Repeatable.")
    p.add_argument("--frames", type=int, default=60, help="Consecutive frames to test.")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence.")
    p.add_argument("--device", default="cpu", help="cpu, or 0 for GPU (.pt only).")
    p.add_argument("--save-dir", default=None, help="Write annotated frames here.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    args = parse_args(argv)
    ensure_cuda_nms()

    url = args.rtsp or rtsp_from_db(args.camera, args.channel)
    if not url:
        print("No RTSP URL (DB lookup failed and no --rtsp given).")
        return 2
    safe = url.split("@")[-1]
    print(f"[stream] connecting to {args.camera}: ...@{safe}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("[stream] FAILED to open. Check network/creds/channel.")
        return 3

    frames = []
    for _ in range(args.frames):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        print("[stream] no frames read.")
        return 3
    h, w = frames[0].shape[:2]
    print(f"[stream] captured {len(frames)} frames at {w}x{h}\n")

    if not args.models:
        print("No --model given.")
        return 2

    save_root = Path(args.save_dir) if args.save_dir else None
    print(f"{'model':22} {'hits':>7} {'rate':>6} {'maxmiss':>8}  per-frame pattern (1=seen 0=miss)")
    print("-" * 100)
    for spec in args.models:
        label, path, imgsz = parse_model_spec(spec)
        try:
            model = YOLO(path, task="detect")
            keep = vehicle_classes(model)
            pattern, counts = [], []
            for i, fr in enumerate(frames):
                kw = dict(conf=args.conf, classes=keep, device=args.device, verbose=False)
                if imgsz:
                    kw["imgsz"] = imgsz
                res = model.predict(fr, **kw)[0]
                n = len(res.boxes) if res.boxes is not None else 0
                counts.append(n)
                pattern.append(1 if n > 0 else 0)
                if save_root:
                    d = save_root / label.replace(" ", "_")
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{i:03d}.jpg"), res.plot())
            hits = sum(pattern)
            rate = hits / len(pattern)
            # longest consecutive-miss run = "will it drop next frame"
            maxmiss = cur = 0
            for v in pattern:
                cur = cur + 1 if v == 0 else 0
                maxmiss = max(maxmiss, cur)
            pat = "".join(str(v) for v in pattern)
            print(f"{label:22.22} {hits:3}/{len(pattern):<3} {rate:6.2f} {maxmiss:8}  {pat}")
        except Exception as exc:  # noqa: BLE001
            print(f"{label:22.22} ERROR: {exc!r}")

    print("\nrate = fraction of frames with >=1 vehicle. maxmiss = longest run of "
          "consecutive MISSED frames (0s) — high = it drops the car and won't "
          "recover for that many frames. A model that can't see CAM-23 shows all 0s.")
    if save_root:
        print(f"Annotated frames in {save_root}/<model>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
