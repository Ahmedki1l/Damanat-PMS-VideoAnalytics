"""capture_finetune_frames.py — record FULL FRAMES off a live camera at a fixed
rate, as a fine-tuning dataset.

Built for CAM-00 (10.1.13.95, "GF-FRONT"), the near-nadir FISHEYE roof camera.
That view is the reason this tool exists: config.yaml's camera_overrides note
that the global yolo11m int8@320 returns ZERO vehicles on it at any confidence,
because COCO has almost no top-down fisheye car prior. yolo11l recovers them
but costs a dedicated model load. Fine-tuning on real frames from THIS mount is
the way out, and that needs frames.

Full view, deliberately: no crop, no dewarp, no CLAHE. CAM-00 has preprocessing
disabled in config.yaml anyway (CLAHE erases the dark car in G3 outright on this
sunlit rooftop), and a training set must show the frame the detector will
actually be handed. Main stream (channel 101) for full resolution, not the 720p
sub-stream.

THE ONE THING THAT WOULD HAVE GONE WRONG
----------------------------------------
The obvious way to get 2 fps is read → sleep(0.5) → read. Do not. Per
tools/probe_camera_lag.py, OpenCV's FFmpeg backend routinely ignores
CAP_PROP_BUFFERSIZE=1, so sleeping between reads lets a backlog build inside
FFmpeg and every frame you pull is progressively staler — minutes behind
reality, while cap.read() still returns instantly and everything LOOKS fine.
You would end up with an hour of filenames spanning an hour and pixels spanning
five minutes.

So this reads the stream CONTINUOUSLY, exactly like camera_manager.py's grabber
thread, and only WRITES one frame per interval. Decode cost is the price of
frames that are actually current.

Usage
-----
    # 2 fps from CAM-00 until Ctrl+C (credentials resolved automatically)
    python tools/capture_finetune_frames.py

    # a bounded 20-minute run into a named dataset dir
    python tools/capture_finetune_frames.py --duration 1200 --out data/finetune/cam00

    # a different camera, explicit password
    python tools/capture_finetune_frames.py --camera CAM-01 --password '...'

    # skip near-identical frames (a rooftop at 2fps is mostly the same picture)
    python tools/capture_finetune_frames.py --min-diff 1.5

Credentials are never stored here. They resolve in this order:
    --password  ->  $CAM_PASSWORD  ->  the Gateway's encrypted cameras table
                ->  VideoAnalytics config.yaml
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    # This laptop has FIVE python.exe on PATH (3.14, 3.12, miniconda, a
    # WindowsApps stub, ...) and `python` resolves to a different one depending
    # on the shell. A bare ModuleNotFoundError sends you hunting; naming the
    # interpreter that is actually running turns it into one copy-paste.
    sys.exit(
        f"\n[cap] missing dependency: {exc.name}\n"
        f"[cap] running interpreter: {sys.executable}\n"
        f"[cap] python version:      {sys.version.split()[0]}\n\n"
        f"[cap] install it for THIS interpreter:\n"
        f'      "{sys.executable}" -m pip install opencv-python-headless numpy\n\n'
        f"[cap] or run the tool with an interpreter that already has it.\n"
    )

# This repo writes em-dashes and arrows in prose, and a Windows console defaults
# to cp1252 — which cannot encode them, so even `--help` dies with a
# UnicodeEncodeError before argparse prints a word. Force UTF-8 on the streams
# we own rather than flattening the text.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

REPO = Path(__file__).resolve().parent.parent
GATEWAY = REPO.parent / "API Gateway"

DEFAULT_CAMERA = "CAM-00"
# Documented in config.yaml's camera_overrides and the gateway cameras row.
FALLBACK = {"ip": "10.1.13.95", "port": 554, "user": "kloudspot"}


# ── credential / endpoint resolution ────────────────────────────────────────

def _read_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser. Deliberately NOT python-dotenv: this has to
    work from whichever venv the operator happens to be in, and dotenv is not
    installed in all of them."""
    values: dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    except OSError:
        return {}
    return values


def _from_gateway_db(camera_id: str) -> dict | None:
    """Read the camera row from the gateway's `cameras` table and decrypt the
    password with the gateway's Fernet key.

    Preferred source: it is the same record VideoAnalytics itself fetches, so a
    rotated password is picked up with no edit here.

    This talks to the DB DIRECTLY rather than importing the gateway app. The
    import path needed the gateway's whole web stack — importing app.config
    pulls pydantic_settings, which is absent from the VideoAnalytics venv, so
    the lookup died with a bare ModuleNotFoundError on the one interpreter that
    HAS OpenCV. sqlalchemy + pyodbc + cryptography are all this actually needs,
    and those are present in both venvs.
    """
    env = _read_env_file(GATEWAY / ".env")
    key = env.get("CAMERAS_ENCRYPTION_KEY")
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
        from sqlalchemy import create_engine, text
    except ModuleNotFoundError:
        return None

    driver = (env.get("DB_DRIVER") or "ODBC Driver 17 for SQL Server").replace(" ", "+")
    host = env.get("DB_SERVER", "localhost")
    port = env.get("DB_PORT", "1433")
    name = env.get("DB_NAME", "damanat_pms")
    if (env.get("DB_TRUSTED_CONNECTION", "")).lower() in ("true", "1", "yes"):
        # The empty @ is required: without it pyodbc 5+ attempts SQL auth with a
        # blank user and fails with "Login failed for user ''".
        url = (f"mssql+pyodbc://@{host}:{port}/{name}?driver={driver}"
               f"&Trusted_Connection=Yes&TrustServerCertificate=Yes")
    else:
        url = (f"mssql+pyodbc://{env.get('DB_USER','')}:{env.get('DB_PASSWORD','')}"
               f"@{host}:{port}/{name}?driver={driver}&TrustServerCertificate=Yes")

    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT ip_address, rtsp_port, rtsp_path, username, password_encrypted
                FROM cameras WHERE camera_id = :cid
            """), {"cid": camera_id}).mappings().first()
        if not row:
            return None
        secret = row.get("password_encrypted")
        return {
            "ip": row.get("ip_address"),
            "port": int(row.get("rtsp_port") or 554),
            "path": row.get("rtsp_path"),
            "user": row.get("username"),
            "password": Fernet(key.encode()).decrypt(secret.encode()).decode()
            if secret else None,
        }
    except Exception as exc:
        print(f"[cap] gateway camera lookup failed ({type(exc).__name__}: {exc}); "
              f"trying config.yaml")
        return None


def _from_va_config(camera_id: str) -> dict | None:
    """Fall back to VideoAnalytics' own config.yaml camera list."""
    try:
        import yaml
        with open(REPO / "config.yaml", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return None
    for cam in cfg.get("cameras") or []:
        if cam.get("id") == camera_id:
            return {
                "ip": cam.get("ip"),
                "port": int(cam.get("rtsp_port") or 554),
                "path": None,
                "user": cam.get("user"),
                "password": cam.get("password"),
            }
    return None


def resolve_endpoint(camera_id: str, args) -> tuple[str, str]:
    """Return (rtsp_url, redacted_url_for_logging)."""
    info = _from_gateway_db(camera_id) or _from_va_config(camera_id) or {}
    ip = args.ip or info.get("ip") or FALLBACK["ip"]
    port = int(args.port or info.get("port") or FALLBACK["port"])
    user = args.user or info.get("user") or FALLBACK["user"]
    password = args.password or os.environ.get("CAM_PASSWORD") or info.get("password")

    if args.channel:
        path = f"/Streaming/Channels/{args.channel}"
    else:
        path = info.get("path") or "/Streaming/Channels/101"

    if not password:
        raise SystemExit(
            f"[cap] no password for {camera_id}. Pass --password, set CAM_PASSWORD, "
            f"or make the gateway's cameras table reachable."
        )

    auth = f"{quote(user, safe='')}:{quote(password, safe='')}@" if user else ""
    shown = f"{quote(user, safe='')}:***@" if user else ""
    return (
        f"rtsp://{auth}{ip}:{port}{path}",
        f"rtsp://{shown}{ip}:{port}{path}",
    )


# ── capture ─────────────────────────────────────────────────────────────────

def open_stream(url: str, redacted: str) -> cv2.VideoCapture:
    # Same FFMPEG options the production grabber uses. `threads;2` keeps FFmpeg
    # from sizing its decoder pool off every visible core (camera_manager.py).
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|threads;2"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # honoured or not, ask anyway
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
    if not cap.isOpened():
        raise ConnectionError(f"could not open {redacted}")
    return cap


def frame_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two frames, 0-255. Cheap: computed on a
    small grayscale thumbnail, not the full 4K frame."""
    small = lambda f: cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(small(a), small(b))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default=DEFAULT_CAMERA, help="camera id (default CAM-00)")
    ap.add_argument("--fps", type=float, default=2.0, help="frames SAVED per second (default 2)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to capture; 0 = until Ctrl+C (default)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 = unlimited")
    ap.add_argument("--out", default="", help="output dir (default data/finetune/<camera>/<timestamp>)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default 95)")
    ap.add_argument("--min-diff", type=float, default=0.0,
                    help="skip a frame unless it differs from the last SAVED one by "
                         "this mean-abs-diff (0-255). 0 = save everything (default)")
    ap.add_argument("--channel", type=int, default=0,
                    help="101 main / 102 sub. Default: the camera's configured path (101)")
    ap.add_argument("--ip"); ap.add_argument("--port", type=int)
    ap.add_argument("--user"); ap.add_argument("--password")
    args = ap.parse_args()

    if args.fps <= 0:
        raise SystemExit("[cap] --fps must be > 0")

    url, redacted = resolve_endpoint(args.camera, args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else REPO / "data" / "finetune" / args.camera / stamp
    out.mkdir(parents=True, exist_ok=True)

    interval = 1.0 / args.fps
    print(f"[cap] {args.camera} <- {redacted}")
    print(f"[cap] saving {args.fps:g} fps -> {out}")
    print(f"[cap] {'until Ctrl+C' if not args.duration else f'for {args.duration:g}s'}"
          f"{f', max {args.max_frames} frames' if args.max_frames else ''}")

    stopping = {"now": False}

    def _stop(signum, frame):
        del signum, frame
        stopping["now"] = True
        print("\n[cap] stopping — finishing the current frame")

    signal.signal(signal.SIGINT, _stop)

    cap = open_stream(url, redacted)
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, args.quality))]

    saved = 0
    skipped = 0
    reconnects = 0
    read_fail = 0
    last_saved: np.ndarray | None = None
    started = time.perf_counter()
    next_save = started
    last_report = started

    try:
        while not stopping["now"]:
            if args.duration and time.perf_counter() - started >= args.duration:
                break
            if args.max_frames and saved >= args.max_frames:
                break

            # CONTINUOUS read — never sleep between reads. See the module
            # docstring: sleeping here is what lets FFmpeg queue stale frames.
            ok, frame = cap.read()
            if not ok or frame is None:
                read_fail += 1
                if read_fail >= 30:
                    reconnects += 1
                    print(f"[cap] stream dropped; reconnecting (#{reconnects})")
                    cap.release()
                    time.sleep(min(2.0 * reconnects, 15.0))
                    try:
                        cap = open_stream(url, redacted)
                        read_fail = 0
                    except ConnectionError as exc:
                        print(f"[cap] reconnect failed: {exc}")
                continue
            read_fail = 0

            now = time.perf_counter()
            if now < next_save:
                continue  # drained, deliberately not saved

            if args.min_diff > 0 and last_saved is not None:
                if frame_delta(frame, last_saved) < args.min_diff:
                    skipped += 1
                    next_save += interval
                    continue

            wall = datetime.now()
            name = f"{args.camera}_{wall:%Y%m%d_%H%M%S}_{wall.microsecond // 1000:03d}.jpg"
            if cv2.imwrite(str(out / name), frame, encode):
                saved += 1
                last_saved = frame
            else:
                print(f"[cap] WARNING: failed to write {name}")

            # Advance on a fixed grid, and re-base if we have fallen behind by
            # more than one interval so a slow disk cannot make the rate creep.
            next_save += interval
            if next_save < now:
                next_save = now + interval

            if now - last_report >= 10.0:
                elapsed = now - started
                print(f"[cap] {saved} frames in {elapsed:.0f}s "
                      f"({saved / elapsed:.2f}/s effective)"
                      f"{f', {skipped} skipped as near-identical' if skipped else ''}"
                      f" | {frame.shape[1]}x{frame.shape[0]}")
                last_report = now
    finally:
        cap.release()

    elapsed = max(time.perf_counter() - started, 1e-6)
    print(f"\n[cap] done: {saved} frames -> {out}")
    print(f"[cap] {elapsed:.0f}s elapsed, {saved / elapsed:.2f} fps effective"
          f"{f', {skipped} skipped' if skipped else ''}"
          f"{f', {reconnects} reconnect(s)' if reconnects else ''}")
    if saved == 0:
        print("[cap] NOTHING CAPTURED — check the camera is reachable from this laptop")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
