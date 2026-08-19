"""capture_all_cameras.py — record FULL FRAMES off EVERY camera at once, at a
fixed rate, as a fine-tuning dataset. Self-contained: drop this one file into
the pod and run it.

This is the fleet-wide sibling of tools/capture_finetune_frames.py (single
camera, same capture discipline). Everything it needs about a camera — IP,
port, RTSP path, username, Fernet-encrypted password — comes from the `cameras`
table, so the list is whatever the DB says it is; nothing is hardcoded.

Main stream by design: the table's rtsp_path is /Streaming/Channels/101, which
on this fleet negotiates 1280x720 — there is no 4K stream to switch to (see
config.yaml's stream_channel note and the boot log's "connected [1280x720]").
That is the exact frame the engine is handed, which is what a fine-tuning set
must show. --channel 102 exists, but here it buys nothing.

WHY THIS OPENS ITS OWN STREAMS INSTEAD OF TAPPING THE RUNNING ENGINE
--------------------------------------------------------------------
The engine already decodes every camera, so reading ITS frames would be nearly
free. It is not reachable, for two structural reasons:

  * supervisor.py splits the fleet across SEPARATE WORKER PROCESSES (b1-gate,
    b1-areas, b2-1, b2-2, ground), each holding its own cam_manager with
    PROCESS-LOCAL frame buffers. Exactly one group runs --api, so an in-process
    frame hook (main.py get_camera_frame -> engine.cam_manager) can only ever
    see that one group's cameras.
  * the only live-frame route, /api/slots/{slot_id}/snapshot/live, returns a
    CROP — vehicle bbox, else slot polygon — not a full frame. A crop is one
    vehicle filling the picture: no localisation signal, useless for detector
    training, as tools/harvest_detector_frames.py spells out.

Closing that gap means a new full-frame endpoint in src/api.py and a VA
redeploy. Until then a sidecar is the only way to get full frames off all 25
cameras — so this keeps its added cost as close to a tap as it can: see the
grab()/retrieve() split in capture_one().

THE ONE THING THAT WOULD HAVE GONE WRONG (inherited, and it still applies)
-------------------------------------------------------------------------
The obvious way to get 1 fps is read -> sleep(1) -> read. Do not. OpenCV's
FFmpeg backend routinely ignores CAP_PROP_BUFFERSIZE=1, so sleeping between
reads lets a backlog build inside FFmpeg and every frame pulled is
progressively staler — minutes behind reality, while cap.read() still returns
instantly and everything LOOKS fine. You would end up with an hour of filenames
spanning an hour and pixels spanning five minutes.

So each camera thread drains its stream continuously with grab() and only
materializes a BGR frame at save time — the same split camera_manager.py uses
to sustain 8 fps of grabbing without paying 8 fps of decode.

THE SECOND THING (specific to running the fleet)
------------------------------------------------
A camera that cannot be resolved is SKIPPED and named in the summary. It never
falls back to another camera's address — capturing camera A's pixels into a
folder labelled camera B is worse than capturing nothing, because the mislabel
survives into the trained weights and is invisible afterwards.

Usage
-----
    # every enabled camera in the DB, 1 fps, 20 minutes
    python tools/capture_all_cameras.py --fps 1 --duration 1200

    # see what WOULD run — resolves every camera, opens no streams
    python tools/capture_all_cameras.py --list

    # a subset, and a hard disk ceiling
    python tools/capture_all_cameras.py --cameras CAM-01,CAM-02,CAM-23 --max-gb 20

    # 6 cameras at a time, each batch for 10 minutes (CPU-friendly)
    python tools/capture_all_cameras.py --batch 6 --duration 600

    # skip near-identical frames — most parking bays are the same picture
    python tools/capture_all_cameras.py --min-diff 1.5

Sources, in order:
    camera list + credentials  ->  $DATABASE_URL or config.yaml database.DATABASE_URL
    Fernet key                 ->  $CAMERAS_ENCRYPTION_KEY or config.yaml root key
    password (last resort)     ->  $CAM_PASSWORD, or the config.yaml camera entry
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    sys.exit(
        f"\n[cap] missing dependency: {exc.name}\n"
        f"[cap] running interpreter: {sys.executable}\n"
        f"[cap] python version:      {sys.version.split()[0]}\n\n"
        f"[cap] install it for THIS interpreter:\n"
        f'      "{sys.executable}" -m pip install opencv-python-headless numpy\n'
    )

# Prose in this repo uses em-dashes and arrows; a cp1252 console cannot encode
# them and --help would die before argparse printed a word.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

REPO = Path(__file__).resolve().parent.parent

# Each thread has its own decoder; OpenCV's internal pool on top of that just
# oversubscribes the box. FFmpeg's own decode threads are set separately below.
cv2.setNumThreads(1)

# LOAD-BEARING, copied verbatim from camera_manager._FFMPEG_OPTIONS.
#
# Omitting `threads;2` is not a tuning miss, it is an OOM. FFmpeg sizes EVERY
# capture's frame-thread pool from the visible CPUs — the whole ~20-core node
# under the unpinned/quota regime — so 28 cameras open ~560 decoder threads,
# each with its own frame buffers, and the pod's memory limit is gone in
# seconds. Measured on the engine 2026-07-16: the 12-camera b2 group carried
# ~240 threads in one process and decode stretched from 10-25ms to 442-844ms
# per frame. Two threads drain 720p comfortably; the total decode WORK is
# identical, only the pointless parallelism is gone.
#
# `rtsp_transport;tcp` matters just as much here: over UDP these streams drop
# packets under load and the decoder emits torn frames — which look fine in a
# thumbnail and quietly poison a training set.
# `stimeout` (microseconds) is the ONLY timeout that actually bites. OpenCV's
# CAP_PROP_OPEN_TIMEOUT_MSEC / CAP_PROP_READ_TIMEOUT_MSEC can only be set on an
# ALREADY-CONSTRUCTED VideoCapture — and the constructor is what opens the
# stream — so by the time they are applied the open has already happened or
# already hung. Measured 2026-08-16: 7 of 26 cameras sat in a single open for
# the entire 191s run and produced nothing, while the "timeouts" above them
# looked perfectly reasonable. 10s is generous for a camera on the LAN.
FFMPEG_OPTIONS = "rtsp_transport;tcp|threads;2|stimeout;10000000"

# ── defaults, tuned for the standing job: the whole fleet, for a day-plus, on a
# fixed disk budget ─────────────────────────────────────────────────────────
#
# 0.2 fps = one frame every 5s. 1 fps was the obvious first choice and it does
# not fit: 26 cameras x 30h at 1 fps is 2.8M frames, 107-214 GB depending on
# JPEG size, against a 55 GB budget. At 0.2 the same run lands at 21-43 GB.
DEFAULT_FPS = 0.2
# 102 = sub-stream. MEASURED 2026-08-16: it serves 1280x720 — the SAME size main
# (101) does on this fleet, which config.yaml's stream_channel note already
# says has no 4K to switch to. So the expected "sub-stream is cheaper" saving
# does not exist here: same pixels, same ~200 KB JPEG, same decode. It is kept
# as the default only because it takes the capture off the profile the engine is
# reading. The lever that ACTUALLY controls size on this fleet is --quality, not
# --channel.
DEFAULT_CHANNEL = 102
# The ANPR pair is deliberately out: they are the entrance/exit plate readers,
# not parking-scene cameras, and their frames belong to a different training
# problem. Overridden by naming them in --cameras explicitly.
DEFAULT_EXCLUDE = ("ANPR-Entry", "ANPR-Exit")


# ── config / camera resolution ──────────────────────────────────────────────

def find_config(explicit: str) -> Path | None:
    """config.yaml may not sit next to this file once it is copied into a pod."""
    for candidate in (
        Path(explicit) if explicit else None,
        REPO / "config.yaml",
        Path.cwd() / "config.yaml",
        Path("/app/config.yaml"),
    ):
        if candidate and candidate.is_file():
            return candidate
    return None


def load_config(path: Path | None) -> dict:
    if not path:
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"[cap] could not read {path} ({type(exc).__name__}: {exc})")
        return {}


def make_decrypt(key: str | None):
    """Mirrors src/utils/crypto.py: a blank key is the expected 'fill it in
    later' state and must yield None rather than raising."""
    if not key:
        return lambda token: None
    try:
        from cryptography.fernet import Fernet, InvalidToken
    except ModuleNotFoundError:
        print("[cap] cryptography not installed — cannot decrypt DB passwords")
        return lambda token: None
    try:
        cipher = Fernet(key.encode())
    except ValueError as exc:
        print(f"[cap] CAMERAS_ENCRYPTION_KEY is malformed ({exc}) — ignoring it")
        return lambda token: None

    def decrypt(token):
        if not token:
            return None
        try:
            return cipher.decrypt(token.encode()).decode()
        except InvalidToken:
            return None

    return decrypt


def cameras_from_db(url: str, decrypt) -> list[dict]:
    """Read every enabled camera from the gateway's `cameras` table.

    Talks to the DB directly rather than importing the gateway app: that import
    pulls the whole web stack (pydantic_settings and friends), which is not
    guaranteed to exist in the interpreter that HAS OpenCV.
    """
    try:
        from sqlalchemy import create_engine, text
    except ModuleNotFoundError:
        print("[cap] sqlalchemy not installed — cannot read the camera list from the DB")
        return []
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT camera_id, name, ip_address, rtsp_port, rtsp_path,
                       username, password_encrypted
                FROM cameras
                WHERE enabled = 1
                ORDER BY camera_id
            """)).mappings().all()
    except Exception as exc:
        print(f"[cap] camera list query failed ({type(exc).__name__}: {exc})")
        return []

    out = []
    for row in rows:
        out.append({
            "id": row.get("camera_id"),
            "name": row.get("name"),
            "ip": row.get("ip_address"),
            "port": int(row.get("rtsp_port") or 554),
            "path": row.get("rtsp_path") or "/Streaming/Channels/101",
            "user": row.get("username"),
            "password": decrypt(row.get("password_encrypted")),
            "source": "db",
        })
    return out


def cameras_from_yaml(cfg: dict) -> list[dict]:
    out = []
    for cam in cfg.get("cameras") or []:
        if not cam.get("id"):
            continue
        out.append({
            "id": cam["id"],
            "name": cam.get("name"),
            "ip": cam.get("ip"),
            "port": int(cam.get("rtsp_port") or 554),
            "path": cam.get("rtsp_path") or "/Streaming/Channels/101",
            "user": cam.get("user"),
            "password": cam.get("password"),
            "source": "config.yaml",
        })
    return out


def resolve_cameras(args, cfg: dict) -> tuple[list[dict], list[str]]:
    """Return (usable cameras, reasons for the ones dropped)."""
    key = os.environ.get("CAMERAS_ENCRYPTION_KEY") or cfg.get("CAMERAS_ENCRYPTION_KEY")
    url = (os.environ.get("DATABASE_URL")
           or (cfg.get("database") or {}).get("DATABASE_URL") or "")

    cams: list[dict] = []
    if url and not args.no_db:
        cams = cameras_from_db(url, make_decrypt(key))
        print(f"[cap] {len(cams)} enabled camera(s) from the DB")
    elif not url:
        print("[cap] no DATABASE_URL (env or config.yaml) — falling back to config.yaml")

    # config.yaml fills gaps rather than replacing: the DB is authoritative for
    # WHICH cameras exist, the YAML often still carries a usable password.
    yaml_cams = {c["id"]: c for c in cameras_from_yaml(cfg)}
    if not cams:
        cams = list(yaml_cams.values())
        print(f"[cap] {len(cams)} camera(s) from config.yaml")
    else:
        for cam in cams:
            if not cam.get("password"):
                fallback = yaml_cams.get(cam["id"]) or {}
                if fallback.get("password"):
                    cam["password"] = fallback["password"]
                    cam["source"] += "+yaml-pw"
            if not cam.get("user"):
                cam["user"] = (yaml_cams.get(cam["id"]) or {}).get("user")

    wanted = {c.strip() for c in args.cameras.split(",") if c.strip()} if args.cameras else None
    if args.exclude is None:
        # Default exclusion applies only to "everything" runs. If someone names
        # cameras explicitly, an invisible default must not overrule them —
        # asking for a camera and silently getting nothing is worse than an
        # error. `--exclude ""` clears it.
        excluded = set() if wanted else set(DEFAULT_EXCLUDE)
    else:
        excluded = {c.strip() for c in args.exclude.split(",") if c.strip()}

    shared_pw = os.environ.get("CAM_PASSWORD")
    usable, dropped = [], []
    seen = set()
    for cam in cams:
        cid = cam["id"]
        if cid in seen:
            continue
        seen.add(cid)
        if wanted is not None and cid not in wanted:
            continue
        if cid in excluded:
            dropped.append(f"{cid}: excluded")
            continue
        if not cam.get("ip"):
            # Never substitute another camera's address. A mislabelled folder
            # poisons the dataset in a way that is invisible after training.
            dropped.append(f"{cid}: no ip_address — SKIPPED (not guessed)")
            continue
        if not cam.get("password"):
            cam["password"] = shared_pw
        if not cam.get("password"):
            dropped.append(f"{cid}: no password (DB decrypt failed, no config.yaml "
                           f"entry, $CAM_PASSWORD unset) — SKIPPED")
            continue
        if args.channel:
            cam["path"] = f"/Streaming/Channels/{args.channel}"
        usable.append(cam)

    if wanted:
        for cid in sorted(wanted - seen):
            dropped.append(f"{cid}: requested but not found in the DB or config.yaml")

    return usable, dropped


def rtsp_url(cam: dict) -> tuple[str, str]:
    """Return (url, redacted). Credentials are URL-encoded: the camera password
    in this deployment contains '@', which truncates an unencoded URL."""
    user, pw = cam.get("user"), cam.get("password")
    auth = f"{quote(user, safe='')}:{quote(pw, safe='')}@" if user else ""
    shown = f"{quote(user, safe='')}:***@" if user else ""
    base = f"{cam['ip']}:{cam['port']}{cam['path']}"
    return f"rtsp://{auth}{base}", f"rtsp://{shown}{base}"


# ── capture ─────────────────────────────────────────────────────────────────

class Stats:
    """Shared counters. The byte total is what enforces --max-gb, and at 25
    cameras of main-stream JPEG that ceiling is the difference between a
    dataset and a full disk on a node that other pods share."""

    def __init__(self):
        self.lock = threading.Lock()
        self.saved: dict[str, int] = {}
        self.skipped: dict[str, int] = {}
        self.reconnects: dict[str, int] = {}
        self.errors: dict[str, str] = {}
        self.resolution: dict[str, str] = {}
        self.bytes = 0

    def add(self, cid: str, nbytes: int) -> int:
        with self.lock:
            self.saved[cid] = self.saved.get(cid, 0) + 1
            self.bytes += nbytes
            return self.bytes

    def bump(self, bucket: dict, cid: str):
        with self.lock:
            bucket[cid] = bucket.get(cid, 0) + 1


def open_stream(url: str, host: str = "", port: int = 0, connect_timeout: float = 4.0):
    """Open an RTSP capture, or None.

    The TCP preflight is not redundant with `stimeout`: that option bounds
    socket I/O AFTER a connection exists, and bounds nothing while the kernel is
    still retrying SYNs. A host that silently drops packets therefore parks
    cv2.VideoCapture() in connect() for the OS default — ~130s on Linux with the
    usual tcp_syn_retries=6. That is exactly what 7 of 26 cameras did on
    2026-08-16: one open, 191 seconds, no frames, and no error to show for it.
    A 4-second connect attempt turns an invisible hang into a fast, loggable
    failure that the retry loop can actually work with.
    """
    if host:
        try:
            with socket.create_connection((host, port or 554), timeout=connect_timeout):
                pass
        except OSError:
            return None
    return _open_capture(url)


def _open_capture(url: str):
    # Set on EVERY open, not once at startup: it is read by FFmpeg when the
    # capture is constructed, and reconnects construct new ones.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FFMPEG_OPTIONS
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # honoured or not, ask anyway
    # No OPEN/READ_TIMEOUT here on purpose: the constructor above has already
    # opened (or hung on) the stream, so setting them now would be theatre. The
    # real timeout is `stimeout` in FFMPEG_OPTIONS.
    return cap if cap.isOpened() else None


def cgroup_memory() -> tuple[int, int] | None:
    """(used_bytes, limit_bytes) for this pod's cgroup, or None if unavailable
    or unlimited. Tries cgroup v2 then v1."""
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),          # v2
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",                          # v1
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            limit_raw = Path(limit_path).read_text().strip()
            used = int(Path(usage_path).read_text().strip())
            if limit_raw == "max":
                return None
            return used, int(limit_raw)
        except (OSError, ValueError):
            continue
    return None


def process_rss() -> int:
    """This process's resident set, in bytes. 0 if it cannot be read.

    Reported next to the cgroup total because the two answer different
    questions: if RSS climbs, this script is the leak; if only the cgroup total
    climbs, the growth is page cache from the frames being written, or the
    engine. After an OOM kill that distinction is the whole investigation.
    """
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return 0


def report_cgroup_memory(n_cams: int, isapi: bool = False) -> None:
    """Print the pod's memory headroom before opening 28 decoders into it.

    This normally runs INSIDE the pms-video-analytics pod, sharing one cgroup
    limit with the engine's worker processes. The OOM killer picks a victim by
    footprint, not by who caused the pressure — so an over-ambitious capture run
    can get a WORKER killed instead of itself, and a worker exiting -9 takes the
    whole supervisor down (config.yaml records exactly that outage on
    2026-07-30). Read-only: it reports, it does not refuse.
    """
    mem = cgroup_memory()
    if mem is None:
        return
    used, limit = mem
    mb = 1 << 20
    free = (limit - used) / mb
    what = "camera(s) over HTTP" if isapi else "decoder(s) to open"
    print(f"[cap] pod memory: {used / mb:.0f} MB used of {limit / mb:.0f} MB "
          f"({free:.0f} MB free) — {n_cams} {what}")
    if isapi:
        # Measured 2026-08-16: 26 cameras on --isapi held a flat 241 MB RSS.
        # There is nothing here to warn about.
        return
    # ~2 threads x buffered frames per decoder, plus the RTSP/FFmpeg context.
    # Empirical, not a formula from the docs: treat it as an alarm, not a spec.
    if free < n_cams * 60:
        print(f"[cap] WARNING: that is tight. If the OOM killer fires it may "
              f"take an ENGINE WORKER, not this script. Consider "
              f"--batch {max(2, int(free // 120))} or a lower --fps.")


def frame_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference, 0-255, on a small grayscale thumbnail — cheap
    enough to run per saved frame on every camera at once."""
    small = lambda f: cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(small(a), small(b))))


class StopSignal:
    """Two reasons a camera thread should quit, behind one interface.

    `group_stop` means "this batch's turn is over" — the next batch is about to
    start and these decoders MUST be gone before it does, or rotating through
    batches would just accumulate open streams until the pod dies. `global_stop`
    means the whole run ends (signal, duration, disk ceiling, memory floor).

    `set()` deliberately raises to the GLOBAL stop: its only caller is
    save_frame hitting the disk ceiling, which is never a per-batch condition.
    """

    def __init__(self, global_stop: threading.Event):
        self.global_stop = global_stop
        self.group_stop = threading.Event()

    def is_set(self) -> bool:
        return self.global_stop.is_set() or self.group_stop.is_set()

    def set(self) -> None:
        self.global_stop.set()

    def wait(self, timeout: float) -> bool:
        """True if we should stop. Polls, because there are two events to watch
        and threading offers no wait-on-any."""
        deadline = time.perf_counter() + timeout
        while True:
            if self.is_set():
                return True
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False
            self.group_stop.wait(min(remaining, 0.25))


def save_bytes(data: bytes, cid: str, out: Path, args, stats: Stats,
               byte_cap: int, stop) -> str:
    """Write one already-encoded JPEG. Returns "saved" or "stop"."""
    wall = datetime.now()
    name = f"{cid}_{wall:%Y%m%d_%H%M%S}_{wall.microsecond // 1000:03d}.jpg"
    try:
        (out / name).write_bytes(data)
    except OSError as exc:
        stats.errors[cid] = f"write failed: {exc}"
        print(f"[cap] {cid}: WRITE FAILED ({exc}) — stopping this camera")
        return "stop"

    total = stats.add(cid, len(data))
    if byte_cap and total >= byte_cap:
        print(f"[cap] disk ceiling of {args.max_gb:g} GB reached — stopping all cameras")
        stop.set()
        return "stop"
    if args.max_frames and stats.saved.get(cid, 0) >= args.max_frames:
        print(f"[cap] {cid}: reached --max-frames {args.max_frames}")
        return "stop"
    return "saved"


def capture_one_isapi(cam: dict, out_root: Path, args, stats: Stats, stop):
    """Fetch JPEGs over HTTP from the camera's own snapshot endpoint.

    NO DECODER ANYWHERE. The camera encodes the JPEG in its own silicon and
    hands it over already compressed; this thread copies those bytes to disk.
    That removes the entire reason the RTSP modes are expensive — an H.264
    stream carries P-frames, which are deltas rather than pictures, so turning
    one into a .jpg means reconstructing it from the last keyframe forward
    (~40-100 MB of decoder state per camera, held open) and re-encoding. Here
    the peak memory per camera is one JPEG, a few hundred KB.

    This is the right mode for exactly this job: stills, at a low rate, from a
    box that is already CPU- and memory-starved. The RTSP modes stay because
    they are the only way to get frames faster than the snapshot endpoint can
    serve them, and the only way to be certain the pixels match what the engine
    sees on the same profile.

    Same endpoint PMS-AI's snapshot_service.py uses, so the credentials and
    privileges are already proven in this deployment. Its warning applies here
    too: a 401 usually means the ISAPI account lacks "Live View / Image
    Capture", not that the password is wrong.
    """
    cid = cam["id"]
    out = out_root / cid
    out.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / args.fps
    byte_cap = int(args.max_gb * (1 << 30)) if args.max_gb else 0

    try:
        import requests
        from requests.auth import HTTPBasicAuth, HTTPDigestAuth
    except ModuleNotFoundError:
        stats.errors[cid] = "requests not installed (needed for --isapi)"
        print(f"[cap] {cid}: `requests` is not installed — --isapi unavailable")
        return

    channel = args.channel or 101
    url = (f"http://{cam['ip']}:{args.isapi_port}"
           f"/ISAPI/Streaming/channels/{channel}/picture")
    user, password = cam.get("user") or "", cam.get("password") or ""

    session = requests.Session()
    session.auth = HTTPDigestAuth(user, password)
    print(f"[cap] {cid}: isapi <- {url} (user {user})")

    last_saved = None
    failures = 0
    tried_basic = False
    next_save = time.perf_counter()

    try:
        while not stop.is_set():
            if stop.wait(max(0.0, next_save - time.perf_counter())):
                break
            next_save = time.perf_counter() + interval

            try:
                resp = session.get(url, timeout=args.isapi_timeout)
            except Exception as exc:
                failures += 1
                if failures in (1, 5, 20) or failures % 100 == 0:
                    print(f"[cap] {cid}: snapshot request failed "
                          f"({type(exc).__name__}) #{failures}")
                continue

            if resp.status_code in (401, 403):
                # Digest is what Hikvision firmware normally wants, but some
                # units are configured for Basic. Try the other scheme once
                # before calling it a permissions problem.
                if not tried_basic:
                    tried_basic = True
                    session.auth = HTTPBasicAuth(user, password)
                    print(f"[cap] {cid}: {resp.status_code} on digest — retrying with basic auth")
                    continue
                stats.errors[cid] = (f"HTTP {resp.status_code} — check ISAPI 'Live View / "
                                     f"Image Capture' privileges for user '{user}'")
                print(f"[cap] {cid}: HTTP {resp.status_code} — the account likely lacks ISAPI "
                      f"'Live View / Image Capture' privileges. Giving up on this camera.")
                return
            if resp.status_code == 404 and channel != 101:
                # The picture endpoint is not guaranteed to expose the same
                # channel ids the RTSP path does — PMS-AI's snapshot_service
                # asks for plain `1`. Fall back to the main channel once rather
                # than logging 404s for 30 hours.
                print(f"[cap] {cid}: no snapshot on channel {channel} (404) — "
                      f"falling back to 101")
                channel = 101
                url = (f"http://{cam['ip']}:{args.isapi_port}"
                       f"/ISAPI/Streaming/channels/101/picture")
                continue

            if resp.status_code != 200 or not resp.content:
                failures += 1
                if failures in (1, 5, 20):
                    print(f"[cap] {cid}: HTTP {resp.status_code}, "
                          f"{len(resp.content)} bytes #{failures}")
                continue

            data = resp.content
            if not data.startswith(b"\xff\xd8"):
                # Not a JPEG: some firmware answers errors with an XML body and
                # a 200. Writing that to a .jpg would seed the training set with
                # files that are not images.
                failures += 1
                if failures in (1, 5):
                    print(f"[cap] {cid}: response is not a JPEG "
                          f"({data[:40]!r}) — skipping")
                continue

            failures = 0

            if cid not in stats.resolution or args.min_diff > 0:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    if cid not in stats.resolution:
                        stats.resolution[cid] = f"{frame.shape[1]}x{frame.shape[0]}"
                        print(f"[cap] {cid}: {stats.resolution[cid]} (isapi)")
                    if args.min_diff > 0:
                        if last_saved is not None and frame_delta(frame, last_saved) < args.min_diff:
                            stats.bump(stats.skipped, cid)
                            continue
                        last_saved = frame

            if save_bytes(data, cid, out, args, stats, byte_cap, stop) == "stop":
                return
    finally:
        session.close()


def save_frame(frame, cid: str, out: Path, encode, args, stats: Stats,
               byte_cap: int, stop: threading.Event, last_saved):
    """Encode and write one frame. Returns (status, last_saved), where status is
    "saved", "skipped" (too similar), "error" (encode failed) or "stop" (the run
    must end — disk ceiling, per-camera limit, or an unwritable path)."""
    if cid not in stats.resolution:
        # Report what the camera ACTUALLY served. --channel is a request, not a
        # guarantee: an NVR without the requested profile can answer with a
        # different one and never say so, and a dataset silently captured at the
        # wrong resolution is only discovered after training on it.
        stats.resolution[cid] = f"{frame.shape[1]}x{frame.shape[0]}"
        print(f"[cap] {cid}: {stats.resolution[cid]}")

    if args.min_diff > 0 and last_saved is not None:
        if frame_delta(frame, last_saved) < args.min_diff:
            stats.bump(stats.skipped, cid)
            return "skipped", last_saved

    ok, buf = cv2.imencode(".jpg", frame, encode)
    if not ok:
        print(f"[cap] {cid}: WARNING encode failed")
        return "error", last_saved

    wall = datetime.now()
    name = f"{cid}_{wall:%Y%m%d_%H%M%S}_{wall.microsecond // 1000:03d}.jpg"
    try:
        (out / name).write_bytes(buf.tobytes())
    except OSError as exc:
        stats.errors[cid] = f"write failed: {exc}"
        print(f"[cap] {cid}: WRITE FAILED ({exc}) — stopping this camera")
        return "stop", last_saved

    total = stats.add(cid, buf.size)
    # Only --min-diff needs the previous frame. Holding it regardless would pin
    # a decoded frame per camera for nothing, and this runs inside a pod whose
    # memory limit is shared with the engine.
    last_saved = frame if args.min_diff > 0 else None

    if byte_cap and total >= byte_cap:
        print(f"[cap] disk ceiling of {args.max_gb:g} GB reached — stopping all cameras")
        stop.set()
        return "stop", last_saved
    if args.max_frames and stats.saved.get(cid, 0) >= args.max_frames:
        print(f"[cap] {cid}: reached --max-frames {args.max_frames}")
        return "stop", last_saved
    return "saved", last_saved


def capture_one_snap(cam: dict, out_root: Path, args, stats: Stats, stop: threading.Event):
    """CONNECT -> grab one frame -> DISCONNECT, once per interval.

    For a long run at a low rate this is dramatically cheaper than holding the
    stream open: a 30-hour job at one frame per 5s spends a second or two per
    cycle inside FFmpeg instead of decoding ~20 fps x 26 cameras continuously
    for 30 hours next to an engine that is already CPU-starved.

    MEASURED ON THIS FLEET (2026-08-16, 26 cameras, sub-stream): a cycle costs
    15-20s, not the ~2s the idea assumes. A 120s run at a nominal 5s interval
    produced 5-7 frames per camera instead of 24. The RTSP handshake plus the
    wait for the next H.264 KEYFRAME dominates, and 26 cameras negotiating at
    once makes it worse.

    So this mode is only worth using when the interval is comfortably longer
    than a cycle — one frame per minute or slower. At anything faster, the
    default continuous mode collects several times more frames for CPU that is
    real but bounded. Left in because a week-long trickle capture is a genuine
    use for it.
    """
    cid = cam["id"]
    url, redacted = rtsp_url(cam)
    out = out_root / cid
    out.mkdir(parents=True, exist_ok=True)
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, args.quality))]
    interval = 1.0 / args.fps
    byte_cap = int(args.max_gb * (1 << 30)) if args.max_gb else 0

    print(f"[cap] {cid}: snap mode <- {redacted}")
    last_saved = None
    consecutive_failures = 0

    while not stop.is_set():
        cycle_started = time.perf_counter()
        cap = open_stream(url, cam["ip"], cam["port"])
        got = False
        if cap is not None:
            try:
                # Give the decoder a bounded window to produce its first frame;
                # most of that wait is the keyframe, not the network.
                deadline = time.perf_counter() + args.snap_timeout
                while time.perf_counter() < deadline and not stop.is_set():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        status, last_saved = save_frame(
                            frame, cid, out, encode, args, stats,
                            byte_cap, stop, last_saved)
                        got = status in ("saved", "skipped")
                        if status == "stop":
                            return
                        break
            finally:
                cap.release()

        if got:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            stats.bump(stats.reconnects, cid)
            if consecutive_failures in (1, 5, 20) or consecutive_failures % 100 == 0:
                print(f"[cap] {cid}: no frame this cycle "
                      f"(#{consecutive_failures} consecutive)")
            if consecutive_failures == 20:
                stats.errors[cid] = "20 consecutive failed snap cycles"

        # Sleep the REMAINDER of the interval. Unlike the continuous mode there
        # is no socket to keep drained — the connection is already closed — so
        # sleeping here cannot build a stale backlog.
        elapsed = time.perf_counter() - cycle_started
        if stop.wait(max(0.0, interval - elapsed)):
            break


def capture_one(cam: dict, out_root: Path, args, stats: Stats, stop: threading.Event):
    cid = cam["id"]
    url, redacted = rtsp_url(cam)
    out = out_root / cid
    out.mkdir(parents=True, exist_ok=True)
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, args.quality))]
    interval = 1.0 / args.fps
    byte_cap = int(args.max_gb * (1 << 30)) if args.max_gb else 0

    cap = open_stream(url, cam["ip"], cam["port"])
    if cap is None:
        stats.errors[cid] = f"could not open {redacted}"
        print(f"[cap] {cid}: COULD NOT OPEN {redacted}")
        return

    print(f"[cap] {cid}: capturing <- {redacted}")
    last_saved = None
    read_fail = 0
    next_save = time.perf_counter()

    try:
        while not stop.is_set():
            # grab() CONTINUOUSLY, retrieve() only at save time — the same split
            # src/camera_manager.py::_grabber_loop uses, and the whole reason
            # this is affordable next to a running engine. grab() consumes the
            # compressed frame and keeps the socket drained (never sleep here:
            # BUFFERSIZE=1 is ignored by the FFmpeg backend, so a sleeping
            # reader falls minutes behind while cap.read() still returns
            # instantly). retrieve() is what actually costs — the H.264 decode,
            # colorspace conversion and copy — and at 1 fps we pay it once a
            # second instead of ~20 times.
            ok = cap.grab()
            if not ok:
                read_fail += 1
                if read_fail >= 30:
                    stats.bump(stats.reconnects, cid)
                    n = stats.reconnects.get(cid, 1)
                    print(f"[cap] {cid}: stream dropped; reconnecting (#{n})")
                    cap.release()
                    if stop.wait(min(2.0 * n, 15.0)):
                        break
                    cap = open_stream(url, cam["ip"], cam["port"])
                    if cap is None:
                        print(f"[cap] {cid}: reconnect failed")
                        cap = cv2.VideoCapture()  # keeps the loop alive to retry
                    read_fail = 0
                continue
            read_fail = 0

            # Deadline is evaluated AFTER the blocking grab, so the frame that
            # crosses it is the one materialized — not the one before it.
            now = time.perf_counter()
            if now < next_save:
                continue  # drained, deliberately not decoded

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            status, last_saved = save_frame(frame, cid, out, encode, args, stats,
                                            byte_cap, stop, last_saved)
            if status == "stop":
                break

            # Advance on a fixed grid, re-basing if we fell more than one
            # interval behind so a slow disk cannot make the rate creep.
            next_save += interval
            if next_save < now:
                next_save = now + interval
    finally:
        try:
            cap.release()
        except Exception:
            pass


def run_group(cams: list[dict], out_root: Path, args, stats: Stats,
              stop: threading.Event, seconds: float) -> None:
    """Capture this group for `seconds` (0 = until stopped). Guarantees every
    thread it starts is gone before it returns, so the caller can rotate to the
    next batch without leaving decoders behind."""
    signal_ = StopSignal(stop)
    threads = []
    worker = (capture_one_isapi if args.isapi
              else capture_one_snap if args.snap
              else capture_one)
    for cam in cams:
        t = threading.Thread(target=worker,
                             args=(cam, out_root, args, stats, signal_),
                             name=cam["id"], daemon=True)
        t.start()
        threads.append(t)
        # Stagger the RTSP handshakes: 25 simultaneous DESCRIBE/SETUP requests
        # make some NVR-fronted cameras refuse the connection outright.
        if signal_.wait(args.stagger):
            break

    started = time.perf_counter()
    last_report = started
    while any(t.is_alive() for t in threads):
        if stop.wait(1.0):
            break
        now = time.perf_counter()
        if seconds and now - started >= seconds:
            break
        mem = cgroup_memory()
        if mem is not None:
            used, limit = mem
            free_mb = (limit - used) / (1 << 20)
            if args.mem_floor > 0 and free_mb < args.mem_floor:
                # Stop OURSELVES rather than let the kernel choose. The cgroup is
                # shared with the engine's workers and the OOM killer picks by
                # footprint, not by blame — a worker exiting -9 takes the whole
                # supervisor down. A truncated dataset is recoverable; that is not.
                print(f"[cap] MEMORY FLOOR: only {free_mb:.0f} MB free "
                      f"(< --mem-floor {args.mem_floor:g} MB) — stopping cleanly "
                      f"before the OOM killer picks a victim")
                stop.set()
                break

        if now - last_report >= args.report:
            elapsed = now - started
            with stats.lock:
                total = sum(stats.saved.values())
                gb = stats.bytes / (1 << 30)
                silent = [c["id"] for c in cams if not stats.saved.get(c["id"])]
            line = f"[cap] {total} frames | {gb:.2f} GB | {elapsed:.0f}s"
            if mem is not None:
                used, limit = mem
                line += (f" | pod {used / (1 << 20):.0f}/{limit / (1 << 20):.0f} MB"
                         f" rss {process_rss() / (1 << 20):.0f} MB")
            if silent:
                line += f" | NO FRAMES YET: {', '.join(silent)}"
            print(line)
            last_report = now

    # Tell THIS batch's threads to finish, whatever ended the loop, and wait for
    # them. Rotating to the next batch while these decoders are still open is
    # how a bounded run turns into an unbounded one.
    signal_.group_stop.set()
    for t in threads:
        t.join(timeout=20.0)
    still_alive = [t.name for t in threads if t.is_alive()]
    if still_alive:
        print(f"[cap] WARNING: {len(still_alive)} thread(s) did not exit in time "
              f"({', '.join(still_alive)}) — their decoders are still holding memory")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS,
                    help=f"frames SAVED per second per camera (default {DEFAULT_FPS:g}, "
                         f"i.e. one frame every {1 / DEFAULT_FPS:g}s)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to capture; 0 = until Ctrl+C / SIGTERM (default)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop a camera after N frames; 0 = unlimited")
    ap.add_argument("--max-gb", type=float, default=0.0,
                    help="stop EVERYTHING once the run has written this many GB; 0 = no ceiling")
    ap.add_argument("--out", default="", help="output dir (default data/finetune/all/<timestamp>)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default 95)")
    ap.add_argument("--min-diff", type=float, default=0.0,
                    help="skip a frame unless it differs from the last SAVED one by this "
                         "mean-abs-diff (0-255). 0 = save everything (default)")
    ap.add_argument("--channel", type=int, default=DEFAULT_CHANNEL,
                    help=f"101 main / 102 sub (default {DEFAULT_CHANNEL}). "
                         f"Pass 0 to use each camera's rtsp_path from the DB")
    ap.add_argument("--cameras", default="", help="comma-separated ids; default = every enabled camera")
    ap.add_argument("--exclude", default=None,
                    help=f"comma-separated ids to skip. Defaults to {','.join(DEFAULT_EXCLUDE)} "
                         f"(only when --cameras is not given). Pass an empty string to skip nothing")
    ap.add_argument("--batch", type=int, default=0,
                    help="hold at most N cameras open at once, in sequential batches; "
                         "0 = all at once (default). This is the memory lever: N decoders "
                         "live instead of 26")
    ap.add_argument("--batch-seconds", type=float, default=0.0,
                    help="seconds each batch holds its cameras before handing over. Set this "
                         "to ROTATE: batches take turns for the whole --duration, so every "
                         "camera is sampled across the entire window instead of one slice of "
                         "it. 0 = one pass, each batch running the full duration")
    ap.add_argument("--mem-floor", type=float, default=400.0,
                    help="stop the run cleanly if the pod's free memory drops below this many "
                         "MB (default 400). Guards the ENGINE: the OOM killer picks by "
                         "footprint, not by blame. 0 disables")
    ap.add_argument("--isapi", action="store_true",
                    help="fetch JPEGs from each camera's HTTP snapshot endpoint instead of "
                         "decoding RTSP. No decoder, no FFmpeg: peak memory per camera is one "
                         "JPEG rather than ~40-100 MB of decoder state. Best choice for stills "
                         "at a low rate on a loaded box")
    ap.add_argument("--isapi-port", type=int, default=80, help="--isapi only: HTTP port (default 80)")
    ap.add_argument("--isapi-timeout", type=float, default=10.0,
                    help="--isapi only: per-request timeout in seconds (default 10)")
    ap.add_argument("--snap", action="store_true",
                    help="connect, grab ONE frame, disconnect, once per interval. MEASURED ON "
                         "THIS FLEET: a connect+keyframe cycle costs 15-20s, so this only makes "
                         "sense at intervals of a minute or more (--fps 0.016 or lower). At the "
                         "default 5s interval it delivers ~1/4 of the requested frames")
    ap.add_argument("--snap-timeout", type=float, default=12.0,
                    help="--snap only: seconds to wait for the first decodable frame (default 12)")
    ap.add_argument("--stagger", type=float, default=1.0, help="seconds between stream opens (default 1)")
    ap.add_argument("--report", type=float, default=30.0, help="progress line every N seconds (default 30)")
    ap.add_argument("--config", default="", help="path to config.yaml (default: alongside the repo)")
    ap.add_argument("--no-db", action="store_true", help="ignore the DB, use config.yaml only")
    ap.add_argument("--list", action="store_true", help="resolve and print the camera list, capture nothing")
    args = ap.parse_args()

    if args.fps <= 0:
        raise SystemExit("[cap] --fps must be > 0")

    cfg_path = find_config(args.config)
    cfg = load_config(cfg_path)
    print(f"[cap] config: {cfg_path or 'none found'}")

    cams, dropped = resolve_cameras(args, cfg)
    for reason in dropped:
        print(f"[cap] DROPPED {reason}")
    if not cams:
        print("[cap] no usable cameras — nothing to do")
        return 1

    if args.list:
        print(f"\n[cap] {len(cams)} camera(s) would be captured:")
        for cam in cams:
            print(f"  {cam['id']:<10} {rtsp_url(cam)[1]:<55} [{cam['source']}]")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else REPO / "data" / "finetune" / "all" / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    report_cgroup_memory(len(cams), args.isapi)

    groups = ([cams[i:i + args.batch] for i in range(0, len(cams), args.batch)]
              if args.batch > 0 else [cams])

    print(f"[cap] {len(cams)} camera(s), {args.fps:g} fps each -> {out_root}")
    print(f"[cap] {len(groups)} batch(es) of up to {args.batch or len(cams)}"
          f", {'until Ctrl+C/SIGTERM' if not args.duration else f'{args.duration:g}s each'}")
    if not args.duration and not args.max_gb:
        # 25 main-stream cameras at 1 fps is ~100 GB/hour. An unbounded run on a
        # shared node fills the disk long before anyone thinks to check.
        print("[cap] NOTE: no --duration and no --max-gb — this runs until stopped. "
              "At main-stream resolution that is ~4 GB/hour PER CAMERA.")

    stats = Stats()
    stop = threading.Event()

    def _stop(signum, frame):
        del signum, frame
        if not stop.is_set():
            print("\n[cap] stopping — finishing frames in flight")
            stop.set()

    signal.signal(signal.SIGINT, _stop)
    # SIGTERM is how a pod is asked to go away (kubectl delete, evictions). Without
    # this the container is SIGKILLed after the grace period, mid-write.
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    started = time.perf_counter()
    deadline = started + args.duration if args.duration else 0.0
    cycle = 0
    while not stop.is_set():
        cycle += 1
        for i, group in enumerate(groups, 1):
            if stop.is_set():
                break
            now = time.perf_counter()
            remaining = deadline - now if deadline else 0.0
            if deadline and remaining <= 0:
                break
            # How long this batch holds the cameras. Without --batch-seconds a
            # batch keeps them for the whole run (one pass, no rotation).
            slice_s = args.batch_seconds or (remaining if deadline else 0.0)
            if deadline:
                slice_s = min(slice_s, remaining)
            if len(groups) > 1:
                print(f"\n[cap] ── cycle {cycle}, batch {i}/{len(groups)} "
                      f"({slice_s:.0f}s): {', '.join(c['id'] for c in group)}")
            run_group(group, out_root, args, stats, stop, slice_s)

        if not args.batch_seconds or len(groups) == 1:
            break   # single pass: every group has had its full turn
        if deadline and time.perf_counter() >= deadline:
            break
    if deadline and time.perf_counter() >= deadline:
        print(f"[cap] duration {args.duration:g}s reached")

    elapsed = max(time.perf_counter() - started, 1e-6)
    total = sum(stats.saved.values())
    print(f"\n[cap] done: {total} frames, {stats.bytes / (1 << 30):.2f} GB "
          f"in {elapsed:.0f}s -> {out_root}")
    for cam in cams:
        cid = cam["id"]
        n = stats.saved.get(cid, 0)
        note = ""
        if stats.skipped.get(cid):
            note += f", {stats.skipped[cid]} near-identical skipped"
        if stats.reconnects.get(cid):
            note += f", {stats.reconnects[cid]} reconnect(s)"
        if stats.errors.get(cid):
            note += f", ERROR: {stats.errors[cid]}"
        flag = "  <-- NOTHING CAPTURED" if n == 0 else ""
        print(f"  {cid:<10} {n:>6} frames{note}{flag}")

    manifest = {
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_s": round(elapsed, 1),
        "fps": args.fps,
        "quality": args.quality,
        "min_diff": args.min_diff,
        "total_frames": total,
        "total_bytes": stats.bytes,
        "dropped": dropped,
        "cameras": [{
            "camera_id": c["id"],
            "name": c.get("name"),
            "rtsp": rtsp_url(c)[1],
            "source": c["source"],
            "frames": stats.saved.get(c["id"], 0),
            "skipped": stats.skipped.get(c["id"], 0),
            "reconnects": stats.reconnects.get(c["id"], 0),
            "resolution": stats.resolution.get(c["id"]),
            "error": stats.errors.get(c["id"]),
        } for c in cams],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[cap] manifest -> {out_root / 'manifest.json'}")

    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
