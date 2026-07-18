"""probe_camera_lag.py — is the RTSP layer QUEUEING frames behind our back?

THE QUESTION THIS ANSWERS. Every queue inside VA is bounded and measured
(latest-frame slot, feed_q, out_q, nireq) — but there is one queue BEFORE our
first timestamp: OpenCV/FFmpeg's internal RTSP buffering. ``capture_ts`` is
stamped when ``cap.read()`` RETURNS, so if FFmpeg is holding a backlog, a frame
that is minutes old gets stamped "fresh" and every downstream measurement
([SLOTTRACE], frame-age, fps) looks perfect while the engine processes the
past. The production grabber reads with a throttle (read → sleep → read, see
camera_manager.py max_grab_fps), which is exactly the pattern that lets a
backlog grow when CAP_PROP_BUFFERSIZE=1 is not honoured — and the FFmpeg
backend is notorious for ignoring it.

WHAT IT DOES (per camera):
  P0  connect, save the first frame           -> always live; OSD reference
  P1  read unthrottled 5s                     -> the camera's true delivery fps
  P2  read at --grab-fps for --throttle-secs  -> the PRODUCTION pattern
      save the last frame                     -> its OSD clock vs wall clock
                                                 IS the accumulated lag
  P3  burst-drain as fast as possible         -> frames returning faster than
      save the last frame                        the camera fps were QUEUED;
                                                 count them = backlog size

READ THE RESULT LIKE THIS:
  * P3 "drained N frames at M/s" with M >> camera fps  -> the queue is REAL;
    backlog/camera_fps = seconds of lag the engine was living in the past.
  * P3 drains at ~camera fps and the P2 frame's OSD clock reads ~now -> no
    queue on this camera; the lag lives elsewhere.
  * The saved JPGs make it operator-visible: compare the burned-in OSD time
    in p2_after_throttle.jpg against the wall= time printed when it was saved.

Usage:
    python tools/probe_camera_lag.py --url "rtsp://user:pass@10.1.13.63:554/Streaming/Channels/102"
    python tools/probe_camera_lag.py --ip 10.1.13.63 --user admin --password '...' [--channel 102]
        [--grab-fps 3] [--throttle-secs 30] [--out probe_out]
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.parse
from datetime import datetime
from pathlib import Path


def wall() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-4]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", help="full rtsp:// URL (overrides ip/user/password)")
    ap.add_argument("--ip")
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--channel", type=int, default=102,
                    help="Hikvision channel: 101 main / 102 sub (default 102)")
    ap.add_argument("--grab-fps", type=float, default=3.0,
                    help="throttled read rate for the production-pattern phase")
    ap.add_argument("--throttle-secs", type=float, default=30.0)
    ap.add_argument("--out", default="probe_out")
    args = ap.parse_args()

    if args.url:
        url = args.url
    else:
        if not (args.ip and args.user and args.password):
            ap.error("give --url or all of --ip/--user/--password")
        pw = urllib.parse.quote(args.password, safe="")
        url = f"rtsp://{args.user}:{pw}@{args.ip}:554/Streaming/Channels/{args.channel}"

    import cv2  # after argparse so --help works without opencv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(tag, frame):
        p = out / f"{tag}.jpg"
        cv2.imwrite(str(p), frame)
        print(f"  saved {p}  wall={wall()}")

    # EXACT production options — camera_manager.py:73,101-108
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|threads;2"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    if not cap.isOpened():
        raise SystemExit("[probe] open failed (check credentials/channel)")

    ok, f0 = cap.read()
    if not ok:
        raise SystemExit("[probe] first read failed")
    print(f"[P0] connected  wall={wall()}  shape={f0.shape}")
    save("p0_connect", f0)

    t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < 5.0:
        ok, _ = cap.read(); n += bool(ok)
    cam_fps = max(n / 5.0, 1.0)
    print(f"[P1] unthrottled: {n} frames / 5s  -> camera delivers {cam_fps:.1f} fps")

    interval = 1.0 / args.grab_fps
    print(f"[P2] PRODUCTION PATTERN: read @{args.grab_fps:g}fps for "
          f"{args.throttle_secs:g}s ...  start wall={wall()}")
    t0 = time.perf_counter(); n = 0; last = None
    while time.perf_counter() - t0 < args.throttle_secs:
        ok, fr = cap.read()
        if ok:
            n += 1; last = fr
        time.sleep(interval)
    print(f"[P2] done: {n} reads  end wall={wall()}")
    if last is not None:
        save("p2_after_throttle", last)
        print("      ^ compare this frame's burned-in OSD clock to the wall= "
              "time above — the difference IS the lag the engine lives in")

    print(f"[P3] burst drain (6s max) ...  start wall={wall()}")
    t0 = time.perf_counter(); times = []; fr = None
    while time.perf_counter() - t0 < 6.0:
        ok, f = cap.read()
        if not ok:
            break
        fr = f
        times.append(time.perf_counter() - t0)
    cap.release()

    if times:
        n = len(times)
        rate = n / times[-1] if times[-1] > 0 else 0.0
        fast = sum(1 for i in range(1, n) if times[i] - times[i - 1] < 0.5 / cam_fps)
        print(f"[P3] drained {n} frames in {times[-1]:.1f}s -> {rate:.0f}/s "
              f"(camera={cam_fps:.0f}/s)")
        print(f"[P3] frames faster than 2x camera rate (QUEUED): {fast} "
              f"~= {fast / cam_fps:.1f} SECONDS of backlog")
        if fr is not None:
            save("p3_after_drain", fr)
        print()
        if fast > cam_fps:  # more than ~1s worth of queued frames
            print("VERDICT: the RTSP layer IS queueing under the throttled read "
                  "pattern — the engine has been processing delayed video. "
                  "Fix: grab()-spin + throttled retrieve() (true latest-frame), "
                  "or read unthrottled and drop after decode.")
        else:
            print("VERDICT: no meaningful backlog on this camera with this "
                  "pattern — the RTSP layer is serving near-live frames.")
    else:
        print("[P3] no frames drained (stream stalled?)")


if __name__ == "__main__":
    main()
