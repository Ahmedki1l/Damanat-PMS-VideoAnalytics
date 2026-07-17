"""
tools/review_labels.py — Keystroke accept/reject review of teacher-consensus
labels. Run this LOCALLY (it opens an OpenCV window). For each frame that has
flagged (single-teacher) boxes, it shows the accepted ground truth in GREEN and
the flagged candidates one at a time in RED; you accept (add to GT) or reject.

It reads the ``proposals/<stem>.json`` written by autolabel_teacher.py and, on
leaving each frame, rewrites ``labels/<stem>.txt`` = accepted boxes + the flagged
boxes you accepted. Re-running is safe (labels are rebuilt from proposals each
time, never double-added).

Keys
----
    a   accept current flagged box  (adds to GT)      A  accept ALL in frame
    r   reject current flagged box                    R  reject ALL in frame
    d   delete the nearest GREEN (accepted) box        (fixes a wrong accept)
    n / SPACE  save frame, go to next
    p          save frame, go to previous
    u   undo last action in this frame
    q   save current frame and quit
    ?   toggle help overlay

Usage
-----
    python tools/review_labels.py --root data/gold_val          # flagged frames only
    python tools/review_labels.py --root data/gold_val --all    # every frame
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

GREEN, RED, YELLOW, BLUE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 160, 0)


def load_frame_state(root: Path, stem: str) -> Optional[Dict]:
    prop = root / "proposals" / (stem + ".json")
    if not prop.exists():
        return None
    d = json.loads(prop.read_text(encoding="utf-8"))
    return {
        "accepted": [np.asarray(b, float) for b in d.get("accepted", [])],
        "flagged": [{"box": np.asarray(b, float), "status": "pending"}
                    for b in d.get("flagged", [])],
        "w": int(d["w"]), "h": int(d["h"]),
    }


def write_labels(root: Path, stem: str, state: Dict) -> None:
    w, h = state["w"], state["h"]
    boxes = list(state["accepted"]) + [f["box"] for f in state["flagged"]
                                       if f["status"] == "accept"]
    lines = []
    for b in boxes:
        cx = (b[0] + b[2]) / 2.0 / w
        cy = (b[1] + b[3]) / 2.0 / h
        bw = (b[2] - b[0]) / w
        bh = (b[3] - b[1]) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (root / "labels" / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")


def draw(im, state, cur_flag_idx, scale, show_help) -> np.ndarray:
    vis = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))

    def rect(b, color, t):
        cv2.rectangle(vis, (int(b[0] * scale), int(b[1] * scale)),
                      (int(b[2] * scale), int(b[3] * scale)), color, t)

    for b in state["accepted"]:
        rect(b, GREEN, 2)
    for i, f in enumerate(state["flagged"]):
        if f["status"] == "accept":
            rect(f["box"], GREEN, 2)
        elif f["status"] == "reject":
            continue
        else:  # pending
            rect(f["box"], RED if i == cur_flag_idx else YELLOW,
                 3 if i == cur_flag_idx else 2)
    n_pending = sum(1 for f in state["flagged"] if f["status"] == "pending")
    bar = f"GT(green)={len(state['accepted']) + sum(1 for f in state['flagged'] if f['status']=='accept')}  pending(red/yellow)={n_pending}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if show_help:
        lines = ["a/A accept  r/R reject  d delete-green  n next  p prev  u undo  q quit  ? help"]
        cv2.rectangle(vis, (0, vis.shape[0] - 26), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
        cv2.putText(vis, lines[0], (8, vis.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def nearest_accepted(state, x, y, scale) -> int:
    """Index into state['accepted'] of the box whose centre is nearest (x,y) in
    display coords; -1 if none."""
    best, bi = 1e18, -1
    for i, b in enumerate(state["accepted"]):
        cx, cy = (b[0] + b[2]) / 2 * scale, (b[1] + b[3]) / 2 * scale
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best:
            best, bi = d, i
    return bi


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="review_labels",
                                description="Accept/reject teacher-flagged boxes.")
    p.add_argument("--root", type=Path, required=True, help="Gold-set root.")
    p.add_argument("--all", action="store_true", help="Review every frame, not just flagged.")
    p.add_argument("--max-w", type=int, default=1600)
    p.add_argument("--max-h", type=int, default=900)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.root
    idx_path = root / "review_index.json"
    index = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else []
    if index:
        order = [r["image"] for r in index if args.all or r["flagged"] > 0]
    else:
        order = sorted(p.name for p in (root / "images").glob("*.jpg"))

    stems = [Path(n).stem for n in order]
    if not stems:
        print("Nothing to review (no flagged frames). Use --all to review everything.")
        return 0
    print(f"Reviewing {len(stems)} frame(s). Keys: a/r accept/reject, n next, q quit, ? help.")

    click = {"xy": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["xy"] = (x, y)

    win = "review_labels"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    i, show_help = 0, True
    while 0 <= i < len(stems):
        stem = stems[i]
        img_path = root / "images" / (order[i])
        im = cv2.imread(str(img_path))
        state = load_frame_state(root, stem)
        if im is None or state is None:
            i += 1
            continue
        scale = min(args.max_w / state["w"], args.max_h / state["h"], 1.0)
        pending = [k for k, f in enumerate(state["flagged"]) if f["status"] == "pending"]
        cur = pending[0] if pending else -1
        history: List[Tuple] = []

        while True:
            cv2.setWindowTitle(win, f"[{i+1}/{len(stems)}] {stem}")
            cv2.imshow(win, draw(im, state, cur, scale, show_help))
            key = cv2.waitKey(20) & 0xFF

            if click["xy"] is not None:
                # click selects nearest pending flagged as current
                x, y = click["xy"]; click["last"] = (x, y); click["xy"] = None
                pend = [k for k, f in enumerate(state["flagged"]) if f["status"] == "pending"]
                if pend:
                    cur = min(pend, key=lambda k: (
                        ((state["flagged"][k]["box"][0]+state["flagged"][k]["box"][2])/2*scale - x)**2 +
                        ((state["flagged"][k]["box"][1]+state["flagged"][k]["box"][3])/2*scale - y)**2))

            def advance_cur():
                pend = [k for k, f in enumerate(state["flagged"]) if f["status"] == "pending"]
                return pend[0] if pend else -1

            if key in (ord("a"), ord("r")) and cur >= 0:
                state["flagged"][cur]["status"] = "accept" if key == ord("a") else "reject"
                history.append((cur, "pending"))
                cur = advance_cur()
            elif key == ord("A"):
                for f in state["flagged"]:
                    if f["status"] == "pending":
                        f["status"] = "accept"
                cur = -1
            elif key == ord("R"):
                for f in state["flagged"]:
                    if f["status"] == "pending":
                        f["status"] = "reject"
                cur = -1
            elif key == ord("d") and state["accepted"]:
                # delete the GREEN box nearest the last mouse click (else most recent)
                if click.get("last"):
                    bi = nearest_accepted(state, click["last"][0], click["last"][1], scale)
                else:
                    bi = len(state["accepted"]) - 1
                if bi >= 0:
                    state["accepted"].pop(bi)
            elif key == ord("u") and history:
                k, st = history.pop(); state["flagged"][k]["status"] = st; cur = k
            elif key in (ord("n"), ord(" ")):
                write_labels(root, stem, state); i += 1; break
            elif key == ord("p"):
                write_labels(root, stem, state); i = max(0, i - 1); break
            elif key == ord("?"):
                show_help = not show_help
            elif key == ord("q"):
                write_labels(root, stem, state); cv2.destroyAllWindows()
                print(f"Saved through frame {i+1}/{len(stems)}. Bye.")
                return 0

    cv2.destroyAllWindows()
    print("Review complete — labels/ updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
