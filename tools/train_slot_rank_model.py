"""Train the slot-identity ranker (LightGBM lambdarank) on the OCR-verified gallery.

WHERE THE LABELS COME FROM
    Every parked-pose crop in vehicle_images/gallery/<plate>/ was written by
    save_parked_reference(), which runs ONLY on an OCR-confirmed identity. So each such
    crop is a verified label: "this crop, taken by this camera, is this car." No manual
    annotation, and the set grows every time a car parks and is read.

WHAT IT LEARNS
    ReID puts the right car in the top-5 ~98% of the time but first only ~88% (cold).
    The ranker's job is that gap: reorder the shortlist using signals ReID's cosine does
    not see — a same-camera parked pose, colour agreement, gallery depth, crop quality,
    and whether the car is already locked into another slot.

HOW IT IS VALIDATED
    LEAVE-PLATES-OUT. With only ~34 distinct cars a model can trivially memorise them,
    and a random row split would look brilliant and be worthless. Every fold holds out
    whole cars; the script ABORTS if any plate leaks across the split. It also refuses to
    write a model that does not beat plain ReID top-1 on held-out cars.

    Two regimes are reported because they are different problems:
      WARM — the car has parked at this camera before (a same-view ref exists).
      COLD — it never has. Every same-camera ref of the true car is withheld.

Usage:
    python tools/train_slot_rank_model.py                 # train + report, writes model
    python tools/train_slot_rank_model.py --dry-run       # report only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.matching.slot_rank_features import (
    FEATURE_NAMES,
    CandidateSignals,
    QuerySignals,
    build_features,
)
from src.reid_matcher.reid_burst import sharpness_score
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_models import VehicleSession

GALLERY = "vehicle_images/gallery"
MODEL_OUT = "models/slot_rank_lgbm.txt"


def load_gallery(reg, gate_cams):
    """plate -> [(camera, vector, crop)] for every reference on disk."""
    gal: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
    for meta_path in sorted(glob.glob(f"{GALLERY}/*/meta.json")):
        d = os.path.dirname(meta_path)
        meta = json.load(open(meta_path))
        refs = []
        for r in meta.get("refs", []):
            img = cv2.imread(os.path.join(d, r["crop"]))
            if img is None or img.size == 0:
                continue
            vec = reg.reid_matcher.extract_feature(img)
            if vec is None:
                continue
            refs.append((r.get("camera", ""), vec, img))
        if refs:
            gal[meta["plate"]] = refs
    return gal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    import lightgbm as lgb

    cfg = load_config("config.yaml")
    reg = VehicleRegistry(matching_config=cfg.matching)
    gate = set(cfg.matching.ground_truth_cameras)
    color = reg.match_decision._color_classifier
    vtype = reg.match_decision._type_classifier

    print("loading gallery + embedding (this is the slow part)...")
    gal = load_gallery(reg, gate)
    plates = sorted(gal)
    print(f"  {len(plates)} cars, {sum(len(v) for v in gal.values())} refs")

    # Classify every reference crop ONCE — colour/type are per-image, not per-pair.
    cls_cache: Dict[Tuple[str, int], Tuple[Tuple[str, float], Tuple[str, float]]] = {}
    for p, refs in gal.items():
        for i, (_c, _v, img) in enumerate(refs):
            cls_cache[(p, i)] = (color.predict(img), vtype.predict(img))

    # Sessions the registry will score against; refs are swapped per query below.
    sess: Dict[str, VehicleSession] = {}
    for p in plates:
        s = VehicleSession(session_id=str(uuid.uuid4()), plate=p)
        reg._sessions[s.session_id] = s
        sess[p] = s

    def install(true_plate: str, hide_cams=(), hide_idx=None, hide_cams_for_all=()):
        """Swap the gallery each session is scored against.

        ``hide_cams_for_all`` strips a camera's references from EVERY car, not just the
        true one. That distinction is load-bearing: an earlier version hid the query
        camera's refs only for the true car, which made ``warm == False`` a perfect
        giveaway (the true car was the only un-warm candidate) and produced a bogus 99.7%
        cold rank-1. Cold must mean "nobody has parked at this camera yet", or the model
        just learns the artifact.
        """
        for p, refs in gal.items():
            keep = [
                (c, v)
                for i, (c, v, _img) in enumerate(refs)
                if not (c in hide_cams_for_all)
                and not (p == true_plate and (c in hide_cams or i == hide_idx))
            ]
            s = sess[p]
            s.feature_vector = next((v for c, v in keep if c in gate), None)
            s.reference_feature_vectors = [v for _c, v in keep]
            s.reference_source_cameras = [c for c, _v in keep]

    # ---- build one group per parked-pose crop, in both regimes ------------------
    queries = [
        (p, i, c)
        for p, refs in gal.items()
        for i, (c, _v, _img) in enumerate(refs)
        if c not in gate
    ]
    print(f"  {len(queries)} OCR-verified parked-pose crops -> {len(queries)} groups per regime")

    def make_groups(regime: str):
        X, y, grp, meta = [], [], [], []
        for tp, qi, qcam in queries:
            if regime == "warm":
                # Leave-one-crop-out: the true car keeps its OTHER same-camera refs, and
                # so does everyone else. No artificial asymmetry.
                install(tp, hide_idx=qi)
            else:
                # Day-1 cold: NOBODY has a reference from this camera, true car included.
                install(tp, hide_cams_for_all={qcam})
            qcrop = gal[tp][qi][2]
            qvec = gal[tp][qi][1]
            kept, _rej = reg.reid_rank(qvec, slot_id=None, slot_camera=qcam, k=len(plates))
            if len(kept) < 2:
                continue
            qcol, qtyp = cls_cache[(tp, qi)]
            h, w = qcrop.shape[:2]
            q = QuerySignals(
                crop_sharpness=float(sharpness_score(qcrop)),
                crop_area=float(h * w),
                crop_aspect=float(w / max(h, 1)),
                active_candidates=len(kept),
            )
            cands = []
            for c in kept:
                # Count refs from the INSTALLED session, never from the raw gallery.
                # Reading `gal[...]` here leaked badly: it still counted the true car's
                # withheld same-camera refs, so `n_same_view_refs > 0` betrayed the answer
                # in the cold regime and the model leaned on it 30x harder than on the
                # ReID score. Production only ever sees the live session, so training must
                # too — this is the exact train/serve mismatch that silently kills rankers.
                s_cams = sess[c.plate].reference_source_cameras or []
                n_same = sum(1 for cc in s_cams if cc == qcam)
                # colour/type of the candidate: use its best available reference
                raw = gal[c.plate]
                same_raw = [i for i, (cc, _v, _i) in enumerate(raw) if cc == qcam]
                bi = same_raw[0] if (same_raw and n_same) else 0
                ccol, ctyp = cls_cache[(c.plate, bi)]
                cands.append(
                    CandidateSignals(
                        plate=c.plate,
                        reid_score=c.score,
                        same_view_score=c.same_view_score,
                        cross_view_score=c.cross_view_score,
                        warm=c.warm,
                        rank=c.rank,
                        n_refs=len(s_cams),
                        n_same_view_refs=n_same,
                        best_ref_is_gate=not c.warm,
                        color_match=(qcol[0] == ccol[0]) if (qcol[0] and ccol[0]) else None,
                        color_conf_query=float(qcol[1]),
                        color_conf_cand=float(ccol[1]),
                        type_match=(qtyp[0] == ctyp[0]) if (qtyp[0] and ctyp[0]) else None,
                        type_conf_query=float(qtyp[1]),
                        locked_elsewhere=False,  # not reconstructible offline; live-only
                    )
                )
            feats = build_features(cands, q)
            X.append(feats)
            y.append(np.array([1 if c.plate == tp else 0 for c in cands], dtype=np.int32))
            grp.append(len(cands))
            meta.append((tp, qcam))
        return X, y, grp, meta

    # Similarity/ranking features are ALLOWED to identify the car on their own — that is
    # literally their job, and reid_score alone is 98% right in the warm regime. The
    # guard below polices the AUXILIARY features: gallery bookkeeping, colour, type, crop
    # quality. If one of THOSE picks the winner by itself, the dataset is leaking.
    SIMILARITY_FEATURES = {
        "reid_score", "same_view_score", "cross_view_score",
        "margin_to_next", "margin_to_top1", "candidate_rank",
    }

    def assert_no_giveaway(X, y, regime):
        """No AUXILIARY feature may separate the true car from the field on its own.

        A ranker that scores ~100% is almost always reading an artifact of how the data
        was built. This caught a real one: hiding the query camera's refs for only the
        TRUE car made `warm == False` identify the answer outright and yielded a bogus
        99.7% cold rank-1. Fail loudly rather than ship that.
        """
        for j, name in enumerate(FEATURE_NAMES):
            if name in SIMILARITY_FEATURES:
                continue
            hit = 0
            for feats, lab in zip(X, y):
                col = np.nan_to_num(feats[:, j], nan=-1e9)
                t = int(np.argmax(lab))
                others = np.delete(col, t)
                if others.size == 0:
                    continue
                # STRICT separation only. Candidates arrive pre-sorted by ReID score, so
                # a tie-broken argsort would credit any weakly-informative binary feature
                # (e.g. `warm`) with ReID's own ordering and flag it as a leak. A feature
                # only gives the answer away if the true car's value beats EVERY other.
                hit += int(col[t] > others.max())
            frac = hit / max(len(X), 1)
            if frac > 0.97:
                print(f"\n  ABORT [{regime}]: feature {name!r} alone identifies the true "
                      f"car in {frac:.1%} of groups. That is a construction artifact, "
                      f"not a signal — the dataset is leaking.")
                raise SystemExit(1)

    results = {}
    for regime in ("warm", "cold"):
        print(f"\n=== {regime.upper()} ===")
        X, y, grp, meta = make_groups(regime)
        plates_of = [m[0] for m in meta]
        assert_no_giveaway(X, y, regime)

        # ---- leave-PLATES-out folds. A car must never be in train and test. -------
        uniq = sorted(set(plates_of))
        folds = defaultdict(list)
        for i, p in enumerate(uniq):
            folds[i % args.folds].append(p)

        base_r1 = base_r5 = mdl_r1 = mdl_r5 = n = 0
        for k in range(args.folds):
            test_plates = set(folds[k])
            tr = [i for i, p in enumerate(plates_of) if p not in test_plates]
            te = [i for i, p in enumerate(plates_of) if p in test_plates]
            if not tr or not te:
                continue
            leak = {plates_of[i] for i in tr} & {plates_of[i] for i in te}
            if leak:
                print(f"  LEAK: {leak}")
                return 1

            Xtr = np.vstack([X[i] for i in tr]); ytr = np.concatenate([y[i] for i in tr])
            gtr = [grp[i] for i in tr]
            ds = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=FEATURE_NAMES)
            params = dict(
                objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                num_leaves=15, max_depth=4, min_data_in_leaf=20,
                learning_rate=0.05, verbosity=-1, num_threads=1,
                label_gain=[0, 1],
            )
            booster = lgb.train(params, ds, num_boost_round=200)

            for i in te:
                s = booster.predict(X[i])
                order_m = np.argsort(-s)
                order_b = np.argsort(-X[i][:, 0])  # feature 0 = reid_score = the baseline
                truth = np.argmax(y[i])
                n += 1
                base_r1 += int(order_b[0] == truth); base_r5 += int(truth in order_b[:5])
                mdl_r1 += int(order_m[0] == truth); mdl_r5 += int(truth in order_m[:5])

        print(f"  held-out groups: {n}")
        print(f"  {'':<12}{'rank-1':>9}{'recall@5':>11}")
        print(f"  {'ReID alone':<12}{base_r1/n:>8.1%}{base_r5/n:>11.1%}")
        print(f"  {'+ ranker':<12}{mdl_r1/n:>8.1%}{mdl_r5/n:>11.1%}"
              f"   ({(mdl_r1-base_r1)/n:+.1%})")
        results[regime] = (base_r1 / n, mdl_r1 / n)

    # ---- ship only if it actually helps on unseen cars -------------------------
    cold_base, cold_model = results["cold"]
    if cold_model <= cold_base:
        print(f"\nREFUSING to write a model: cold rank-1 {cold_model:.1%} does not beat "
              f"plain ReID {cold_base:.1%} on held-out cars.")
        return 1

    if args.dry_run:
        print("\n--dry-run: not writing the model")
        return 0

    # Final model on ALL data (folds were only to measure honestly).
    X, y, grp, _m = make_groups("cold")
    Xw, yw, gw, _mw = make_groups("warm")
    ds = lgb.Dataset(np.vstack(X + Xw), label=np.concatenate(y + yw),
                     group=grp + gw, feature_name=FEATURE_NAMES)
    booster = lgb.train(
        dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1], num_leaves=15,
             max_depth=4, min_data_in_leaf=20, learning_rate=0.05, verbosity=-1,
             num_threads=1, label_gain=[0, 1]),
        ds, num_boost_round=200,
    )
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    booster.save_model(MODEL_OUT)
    print(f"\nwrote {MODEL_OUT}")
    imp = sorted(zip(FEATURE_NAMES, booster.feature_importance("gain")),
                 key=lambda kv: -kv[1])
    print("\ntop features by gain:")
    for name, g in imp[:10]:
        print(f"   {name:<22}{g:>12.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
