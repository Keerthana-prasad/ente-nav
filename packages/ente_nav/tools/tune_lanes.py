#!/usr/bin/env python3
"""Offline lane-perception check on saved frames, using the SAME code and config as the robot.
usage: tune_lanes.py FRAMES_DIR OUT_DIR [--set lanes.white.s=[0,80] ...]
Writes annotated images (camera view + masks + bird's-eye) and prints per-frame d/phi/conf/stop."""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "include"))
from ente_nav.lanes import LanePerception          # noqa: E402
from ente_nav.perception import GroundProjector    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("frames")
ap.add_argument("out")
ap.add_argument("--set", action="append", default=[], help="override, e.g. lanes.white.s=[0,80]")
a = ap.parse_args()
cfg = yaml.safe_load(open(os.path.join(HERE, "..", "config", "ente_nav.yaml")))
for kv in a.set:
    k, v = kv.split("=", 1)
    node, *path = k.split(".")
    d = cfg.setdefault(node, {}) if path else cfg
    for p in path[:-1]:
        d = d.setdefault(p, {})
    d[path[-1] if path else node] = yaml.safe_load(v)
cam = json.load(open(os.path.join(a.frames, "camera.json")))
proj = GroundProjector("none", *cam["pinhole"])
if cam.get("H"):
    proj.H = np.array(cam["H"]).reshape(3, 3)
    proj.Hinv = np.linalg.inv(proj.H)
proj.D = cam["D"]
proj.scale = float(cfg.get("ground_scale", 1.0))
geom = {k: cfg.get(k) for k in ("lane_width", "white_tape_width", "yellow_tape_width") if cfg.get(k)}
os.makedirs(a.out, exist_ok=True)
seps = []
for fn in sorted(glob.glob(os.path.join(a.frames, "*.jpg"))):
    img = cv2.imread(fn)
    lp = LanePerception(cfg.get("lanes", {}), proj, geom)       # fresh per frame: no temporal help
    for _ in range(3):                                          # let the filter settle on a still frame
        st = lp.process(img, cam["K"], cam["wh"], 0, 0)
    vis = lp.draw(img.copy(), cam["K"])
    m = lp.dbg["masks"]
    strip = np.hstack([cv2.cvtColor(cv2.resize(m[k], (img.shape[1] // 3, img.shape[0] // 3 * m[k].shape[0] // m[k].shape[0])), cv2.COLOR_GRAY2BGR) for k in ("white", "yellow", "red")])
    strip = cv2.resize(strip, (img.shape[1], strip.shape[0] * img.shape[1] // strip.shape[1]))
    for i, k in enumerate(("white", "yellow", "red")):
        cv2.putText(strip, k, (8 + i * img.shape[1] // 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(a.out, os.path.basename(fn)), np.vstack([vis, strip]))
    f = lp.dbg
    if f["fy"] and f["fw"]:            # yellow-white centre separation at x=0.25 m, for checking ground_scale
        x = 0.25
        seps.append((f["fy"]["b"] + x * np.tan(f["fy"]["ang"])) - (f["fw"]["b"] + x * np.tan(f["fw"]["ang"])))
    print("%s  d=%+.3f phi=%+5.1fdeg conf=%.2f src=%-12s Y=%3d W=%3d R=%3d stop=%s exits=%s" % (
        os.path.basename(fn), st["d"], np.degrees(st["phi"]), st["conf"], st["src"],
        len(f["Y"]), len(f["W"]), len(f["R"]), st["stop_dist"], st["exits"]))
if seps:
    exp = lp.lane_w + (lp.yw + lp.ww) / 2
    med = float(np.median(seps))
    print("yellow-white separation: median %.3f m over %d frames, expected %.3f -> ground_scale x %.3f" % (med, len(seps), exp, exp / med))
