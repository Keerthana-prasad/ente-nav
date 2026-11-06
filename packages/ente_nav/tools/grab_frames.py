#!/usr/bin/env python3
"""Save raw camera frames from the robot for offline lane-threshold tuning.
usage: grab_frames.py [--host entebot209.local] [--n 30] [--every 1.0] [--out ~/ente_frames]"""
import argparse
import json
import os
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="entebot209.local")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--every", type=float, default=1.0)
ap.add_argument("--out", default=os.path.expanduser("~/ente_frames"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
base = "http://%s:8090" % a.host
json.dump(json.loads(urllib.request.urlopen(base + "/api/camera", timeout=5).read()),
          open(os.path.join(a.out, "camera.json"), "w"), indent=1)
start = len([f for f in os.listdir(a.out) if f.endswith(".jpg")])
for i in range(a.n):
    jpg = urllib.request.urlopen(base + "/api/snapshot", timeout=5).read()
    fn = os.path.join(a.out, "frame_%03d.jpg" % (start + i))
    open(fn, "wb").write(jpg)
    print("saved", fn, flush=True)
    time.sleep(a.every)
