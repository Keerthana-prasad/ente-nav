#!/usr/bin/env python3
"""Checks on the real robot that ente_nav actually drives it. Run on the laptop while
`dts devel run -H entebot209 -L ente-nav` is running:

    python3 verify_motion.py                # default host entebot209.local
    python3 verify_motion.py 10.26.0.4      # or by IP

Give the robot ~40 cm of clear floor in front. It drives forward 20 cm, turns left 90 deg, then stops.
PASS needs: wheel encoder ticks changed AND the pose moved the expected amount."""
import json
import math
import sys
import time
import urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "entebot209.local"
PORT = sys.argv[2] if len(sys.argv) > 2 else "8090"
BASE = "http://%s:%s" % (HOST, PORT)


def get():
    return json.loads(urllib.request.urlopen(BASE + "/api/state", timeout=3).read())


def cmd(o):
    req = urllib.request.Request(BASE + "/api/cmd", json.dumps(o).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def wait_idle(timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = get()
        if s["nav_state"] in ("IDLE", "ARRIVED") and time.time() - t0 > 1.0:
            return s
        time.sleep(0.2)
    return get()


def main():
    try:
        s = get()
    except Exception as e:
        print("FAIL: dashboard not reachable at %s (%s). Is the node running?" % (BASE, e))
        return 1
    print("connected to %s." % BASE, "mode=%s motor_out=%s ticks=%s gyro=%s" % (s["mode"], s.get("motor_out"), s.get("ticks"), s["slam"]["gyro"]))
    t = s.get("ticks") or {}
    if t.get("l") is None or t.get("r") is None:
        print("FAIL: no wheel encoder ticks received. Check `rostopic echo /entebot209/left_wheel_encoder_driver_node/tick`.")
        return 1

    ok = True
    try:
        # 1) forward 20 cm
        p0, k0 = s["pose"], s["ticks"]
        print("> drive forward 20 cm :", cmd({"type": "nlp", "text": "drive forward 20 cm"}).get("reply"))
        s = wait_idle(15)
        p1, k1 = s["pose"], s["ticks"]
        moved = math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
        dl, dr = k1["l"] - k0["l"], k1["r"] - k0["r"]
        good = abs(dl) > 20 and abs(dr) > 20 and 0.12 < moved < 0.30
        ok &= good
        print("  %s  moved %.3f m, ticks L%+d R%+d, state=%s" % ("PASS" if good else "FAIL", moved, dl, dr, s["nav_state"]))

        # 2) turn left 90
        th0, k0 = s["pose"]["th"], s["ticks"]
        print("> turn left 90 :", cmd({"type": "nlp", "text": "turn left 90"}).get("reply"))
        s = wait_idle(15)
        dth = math.degrees((s["pose"]["th"] - th0 + math.pi) % (2 * math.pi) - math.pi)
        dl, dr = s["ticks"]["l"] - k0["l"], s["ticks"]["r"] - k0["r"]
        good = 60 < dth < 120 and dl * dr < 0
        ok &= good
        print("  %s  turned %+.0f deg, ticks L%+d R%+d" % ("PASS" if good else "FAIL", dth, dl, dr))
    finally:
        cmd({"type": "nlp", "text": "stop"})
        cmd({"type": "mode", "mode": "MANUAL"})

    print("RESULT:", "PASS - robot drives under ente_nav" if ok else "FAIL - see lines above")
    if not ok:
        print("hints: ticks 0 -> motors got nothing (look for 'motor output ->' in the node log);"
              " ticks moved but distance off -> check wheel_radius/encoder_resolution in ente_nav.yaml")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
