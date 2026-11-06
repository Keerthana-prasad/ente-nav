"""Lane perception for a Duckietown-style mat (right-hand traffic).

HSV masks for the cream-white edge, the mustard-yellow dashed centre line and the salmon-red stop lines
are computed on the lower part of a down-scaled frame. Mask pixels are projected to the floor with the
GroundProjector, robust lines are fitted, and the result is
    d    lateral offset (m) of the robot from the RIGHT-lane centre, + = robot is left of centre
    phi  heading error (rad), + = robot points left of the lane direction
    conf 0..1
Uses the yellow line, the white line or both. A small complementary filter predicts d/phi with odometry
between frames. Also reports the distance to a red stop line across our lane and, near a stop line, which
exits (left / straight / right) look open. Nothing about the track layout is assumed.
"""
import math
import time

import cv2
import numpy as np

DEFAULTS = {
    "scale": 0.25, "roi_top": 0.45, "roi_bottom": 1.0, "open_k": 3,
    "white": {"h": [0, 180], "s": [0, 70], "v": [150, 255]},
    "yellow": {"h": [14, 36], "s": [80, 255], "v": [90, 255]},
    "red": {"h1": [0, 10], "h2": [165, 180], "s": [70, 255], "v": [90, 255]},
    "yellow_min_b": 0,         # >0: yellow must also have Lab b* (yellowness) >= this; separates mustard from cream
    "white_rel_v": 0,          # >0: white V must also exceed the ROI median V by this much (adapts to lighting)
    "white_tophat": 0,         # >0: white V minus local background (top-hat) must exceed this (rejects broad glare)
    "white_tophat_k": 15,      # top-hat kernel (px at processing scale); must be wider than the tape
    "max_pts": 600,            # per colour, per frame
    "x_min": 0.06, "x_max": 0.65, "y_max": 0.6,     # metres, robot frame
    "min_pts": 12, "min_span": 0.06, "inlier": 0.03,
    "gate": 0.07,              # metres around the predicted line once the lane is locked
    "consist_tol": 0.06,       # max disagreement between yellow- and white-based d
    "alpha": 0.6,              # filter gain at confidence 1
    "lock_conf": 0.35,
    "red_min_pts": 10, "stop_min_width": 0.07, "stop_max_x": 0.6, "stop_hold": 1.2,
    "exit_min_pts": 8,
}
LANE_DEFAULTS = {"lane_width": 0.23, "white_tape_width": 0.048, "yellow_tape_width": 0.025}


def _merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def _in(hsv, h, s, v):
    return cv2.inRange(hsv, (int(h[0]), int(s[0]), int(v[0])), (int(h[1]), int(s[1]), int(v[1])))


class LanePerception:
    def __init__(self, cfg=None, projector=None, geom=None):
        self.c = _merge(DEFAULTS, cfg)
        g = dict(LANE_DEFAULTS, **(geom or {}))
        self.lane_w = float(g["lane_width"])
        self.off_y = self.lane_w / 2 + g["yellow_tape_width"] / 2      # lane centre -> yellow tape centre
        self.off_w = self.lane_w / 2 + g["white_tape_width"] / 2       # lane centre -> white tape centre
        self.yw, self.ww = g["yellow_tape_width"], g["white_tape_width"]
        self.proj = projector
        self.d = self.phi = self.kappa = 0.0
        self.conf = 0.0
        self.t = None
        self.stop = None
        self.stop_t = 0.0
        self.stop_hits = 0
        self.exits = None
        self.src = "none"
        self.dbg = {}

    # ---------- image -> masks ----------
    def masks(self, bgr):
        c, s = self.c, float(self.c["scale"])
        small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h = small.shape[0]
        r0, r1 = int(h * c["roi_top"]), int(h * c["roi_bottom"])
        hsv = cv2.cvtColor(small[r0:r1], cv2.COLOR_BGR2HSV)
        W, Y, R = c["white"], c["yellow"], c["red"]
        mw = _in(hsv, W["h"], W["s"], W["v"])
        V = hsv[:, :, 2]
        if c["white_rel_v"] > 0:
            mw &= ((V.astype(np.int16) >= np.median(V) + c["white_rel_v"]) * 255).astype(np.uint8)
        if c["white_tophat"] > 0:
            k = int(c["white_tophat_k"]) | 1
            th = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            mw &= ((th >= c["white_tophat"]) * 255).astype(np.uint8)
        my = _in(hsv, Y["h"], Y["s"], Y["v"])
        if c["yellow_min_b"] > 0:
            b = cv2.cvtColor(small[r0:r1], cv2.COLOR_BGR2Lab)[:, :, 2]
            my &= ((b >= 128 + c["yellow_min_b"]) * 255).astype(np.uint8)
        mr = _in(hsv, R["h1"], R["s"], R["v"]) | _in(hsv, R["h2"], R["s"], R["v"])
        mw &= ~(my | mr)
        if c["open_k"]:
            ker = np.ones((int(c["open_k"]),) * 2, np.uint8)
            mw, my, mr = (cv2.morphologyEx(m, cv2.MORPH_OPEN, ker) for m in (mw, my, mr))
        return {"white": mw, "yellow": my, "red": mr}, r0, s

    def _ground(self, mask, r0, s, K, wh):
        ys, xs = np.nonzero(mask)
        n = int(self.c["max_pts"])
        if len(xs) > n:
            i = np.linspace(0, len(xs) - 1, n).astype(int)
            ys, xs = ys[i], xs[i]
        if len(xs) == 0:
            return np.zeros((0, 2))
        uv = np.stack([(xs + 0.5) / s, (ys + r0 + 0.5) / s], 1)
        g = self.proj.project_many(uv, K, wh)
        c = self.c
        ok = np.isfinite(g[:, 0]) & (g[:, 0] >= c["x_min"]) & (g[:, 0] <= c["x_max"]) & (np.abs(g[:, 1]) <= c["y_max"])
        return g[ok]

    # ---------- geometry ----------
    def _fit(self, P, tape_w=0.03):
        """RANSAC line through floor points (keeps the biggest consistent stripe, ignores other tapes),
        then least squares on its inliers. Returns angle/intercept/quality or None."""
        c = self.c
        if len(P) < c["min_pts"] or np.ptp(P[:, 0]) < c["min_span"]:
            return None
        tol = max(c["inlier"], tape_w * 0.7)
        rng = np.random.RandomState(len(P))
        best, best_n = None, 0
        for _ in range(40):
            i, j = rng.randint(0, len(P), 2)
            dx, dy = P[j] - P[i]
            L = math.hypot(dx, dy)
            if L < 0.04 or abs(dx) < 0.5 * L:              # too short, or runs across us (>60 deg)
                continue
            res = np.abs((P[:, 0] - P[i, 0]) * dy - (P[:, 1] - P[i, 1]) * dx) / L
            inl = res < tol
            n = int(inl.sum())
            if n > best_n:
                best, best_n = inl, n
        if best is None or best_n < c["min_pts"]:
            return None
        Q = P[best]
        if np.ptp(Q[:, 0]) < c["min_span"]:
            return None
        vx, vy, x0, y0 = cv2.fitLine(Q.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        if vx < 0:
            vx, vy = -vx, -vy
        if vx < 0.5:
            return None
        res = np.abs((Q[:, 0] - x0) * vy - (Q[:, 1] - y0) * vx)
        ang = math.atan2(vy, vx)
        b = y0 - x0 * vy / vx
        rms = float(np.sqrt(np.mean(res ** 2)))
        span = float(np.ptp(Q[:, 0]))
        q = min(1.0, len(Q) / 40.0) * min(1.0, span / 0.2) * math.exp(-max(0.0, rms - tape_w / 3.46) / 0.01)
        return {"ang": ang, "b": float(b), "rms": rms, "n": int(len(Q)), "span": span, "q": q, "pts": Q}

    def _line_y(self, off, x, d=None, phi=None):
        """Lateral position (robot frame) at forward x of a line `off` m left of the lane centre."""
        d = self.d if d is None else d
        phi = self.phi if phi is None else phi
        cp = max(math.cos(phi), 0.3)
        return (off - d) / cp - x * math.tan(phi)

    def to_lane(self, P):
        """Robot-frame points -> lane frame (X along the lane from the robot, Y left of the lane centre)."""
        cph, sph = math.cos(self.phi), math.sin(self.phi)
        return np.stack([P[:, 0] * cph - P[:, 1] * sph, P[:, 0] * sph + P[:, 1] * cph + self.d], 1)

    def reset(self):
        """Forget the lane estimate (after a turn through an intersection): next frames re-acquire from scratch."""
        self.d = self.phi = self.kappa = 0.0
        self.conf = 0.0
        self.stop, self.stop_hits, self.exits = None, 0, None

    # ---------- filter ----------
    def predict(self, v, w, dt):
        if dt <= 0 or dt > 0.5:
            return
        self.d += v * math.sin(self.phi) * dt
        self.phi += (w - v * self.kappa) * dt
        if self.stop is not None:
            self.stop -= v * dt

    # ---------- main ----------
    def process(self, bgr, K, wh, v=0.0, w=0.0, now=None):
        now = time.time() if now is None else now
        if self.t is not None:
            self.predict(v, w, now - self.t)
        self.t = now
        c = self.c
        m, r0, s = self.masks(bgr)
        Yp, Wp, Rp = (self._ground(m[k], r0, s, K, wh) for k in ("yellow", "white", "red"))
        locked = self.conf >= c["lock_conf"]

        if locked:
            g = c["gate"]                                    # wider further ahead, where curves bend away
            Ys = Yp[np.abs(Yp[:, 1] - self._line_y(self.off_y, Yp[:, 0])) < g + 0.25 * Yp[:, 0]] if len(Yp) else Yp
            Ws = Wp[np.abs(Wp[:, 1] - self._line_y(-self.off_w, Wp[:, 0])) < g + 0.25 * Wp[:, 0]] if len(Wp) else Wp
            fy = self._fit(Ys, self.yw)
            fw = self._fit(Ws, self.ww)
        else:
            fy = self._fit(Yp, self.yw)
            if fy is not None and len(Wp):
                yl = fy["b"] + Wp[:, 0] * math.tan(fy["ang"])
                Ws = Wp[(Wp[:, 1] < yl - 0.5 * self.lane_w) & (Wp[:, 1] > yl - self.lane_w - 0.15)]
            elif len(Wp):
                right = Wp[Wp[:, 1] < 0.05]
                Ws = right[right[:, 1] > np.percentile(right[:, 1], 85) - 0.08] if len(right) else right
            else:
                Ws = Wp
            fw = self._fit(Ws, self.ww)

        meas = []
        if fy:
            cp = math.cos(fy["ang"])
            meas.append(("yellow", self.off_y - fy["b"] * cp, -fy["ang"], fy["q"]))
        if fw:
            cp = math.cos(fw["ang"])
            meas.append(("white", -self.off_w - fw["b"] * cp, -fw["ang"], fw["q"]))
        penalty = 1.0
        if len(meas) == 2 and abs(meas[0][1] - meas[1][1]) > c["consist_tol"]:
            ref = self.d if locked else None
            if ref is not None:
                meas = [min(meas, key=lambda q: abs(q[1] - ref))]
            else:
                meas = [max(meas, key=lambda q: q[3])]
            penalty = 0.6
        if meas:
            wsum = sum(q[3] for q in meas) + 1e-9
            dm = sum(q[1] * q[3] for q in meas) / wsum
            pm = sum(q[2] * q[3] for q in meas) / wsum
            cm = (1 - np.prod([1 - q[3] for q in meas])) * penalty
            self.src = "+".join(q[0] for q in meas)
        else:
            dm = pm = None
            cm = 0.0
            self.src = "none"

        if dm is not None and locked and (abs(dm) > 0.2 or (len(meas) == 1 and (abs(pm - self.phi) > 0.3 or abs(dm - self.d) > 0.08))):
            dm = pm = None                                   # one line implying a jump: probably the wrong tape
            cm = 0.0
            self.src = "rejected"
        if dm is not None:
            a = c["alpha"] * cm
            if not locked:
                a = max(a, min(0.9, cm))                     # re-acquire: snap to the measurement
            elif abs(dm - self.d) > 2 * c["gate"]:
                a *= 0.3                                     # suspicious jump
            self.d += a * (dm - self.d)
            self.phi += a * (pm - self.phi)
        self.conf = 0.6 * self.conf + 0.4 * cm

        best = max([f for f in (fy, fw) if f], key=lambda f: f["span"], default=None)
        if best is not None and best["span"] > 0.25 and best["n"] > 30:
            P = best["pts"]
            cc = np.polyfit(P[:, 0], P[:, 1], 2)
            k = 2 * cc[0] / (1 + cc[1] ** 2) ** 1.5
            self.kappa += 0.3 * (float(np.clip(k, -8, 8)) - self.kappa)
        elif best is None:
            self.kappa *= 0.9

        self._stopline(Rp, now)
        self.exits = self._exits(Wp, Yp) if self.stop is not None and self.stop < 0.4 else None
        self.dbg = {"masks": m, "r0": r0, "s": s, "fy": fy, "fw": fw, "Y": Yp, "W": Wp, "R": Rp, "meas_conf": cm}
        return self.state()

    def _stopline(self, Rp, now):
        c = self.c
        found = None
        if len(Rp) >= c["red_min_pts"]:
            L = self.to_lane(Rp) if self.conf >= c["lock_conf"] else Rp
            sel = (np.abs(L[:, 1]) < self.lane_w / 2 + 0.05) & (L[:, 0] > 0.02) & (L[:, 0] < c["stop_max_x"])
            L = L[sel]
            if len(L) >= c["red_min_pts"]:
                near = float(np.percentile(L[:, 0], 10))
                band = L[L[:, 0] < near + 0.12]
                if np.ptp(band[:, 1]) >= c["stop_min_width"] and band[:, 1].min() < -0.02 and band[:, 1].max() > 0.02:
                    found = near
        if found is not None:
            self.stop_hits += 1
            if self.stop_hits >= 2:
                self.stop = found if self.stop is None or abs(self.stop - found) > 0.1 else self.stop + 0.5 * (found - self.stop)
                self.stop_t = now
        else:
            self.stop_hits = 0
            if self.stop is not None and (now - self.stop_t > c["stop_hold"] or self.stop < -0.15):
                self.stop = None

    def _exits(self, Wp, Yp):
        """Evidence for which roads leave this stop line. A white edge continuing past the stop line on the right
        means no right turn; white running across ahead means no straight; white beyond the far lane on the left
        means no left turn. Returns point counts per direction (blocking evidence) for the caller to vote on."""
        if len(Wp) == 0:
            return {"left": 0, "straight": 0, "right": 0}
        L = self.to_lane(Wp)
        sd, lw = self.stop, self.lane_w
        X, Y = L[:, 0], L[:, 1]
        right = ((X > sd + 0.06) & (X < sd + 0.30) & (np.abs(Y + self.off_w) < 0.05)).sum()
        sel = (X > sd + 0.25) & (X < sd + 0.9) & (np.abs(Y) < lw / 2)
        straight = sel.sum() if sel.sum() and np.ptp(Y[sel]) > 0.08 else 0
        far_left = self.off_y + self.yw / 2 + lw + self.ww / 2
        left = ((X > sd + 0.10) & (X < sd + 0.55) & (np.abs(Y - far_left) < 0.06)).sum()
        return {"left": int(left), "straight": int(straight), "right": int(right)}

    def state(self):
        return {"d": round(self.d, 4), "phi": round(self.phi, 4), "conf": round(float(self.conf), 3),
                "kappa": round(self.kappa, 3), "src": self.src,
                "stop_dist": None if self.stop is None else round(self.stop, 3),
                "exits": self.exits}

    # ---------- debug drawing ----------
    def draw(self, img, K):
        dbg = self.dbg
        if not dbg:
            return img
        s, r0 = dbg["s"], dbg["r0"]
        h, w = img.shape[:2]
        y0 = int(r0 / s)
        tint = {"white": (255, 255, 0), "yellow": (0, 220, 255), "red": (255, 0, 255)}
        roi = img[y0:]
        for k, col in tint.items():
            mk = cv2.resize(dbg["masks"][k], (w, roi.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            roi[mk] = (0.45 * roi[mk] + 0.55 * np.array(col)).astype(np.uint8)
        cv2.line(img, (0, y0), (w, y0), (120, 120, 120), 1)
        xs = np.linspace(self.c["x_min"], self.c["x_max"], 14)

        def poly(fwd, left, col, th=2):
            px = self.proj.to_pixel(fwd, left, K)
            px = px[np.isfinite(px[:, 0])]
            if len(px) > 1:
                cv2.polylines(img, [px.astype(np.int32)], False, col, th, cv2.LINE_AA)

        for key, col in (("fy", (0, 200, 255)), ("fw", (255, 255, 255))):
            f = dbg.get(key)
            if f:
                poly(xs, f["b"] + xs * math.tan(f["ang"]), col, 3)
        if self.conf > 0.15:
            poly(xs, self._line_y(0.0, xs), (80, 255, 80), 2)                          # lane centre
            for off in (self.lane_w / 2, -self.lane_w / 2):
                poly(xs, self._line_y(off, xs), (80, 180, 80), 1)
        if self.stop is not None and self.stop > 0.05:
            ys = np.linspace(-self.lane_w / 2, self.lane_w / 2, 6)
            cph, sph = math.cos(self.phi), math.sin(self.phi)
            X, Yl = np.full_like(ys, self.stop), ys - self.d                              # lane frame -> robot frame
            poly(X * cph + Yl * sph, -X * sph + Yl * cph, (60, 60, 255), 4)
        txt = "d %+.3f m  phi %+.1f deg  conf %.2f  [%s]" % (self.d, math.degrees(self.phi), self.conf, self.src)
        if self.stop is not None:
            txt += "  STOP %.2f m" % self.stop
        cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if self.exits:
            e = self.exits
            cv2.putText(img, "exit evidence (blocked pts) L%d S%d R%d" % (e["left"], e["straight"], e["right"]),
                        (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 255), 1, cv2.LINE_AA)
        # bird's-eye inset: 0.7 m ahead, +-0.35 m sideways
        S, sc = 140, 200.0
        ins = np.full((S, S, 3), 40, np.uint8)
        for k, col in (("W", (255, 255, 255)), ("Y", (0, 220, 255)), ("R", (60, 60, 255))):
            P = dbg[k]
            if len(P):
                u = (S / 2 - P[:, 1] * sc).astype(int)
                v = (S - 4 - P[:, 0] * sc).astype(int)
                ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
                ins[v[ok], u[ok]] = col
        cv2.drawMarker(ins, (S // 2, S - 4), (80, 255, 80), cv2.MARKER_TRIANGLE_UP, 8, 2)
        img[h - S - 6:h - 6, w - S - 6:w - 6] = ins
        return img
