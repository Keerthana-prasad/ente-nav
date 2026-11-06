"""Lane-following controller: PID on (d, phi) -> omega, speed shaping, drift recovery, line-loss handling.

Conventions (from lanes.py): d > 0 robot is left of the right-lane centre, phi > 0 robot points left of the lane.
omega > 0 turns left. So omega = -(kd*d + ki*int(d) + kphi*phi) + v*kappa (curvature feed-forward).
"""
import math

DEFAULTS = {
    "v_nom": 0.15,           # m/s default lane speed
    "v_min": 0.05,
    "kd": 8.0, "kphi": 3.0, "kd_rate": 0.2,
    "ki": 0.5,               # integral on the combined steering error (cancels wheel trim / slope)
    "i_w_max": 0.35,         # anti-windup: max |omega| the integral may contribute (rad/s)
    "w_max": 3.0,
    "phi_max": 0.30,         # rad: never aim more than ~17 deg across the lane when correcting
    "kappa_ff": 0.8,         # fraction of measured curvature fed forward
    "curve_w": 2.5,          # v is scaled by 1/(1+|omega|/curve_w)
    "conf_ok": 0.45,         # below this the speed is reduced proportionally
    "drift_d": 0.05, "drift_phi": 0.35, "drift_gain": 1.8, "drift_v": 0.7,
    "yellow_guard": 0.045,   # d beyond which we never steer further left (robot edge near the yellow line)
    "lost_conf": 0.25,       # "low confidence"
    "slow_after": 0.5, "stop_after": 1.5, "manual_after": 8.0,
    "reacq_w": 0.6,          # rad/s while sweeping for the lines
    "reacq_conf": 0.45, "reacq_hold": 0.3,
}


def clip(x, lo, hi):
    return max(lo, min(hi, x))


class LaneController:
    def __init__(self, cfg=None):
        self.c = dict(DEFAULTS, **(cfg or {}))
        self.reset()

    def reset(self):
        self.i = 0.0
        self.prev_d = None
        self.low_since = None
        self.reacq_start = None
        self.reacq_phase = 0.0
        self.reacq_dir = 1.0
        self.good_since = None
        self.state = "FOLLOW"
        self.last_good = (0.0, 0.0)

    def step(self, lane, dt, now, v_nom=None):
        """lane: dict with d, phi, conf, kappa. Returns (v, w). self.state is FOLLOW | DRIFT | SLOW | REACQUIRE | LOST."""
        c = self.c
        v_nom = c["v_nom"] if v_nom is None else v_nom
        d, phi, conf = lane["d"], lane["phi"], lane["conf"]
        kappa = lane.get("kappa", 0.0) or 0.0

        # ---- line-loss timeline ----
        if conf < c["lost_conf"]:
            self.low_since = self.low_since if self.low_since is not None else now
            self.good_since = None
        else:
            self.good_since = self.good_since if self.good_since is not None else now
            self.last_good = (d, phi)
        low_for = now - self.low_since if self.low_since is not None else 0.0

        if self.state in ("REACQUIRE", "LOST"):
            if conf >= c["reacq_conf"] and self.good_since is not None and now - self.good_since >= c["reacq_hold"]:
                self.state, self.low_since, self.reacq_start = "FOLLOW", None, None
                self.i = 0.0
            elif self.state == "LOST":
                return 0.0, 0.0
            else:
                return self._reacquire(now, dt)
        elif low_for > c["stop_after"]:
            self.state = "REACQUIRE"
            self.reacq_start, self.reacq_phase = now, 0.0
            # sweep first toward where the lane probably is: if we were left of centre, look right
            self.reacq_dir = -1.0 if self.last_good[0] > 0 or self.last_good[1] > 0 else 1.0
            return 0.0, 0.0
        if conf < c["lost_conf"]:
            # lines faint (e.g. inside an intersection): don't steer on a stale estimate, roll straight & slow
            self.i *= 0.95
            self.state = "SLOW" if low_for > c["slow_after"] else "FOLLOW"
            return max(c["v_min"], 0.5 * v_nom) if v_nom > 0 else 0.0, 0.0
        elif self.low_since is not None:
            self.low_since = None

        # ---- PID ----
        drift = abs(d) > c["drift_d"] or abs(phi) > c["drift_phi"]
        g = c["drift_gain"] if drift else 1.0            # drifting: steer back to the centre more steeply
        # cascade: offset -> wanted heading (capped so the lines stay in view) -> heading error
        phi_ref = clip(-g * c["kd"] * d / c["kphi"], -c["phi_max"], c["phi_max"])
        e = c["kphi"] * (phi - phi_ref)
        if conf >= c["lost_conf"] and c["ki"] > 0:
            lim = c["i_w_max"] / c["ki"]
            self.i = clip(self.i + e * dt, -lim, lim)
        rate = 0.0 if self.prev_d is None or dt <= 0 else (d - self.prev_d) / dt
        self.prev_d = d
        w = -e - c["ki"] * self.i - c["kd_rate"] * rate

        # ---- speed ----
        v = v_nom
        v *= 1.0 / (1.0 + abs(w) / c["curve_w"])
        if conf < c["conf_ok"]:
            v *= clip(conf / c["conf_ok"], 0.3, 1.0)
        if drift:
            v *= c["drift_v"]
        if low_for > c["slow_after"]:
            v *= 0.5
        v = max(c["v_min"], v) if v_nom > 0 else 0.0
        w += c["kappa_ff"] * v * kappa

        # ---- never cross the yellow line ----
        if d > c["yellow_guard"] and w > 0 and phi >= -0.1:
            w = min(w, -0.5 * c["kd"] * (d - c["yellow_guard"]))
        w = clip(w, -c["w_max"], c["w_max"])
        self.state = "SLOW" if low_for > c["slow_after"] else ("DRIFT" if drift else "FOLLOW")
        return v, w

    def _reacquire(self, now, dt):
        """Stop and sweep slowly left/right with growing amplitude; LOST after manual_after seconds."""
        c = self.c
        if now - self.reacq_start > c["manual_after"]:
            self.state = "LOST"
            return 0.0, 0.0
        # amplitude (rad) grows 30 -> 60 -> 90 deg; phase integrates the commanded rate
        self.reacq_phase += self.reacq_dir * c["reacq_w"] * dt
        amp = math.radians(min(90, 30 * (1 + int((now - self.reacq_start) / 2.5))))
        if abs(self.reacq_phase) >= amp:
            self.reacq_phase = math.copysign(amp, self.reacq_phase)
            self.reacq_dir = -self.reacq_dir
        return 0.0, self.reacq_dir * c["reacq_w"]
