"""Closed-loop wheel speed control from the wheel encoders.

The DB21 motors do not move below roughly 0.2-0.3 duty on the foam mat, so open-loop (v, omega) commands at
lane-following speeds either stall or lurch. Each wheel gets feed-forward (kinematics gain) + a static-friction
offset + PI on the encoder-measured speed. Stall protection cuts a wheel that is driven hard but does not turn.
"""
from collections import deque

DEFAULTS = {
    "kff": None,           # duty per (m/s); default gain / (radius * k) from the kinematics calibration
    "u_static": 0.15,      # duty added in the direction of motion to break static friction
    "s_static": 0.01,      # m/s: below this target the wheel is commanded to stop (no creeping)
    "kp": 1.2, "ki": 3.0, "i_max": 0.25,
    "window": 0.15,        # s of encoder history used for the speed estimate
    "stall_u": 0.45, "stall_t": 1.5,
    "limit": 1.0,
}


class WheelPI:
    def __init__(self, cfg=None, kin=None, res=135, radius=0.0318, log=print):
        c = dict(DEFAULTS, **{k: v for k, v in (cfg or {}).items() if v is not None})
        if not c["kff"]:
            k = kin or {}
            c["kff"] = k.get("gain", 1.0) / (k.get("radius", radius) * k.get("k", 27.0))
        self.c = c
        self.m_per_tick = 2 * 3.141592653589793 * radius / res
        self.hist = {"l": deque(), "r": deque()}
        self.i = {"l": 0.0, "r": 0.0}
        self.speed = {"l": 0.0, "r": 0.0}
        self.stall_since = {"l": None, "r": None}
        self.stalled = False
        self.log = log

    def update_ticks(self, side, ticks, t):
        h = self.hist[side]
        h.append((t, ticks))
        while len(h) > 2 and t - h[1][0] > self.c["window"]:
            h.popleft()
        if len(h) >= 2 and h[-1][0] > h[0][0]:
            self.speed[side] = (h[-1][1] - h[0][1]) * self.m_per_tick / (h[-1][0] - h[0][0])

    def measured(self, now, stale=0.3):
        """Wheel speeds (m/s); 0 if the encoders went quiet (robot stopped)."""
        out = {}
        for s in ("l", "r"):
            h = self.hist[s]
            out[s] = self.speed[s] if h and now - h[-1][0] < stale else 0.0
        return out

    def step(self, s_l, s_r, now, dt):
        """Target wheel speeds (m/s) -> duty (u_l, u_r)."""
        c = self.c
        meas = self.measured(now)
        out = {}
        for side, s in (("l", s_l), ("r", s_r)):
            if abs(s) < c["s_static"]:
                self.i[side] = 0.0
                self.stall_since[side] = None
                out[side] = 0.0
                continue
            sg = 1.0 if s > 0 else -1.0
            err = s - meas[side]
            self.i[side] = max(-c["i_max"], min(c["i_max"], self.i[side] + c["ki"] * err * dt))
            u = sg * c["u_static"] + c["kff"] * s + c["kp"] * err + self.i[side]
            if u * sg < 0:
                u = 0.0                                       # never drive against the wanted direction
            u = max(-c["limit"], min(c["limit"], u))
            # stall protection: pushing hard, wheel not turning
            if abs(u) >= c["stall_u"] and abs(meas[side]) < 0.01:
                self.stall_since[side] = self.stall_since[side] or now
                if now - self.stall_since[side] > c["stall_t"]:
                    if not self.stalled:
                        self.log("MOTOR STALL on %s wheel: output cut. Check that the wheel can turn / the encoder works." % side)
                    self.stalled = True
            else:
                self.stall_since[side] = None
            out[side] = 0.0 if self.stalled else u
        return out["l"], out["r"]

    def reset_stall(self):
        self.stalled = False
        self.stall_since = {"l": None, "r": None}
        self.i = {"l": 0.0, "r": 0.0}
