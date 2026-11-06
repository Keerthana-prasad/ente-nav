"""Obstacle safety for every autonomous mode.

Fuses (a) YOLO detections ground-projected into the current lane corridor and (b) the front time-of-flight
range. Distances are measured from the robot's FRONT bumper. Slow inside slow_dist, hard stop inside stop_dist,
resume only after clear_time seconds without anything in range. A source that has gone stale counts as an
obstacle (never as clear). Every stop and resume is logged with its reason.
"""
import math

DEFAULTS = {
    "slow_dist": 0.50, "stop_dist": 0.25, "clear_time": 1.5,
    "front_offset": 0.08,         # robot origin (axle) -> front bumper, metres
    "corridor_margin": 0.03,      # added to half the lane width
    "yolo_stale": 1.0, "tof_stale": 0.5,
    "v_crawl": 0.05,              # speed cap right before the stop zone
}


class Safety:
    def __init__(self, cfg=None, log=print):
        self.c = dict(DEFAULTS, **(cfg or {}))
        self.log = log
        self.yolo = {"on": False, "t": 0.0, "dist": None, "s0": None, "period": None}   # nearest in-lane detection
        self.tof = {"on": False, "t": 0.0, "dist": None, "max": 2.0}
        self.stopped = False
        self.reason = ""
        self.clear_since = None
        self.nearest = None
        self.sources = {}
        self.stops = 0

    # ---------- inputs ----------
    def set_yolo(self, points, t, lane=None, lane_w=0.23, corridor=0.14, s0=None):
        """points: (forward, left) floor points (robot origin) of detections, already class-filtered.
        t: capture time of the frame; s0: odometer reading (m) at capture, used to age the distance."""
        if self.yolo["on"] and t > self.yolo["t"]:
            dtp = t - self.yolo["t"]
            self.yolo["period"] = dtp if self.yolo["period"] is None else 0.7 * self.yolo["period"] + 0.3 * dtp
        self.yolo["on"] = True
        self.yolo["s0"] = s0
        best = None
        for fwd, left in points:
            if fwd <= 0:
                continue
            if lane is not None and lane.get("conf", 0) >= 0.35:
                phi, d = lane["phi"], lane["d"]
                lat = fwd * math.sin(phi) + left * math.cos(phi) + d       # lateral in the lane frame
                inside = abs(lat) < lane_w / 2 + self.c["corridor_margin"]
            else:
                inside = abs(left) < corridor
            if inside:
                fd = fwd - self.c["front_offset"]
                best = fd if best is None else min(best, fd)
        self.yolo["t"], self.yolo["dist"] = t, best

    def set_tof(self, rng, t, max_range=None, min_range=0.0):
        """sensor_msgs/Range. Beyond max_range (the sensor's 'no target' value) means clear."""
        self.tof["on"] = True
        self.tof["t"] = t
        mx = max_range if max_range and max_range > 0 else self.tof["max"]
        if rng is None or not math.isfinite(rng) or rng > mx or rng <= min_range:
            self.tof["dist"] = None
        else:
            self.tof["dist"] = rng

    # ---------- decision ----------
    def update(self, now, odo_s=None):
        """Returns (v_cap or None, hard_stop: bool). v_cap None = no limit. odo_s: odometer (m) to age YOLO distances."""
        c = self.c
        cand, src = [], {}
        y = self.yolo
        # a slow detector is stale only after missing ~2.5 of its own cycles (never less than yolo_stale)
        y_stale = max(c["yolo_stale"], 4.0 * y["period"]) if y["period"] else max(c["yolo_stale"], 6.0)
        y_dist = y["dist"]
        if y_dist is not None and odo_s is not None and y["s0"] is not None:
            y_dist = y_dist - max(0.0, odo_s - y["s0"])          # we drove towards it since that frame
        for name, s, stale, dist in (("yolo", y, y_stale, y_dist), ("tof", self.tof, c["tof_stale"], self.tof["dist"])):
            if not s["on"]:
                src[name] = "off"
                continue
            age = now - s["t"]
            if age > stale:
                src[name] = "STALE %.1fs" % age
                cand.append((0.0, "%s stale (%.1f s old)" % (name, age)))
            elif dist is not None:
                src[name] = "%.2f m" % dist
                cand.append((dist, "%s obstacle at %.2f m" % (name, dist)))
            else:
                src[name] = "clear"
        self.sources = src
        near = min(cand) if cand else None
        self.nearest = near[0] if near else None
        in_slow = near is not None and near[0] < c["slow_dist"]
        in_stop = near is not None and near[0] < c["stop_dist"]

        if in_stop:
            if not self.stopped:
                self.stops += 1
                self.log("SAFETY STOP #%d: %s" % (self.stops, near[1]))
            self.stopped, self.reason, self.clear_since = True, near[1], None
            return 0.0, True
        if self.stopped:
            if in_slow:                                    # still something close: stay stopped
                self.clear_since = None
                self.reason = near[1]
                return 0.0, True
            self.clear_since = self.clear_since or now
            if now - self.clear_since < c["clear_time"]:
                return 0.0, True
            self.stopped, self.clear_since = False, None
            self.log("SAFETY resume: clear for %.1f s (was: %s)" % (c["clear_time"], self.reason))
            self.reason = ""
        if in_slow:
            span = c["slow_dist"] - c["stop_dist"]
            frac = (near[0] - c["stop_dist"]) / span if span > 0 else 0.0
            self.reason = "slow: " + near[1]
            return c["v_crawl"] + frac * 0.10, False        # 0.05 m/s at the stop line .. 0.15 m/s at slow_dist
        self.reason = ""
        return None, False

    def state(self):
        return {"stopped": self.stopped, "reason": self.reason, "nearest": self.nearest,
                "sources": self.sources, "stops": self.stops}
