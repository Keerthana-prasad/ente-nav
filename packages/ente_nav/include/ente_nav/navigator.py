"""Task-queue navigator: goto / spin / forward / explore, with obstacle wait-and-detour,
relocalisation spin when pose uncertainty is high, and off-course detection.
Lane mode adds: lane (follow until a red line / a distance), stopwait (1 s at the red line, collecting exit
evidence), decide (ask the mission which way), arc (encoder+gyro turn through the intersection)."""
import math
import random
from collections import deque
from ente_nav.ekf_slam import wrap
from ente_nav.lane_control import LaneController

TURN_SIGN = {"left": 1, "straight": 0, "right": -1}


def clip(x, lo, hi):
    return max(lo, min(hi, x))


class Navigator:
    def __init__(self, c, lane_cfg=None, geom=None, turn_cfg=None):
        self.c = c                      # dict of tunables (see config/ente_nav.yaml)
        self.lc = LaneController(lane_cfg)
        g = dict({"lane_width": 0.23, "white_tape_width": 0.048, "yellow_tape_width": 0.025}, **(geom or {}))
        lw, ww, yw = g["lane_width"], g["white_tape_width"], g["yellow_tape_width"]
        # intersection geometry from the lane width (Duckietown-style tiles); every value can be overridden
        self.turn = dict({"stop_target": 0.12, "stop_wait": 1.0, "v_arc": 0.12, "k_th": 3.0, "pre_extra": 0.0,
                          "r_right": lw / 2 + ww, "r_left": 1.5 * lw + ww + yw,
                          "straight_len": 2 * lw + 2 * ww + yw + 0.05, "post": 0.03,
                          "stop_ignore": 0.30, "approach": 0.35, "exit_block_pts": 30, "timeout": 12.0},
                         **{k: v for k, v in (turn_cfg or {}).items() if v not in (None, 0, 0.0) or k == "pre_extra"})
        self.hooks = {}                 # stopline(votes) / decide() -> dict, turn_failed(), arrived(msg), lane_reset()
        self.lane_state = ""
        self.message = ""
        self.tasks = deque()
        self.cur = None
        self.state = "IDLE"
        self.goal_name = ""
        self.speed_scale = 1.0
        self.blocked_since = None
        self.reloc_since = None
        self.alert = ""
        self.lost = False
        self.xte = 0.0
        self.dist_to_goal = None

    # ---------- task API ----------
    def cancel(self):
        self.tasks.clear()
        self.cur = None
        self.state = "IDLE"
        self.lane_state = ""
        self.lc.reset()
        self.dist_to_goal = None
        self.blocked_since = self.reloc_since = None
        self.alert = ""

    def goto(self, x, y, name=""):
        self.cancel()
        self.lost = False
        self.goal_name = name
        self.tasks.append({"t": "goto", "x": x, "y": y, "name": name})

    def spin(self, rad):
        self.cancel()
        self.tasks.append({"t": "spin", "rad": rad})

    def forward(self, d, pose):
        x, y, th = pose
        self.goto(x + d * math.cos(th), y + d * math.sin(th), "ahead")

    def explore(self, n=6):
        self.cancel()
        self.lost = False
        for _ in range(n):
            self.tasks.append({"t": "spin", "rad": 2 * math.pi})
            self.tasks.append({"t": "fwd", "d": 0.5})
        self.goal_name = "exploring"

    def lane_follow(self, stop_s=None, decide_first=False):
        """Follow the lane; at red lines stop, then let the mission choose the turn. stop_s: stop after this many
        metres along the current edge (a saved place)."""
        self.cancel()
        self.lost = False
        if decide_first:
            self.tasks.append({"t": "decide"})
        else:
            self.tasks.append({"t": "lane", "stop_s": stop_s, "ignore_until": 0.0})

    def _hook(self, name, *a):
        f = self.hooks.get(name)
        return f(*a) if f else None

    def _after_decision(self, res, lane, edge_dist):
        """Queue the arc + next lane leg for a mission decision, or finish."""
        self.cur = None
        if res and res.get("continue"):
            self.tasks.appendleft({"t": "lane", "stop_s": res.get("stop_s"), "ignore_until": 0.0})
            return
        if not res or "turn" not in res:
            self.message = (res or {}).get("done", "")
            self.state = "ARRIVED" if (res or {}).get("arrived") else "IDLE"
            self.lane_state = ""
            self.tasks.clear()
            if self.message:
                self._hook("arrived", self.message)
            return
        sd = lane.get("stop_dist") if lane else None
        pre = (sd if sd is not None and 0.0 < sd < 0.3 else self.turn["stop_target"]) + self.turn["pre_extra"]
        self.tasks.appendleft({"t": "lane", "stop_s": res.get("stop_s"), "ignore_until": self.turn["stop_ignore"],
                               "after_turn": True})
        self.tasks.appendleft({"t": "arc", "turn": res["turn"], "pre": max(0.0, pre)})
        self.message = "turning %s" % res["turn"]

    # ---------- internals ----------
    def _lane_step(self, t, pose, now, lane, dt, edge_dist):
        T = self.turn
        if t["t"] == "decide":
            self._after_decision(self._hook("decide"), lane, edge_dist)
            return 0.0, 0.0

        if t["t"] == "stopwait":
            self.state = self.lane_state = "STOP_LINE"
            t.setdefault("t0", now)
            if lane and lane.get("exits"):
                t["votes"].append(lane["exits"])
            if now - t["t0"] < T["stop_wait"]:
                return 0.0, 0.0
            opened = None
            if t["votes"]:
                opened = {k: sum(v[k] for v in t["votes"]) / len(t["votes"]) < T["exit_block_pts"]
                          for k in ("left", "straight", "right")}
            self._after_decision(self._hook("stopline", opened), lane, edge_dist)
            return 0.0, 0.0

        if t["t"] == "arc":
            return self._arc(t, pose, now)

        # ---- lane following ----
        if lane is None:
            self.state = self.lane_state = "NO_LANE_DATA"
            return 0.0, 0.0
        if t.get("stop_s") is not None and edge_dist >= t["stop_s"]:
            self.cur = None
            self.tasks.clear()
            self.state, self.lane_state = "ARRIVED", ""
            self._hook("arrived", "")
            return 0.0, 0.0
        v_nom = self.lc.c["v_nom"] * self.speed_scale
        sd = lane.get("stop_dist")
        if sd is not None and edge_dist >= t["ignore_until"]:
            if sd <= T["stop_target"]:
                self.cur = {"t": "stopwait", "votes": []}
                self.lc.reset()
                return 0.0, 0.0
            if sd < T["approach"]:                     # ease in so we stop on the mark
                v_nom = max(0.06, v_nom * min(1.0, (sd - T["stop_target"]) / (T["approach"] - T["stop_target"]) + 0.3))
        if t.get("stop_s") is not None:
            rem = t["stop_s"] - edge_dist
            if rem < 0.2:
                v_nom = max(0.05, v_nom * (rem / 0.2 + 0.2))
        v, w = self.lc.step(lane, dt, now, v_nom)
        self.state, self.lane_state = "LANE_" + self.lc.state, self.lc.state
        if self.lc.state == "LOST":
            if t.get("after_turn") and edge_dist < 0.6:
                self._hook("turn_failed")
            self.cancel()
            self.lost = True
            self.alert = "LANE LOST: no lane lines found for 8 s. Switched to manual."
            return 0.0, 0.0
        return v, w

    def _arc(self, t, pose, now):
        """Encoder+gyro manoeuvre through the intersection: straight `pre` metres to the intersection entry,
        then a quarter circle (left/right) or a straight crossing, holding the planned heading."""
        T = self.turn
        x, y, th = pose
        if "x0" not in t:
            t.update(x0=x, y0=y, th0=th, px=x, py=y, s=0.0, t0=now)
            sg = TURN_SIGN[t["turn"]]
            t["sg"] = sg
            t["R"] = T["r_left"] if sg > 0 else T["r_right"]
            t["L"] = T["straight_len"] if sg == 0 else 0.0
        t["s"] += math.hypot(x - t["px"], y - t["py"])
        t["px"], t["py"] = x, y
        self.state = self.lane_state = "TURN_" + t["turn"].upper()
        v = T["v_arc"] * min(1.0, self.speed_scale + 0.2)
        k = T["k_th"]
        if now - t["t0"] > T["timeout"]:
            return self._arc_done()
        if t["s"] < t["pre"]:
            return v, clip(k * wrap(t["th0"] - th), -2.0, 2.0)
        if t["sg"] == 0:
            if t["s"] >= t["pre"] + t["L"]:
                return self._arc_done()
            return v, clip(k * wrap(t["th0"] - th), -2.0, 2.0)
        sg, R = t["sg"], t["R"]
        turned = wrap(th - t["th0"]) * sg
        if turned >= math.pi / 2 - 0.06:
            t.setdefault("s_end", t["s"])
            if t["s"] - t["s_end"] >= T["post"]:
                return self._arc_done()
            return v, clip(k * wrap(t["th0"] + sg * math.pi / 2 - th), -2.0, 2.0)
        th_ref = t["th0"] + sg * min((t["s"] - t["pre"]) / R, math.pi / 2)
        return v, clip(sg * v / R + k * wrap(th_ref - th), -3.0, 3.0)

    def _arc_done(self):
        self.cur = None
        self.lc.reset()
        self._hook("lane_reset")
        return 0.0, 0.0

    def _begin(self, t, pose):
        x, y, th = pose
        if t["t"] == "fwd":
            t.update(t="goto", x=x + t["d"] * math.cos(th), y=y + t["d"] * math.sin(th), name="")
        if t["t"] == "spin":
            t["acc"], t["last"] = 0.0, th
        if t["t"] == "goto":
            t["sx"], t["sy"] = x, y

    def _done(self, arrived=False):
        self.cur = None
        self.blocked_since = self.reloc_since = None
        if not self.tasks:
            self.state = "ARRIVED" if arrived and self.goal_name not in ("", "ahead") else "IDLE"
            self.dist_to_goal = None

    def step(self, pose, now, obst, unc, tag_age, lane=None, dt=1 / 30.0, edge_dist=0.0):
        """pose=(x,y,th). obst: distance (m) to nearest obstacle in path or None. lane: lanes.py state.
        edge_dist: metres driven since leaving the last stop line. Returns (v, w)."""
        c = self.c
        if self.cur is None:
            if not self.tasks:
                return 0.0, 0.0
            self.cur = self.tasks.popleft()
            self._begin(self.cur, pose)
        t = self.cur
        x, y, th = pose
        sc = self.speed_scale

        if t["t"] in ("lane", "stopwait", "decide", "arc"):
            return self._lane_step(t, pose, now, lane, dt, edge_dist)

        if t["t"] == "spin":
            d = wrap(th - t["last"])
            t["last"] = th
            t["acc"] += d
            if abs(t["acc"]) >= abs(t["rad"]) - 0.05:
                self._done()
                return 0.0, 0.0
            self.state = "SPINNING"
            return 0.0, math.copysign(c["spin_w"] * sc, t["rad"])

        # ---- goto ----
        dx, dy = t["x"] - x, t["y"] - y
        dist = math.hypot(dx, dy)
        self.dist_to_goal = dist
        if dist < c["goal_tol"]:
            self._done(arrived=True)
            return 0.0, 0.0

        # obstacle: stop, wait, then detour
        if obst is not None and obst < c["stop_dist"]:
            self.blocked_since = self.blocked_since or now
            if now - self.blocked_since > c["obstacle_wait"]:
                sgn = random.choice([-1, 1])
                self.tasks.appendleft(t)
                self.tasks.appendleft({"t": "fwd", "d": 0.35})
                self.tasks.appendleft({"t": "spin", "rad": sgn * math.pi / 2})
                self.cur = None
                self.blocked_since = None
                self.state = "AVOIDING"
                return 0.0, 0.0
            self.state = "OBSTACLE"
            return 0.0, 0.0
        self.blocked_since = None

        # localisation health: spin to find a tag; give up -> LOST (operator takes over)
        if unc > c["unc_max"] and tag_age > c["reloc_after"]:
            self.reloc_since = self.reloc_since or now
            if now - self.reloc_since > c["reloc_timeout"]:
                self.cancel()
                self.lost = True
                self.alert = "LOST: no landmarks found, switch to manual"
                return 0.0, 0.0
            self.state = "RELOCALIZING"
            return 0.0, c["spin_w"] * sc
        self.reloc_since = None

        # off-course (cross-track error from the start->goal line)
        L = math.hypot(t["x"] - t["sx"], t["y"] - t["sy"])
        if L > 0.2:
            self.xte = abs((t["x"] - t["sx"]) * (t["sy"] - y) - (t["sx"] - x) * (t["y"] - t["sy"])) / L
        else:
            self.xte = 0.0
        if self.xte > c["offcourse"]:
            self.alert = "OFF_COURSE: %.2f m from planned line" % self.xte
        elif self.alert.startswith("OFF_COURSE"):
            self.alert = ""

        err = wrap(math.atan2(dy, dx) - th)
        w = clip(c["k_heading"] * err, -c["w_max"], c["w_max"])
        if abs(err) > c["turn_in_place"]:
            v = 0.0
        else:
            v = c["v_max"] * sc * max(0.0, math.cos(err)) * min(1.0, dist / c["slow_radius"] + 0.3)
        self.state = "NAVIGATING"
        return v, w
