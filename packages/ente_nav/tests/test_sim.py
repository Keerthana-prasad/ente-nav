#!/usr/bin/env python3
"""Laptop simulation tests for the lane controller, intersection manoeuvres, the topological planner and
the mapping mission. No ROS needed: python3 tests/test_sim.py"""
import math
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
from ente_nav.lane_control import LaneController          # noqa: E402
from ente_nav.navigator import Navigator                   # noqa: E402
from ente_nav.safety import Safety                         # noqa: E402
from ente_nav.motor import WheelPI                         # noqa: E402
from ente_nav.topo_map import TopoMap, Mission, TURNS      # noqa: E402

LW = 0.23
ROBOT_HALF = 0.065          # DB21 half width: lines are "touched" when |d| > LW/2 - ROBOT_HALF + tape slack
DT = 1 / 30.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class LoopTrack:
    """Right-lane centre line of a rounded rectangle, driven counter-clockwise (right-hand traffic)."""

    def __init__(self, a=1.2, b=0.8, r=0.30, n=4000):
        pts = []
        for i in range(n):
            s = i / n
            pts.append(self._at(s, a, b, r))
        self.P = np.array(pts)
        seg = np.diff(np.vstack([self.P, self.P[:1]]), axis=0)
        self.th = np.arctan2(seg[:, 1], seg[:, 0])
        self.L = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
        dth = np.array([wrap(x) for x in np.diff(np.concatenate([self.th, self.th[:1]]))])
        self.kappa = dth / (self.L / n)

    @staticmethod
    def _at(s, a, b, r):
        # perimeter parametrisation of a rounded rectangle centred at the origin
        sx, sy = a - 2 * r, b - 2 * r
        per = 2 * sx + 2 * sy + 2 * math.pi * r
        u = s * per
        corners = [(sx / 2, -sy / 2, -math.pi / 2), (sx / 2, sy / 2, 0), (-sx / 2, sy / 2, math.pi / 2), (-sx / 2, -sy / 2, math.pi)]
        segs = [("L", (-sx / 2, -b / 2), (sx / 2, -b / 2)), ("A", 0), ("L", (a / 2, -sy / 2), (a / 2, sy / 2)), ("A", 1),
                ("L", (sx / 2, b / 2), (-sx / 2, b / 2)), ("A", 2), ("L", (-a / 2, sy / 2), (-a / 2, -sy / 2)), ("A", 3)]
        for sg in segs:
            if sg[0] == "L":
                (x0, y0), (x1, y1) = sg[1], sg[2]
                ln = math.hypot(x1 - x0, y1 - y0)
                if u <= ln:
                    f = u / ln
                    return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
                u -= ln
            else:
                cx, cy, a0 = corners[sg[1]]
                ln = math.pi / 2 * r
                if u <= ln:
                    ang = a0 + u / r
                    return (cx + r * math.cos(ang), cy + r * math.sin(ang))
                u -= ln
        return (-sx / 2, -b / 2)

    def lane(self, x, y, th):
        i = int(np.argmin((self.P[:, 0] - x) ** 2 + (self.P[:, 1] - y) ** 2))
        tth = self.th[i]
        dx, dy = x - self.P[i, 0], y - self.P[i, 1]
        d = -math.sin(tth) * dx + math.cos(tth) * dy             # + = left of the centre line
        return d, wrap(th - tth), float(self.kappa[i]), i


class Robot:
    """Diff drive with actuator bias (trim), speed scale error, slew limit and gyro-fused heading."""

    def __init__(self, x, y, th, w_bias=0.25, v_scale=0.9, seed=1):
        self.x, self.y, self.th = x, y, th
        self.v = self.w = 0.0
        self.w_bias, self.v_scale = w_bias, v_scale
        self.rng = random.Random(seed)
        self.odo = [x, y, th]           # what the robot believes (encoders + gyro, with small drift)

    def step(self, v, w, dt=DT):
        dv, dw = 0.6 * dt, 6.0 * dt
        self.v += max(-dv, min(dv, v - self.v))
        self.w += max(-dw, min(dw, w - self.w))
        va = self.v * self.v_scale * (1 + self.rng.gauss(0, 0.03))
        wa = self.w + (self.w_bias if abs(self.v) > 1e-3 else 0.0) + self.rng.gauss(0, 0.05)
        self.th = wrap(self.th + wa * dt)
        self.x += va * math.cos(self.th) * dt
        self.y += va * math.sin(self.th) * dt
        ov = self.v * (1 + self.rng.gauss(0, 0.02))                  # odometry: slightly wrong distance
        self.odo[2] = wrap(self.odo[2] + (wa + 0.003) * dt)          # gyro drift
        self.odo[0] += ov * math.cos(self.odo[2]) * dt
        self.odo[1] += ov * math.sin(self.odo[2]) * dt
        return ov * dt


def noisy_lane(track, rob, rng, conf=0.8, sd=0.01, sp=0.03):
    d, phi, k, _ = track.lane(rob.x, rob.y, rob.th)
    return {"d": d + rng.gauss(0, sd), "phi": phi + rng.gauss(0, sp), "conf": conf, "kappa": k, "stop_dist": None, "exits": None}


class TestLaneController(unittest.TestCase):
    def run_loop(self, d0=0.0, phi0=0.0, laps=1.0, dropouts=True, seed=3):
        tr = LoopTrack()
        rng = random.Random(seed)
        x, y = tr.P[0]
        th = tr.th[0]
        rob = Robot(x - math.sin(th) * d0, y + math.cos(th) * d0, th + phi0, seed=seed)
        lc = LaneController({"v_nom": 0.15})
        t, dist, worst, worst_after = 0.0, 0.0, 0.0, 0.0
        while dist < laps * tr.L and t < 120:
            lane = noisy_lane(tr, rob, rng)
            if dropouts and int(t * 10) % 50 in (7, 8, 9):            # 0.3 s of no lines every 5 s
                lane["conf"] = 0.05
            v, w = lc.step(lane, DT, t)
            dist += abs(rob.step(v, w))
            d = abs(tr.lane(rob.x, rob.y, rob.th)[0])
            worst = max(worst, d)
            if t > 3.0:
                worst_after = max(worst_after, d)
            t += DT
        return dist, tr.L, worst, worst_after, t

    def test_full_loop_stays_in_lane(self):
        dist, L, worst, _, t = self.run_loop()
        self.assertGreaterEqual(dist, L)
        self.assertLess(worst, LW / 2 - ROBOT_HALF + 0.01, "touched a line: max |d| %.3f" % worst)
        self.assertLess(t, L / 0.15 * 2.2, "too slow")

    def test_drift_recovery(self):
        _, _, worst, worst_after, _ = self.run_loop(d0=0.07, phi0=0.35, laps=0.5)
        self.assertLess(worst_after, 0.04, "did not converge back to the lane centre: %.3f" % worst_after)

    def test_never_crosses_yellow(self):
        _, _, worst, _, _ = self.run_loop(d0=0.08, phi0=0.5, laps=0.3, seed=7)
        self.assertLess(worst, LW / 2, "crossed a lane line")

    def test_line_loss_timeline(self):
        lc = LaneController()
        good = {"d": 0.0, "phi": 0.0, "conf": 0.9, "kappa": 0.0}
        bad = dict(good, conf=0.0)
        t = 0.0
        for _ in range(30):
            lc.step(good, DT, t)
            t += DT
        v_ok, _ = lc.step(good, DT, t)
        states = {}
        t0 = t
        while t - t0 < 10.0:
            v, w = lc.step(bad, DT, t)
            states.setdefault(lc.state, (round(t - t0, 2), v, w))
            t += DT
        self.assertIn("SLOW", states)
        self.assertAlmostEqual(states["SLOW"][0], 0.5, delta=0.1)
        self.assertLess(states["SLOW"][1], v_ok)
        self.assertAlmostEqual(states["REACQUIRE"][0], 1.5, delta=0.1)
        self.assertEqual(states["REACQUIRE"][1], 0.0)
        self.assertAlmostEqual(states["LOST"][0], 9.5, delta=0.15)
        # lines come back during the sweep -> resumes
        lc = LaneController()
        t = 0.0
        for _ in range(int(2.5 / DT)):
            lc.step(bad, DT, t)
            t += DT
        self.assertEqual(lc.state, "REACQUIRE")
        for _ in range(int(0.6 / DT)):
            lc.step(good, DT, t)
            t += DT
        self.assertEqual(lc.state, "FOLLOW")


class TestIntersection(unittest.TestCase):
    def drive_turn(self, turn):
        nav = Navigator({"v_max": 0.18, "w_max": 2.5, "k_heading": 1.8, "turn_in_place": 0.7, "slow_radius": 0.4,
                         "goal_tol": 0.02, "stop_dist": 0.3, "obstacle_wait": 4, "unc_max": 0.35, "reloc_after": 3,
                         "reloc_timeout": 20, "offcourse": 0.3, "spin_w": 1.0})
        events = []
        nav.hooks = {"stopline": lambda opened: events.append(("stop", opened)) or {"turn": turn},
                     "lane_reset": lambda: events.append(("reset",))}
        rob = Robot(0.0, 0.0, 0.0, w_bias=0.1, seed=5)
        nav.lane_follow()
        t, edge = 0.0, 0.0
        line_x = 0.6                      # red line near edge 0.6 m ahead on a straight lane along +x
        trace = []
        while t < 25:
            sd = line_x - rob.x
            lane = {"d": rob.y, "phi": rob.th, "conf": 0.9, "kappa": 0.0,
                    "stop_dist": sd if -0.15 < sd < 0.6 else None, "exits": {"left": 0, "straight": 0, "right": 0}}
            v, w = nav.step(tuple(rob.odo), t, None, 0.0, 0.0, lane, DT, edge)
            edge += rob.step(v, w)
            trace.append((t, nav.state, rob.x, rob.y, rob.th))
            t += DT
            if ("reset",) in events:
                break
        return nav, events, rob, trace

    def test_stop_then_turns(self):
        for turn, dth in (("right", -math.pi / 2), ("left", math.pi / 2), ("straight", 0.0)):
            nav, ev, rob, tr = self.drive_turn(turn)
            stops = [r for r in tr if r[1] == "STOP_LINE"]
            self.assertTrue(stops, turn)
            wait = stops[-1][0] - stops[0][0]
            self.assertAlmostEqual(wait, 1.0, delta=0.1)
            self.assertAlmostEqual(stops[0][2], 0.6 - 0.12, delta=0.03, msg="stop position (%s)" % turn)
            self.assertEqual(ev[0][0], "stop")
            self.assertIn(("reset",), ev)
            self.assertLess(abs(wrap(rob.th - dth)), 0.2, "%s: final heading %.2f" % (turn, rob.th))
            lw, ww, yw = 0.23, 0.048, 0.025
            if turn == "right":     # exit lane centre: x = line + R_right, heading -y
                self.assertAlmostEqual(rob.x, 0.6 + lw / 2 + ww, delta=0.06)
            if turn == "left":
                self.assertAlmostEqual(rob.x, 0.6 + 1.5 * lw + ww + yw, delta=0.08)
                self.assertAlmostEqual(rob.y, 1.5 * lw + ww + yw, delta=0.08)


class TestSafety(unittest.TestCase):
    def test_stop_resume_and_stale(self):
        logs = []
        s = Safety({}, logs.append)
        s.set_tof(None, 0.0, 99.0)
        s.yolo["on"] = True
        lane = {"d": 0.0, "phi": 0.0, "conf": 0.9}
        s.set_yolo([(0.9, 0.0)], 0.0, lane)
        self.assertEqual(s.update(0.05), (None, False))
        s.set_yolo([(0.08 + 0.4, 0.0)], 0.1, lane)          # 0.40 m from the bumper -> slow
        cap, hard = s.update(0.15)
        self.assertFalse(hard)
        self.assertLess(cap, 0.15)
        s.set_yolo([(0.08 + 0.2, 0.0)], 0.2, lane)          # 0.20 m -> hard stop
        self.assertEqual(s.update(0.25), (0.0, True))
        s.set_yolo([(0.08 + 0.2, 0.30)], 0.3, lane)         # beside the lane -> ignored
        s.set_tof(None, 0.3, 99.0)
        self.assertTrue(s.update(0.35)[1])                  # still waiting for 1.5 s clear
        t = 0.35
        while t < 2.0:
            s.set_yolo([], t, lane)
            s.set_tof(150.0, t, 99.0)
            r = s.update(t)
            t += 0.05
        self.assertEqual(r, (None, False))
        self.assertTrue(any("SAFETY STOP" in m for m in logs) and any("resume" in m for m in logs))
        self.assertTrue(s.update(t + 1.2)[1], "stale sensors must count as an obstacle")
        s.set_tof(0.2, t + 1.2, 99.0)
        s.set_yolo([], t + 1.2, lane)
        self.assertTrue(s.update(t + 1.25)[1], "ToF obstacle at 0.20 m")


class TestSafetySlowYolo(unittest.TestCase):
    def test_aging_and_adaptive_stale(self):
        s = Safety({}, lambda m: None)
        lane = {"d": 0.0, "phi": 0.0, "conf": 0.9}
        for k in range(4):                                   # YOLO every 3 s (slow Jetson CPU)
            s.set_yolo([], 3.0 * k, lane, s0=0.0)
        self.assertAlmostEqual(s.yolo["period"], 3.0, places=3)
        self.assertFalse(s.update(9.0 + 10.0, 0.0)[1], "10 s old is not stale for a 3 s detector")
        self.assertTrue(s.update(9.0 + 13.0, 0.0)[1], "13 s old is stale -> obstacle")
        s2 = Safety({}, lambda m: None)
        s2.set_yolo([(0.08 + 0.60, 0.0)], 0.0, lane, s0=1.00)  # duck 0.60 m ahead of the bumper when captured
        s2.set_yolo([(0.08 + 0.60, 0.0)], 0.1, lane, s0=1.00)
        cap, hard = s2.update(0.5, 1.30)                     # drove 0.30 m since -> 0.30 m away: slow
        self.assertFalse(hard)
        self.assertIsNotNone(cap)
        self.assertTrue(s2.update(0.6, 1.40)[1])             # 0.20 m -> hard stop


class TestWheelPI(unittest.TestCase):
    """Motor with a 0.22 duty deadband and lag; encoders at 135 ticks/rev published at ~30 Hz."""

    def sim(self, target, t_end=2.0, dead=0.22, blocked=False):
        logs = []
        pi = WheelPI({}, {"gain": 2.0, "radius": 0.0318, "k": 27.0}, 135, 0.0318, logs.append)
        speed, pos, t, hist = 0.0, 0.0, 0.0, []
        while t < t_end:
            u, _ = pi.step(target, target, t, DT)
            want = 0.0 if blocked else (math.copysign(max(0.0, abs(u) - dead) / 2.33, u))
            speed += (want - speed) * DT / 0.08
            pos += speed * DT
            pi.update_ticks("l", int(pos / pi.m_per_tick), t)
            pi.update_ticks("r", int(pos / pi.m_per_tick), t)
            hist.append(speed)
            t += DT
        return hist, pi, logs

    def test_tracks_slow_and_normal_speeds(self):
        for target in (0.05, 0.10, 0.15, -0.08):
            hist, _, _ = self.sim(target)
            tail = hist[int(1.0 / DT):]
            mean = sum(tail) / len(tail)
            self.assertAlmostEqual(mean, target, delta=0.15 * abs(target) + 0.005, msg="target %.2f got %.3f" % (target, mean))

    def test_stall_cut(self):
        _, pi, logs = self.sim(0.15, t_end=4.0, blocked=True)
        self.assertTrue(pi.stalled)
        self.assertTrue(any("STALL" in m for m in logs))

    def test_zero_means_zero(self):
        pi = WheelPI({}, {"gain": 2.0, "radius": 0.0318, "k": 27.0})
        self.assertEqual(pi.step(0.0, 0.005, 0.0, DT), (0.0, 0.0))


class World:
    """Ground truth topology for planner tests: two loops joined by T-intersections (like the user's mat)."""
    # node -> {turn: (next_node, dist)}; node coords (x, y, heading)
    EDGES = {1: {"left": (4, 1.6), "right": (2, 0.9)}, 2: {"straight": (3, 1.2), "left": (5, 1.0)},
             3: {"right": (1, 2.1), "straight": (6, 1.4)}, 4: {"left": (2, 1.1), "straight": (6, 0.8)},
             5: {"right": (6, 0.7), "straight": (1, 1.9)}, 6: {"left": (3, 1.0), "right": (5, 1.3)}}
    POS = {1: (0, 0, 0), 2: (1.0, 0.2, 1.57), 3: (1.0, 1.5, 3.14), 4: (0.2, 1.0, 1.57), 5: (2.0, 0.5, -1.57), 6: (1.8, 1.6, 3.14)}


class TestPlanner(unittest.TestCase):
    def build(self):
        tp = TopoMap()
        ids = {}
        for n, (x, y, th) in World.POS.items():
            ids[n] = tp.add_node(x, y, th)
        for a, outs in World.EDGES.items():
            for t, (b, d) in outs.items():
                tp.set_edge(ids[a], t, ids[b], d)
            for t in TURNS:
                if t not in outs:
                    tp.mark_blocked(ids[a], t)
        return tp

    def test_shortest_path(self):
        tp = self.build()
        p = tp.path(1, 6)
        self.assertEqual(p, [(1, "left"), (4, "straight")])        # 2.4 m beats 1->2->3->6 (3.5 m)
        dist, _ = tp.dijkstra(1)
        self.assertAlmostEqual(dist[6], 2.4 + 0.05, places=6)

    def test_route_to_place(self):
        tp = self.build()
        r = tp.route_to_place(1, {"from": 2, "turn": "left", "s": 0.4})
        self.assertEqual(r["steps"], [(1, "right"), (2, "left")])
        self.assertAlmostEqual(r["stop_s"], 0.4)
        r = tp.route_to_place(3, {"to": 6, "rem": 0.3})               # 0.3 m before stop line 6
        self.assertEqual(r["steps"][-1], (3, "straight"))
        self.assertAlmostEqual(r["stop_s"], 1.4 - 0.3)
        r = tp.route_to_place(5, {"to": 5, "rem": 0.0})
        self.assertEqual(r["steps"], [])

    def test_mapping_and_goto(self):
        """Mission maps the world from scratch with noisy odometry, then plans routes to saved places."""
        rng = random.Random(4)
        tp = TopoMap()
        logs = []
        m = Mission(tp, logs.append)
        m.start("map")
        truth = 1
        drift = [0.0, 0.0]
        id_of = {}
        done = None
        for _ in range(60):
            x, y, th = World.POS[truth]
            pose = (x + drift[0], y + drift[1], th + rng.gauss(0, 0.05))
            n, known = m.identify(*pose)
            if known is not None:
                drift = [0.0, 0.0]                                    # stop line recognised -> odometry reset
            id_of.setdefault(truth, n)
            self.assertEqual(id_of[truth], n, "node confusion")
            tp.observe_exits(n, {t: t in World.EDGES[truth] for t in TURNS})
            res = m.decide(n)
            if "done" in res:
                done = res["done"]
                break
            nxt, dist = World.EDGES[truth][res["turn"]]
            m.edge_dist = dist * rng.uniform(0.97, 1.03)
            drift[0] += rng.gauss(0, 0.04 * dist)
            drift[1] += rng.gauss(0, 0.04 * dist)
            truth = nxt
        self.assertIsNotNone(done, "mapping did not finish")
        self.assertIn("mapped", done)
        self.assertEqual(len(tp.nodes), 6)
        self.assertEqual(len(tp.edges), sum(len(v) for v in World.EDGES.values()))
        for a, outs in World.EDGES.items():
            for t, (b, _) in outs.items():
                self.assertEqual(tp.edges["%d:%s" % (id_of[a], t)]["to"], id_of[b])
        # save a place 0.5 m after node 4 going straight, then go there from node 3
        m.places["dock"] = {"from": id_of[4], "turn": "straight", "s": 0.5}
        m.start("goto", "dock", m.places["dock"])
        cur = id_of[3]
        steps = []
        while True:
            res = m.decide(cur)
            if "done" in res:
                break
            steps.append(res["turn"])
            truth_cur = [k for k, v in id_of.items() if v == cur][0]
            if res.get("stop_s") is not None:
                self.assertAlmostEqual(res["stop_s"], 0.5)
                self.assertEqual((truth_cur, res["turn"]), (4, "straight"))
                break
            cur = id_of[World.EDGES[truth_cur][res["turn"]][0]]
            m.at_node = cur
        self.assertTrue(steps)

    def test_goto_unknown_explores(self):
        tp = TopoMap()
        m = Mission(tp, lambda s: None)
        n1 = tp.add_node(0, 0, 0)
        tp.observe_exits(n1, {"left": True, "straight": False, "right": True})
        m.start("goto", "x", {"to": 99, "rem": 0.2})
        res = m.decide(n1)
        self.assertIn(res["turn"], ("left", "right"))                # explores an unknown exit


if __name__ == "__main__":
    unittest.main(verbosity=2)
