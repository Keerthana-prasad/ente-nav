#!/usr/bin/env python3
"""Renders a synthetic camera frame of a straight lane (real entebot209 intrinsics + extrinsic homography)
at known (d, phi) and checks LanePerception recovers them, plus a stop line distance."""
import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
from ente_nav.lanes import LanePerception      # noqa: E402
from ente_nav.perception import GroundProjector  # noqa: E402

K = [445.40476031606283, 0.0, 322.7093409131285, 0.0, 448.3546110882771, 285.0989720099002, 0.0, 0.0, 1.0]
D = [-0.5621711310294815, 0.391583575333381, 0.009434294249617406, -0.002764599547256031, -0.160019140296408]
H = [-0.01615349486829033, -0.20858844311687008, 0.5572420319003281, -0.6906502850626461, 0.030002915655341984,
     -0.004338628959170905, -0.05883445158535966, 6.323194881545779, 1.0]
LW, WW, YW = 0.23, 0.048, 0.025
BGR = {"mat": (62, 60, 58), "white": (200, 228, 236), "yellow": (60, 170, 205), "red": (110, 120, 235)}


def make_proj():
    p = GroundProjector("none", 0.10, math.radians(12), 0.06)
    p.H = np.array(H).reshape(3, 3)
    p.Hinv = np.linalg.inv(p.H)
    p.D = D
    return p


def render(proj, d, phi, stop=None, dashed=True):
    """Draw tape rectangles given in LANE coordinates (X along lane, Y left of lane centre)."""
    img = np.full((480, 640, 3), BGR["mat"], np.uint8)
    img = (img + np.random.RandomState(0).randint(-6, 7, img.shape)).clip(0, 255).astype(np.uint8)
    cph, sph = math.cos(phi), math.sin(phi)

    def rect(x0, x1, y0, y1, col):
        X = np.concatenate([np.linspace(x0, x1, 20), np.full(8, x1), np.linspace(x1, x0, 20), np.full(8, x0)])
        Y = np.concatenate([np.full(20, y0), np.linspace(y0, y1, 8), np.full(20, y1), np.linspace(y1, y0, 8)])
        Yl = Y - d
        fwd, left = X * cph + Yl * sph, -X * sph + Yl * cph
        # keep the polygon inside the radius where the fisheye polynomial is monotonic (real lens FOV)
        q = proj.Hinv @ np.vstack([fwd, left, np.ones_like(fwd)])
        r = np.hypot(q[0] / q[2], q[1] / q[2])
        if (r > 0.95).any() or (q[2] <= 0).any():
            return
        px = proj.to_pixel(fwd, left, K)
        cv2.fillPoly(img, [px.astype(np.int32)], BGR[col])

    off_y, off_w = LW / 2 + YW / 2, LW / 2 + WW / 2
    for x in np.arange(0.05, 1.5, 0.05):                       # piecewise so near pieces clip cleanly
        rect(x, x + 0.05, -off_w - WW / 2, -off_w + WW / 2, "white")
    for x in np.arange(0.05, 1.5, 0.1 if dashed else 1.6):
        rect(x, x + (0.05 if dashed else 1.6), off_y - YW / 2, off_y + YW / 2, "yellow")
    if stop is not None:
        rect(stop, stop + 0.05, -LW / 2, LW / 2 + YW, "red")
    return img


class TestLanes(unittest.TestCase):
    def run_case(self, d, phi, stop=None):
        proj = make_proj()
        lp = LanePerception({}, proj, {"lane_width": LW})
        img = render(proj, d, phi, stop)
        for _ in range(6):
            st = lp.process(img, K, (640, 480))
        return st

    def test_centered(self):
        st = self.run_case(0.0, 0.0)
        self.assertAlmostEqual(st["d"], 0.0, delta=0.02)
        self.assertAlmostEqual(st["phi"], 0.0, delta=0.05)
        self.assertGreater(st["conf"], 0.5)
        self.assertEqual(st["src"], "yellow+white")

    def test_offsets(self):
        for d, phi in ((0.05, 0.0), (-0.05, 0.0), (0.0, 0.25), (0.03, -0.2), (-0.04, 0.15)):
            st = self.run_case(d, phi)
            self.assertAlmostEqual(st["d"], d, delta=0.025, msg=(d, phi, st))
            self.assertAlmostEqual(st["phi"], phi, delta=0.06, msg=(d, phi, st))

    def test_stopline(self):
        st = self.run_case(0.0, 0.0, stop=0.30)
        self.assertIsNotNone(st["stop_dist"])
        self.assertAlmostEqual(st["stop_dist"], 0.30, delta=0.04)

    def test_roundtrip(self):
        p = make_proj()
        g = np.array([[0.2, 0.05], [0.4, -0.1], [0.15, 0.12]])
        px = p.to_pixel(g[:, 0], g[:, 1], K)
        back = p.project_many(px, K)
        np.testing.assert_allclose(back, g, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
