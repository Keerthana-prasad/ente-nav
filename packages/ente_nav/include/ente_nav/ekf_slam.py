"""Landmark EKF-SLAM (AprilTags as landmarks). State = [x, y, theta, l1x, l1y, ...]."""
import math
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class EKFSlam:
    def __init__(self, q_dist=0.05, q_theta=0.04, r_range=0.06, r_bearing=0.05, gate=13.8):
        self.mu = np.zeros(3)
        self.P = np.diag([1e-6, 1e-6, 1e-6])
        self.ids = {}                       # tag id -> index into mu
        self.q_dist, self.q_theta = q_dist, q_theta
        self.R = np.diag([r_range ** 2, r_bearing ** 2])
        self.gate = gate                    # chi2(2 dof, 99.9%) outlier gate
        self.rejected = 0
        self.accepted = 0

    # ---------- accessors ----------
    @property
    def pose(self):
        return float(self.mu[0]), float(self.mu[1]), float(self.mu[2])

    def landmarks(self):
        return {i: (float(self.mu[k]), float(self.mu[k + 1])) for i, k in self.ids.items()}

    def uncertainty(self):
        """1-sigma position uncertainty in metres."""
        return float(math.sqrt(max(np.trace(self.P[:2, :2]), 0.0)))

    def cov_ellipse(self):
        w, v = np.linalg.eigh(self.P[:2, :2])
        w = np.sqrt(np.maximum(w, 0))
        return float(w[1]), float(w[0]), float(math.atan2(v[1, 1], v[0, 1]))

    # ---------- prediction ----------
    def predict(self, d, dth):
        x, y, th = self.mu[:3]
        tm = th + dth / 2.0
        self.mu[0] = x + d * math.cos(tm)
        self.mu[1] = y + d * math.sin(tm)
        self.mu[2] = wrap(th + dth)
        F = np.array([[1, 0, -d * math.sin(tm)], [0, 1, d * math.cos(tm)], [0, 0, 1]])
        sd = self.q_dist * abs(d) + 1e-4
        st = self.q_theta * abs(dth) + 0.2 * self.q_dist * abs(d) + 1e-4
        Q = np.diag([sd ** 2, sd ** 2, st ** 2])
        self.P[:3, :3] = F @ self.P[:3, :3] @ F.T + Q
        if self.P.shape[0] > 3:
            self.P[:3, 3:] = F @ self.P[:3, 3:]
            self.P[3:, :3] = self.P[:3, 3:].T

    def reset_pose(self, x, y, th, var=1e-4):
        """Snap the robot pose to a known place (a recognised stop line) and shrink its uncertainty."""
        self.mu[0], self.mu[1], self.mu[2] = x, y, wrap(th)
        self.P[:3, :3] = np.diag([var, var, var / 4])
        if self.P.shape[0] > 3:
            self.P[:3, 3:] = 0.0
            self.P[3:, :3] = 0.0

    # ---------- landmarks ----------
    def add_known(self, tag_id, x, y, var=0.01):
        """Pre-seed a landmark whose world position is known (anchors the map)."""
        self._extend(tag_id, np.array([x, y]), np.eye(2) * var, np.zeros((2, self.P.shape[0])))

    def _extend(self, tag_id, pos, Pll, Plx):
        n = len(self.mu)
        P = np.zeros((n + 2, n + 2))
        P[:n, :n] = self.P
        P[n:, :n] = Plx
        P[:n, n:] = Plx.T
        P[n:, n:] = Pll
        self.P = P
        self.mu = np.concatenate([self.mu, pos])
        self.ids[tag_id] = n

    def update(self, tag_id, r, b):
        """r, b: range (m) and bearing (rad, CCW+) of tag in the robot frame."""
        x, y, th = self.mu[:3]
        if tag_id not in self.ids:
            a = th + b
            pos = np.array([x + r * math.cos(a), y + r * math.sin(a)])
            Gx = np.array([[1, 0, -r * math.sin(a)], [0, 1, r * math.cos(a)]])
            Gz = np.array([[math.cos(a), -r * math.sin(a)], [math.sin(a), r * math.cos(a)]])
            n = len(self.mu)
            Plx = Gx @ self.P[:3, :n]
            Pll = Gx @ self.P[:3, :3] @ Gx.T + Gz @ self.R @ Gz.T
            self._extend(tag_id, pos, Pll, Plx)
            return True
        k = self.ids[tag_id]
        dx, dy = self.mu[k] - x, self.mu[k + 1] - y
        q = dx * dx + dy * dy
        sq = math.sqrt(q)
        if sq < 1e-6:
            return False
        zhat = np.array([sq, wrap(math.atan2(dy, dx) - th)])
        n = len(self.mu)
        H = np.zeros((2, n))
        H[:, :3] = [[-dx / sq, -dy / sq, 0], [dy / q, -dx / q, -1]]
        H[:, k:k + 2] = [[dx / sq, dy / sq], [-dy / q, dx / q]]
        innov = np.array([r, b]) - zhat
        innov[1] = wrap(innov[1])
        S = H @ self.P @ H.T + self.R
        if innov @ np.linalg.solve(S, innov) > self.gate:
            self.rejected += 1
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ innov
        self.mu[2] = wrap(self.mu[2])
        I_KH = np.eye(n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T      # Joseph form
        self.P = (self.P + self.P.T) / 2
        self.accepted += 1
        return True
