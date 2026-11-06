"""Wheel-encoder odometry with gyro heading fusion (cuts yaw drift from wheel slip)."""
import math


class Odometry:
    def __init__(self, wheel_radius=0.0318, baseline=0.102, gyro_weight=0.85):
        self.R, self.L, self.alpha = wheel_radius, baseline, gyro_weight
        self.prev = None
        self.gyro_acc = 0.0
        self.gyro_bias = 0.0
        self._bias_samples = []
        self._calibrated = False
        self.have_gyro = False
        self.last_gz = 0.0
        self._last_t = None
        self.v = 0.0
        self.w = 0.0
        self.vs = 0.0                   # smoothed v, w (control-loop jitter makes the raw ratios spiky)
        self.ws = 0.0
        self.th_gyro = 0.0              # diagnostics: total heading change seen by gyro / by encoders
        self.th_enc = 0.0

    def on_imu(self, gz, t):
        """gz: yaw rate rad/s, t: seconds. Calibrates bias from the first ~100 samples (keep robot still)."""
        self.have_gyro = True
        if not self._calibrated:
            self._bias_samples.append(gz)
            if len(self._bias_samples) >= 100:
                self.gyro_bias = sum(self._bias_samples) / len(self._bias_samples)
                self._calibrated = True
            self._last_t = t
            return
        if self._last_t is not None:
            self.gyro_acc += (gz - self.gyro_bias) * (t - self._last_t)
        self.last_gz = gz - self.gyro_bias
        self._last_t = t

    def step(self, left_ticks, right_ticks, res, dt):
        """Returns (distance, dtheta) travelled since the last call, or None on the first call."""
        if self.prev is None:
            self.prev = (left_ticks, right_ticks)
            self.gyro_acc = 0.0
            return None
        dl = 2 * math.pi * self.R * (left_ticks - self.prev[0]) / res
        dr = 2 * math.pi * self.R * (right_ticks - self.prev[1]) / res
        self.prev = (left_ticks, right_ticks)
        d = (dl + dr) / 2.0
        dth_enc = (dr - dl) / self.L
        self.th_enc += dth_enc
        self.th_gyro += self.gyro_acc
        if self.have_gyro and self._calibrated:
            dth = self.alpha * self.gyro_acc + (1 - self.alpha) * dth_enc
        else:
            dth = dth_enc
        self.gyro_acc = 0.0
        if dt > 1e-4:
            self.v, self.w = d / dt, dth / dt
            a = min(1.0, dt / 0.15)
            self.vs += a * (max(-0.4, min(0.4, self.v)) - self.vs)
            self.ws += a * (max(-5.0, min(5.0, self.w)) - self.ws)
        return d, dth
