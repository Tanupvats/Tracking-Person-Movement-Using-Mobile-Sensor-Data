

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .tracker import GPSSample, IMUSample, MagSample


def _angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class SimpleINSConfig:
    init_pos_std: float = 3.0
    init_vel_std: float = 1.0
    init_psi_std: float = 0.1

    # Process noise: we tolerate significant accel/gyro uncertainty so the
    # filter can absorb small biases as random-walk model mismatch.
    q_acc: float  = 0.6    # m/s^2 / sqrt(s)
    q_gyro: float = 0.03   # rad/s / sqrt(s)

    nis_reject_threshold: float = 16.0

    zupt_acc_std_threshold: float  = 0.25
    zupt_gyro_mean_threshold: float = 0.08
    zupt_window: int = 15
    enable_zupt: bool = True


@dataclass
class _Hist:
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    P: list = field(default_factory=list)
    nis: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    mode: list = field(default_factory=list)


class SimpleINSTracker:
    STATE_DIM = 5
    I5 = np.eye(5)

    def __init__(self, cfg: Optional[SimpleINSConfig] = None):
        self.cfg = cfg or SimpleINSConfig()
        self.x = np.zeros(self.STATE_DIM)
        self.P = np.diag([
            self.cfg.init_pos_std ** 2, self.cfg.init_pos_std ** 2,
            self.cfg.init_vel_std ** 2, self.cfg.init_vel_std ** 2,
            self.cfg.init_psi_std ** 2,
        ])
        self.hist = _Hist()
        self._last_t: Optional[float] = None
        self._recent_imu: list[IMUSample] = []
        self._last_acc = np.zeros(2)
        self._last_omega = 0.0
        self._mode_bias = 0.0

    # ------------------------------------------------------------------ #
    def initialize(self, pos: np.ndarray, heading: float = 0.0,
                   speed: float = 0.0, t0: float = 0.0) -> None:
        self.x[:2] = pos
        self.x[2] = speed * np.cos(heading)
        self.x[3] = speed * np.sin(heading)
        self.x[4] = _angle_wrap(heading)
        self._last_t = t0
        self._record(t0)

    def _record(self, t, nis=None, rejected=False):
        self.hist.t.append(t)
        self.hist.x.append(self.x.copy())
        self.hist.P.append(self.P.copy())
        self.hist.nis.append(nis if nis is not None else np.nan)
        self.hist.rejected.append(rejected)
        self.hist.mode.append(self._mode_bias)

    # ------------------------------------------------------------------ #
    def _Q(self, dt: float) -> np.ndarray:
        scale = 1.0 + 3.0 * self._mode_bias
        qa = (self.cfg.q_acc * scale) ** 2
        qg = (self.cfg.q_gyro * scale) ** 2
        Q = np.zeros((5, 5))
        Q[0, 0] = Q[1, 1] = qa * dt ** 4 / 4
        Q[2, 2] = Q[3, 3] = qa * dt ** 2
        Q[0, 2] = Q[2, 0] = qa * dt ** 3 / 2
        Q[1, 3] = Q[3, 1] = qa * dt ** 3 / 2
        Q[4, 4] = qg * dt ** 2
        return Q

    # ------------------------------------------------------------------ #
    def _predict(self, t: float, acc_b: np.ndarray, omega: float) -> None:
        if self._last_t is None:
            self._last_t = t
            return
        dt = t - self._last_t
        if dt <= 0:
            return
        if dt > 2.0:
            dt = 2.0

        psi = self.x[4]
        c, s = np.cos(psi), np.sin(psi)
        ax = acc_b[0]
        ay = acc_b[1]
        ax_g = c * ax - s * ay
        ay_g = s * ax + c * ay

        vx_new = self.x[2] + ax_g * dt
        vy_new = self.x[3] + ay_g * dt
        px_new = self.x[0] + self.x[2] * dt + 0.5 * ax_g * dt ** 2
        py_new = self.x[1] + self.x[3] * dt + 0.5 * ay_g * dt ** 2
        psi_new = _angle_wrap(psi + omega * dt)

        x_new = np.array([px_new, py_new, vx_new, vy_new, psi_new])

        F = np.eye(5)
        F[0, 2] = dt
        F[1, 3] = dt
        dax_dpsi = -s * ax - c * ay
        day_dpsi =  c * ax - s * ay
        F[2, 4] = dax_dpsi * dt
        F[3, 4] = day_dpsi * dt
        F[0, 4] = 0.5 * dax_dpsi * dt ** 2
        F[1, 4] = 0.5 * day_dpsi * dt ** 2

        Q = self._Q(dt)
        self.x = x_new
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self._last_t = t

    # ------------------------------------------------------------------ #
    def _gps_update(self, pos: np.ndarray, sigma: float):
        H = np.zeros((2, 5))
        H[0, 0] = 1; H[1, 1] = 1
        R = np.eye(2) * sigma ** 2
        y = pos - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S += 1e-6 * np.eye(2); S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)

        mag = float(np.linalg.norm(y))
        target = 1.0 if mag > 3.0 * sigma else 0.0
        self._mode_bias = 0.85 * self._mode_bias + 0.15 * target

        if nis > self.cfg.nis_reject_threshold:
            return nis, True

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.x[4] = _angle_wrap(self.x[4])
        I_KH = self.I5 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return nis, False

    def _mag_update(self, heading: float, sigma: float):
        H = np.zeros((1, 5)); H[0, 4] = 1
        R = np.array([[sigma ** 2]])
        y = np.array([_angle_wrap(heading - self.x[4])])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).ravel()
        self.x[4] = _angle_wrap(self.x[4])
        I_KH = self.I5 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def _zupt_update(self):
        H = np.zeros((2, 5)); H[0, 2] = 1; H[1, 3] = 1
        R = np.eye(2) * 0.02 ** 2
        y = -H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = self.I5 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def _detect_stationary(self):
        """
        Stationarity detection: requires both quiet IMU AND the filter's
        current velocity estimate to be small. A vehicle at highway cruise
        has zero body-frame accel but is decidedly not stationary.
        """
        if not self.cfg.enable_zupt or len(self._recent_imu) < self.cfg.zupt_window:
            return False
        window = self._recent_imu[-self.cfg.zupt_window:]
        acc_mag = np.array([np.linalg.norm(s.acc) for s in window])
        omega = np.array([abs(s.omega_z) for s in window])
        imu_quiet = (acc_mag.mean() < self.cfg.zupt_acc_std_threshold
                     and acc_mag.std() < self.cfg.zupt_acc_std_threshold
                     and omega.mean() < self.cfg.zupt_gyro_mean_threshold)
        # Guard: filter must also believe we're (nearly) stationary
        current_speed = float(np.hypot(self.x[2], self.x[3]))
        return imu_quiet and current_speed < 0.5

    # ------------------------------------------------------------------ #
    def step(self, sample):
        if isinstance(sample, IMUSample):
            self._recent_imu.append(sample)
            if len(self._recent_imu) > 256:
                self._recent_imu.pop(0)
            self._predict(sample.t, sample.acc, sample.omega_z)
            self._last_acc = sample.acc; self._last_omega = sample.omega_z
            if self._detect_stationary():
                self._zupt_update()
            self._record(sample.t)
        elif isinstance(sample, GPSSample):
            self._predict(sample.t, self._last_acc, self._last_omega)
            nis, rej = self._gps_update(np.asarray(sample.pos), sample.sigma)
            self._record(sample.t, nis=nis, rejected=rej)
        elif isinstance(sample, MagSample):
            self._predict(sample.t, self._last_acc, self._last_omega)
            self._mag_update(sample.heading, sample.sigma)
            self._record(sample.t)
        else:
            raise TypeError(f"Unknown sample {type(sample)}")

    # ------------------------------------------------------------------ #
    def trajectory(self):
        ts = np.array(self.hist.t)
        xs = np.stack(self.hist.x)
        Ps = np.stack(self.hist.P)
        return ts, xs[:, :4], Ps[:, :4, :4]

    def mode_history(self):
        m = np.array(self.hist.mode)
        return np.stack([1 - m, m], axis=1)
