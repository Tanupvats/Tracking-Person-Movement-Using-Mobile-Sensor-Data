

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .tracker import GPSSample, IMUSample, MagSample


def _angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi




@dataclass
class INSConfig:
    # Initial 1-sigma values
    init_pos_std: float = 3.0
    init_vel_std: float = 2.0        # loose initial velocity
    init_psi_std: float = 0.15
    init_ba_std: float  = 0.03       # tight: biases are small and nearly constant
    init_bg_std: float  = 0.003

    # Process noise PSD (per sqrt(second)).
    # q_acc is large enough that the filter accepts large true accelerations
    # (e.g. a car accelerating from rest) as NOT being bias.
    q_acc: float  = 1.0
    q_gyro: float = 0.05
    q_ba: float   = 0.0002   # very slow bias drift
    q_bg: float   = 0.0001

    # Mahalanobis (NIS) threshold for GPS outlier rejection.
    # Chi-square inverse CDF at p=0.999 for 2 dof ≈ 13.82.
    nis_reject_threshold: float = 16.0

    # ZUPT (zero-velocity update) detection
    zupt_acc_std_threshold: float  = 0.25
    zupt_gyro_mean_threshold: float = 0.08
    zupt_window: int = 15
    enable_zupt: bool = True


@dataclass
class TrackHistory:
    t:        list = field(default_factory=list)
    x:        list = field(default_factory=list)
    P:        list = field(default_factory=list)
    nis:      list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    mode:     list = field(default_factory=list)




class INSTracker:
    STATE_DIM = 8
    I8 = np.eye(8)

    def __init__(self, cfg: Optional[INSConfig] = None):
        self.cfg = cfg or INSConfig()
        self.x = np.zeros(self.STATE_DIM)
        self.P = np.diag([
            self.cfg.init_pos_std ** 2, self.cfg.init_pos_std ** 2,
            self.cfg.init_vel_std ** 2, self.cfg.init_vel_std ** 2,
            self.cfg.init_psi_std ** 2,
            self.cfg.init_ba_std ** 2, self.cfg.init_ba_std ** 2,
            self.cfg.init_bg_std ** 2,
        ])

        self.hist = TrackHistory()
        self._last_t: Optional[float] = None
        self._recent_imu: list[IMUSample] = []
        self._last_acc_b = np.zeros(2)
        self._last_omega = 0.0

        # EMA mode indicator (0 = cruise, 1 = manoeuvre)
        self._mode_bias = 0.0
        self._mode_alpha = 0.15

    # ------------------------------------------------------------------ #
    def initialize(self, pos: np.ndarray, heading: float = 0.0,
                   speed: float = 0.0, t0: float = 0.0) -> None:
        self.x[:2] = pos
        self.x[2] = speed * np.cos(heading)
        self.x[3] = speed * np.sin(heading)
        self.x[4] = _angle_wrap(heading)
        self.x[5:] = 0.0
        self._last_t = t0
        self._record(t0)

    # ------------------------------------------------------------------ #
    def _record(self, t: float, nis: Optional[float] = None,
                rejected: bool = False) -> None:
        self.hist.t.append(t)
        self.hist.x.append(self.x.copy())
        self.hist.P.append(self.P.copy())
        self.hist.nis.append(nis if nis is not None else np.nan)
        self.hist.rejected.append(rejected)
        self.hist.mode.append(self._mode_bias)

    # ------------------------------------------------------------------ #
    def _process_noise(self, dt: float) -> np.ndarray:
        """Time-varying Q with mode-adaptive inflation."""
        scale = 1.0 + 3.0 * self._mode_bias
        qa  = (self.cfg.q_acc  * scale) ** 2
        qg  = (self.cfg.q_gyro * scale) ** 2
        qba = self.cfg.q_ba ** 2
        qbg = self.cfg.q_bg ** 2

        Q = np.zeros((8, 8))
        # Position-velocity white-noise-acceleration structure
        Q[0, 0] = Q[1, 1] = qa * (dt ** 4) / 4
        Q[2, 2] = Q[3, 3] = qa * dt ** 2
        Q[0, 2] = Q[2, 0] = qa * (dt ** 3) / 2
        Q[1, 3] = Q[3, 1] = qa * (dt ** 3) / 2
        # Heading
        Q[4, 4] = qg * dt ** 2
        # Bias random walks
        Q[5, 5] = qba * dt
        Q[6, 6] = qba * dt
        Q[7, 7] = qbg * dt
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
            dt = 2.0   # cap long gaps (GPS outage)

        psi = self.x[4]
        bax, bay, bgz = self.x[5], self.x[6], self.x[7]

        # Correct sensors
        a_x = acc_b[0] - bax
        a_y = acc_b[1] - bay
        om  = omega - bgz

        # Rotate to global frame
        c, s = np.cos(psi), np.sin(psi)
        ax_g = c * a_x - s * a_y
        ay_g = s * a_x + c * a_y

        # Propagate
        vx_new = self.x[2] + ax_g * dt
        vy_new = self.x[3] + ay_g * dt
        px_new = self.x[0] + self.x[2] * dt + 0.5 * ax_g * dt ** 2
        py_new = self.x[1] + self.x[3] * dt + 0.5 * ay_g * dt ** 2
        psi_new = _angle_wrap(psi + om * dt)

        x_new = np.array([px_new, py_new, vx_new, vy_new, psi_new, bax, bay, bgz])

        # Jacobian F = df/dx
        F = np.eye(8)
        F[0, 2] = dt
        F[1, 3] = dt

        # d(ax_g)/d(psi) = -s*a_x - c*a_y ; d(ay_g)/d(psi) = c*a_x - s*a_y
        dax_dpsi = -s * a_x - c * a_y
        day_dpsi =  c * a_x - s * a_y
        F[2, 4] = dax_dpsi * dt
        F[3, 4] = day_dpsi * dt
        F[0, 4] = 0.5 * dax_dpsi * dt ** 2
        F[1, 4] = 0.5 * day_dpsi * dt ** 2

        # d(ax_g)/d(bax) = -c ; d(ax_g)/d(bay) =  s
        # d(ay_g)/d(bax) = -s ; d(ay_g)/d(bay) = -c
        F[2, 5] = -c * dt
        F[2, 6] =  s * dt
        F[3, 5] = -s * dt
        F[3, 6] = -c * dt
        F[0, 5] = -0.5 * c * dt ** 2
        F[0, 6] =  0.5 * s * dt ** 2
        F[1, 5] = -0.5 * s * dt ** 2
        F[1, 6] = -0.5 * c * dt ** 2

        # Heading depends on gyro bias
        F[4, 7] = -dt

        Q = self._process_noise(dt)
        self.x = x_new
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self._last_t = t

    # ------------------------------------------------------------------ #
    def _gps_update(self, pos: np.ndarray, sigma: float) -> tuple[float, bool]:
        H = np.zeros((2, 8))
        H[0, 0] = 1
        H[1, 1] = 1
        R = np.eye(2) * sigma ** 2

        y = pos - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S = S + 1e-6 * np.eye(2)
            S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)

        # Mode indicator: large innovation -> manoeuvre
        mag = float(np.linalg.norm(y))
        target = 1.0 if mag > 3.0 * sigma else 0.0
        self._mode_bias = (1 - self._mode_alpha) * self._mode_bias + self._mode_alpha * target

        if nis > self.cfg.nis_reject_threshold:
            return nis, True   # outlier rejected

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.x[4] = _angle_wrap(self.x[4])
        I_KH = self.I8 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T    # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return nis, False

    # ------------------------------------------------------------------ #
    def _mag_update(self, heading: float, sigma: float) -> None:
        H = np.zeros((1, 8))
        H[0, 4] = 1
        R = np.array([[sigma ** 2]])
        y = np.array([_angle_wrap(heading - self.x[4])])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).ravel()
        self.x[4] = _angle_wrap(self.x[4])
        I_KH = self.I8 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------ #
    def _zupt_update(self) -> None:
        """Zero-velocity pseudo-measurement: v = 0."""
        H = np.zeros((2, 8))
        H[0, 2] = 1
        H[1, 3] = 1
        R = np.eye(2) * 0.02 ** 2
        y = -H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = self.I8 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------ #
    def _detect_stationary(self) -> bool:
        if not self.cfg.enable_zupt or len(self._recent_imu) < self.cfg.zupt_window:
            return False
        window = self._recent_imu[-self.cfg.zupt_window:]
        acc_mag = np.array([np.linalg.norm(s.acc) for s in window])
        omega   = np.array([abs(s.omega_z) for s in window])
        imu_quiet = (acc_mag.mean() < self.cfg.zupt_acc_std_threshold
                     and acc_mag.std() < self.cfg.zupt_acc_std_threshold
                     and omega.mean() < self.cfg.zupt_gyro_mean_threshold)
        # Require the filter to also believe we're slow. Without this a
        # vehicle in steady cruise (zero body-frame accel) would wrongly
        # trigger ZUPT and get its velocity zeroed.
        current_speed = float(np.hypot(self.x[2], self.x[3]))
        return imu_quiet and current_speed < 0.5

    # ------------------------------------------------------------------ #
    def step(self, sample) -> None:
        if isinstance(sample, IMUSample):
            self._recent_imu.append(sample)
            if len(self._recent_imu) > 256:
                self._recent_imu.pop(0)
            self._predict(sample.t, sample.acc, sample.omega_z)
            self._last_acc_b = sample.acc
            self._last_omega = sample.omega_z
            if self._detect_stationary():
                self._zupt_update()
            self._record(sample.t)

        elif isinstance(sample, GPSSample):
            self._predict(sample.t, self._last_acc_b, self._last_omega)
            nis, rejected = self._gps_update(np.asarray(sample.pos), sample.sigma)
            self._record(sample.t, nis=nis, rejected=rejected)

        elif isinstance(sample, MagSample):
            self._predict(sample.t, self._last_acc_b, self._last_omega)
            self._mag_update(sample.heading, sample.sigma)
            self._record(sample.t)
        else:
            raise TypeError(f"Unknown sample {type(sample)}")

    # ------------------------------------------------------------------ #
    def trajectory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ts = np.array(self.hist.t)
        xs = np.stack(self.hist.x)
        Ps = np.stack(self.hist.P)
        poses = xs[:, :4]
        covs  = Ps[:, :4, :4]
        return ts, poses, covs

    def mode_history(self) -> np.ndarray:
        """Return (N, 2) array [cruise_prob, manoeuvre_prob]."""
        m = np.array(self.hist.mode)
        return np.stack([1 - m, m], axis=1)




def rts_smooth_track(times: np.ndarray,
                     poses: np.ndarray,
                     covs: np.ndarray,
                     q_acc: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """CV-model RTS smoother over the filtered (x, y, vx, vy) track."""
    from .motion_models import ConstantVelocity
    N = len(times)
    xs = poses.copy()
    Ps = covs.copy()
    for k in range(N - 2, -1, -1):
        dt = times[k + 1] - times[k]
        if dt <= 0:
            continue
        F = ConstantVelocity.F(xs[k], dt)
        Q = ConstantVelocity.Q(dt, q_acc)
        P_pred = F @ Ps[k] @ F.T + Q
        try:
            C = Ps[k] @ F.T @ np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            continue
        xs[k] = xs[k] + C @ (xs[k + 1] - F @ xs[k])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - P_pred) @ C.T
    return xs, Ps
