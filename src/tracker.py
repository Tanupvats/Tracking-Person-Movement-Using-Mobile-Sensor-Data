

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .imm import IMMFilter
from .motion_models import CTRV, ConstantAcceleration, ConstantVelocity
from .ukf import UKF, MerweSigmaPoints




@dataclass
class IMUSample:
    t: float
    acc: np.ndarray     # (2,) body-frame accel, m/s^2 (x forward, y left)
    omega_z: float      # yaw rate, rad/s


@dataclass
class GPSSample:
    t: float
    pos: np.ndarray     # (2,) local ENU metres
    sigma: float = 3.0  # 1-sigma position noise


@dataclass
class MagSample:
    t: float
    heading: float      # rad, [-pi, pi]
    sigma: float = 0.2




def _angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _residual_ctrv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = a - b
    r[3] = _angle_wrap(r[3])
    return r


def _weighted_mean_ctrv(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    # Handle circular yaw component via sin/cos averaging.
    m = Wm @ sigmas
    psis = sigmas[:, 3]
    sin_m = Wm @ np.sin(psis)
    cos_m = Wm @ np.cos(psis)
    m[3] = np.arctan2(sin_m, cos_m)
    return m




def build_cv_ukf() -> UKF:
    m = ConstantVelocity
    ukf = UKF(
        dim_x=m.dim, dim_z=2,
        fx=m.f,
        hx=lambda x: x[:2],
        sigma=MerweSigmaPoints(m.dim),
    )
    ukf.P = np.diag([25.0, 25.0, 5.0, 5.0])
    return ukf


def build_ca_ukf() -> UKF:
    m = ConstantAcceleration
    ukf = UKF(
        dim_x=m.dim, dim_z=2,
        fx=m.f,
        hx=lambda x: x[:2],
        sigma=MerweSigmaPoints(m.dim),
    )
    ukf.P = np.diag([25.0, 25.0, 5.0, 5.0, 2.0, 2.0])
    return ukf


def build_ctrv_ukf() -> UKF:
    m = CTRV
    ukf = UKF(
        dim_x=m.dim, dim_z=2,
        fx=m.f,
        hx=lambda x: x[:2],
        sigma=MerweSigmaPoints(m.dim),
        residual_x=_residual_ctrv,
        mean_x=_weighted_mean_ctrv,
    )
    ukf.P = np.diag([25.0, 25.0, 5.0, 0.5, 0.5])
    return ukf




class SensorFusionTracker:
    """
    IMM-UKF tracker fusing GPS + IMU + optional magnetometer.

    Usage:
        tracker = SensorFusionTracker()
        tracker.initialize(pos=[0,0], heading=0.0)
        for sample in stream:
            tracker.step(sample)
        traj = tracker.trajectory()
    """

    def __init__(
        self,
        sigma_a_cv: float = 0.6,
        sigma_a_ctrv: float = 0.8,
        sigma_psidd_ctrv: float = 0.15,   # tighter, was 0.5
        sigma_j_ca: float = 0.6,
        zupt_acc_threshold: float = 0.35,
        zupt_gyro_threshold: float = 0.08,
        zupt_window: int = 10,
        enable_zupt: bool = True,
        enable_adaptive_R: bool = False,
    ):
        # Build three sub-filters
        self._cv = build_cv_ukf()
        self._ca = build_ca_ukf()
        self._ctrv = build_ctrv_ukf()

        # Toggle adaptive R per filter
        for f in (self._cv, self._ca, self._ctrv):
            f.adaptive = enable_adaptive_R

        # Process-noise tuning
        self._sigma_a_cv = sigma_a_cv
        self._sigma_a_ctrv = sigma_a_ctrv
        self._sigma_psidd_ctrv = sigma_psidd_ctrv
        self._sigma_j_ca = sigma_j_ca

        # IMM transition matrix: high self-persistence, small cross-coupling
        M_trans = np.array(
            [
                [0.95, 0.03, 0.02],   # CV -> CV/CTRV/CA
                [0.03, 0.95, 0.02],   # CTRV
                [0.05, 0.05, 0.90],   # CA
            ]
        )
        self._imm = IMMFilter(
            filters=[self._cv, self._ctrv, self._ca],
            mu=np.array([0.5, 0.4, 0.1]),
            M_trans=M_trans,
        )

        # State for ZUPT
        self._enable_zupt = enable_zupt
        self._zupt_thresh_acc = zupt_acc_threshold
        self._zupt_thresh_gyro = zupt_gyro_threshold
        self._zupt_window = zupt_window
        self._recent_imu: list[IMUSample] = []

        self._last_t: Optional[float] = None
        self._history: list[tuple[float, np.ndarray, np.ndarray]] = []
        # (t, [x,y,vx,vy], cov(4,4))

    # ------------------------------------------------------------------ #
    def initialize(
        self,
        pos: np.ndarray,
        heading: float = 0.0,
        speed: float = 0.0,
        t0: float = 0.0,
    ) -> None:
        pos = np.asarray(pos, dtype=float)
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)

        self._cv.x = np.array([pos[0], pos[1], vx, vy])
        self._ca.x = np.array([pos[0], pos[1], vx, vy, 0.0, 0.0])
        self._ctrv.x = np.array([pos[0], pos[1], speed, heading, 0.0])
        self._last_t = t0
        self._history.append((t0, self._cv.x.copy(), self._cv.P.copy()))

    # ------------------------------------------------------------------ #
    def _apply_process_noise(self, dt: float) -> None:
        """Refresh each sub-filter's Q based on dt."""
        self._cv.Q = ConstantVelocity.Q(dt, self._sigma_a_cv)
        self._ca.Q = ConstantAcceleration.Q(dt, self._sigma_j_ca)
        self._ctrv.Q = CTRV.Q(dt, self._sigma_a_ctrv, self._sigma_psidd_ctrv)

    # ------------------------------------------------------------------ #
    def _predict_to(self, t: float) -> None:
        if self._last_t is None:
            self._last_t = t
            return
        dt = t - self._last_t
        if dt <= 0:
            return
        self._apply_process_noise(dt)
        self._imm.predict(dt)
        self._last_t = t
        # Guard against divergence: cap each sub-filter's P trace
        for f in (self._cv, self._ca, self._ctrv):
            tr = np.trace(f.P)
            if not np.isfinite(tr) or tr > 1e6:
                # Reset to a large but finite covariance
                f.P = np.eye(f.dim_x) * 100.0

    # ------------------------------------------------------------------ #
    def _detect_stationary(self) -> bool:
        if not self._enable_zupt or len(self._recent_imu) < self._zupt_window:
            return False
        window = self._recent_imu[-self._zupt_window:]
        acc_mag = np.array([np.linalg.norm(s.acc) for s in window])
        omega = np.array([abs(s.omega_z) for s in window])
        # Detrend gravity / bias by looking at std instead of magnitude
        return (acc_mag.std() < self._zupt_thresh_acc
                and omega.mean() < self._zupt_thresh_gyro)

    # ------------------------------------------------------------------ #
    def step(self, sample) -> None:
        """Process one sensor sample, updating the filter state."""
        if isinstance(sample, IMUSample):
            self._recent_imu.append(sample)
            if len(self._recent_imu) > 200:
                self._recent_imu.pop(0)
            self._predict_to(sample.t)

            # Zero-velocity update: if the device is stationary, snap velocity
            # to zero. This is critical for pedestrian dead-reckoning (e.g.
            # Foxlin 2005, NavShoe).
            if self._detect_stationary():
                self._apply_zupt()

        elif isinstance(sample, GPSSample):
            self._predict_to(sample.t)
            # Set R from sample noise
            R = np.eye(2) * sample.sigma ** 2
            for f in (self._cv, self._ca, self._ctrv):
                f.R = R
            self._imm.update(np.asarray(sample.pos, dtype=float))

        elif isinstance(sample, MagSample):
            self._predict_to(sample.t)
            self._apply_heading_update(sample.heading, sample.sigma)
        else:
            raise TypeError(f"Unknown sample type {type(sample)}")

        # Record
        mean, cov = self._imm.combined_pose()
        self._history.append((sample.t, mean, cov))

    # ------------------------------------------------------------------ #
    def _apply_zupt(self) -> None:
        """Soft zero-velocity update: pseudo-measurement v = 0."""
        # We directly shrink velocity in the state while inflating covariance
        # slightly on position (to avoid over-confidence).
        for f in (self._cv, self._ca):
            f.x[2] *= 0.1
            f.x[3] *= 0.1
            f.P[2, 2] = max(f.P[2, 2], 0.05)
            f.P[3, 3] = max(f.P[3, 3], 0.05)
        # CTRV: v -> 0
        self._ctrv.x[2] *= 0.1
        self._ctrv.P[2, 2] = max(self._ctrv.P[2, 2], 0.05)

    def _apply_heading_update(self, heading: float, sigma: float) -> None:
        # Only CTRV has an explicit heading state; update it directly.
        H = np.zeros((1, 5))
        H[0, 3] = 1.0
        innov = _angle_wrap(heading - self._ctrv.x[3])
        S = (H @ self._ctrv.P @ H.T + np.array([[sigma ** 2]]))
        K = self._ctrv.P @ H.T / S
        self._ctrv.x = self._ctrv.x + (K * innov).ravel()
        self._ctrv.x[3] = _angle_wrap(self._ctrv.x[3])
        I_KH = np.eye(5) - K @ H
        self._ctrv.P = I_KH @ self._ctrv.P @ I_KH.T + K @ np.array([[sigma ** 2]]) @ K.T

    # ------------------------------------------------------------------ #
    def trajectory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (times, poses [N,4], covariances [N,4,4])."""
        ts = np.array([h[0] for h in self._history])
        poses = np.stack([h[1] for h in self._history])
        covs = np.stack([h[2] for h in self._history])
        return ts, poses, covs

    def mode_probabilities(self) -> np.ndarray:
        """Return history of IMM mode probabilities, shape (N, 3)."""
        if not self._imm.mode_history:
            return np.zeros((0, 3))
        return np.stack(self._imm.mode_history)


# --------------------------------------------------------------------------- #
# RTS smoother for off-line refinement                                        #
# --------------------------------------------------------------------------- #


def rts_smoother(
    times: np.ndarray,
    filtered_mean: np.ndarray,
    filtered_cov: np.ndarray,
    sigma_a: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel smoother applied to a pose-space track (CV model).

    Produces smoother output than the filter alone by running a backward
    pass that uses future measurements to refine past estimates. Commonly
    used for post-processing in offline analysis.
    """
    N = len(times)
    xs = filtered_mean.copy()
    Ps = filtered_cov.copy()

    for k in range(N - 2, -1, -1):
        dt = times[k + 1] - times[k]
        if dt <= 0:
            continue
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        Q = ConstantVelocity.Q(dt, sigma_a)
        P_pred = F @ Ps[k] @ F.T + Q
        try:
            C = Ps[k] @ F.T @ np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            continue
        xs[k] = xs[k] + C @ (xs[k + 1] - F @ xs[k])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - P_pred) @ C.T
    return xs, Ps
