

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .tracker import GPSSample, IMUSample, MagSample


@dataclass
class Truth:
    t: np.ndarray      # (N,)
    pos: np.ndarray    # (N, 2)
    vel: np.ndarray    # (N, 2)
    acc: np.ndarray    # (N, 2) body-frame
    heading: np.ndarray  # (N,)
    omega: np.ndarray  # (N,)





def _derivatives(t: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return vel (global), acc (body), heading, omega from pos(t)."""
    dt = np.gradient(t)
    vx = np.gradient(pos[:, 0], t)
    vy = np.gradient(pos[:, 1], t)
    vel = np.column_stack([vx, vy])
    speed = np.hypot(vx, vy)

    heading = np.arctan2(vy, vx)
    # Unwrap to avoid jumps when computing omega
    heading_un = np.unwrap(heading)
    omega = np.gradient(heading_un, t)

    # Body-frame acceleration = R(-heading) @ global acc
    ax_g = np.gradient(vx, t)
    ay_g = np.gradient(vy, t)
    c, s = np.cos(heading), np.sin(heading)
    ax_b = c * ax_g + s * ay_g
    ay_b = -s * ax_g + c * ay_g
    acc = np.column_stack([ax_b, ay_b])

    return vel, acc, heading, omega


def _make_truth(t: np.ndarray, pos: np.ndarray) -> Truth:
    vel, acc, heading, omega = _derivatives(t, pos)
    return Truth(t=t, pos=pos, vel=vel, acc=acc, heading=heading, omega=omega)


def traj_circle(T: float = 60.0, radius: float = 20.0, speed: float = 2.0, dt: float = 0.01) -> Truth:
    t = np.arange(0, T, dt)
    omega = speed / radius
    theta = omega * t
    pos = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    return _make_truth(t, pos)


def traj_figure_eight(T: float = 80.0, A: float = 15.0, speed: float = 1.5, dt: float = 0.01) -> Truth:
    t = np.arange(0, T, dt)
    omega = speed / A
    pos = np.column_stack([A * np.sin(omega * t), A * np.sin(omega * t) * np.cos(omega * t)])
    return _make_truth(t, pos)


def traj_urban_canyon(T: float = 100.0, dt: float = 0.01) -> Truth:
    """
    Piecewise route mimicking a pedestrian in an urban canyon: straight
    segments + sharp 90-deg turns. This is the hardest regime for GPS-only
    tracking because of multipath.
    """
    t = np.arange(0, T, dt)
    # Define waypoint times and positions
    waypoints = np.array(
        [
            [0.0, 0.0],
            [30.0, 0.0],
            [30.0, 25.0],
            [55.0, 25.0],
            [55.0, 5.0],
            [80.0, 5.0],
            [80.0, 40.0],
        ]
    )
    # Travel at roughly constant speed
    segs = np.diff(waypoints, axis=0)
    seg_len = np.linalg.norm(segs, axis=1)
    total_len = seg_len.sum()
    speed = total_len / T
    seg_t = seg_len / speed
    cum_t = np.concatenate([[0], np.cumsum(seg_t)])

    pos = np.zeros((len(t), 2))
    for i in range(len(segs)):
        mask = (t >= cum_t[i]) & (t <= cum_t[i + 1])
        if mask.sum() == 0:
            continue
        alpha = (t[mask] - cum_t[i]) / (seg_t[i] + 1e-9)
        pos[mask] = waypoints[i] + alpha[:, None] * segs[i]
    # Smooth corners slightly via low-pass on position
    from scipy.signal import savgol_filter
    pos[:, 0] = savgol_filter(pos[:, 0], 201, 3)
    pos[:, 1] = savgol_filter(pos[:, 1], 201, 3)
    return _make_truth(t, pos)


def traj_highway(T: float = 60.0, dt: float = 0.01) -> Truth:
    """Vehicle-like: accelerate, cruise, lane-change, brake."""
    t = np.arange(0, T, dt)
    # Forward-speed profile
    v_target = np.piecewise(
        t,
        [t < 10, (t >= 10) & (t < 30), (t >= 30) & (t < 40), t >= 40],
        [lambda tt: 2 * tt,            # 0 -> 20 m/s over 10s
         lambda tt: 20.0 + 0 * tt,     # cruise
         lambda tt: 20.0 + 0 * tt,     # lane change, same speed
         lambda tt: np.maximum(20 - 1.0 * (tt - 40), 5.0)],
    )
    x = np.cumsum(v_target) * dt
    # Lane change: sinusoidal lateral between t=30 and t=40
    y = np.zeros_like(t)
    lane_mask = (t >= 30) & (t < 40)
    tau = (t[lane_mask] - 30) / 10
    y[lane_mask] = 3.5 * (0.5 - 0.5 * np.cos(np.pi * tau))
    y[t >= 40] = 3.5
    pos = np.column_stack([x, y])
    return _make_truth(t, pos)


def traj_pedestrian_mixed(T: float = 90.0, dt: float = 0.01) -> Truth:
    """
    Realistic pedestrian pattern: walk, stop (wait at crossing), walk again,
    turn, jog. Tests ZUPT and the IMM mode-switching.
    """
    t = np.arange(0, T, dt)
    pos = np.zeros((len(t), 2))

    def seg_const(t0, t1, start, vel):
        mask = (t >= t0) & (t < t1)
        dt_seg = (t[mask] - t0)[:, None]
        return mask, start + dt_seg * vel

    phases: list = [
        ("walk_east", 0.0, 25.0, np.array([0.0, 0.0]), np.array([1.2, 0.0])),
        ("stop",      25.0, 35.0, None, np.array([0.0, 0.0])),
        ("walk_ne",   35.0, 55.0, None, np.array([1.2, 1.2]) / np.sqrt(2) * 1.2),
        ("turn_n",    55.0, 60.0, None, None),
        ("jog_n",     60.0, 90.0, None, np.array([0.0, 2.5])),
    ]

    cursor = np.array([0.0, 0.0])
    for name, t0, t1, _, vel in phases:
        mask = (t >= t0) & (t < t1)
        if not mask.any():
            continue
        if name == "turn_n":
            # Smooth quarter-turn
            tau = (t[mask] - t0) / (t1 - t0)
            r = 3.0
            c_x = cursor[0] - r
            c_y = cursor[1]
            theta0 = 0.0
            theta = theta0 + tau * (np.pi / 2)  # turn from east to north
            pos[mask, 0] = c_x + r * np.cos(theta)
            pos[mask, 1] = c_y + r * np.sin(theta)
            cursor = pos[mask][-1]
            continue
        dt_seg = (t[mask] - t0)[:, None]
        segment_pos = cursor + dt_seg * vel
        pos[mask] = segment_pos
        cursor = segment_pos[-1]

    # Keep positions monotone / no NaN for zero-velocity phases
    for i in range(1, len(pos)):
        if np.all(pos[i] == 0) and i > 0:
            pos[i] = pos[i - 1]
    return _make_truth(t, pos)




@dataclass
class SensorConfig:
    imu_rate: float = 100.0
    gps_rate: float = 1.0
    mag_rate: float = 10.0

    acc_noise: float = 0.2           # m/s^2
    acc_bias: np.ndarray = None      # set in __post_init__
    gyro_noise: float = 0.01         # rad/s
    gyro_bias: float = 0.002

    gps_noise: float = 2.5           # m (open sky)
    gps_outage_intervals: list = None        # [(t_start, t_end)]
    gps_multipath_prob: float = 0.0
    gps_multipath_magnitude: float = 25.0

    mag_noise: float = 0.05          # rad

    seed: int = 0

    def __post_init__(self):
        if self.acc_bias is None:
            self.acc_bias = np.array([0.05, -0.03])
        if self.gps_outage_intervals is None:
            self.gps_outage_intervals = []


def stream_sensors(truth: Truth, cfg: SensorConfig) -> Iterable:
    """Yield IMU / GPS / Mag samples in timestamp order."""
    rng = np.random.default_rng(cfg.seed)

    t_end = truth.t[-1]
    imu_times = np.arange(0, t_end, 1.0 / cfg.imu_rate)
    gps_times = np.arange(0, t_end, 1.0 / cfg.gps_rate)
    mag_times = np.arange(0, t_end, 1.0 / cfg.mag_rate)

    def interp(ts: np.ndarray, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return np.interp(ts, truth.t, arr)
        return np.stack([np.interp(ts, truth.t, arr[:, i]) for i in range(arr.shape[1])], axis=1)

    imu_acc_b = interp(imu_times, truth.acc)
    imu_omega = interp(imu_times, truth.omega)
    # Add bias + white noise
    imu_acc_b = imu_acc_b + cfg.acc_bias + rng.normal(0, cfg.acc_noise, size=imu_acc_b.shape)
    imu_omega = imu_omega + cfg.gyro_bias + rng.normal(0, cfg.gyro_noise, size=imu_omega.shape)

    gps_pos = interp(gps_times, truth.pos)
    gps_pos = gps_pos + rng.normal(0, cfg.gps_noise, size=gps_pos.shape)

    mag_heading = interp(mag_times, truth.heading) + rng.normal(0, cfg.mag_noise, size=mag_times.shape)

    # Multipath spikes
    mpath_mask = rng.random(len(gps_times)) < cfg.gps_multipath_prob
    mpath_offsets = rng.normal(0, cfg.gps_multipath_magnitude, size=(len(gps_times), 2))
    gps_pos[mpath_mask] += mpath_offsets[mpath_mask]

    # Outages remove GPS samples
    def in_outage(t):
        for (a, b) in cfg.gps_outage_intervals:
            if a <= t <= b:
                return True
        return False

    # Merge streams sorted by time
    events = []
    for k, t in enumerate(imu_times):
        events.append((t, "imu", k))
    for k, t in enumerate(gps_times):
        if not in_outage(t):
            events.append((t, "gps", k))
    for k, t in enumerate(mag_times):
        events.append((t, "mag", k))
    events.sort(key=lambda e: e[0])

    for t, kind, k in events:
        if kind == "imu":
            yield IMUSample(t=t, acc=imu_acc_b[k], omega_z=imu_omega[k])
        elif kind == "gps":
            yield GPSSample(t=t, pos=gps_pos[k], sigma=cfg.gps_noise)
        elif kind == "mag":
            yield MagSample(t=t, heading=mag_heading[k], sigma=cfg.mag_noise)




def extract_gps_only(samples: List) -> tuple[np.ndarray, np.ndarray]:
    ts = []
    pos = []
    for s in samples:
        if isinstance(s, GPSSample):
            ts.append(s.t)
            pos.append(s.pos)
    return np.array(ts), np.array(pos)




def interpolate_truth(times: np.ndarray, truth: Truth) -> np.ndarray:
    return np.stack(
        [np.interp(times, truth.t, truth.pos[:, 0]),
         np.interp(times, truth.t, truth.pos[:, 1])],
        axis=1,
    )


def rmse(est: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((est - ref) ** 2, axis=1))))


def cep50(est: np.ndarray, ref: np.ndarray) -> float:
    """Circular Error Probable (50%) - the median radial error."""
    errs = np.linalg.norm(est - ref, axis=1)
    return float(np.median(errs))


def max_err(est: np.ndarray, ref: np.ndarray) -> float:
    errs = np.linalg.norm(est - ref, axis=1)
    return float(errs.max())
