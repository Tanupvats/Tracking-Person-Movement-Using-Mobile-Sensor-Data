

from __future__ import annotations

import numpy as np




class ConstantVelocity:
    """2-D constant-velocity model. State = [x, y, vx, vy]."""

    dim = 4
    name = "CV"

    @staticmethod
    def f(x: np.ndarray, dt: float) -> np.ndarray:
        nx = x.copy()
        nx[0] += x[2] * dt
        nx[1] += x[3] * dt
        return nx

    @staticmethod
    def F(x: np.ndarray, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    @staticmethod
    def Q(dt: float, sigma_a: float = 0.5) -> np.ndarray:
        """Continuous-white-noise-acceleration (CWNA) discretisation."""
        q = sigma_a ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        return q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ]
        )





class ConstantAcceleration:
    """2-D constant-acceleration model. State = [x, y, vx, vy, ax, ay]."""

    dim = 6
    name = "CA"

    @staticmethod
    def f(x: np.ndarray, dt: float) -> np.ndarray:
        nx = x.copy()
        nx[0] += x[2] * dt + 0.5 * x[4] * dt ** 2
        nx[1] += x[3] * dt + 0.5 * x[5] * dt ** 2
        nx[2] += x[4] * dt
        nx[3] += x[5] * dt
        return nx

    @staticmethod
    def F(x: np.ndarray, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * dt ** 2
        F[1, 5] = 0.5 * dt ** 2
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    @staticmethod
    def Q(dt: float, sigma_j: float = 0.5) -> np.ndarray:
        """Continuous-white-noise-jerk (CWNJ) discretisation."""
        q = sigma_j ** 2
        dt2, dt3, dt4, dt5 = dt ** 2, dt ** 3, dt ** 4, dt ** 5
        return q * np.array(
            [
                [dt5 / 20, 0, dt4 / 8, 0, dt3 / 6, 0],
                [0, dt5 / 20, 0, dt4 / 8, 0, dt3 / 6],
                [dt4 / 8, 0, dt3 / 3, 0, dt2 / 2, 0],
                [0, dt4 / 8, 0, dt3 / 3, 0, dt2 / 2],
                [dt3 / 6, 0, dt2 / 2, 0, dt, 0],
                [0, dt3 / 6, 0, dt2 / 2, 0, dt],
            ]
        )





class CTRV:
    """
    Constant-Turn-Rate-and-Velocity model.

    State = [x, y, v, psi, psi_dot]

        * x, y    - position
        * v       - forward speed
        * psi     - yaw (heading, rad)
        * psi_dot - yaw rate (rad/s)

    This is the model used by the Apollo / Autoware autonomous-driving stacks
    and handles curved motion much better than CV or CA for pedestrians,
    cyclists and vehicles.
    """

    dim = 5
    name = "CTRV"
    _EPS = 1e-6

    @staticmethod
    def f(x: np.ndarray, dt: float) -> np.ndarray:
        px, py, v, psi, psid = x
        # Clamp yaw-rate to physically plausible range to prevent numerical
        # explosion of the sigma points (radius r = v / psid -> 0 as psid -> inf)
        psid = np.clip(psid, -3.0, 3.0)   # max 3 rad/s (~172 deg/s)
        v    = np.clip(v, -60.0, 60.0)    # |speed| <= 60 m/s (~216 km/h)
        nx = x.copy()
        nx[2] = v
        nx[4] = psid
        if abs(psid) > CTRV._EPS:
            nx[0] = px + (v / psid) * (np.sin(psi + psid * dt) - np.sin(psi))
            nx[1] = py + (v / psid) * (-np.cos(psi + psid * dt) + np.cos(psi))
        else:
            # Straight-line limit, avoids division by zero
            nx[0] = px + v * np.cos(psi) * dt
            nx[1] = py + v * np.sin(psi) * dt
        nx[3] = psi + psid * dt
        # Wrap yaw to [-pi, pi]
        nx[3] = (nx[3] + np.pi) % (2 * np.pi) - np.pi
        return nx

    @staticmethod
    def F(x: np.ndarray, dt: float) -> np.ndarray:
        _, _, v, psi, psid = x
        F = np.eye(5)
        if abs(psid) > CTRV._EPS:
            F[0, 2] = (np.sin(psi + psid * dt) - np.sin(psi)) / psid
            F[0, 3] = (v / psid) * (np.cos(psi + psid * dt) - np.cos(psi))
            F[0, 4] = (
                (v * dt / psid) * np.cos(psi + psid * dt)
                - (v / psid ** 2) * (np.sin(psi + psid * dt) - np.sin(psi))
            )
            F[1, 2] = (-np.cos(psi + psid * dt) + np.cos(psi)) / psid
            F[1, 3] = (v / psid) * (np.sin(psi + psid * dt) - np.sin(psi))
            F[1, 4] = (
                (v * dt / psid) * np.sin(psi + psid * dt)
                - (v / psid ** 2) * (-np.cos(psi + psid * dt) + np.cos(psi))
            )
        else:
            F[0, 2] = np.cos(psi) * dt
            F[0, 3] = -v * np.sin(psi) * dt
            F[1, 2] = np.sin(psi) * dt
            F[1, 3] = v * np.cos(psi) * dt
        F[3, 4] = dt
        return F

    @staticmethod
    def Q(dt: float, sigma_a: float = 1.0, sigma_psidd: float = 0.5) -> np.ndarray:
        """
        CTRV process noise, modelling white-noise longitudinal acceleration
        and yaw-acceleration. See Schubert et al. 2008.
        """
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        sa2 = sigma_a ** 2
        sp2 = sigma_psidd ** 2
        Q = np.zeros((5, 5))
        Q[0, 0] = 0.25 * dt4 * sa2
        Q[1, 1] = 0.25 * dt4 * sa2
        Q[2, 2] = dt2 * sa2
        Q[0, 2] = Q[2, 0] = 0.5 * dt3 * sa2
        Q[1, 2] = Q[2, 1] = 0.5 * dt3 * sa2
        Q[3, 3] = 0.25 * dt4 * sp2
        Q[4, 4] = dt2 * sp2
        Q[3, 4] = Q[4, 3] = 0.5 * dt3 * sp2
        return Q


MODEL_REGISTRY = {
    "CV": ConstantVelocity,
    "CA": ConstantAcceleration,
    "CTRV": CTRV,
}


def get_model(name: str):
    """Lookup a motion model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown motion model '{name}'. Choose from {list(MODEL_REGISTRY)}."
        )
    return MODEL_REGISTRY[name]
