

from __future__ import annotations

from typing import Sequence

import numpy as np


class IMMFilter:
    """
    Manage ``M`` sub-filters with a Markov-chain model-switching probability.

    Parameters
    ----------
    filters : sequence of UKF
        Sub-filters. Each must implement ``predict(dt)`` and ``update(z)``
        and expose ``x``, ``P``, and ``dim_x``.
    mu : np.ndarray of shape (M,)
        Prior model probabilities, summing to 1.
    M_trans : np.ndarray of shape (M, M)
        Markov transition matrix. ``M_trans[i, j]`` = P(model j at k | model i at k-1).
    lift : list of callables, optional
        For heterogeneous states, provides ``lift[i](x_common) -> x_i`` and
        ``project[i](x_i) -> x_common``. Not needed when all filters share
        the same state dimension.
    """

    def __init__(
        self,
        filters: Sequence,
        mu: np.ndarray,
        M_trans: np.ndarray,
    ):
        self.filters = list(filters)
        self.M = len(self.filters)
        self.mu = np.asarray(mu, dtype=float)
        self.M_trans = np.asarray(M_trans, dtype=float)
        assert self.mu.shape == (self.M,)
        assert self.M_trans.shape == (self.M, self.M)
        assert np.allclose(self.mu.sum(), 1.0, atol=1e-6)
        assert np.allclose(self.M_trans.sum(axis=1), 1.0, atol=1e-6)

        self.likelihoods = np.ones(self.M)
        self.mode_history: list[np.ndarray] = []
        # Initialise mu_bar to current mu so update() works before any predict()
        self.mu_bar = self.mu.copy()



    @staticmethod
    def _pose(f) -> np.ndarray:
        """Return [x, y, vx, vy] of a filter regardless of its underlying model."""
        if f.dim_x == 5:  # CTRV: [x, y, v, psi, psi_dot]
            x, y, v, psi, _ = f.x
            return np.array([x, y, v * np.cos(psi), v * np.sin(psi)])
        elif f.dim_x == 4:  # CV
            return f.x[:4].copy()
        elif f.dim_x == 6:  # CA
            return f.x[:4].copy()
        else:
            return f.x[:4].copy()

    @staticmethod
    def _pose_cov(f) -> np.ndarray:
        """Return 4x4 pose covariance [x, y, vx, vy]."""
        if f.dim_x == 5:
            x, y, v, psi, _ = f.x
            J = np.zeros((4, 5))
            J[0, 0] = 1
            J[1, 1] = 1
            J[2, 2] = np.cos(psi)
            J[2, 3] = -v * np.sin(psi)
            J[3, 2] = np.sin(psi)
            J[3, 3] = v * np.cos(psi)
            return J @ f.P @ J.T
        else:
            return f.P[:4, :4].copy()



    def predict(self, dt: float) -> None:
        """Mixing + prediction."""
        # Mixing probabilities mu_{i|j} = M_trans[i,j] * mu[i] / c_bar[j]
        c_bar = self.M_trans.T @ self.mu  # normalisation for each target j
        mix_prob = (self.M_trans * self.mu[:, None]) / (c_bar[None, :] + 1e-12)

        # Mix each filter's state/cov in pose space
        mixed_poses = np.zeros((self.M, 4))
        mixed_covs = np.zeros((self.M, 4, 4))

        poses = np.stack([self._pose(f) for f in self.filters])       # (M, 4)
        pcovs = np.stack([self._pose_cov(f) for f in self.filters])   # (M, 4, 4)

        for j in range(self.M):
            w = mix_prob[:, j]  # column j
            mixed_poses[j] = w @ poses
            P_j = np.zeros((4, 4))
            for i in range(self.M):
                dy = poses[i] - mixed_poses[j]
                P_j += w[i] * (pcovs[i] + np.outer(dy, dy))
            mixed_covs[j] = P_j

        # Inject mixed pose into each filter's native state (keep non-pose
        # components of the existing posterior).
        for j, f in enumerate(self.filters):
            self._set_pose(f, mixed_poses[j], mixed_covs[j])

        # Prior model probabilities for the new step
        self.mu_bar = c_bar

        # Predict each filter
        for f in self.filters:
            f.predict(dt)

    def update(self, z: np.ndarray) -> None:
        """Update each filter and recompute mixing weights."""
        log_likelihoods = np.zeros(self.M)
        for j, f in enumerate(self.filters):
            # Snapshot to compute likelihood from the innovation
            accepted = f.update(z)
            if accepted and len(f.diag.likelihood) > 0:
                log_likelihoods[j] = f.diag.likelihood[-1]
            else:
                # Treat rejected observation as low likelihood
                log_likelihoods[j] = -50.0

        # Normalise likelihoods in log-domain for numerical stability
        log_likelihoods -= log_likelihoods.max()
        likelihoods = np.exp(log_likelihoods)

        # Posterior mode probabilities
        w = likelihoods * self.mu_bar
        w /= w.sum() + 1e-12
        self.mu = w
        self.mode_history.append(self.mu.copy())



    def combined_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the mixed posterior in pose space."""
        poses = np.stack([self._pose(f) for f in self.filters])
        pcovs = np.stack([self._pose_cov(f) for f in self.filters])
        mu = self.mu[:, None]
        mean = (mu * poses).sum(axis=0)
        cov = np.zeros((4, 4))
        for i in range(self.M):
            dy = poses[i] - mean
            cov += self.mu[i] * (pcovs[i] + np.outer(dy, dy))
        return mean, cov



    def _set_pose(self, f, pose: np.ndarray, pose_cov: np.ndarray) -> None:
        px, py, vx, vy = pose
        if f.dim_x == 5:  # CTRV
            v = np.hypot(vx, vy)
            psi = np.arctan2(vy, vx) if v > 0.5 else f.x[3]
            f.x[0] = px
            f.x[1] = py
            f.x[2] = v
            f.x[3] = psi
            # Rebuild covariance: use the pose-space cov for position, map
            # velocity-cov into (v, psi) via linearisation, keep psi_dot cov.
            # Floor v in the Jacobian at 1 m/s to avoid 1/v singularity at rest;
            # at low speed heading is unobservable so we also inflate psi var.
            P_new = np.zeros((5, 5))
            P_new[:2, :2] = pose_cov[:2, :2]
            v_floor = max(v, 1.0)
            J = np.array([
                [np.cos(psi),             np.sin(psi)],
                [-np.sin(psi) / v_floor,  np.cos(psi) / v_floor],
            ])
            P_vv = J @ pose_cov[2:4, 2:4] @ J.T
            P_new[2, 2] = P_vv[0, 0]
            P_new[3, 3] = P_vv[1, 1] + max(0.0, 1.0 - v) * 0.5
            # Skip pos-vel cross terms (small and source of numerical issues)
            P_new[4, 4] = max(f.P[4, 4], 0.1)
            f.P = 0.5 * (P_new + P_new.T) + 1e-6 * np.eye(5)
        elif f.dim_x == 4:  # CV
            f.x[:4] = pose
            f.P = 0.5 * (pose_cov + pose_cov.T) + 1e-6 * np.eye(4)
        elif f.dim_x == 6:  # CA
            f.x[:4] = pose
            # Preserve acceleration block, replace pose block
            P_new = f.P.copy()
            P_new[:4, :4] = pose_cov
            # Zero out cross terms between pose and accel (we don't have them)
            P_new[:4, 4:] = 0
            P_new[4:, :4] = 0
            # Ensure acc block PD
            P_new[4:, 4:] = 0.5 * (P_new[4:, 4:] + P_new[4:, 4:].T) + 1e-6 * np.eye(2)
            f.P = 0.5 * (P_new + P_new.T) + 1e-6 * np.eye(6)
        else:
            f.x[:4] = pose
            f.P[:4, :4] = pose_cov
