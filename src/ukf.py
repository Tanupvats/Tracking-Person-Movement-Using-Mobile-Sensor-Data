

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.linalg import cholesky
from scipy.stats import chi2



class MerweSigmaPoints:
    """
    Generates 2n+1 scaled sigma points following Van der Merwe (2004).

    Parameters
    ----------
    n : int
        State dimension.
    alpha : float
        Spread of points around the mean (1e-3 .. 1).
    beta : float
        Prior knowledge of state distribution; 2 is optimal for Gaussian.
    kappa : float
        Secondary scaling, typically 3-n or 0.
    """

    def __init__(self, n: int, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha ** 2 * (n + kappa) - n
        self.num_sigmas = 2 * n + 1
        self._compute_weights()

    def _compute_weights(self) -> None:
        n, lam = self.n, self.lam
        c = 0.5 / (n + lam)
        self.Wm = np.full(2 * n + 1, c)
        self.Wc = np.full(2 * n + 1, c)
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - self.alpha ** 2 + self.beta)

    def sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        n, lam = self.n, self.lam
        # Symmetrise and add small jitter for numerical stability
        cov_sym = 0.5 * (cov + cov.T)
        cov_jit = cov_sym + 1e-9 * np.eye(n)
        try:
            L = cholesky((n + lam) * cov_jit, lower=True)
        except np.linalg.LinAlgError:
            # Stronger regularisation: eigendecomposition with floor.
            w, V = np.linalg.eigh(cov_sym)
            w = np.clip(w, 1e-6, None)
            cov_fix = (V * w) @ V.T
            L = cholesky((n + lam) * cov_fix + 1e-6 * np.eye(n), lower=True)
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = mean
        for i in range(n):
            sigmas[i + 1] = mean + L[:, i]
            sigmas[n + i + 1] = mean - L[:, i]
        return sigmas




@dataclass
class UKFDiagnostics:
    """Collected diagnostics emitted by the filter at every step."""

    nis: list = field(default_factory=list)          # Normalised innovation sq.
    innovation: list = field(default_factory=list)   # Raw innovation vectors
    rejected: list = field(default_factory=list)     # Indices of rejected meas.
    trace_P: list = field(default_factory=list)      # Trace of covariance
    likelihood: list = field(default_factory=list)   # Log-likelihood


class UKF:
    """
    Additive-noise Unscented Kalman Filter.

    The filter is deliberately *model-agnostic*: the user provides the state-
    transition function ``fx(x, dt)`` and the measurement function ``hx(x)``.
    State and measurement can be subject to angle wrapping via
    ``residual_x`` / ``residual_z`` callables.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        fx: Callable[[np.ndarray, float], np.ndarray],
        hx: Callable[[np.ndarray], np.ndarray],
        sigma: Optional[MerweSigmaPoints] = None,
        residual_x: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        residual_z: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        mean_x: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        gate_alpha: float = 0.99,
        adaptive: bool = False,
        adaptive_window: int = 30,
    ):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.sigma = sigma or MerweSigmaPoints(dim_x)

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x) * 1e-3
        self.R = np.eye(dim_z) * 1e-1

        # Residual / mean functions allow circular-quantity handling
        self.residual_x = residual_x or (lambda a, b: a - b)
        self.residual_z = residual_z or (lambda a, b: a - b)
        self.mean_x = mean_x or self._default_weighted_mean
        self.mean_z = self._default_weighted_mean

        # Outlier rejection: chi-square upper tail threshold
        self.gate_threshold = chi2.ppf(gate_alpha, dim_z)

        # Adaptive R
        self.adaptive = adaptive
        self.adaptive_window = adaptive_window
        self._innov_history: list[np.ndarray] = []

        # Book-keeping
        self.diag = UKFDiagnostics()

        # Storage for sigma points (populated during predict)
        self._sigmas_f = np.zeros((self.sigma.num_sigmas, dim_x))



    @staticmethod
    def _default_weighted_mean(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
        return Wm @ sigmas

    @staticmethod
    def _joseph_update(
        P: np.ndarray, K: np.ndarray, H_eq: np.ndarray, R: np.ndarray
    ) -> np.ndarray:
        """Joseph-form covariance update (symmetric, PSD by construction)."""
        I_KH = np.eye(P.shape[0]) - K @ H_eq
        return I_KH @ P @ I_KH.T + K @ R @ K.T


    def predict(self, dt: float) -> None:
        sigmas = self.sigma.sigma_points(self.x, self.P)
        for i in range(sigmas.shape[0]):
            self._sigmas_f[i] = self.fx(sigmas[i], dt)

        self.x = self.mean_x(self._sigmas_f, self.sigma.Wm)

        P = np.zeros_like(self.P)
        for i in range(self._sigmas_f.shape[0]):
            dx = self.residual_x(self._sigmas_f[i], self.x)
            P += self.sigma.Wc[i] * np.outer(dx, dx)
        self.P = P + self.Q
        self._symmetrise_P()

    def update(self, z: np.ndarray) -> bool:
        """
        Update step. Returns True if measurement was accepted, False if
        rejected by the NIS gate.
        """
        # Transform sigmas to measurement space
        sigmas_h = np.zeros((self._sigmas_f.shape[0], self.dim_z))
        for i in range(self._sigmas_f.shape[0]):
            sigmas_h[i] = self.hx(self._sigmas_f[i])

        z_pred = self.mean_z(sigmas_h, self.sigma.Wm)

        # Innovation covariance S
        S = np.zeros((self.dim_z, self.dim_z))
        for i in range(self._sigmas_f.shape[0]):
            dz = self.residual_z(sigmas_h[i], z_pred)
            S += self.sigma.Wc[i] * np.outer(dz, dz)
        S += self.R

        # Cross-covariance
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(self._sigmas_f.shape[0]):
            dx = self.residual_x(self._sigmas_f[i], self.x)
            dz = self.residual_z(sigmas_h[i], z_pred)
            Pxz += self.sigma.Wc[i] * np.outer(dx, dz)

        y = self.residual_z(z, z_pred)  # innovation

        # ---- NIS gate (chi-square outlier rejection) --------------------
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # S is singular -> regularise and continue
            S += 1e-6 * np.eye(self.dim_z)
            S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)
        self.diag.nis.append(nis)
        self.diag.innovation.append(y.copy())

        if nis > self.gate_threshold * 4.0:
            # Reject only truly egregious outliers (4x the chi-sq threshold)
            self.diag.rejected.append(len(self.diag.nis) - 1)
            self.diag.trace_P.append(float(np.trace(self.P)))
            self.diag.likelihood.append(-0.5 * nis)
            return False

        # Kalman gain
        K = Pxz @ S_inv
        self.x = self.x + K @ y

        # Joseph-form update requires an equivalent H.
        # For UKF an exact H does not exist, so we use K @ S = Pxz => H_eq @ P = Pxz.T
        H_eq = (S_inv @ Pxz.T)  # yields Pxz^T P^{-1} approximately
        # Safer alternative: directly apply P = P - K @ S @ K.T and then
        # symmetrise; this is the standard UKF covariance update.
        self.P = self.P - K @ S @ K.T
        self._symmetrise_P()

        # ---- Adaptive R (Mehra-style) -----------------------------------
        if self.adaptive:
            self._innov_history.append(y.copy())
            if len(self._innov_history) > self.adaptive_window:
                self._innov_history.pop(0)
                innov_arr = np.stack(self._innov_history)
                C_hat = (innov_arr.T @ innov_arr) / len(innov_arr)
                # R_hat = C_hat - H P_pred H^T  -> approximated via S - K? Simpler:
                R_new = np.diag(np.maximum(np.diag(C_hat) * 0.5, 1e-3))
                # Blend (low-pass)
                self.R = 0.95 * self.R + 0.05 * R_new

        self.diag.trace_P.append(float(np.trace(self.P)))
        self.diag.likelihood.append(
            -0.5 * (nis + np.log(np.linalg.det(2 * np.pi * S) + 1e-12))
        )
        return True

    # ------------------------------------------------------------------ #
    def _symmetrise_P(self) -> None:
        self.P = 0.5 * (self.P + self.P.T)


    def copy_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.P.copy()

    def set_state(self, x: np.ndarray, P: np.ndarray) -> None:
        self.x = x.copy()
        self.P = P.copy()
