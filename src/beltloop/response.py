"""Start-up response by modal superposition with the mode-acceleration correction.

Modal equations (Section "Model", Eq. modal):
    p_k'' + 2 zeta_k Om_k p_k' + Om_k^2 p_k = F_k(tau),  F_k = -Gamma_k a_hat - R_k phi,
    zeta_k = zeta_hat Om_k  (Kelvin-Voigt, stiffness-proportional).
Each p_k is integrated exactly (matrix exponential of the modal equation augmented with
the load exosystem). The dynamic tension uses the mode-acceleration form
    T = a Q_mu + phi Q_r + 2 zeta_hat (a' Q_mu + phi' Q_r)
        + sum_k [(p_k - p_k^qs) + 2 zeta_hat (p_k' - p_k^qs')] W_k',
    p_k^qs = F_k / Om_k^2,
whose first line is the exact quasi-static tension (take-up holds T = 0) and whose
modal remainder converges much faster than the plain sum. The take-up motion is
y = (a I_mu + phi I_r)/2 + sum_k (p_k - p_k^qs) Y_k, and beta y'' = -2 T(xi).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .eigen import ModalBasis
from .forcing import StartProfile


def _modal_generators(basis: ModalBasis, zeta_hat: float, chunk) -> np.ndarray:
    """Stacked augmented generators H_k for one chunk, shape (n_modes, 2+nz, 2+nz)."""
    n, nz = basis.n, len(chunk.z0)
    Om = basis.Om
    H = np.zeros((n, 2 + nz, 2 + nz))
    H[:, 0, 1] = 1.0
    H[:, 1, 0] = -Om ** 2
    H[:, 1, 1] = -2.0 * zeta_hat * Om ** 2
    H[:, 1, 2:] = -(basis.Gamma[:, None] * chunk.ca[None, :] + basis.R[:, None] * chunk.cphi[None, :])
    H[:, 2:, 2:] = chunk.E
    return H


def integrate_modes(basis: ModalBasis, profile: StartProfile, zeta_hat: float, tau) -> tuple[np.ndarray, np.ndarray]:
    """Exact modal coordinates p_k(tau) and p_k'(tau) on a sorted time grid (tau >= 0).
    Returns arrays of shape (n_modes, n_tau)."""
    tau = np.asarray(tau, dtype=float)
    if tau.ndim != 1 or np.any(np.diff(tau) < 0) or tau[0] < 0:
        raise ValueError("tau must be a sorted 1-D array of non-negative times")
    n = basis.n
    P = np.zeros((n, tau.size)); dP = np.zeros((n, tau.size))
    chunks = profile.chunks()
    q = np.zeros((n, 2))                 # (p, p') at time t_now
    t_now, ic = 0.0, 0
    cache: dict = {}

    def prop(ci, X, dt):
        key = (ci, round(dt, 14))
        if key not in cache:
            cache[key] = expm(_modal_generators(basis, zeta_hat, chunks[ci]) * dt)
        return np.einsum("kij,kj->ki", cache[key], X)

    # state including the exosystem of the current chunk
    X = np.concatenate([q, np.tile(chunks[0].z0, (n, 1))], axis=1)
    for j, tj in enumerate(tau):
        while tj > chunks[ic].end:                      # cross chunk boundaries
            X = prop(ic, X, chunks[ic].end - t_now)
            t_now = chunks[ic].end
            ic += 1
            X = np.concatenate([X[:, :2], np.tile(chunks[ic].z0, (n, 1))], axis=1)
        if tj > t_now:
            X = prop(ic, X, tj - t_now)
            t_now = tj
        P[:, j], dP[:, j] = X[:, 0], X[:, 1]
    return P, dP


@dataclass
class StartupResponse:
    """Dimensionless start-up response. Tensions in units of mu_r L a_m, displacements in
    units of a_m L^2 / c_r^2, take-up acceleration in units of a_m."""

    basis: ModalBasis
    profile: StartProfile
    zeta_hat: float
    tau: np.ndarray
    p: np.ndarray
    dp: np.ndarray

    # ---------------------------------------------------------------- loads
    @property
    def a(self) -> np.ndarray:
        return self.profile.a(self.tau)

    @property
    def phi(self) -> np.ndarray:
        return self.profile.phi(self.tau)

    def _residual(self):
        b = self.basis
        F = -(b.Gamma[:, None] * self.a[None, :] + b.R[:, None] * self.phi[None, :])
        dF = -(b.Gamma[:, None] * self.profile.da(self.tau)[None, :]
               + b.R[:, None] * self.profile.dphi(self.tau)[None, :])
        Om2 = b.Om[:, None] ** 2
        return self.p - F / Om2, self.dp - dF / Om2

    # ---------------------------------------------------------------- outputs
    def tension(self, x, method: str = "mode-acceleration") -> np.ndarray:
        """Dynamic tension T_hat(x, tau), shape (n_x, n_tau). method='plain' gives the
        plain modal sum (for convergence checks)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        dW = self.basis.dW(x)                         # (n_modes, n_x)
        zh = self.zeta_hat
        if method == "plain":
            return dW.T @ (self.p + 2 * zh * self.dp)
        if method != "mode-acceleration":
            raise ValueError("method must be 'mode-acceleration' or 'plain'")
        lp = self.basis.loop
        Qm, Qr = lp.quasi_static("mu", x), lp.quasi_static("r", x)
        pr, dpr = self._residual()
        a, phi = self.a, self.phi
        da, dphi = self.profile.da(self.tau), self.profile.dphi(self.tau)
        qs = np.outer(Qm, a + 2 * zh * da) + np.outer(Qr, phi + 2 * zh * dphi)
        return qs + dW.T @ (pr + 2 * zh * dpr)

    def quasi_static_tension(self, x) -> np.ndarray:
        """Reference for dynamic amplification: a Q_mu(x) + phi Q_r(x) (elastic part)."""
        lp = self.basis.loop
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.outer(lp.quasi_static("mu", x), self.a) + np.outer(lp.quasi_static("r", x), self.phi)

    def takeup_displacement(self) -> np.ndarray:
        """y_hat(tau), positive when the loop lengthens (counterweight descends)."""
        lp, b = self.basis.loop, self.basis
        pr, _ = self._residual()
        static = 0.5 * (self.a * lp.quasi_static_integral("mu") + self.phi * lp.quasi_static_integral("r"))
        return static + b.Y @ pr

    def takeup_acceleration(self) -> np.ndarray:
        """y_hat''(tau) from the take-up equation beta y'' = -2 T(xi)."""
        T_xi = self.tension([self.basis.loop.xi])[0]
        return -2.0 * T_xi / self.basis.beta


def startup_response(basis: ModalBasis, profile: StartProfile, zeta_hat: float, tau) -> StartupResponse:
    """Integrate the modal equations and wrap the result."""
    P, dP = integrate_modes(basis, profile, zeta_hat, tau)
    return StartupResponse(basis, profile, zeta_hat, np.asarray(tau, dtype=float), P, dP)


def residual_amplitude_sine(basis: ModalBasis, tau_a: float) -> np.ndarray:
    """Undamped free amplitude P_k left in each mode by the sine acceleration phase
    (Eq. residual), inertial load only."""
    w = np.pi / tau_a
    Om, G = basis.Om, np.abs(basis.Gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = G / Om * 2 * w * np.abs(np.cos(Om * tau_a / 2)) / np.abs(w ** 2 - Om ** 2)
    return np.where(np.isclose(Om, w, rtol=1e-10), G * tau_a / (2 * Om), P)
