"""Start-up response by modal superposition with a quasi-static (mode-acceleration) split.

Modal equations (Section "Model", Eq. modal):
    p_k'' + 2 zeta_k Om_k p_k' + Om_k^2 p_k = F_k(tau),  F_k = -Gamma_k a_hat - R_k phi,
    zeta_k = zeta_hat Om_k  (Kelvin-Voigt, stiffness-proportional).
Each p_k is integrated exactly (matrix exponential of the modal equation augmented with
the load exosystem).

Quasi-static split. Without inertia, a Kelvin-Voigt belt carries exactly the static
tension T_qs = a Q_mu + phi Q_r (Q = int_xi^x q), whatever the damping; only the strain
lags, through the first-order filter  2 zeta_hat f' + f = load,  whose time constant
2 zeta_hat is the same for every mode. With the lagged loads a_f, phi_f the quasi-static
modal coordinates are p_k^qs = -(Gamma_k a_f + R_k phi_f)/Om_k^2, and
    T = T_qs + sum_k [q_k + 2 zeta_hat q_k'] W_k',   q_k = p_k - p_k^qs,
    y = (a_f I_mu + phi_f I_r)/2 + sum_k q_k Y_k,     I = int_0^2 Q dx,
    beta y'' = -2 T(xi).
The remainder q_k is driven only by the rate of change of the quasi-static response and
converges fast with or without damping. (Using the elastic split p_k - F_k/Om_k^2 instead
leaves a remainder of order 2 zeta_hat F'/Om_k^2 that converges as slowly as the plain sum.)
For zeta_hat = 0, a_f = a and phi_f = phi.
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


def _lag_generator(zeta_hat: float, chunk) -> np.ndarray:
    """Generator for the lagged loads (a_f, phi_f): 2 zeta_hat f' + f = load. Shape (1, 2+nz, 2+nz)."""
    nz = len(chunk.z0)
    c = 1.0 / (2.0 * zeta_hat)
    H = np.zeros((1, 2 + nz, 2 + nz))
    H[0, 0, 0] = H[0, 1, 1] = -c
    H[0, 0, 2:] = c * chunk.ca
    H[0, 1, 2:] = c * chunk.cphi
    H[0, 2:, 2:] = chunk.E
    return H


def _propagate(profile: StartProfile, generator, n_sys: int, tau: np.ndarray) -> np.ndarray:
    """Exact propagation of n_sys stacked systems x' = H x, x = (two dynamic states, z),
    with the exosystem z re-initialised at each chunk start and the two dynamic states
    carried over. Returns the dynamic states at tau, shape (n_sys, 2, n_tau)."""
    tau = np.asarray(tau, dtype=float)
    if tau.ndim != 1 or np.any(np.diff(tau) < 0) or tau[0] < 0:
        raise ValueError("tau must be a sorted 1-D array of non-negative times")
    chunks = profile.chunks()
    out = np.zeros((n_sys, 2, tau.size))
    cache: dict = {}

    def prop(ci, X, dt):
        key = (ci, round(dt, 14))
        if key not in cache:
            cache[key] = expm(generator(chunks[ci]) * dt)
        return np.einsum("kij,kj->ki", cache[key], X)

    X = np.concatenate([np.zeros((n_sys, 2)), np.tile(chunks[0].z0, (n_sys, 1))], axis=1)
    t_now, ic = 0.0, 0
    for j, tj in enumerate(tau):
        while tj > chunks[ic].end:                      # cross chunk boundaries
            X = prop(ic, X, chunks[ic].end - t_now)
            t_now = chunks[ic].end
            ic += 1
            X = np.concatenate([X[:, :2], np.tile(chunks[ic].z0, (n_sys, 1))], axis=1)
        if tj > t_now:
            X = prop(ic, X, tj - t_now)
            t_now = tj
        out[:, :, j] = X[:, :2]
    return out


def integrate_modes(basis: ModalBasis, profile: StartProfile, zeta_hat: float, tau) -> tuple[np.ndarray, np.ndarray]:
    """Exact modal coordinates p_k(tau) and p_k'(tau) on a sorted time grid (tau >= 0).
    Returns arrays of shape (n_modes, n_tau)."""
    out = _propagate(profile, lambda ch: _modal_generators(basis, zeta_hat, ch), basis.n, tau)
    return out[:, 0, :], out[:, 1, :]


def lagged_loads(profile: StartProfile, zeta_hat: float, tau) -> tuple[np.ndarray, np.ndarray]:
    """Loads filtered by the Kelvin-Voigt lag, 2 zeta_hat f' + f = load, f(0) = 0:
    returns (a_f, phi_f). For zeta_hat = 0 they equal a_hat and phi."""
    tau = np.asarray(tau, dtype=float)
    if zeta_hat == 0:
        return profile.a(tau), profile.phi(tau)
    out = _propagate(profile, lambda ch: _lag_generator(zeta_hat, ch), 1, tau)
    return out[0, 0], out[0, 1]


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

    def _lagged(self):
        if not hasattr(self, "_lag_cache"):
            self._lag_cache = lagged_loads(self.profile, self.zeta_hat, self.tau)
        return self._lag_cache

    def _residual(self):
        """q_k = p_k - p_k^qs and q_k' (see module docstring)."""
        b = self.basis
        af, phif = self._lagged()
        Om2 = b.Om[:, None] ** 2
        pqs = -(b.Gamma[:, None] * af[None, :] + b.R[:, None] * phif[None, :]) / Om2
        if self.zeta_hat == 0:
            da, dphi = self.profile.da(self.tau), self.profile.dphi(self.tau)
        else:
            c = 1.0 / (2.0 * self.zeta_hat)
            da, dphi = c * (self.a - af), c * (self.phi - phif)
        dpqs = -(b.Gamma[:, None] * da[None, :] + b.R[:, None] * dphi[None, :]) / Om2
        return self.p - pqs, self.dp - dpqs

    # ---------------------------------------------------------------- outputs
    def tension(self, x, method: str = "mode-acceleration") -> np.ndarray:
        """Dynamic tension T_hat(x, tau) (elastic + viscous), shape (n_x, n_tau).
        method='plain' gives the plain modal sum (for convergence checks)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        dW = self.basis.dW(x)                         # (n_modes, n_x)
        zh = self.zeta_hat
        if method == "plain":
            return dW.T @ (self.p + 2 * zh * self.dp)
        if method != "mode-acceleration":
            raise ValueError("method must be 'mode-acceleration' or 'plain'")
        q, dq = self._residual()
        return self.quasi_static_tension(x) + dW.T @ (q + 2 * zh * dq)

    def quasi_static_tension(self, x) -> np.ndarray:
        """Quasi-static tension a Q_mu(x) + phi Q_r(x): exact without inertia for any damping;
        reference for dynamic amplification."""
        lp = self.basis.loop
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.outer(lp.quasi_static("mu", x), self.a) + np.outer(lp.quasi_static("r", x), self.phi)

    def takeup_displacement(self) -> np.ndarray:
        """y_hat(tau), positive when the loop lengthens (counterweight descends)."""
        lp, b = self.basis.loop, self.basis
        q, _ = self._residual()
        af, phif = self._lagged()
        static = 0.5 * (af * lp.quasi_static_integral("mu") + phif * lp.quasi_static_integral("r"))
        return static + b.Y @ q

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
