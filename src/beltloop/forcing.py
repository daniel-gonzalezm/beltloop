"""Start-up profiles and resistance onset, as piecewise exosystems.

Every load history used here is, on each time interval ("chunk"), the output of a
linear autonomous system z' = E z (polynomials and harmonics). Augmenting the modal
equation with z makes each chunk an LTI system that is integrated exactly with the
matrix exponential, with or without damping and including exact resonance.

Dimensionless time tau = c_r t / L; accelerations scaled by the peak a_m, so the
peak of a_hat is 1 for every profile. Profiles (thesis, Ch. 2):
  sine:        a = a_m sin(pi t/t_a),           a_m = pi V / (2 t_a)
  triangular:  a = 2 a_m t/t_a, then 2 a_m (1 - t/t_a),  a_m = 2 V / t_a
  parabolic:   a = 4 a_m (t/t_a)(1 - t/t_a),    a_m = 3 V / (2 t_a)
Resistance onset phi(tau): 'velocity' (phi = V/V_inf), 'step' (phi = 1 for tau >= 0)
or 'none'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROFILES = ("sine", "triangular", "parabolic")
ONSETS = ("velocity", "step", "none")

# a_m * t_a / V_inf for each profile
PEAK_FACTOR = {"sine": np.pi / 2, "triangular": 2.0, "parabolic": 1.5}


@dataclass(frozen=True)
class Chunk:
    """Load history on [start, end): a_hat = ca @ z, phi = cphi @ z, z' = E z, z(start) = z0."""

    start: float
    end: float
    E: np.ndarray
    z0: np.ndarray
    ca: np.ndarray
    cphi: np.ndarray


def _poly_E(deg: int) -> np.ndarray:
    """z = (1, t, ..., t^deg), z' = E z."""
    E = np.zeros((deg + 1, deg + 1))
    for j in range(1, deg + 1):
        E[j, j - 1] = j
    return E


@dataclass(frozen=True)
class StartProfile:
    """Dimensionless start-up: drive acceleration a_hat(tau) and resistance onset phi(tau)."""

    kind: str = "sine"
    tau_a: float = 10.0
    onset: str = "velocity"

    def __post_init__(self):
        if self.kind not in PROFILES:
            raise ValueError(f"kind must be one of {PROFILES}")
        if self.onset not in ONSETS:
            raise ValueError(f"onset must be one of {ONSETS}")
        if not self.tau_a > 0:
            raise ValueError("tau_a must be positive")

    # --------------------------------------------------------------- closed forms
    def a(self, tau) -> np.ndarray:
        """Dimensionless drive acceleration a_hat."""
        u = np.asarray(tau, dtype=float) / self.tau_a
        on = (u >= 0) & (u <= 1)
        if self.kind == "sine":
            v = np.sin(np.pi * u)
        elif self.kind == "triangular":
            v = np.where(u <= 0.5, 2 * u, 2 * (1 - u))
        else:
            v = 4 * u * (1 - u)
        return np.where(on, v, 0.0)

    def da(self, tau) -> np.ndarray:
        """d a_hat / d tau (one-sided where the profile has corners)."""
        u = np.asarray(tau, dtype=float) / self.tau_a
        on = (u >= 0) & (u < 1)
        if self.kind == "sine":
            v = np.pi * np.cos(np.pi * u)
        elif self.kind == "triangular":
            v = np.where(u < 0.5, 2.0, -2.0)
        else:
            v = 4 * (1 - 2 * u)
        return np.where(on, v / self.tau_a, 0.0)

    def velocity(self, tau) -> np.ndarray:
        """V / V_inf."""
        u = np.clip(np.asarray(tau, dtype=float) / self.tau_a, 0.0, 1.0)
        if self.kind == "sine":
            return 0.5 * (1 - np.cos(np.pi * u))
        if self.kind == "triangular":
            return np.where(u <= 0.5, 2 * u ** 2, 1 - 2 * (1 - u) ** 2)
        return 3 * u ** 2 - 2 * u ** 3

    @property
    def a_integral(self) -> float:
        """int a_hat d tau = tau_a / PEAK_FACTOR (dimensionless V_inf)."""
        return self.tau_a / PEAK_FACTOR[self.kind]

    def phi(self, tau) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        if self.onset == "velocity":
            return self.velocity(tau)
        if self.onset == "step":
            return np.where(tau >= 0, 1.0, 0.0)
        return np.zeros_like(tau)

    def dphi(self, tau) -> np.ndarray:
        """d phi / d tau, excluding the Dirac impulse of the step onset at tau = 0."""
        if self.onset == "velocity":
            return self.a(tau) / self.a_integral
        return np.zeros_like(np.asarray(tau, dtype=float))

    # --------------------------------------------------------------- exosystem chunks
    def chunks(self) -> list[Chunk]:
        ta, on = self.tau_a, self.onset
        step = 1.0 if on == "step" else 0.0
        vel = 1.0 if on == "velocity" else 0.0
        out = []
        if self.kind == "sine":
            w = np.pi / ta
            E = np.array([[0.0, w, 0.0], [-w, 0.0, 0.0], [0.0, 0.0, 0.0]])   # (sin, cos, 1)
            out.append(Chunk(0.0, ta, E, np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.0, 0.0]),
                             np.array([0.0, -0.5 * vel, 0.5 * vel + step])))
        elif self.kind == "triangular":
            E = _poly_E(2)
            out.append(Chunk(0.0, 0.5 * ta, E, np.array([1.0, 0.0, 0.0]),
                             np.array([0.0, 2 / ta, 0.0]),
                             np.array([step, 0.0, 2 * vel / ta ** 2])))
            out.append(Chunk(0.5 * ta, ta, E, np.array([1.0, 0.0, 0.0]),
                             np.array([1.0, -2 / ta, 0.0]),
                             np.array([0.5 * vel + step, 2 * vel / ta, -2 * vel / ta ** 2])))
        else:
            E = _poly_E(3)
            out.append(Chunk(0.0, ta, E, np.array([1.0, 0.0, 0.0, 0.0]),
                             np.array([0.0, 4 / ta, -4 / ta ** 2, 0.0]),
                             np.array([step, 0.0, 3 * vel / ta ** 2, -2 * vel / ta ** 3])))
        out.append(Chunk(ta, np.inf, np.zeros((1, 1)), np.array([1.0]), np.array([0.0]),
                         np.array([vel + step])))
        return out
