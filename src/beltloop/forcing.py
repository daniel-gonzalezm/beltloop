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

PiecewiseProfile covers piecewise-polynomial accelerations (linear start with an initial
speed offset, linear start with a rest period; Lodewijks 1996, Eqs. 8.35-8.36), scaled by
a reference acceleration chosen by the caller.
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


@dataclass(frozen=True)
class PiecewiseProfile:
    """Start-up with a piecewise-polynomial drive acceleration (Lodewijks 1996, Sec. 8.6.2).

    On [breaks[i], breaks[i+1]) the dimensionless acceleration is the polynomial
    a_hat = sum_j coeffs[i][j] u**j in the local time u = tau - breaks[i]; a_hat = 0 after
    breaks[-1]. Velocity jumps are not represented: a jump is replaced by a short ramp
    (see :meth:`linear`), so that the load stays bounded and the exosystem stays exact.
    The acceleration scale a_m is the caller's choice (the SI helpers use the mean
    acceleration V_inf/t_a); the final speed in units of a_m L/c_r is ``a_integral``.
    onset as in StartProfile ('velocity': phi = V/V_inf; 'step'; 'none').
    """

    breaks: tuple
    coeffs: tuple
    onset: str = "velocity"

    def __post_init__(self):
        b = np.asarray(self.breaks, dtype=float)
        if b.ndim != 1 or b.size < 2 or b[0] != 0.0 or np.any(np.diff(b) <= 0):
            raise ValueError("breaks must start at 0 and increase strictly")
        if len(self.coeffs) != b.size - 1:
            raise ValueError("one coefficient array per interval is required")
        if self.onset not in ONSETS:
            raise ValueError(f"onset must be one of {ONSETS}")
        object.__setattr__(self, "breaks", tuple(float(x) for x in b))
        object.__setattr__(self, "coeffs", tuple(tuple(float(c) for c in cs) for cs in self.coeffs))
        if not self.a_integral > 0:
            raise ValueError("the profile must end at a positive speed")

    # --------------------------------------------------------------- builders
    @classmethod
    def linear(cls, tau_a: float, v0: float = 0.0, tau_j: float = 0.0,
               onset: str = "velocity") -> "PiecewiseProfile":
        """Linear speed increase over tau_a (Lodewijks Eq. 8.35), scaled by the mean
        acceleration (a_hat = 1 for v0 = 0). v0 = V_0/V_inf is the initial speed offset;
        the jump to v0 is a ramp of length tau_j superposed on the linear increase, so the
        speed equals V_0 + (V_inf - V_0) tau/tau_a exactly for tau >= tau_j."""
        if not 0.0 <= v0 < 1.0:
            raise ValueError("v0 must lie in [0, 1)")
        base = 1.0 - v0
        if v0 == 0.0:
            return cls((0.0, tau_a), ((base,),), onset)
        if not 0.0 < tau_j < tau_a:
            raise ValueError("an offset needs a jump ramp 0 < tau_j < tau_a")
        return cls((0.0, tau_j, tau_a), ((base + v0 * tau_a / tau_j,), (base,)), onset)

    @classmethod
    def delayed(cls, tau_a: float, tau_r: float, dtau: float,
                onset: str = "velocity") -> "PiecewiseProfile":
        """Linear increase with a rest period (Lodewijks Eq. 8.36): a_hat = 1 on [0, tau_r),
        0 on [tau_r, tau_r + dtau), 1 on [tau_r + dtau, tau_a + dtau)."""
        if not (0.0 < tau_r < tau_a and dtau > 0.0):
            raise ValueError("need 0 < tau_r < tau_a and dtau > 0")
        return cls((0.0, tau_r, tau_r + dtau, tau_a + dtau), ((1.0,), (0.0,), (1.0,)), onset)

    # --------------------------------------------------------------- closed forms
    def _locate(self, tau):
        tau = np.asarray(tau, dtype=float)
        b = np.asarray(self.breaks)
        i = np.clip(np.searchsorted(b, tau, side="right") - 1, 0, len(self.coeffs) - 1)
        inside = (tau >= 0.0) & (tau < b[-1])
        return tau, i, tau - b[i], inside

    def a(self, tau) -> np.ndarray:
        tau, i, u, inside = self._locate(tau)
        out = np.zeros_like(tau)
        for k, cs in enumerate(self.coeffs):
            m = inside & (i == k)
            out[m] = np.polynomial.polynomial.polyval(u[m], cs)
        return out

    def da(self, tau) -> np.ndarray:
        """d a_hat / d tau (right derivative at the breaks)."""
        tau, i, u, inside = self._locate(tau)
        out = np.zeros_like(tau)
        for k, cs in enumerate(self.coeffs):
            m = inside & (i == k)
            out[m] = np.polynomial.polynomial.polyval(u[m], np.polynomial.polynomial.polyder(cs))
        return out

    def _speed_at_breaks(self) -> np.ndarray:
        b = np.asarray(self.breaks)
        v = [0.0]
        for k, cs in enumerate(self.coeffs):
            v.append(v[-1] + np.polynomial.polynomial.polyval(b[k + 1] - b[k],
                                                              np.polynomial.polynomial.polyint(cs)))
        return np.array(v)

    @property
    def a_integral(self) -> float:
        """Final speed, int a_hat d tau (units a_m L / c_r)."""
        return float(self._speed_at_breaks()[-1])

    @property
    def tau_end(self) -> float:
        return self.breaks[-1]

    def velocity(self, tau) -> np.ndarray:
        """V / V_inf."""
        tau, i, u, inside = self._locate(tau)
        vb = self._speed_at_breaks()
        out = np.where(tau >= self.breaks[-1], vb[-1], 0.0)
        for k, cs in enumerate(self.coeffs):
            m = inside & (i == k)
            out[m] = vb[k] + np.polynomial.polynomial.polyval(u[m], np.polynomial.polynomial.polyint(cs))
        return out / vb[-1]

    def displacement(self, tau) -> np.ndarray:
        """Drive displacement int_0^tau V d tau' (units a_m L^2 / c_r^2)."""
        P = np.polynomial.polynomial
        tau, i, u, inside = self._locate(tau)
        b = np.asarray(self.breaks)
        vb = self._speed_at_breaks()
        db = [0.0]
        for k, cs in enumerate(self.coeffs):
            db.append(db[-1] + vb[k] * (b[k + 1] - b[k]) + P.polyval(b[k + 1] - b[k], P.polyint(cs, 2)))
        out = np.where(tau >= b[-1], db[-1] + vb[-1] * (tau - b[-1]), 0.0)
        for k, cs in enumerate(self.coeffs):
            m = inside & (i == k)
            out[m] = db[k] + vb[k] * u[m] + P.polyval(u[m], P.polyint(cs, 2))
        return out

    def phi(self, tau) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        if self.onset == "velocity":
            return self.velocity(tau)
        if self.onset == "step":
            return np.where(tau >= 0, 1.0, 0.0)
        return np.zeros_like(tau)

    def dphi(self, tau) -> np.ndarray:
        if self.onset == "velocity":
            return self.a(tau) / self.a_integral
        return np.zeros_like(np.asarray(tau, dtype=float))

    # --------------------------------------------------------------- exosystem chunks
    def chunks(self) -> list[Chunk]:
        vb = self._speed_at_breaks()
        vinf = vb[-1]
        vel = 1.0 if self.onset == "velocity" else 0.0
        step = 1.0 if self.onset == "step" else 0.0
        out = []
        for k, cs in enumerate(self.coeffs):
            d = len(cs)                       # phi has degree d (a has degree d - 1)
            E = _poly_E(d)
            ca = np.r_[cs, 0.0]
            cphi = vel * np.r_[vb[k], [c / (j + 1) for j, c in enumerate(cs)]] / vinf
            cphi[0] += step
            out.append(Chunk(self.breaks[k], self.breaks[k + 1], E, np.eye(d + 1)[0], ca, cphi))
        out.append(Chunk(self.breaks[-1], np.inf, np.zeros((1, 1)), np.array([1.0]),
                         np.array([0.0]), np.array([vel + step])))
        return out
