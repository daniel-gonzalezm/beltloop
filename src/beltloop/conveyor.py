"""Dimensional conveyor: static state, take-up reeving, scaling and validity checks.

Geometric coordinate sigma (m) from the head pulley along belt travel: return strand
[0, L), carry strand [L, 2L). The carry-strand elevation is given as a piecewise-linear
profile measured from the tail; the return strand runs beneath it, h_r(sigma) =
h_c(L - sigma). Gravity enters only the static state (Section "Model").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.constants import g as G_STD

from .eigen import modal_basis
from .forcing import PEAK_FACTOR, StartProfile
from .loop import Loop
from .response import StartupResponse, startup_response


@dataclass(frozen=True)
class GravityTakeUp:
    """Counterweight M_w, pulley + carriage M_c, reeving ratio i (counterweight travel per
    unit carriage travel; i = 1 for a directly hung counterweight), kappa = sine of the
    carriage inclination (1 vertical, 0 horizontal), and n = strands: number of belt
    strands that carry the carriage (2 for a single loop around one take-up pulley, 4 for
    a double loop as in Harrison (1985), Fig. 2a).

    With n strands the belt kinematics is n:1: a carriage travel y changes the belt length
    stored in the loop by n y, and the carriage equation is M y'' = force - n T. The
    dimensionless model is written for n = 2; any n maps onto it exactly with the
    belt-side mass M_belt = 4 M / n**2 and the belt-side travel (n/2) y (the kinetic
    energy M y'^2 / 2 is preserved)."""

    M_w: float
    M_c: float = 0.0
    i: float = 1.0
    kappa: float = 1.0
    g: float = G_STD
    strands: int = 2

    def __post_init__(self):
        if int(self.strands) != self.strands or self.strands < 2 or self.strands % 2:
            raise ValueError("strands must be an even integer >= 2")

    @property
    def force(self) -> float:
        """Static force on the carriage, n T_t = g (i M_w + kappa M_c)."""
        return self.g * (self.i * self.M_w + self.kappa * self.M_c)

    @property
    def T_t(self) -> float:
        """Static belt tension at the take-up, force / n."""
        return self.force / self.strands

    @property
    def M(self) -> float:
        """Effective mass referred to the carriage displacement, M = M_c + i^2 M_w."""
        return self.M_c + self.i ** 2 * self.M_w

    @property
    def M_belt(self) -> float:
        """Mass seen by the belt in the 2:1 formulation, 4 M / n^2 (= M for n = 2)."""
        return 4.0 * self.M / self.strands ** 2

    @property
    def travel_factor(self) -> float:
        """Carriage travel per unit belt-side travel of the 2:1 formulation, 2 / n."""
        return 2.0 / self.strands

    @property
    def max_acceleration(self) -> float:
        """Largest carriage acceleration (lengthening the loop) with positive belt tension
        at the take-up: n T_t / M (= g for a directly hung vertical counterweight)."""
        return self.force / self.M


@dataclass
class Conveyor:
    """Belt conveyor with uniform strands (SI units).

    mu_r, mu_c: inertial line densities (belt + reduced idler mass + coupled material);
    m_r, m_c: weight line densities (belt + material, default: equal to mu_r, mu_c);
    r_r, r_c: running motion resistances per unit length (N/m, opposing travel);
    t_v: Kelvin-Voigt retardation time eta/E (s).
    """

    L: float
    EA: float
    mu_r: float
    mu_c: float
    drive_position: float          # sigma_d (m)
    takeup_position: float         # sigma_t (m), on the return strand
    takeup: GravityTakeUp
    m_r: float | None = None
    m_c: float | None = None
    r_r: float = 0.0
    r_c: float = 0.0
    t_v: float = 0.0
    carry_profile: tuple = ((0.0, 0.0),)   # (distance from tail along carry, elevation), m
    g: float = G_STD
    _prof: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.m_r is None:
            self.m_r = self.mu_r
        if self.m_c is None:
            self.m_c = self.mu_c
        p = np.atleast_2d(np.asarray(self.carry_profile, dtype=float))
        if p.shape[0] == 1:
            p = np.array([[0.0, p[0, 1]], [self.L, p[0, 1]]])
        if abs(p[0, 0]) > 1e-9 or abs(p[-1, 0] - self.L) > 1e-6 or np.any(np.diff(p[:, 0]) <= 0):
            raise ValueError("carry_profile must start at 0 (tail), end at L (head) and increase")
        self._prof = p

    # ---------------------------------------------------------------- dimensionless groups
    @property
    def c_r(self) -> float:
        return float(np.sqrt(self.EA / self.mu_r))

    @property
    def c_c(self) -> float:
        return float(np.sqrt(self.EA / self.mu_c))

    @property
    def gamma(self) -> float:
        return self.c_r / self.c_c

    @property
    def beta(self) -> float:
        """Belt-side take-up mass ratio, 4 M / (n^2 mu_r L) for n strands."""
        return self.takeup.M_belt / (self.mu_r * self.L)

    @property
    def zeta_hat(self) -> float:
        return self.c_r * self.t_v / (2.0 * self.L)

    @property
    def xi(self) -> float:
        return ((self.takeup_position - self.drive_position) / self.L) % 2.0

    def loop(self, a_m: float = 1.0) -> Loop:
        """Dimensionless loop; resistances scaled by mu_r a_m."""
        s = self.mu_r * a_m
        return Loop.from_positions(self.drive_position / self.L, self.takeup_position / self.L,
                                   self.gamma, self.r_r / s, self.r_c / s)

    # ---------------------------------------------------------------- geometry and statics
    def elevation(self, sigma) -> np.ndarray:
        """Elevation h(sigma) along the loop (sigma in m, geometric coordinate)."""
        sig = np.asarray(sigma, dtype=float) % (2 * self.L)
        d = np.where(sig >= self.L, sig - self.L, self.L - sig)    # distance from tail
        return np.interp(d, self._prof[:, 0], self._prof[:, 1])

    def _to_sigma(self, s):
        return (np.asarray(s, dtype=float) + self.drive_position) % (2 * self.L)

    def static_tension(self, s) -> np.ndarray:
        """Static tension at rest, T0(s), s in m along the loop coordinate (drive exit = 0).
        dT0/ds = m g dh/ds, T0(s_t) = T_t; exact for the piecewise-linear profile."""
        s = np.asarray(s, dtype=float)
        L2 = 2 * self.L
        to_s = lambda sig: (np.asarray(sig) - self.drive_position) % L2
        brk = [0.0, L2, to_s(0.0), to_s(self.L), self.xi * self.L]
        d = self._prof[:, 0]
        brk += list(to_s(self.L + d)) + list(to_s(self.L - d))
        nodes = np.unique(np.clip(np.r_[brk, np.linspace(0, L2, 2001)], 0, L2))
        h = self.elevation(self._to_sigma(nodes))
        mid = self._to_sigma(0.5 * (nodes[1:] + nodes[:-1]))
        m = np.where(mid >= self.L, self.m_c, self.m_r)
        cum = np.r_[0.0, np.cumsum(m * self.g * np.diff(h))]
        cum_t = np.interp(self.xi * self.L, nodes, cum)
        return self.takeup.T_t + np.interp(s, nodes, cum) - cum_t

    def running_tension(self, s) -> np.ndarray:
        """Static tension plus the quasi-static resistance tension (steady running state
        with the take-up holding T_t), N."""
        lp = self.loop(1.0)
        return self.static_tension(s) + self.mu_r * self.L * lp.quasi_static("r", np.asarray(s) / self.L)

    # ---------------------------------------------------------------- start-up
    def start(self, V: float, t_a: float, kind: str = "sine", onset: str = "velocity",
              n_modes: int = 60, t_end: float | None = None, n_t: int = 2001) -> "DimensionalStartup":
        """Start-up from rest to belt speed V (m/s) in t_a (s)."""
        a_m = PEAK_FACTOR[kind] * V / t_a
        lp = self.loop(a_m)
        basis = modal_basis(lp, self.beta, n_modes)
        tau_a = self.c_r * t_a / self.L
        if t_end is None:
            t_end = t_a + 3 * 2 * np.pi * self.L / (self.c_r * basis.Om[0])
        tau = np.linspace(0.0, self.c_r * t_end / self.L, n_t)
        resp = startup_response(basis, StartProfile(kind, tau_a, onset), self.zeta_hat, tau)
        return DimensionalStartup(self, resp, a_m)

    def start_profile(self, profile, a_m: float, t_end: float, n_modes: int = 60,
                      n_t: int = 2001) -> "DimensionalStartup":
        """Start-up with any dimensionless profile (e.g. PiecewiseProfile) whose
        accelerations are scaled by a_m (m/s^2); t_end in s."""
        basis = modal_basis(self.loop(a_m), self.beta, n_modes)
        tau = np.linspace(0.0, self.c_r * t_end / self.L, n_t)
        resp = startup_response(basis, profile, self.zeta_hat, tau)
        return DimensionalStartup(self, resp, a_m)

    def profile_kinematics(self, profile, a_m: float):
        """SI drive kinematics t -> (a, v, d) of a dimensionless profile (for the lumped model)."""
        tsc, vsc, dsc = self.L / self.c_r, a_m * self.L / self.c_r, a_m * self.L ** 2 / self.c_r ** 2

        def kin(t):
            tau = np.asarray(t, dtype=float) / tsc
            v = profile.velocity(tau) * profile.a_integral
            return (float(a_m * profile.a(tau)), float(vsc * v), float(dsc * profile.displacement(tau)))
        return kin


@dataclass
class DimensionalStartup:
    """Start-up response in SI units, with the a-posteriori validity checks."""

    conveyor: Conveyor
    response: StartupResponse
    a_m: float

    @property
    def t(self) -> np.ndarray:
        cv = self.conveyor
        return self.response.tau * cv.L / cv.c_r

    @property
    def T_scale(self) -> float:
        return self.conveyor.mu_r * self.conveyor.L * self.a_m

    @property
    def y_scale(self) -> float:
        cv = self.conveyor
        return self.a_m * cv.L ** 2 / cv.c_r ** 2

    def dynamic_tension(self, s) -> np.ndarray:
        """T_d(s, t), N, shape (n_s, n_t); s in m along the loop coordinate."""
        return self.T_scale * self.response.tension(np.atleast_1d(s) / self.conveyor.L)

    def total_tension(self, s) -> np.ndarray:
        """T0(s) + T_d(s, t), N."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        return self.conveyor.static_tension(s)[:, None] + self.dynamic_tension(s)

    def takeup_displacement(self) -> np.ndarray:
        """Carriage travel from rest, m, positive when the loop lengthens."""
        f = self.conveyor.takeup.travel_factor
        return f * self.y_scale * self.response.takeup_displacement()

    def takeup_acceleration(self) -> np.ndarray:
        """Carriage acceleration, m/s^2."""
        return self.conveyor.takeup.travel_factor * self.a_m * self.response.takeup_acceleration()

    def loop_storage(self) -> np.ndarray:
        """Belt length taken into the take-up loop since rest, m (= n * carriage travel)."""
        return self.conveyor.takeup.strands * self.takeup_displacement()

    def checks(self, n_s: int = 801) -> dict:
        """Validity of the linear model: positive total tension along the loop (including
        both faces of the drive) and carriage acceleration below n T_t / M."""
        cv = self.conveyor
        s = np.unique(np.r_[np.linspace(0, 2 * cv.L, n_s), cv.xi * cv.L])
        T = self.total_tension(s)
        i, j = np.unravel_index(np.argmin(T), T.shape)
        ydd = self.takeup_acceleration()
        k = int(np.argmax(ydd))
        lim = cv.takeup.max_acceleration
        return {
            "min_total_tension_N": float(T[i, j]),
            "min_tension_s_m": float(s[i]),
            "min_tension_t_s": float(self.t[j]),
            "tension_positive": bool(T[i, j] > 0),
            "max_takeup_accel_ms2": float(ydd[k]),
            "takeup_accel_limit_ms2": float(lim),
            "max_takeup_accel_t_s": float(self.t[k]),
            "takeup_follows": bool(ydd[k] < lim),
            "takeup_travel_m": (float(self.takeup_displacement().min()),
                                float(self.takeup_displacement().max())),
        }
