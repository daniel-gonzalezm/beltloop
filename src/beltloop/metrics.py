"""Start-up metrics and the universal curves of a fixed-free strand (phase 4.5).

With beta -> 0 the take-up is a free end and the loop splits into two strands, fixed at
the drive and free at the take-up (eigen.strand_participation). During a start each strand
responds on its own, so

  * the peak dynamic tension at the drive entry belongs to the downstream strand B
    (take-up -> tail -> carry -> drive entry);
  * the minimum at the drive exit belongs to the upstream strand A (drive exit -> take-up),
    a uniform return-strand segment of length xi;
  * the take-up travel is half the difference of the free-end displacements of B and A.

For a uniform fixed-free strand the response to a given acceleration profile depends only on
tau_a / T_s (T_s: the strand's fundamental period) and on the damping ratio of its first mode.
:func:`strand_curves` gives these universal curves: the dynamic amplification D of the tension
at the fixed end and of the displacement of the free end, relative to their quasi-static peaks.
Two limits bound them: for slow starts D -> 1 (quasi-static, T = m a); for fast starts the
fixed end carries the wave law T = Z V (Z = mu c, the strand impedance), i.e. D = (8/pi) r for
the sine profile (r = tau_a / T_s). Strand A is uniform, so the exit follows the curve exactly;
strand B is uniform only for gamma = 1, but the entry follows the curve within about 2 % for
tau_a / T_1 >= 0.8 (the fast-start limit depends on the carry impedance alone, which is why the
collapse fails for very fast starts).

Units as in response.StartupResponse: tensions in mu_r L a_m, displacements in a_m L^2/c_r^2,
time in L/c_r; a_m is the peak acceleration of the profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eigen import modal_basis, strand_participation
from .forcing import PEAK_FACTOR, StartProfile
from .loop import Loop, Segment
from .response import startup_response

_EPS_A = 1e-3          # length of the dummy upstream strand used to isolate a single strand
_BETA0 = 1e-8          # take-up mass ratio used for the beta -> 0 limit


def _time_grid(tau_end: float, dt: float) -> np.ndarray:
    n = int(np.ceil(tau_end / dt))
    return np.linspace(0.0, n * dt, n + 1)


def strand_curves(ratios, kind: str = "sine", zeta1: float = 0.0, load: str = "inertia",
                  n_modes: int = 60, points: int = 300, periods_after: float = 3.0):
    """Universal amplification curves of a uniform fixed-free strand.

    ratios: tau_a / T_s values (T_s = 4 l / c, the strand's fundamental period).
    kind: 'sine', 'triangular' or 'parabolic' (StartProfile); accelerations scaled by their peak.
    zeta1: damping ratio of the first strand mode (Kelvin-Voigt: zeta_k = (2k - 1) zeta1).
    load: 'inertia' (drive acceleration) or 'resistance' (uniform resistance with onset
      phi = V/V_inf, inertia excluded by linearity).
    Returns (D_end, D_free): peak fixed-end tension and peak free-end displacement over the
    start and `periods_after` periods after it, divided by their quasi-static peaks.
    """
    if load not in ("inertia", "resistance"):
        raise ValueError("load must be 'inertia' or 'resistance'")
    seg_r = 1.0 if load == "resistance" else 0.0
    lp = Loop((Segment(_EPS_A, r=seg_r),), (Segment(1.0, r=seg_r),))
    b = modal_basis(lp, _BETA0, n_modes)
    Ts = 2 * np.pi / b.Om[0]                        # = 4 (unit length, unit wave speed)
    zeta_hat = zeta1 / b.Om[0]
    x_end = np.array([lp.length])
    attr = "mu" if load == "inertia" else "r"
    qs_end = abs(lp.quasi_static(attr, x_end)[0])
    I = lp.quasi_static_integral(attr)
    De, Df = [], []
    for r in np.atleast_1d(ratios):
        ta = float(r) * Ts
        tau = _time_grid(ta + periods_after * Ts, min(Ts, ta) / points)
        if load == "inertia":
            res = startup_response(b, StartProfile(kind, ta, "none"), zeta_hat, tau)
            T = res.tension(x_end)[0]
            y = res.takeup_displacement()
        else:
            full = startup_response(b, StartProfile(kind, ta, "velocity"), zeta_hat, tau)
            inert = startup_response(b, StartProfile(kind, ta, "none"), zeta_hat, tau)
            T = full.tension(x_end)[0] - inert.tension(x_end)[0]
            y = full.takeup_displacement() - inert.takeup_displacement()
        De.append(T.max() / qs_end)
        Df.append(y.max() / (0.5 * I))
    return np.array(De), np.array(Df)


def fast_start_limit(ratio, kind: str = "sine") -> np.ndarray:
    """Wave-law limit of the fixed-end amplification of a uniform strand for fast starts,
    D = Z V_inf / (m a_m) = 4 ratio / PEAK_FACTOR (= (8/pi) ratio for the sine profile)."""
    return 4.0 * np.asarray(ratio, dtype=float) / PEAK_FACTOR[kind]


@dataclass
class StartupMetrics:
    """Start-up metrics of a loop (dimensionless). *_qs are the quasi-static references
    (peak drive-entry tension, minimum drive-exit tension, peak take-up travel)."""

    T1: float              # fundamental period of the loop
    T_A1: float            # fundamental period of the upstream strand (drive exit -> take-up)
    T_B1: float            # fundamental period of the downstream strand
    entry_peak: float
    exit_min: float
    travel_max: float
    entry_qs: float
    exit_qs: float
    travel_qs: float

    @property
    def D_entry(self) -> float:
        return self.entry_peak / self.entry_qs

    @property
    def D_exit(self) -> float:
        return self.exit_min / self.exit_qs

    @property
    def D_travel(self) -> float:
        return self.travel_max / self.travel_qs


def startup_metrics(loop: Loop, beta: float, profile: StartProfile, zeta_hat: float = 0.0,
                    n_modes: int = 40, points: int = 200, periods_after: float = 3.0) -> StartupMetrics:
    """Peak drive-entry tension, minimum drive-exit tension and peak take-up travel during a
    start (and `periods_after` loop periods after it), with their quasi-static references.
    Drive entry is x = 2 (end of the chain), drive exit x = 0."""
    b = modal_basis(loop, beta, n_modes)
    T1 = 2 * np.pi / b.Om[0]
    TA = 2 * np.pi / strand_participation(loop.upstream, 1)[0][0]
    TB = 2 * np.pi / strand_participation(loop.downstream[::-1], 1)[0][0]
    ta = profile.tau_a
    tau = _time_grid(ta + periods_after * T1, min(T1, ta) / points)
    res = startup_response(b, profile, zeta_hat, tau)
    xs = np.array([0.0, loop.length])
    T = res.tension(xs)
    qs = res.quasi_static_tension(xs)
    y = res.takeup_displacement()
    a, phi = profile.a(tau), profile.phi(tau)
    y_qs = 0.5 * (a * loop.quasi_static_integral("mu") + phi * loop.quasi_static_integral("r"))
    return StartupMetrics(T1=T1, T_A1=TA, T_B1=TB, entry_peak=T[1].max(), exit_min=T[0].min(),
                          travel_max=y.max(), entry_qs=qs[1].max(), exit_qs=qs[0].min(),
                          travel_qs=y_qs.max())


def strand_travel_estimate(loop: Loop, D_free_B: float) -> float:
    """Peak take-up travel (inertial load, unit peak acceleration) from the strand picture:
    y ~ (D_free_B S_B - S_A) / 2, with S_B = int_xi^2 Q dx (downstream free-end displacement)
    and S_A = xi^2 / 2 for the uniform upstream strand. Within about 5 % for gamma >= 2 and
    tau_a / T_1 >= 1; strand A is not negligible when gamma ~ 1 and the take-up is near the
    tail."""
    xs = np.linspace(loop.xi, loop.length, 4001)
    S_B = np.trapezoid(loop.quasi_static("mu", xs), xs)
    S_A = -np.trapezoid(loop.quasi_static("mu", np.linspace(0.0, loop.xi, 2001)),
                        np.linspace(0.0, loop.xi, 2001))
    return 0.5 * (D_free_B * S_B - S_A)


# --------------------------------------------------------------- crawl start (plateau)
def crawl_start(tau_r: float, tau_p: float, tau_a: float, v_p: float):
    """Start with an initial crawl (Harrison 1985b, ZISCO, Henderson): constant acceleration to
    the crawl speed v_p V_inf over tau_r, hold for tau_p, then a parabolic acceleration of peak 1
    (the a_m scale) over tau_a to V_inf. Returns a PiecewiseProfile with onset 'none'; combine it
    with :func:`crawl_onset` for resistances that build up as the belt starts to move."""
    from .forcing import PiecewiseProfile
    if not (0.0 <= v_p < 1.0 and tau_a > 0):
        raise ValueError("need 0 <= v_p < 1 and tau_a > 0")
    v_main = 2.0 * tau_a / 3.0
    v_inf = v_main / (1.0 - v_p)
    breaks, coeffs = [0.0], []
    if v_p > 0:
        if not tau_r > 0:
            raise ValueError("a crawl speed needs tau_r > 0")
        breaks.append(tau_r)
        coeffs.append((v_p * v_inf / tau_r,))
        if tau_p > 0:
            breaks.append(tau_r + tau_p)
            coeffs.append((0.0,))
    breaks.append(breaks[-1] + tau_a)
    coeffs.append((0.0, 4.0 / tau_a, -4.0 / tau_a ** 2))
    return PiecewiseProfile(tuple(breaks), tuple(coeffs), onset="none")


def crawl_onset(tau_r: float):
    """Resistance onset phi = min(1, tau / tau_r): resistances fully established once the belt
    reaches the crawl speed (Coulomb-like, speed independent). Used as the load history of the
    resistance part in :func:`startup_with_onset` (its own acceleration is discarded)."""
    from .forcing import PiecewiseProfile
    return PiecewiseProfile((0.0, float(tau_r)), ((1.0,),), onset="velocity")


def startup_with_onset(basis, profile, onset_profile, zeta_hat: float, tau, x):
    """Superpose the inertial response to `profile` (its own onset is ignored) and the
    resistance response with phi(tau) = onset_profile.phi(tau) (by linearity: response with the
    onset minus the same without it). Returns (T, y): dynamic tension at x, shape (n_x, n_tau),
    and take-up travel."""
    from dataclasses import replace
    x = np.atleast_1d(np.asarray(x, dtype=float))
    inert = startup_response(basis, replace(profile, onset="none"), zeta_hat, tau)
    full = startup_response(basis, onset_profile, zeta_hat, tau)
    bare = startup_response(basis, replace(onset_profile, onset="none"), zeta_hat, tau)
    T = inert.tension(x) + full.tension(x) - bare.tension(x)
    y = inert.takeup_displacement() + full.takeup_displacement() - bare.takeup_displacement()
    return T, y
