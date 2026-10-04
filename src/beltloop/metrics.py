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


# --------------------------------------------------------------- take-up kinematics (4.6)
@dataclass
class TakeupKinematics:
    """Extremes of the belt-side take-up velocity and acceleration during a start.

    v_*: y' / V_inf (V_inf: final belt speed); a_*: y'' / a_m. Belt-side (2:1) quantities:
    the carriage values are (2/n) times these. Positive = loop lengthening (counterweight
    descending), the direction that unloads the belt at the take-up."""

    v_max: float
    v_min: float
    a_max: float
    a_min: float


def takeup_kinematics(loop: Loop, profile, zeta_hat: float = 0.0, beta: float = _BETA0,
                      n_modes: int = 120, points: int = 400,
                      periods_after: float = 3.0) -> TakeupKinematics:
    """Take-up velocity (exact, modal) and acceleration (differentiated on a fine grid)
    during a start and `periods_after` loop periods after it, inertial load only unless
    the profile carries a resistance onset. With beta -> 0 the take-up is the free end of
    both strands; for gamma = 1 the exact bounds are |y'| <= V_inf and |y''| <= 2 a_m
    (d'Alembert: each free end moves at an alternating sum of delayed drive velocities).
    Numerical envelope for 1 <= gamma <= 3, any xi, sine profile (maps/validity.py):
    |y''| <= 2.5 a_m for every start duration (reached with gamma = 3 and fast starts);
    |y'| <= 1.75 V_inf for tau_a -> 0 (impedance step at the tail), <= 1.14 V_inf for
    tau_a >= 0.2 T_1, <= 0.30 V_inf for tau_a >= T_1 and <= 0.09 V_inf for tau_a >= 1.4 T_1.
    Since the carriage acceleration is (2/n) y'', the take-up keeps the belt taut
    (y''_carriage < n T_t / M, = g for a direct counterweight) unless a_m approaches
    g / 2.5: the condition is inactive for any realistic start."""
    b = modal_basis(loop, beta, n_modes)
    T1 = 2 * np.pi / b.Om[0]
    ta = profile.tau_a
    tau = _time_grid(ta + periods_after * T1, min(T1, ta) / points)
    res = startup_response(b, profile, zeta_hat, tau)
    v = res.takeup_velocity()
    a = np.gradient(v, tau)
    V_inf = profile.a_integral
    return TakeupKinematics(v.max() / V_inf, v.min() / V_inf, a.max(), a.min())


# --------------------------------------------------------------- validity zones (4.6)
def strand_rebound(ratios, kind: str = "sine", zeta1: float = 0.0, n_modes: int = 60,
                   points: int = 300, periods_after: float = 3.0):
    """Rebound curves of a uniform fixed-free strand (complement of strand_curves).

    Returns (D_minus, R_post): D_minus = -min_t T_end / (m a_m), the largest tension drop
    below zero at the fixed end under the inertial load (the free oscillation left by the
    start swings the drive-entry tension below its running value); R_post = min over
    t >= t_a of T_end / R under the resistance load with onset phi = V/V_inf (fraction of
    the running resistance tension left at the worst post-start instant).
    Slow starts: D_minus ~ 2 (D - 1) ~ 0.8 / r for the sine profile (jerk jumps at both
    ends); D_minus = 0 where the residual of the first mode vanishes (t_a = (j + 1/2) T_s).
    """
    lp = Loop((Segment(_EPS_A),), (Segment(1.0),))
    lr = Loop((Segment(_EPS_A, r=1.0),), (Segment(1.0, r=1.0),))
    b, br = modal_basis(lp, _BETA0, n_modes), modal_basis(lr, _BETA0, n_modes)
    Ts = 2 * np.pi / b.Om[0]
    zh = zeta1 / b.Om[0]
    xe = np.array([lp.length])
    Dm, Rp = [], []
    for r in np.atleast_1d(ratios):
        ta = float(r) * Ts
        tau = _time_grid(ta + periods_after * Ts, min(Ts, ta) / points)
        Ti = startup_response(b, StartProfile(kind, ta, "none"), zh, tau).tension(xe)[0]
        Tr = (startup_response(br, StartProfile(kind, ta, "velocity"), zh, tau).tension(xe)[0]
              - startup_response(br, StartProfile(kind, ta, "none"), zh, tau).tension(xe)[0])
        Dm.append(max(0.0, -Ti.min()))
        Rp.append(Tr[tau >= ta].min())
    return np.array(Dm), np.array(Rp)


def strand_extremes(ratios, rho: float, kind: str = "sine", zeta1: float = 0.0,
                    n_modes: int = 60, points: int = 300, periods_after: float = 3.0,
                    n_x: int = 81):
    """Extremes over the whole strand and the whole start (plus `periods_after` periods) of
    the dynamic tension of a uniform fixed-free strand of unit mass under the combined load:
    unit peak acceleration plus a resistance rho per unit mass with onset phi = V/V_inf.

    Returns (C_max, C_min), in units of (strand mass) * a_m. At the free end the tension is
    zero, so C_min <= 0 <= C_max. C_max is reached at the fixed end during the start; C_min
    is the rebound after it, at the fixed end without resistances and inside the strand
    with them (the running resistance tension grows from the free end)."""
    lp = Loop((Segment(_EPS_A, r=rho),), (Segment(1.0, r=rho),))
    b = modal_basis(lp, _BETA0, n_modes)
    Ts = 2 * np.pi / b.Om[0]
    zh = zeta1 / b.Om[0]
    x = np.linspace(_EPS_A, lp.length, n_x)
    cmax, cmin = [], []
    for r in np.atleast_1d(ratios):
        ta = float(r) * Ts
        tau = _time_grid(ta + periods_after * Ts, min(Ts, ta) / points)
        T = startup_response(b, StartProfile(kind, ta, "velocity"), zh, tau).tension(x)
        cmax.append(max(T.max(), 0.0))
        cmin.append(min(T.min(), 0.0))
    return np.array(cmax), np.array(cmin)


@dataclass
class TensionRequirement:
    """Minimum running slack-side tension T_2 (at the drive exit) for positive tension
    during a start and its aftermath, in units of mu_r L a_m (horizontal conveyor, drive at
    the head, resistances proportional to the inertial line density, r = rho mu a_m, so the
    take-up holds T_t = T_2 + rho m_A). From the strand extremes (strand_extremes):

    exit (strand A, during the start): T_2 >= m_A [C_max(r_A, rho) - rho]
    B    (rebound after the start):    T_2 >= -m_B C_min(r, rho) - rho m_A

    r_A = tau_a / T_A1, r = tau_a / T_1 (= T_B1 with beta -> 0); m_A = xi and m_B are the
    strand masses. Exact for strand A (uniform) with beta -> 0; strand B follows the uniform
    curve only approximately when gamma != 1. Add the sag threshold (or any other minimum
    tension) to the right-hand sides for a practical criterion.

    grip (prescribed drive velocity needs no slip; Euler-Eytelwein, E = exp(mu theta)):
    T_entry <= E T_exit with both peaks taken together,
    T_2 >= [m_B C_max(r, rho) + E m_A C_max(r_A, rho)] / (E - 1) - rho m_A;
    in steady running it reduces to the usual T_2 >= rho m_belt / (E - 1).

    Accuracy against the full loop (drive at the head, 1 <= gamma <= 3, rho = (0.3 to 3) r,
    sine profile, undamped): for tau_a >= T_1 / 2 the slack estimate is at most 0.04 m_belt
    below the exact requirement and the grip estimate at most 0.006 m_belt below (it is
    usually above, by up to 0.34 m_belt with E = 3, because the two peaks do not coincide);
    for tau_a >= 2 T_1 both are within 0.002 m_belt on the unsafe side. Faster starts with
    gamma != 1 are not covered: the impedance step at the tail leaves negative tension in
    strand B that the uniform curves miss (0.19 m_belt with gamma = 3, tau_a = 0.3 T_1); use
    the full model (DimensionalStartup.checks) there. Intermediate drives (strand B or A
    non-uniform): about 5 % of the requirement on the unsafe side.

    Coupled static state: the take-up must hold T_t = T_2 + rho m_A (T_t_min). With a
    gravity take-up, beta = (4/n) lam T_t / (g mu_r L), lam = g M / (n T_t) = g /
    (largest carriage acceleration) (1 for a directly hung counterweight), so the
    requirement is a lower bound on beta: beta >= (4/n) lam (a_m/g) T_t_min (beta_min).
    With rho = f g / a_m (resistance coefficient f), the running grip part alone gives
    beta >= (4/n) lam f [m_belt / (E - 1) + m_A]."""

    exit: float
    rebound: float
    grip: float = float("nan")      # only with euler = exp(mu theta)
    resistance_A: float = 0.0       # rho m_A: running resistance between drive exit and take-up

    @property
    def T2_min(self) -> float:
        return float(np.nanmax([self.exit, self.rebound, self.grip, 0.0]))

    @property
    def T_t_min(self) -> float:
        """Minimum static take-up tension, T_2_min + rho m_A (units mu_r L a_m)."""
        return self.T2_min + self.resistance_A

    def beta_min(self, a_over_g: float, strands: int = 2, lam: float = 1.0) -> float:
        """Smallest take-up mass ratio of a gravity take-up that meets the requirement:
        (4/n) lam (a_m/g) T_t_min. lam = g M / force (GravityTakeUp: g / max_acceleration)."""
        return 4.0 / strands * lam * a_over_g * self.T_t_min

    @property
    def governing(self) -> str:
        v = {"exit": self.exit, "rebound": self.rebound, "grip": self.grip}
        return max((k for k in v if not np.isnan(v[k])), key=lambda k: v[k])


def tension_requirement(loop: Loop, tau_a: float, rho: float, kind: str = "sine",
                        zeta1: float = 0.0, euler: float | None = None) -> TensionRequirement:
    """Strand-curve estimate of the minimum running slack-side tension; see
    TensionRequirement. loop: the standard loop (its own resistances are ignored; rho sets
    them)."""
    TA = 2 * np.pi / strand_participation(loop.upstream, 1)[0][0]
    TB = 2 * np.pi / strand_participation(loop.downstream[::-1], 1)[0][0]
    mA = sum(s.mu * s.length for s in loop.upstream)
    mB = loop.belt_mass - mA
    cA, _ = strand_extremes([tau_a / TA], rho, kind, zeta1)
    cBmax, cB = strand_extremes([tau_a / TB], rho, kind, zeta1)
    grip = float("nan")
    if euler is not None:
        if not euler > 1:
            raise ValueError("euler = exp(mu theta) must exceed 1")
        grip = (mB * cBmax[0] + euler * mA * cA[0]) / (euler - 1.0) - rho * mA
    return TensionRequirement(mA * (cA[0] - rho), -mB * cB[0] - rho * mA, grip, rho * mA)
