"""Natural frequencies and modes of the loop with a gravity take-up.

Root isolation (no grid scanning). Write the characteristic equation as
    F(Om) = A12/A22 + B12/B11 = 4 / (beta Om^2),
where A12/A22 and B12/B11 are the end receptances of the upstream and downstream
parts (fixed at the drive, free at the take-up). Both increase strictly with Om
between their poles (Green's identity: d/d(Om^2) [W/W'] = int mu W^2 / W'^2 > 0),
while the right-hand side decreases. Hence the roots of D strictly interlace with the
merged, sorted zeros p_1 <= p_2 <= ... of A22 and B11 (the fixed-free modes of the two
parts): exactly one root in (0, p_1) and one in each (p_{k-1}, p_k); a coincident pair
p_{k-1} = p_k is itself a root. The fixed-free zeros are bracketed exactly with the
Pruefer angle. The method finds every root, however close two roots are (small beta).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, newton

from .loop import Loop
from .transfer import AB, char_fun, char_fun_torque, prufer_angle

_XTOL = 1e-14


# ---------------------------------------------------------------------------- roots
def fixed_free_roots(segments, n: int) -> np.ndarray:
    """First n frequencies of a chain fixed at its start and free at its end."""
    segments = tuple(segments)
    phase = sum(s.g * s.length for s in segments)          # d(phi)/d(Om) without interfaces
    m = sum(1 for a, b in zip(segments[:-1], segments[1:]) if a.g != b.g)
    out = np.empty(n)
    for j in range(1, n + 1):
        target = (j - 0.5) * np.pi
        f = lambda o: prufer_angle(segments, o) - target
        lo = max((target - 0.5 * m * np.pi - 0.1) / phase, 1e-300)
        hi = (target + 0.5 * m * np.pi + 0.1) / phase
        if f(lo) > 0:   # cannot happen with the bounds above; kept as a guard
            lo = 1e-300
        out[j - 1] = brentq(f, lo, hi, xtol=_XTOL, rtol=4 * np.finfo(float).eps)
    return out


def poles(loop: Loop, n: int) -> np.ndarray:
    """Merged zeros of A22 and B11 (fixed-free modes of the two parts), first n."""
    pa = fixed_free_roots(loop.upstream, n)
    pb = fixed_free_roots(loop.downstream[::-1], n)
    return np.sort(np.concatenate((pa, pb)))[:n]


def natural_frequencies(loop: Loop, beta: float, n: int) -> np.ndarray:
    """First n dimensionless natural frequencies Om = omega L / c_r (prescribed drive
    velocity). beta = 0 gives the constant-force limit (the poles themselves)."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    p = poles(loop, n)
    if beta == 0:
        return p
    out = np.empty(n)
    lo = 0.0
    for k in range(n):
        hi = p[k]
        if hi - lo <= 1e-12 * hi:          # coincident fixed-free modes: exact root
            out[k] = hi
        else:
            # Inside (lo, hi), D = A22 B11 beta Om^2 H with H strictly increasing from -inf
            # to +inf and A22 B11 of constant sign: sign-corrected D is negative near lo and
            # positive near hi. The end values are replaced by these limit signs, which
            # stays correct when an end is a coincident pole (where D = 0 exactly).
            mid = 0.5 * (lo + hi)
            A, B = AB(loop, mid)
            sk = np.sign(A[1, 1] * B[0, 0])
            a, b = lo, hi

            def f(o, a=a, b=b, sk=sk):
                if o <= a:
                    return -1.0
                if o >= b:
                    return 1.0
                return sk * char_fun(loop, o, beta)

            out[k] = brentq(f, a, b, xtol=_XTOL, rtol=4 * np.finfo(float).eps)
        lo = hi
    return out


def natural_frequencies_torque(loop: Loop, beta: float, md: float, Om_max: float,
                               n_grid: int = 20000) -> np.ndarray:
    """Elastic natural frequencies (rigid mode excluded) below Om_max for a
    torque-controlled drive. Limit-of-validity tool: grid scan with sign changes,
    so very close roots may be missed; check the grid density for small beta."""
    f = lambda o: char_fun_torque(loop, o, beta, md)
    g = np.linspace(1e-3 * min(1.0, Om_max), Om_max, n_grid)
    v = np.array([f(o) for o in g])
    idx = np.nonzero(v[:-1] * v[1:] < 0)[0]
    return np.array([brentq(f, g[i], g[i + 1], xtol=_XTOL) for i in idx])


def damped_drive_root(loop: Loop, beta: float, md: float, cd: float, guess: complex) -> complex:
    """Complex natural frequency (Om = Re + i Im, Im > 0 for decay) for a drive with mass
    md and slip dashpot cd (see transfer.char_fun_torque), by Newton from a guess, e.g.
    the prescribed-velocity root (the limit cd -> infinity). Limit-of-validity tool."""
    f = lambda o: char_fun_torque(loop, o, beta, md, cd)
    return complex(newton(f, complex(guess), tol=1e-13, maxiter=200))


# ---------------------------------------------------------------------------- modes
@dataclass
class ModalBasis:
    """Mass-normalised modes (<phi_k, phi_k> = 1 with the weighted inner product).

    Attributes (arrays over modes k):
      Om     natural frequencies;  Y  take-up amplitudes;
      Gamma  inertial participation factors int mu_hat W_k dx;
      R      resistance participation factors int r_hat W_k dx;
      W0, dW0 (n_modes, n_segments): state at the start of each segment
      (downstream segments start at xi+, after the take-up jump).
    """

    loop: Loop
    beta: float
    Om: np.ndarray
    Y: np.ndarray
    Gamma: np.ndarray
    R: np.ndarray
    W0: np.ndarray
    dW0: np.ndarray
    stiffness_check: np.ndarray   # int W_k'^2 dx / Om_k^2 (should be 1)

    @property
    def n(self) -> int:
        return len(self.Om)

    @property
    def effective_mass_fraction(self) -> np.ndarray:
        """Gamma_k^2 m_k / belt mass: share of the belt inertia excited in mode k."""
        return self.Gamma ** 2 / self.loop.belt_mass

    def _eval(self, x, side: str):
        x = np.asarray(x, dtype=float)
        lp = self.loop
        i = lp.segment_index(x)
        if side == "left":       # take the upstream side at x = xi exactly
            nu = len(lp.upstream)
            at_xi = np.isclose(x, lp.xi, rtol=0, atol=1e-13)
            i = np.where(at_xi, nu - 1, i)
        g = np.array([s.g for s in lp.segments])[i]
        u = x - lp.x0[i]
        k = self.Om[:, None] * g[None, :]
        c, s = np.cos(k * u), np.sin(k * u)
        W0, dW0 = self.W0[:, i], self.dW0[:, i]
        W = W0 * c + dW0 * s / k
        dW = -k * W0 * s + dW0 * c
        return W, dW

    def W(self, x, side: str = "right") -> np.ndarray:
        """Mode shapes at x, shape (n_modes, n_x). At x = xi, side selects the face."""
        return self._eval(x, side)[0]

    def dW(self, x) -> np.ndarray:
        """Modal dynamic tensions W_k'(x), shape (n_modes, n_x) (continuous at xi)."""
        return self._eval(x, "right")[1]


def _segment_integrals(W0, dW0, l, k):
    """Exact int W, int W^2, int W'^2 over a segment, plus the end state."""
    c, s = np.cos(k * l), np.sin(k * l)
    W1 = W0 * c + dW0 * s / k
    dW1 = -k * W0 * s + dW0 * c
    C = dW0 ** 2 + (k * W0) ** 2
    jump = W1 * dW1 - W0 * dW0
    I1 = (dW0 - dW1) / k ** 2
    I2 = 0.5 * l * C / k ** 2 - 0.5 * jump / k ** 2
    I3 = 0.5 * (C * l + jump)
    return I1, I2, I3, W1, dW1


def modal_basis(loop: Loop, beta: float, n: int) -> ModalBasis:
    """Modes for prescribed drive velocity (beta > 0), with exact integrals."""
    if not beta > 0:
        raise ValueError("modal_basis needs beta > 0 (use a small beta for the limit)")
    Om = natural_frequencies(loop, beta, n)
    nseg, nu = len(loop.segments), len(loop.upstream)
    W0 = np.zeros((n, nseg)); dW0 = np.zeros((n, nseg))
    Y = np.zeros(n); G = np.zeros(n); R = np.zeros(n); m = np.zeros(n); K = np.zeros(n)
    for j, o in enumerate(Om):
        w, dw = 0.0, 1.0
        for i, seg in enumerate(loop.segments):
            if i == nu:                                  # take-up pulley
                Y[j] = 2.0 * dw / (beta * o ** 2)
                w = w - 2.0 * Y[j]
            W0[j, i], dW0[j, i] = w, dw
            I1, I2, I3, w, dw = _segment_integrals(w, dw, seg.length, seg.g * o)
            G[j] += seg.mu * I1; R[j] += seg.r * I1; m[j] += seg.mu * I2; K[j] += I3
        m[j] += beta * Y[j] ** 2
    sc = 1.0 / np.sqrt(m)
    return ModalBasis(loop=loop, beta=beta, Om=Om, Y=Y * sc, Gamma=G * sc, R=R * sc,
                      W0=W0 * sc[:, None], dW0=dW0 * sc[:, None],
                      stiffness_check=K / (m * Om ** 2))


def rayleigh_takeup_bound(loop: Loop, beta: float) -> float:
    """Upper bound Om_0 <= sqrt(2/(beta + beta_b)) for the slow take-up mode, with the
    quasi-static shape W = x upstream, W = x - 2 downstream (accurate for beta >> beta_b)."""
    bb = 0.0
    for a, b, seg in zip(loop.x0[:-1], loop.x0[1:], loop.segments):
        shift = 0.0 if b <= loop.xi + 1e-13 else loop.length
        bb += seg.mu * ((b - shift) ** 3 - (a - shift) ** 3) / 3.0
    return float(np.sqrt(2.0 / (beta + bb)))


# ------------------------------------------------------------- take-up mass regime
def strand_modes(segments, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-free modes of a chain fixed at its start and free at its end.

    Returns the first n frequencies Om_j and the effective modal masses at the free end,
    m_tilde_j = int mu_hat phi_j^2 dx / phi_j(end)^2 (the modal mass of a mode scaled to
    unit amplitude at the free end). A uniform strand of length l and density mu_hat has
    m_tilde = mu_hat l / 2 for every mode. The end receptance of the chain is
        W(end)/W'(end) = sum_j 1 / (m_tilde_j (Om_j^2 - Om^2)).
    """
    segments = tuple(segments)
    Om = fixed_free_roots(segments, n)
    mt = np.empty(n)
    for j, o in enumerate(Om):
        w, dw, m = 0.0, 1.0, 0.0
        for seg in segments:
            _, I2, _, w, dw = _segment_integrals(w, dw, seg.length, seg.g * o)
            m += seg.mu * I2
        mt[j] = m / w ** 2
    return Om, mt


def takeup_mass_approx(loop: Loop, beta: float, n: int) -> np.ndarray:
    """Small-beta approximation of the first n natural frequencies (prescribed drive).

    The characteristic equation is H_A + H_B = 4/(beta Om^2), with H_A, H_B the end
    receptances of the two strands (fixed at the drive, free at the take-up). As beta -> 0
    each root tends to a fixed-free pole Om_p from below. Keeping the pole of the root and
    the nearest pole of the other strand (u = Om^2, a = 1/m_tilde):
        a/(P_a - u) + b/(P_b - u) = 4/(beta u),
    a quadratic in u. With one pole only this is Rayleigh's quotient with the frozen
    fixed-free shape and a mass beta/4 at the free end: Om = Om_p / sqrt(1 + beta/(4 m_tilde)).
    The two-pole form also covers coincident poles (one root stays at Om_p, the other
    shifts by both strands). Accurate to a few 1e-3 for beta <= 0.1 (modes 1-3, 1 <= gamma
    <= 3); for beta ~ 1 the fundamental stays within ~3 % but higher modes do not, because
    the take-up's own mode, Om ~ sqrt(2/beta), enters the belt spectrum.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    pa, ma = strand_modes(loop.upstream, n)
    pb, mb = strand_modes(loop.downstream[::-1], n)
    P = np.concatenate((pa, pb))
    a_all = 1.0 / np.concatenate((ma, mb))
    side = np.r_[np.zeros(n), np.ones(n)]
    order = np.argsort(P, kind="stable")[:n]
    out = np.empty(n)
    for k, i in enumerate(order):
        p, a = P[i] ** 2, a_all[i]
        other = np.flatnonzero(side != side[i])
        j = other[np.argmin(np.abs(P[other] - P[i]))]
        q, b = P[j] ** 2, a_all[j]
        u = np.sort(np.roots([4.0 + beta * (a + b),
                              -(4.0 * (p + q) + beta * (a * q + b * p)),
                              4.0 * p * q]).real)
        if np.isclose(p, q, rtol=1e-12, atol=0.0):
            first = k == 0 or not np.isclose(P[order[k - 1]], P[i], rtol=1e-12, atol=0.0)
            out[k] = np.sqrt(u[0] if first else u[1])
        else:
            out[k] = np.sqrt(u[1] if p > q else u[0])
    return out
