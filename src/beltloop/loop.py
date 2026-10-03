"""Loop geometry: the belt loop as an open chain of homogeneous segments.

Conventions (Section "Model"):
  * geometric coordinate sigma in [0, 2) (units of L), from the head pulley along belt
    travel; return strand [0, 1), carry strand [1, 2), tail pulley at sigma = 1;
  * loop coordinate x = (sigma - sigma_d) mod 2, from the drive exit (x = 0) to the
    drive entry (x = 2);
  * the take-up pulley is at x = xi and splits the chain into an upstream part
    (drive exit -> take-up, transfer matrix A) and a downstream part
    (take-up -> drive entry, transfer matrix B).

Each segment carries its wavenumber factor g (kappa = g * Omega), so that the
relative line density is mu_hat = g**2 (1 on the return strand, gamma**2 on the carry
strand), and a dimensionless motion resistance r_hat (force per unit length, scaled
by mu_r * a_m).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_TOL = 1e-12


@dataclass(frozen=True)
class Segment:
    """Homogeneous belt segment in the loop coordinate (all quantities dimensionless)."""

    length: float
    g: float = 1.0          # wavenumber factor: kappa = g * Omega (= c_r / c_j)
    r: float = 0.0          # resistance per unit length, r_hat = r / (mu_r a_m)
    strand: str = "return"  # 'return' or 'carry' (informative)

    @property
    def mu(self) -> float:
        """Relative line density mu_hat = g**2."""
        return self.g ** 2


@dataclass(frozen=True)
class Loop:
    """Belt loop split at the take-up into upstream and downstream segment chains.

    Use :meth:`from_positions` for the standard two-strand conveyor; the general
    constructor accepts arbitrary chains (used in tests and for future extensions).
    """

    upstream: tuple[Segment, ...]
    downstream: tuple[Segment, ...]
    sigma_d: float = 0.0               # drive position (geometric, units of L)
    gamma: float | None = None         # c_r / c_c, if built from positions
    _x0: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.upstream or not self.downstream:
            raise ValueError("both upstream and downstream chains must be non-empty")
        for s in self.segments:
            if s.length <= 0 or s.g <= 0:
                raise ValueError("segment lengths and wavenumber factors must be positive")
        x0 = np.concatenate(([0.0], np.cumsum([s.length for s in self.segments])))
        object.__setattr__(self, "_x0", x0)

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_positions(cls, sigma_d: float, sigma_t: float, gamma: float,
                       r_return: float = 0.0, r_carry: float = 0.0) -> "Loop":
        """Standard conveyor: drive anywhere at sigma_d in [0, 2), take-up on the return
        strand at sigma_t in (0, 1), carry/return wave-speed ratio gamma = c_r / c_c."""
        sigma_d = float(sigma_d) % 2.0
        if not (0.0 < sigma_t < 1.0):
            raise ValueError("the take-up must lie on the return strand: 0 < sigma_t < 1")
        if abs(sigma_d - sigma_t) < _TOL:
            raise ValueError("drive and take-up cannot coincide")
        to_x = lambda sig: (sig - sigma_d) % 2.0
        xi = to_x(sigma_t)
        cuts = sorted({to_x(0.0), to_x(1.0), xi} - {0.0} | {2.0})
        segs, a = [], 0.0
        for b in cuts:
            if b - a > _TOL:
                mid = (0.5 * (a + b) + sigma_d) % 2.0
                carry = mid >= 1.0
                segs.append((a, b, Segment(b - a, gamma if carry else 1.0,
                                           r_carry if carry else r_return,
                                           "carry" if carry else "return")))
            a = b
        up = _merge([s for a_, b_, s in segs if b_ <= xi + _TOL])
        dn = _merge([s for a_, b_, s in segs if a_ >= xi - _TOL])
        return cls(tuple(up), tuple(dn), sigma_d=sigma_d, gamma=float(gamma))

    # ------------------------------------------------------------------ properties
    @property
    def segments(self) -> tuple[Segment, ...]:
        return self.upstream + self.downstream

    @property
    def xi(self) -> float:
        """Take-up position in the loop coordinate."""
        return float(sum(s.length for s in self.upstream))

    @property
    def length(self) -> float:
        return float(sum(s.length for s in self.segments))

    @property
    def x0(self) -> np.ndarray:
        """Start of each segment (and total length as last entry), loop coordinate."""
        return self._x0

    @property
    def belt_mass(self) -> float:
        """Dimensionless belt mass, int mu_hat dx (= 1 + gamma**2 for the standard loop)."""
        return float(sum(s.mu * s.length for s in self.segments))

    def reversed(self) -> "Loop":
        """Same loop traversed against belt travel (the spectrum is invariant)."""
        return Loop(tuple(self.downstream[::-1]), tuple(self.upstream[::-1]),
                    sigma_d=self.sigma_d, gamma=self.gamma)

    def segment_index(self, x) -> np.ndarray:
        """Index of the segment containing x (right-continuous, x = 2 in the last one)."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self._x0, x, side="right") - 1
        return np.clip(idx, 0, len(self.segments) - 1)

    def sigma(self, x) -> np.ndarray:
        """Geometric coordinate of loop position x."""
        return (np.asarray(x, dtype=float) + self.sigma_d) % 2.0

    # ------------------------------------------------------------------ static fields
    def cumulative(self, attr: str, x) -> np.ndarray:
        """int_0^x q dx' for the piecewise-constant segment property q = 'mu' or 'r'."""
        x = np.asarray(x, dtype=float)
        q = np.array([getattr(s, attr) for s in self.segments])
        base = np.concatenate(([0.0], np.cumsum(q * np.diff(self._x0))))
        i = self.segment_index(x)
        return base[i] + q[i] * (x - self._x0[i])

    def quasi_static(self, attr: str, x) -> np.ndarray:
        """Q(x) = int_xi^x q dx': quasi-static dynamic tension per unit load
        (q = 'mu': inertial load, unit acceleration; q = 'r': resistances, phi = 1)."""
        return self.cumulative(attr, x) - self.cumulative(attr, self.xi)

    def quasi_static_integral(self, attr: str) -> float:
        """int_0^2 Q dx, exact (Q is piecewise linear). Static take-up motion = half of it."""
        xs = self._x0
        c = self.cumulative(attr, xs)
        return float(np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(xs))
                     - self.length * self.cumulative(attr, self.xi))


def _merge(segs: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        if out and out[-1].g == s.g and out[-1].r == s.r and out[-1].strand == s.strand:
            p = out.pop()
            s = Segment(p.length + s.length, s.g, s.r, s.strand)
        out.append(s)
    return out
