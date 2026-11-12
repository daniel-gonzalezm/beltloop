"""Independent lumped-mass model of the conveyor loop, integrated in time (SI units).

Used to validate the modal solution. It shares only input data with it (Conveyor
parameters and the elevation profile); every modelling step is redone differently:

* absolute belt displacements u along the path, with both faces of the drive pulley
  moving with the prescribed rigid motion U(t): the inertial load of the start-up is not
  applied, it emerges;
* gravity (belt and material weight, counterweight) is applied as a constant load and the
  run starts from the static equilibrium computed by a linear solve; nothing assumes that
  gravity cancels;
* the take-up is an extra degree of freedom y (carriage travel) with mass M, loaded by
  the counterweight force n T_t; the belt element that wraps the take-up has elongation
  u_R - u_L + n y (n:1 kinematics, n = number of strands carrying the carriage, 2 for a
  single loop), so the rigid motion U does not move the take-up only if the formulation is
  right. The strand count is built in here directly, not through the belt-side mass
  4 M / n^2 used by the modal solution;
* lumped masses, linear springs, Kelvin-Voigt dashpots (C = t_v K) and the Newmark
  average-acceleration integrator (implicit, second order, no numerical damping);
* the start-up kinematics U, V, a are re-implemented here from the thesis formulas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .conveyor import Conveyor


# ---------------------------------------------------------------------------- kinematics
def drive_kinematics(kind: str, V: float, t_a: float, t):
    """Prescribed drive acceleration, velocity and displacement (SI)."""
    t = np.asarray(t, dtype=float)
    u = np.clip(t / t_a, 0.0, 1.0)
    if kind == "sine":
        am = np.pi * V / (2 * t_a)
        a = am * np.sin(np.pi * u)
        v = 0.5 * V * (1 - np.cos(np.pi * u))
        x_in = 0.5 * V * (t_a * u - t_a / np.pi * np.sin(np.pi * u))
    elif kind == "triangular":
        a = np.where(u <= 0.5, 4 * V * t_a * u / t_a ** 2, 4 * V / t_a * (1 - u))
        v = np.where(u <= 0.5, 2 * V * u ** 2, V * (1 - 2 * (1 - u) ** 2))
        x_in = np.where(u <= 0.5, 2 * V * t_a * u ** 3 / 3,
                        V * t_a * (1 / 12 + (u - 0.5) + 2 / 3 * ((1 - u) ** 3 - 0.125)))
    elif kind == "parabolic":
        a = 1.5 * V / t_a * 4 * u * (1 - u)
        v = V * (3 * u ** 2 - 2 * u ** 3)
        x_in = V * t_a * (u ** 3 - 0.5 * u ** 4)
    else:
        raise ValueError(kind)
    a = np.where((t >= 0) & (t <= t_a), a, 0.0)
    disp = np.where(t <= t_a, x_in, x_in + V * (t - t_a))
    return a, v, disp


# ---------------------------------------------------------------------------- model
@dataclass
class LumpedResult:
    t: np.ndarray
    x_mid: np.ndarray          # element midpoints, m, loop coordinate
    T_total: np.ndarray        # element tension (elastic + viscous), N, (n_elem, n_t)
    T_static: np.ndarray       # static element tension at rest, N, (n_elem,)
    y: np.ndarray              # carriage travel from rest, m, (n_t,)

    @property
    def T_dynamic(self) -> np.ndarray:
        return self.T_total - self.T_static[:, None]


class LumpedModel:
    """Chain of n_elements belt elements from the drive exit (node 0) to the drive entry
    (node N), plus the take-up dof. Nodes are placed at the head, tail and take-up."""

    def __init__(self, conveyor: Conveyor, n_elements: int = 1000):
        cv = self.cv = conveyor
        self.n = int(cv.takeup.strands)
        L2 = 2 * cv.L
        to_s = lambda sig: (sig - cv.drive_position) % L2
        s_t = to_s(cv.takeup_position)
        cuts = sorted({to_s(0.0), to_s(cv.L), s_t, L2} - {0.0})
        nodes, a = [0.0], 0.0
        for b in cuts:
            ne = max(2, int(round(n_elements * (b - a) / L2)))
            nodes += list(np.linspace(a, b, ne + 1)[1:])
            a = b
        x = np.array(nodes)
        self.x = x
        self.N = len(x) - 1                         # number of elements
        self.h = np.diff(x)
        self.x_mid = 0.5 * (x[1:] + x[:-1])
        self.i_tu = int(np.argmin(np.abs(x - s_t)))  # node at the take-up (upstream face)
        sig_mid = (self.x_mid + cv.drive_position) % L2
        carry = sig_mid >= cv.L
        self.mu = np.where(carry, cv.mu_c, cv.mu_r)
        self.m = np.where(carry, cv.m_c, cv.m_r)
        self.r = np.where(carry, cv.r_c, cv.r_r)
        self.dh = np.diff(cv.elevation((x + cv.drive_position) % L2))
        self._assemble()

    # dofs: 0..N belt nodes (u), N+1 take-up y
    def _element_map(self, e):
        """Rows: (left, right) element-end displacements in terms of dofs, with the
        take-up jump on the element that starts at the take-up node."""
        T = [(e, 1.0)], [(e + 1, 1.0)]
        if e == self.i_tu:
            T = [(e, 1.0), (self.N + 1, -float(self.n))], [(e + 1, 1.0)]
        return T

    def _assemble(self):
        cv, N = self.cv, self.N
        nd = N + 2
        K = sp.lil_matrix((nd, nd)); M = sp.lil_matrix((nd, nd))
        fg = np.zeros(nd); fr = np.zeros(nd)
        for e in range(N):
            k = cv.EA / self.h[e]
            mh = 0.5 * self.mu[e] * self.h[e]
            left, right = self._element_map(e)
            # strain = (u_right_end - u_left_end)/h; elastic and mass contributions
            B = {}
            for d, c in left:
                B[d] = B.get(d, 0.0) - c
            for d, c in right:
                B[d] = B.get(d, 0.0) + c
            for di, ci in B.items():
                for dj, cj in B.items():
                    K[di, dj] += k * ci * cj
            for end in (left, right):
                for di, ci in end:
                    for dj, cj in end:
                        M[di, dj] += mh * ci * cj
                    # gravity (tangential weight) and resistances, half to each end
                    fg[di] += -0.5 * self.m[e] * cv.g * self.dh[e] * ci
                    fr[di] += -0.5 * self.r[e] * self.h[e] * ci
        M[N + 1, N + 1] += cv.takeup.M
        fg[N + 1] += cv.takeup.force                     # counterweight: n T_t on the carriage
        self.K, self.M = K.tocsr(), M.tocsr()
        self.C = cv.t_v * self.K
        self.fg, self.fr = fg, fr
        self.free = np.array([i for i in range(nd) if i not in (0, N)])
        self.pres = np.array([0, N])

    def element_tension(self, u, v):
        """Element tensions EA (eps + t_v eps_t) for state vectors u, v (dofs x n)."""
        N = self.N
        du = u[1:N + 1] - u[0:N]
        dv = v[1:N + 1] - v[0:N]
        du[self.i_tu] += self.n * u[N + 1]
        dv[self.i_tu] += self.n * v[N + 1]
        return self.cv.EA * (du + self.cv.t_v * dv) / self.h[:, None]

    def static_state(self) -> np.ndarray:
        f, p = self.free, self.free
        u = np.zeros(self.N + 2)
        u[f] = splu(self.K[f][:, f].tocsc()).solve(self.fg[f])
        return u

    def eigenfrequencies(self, n: int) -> np.ndarray:
        """Lowest n circular frequencies (rad/s), prescribed drive (faces fixed)."""
        from scipy.sparse.linalg import eigsh
        f = self.free
        w2 = eigsh(self.K[f][:, f].tocsc(), k=n, M=self.M[f][:, f].tocsc(), sigma=0.0,
                   which="LM", return_eigenvectors=False)
        return np.sqrt(np.sort(w2))

    def start(self, V: float, t_a: float, kind: str = "sine", onset: str = "velocity",
              dt: float = 0.01, t_end: float = 100.0, store_every: int = 10,
              kinematics=None) -> LumpedResult:
        """Start-up from the static state. kinematics(t) -> (a, v, d) in SI overrides the
        profile given by kind and t_a (V is then the final speed, used for phi = v/V)."""
        kin = kinematics if kinematics is not None else (lambda t: drive_kinematics(kind, V, t_a, t))
        f, p = self.free, self.pres
        K, C, M = self.K, self.C, self.M
        Kff, Cff, Mff = K[f][:, f], C[f][:, f], M[f][:, f]
        Kfp, Cfp, Mfp = K[f][:, p], C[f][:, p], M[f][:, p]
        bN, gN = 0.25, 0.5
        Keff = (Kff + gN / (bN * dt) * Cff + 1 / (bN * dt ** 2) * Mff).tocsc()
        lu = splu(Keff)

        def phi(t):
            if onset == "velocity":
                return kin(t)[1] / V
            if onset == "step":
                return 1.0
            return 0.0

        u = self.static_state(); v = np.zeros_like(u); acc = np.zeros_like(u)
        u0 = u.copy()
        a0, v0, d0 = kin(0.0)
        u[p] = d0; v[p] = v0; acc[p] = a0
        rhs0 = self.fg + self.fr * phi(0.0) - K @ u - C @ v - M[:, p] @ acc[p]
        acc[f] = splu(Mff.tocsc()).solve(rhs0[f])

        n_steps = int(round(t_end / dt))
        ts, U, Vv = [0.0], [u.copy()], [v.copy()]
        for n in range(1, n_steps + 1):
            t = n * dt
            ut = u + dt * v + dt ** 2 * (0.5 - bN) * acc
            vt = v + dt * (1 - gN) * acc
            ap, vp, up = kin(t)
            F = self.fg + self.fr * phi(t)
            rhs = (F[f] - Kfp @ [up, up] - Cfp @ [vp, vp] - Mfp @ [ap, ap]
                   + Mff @ ut[f] / (bN * dt ** 2) + Cff @ (gN / (bN * dt) * ut[f] - vt[f]))
            un = np.empty_like(u)
            un[f] = lu.solve(rhs); un[p] = up
            an = (un - ut) / (bN * dt ** 2); vn = vt + gN * dt * an
            an[p] = ap; vn[p] = vp
            u, v, acc = un, vn, an
            if n % store_every == 0:
                ts.append(t); U.append(u.copy()); Vv.append(v.copy())
        U = np.array(U).T; Vv = np.array(Vv).T
        T = self.element_tension(U, Vv)
        Ts = self.element_tension(u0[:, None], np.zeros((len(u0), 1)))[:, 0]
        return LumpedResult(np.array(ts), self.x_mid, T, Ts, U[self.N + 1] - u0[self.N + 1])
