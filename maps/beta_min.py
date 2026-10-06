"""Phase 5.4: what sets the take-up tension, and why beta does not grow with length.

With a gravity take-up the belt-side mass ratio follows from the take-up tension:

    beta = K T_t / W_r,    K = (4 / n) lam,    W_r = g mu_r L,

W_r the (inertial) weight of the return strand, lam = g M / (n T_t) (1 for a directly hung
counterweight). K = 2 for a direct counterweight on two strands, 1 on four. The design of
T_t therefore fixes beta, and each minimum-tension requirement gives a floor:

  grip at the drive (DIN 22101 eqs. 47-50), running and start:
      T_2 >= p F_U / (E - 1),  T_t = T_2 + (drive exit -> take-up),
      beta_grip = K [p psi / (E - 1) + psi_A],  psi = F_U / W_r,
      psi = C f (m_R + 2 m_b + m_L) / mu_r + (H / L) m_L / mu_r;
  sag at the low-tension point (DIN 22101 eqs. 51-52):
      T(s) >= g m' l / (8 h_rel), carried to the take-up through the running state.

psi and psi_A do not depend on L at a given slope and load: with a length-proportional
requirement beta is independent of L. Fixed requirements (sag on a level belt, secondary
resistances through C(L)) give beta ~ 1 / L and dominate short belts. The source of each
case decides which one was used (`Case.takeup_basis`).

All requirements are computed in the steady running state of `Conveyor.running_tension`
(gravity from the carry profile, take-up holding T_t) with strand resistances from DIN eq.
14 scaled to the case's F_U. Start: quasi-static uniform acceleration that raises the drive
force to p F_U (the universal start curve of step 4.5 is ~1 for t_a >= 2-3 T_1).

    python maps/beta_min.py         # table and the working figure figures/beta_min.pdf
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import cases as cs

G = cs.G_STD
P_DIN_MAX = 1.7               # DIN 22101 section 7.3.1: start factor p_A,max <= 1.7
H_REL = (0.01, 0.02)          # DIN 22101 sag limit (steady running) and common practice
ORDER = ["Lo", "S", "LL", "WR", "Pa", "Su", "SM", "Si", "H"]


# ------------------------------------------------------------------------- running state
def strand_resistances(c: cs.Case, F_U: float | None = None, q_r: float = 1.0):
    """(r_r, r_c, C f): running resistances per unit length (N/m) of the return and carry
    strands, DIN eq. 14 with a common C f fitted so that (r_r + r_c) L + g m_l H = F_U.
    q_r scales the return-strand coefficient against the carry one (1: common f, DIN's
    estimate; QNK gives about 0.55 for SM, phase 5.4); the total stays F_U."""
    F = c.F_U.value if F_U is None else F_U
    k = cs.KAPPA_IDLER
    m_ro = (0.0 if c.m_ic is None else c.m_ic.value) / k
    m_ru = (0.0 if c.m_ir is None else c.m_ir.value) / k
    wr, wc = q_r * (m_ru + c.m_b.value), m_ro + c.m_b.value + c.m_l.value
    H = lift(c)
    Cf = (F - G * c.m_l.value * H) / (G * c.L.value * (wr + wc))
    return Cf * G * wr, Cf * G * wc, Cf


def lift(c: cs.Case) -> float:
    if c.carry_profile is None:
        return 0.0
    p = c.carry_profile.value
    return p[-1][1] - p[0][1]


def running_conveyor(c: cs.Case, sigma_t=None, M_w=None, F_U=None, q_r=1.0):
    """Dimensional Conveyor with the strand resistances (any EA: statics do not need it)."""
    r_r, r_c, _ = strand_resistances(c, F_U, q_r)
    cc = c if M_w is None else replace(c, M_w=cs.P(M_w), T_t=None)
    EA = 1e8 if c.EA is None else None
    return cc.conveyor(sigma_t=sigma_t, EA=EA, r_r=r_r, r_c=r_c)


@dataclass
class State:
    """Tensions as T_t + offset along the loop (s from the drive exit), running and start."""
    s: np.ndarray
    T_run: np.ndarray
    T_start: np.ndarray
    carry: np.ndarray
    T_t: float
    p: float

    @property
    def exit_entry(self):
        return (self.T_run[0], self.T_run[-1]), (self.T_start[0], self.T_start[-1])


def state(c: cs.Case, p: float = 1.0, sigma_t=None, M_w=None, F_U=None, n_s: int = 4001,
          q_r: float = 1.0) -> State:
    """Running state and quasi-static start with peak drive force p F_U (uniform belt
    acceleration a = (p - 1) F_U / ((mu_r + mu_c) L))."""
    cv = running_conveyor(c, sigma_t, M_w, F_U, q_r)
    L = cv.L
    s = np.unique(np.r_[np.linspace(0.0, 2 * L, n_s), cv.xi * L])
    T = cv.running_tension(s)
    F = c.F_U.value if F_U is None else F_U
    a = (p - 1.0) * F / ((cv.mu_r + cv.mu_c) * L)
    T_s = T + cv.mu_r * L * a * cv.loop(1.0).quasi_static("mu", s / L)
    sig = (s + cv.drive_position) % (2 * L)
    return State(s, T, T_s, sig >= L, cv.takeup.T_t, p)


def grip_requirement(st: State, E: float, start: bool = False) -> float:
    """Smallest T_t with T_entry <= E T_exit (DIN eqs. 48-49); tensions are T_t + offsets."""
    T = st.T_start if start else st.T_run
    a, b = T[0] - st.T_t, T[-1] - st.T_t
    return (b - E * a) / (E - 1.0)


def sag_requirement(c: cs.Case, st: State, h_rel: float = 0.01) -> tuple[float, float, bool]:
    """Smallest T_t with T(s) >= g m' l / (8 h_rel) on both strands in running (DIN eqs.
    51-52): (T_t required, s of the critical point, on the carry strand?)."""
    T_min = np.where(st.carry, G * (c.m_b.value + c.m_l.value) * c.l_o.value,
                     G * c.m_b.value * c.l_u.value) / (8.0 * h_rel)
    d = T_min - st.T_run
    i = int(np.argmax(d))
    return st.T_t + float(d[i]), float(st.s[i]), bool(st.carry[i])


def start_factor_from_time(c: cs.Case, kind: str = "sine") -> float | None:
    """Quasi-static start factor of the published start: 1 + a_max (mu_r + mu_c) L / F_U."""
    if c.t_a is None or c.V is None or c.F_U is None:
        return None
    from beltloop import PEAK_FACTOR
    a = PEAK_FACTOR[kind] * c.V.value / c.t_a.value
    return 1.0 + a * (c.mu_r + c.mu_c()) * c.L.value / c.F_U.value


# ------------------------------------------------------------------------- per case
@dataclass
class Design:
    tag: str
    L: float
    beta: float
    K: float                 # beta W_r / T_t = (4 / n) lam
    T_t: float
    W_r: float
    F_U: float
    psi: float               # F_U / W_r
    E: float | None
    p: float | None          # start factor of the source (None: DIN maximum 1.7 used for grip)
    p_time: float | None     # quasi-static start factor of the published start time
    E_run: float             # e^(mu theta) needed in running with the actual T_t
    E_start: float | None    # ... at start with p (DIN maximum 1.7 if the source gives none)
    T_grip_run: float | None
    T_grip_start: float | None
    T_sag: dict              # h_rel -> T_t required
    sag_point: str
    basis: str

    @property
    def tau(self) -> float:
        """T_t / F_U: beta = K psi tau."""
        return self.T_t / self.F_U

    @property
    def ratio(self) -> float:
        """T_t / W_r: beta = K T_t / W_r."""
        return self.T_t / self.W_r

    def margin(self, which: str) -> float | None:
        req = {"grip_run": self.T_grip_run, "grip_start": self.T_grip_start,
               "sag1": self.T_sag.get(0.01), "sag2": self.T_sag.get(0.02)}[which]
        return None if req is None else self.T_t / req

    @property
    def floor(self) -> float | None:
        """Largest computable requirement: DIN grip with the source's E and p, and sag at
        h_rel = 2 % (the practice of the sources: Surtees 2 %, Lodewijks below 1.5 %; DIN's
        1 % is reported separately). None without E (the grip floor is then unknown)."""
        if self.T_grip_run is None:
            return None
        reqs = [r for r in (self.T_grip_run, self.T_grip_start, self.T_sag.get(0.02)) if r]
        return max(reqs)

    def beta_of(self, T: float) -> float:
        return self.K * T / self.W_r


def design(c: cs.Case, sigma_t=None, M_w=None, q_r: float = 1.0) -> Design:
    """Take-up tension of a case against its grip and sag requirements."""
    cc = c if M_w is None else replace(c, M_w=cs.P(M_w), T_t=None)
    beta = cc.beta
    if c.F_A is not None:
        p = c.F_A.value / c.F_U.value
    else:
        p = None if c.p_A is None else c.p_A.value
    st = state(cc, P_DIN_MAX if p is None else p, sigma_t, q_r=q_r)
    T_t, L = st.T_t, c.L.value
    W_r = G * c.mu_r * L
    (Tr0, Tr1), (Ts0, Ts1) = st.exit_entry
    E = None if c.E is None else c.E.value
    Tsag, pt = {}, ""
    for h in H_REL:
        T, s_c, carry = sag_requirement(c, st, h)
        Tsag[h] = T
        if h == H_REL[0]:
            pt = f"{'carry' if carry else 'return'} at s/L = {s_c / L:.2f}"
    return Design(c.tag, L, beta, beta * W_r / T_t, T_t, W_r, c.F_U.value, c.F_U.value / W_r,
                  E, p, start_factor_from_time(c), Tr1 / Tr0, Ts1 / Ts0,
                  None if E is None else grip_requirement(st, E),
                  None if E is None else grip_requirement(st, E, start=True),
                  Tsag, pt, c.takeup_basis)


def design_simple(c: cs.Case) -> Design:
    """Cases without line masses (H): take-up at the drive, horizontal, T_out = T_t."""
    T_t = c.takeup().T_t
    W_r = G * c.mu_r * c.L.value
    F = c.F_U.value
    return Design(c.tag, c.L.value, c.beta, c.beta * W_r / T_t, T_t, W_r, F, F / W_r, None,
                  None, None, (T_t + F) / T_t, None, None, None, {}, "", c.takeup_basis)


def designs():
    out = {}
    for tag in ORDER:
        c = cs.BY_TAG[tag]
        if tag == "H":
            out[tag] = design_simple(c)
        elif tag == "S":
            out[tag] = design(c, sigma_t=0.999)        # 173 kN is the tail take-up (their model)
        elif tag == "Su":
            out[tag] = design(c)                       # design 2 (base), 100 kN
            out["Su1"] = design(c, M_w=c.M_w.lo)       # design 1, 23.6 kN
            out["Su1"].tag = "Su1"
        else:
            out[tag] = design(c)
    return out


# ------------------------------------------------------------------------- scaling
def scaled_case(c: cs.Case, k: float) -> cs.Case:
    """Same belt, load, slope and resistance coefficient on a k times longer route: F_U and H
    scale with k at constant C f (pure length scaling of the grip requirement)."""
    r_r, r_c, Cf = strand_resistances(c)
    prof = None if c.carry_profile is None else cs.D(tuple((d * k, h * k) for d, h in
                                                          c.carry_profile.value))
    F = Cf * G * k * c.L.value * (2 * c.m_b.value + c.m_l.value
                                  + ((c.m_ic.value if c.m_ic else 0) + (c.m_ir.value if c.m_ir else 0))
                                  / cs.KAPPA_IDLER) + G * c.m_l.value * lift(c) * k
    return replace(c, L=cs.D(k * c.L.value), carry_profile=prof, F_U=cs.D(F))


# ------------------------------------------------------------------------- short belts
def wr_siblings():
    """Conveyors B and C of Wheatley and Rubel (2021, Table 1): same belt, idlers and
    material as A, published counterweights (n = 2 assumed). (tag, L, H, M_w, F_U)."""
    out = []
    for tag, Lh, H, Mw, P in (("WR-B", 247.0, 11.0, 10000.0, 88e3), ("WR-C", 89.0, 17.0, 6000.0, 94e3)):
        out.append((tag, float(np.hypot(Lh, H)), H, Mw, P / 1.8))
    return out


def wr_sibling_cases():
    a = cs.BY_TAG["WR"]
    ml = 1500 / 3.6 / 1.8
    res = []
    for tag, L, H, Mw, F in wr_siblings():
        res.append(replace(a, tag=tag, L=cs.D(L), M_w=cs.P(Mw), m_l=cs.D(ml), F_U=cs.D(F),
                           carry_profile=cs.A(((0.0, 0.0), (L, H)))))
    return res


# ------------------------------------------------------------------------- output
def table():
    ds = designs()
    print("Take-up tension against its requirements (phase 5.4). Tensions in kN.")
    print(f"{'case':5s} {'L km':>5s} {'beta':>6s} {'K':>4s} {'Tt/Wr':>6s} {'psi':>5s} {'tau':>5s} {'T_t':>6s} "
          f"{'E':>5s} {'p':>5s} {'p(t_a)':>6s} {'E_run':>6s} {'E_st':>5s} {'grip_r':>7s} "
          f"{'grip_s':>7s} {'sag1%':>6s} {'sag2%':>6s}  sag point")
    f = lambda x, w=6, d=1: f"{x:{w}.{d}f}" if x is not None else " " * (w - 1) + "-"
    for k, d in ds.items():
        print(f"{k:5s} {d.L / 1e3:5.2f} {d.beta:6.3f} {d.K:4.2f} {d.ratio:6.3f} {d.psi:5.3f} {d.tau:5.2f} "
              f"{d.T_t / 1e3:6.1f} {f(d.E, 5, 2)} {f(d.p, 5, 2)} {f(d.p_time, 6, 2)} "
              f"{d.E_run:6.2f} {f(d.E_start, 5, 2)} "
              f"{f(None if d.T_grip_run is None else d.T_grip_run / 1e3, 7)} "
              f"{f(None if d.T_grip_start is None else d.T_grip_start / 1e3, 7)} "
              f"{f(d.T_sag[0.01] / 1e3 if d.T_sag else None)} {f(d.T_sag[0.02] / 1e3 if d.T_sag else None)}"
              f"  {d.sag_point}")
    print("\nBasis of T_t as documented by each source:")
    for k, d in ds.items():
        print(f"  {k:4s} {d.basis}")
    print("\nShort belts of the same plant (Wheatley and Rubel 2021, n = 2 assumed):")
    for c in [cs.BY_TAG["WR"]] + wr_sibling_cases():
        d = design(c)
        print(f"  {c.tag:5s} L = {d.L:5.0f} m  T_t = {d.T_t / 1e3:5.1f} kN  beta = {d.beta:5.2f}  "
              f"T_t/W_r = {d.ratio:4.2f}  psi = {d.psi:4.2f}  E_run = {d.E_run:4.2f}  "
              f"sag 1 % = {d.T_sag[0.01] / 1e3:5.1f} kN")
    return ds


def beta5_B1(c: cs.Case, sigma_t=None) -> float:
    """Exact 5 % threshold of the fundamental (strand B, mode 1) at the case's geometry."""
    from beltloop import Loop, takeup_mass_threshold
    st = c.sigma_t[0] if sigma_t is None else sigma_t
    return takeup_mass_threshold(Loop.from_positions(c.sigma_d.value, st, c.gamma()), "B", 1)


def regime():
    """(tag, L, beta, beta_5(B1), beta / beta_5) for the cases and the WR siblings."""
    rows = []
    for k, d in designs().items():
        c = cs.BY_TAG[k.rstrip("1")]
        b5 = beta5_B1(c, 0.999 if k == "S" else None)
        rows.append((k, d.L, d.beta, b5, d.beta / b5))
    for c in wr_sibling_cases():
        b5 = beta5_B1(c)
        rows.append((c.tag, c.L.value, c.beta, b5, c.beta / b5))
    return rows


def figure(path=None):
    """Working figure. (a) beta against L: filled, published T_t; open, DIN floor with the
    source's E and p (p = 1.7 if not given), sag at 2 %. (b) beta / beta_5(B1)."""
    import matplotlib.pyplot as plt
    ds = designs()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for k, d in ds.items():
        ax.loglog(d.L, d.beta, "o", color="k", ms=5)
        ax.annotate(k, (d.L, d.beta), xytext=(4, 3), textcoords="offset points", fontsize=8)
        if d.floor is not None:
            b0 = d.beta_of(d.floor)
            ax.plot([d.L, d.L], [b0, d.beta], "-", color="0.6", lw=1)
            ax.plot(d.L, b0, "o", mfc="w", mec="k", ms=5)
    for c in wr_sibling_cases():
        ax.loglog(c.L.value, c.beta, "s", color="0.4", ms=4)
        ax.annotate(c.tag, (c.L.value, c.beta), xytext=(4, 3), textcoords="offset points",
                    fontsize=7)
    L = np.logspace(1.8, 4.3, 50)
    for T, ls in ((20e3, ":"), (50e3, "--")):
        ax.plot(L, 2 * T / (G * 35.0 * L), ls, color="0.5", lw=0.8,
                label=f"fixed T_t = {T / 1e3:.0f} kN (mu_r = 35 kg/m, n = 2)")
    ax.set_xlabel("L (m)")
    ax.set_ylabel("beta = K T_t / W_r")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(a) filled: published T_t; open: DIN floor", fontsize=9)
    for tag, Lc, beta, b5, r in regime():
        mk = "s" if tag.startswith("WR-") else "o"
        bx.loglog(Lc, r, mk, color="k" if mk == "o" else "0.4", ms=5 if mk == "o" else 4)
        bx.annotate(tag, (Lc, r), xytext=(4, 3), textcoords="offset points", fontsize=7)
    bx.axhline(1.0, color="0.3", lw=0.8)
    bx.text(400, 1.08, "5 % on the fundamental", fontsize=7)
    bx.set_xlabel("L (m)")
    bx.set_ylabel("beta / beta_5(B1)")
    bx.set_title("(b) distance to the 5 % threshold", fontsize=9)
    fig.tight_layout()
    path = Path(__file__).parent / "figures" / "beta_min.pdf" if path is None else path
    fig.savefig(path)
    fig.savefig(Path(path).with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    table()
    print("\nRegime (beta / beta_5 of the fundamental):")
    for tag, Lc, beta, b5, r in regime():
        print(f"  {tag:5s} L = {Lc:6.0f} m  beta = {beta:5.3f}  beta_5 = {b5:5.3f}  ratio = {r:5.3f}")
    print(figure())
