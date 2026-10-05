"""Conveyors from the literature: single source of case data for all figures and tables.

Phase 5.2 rewrite. Every input is a `Datum` that records where it comes from, so that the
parameter table of the paper can show the provenance of each number and the figures can be
regenerated after the source checks of phase 5.3. Tags follow the first author of the source.

Conventions (phase 5.2)
-----------------------
Lengths and positions
    L is the strand length (centre distance along the belt). Positions are geometric, in
    units of L: sigma = 0 head pulley, sigma = 1 tail pulley, return strand in (0, 1), as in
    `Loop.from_positions`. For a head drive xi = sigma_t.

Inertial line densities
    mu_r = belt + reduced mass of the return idlers;
    mu_c = belt + reduced mass of the carry idlers + coupled bulk material.
    Reduced idler mass = sum of I / r^2 of the rolls per metre (4 I / d^2). When a source gives
    only the roll mass, that mass is used and flagged: it is an upper bound of the reduced
    mass, since I <= m r^2 for any roll (thin-shell limit).

Material coupling (Lodewijks 2002, eqs. 3-5)
    Eq. 4 (full coupling, all material mass in the carry strand) is his lower limit of the
    wave speed and eq. 3 (no material) the upper limit. Field values of the coupling factor
    alpha: 0.8-1 for fine wet materials (mineral sands, bauxite, run-of-mine coal); lowest
    0.3 for large lumps of dry rock ore. Eq. 5 as printed, c_eff = c_effU - alpha C_act/C_des,
    is dimensionally inconsistent; we read it as an interpolation of the wave speed between
    the two limits, c = c_U - alpha (c_U - c_L) at design capacity, which recovers eq. 4 at
    alpha = 1 and eq. 3 at alpha = 0. The equivalent inertial density is
    mu_c = 1 / [(1 - alpha) / sqrt(m_U) + alpha / sqrt(m_L)]^2, independent of EA.
    (His worked example in section 5 uses the name "load coupling factor" for the ratio of
    loaded to bare-belt wave speed, 0.73 = 455/621 and 0.77 = 1098/1432: a different
    quantity. We follow eq. 5.) Base value alpha = 1; the range per case depends on the
    material.

Belt stiffness EA
    Hierarchy: published > derived from a published wave speed or stiffness > catalogue.
    EA only sets c_r (seconds); beta, gamma and xi do not depend on it.

Take-up (beta)
    The belt sees the 2:1-equivalent mass M_belt = 4 M / n^2 with M = M_c + i^2 M_w
    (`GravityTakeUp`). Rules, in order:
      1. rigging known (n, i, M_w): exact mapping;
      2. T_t and M_w known: for ideal rigging and negligible carriage mass,
         beta = 4 T_t^2 / (M_w g^2 mu_r L), whatever n and i (n T_t = i M_w g);
      3. only M_w known: direct counterweight on n = 2 strands assumed;
      4. only T_t known: direct counterweight on n = 2 strands assumed (M_w = 2 T_t / g).
    Rules 3 and 4 give the largest beta among direct counterweights with n >= 2 (and among
    roped rigs with i <= 1), i.e. an upper bound for the take-up mass effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G_STD = 9.80665
KINDS = ("published", "measured", "derived", "catalogue", "assumed")


@dataclass(frozen=True)
class Datum:
    """A value with its provenance and an optional plausible range [lo, hi]."""
    value: object
    kind: str
    note: str = ""
    lo: float | None = None
    hi: float | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown provenance {self.kind!r}")
        if self.lo is not None and self.hi is not None and not self.lo <= self.value <= self.hi:
            raise ValueError(f"value {self.value} outside its range [{self.lo}, {self.hi}]")

    @property
    def range(self):
        return (self.value if self.lo is None else self.lo,
                self.value if self.hi is None else self.hi)


def P(v, note="", lo=None, hi=None): return Datum(v, "published", note, lo, hi)
def Ms(v, note="", lo=None, hi=None): return Datum(v, "measured", note, lo, hi)
def D(v, note="", lo=None, hi=None): return Datum(v, "derived", note, lo, hi)
def C(v, note="", lo=None, hi=None): return Datum(v, "catalogue", note, lo, hi)
def A(v, note="", lo=None, hi=None): return Datum(v, "assumed", note, lo, hi)


def coupled_carry_density(m_unloaded: float, m_loaded: float, alpha: float) -> float:
    """Inertial carry density for coupling factor alpha (Lodewijks 2002, eq. 5 read as a
    wave-speed interpolation between eqs. 3 and 4). m_unloaded = belt + carry idlers,
    m_loaded = m_unloaded + material."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return 1.0 / ((1.0 - alpha) / np.sqrt(m_unloaded) + alpha / np.sqrt(m_loaded)) ** 2


def _v(d):
    return None if d is None else d.value


@dataclass
class Case:
    tag: str
    name: str
    bibkey: str
    L: Datum
    sigma_t: list                      # take-up positions (units of L); several if unknown
    sigma_d: Datum = field(default_factory=lambda: P(0.0, "head drive"))
    # line densities, kg/m
    m_b: Datum | None = None           # belt
    m_ic: Datum | None = None          # carry idlers, reduced
    m_ir: Datum | None = None          # return idlers, reduced
    m_l: Datum | None = None           # material at the capacity used
    alpha: Datum = field(default_factory=lambda: A(1.0, "full coupling", 0.8, 1.0))
    mu_r_given: Datum | None = None    # when only the inertial density is known
    gamma_given: Datum | None = None   # when the wave-speed ratio is known directly
    c_r_given: Datum | None = None     # return wave speed, m/s
    EA: Datum | None = None            # N
    # take-up
    M_w: Datum | None = None           # counterweight, kg
    T_t: Datum | None = None           # static belt tension at the take-up, N
    n: Datum | None = None             # belt strands carrying the carriage
    i: Datum | None = None             # counterweight travel per carriage travel
    M_c: Datum | None = None           # carriage and pulleys, kg
    # operation (filled in phase 5.3)
    V: Datum | None = None             # m/s
    t_a: Datum | None = None           # start duration, s
    profile: str | None = None         # start profile kind
    E: Datum | None = None             # Euler-Eytelwein factor e^(mu theta) of the drive
    F_U: Datum | None = None           # running peripheral force, N
    carry_profile: Datum | None = None  # ((distance from tail, elevation), ...), m
    notes: str = ""

    # ------------------------------------------------------------ densities and speeds
    @property
    def mu_r(self) -> float:
        if self.mu_r_given is not None:
            return self.mu_r_given.value
        return self.m_b.value + self.m_ir.value

    def mu_c(self, alpha: float | None = None) -> float:
        if self.gamma_given is not None:
            return self.mu_r * self.gamma_given.value ** 2
        a = self.alpha.value if alpha is None else alpha
        mu_u = self.m_b.value + (0.0 if self.m_ic is None else self.m_ic.value)
        return coupled_carry_density(mu_u, mu_u + self.m_l.value, a)

    def gamma(self, alpha: float | None = None) -> float:
        if self.gamma_given is not None:
            return self.gamma_given.value
        return float(np.sqrt(self.mu_c(alpha) / self.mu_r))

    @property
    def c_r(self) -> float | None:
        if self.c_r_given is not None:
            return self.c_r_given.value
        if self.EA is None:
            return None
        return float(np.sqrt(self.EA.value / self.mu_r))

    # ------------------------------------------------------------ take-up
    def takeup_rule(self):
        """(rule, M_w, i, n) with the rigging used to compute beta (see the module notes)."""
        Mw, Tt, n, i = _v(self.M_w), _v(self.T_t), _v(self.n), _v(self.i)
        if Mw is not None and n is not None:
            return "rigging", Mw, (1.0 if i is None else i), int(n)
        if Mw is not None and Tt is not None:
            # n = 2 with i chosen so that i M_w g = 2 T_t: exact for any ideal rigging
            return "T_t and M_w", Mw, 2.0 * Tt / (Mw * G_STD), 2
        if Mw is not None:
            return "M_w only, n = 2 assumed", Mw, 1.0, 2
        if Tt is not None:
            return "T_t only, n = 2 assumed", 2.0 * Tt / G_STD, 1.0, 2
        return "missing", None, None, None

    def takeup(self):
        """GravityTakeUp reproducing the case (None if the take-up is unknown)."""
        from beltloop import GravityTakeUp
        rule, Mw, i, n = self.takeup_rule()
        if Mw is None:
            return None
        return GravityTakeUp(M_w=Mw, M_c=0.0 if self.M_c is None else self.M_c.value,
                             i=i, strands=n)

    @property
    def beta(self) -> float | None:
        rule, Mw, i, n = self.takeup_rule()
        if Mw is None:
            return None
        Mc = 0.0 if self.M_c is None else self.M_c.value
        M = Mc + i * i * Mw
        return 4.0 * M / (n * n * self.mu_r * self.L.value)

    # ------------------------------------------------------------ geometry
    @property
    def head_drive(self) -> bool:
        return self.sigma_d.value == 0.0

    @property
    def xis(self):
        """Take-up positions in the loop coordinate from the drive exit."""
        return [(s - self.sigma_d.value) % 2.0 for s in self.sigma_t]

    def assumed_inputs(self):
        """Names of the inputs that are assumptions (for the provenance column)."""
        out = [k for k, v in self.__dict__.items() if isinstance(v, Datum) and v.kind == "assumed"
               and not (k == "alpha" and self.gamma_given is not None)]
        if self.takeup_rule()[0].endswith("assumed") and "n" not in out:
            out.append("n")
        return out

    # ------------------------------------------------------------ model
    def conveyor(self, sigma_t: float | None = None, alpha: float | None = None,
                 EA: float | None = None, **kw):
        """Dimensional `Conveyor` (horizontal unless carry_profile is set). Needs EA."""
        from beltloop import Conveyor
        EA = (None if self.EA is None else self.EA.value) if EA is None else EA
        if EA is None:
            raise ValueError(f"{self.tag}: EA unknown (phase 5.3)")
        L = self.L.value
        st = self.sigma_t[0] if sigma_t is None else sigma_t
        m_r = None if self.m_b is None else self.m_b.value
        m_c = None if (self.m_b is None or self.m_l is None) else self.m_b.value + self.m_l.value
        prof = ((0.0, 0.0),) if self.carry_profile is None else self.carry_profile.value
        return Conveyor(L=L, EA=EA, mu_r=self.mu_r, mu_c=self.mu_c(alpha),
                        drive_position=self.sigma_d.value * L, takeup_position=st * L,
                        takeup=self.takeup(), m_r=m_r, m_c=m_c, carry_profile=prof, **kw)


# =============================================================================== cases
# Values as derived in phases 3-4 (sources in the project notes, sections 4.10-4.20 and 7).
# Phase 5.3 checks each one against its source; until then, "published" means "as read in
# phases 3-4".
FT = 0.3048

CASES_FULL = [
    Case("H", "Harrison 1983/85", "harrison1983", L=P(5100.0), sigma_t=[0.005],
         mu_r_given=D(79.0, "consistent with Harrison's eq. 2; 39 kg/m from the text", 39.0, 79.0),
         gamma_given=Ms(0.97, "S1->S3->S2 times, Harrison 1985b Fig. 4a", 0.90, 1.06),
         c_r_given=Ms(1450.0, "7 s wave period on 5.1 km"),
         M_w=P(20e3), n=P(4, "Harrison 1985b Fig. 2a"), i=A(1.0, "direct"), V=P(3.7),
         notes="consistency case; plotted at gamma = 1 on the maps; xi read from the inset "
               "of Fig. 5a (0.002-0.02)"),
    Case("S", "Song et al. 2012", "song2012", L=P(7117.0), sigma_t=[0.001, 0.999],
         m_b=P(27.0), m_ic=P(11.25), m_ir=P(10.8), m_l=D(66.7, "600 t/h at 2.5 m/s"),
         alpha=A(1.0, "coal", 0.8, 1.0), EA=P(104e6, "1 m width"),
         M_w=P(4500.0), n=P(2, "2:1 in their eqs. 5-6"), i=A(1.0),
         V=P(2.5), t_a=P(300.0), profile="sine",
         notes="head (their Fig. 1) / tail (their model); inconsistent running tensions: "
               "structure only"),
    Case("G", "Gao et al. 2026", "gao2026", L=P(4500.0), sigma_t=[0.001],
         m_b=P(40.1), m_ir=P(0.0, "no idler masses given"), m_l=P(154.2),
         alpha=A(1.0, "material not specified", 0.3, 1.0),
         EA=P(1.56e8, "1.3e8 N/m per width x 1.2 m"), M_w=P(1000.0),
         notes="no idler masses given"),
    Case("LL", "Li and Li 2009", "li2009", L=P(7600.0), sigma_t=[0.005],
         m_b=P(54.0), m_ic=D(32.2, "from their c = 837 m/s with full coupling"),
         m_ir=D(12.9, "carry idlers scaled by the 3 m / 1.2 m spacing", 6.45, 19.35),
         m_l=D(173.6, "2500 t/h at 4 m/s"), alpha=A(1.0, "material not specified", 0.8, 1.0),
         EA=P(182e6, "1300 kN/cm x 140 cm"), M_w=P(42800.0),
         n=A(2, "their Fig. 4: ~200 kN on the slack side vs 210 kN = M g / 2"), i=A(1.0),
         V=P(4.0),
         notes="175 m descent: static state only; motors in steps every 4 s (no speed control)"),
    Case("SM", "Suchorab-Matuszewska et al. 2025 (KGHM, variant 1)", "suchorab2025",
         L=P(3000.0), sigma_t=[0.1], m_b=D(64.8, "54 kg/m2 x 1.2 m"),
         m_ic=P(3 * 9.10 / 0.83, "roll masses (upper bound of reduced mass): 3 x 9.10 kg / 0.83 m"),
         m_ir=P(2 * 12.30 / 2.5, "roll mass 12.30 kg / 2.5 m, two rolls per set assumed",
                12.30 / 2.5, 2 * 12.30 / 2.5),
         m_l=D(2000 / 3.6 / 3.0, "2000 t/h at 3 m/s"),
         alpha=A(1.0, "run-of-mine copper ore", 0.3, 1.0),
         T_t=P(140e3, "S(3), QNK-TT report"), V=P(3.0),
         E=D(float(np.exp(0.35 * np.radians(458.0))), "mu = 0.35, 458 deg wrap (QNK-TT report)"),
         notes="EA missing (St 4000, catalogue in 5.3); take-up at node 3, 300 m from the head"),
    Case("Lo", "Lodewijks 1996, ch. 8", "lodewijks1996", L=P(1000.0), sigma_t=[0.001],
         m_b=P(14.28), m_ic=P(13.38), m_ir=P(6.95), m_l=P(133.5),
         alpha=A(1.0, "coal", 0.8, 1.0), EA=D(4.214e6, "E = 340.9 MPa times the belt section"),
         T_t=P(42.66e3 / 2, "take-up force 42.66 kN = 2 T_t"), V=P(5.0), t_a=P(30.0),
         notes="tensioning pulley 1606 kg (reduced) not included (M_c candidate, 5.3)"),
    Case("Pa", "Pascual et al. 2005", "pascual2005", L=P(2561.0), sigma_t=[0.999],
         m_b=P(118.0), m_ic=P(67.0), m_ir=P(20.0), m_l=P(287.0),
         alpha=A(1.0, "large rocks: Lodewijks' lowest field value applies", 0.3, 1.0),
         EA=P(1.74e9, "k = mu_ef v0^2, their eq. 32 (to verify: c_r ~ 3550 m/s)"),
         M_w=P(45.5e3), V=P(4.75),
         notes="take-up at the tail; n not stated; start time of the thesis not usable"),
    Case("Si", "Sinaga 2008 (KPC)", "sinaga2008", L=P(13100.0), sigma_t=[0.001],
         m_b=P(29.7), m_ic=C(9.5, "provisional catalogue estimate", 8.0, 11.0),
         m_ir=C(4.0, "provisional catalogue estimate", 3.0, 5.0),
         m_l=D(4200 / 3.6 / 8.5, "4200 t/h at 8.5 m/s (design 4500 t/h)"),
         alpha=A(1.0, "coal", 0.8, 1.0), M_w=P(46.9e3), T_t=P(115e3), V=P(8.5), t_a=P(780.0),
         notes="EA missing (ST2100, catalogue in 5.3)"),
    Case("WR", "Wheatley and Rubel 2021", "wheatley2021",
         L=D(274.6, "from 274 m and 18 m lift"), sigma_t=[0.05, 0.5, 0.999],
         m_b=C(25.83, "PN1250/4 with 10 + 4 mm covers, Fenner Dunlop 2009"),
         m_ic=C(13.10, "CEMA C6, 4 I / d^2 at 1.2 m"), m_ir=C(5.26, "CEMA C6 at 3 m"),
         m_l=D(1800 / 3.6 / 2.2, "1800 t/h at 2.2 m/s"), alpha=A(1.0, "iron ore fines", 0.8, 1.0),
         EA=C(12000e3 * 0.9, "12000 N/mm x 0.9 m; Zarzycki 2023: 13-30 % lower at low load",
              0.70 * 12000e3 * 0.9, 12000e3 * 0.9),
         M_w=P(7550.0), V=P(2.2),
         notes="take-up position unknown (dotted line on the maps); thesis start time not usable"),
    Case("Su", "Surtees 1995 (SASOL)", "surtees1995", L=P(805.0),
         sigma_d=P(152.0 / 805.0, "drives 152 m from the head"),
         sigma_t=[152.0 / 805.0 + 0.01],
         m_b=P(35.6), m_ic=P(12.0, "18 kg per 1.5 m"), m_ir=P(5.33, "16 kg per 3 m"),
         m_l=D(3500 / 3.6 / 4.4, "3500 t/h at 4.4 m/s"), alpha=A(1.0, "coal", 0.8, 1.0),
         M_w=P(4795.0, "design 1; design 2 (steel cord ST1250): 20 387 kg", 4795.0, 20387.0),
         n=P(2, "vertical direct, M = 2 T_2 / g"), i=A(1.0), V=P(4.4),
         notes="design data; take-up assumed right after the drives; intermediate-drive panel only"),
    Case("NC", "Nordell and Ciozda 1984, case 1", "nordell1984", L=P(8150 * FT),
         sigma_d=P(5150 / 8150, "primary and secondary drives together"), sigma_t=[5450 / 8150],
         c_r_given=P(1450.0, "BELTFLEX"), gamma_given=D(1450.0 / 590.0, "1450 / 590 m/s"),
         mu_r_given=A(1.0, "placeholder: not given (only frequencies are used)"),
         notes="no take-up mass nor EA: frequencies in the limit beta -> 0 only; retarder"),
]

BY_TAG = {c.tag: c for c in CASES_FULL}

# Cases on the head-drive maps, in the legacy order.
MAP_TAGS = ["H", "S", "G", "LL", "SM", "Lo", "Pa", "Si", "WR"]

# Case whose take-up position is unknown (drawn as a line across xi).
UNKNOWN_POSITION = "WR"


def _legacy(c: Case):
    """(tag, name, L, mu_r, mu_c, M, n, xis, note) as used by beta_regime and modal_maps:
    M and n are chosen so that 4 M / (n^2 mu_r L) is the case's beta."""
    rule, Mw, i, n = c.takeup_rule()
    M = c.beta * n * n * c.mu_r * c.L.value / 4.0
    return (c.tag, c.name, c.L.value, c.mu_r, c.mu_c(), M, n, c.xis, c.notes or rule)


CASES = [_legacy(BY_TAG[t]) for t in MAP_TAGS]

# Intermediate drives (panel of T_1 / (4 t_B) against the drive offset l1 = sigma_d).
_nc, _su = BY_TAG["NC"], BY_TAG["Su"]
NORDELL_CIOZDA = dict(tag="NC", L=_nc.L.value, c_r=_nc.c_r, gamma=_nc.gamma(),
                      sigma_d=_nc.sigma_d.value, sigma_t=_nc.sigma_t[0])
SASOL = dict(tag="Su", L=_su.L.value, gamma=_su.gamma(), sigma_d=_su.sigma_d.value,
             sigma_t=_su.sigma_t[0], note="design data; take-up position assumed next to the drives")


def points():
    """(tag, beta, gamma, xi, unknown_position) for every case and take-up position."""
    out = []
    for t in MAP_TAGS:
        c = BY_TAG[t]
        for xi in c.xis:
            out.append((t + ("t" if t == "S" and xi > 0.5 else ""), c.beta, c.gamma(), xi,
                        t == UNKNOWN_POSITION))
    return out


def table(alphas=(1.0, 0.8, 0.3)):
    """Print the case table with provenance and the sensitivity of gamma to the coupling."""
    print(f"{'tag':4s}{'L':>8}{'mu_r':>7}{'mu_c':>7}{'gamma':>7}"
          + "".join(f"{'g(a=' + format(a, 'g') + ')':>9}" for a in alphas[1:])
          + f"{'beta':>8}{'c_r':>7}  take-up rule; assumed inputs")
    for c in CASES_FULL:
        sens = "".join(f"{(np.nan if c.gamma_given is not None else c.gamma(a)):9.2f}"
                       for a in alphas[1:])
        b, cr = c.beta, c.c_r
        print(f"{c.tag:4s}{c.L.value:8.0f}{c.mu_r:7.1f}{c.mu_c():7.1f}{c.gamma():7.2f}{sens}"
              f"{(np.nan if b is None else b):8.4f}{(np.nan if cr is None else cr):7.0f}  "
              f"{c.takeup_rule()[0]}; {', '.join(c.assumed_inputs())}")


if __name__ == "__main__":
    table()
