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
    only the roll mass m, the reduced mass is KAPPA_IDLER * m (phase 5.3): CEMA (2007) tables
    5.41-5.44 give, for steel rolls of classes B4-E7 and belt widths 18-48 in, a ratio
    WK^2 / (W (d/2)^2) of 0.72-0.92 (0.67-0.95 if the table pairing is read the other way),
    median about 0.8. Range 0.6-0.92: the lower end covers a source mass that includes the
    shaft. m itself is the upper bound (I <= m r^2 for any roll).

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
    (`GravityTakeUp`). The take-up tension takes precedence over a published mass (phase
    5.3(c)): published masses proved unreliable (Pascual: tail pulley; Song: 4500 kg against
    173 kN), while the tension is checked by the static balance. Rules, in order:
      0. T_t and n known: direct counterweight on the n strands, M_w = n T_t / g (also when
         a mass is published; the source note keeps it as the alternative);
      1. rigging known (n, i, M_w): exact mapping;
      2. T_t and M_w known: for ideal rigging and negligible carriage mass,
         beta = 4 T_t^2 / (M_w g^2 mu_r L), whatever n and i (n T_t = i M_w g);
      3. only M_w known: direct counterweight on n = 2 strands assumed;
      4. only T_t known: direct counterweight on n strands assumed (M_w = n T_t / g), with
         n = 2 unless the source shows the reeving (phase 5.3(b): Pascual's Fig. 5).
    Rules 3 and 4 give the largest beta among direct counterweights with n >= 2 (and among
    roped rigs with i <= 1), i.e. an upper bound for the take-up mass effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G_STD = 9.80665
KINDS = ("published", "measured", "derived", "catalogue", "assumed")

# Reduced mass of an idler roll per unit roll mass, I / (m r^2): CEMA tables 5.41-5.44.
KAPPA_IDLER, KAPPA_LO, KAPPA_HI = 0.80, 0.60, 0.92


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


LBIN2 = 0.45359237 * 0.0254 ** 2    # lbm in^2 -> kg m^2


def cema_reduced(wk2_lb_in2, d_in, spacing):
    """Reduced idler density, kg/m, from a CEMA WK^2 (lb in^2) per idler set of roll
    diameter d_in (in) at the given spacing (m): 4 I / d^2 / spacing."""
    return 4.0 * wk2_lb_in2 * LBIN2 / (d_in * 0.0254) ** 2 / spacing


def reduced_idlers(mass_per_metre, note):
    """Reduced idler density from a published roll (or set) mass per metre, with the CEMA
    ratio and its range (phase 5.3)."""
    m = mass_per_metre
    return D(KAPPA_IDLER * m, f"{note}; x {KAPPA_IDLER} (CEMA I/(m r^2))",
             KAPPA_LO * m, KAPPA_HI * m)


def coupled_carry_density(m_unloaded: float, m_loaded: float, alpha: float) -> float:
    """Inertial carry density for coupling factor alpha (Lodewijks 2002, eq. 5 read as a
    wave-speed interpolation between eqs. 3 and 4). m_unloaded = belt + carry idlers,
    m_loaded = m_unloaded + material."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return 1.0 / ((1.0 - alpha) / np.sqrt(m_unloaded) + alpha / np.sqrt(m_loaded)) ** 2


def din_peripheral_force(L, f, m_ic, m_ir, m_b, m_l, H=0.0, C=1.0, kappa=None):
    """Running peripheral force, DIN 22101 eqs. 14 and 26 with cos(delta) = 1:
    F_U = C f g L (m_Ro + m_Ru + 2 m_b + m_l) + g m_l H. The rotating idler masses are taken
    as the reduced masses divided by kappa (phase 5.4; the split only matters through the
    reduced/rotating ratio, 0.6-0.92)."""
    k = KAPPA_IDLER if kappa is None else kappa
    return C * f * G_STD * L * ((m_ic + m_ir) / k + 2 * m_b + m_l) + G_STD * m_l * H


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
    f: Datum | None = None             # DIN 22101 fictitious friction coefficient
    carry_profile: Datum | None = None  # ((distance from tail, elevation), ...), m
    # take-up tension design (phase 5.4)
    l_o: Datum | None = None           # carry idler spacing, m
    l_u: Datum | None = None           # return idler spacing, m
    p_A: Datum | None = None           # start factor: peak peripheral force at start / F_U
    F_A: Datum | None = None           # peak peripheral force at start, N (when published)
    takeup_basis: str = ""             # criterion that set T_t, as documented by the source
    basis_short: str = ""              # the same, abridged for the case table (phase 5.6)
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

    def gamma_range(self):
        """(gamma at the lowest, gamma at the highest coupling of the material class), or None
        when gamma is given directly (measured or published): no coupling bar (phase 5.3(d))."""
        if self.gamma_given is not None:
            return None
        lo, hi = self.alpha.range
        return self.gamma(lo), self.gamma(hi)

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
        if Tt is not None and n is not None:
            return "T_t and n, direct counterweight assumed", int(n) * Tt / G_STD, 1.0, int(n)
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
        rule = self.takeup_rule()[0]
        if "n = 2 assumed" in rule and "n" not in out:
            out.append("n")
        if "direct counterweight assumed" in rule and "i" not in out:
            out.append("i")
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
_SI_IDLERS = (round(cema_reduced(880.0, 7, 3.0), 2), round(cema_reduced(907.0, 7, 6.0), 2))

CASES_FULL = [
    Case("H", "Harrison 1983/85", "harrison1983", L=P(5100.0), sigma_t=[0.005],
         # Phase 5.6: strand values from the loop quantities, as in validation/harrison_case.py
         # (transit basis), so that the case table gives the T_1 of the validation section
         # (28.4 s). Before: mu_r = 79 and c_r = 1450 as if gamma = 1 (T_1 = 28.0 s).
         mu_r_given=D(2 * 79.0 / (1 + 0.97 ** 2),
                      "mean loop density 79 kg/m (consistent with Harrison's eq. 2; 39 from the "
                      "text) split so that mu_c / mu_r = gamma^2",
                      2 * 39.0 / (1 + 0.97 ** 2), 2 * 79.0 / (1 + 0.97 ** 2)),
         gamma_given=Ms(0.97, "S1->S3->S2 times, Harrison 1985b Fig. 4a", 0.90, 1.06),
         c_r_given=Ms(5100.0 * (1 + 0.97) / (2 * 5100.0 / 1450.0),
                      "loop transit 2L / 1450 m/s = 7.03 s (7 s wave period on 5.1 km) split with "
                      "gamma = 0.97: L / c_r + L / c_c = 7.03 s"),
         M_w=P(20e3, "1983 text and 1985b"), n=P(4, "Harrison 1985b Fig. 2a"),
         i=A(1.0, "direct"), V=Ms(3.7, "final speed read from Fig. 5a (1983), two-step start"),
         profile="two torque steps (wound-rotor motors, 1985b): not speed-controlled",
         F_U=D(151e3, "running T1 = 200 kN (1983, section 6) less T_t = 49 kN (20 t on 4 strands)"),
         takeup_basis="not stated; running T1 / T2 = 200 / 49 kN needs e^(mu theta) >= 4.1",
         basis_short="not stated (grip at $\\mathrm{e}^{\\mu\\theta} \\ge 4.1$)",
         notes="consistency case; plotted at gamma = 1 on the maps; xi read from the inset "
               "of Fig. 5a (0.002-0.02). Belt SR2250 in 1983, SR2400 in 1985b. Running power "
               "900 kW (1985b). Not a start-up case: stepped-torque drive. EA not set (c_r is "
               "measured): his eq. 2 implies 4.34 km/s squared x 9 kg/m of cord = 170 MN"),
    Case("S", "Song et al. 2012", "song2012", L=P(7117.0), sigma_t=[0.001, 0.999],
         m_b=P(27.0), m_ic=P(11.25), m_ir=P(10.8), m_l=D(66.7, "600 t/h at 2.5 m/s"),
         alpha=A(1.0, "coal", 0.8, 1.0), EA=P(104e6, "1 m width"),
         M_w=P(4500.0, "their tensioning weight, neglected in their solution; on 2 strands it "
                       "holds only 22 kN. Alternative: beta = 0.017, fundamental +0.14 %"),
         T_t=P(173e3, "tension at the tail take-up, both of their methods (section 3.3); "
                      "basis of beta (rule 0): 35.3 t on 2 strands, beta = 0.131"),
         n=P(2, "2:1 in their eqs. 5-6"), i=A(1.0),
         V=P(2.5), t_a=P(300.0), profile="sine (Harrison cycloid)",
         F_U=P(225e3, "steady drive force (2.268e5 N from the resistances)"),
         f=P(0.016, "speed-independent resistance coefficient, both strands (plus 0.0026 "
                    "speed-dependent); drive friction 0.3, wrap not given"),
         l_o=A(1.2, "not published: DIN 22101 Table 4 standard range", 1.0, 1.5),
         l_u=A(3.0, "not published: DIN 22101 Table 4 standard range", 2.5, 3.5),
         p_A=P(236.0 / 225.0, "peak 2.36e5 N against the steady 2.25e5 N (Fig. 5 text)"),
         takeup_basis="not stated; their running tensions (337 / 113 kN, text) need "
                      "e^(mu theta) = 2.98: one pulley with mu = 0.3 and ~210 deg wrap",
         basis_short="not stated (grip at $\\mathrm{e}^{\\mu\\theta} = 2.98$)",
         notes="head (their Fig. 1) / tail (their model); inconsistent running tensions: "
               "structure only. ST1600, 1 m, 14.8 mm; relaxation coefficient 0.14"),
    Case("G", "Gao et al. 2026", "gao2026", L=P(4500.0), sigma_t=[0.001],
         m_b=P(40.1), m_ir=P(0.0, "no idler masses given"), m_l=P(154.2),
         alpha=A(1.0, "coal-mine project, material not named: fine/wet class (the 0.3 class "
                      "needs coarse dry rock in the source)", 0.8, 1.0),
         EA=P(1.56e8, "1.3e8 N/m per width x 1.2 m"),
         M_w=P(1000.0, "head take-up; their z (number of counterweights) not given"),
         V=P(4.0), t_a=P(60.0, "base case; 60-120 s in their sweep"), profile="sine (Harrison)",
         takeup_basis="unknown: the number of counterweights z is not given, so T_t is unknown",
         basis_short="unknown",
         notes="idler spacings 1.5 / 3 m but no idler masses; drums 600 kg (bend) and 500 kg "
               "(drive), rotational; horizontal"),
    Case("LL", "Li and Li 2009", "li2009amesim", L=P(7600.0), sigma_t=[0.005],
         m_b=P(54.0), m_ic=D(32.2, "from their c = 837 m/s with full coupling"),
         m_ir=D(12.9, "carry idlers scaled by the 3 m / 1.2 m spacing", 6.45, 19.35),
         m_l=D(173.6, "2500 t/h at 4 m/s"), alpha=A(1.0, "material not specified", 0.8, 1.0),
         EA=P(182e6, "1300 kN/cm x 140 cm (catalogue ST2000 x 1.4 m: 202 MN)"),
         M_w=P(42800.0),
         n=A(2, "their Fig. 4: ~200 kN on the slack side vs 210 kN = M g / 2"), i=A(1.0),
         V=P(4.0), t_a=P(70.0, "4.05 m/s at ~70 s; motors on in 4 s steps over the first 10 s",
                         60.0, 70.0),
         profile="motor steps, no speed control",
         E=P(4.81, "drive factor e^(mu alpha); tension ratio <= 4.25 in the start"),
         carry_profile=A(((0.0, 0.0), (7600.0, -175.0)), "straight decline, -1.3 deg mean"),
         l_o=P(1.2), l_u=P(3.0),
         F_U=A(round(din_peripheral_force(7600.0, 0.020, 32.2, 12.9, 54.0, 173.6, -175.0)),
               "not published: DIN eq. 14 with f = 0.020 (standard), C = 1, -175 m",
               round(din_peripheral_force(7600.0, 0.016, 32.2, 12.9, 54.0, 173.6, -175.0)),
               round(din_peripheral_force(7600.0, 0.025, 32.2, 12.9, 54.0, 173.6, -175.0))),
         F_A=Ms(650e3, "Fig. 4: ~850 kN at the drive entry with ~200 kN at the exit; tension "
                       "ratio <= 4.25 (text)", 600e3, 700e3),
         takeup_basis="start grip (checked, not stated as the design basis): peak tension ratio "
                      "4.25 against e^(mu alpha) = 4.81 with motor steps and no speed control",
         basis_short="start grip (checked)",
         notes="175 m descent: static state only; motors in steps every 4 s (no speed control); "
               "two head drive pulleys (2 + 1 motors of 800 kW), take-up by the second; "
               "take-up settles ~12 m down"),
    Case("SM", "Suchorab-Matuszewska et al. 2025 (KGHM, variant 1)", "suchorab2025longdistance",
         L=P(3000.0, "QNK-TT route: 300 + 6 x 400 + 300 m"),
         sigma_t=[0.1],   # take-up at QNK node 3, end of the 300 m section after the head
         m_b=P(64.8, "GTP-St-4000-X-(14+10), 54.00 kg/m2 x 1.2 m; matches the QNK belt "
                     "gravity forces"),
         m_ic=reduced_idlers(3 * 9.10 / 0.83, "3 rolls of 9.10 kg (465 mm, 159 mm) every 0.83 m"),
         m_ir=reduced_idlers(2 * 12.30 / 2.5, "V return, 2 rolls of 12.30 kg (670 mm) every "
                                              "2.5 m (QNK side rolls only; V-type in the paper)"),
         m_l=D(2000 / 3.6 / 3.0, "2000 t/h at 3 m/s; matches the QNK material gravity forces"),
         alpha=A(1.0, "run-of-mine copper ore (underground, lumps)", 0.3, 1.0),
         EA=C(72 * 4000e3 * 1.2, "St 4000 x 1.2 m, modulus 72 x class (Continental, Fenner)"),
         T_t=P(140e3, "S(3), QNK-TT report"), V=P(3.0),
         E=D(float(np.exp(0.35 * np.radians(458.0))), "mu = 0.35, 458 deg wrap (QNK-TT report)"),
         F_U=P(514554.0, "QNK peripheral force, section 2-3"),
         f=P(0.0241, "QNK equivalent, C = 1.03"),
         carry_profile=P(((0.0, 0.0), (300.0, 10.47), (700.0, 24.43), (1100.0, 38.39),
                          (1500.0, 52.35), (1900.0, 66.31), (2300.0, 80.27), (2700.0, 94.23),
                          (3000.0, 130.79)), "QNK route table, sections 1-8 from the tail"),
         l_o=P(0.83, "QNK-TT report"), l_u=P(2.5, "QNK-TT report"),
         takeup_basis="not stated: S(3) = 140 kN is a QNK input; QNK checks slip with a safety "
                      "factor 1.2. Its start factor (Kr = 1.12, Pr / Pu = 3.3) belongs to a "
                      "rigid-body direct start, not to the VFD drives",
         basis_short="not stated (model input)",
         notes="head drive (paper); take-up at QNK node 3, 300 m down the return. Node 3 is "
               "the first node after the 300 m section that holds the drive, so the take-up may "
               "sit anywhere from the drive to 300 m (xi 0-0.1). No start time: QNK's 2.7 s is "
               "a rigid-body estimate; KGHM drives have VFDs"),
    Case("Lo", "Lodewijks 1996, ch. 8", "lodewijks1996", L=P(1000.0), sigma_t=[0.001],
         m_b=P(14.28), m_ic=P(13.38), m_ir=P(6.95), m_l=P(133.5),
         alpha=A(1.0, "coal", 0.8, 1.0), EA=D(4.214e6, "E = 340.9 MPa times the belt section"),
         T_t=P(42.66e3 / 2, "take-up force 42.66 kN = 2 T_t"),
         n=P(2, "tensioning weight hung on the take-up pulley loop (Fig. 8.1; force = 2 T_t)"),
         V=P(5.2, "694.44 kg/s / 133.54 kg/m (5.11 m/s effective in Table 8.7: motor slip)"),
         t_a=P(30.0, "Table 8.7"), profile="five 30 s profiles (Table 8.7)",
         E=D(float(np.exp(0.35 * np.pi)), "wrap pi, mu 0.35 (DIN 22101), eq. 8.8"),
         F_U=P(35.55e3, "DIN, loaded, C = 1.09"), f=P(0.018, "from his rolling-resistance model"),
         carry_profile=P(((0.0, 0.0), (1000.0, 0.0)), "horizontal (Fig. 8.1)"),
         l_o=P(1.5), l_u=P(2.5),
         p_A=P(1.2, "start-up factor K_s, drain-type fluid coupling (Table 8.2, Simonsen 1987)"),
         takeup_basis="start grip, sized: take-up force 2 F_a / (e^(mu theta) - 1) with F_a = "
                      "1.2 F_U and e^(mu theta) = 3 (eqs. 8.7-8.9); sag below 1.5 % checked",
         basis_short="start grip (sized)",
         notes="vertical tensioning weight (Fig. 8.1): the take-up pulley (1606 kg reduced, "
               "~1.7 t shell; eq. 8.16) is part of the 42.66 kN weight, so M = 2 T_t / g "
               "already holds it. Idlers: 90 % of the roll mass (Simonsen 1987)"),
    Case("Pa", "Pascual et al. 2005", "pascual2005",
         L=P(2561.0, "belt half length l / 2, Table 1 (Fig. 5: 1800 + 770 = 2570 m)"),
         # Phase 5.3(b): take-up in the belt loop just after the head drive (Fig. 5, "tensor
         # pulley", drawn in two positions), not at the tail as the thesis read it.
         sigma_t=[0.02],
         m_b=P(118.0), m_ic=P(67.0, "\"equivalent rollers' mass\" (eq. 2): taken as reduced"),
         m_ir=P(20.0, "\"equivalent rollers' mass\" (eq. 2): taken as reduced"),
         m_l=P(287.0, "4920 t/h at 4.75 m/s gives 287.7 kg/m"),
         alpha=A(1.0, "large-diameter rocks: Lodewijks' lowest field value applies", 0.3, 1.0),
         EA=D(0.98e9, "their v0 = 2355 m/s is the mean of the return and carry speeds (eq. 8); "
                      "with their own alpha = 0.3 for rocks (eq. 7) it gives EA = 0.98 GN "
                      "(1.29 GN with alpha = 1). Their k = mu_ef v0^2 = 1.74 GN (eq. 32) is not "
                      "EA: loop-average mass, tail pulley included, times the mean speed "
                      "squared. Lower end: catalogue, 118 kg/m is ST5000-ST6300 at 1.9-2.3 m, "
                      "72 x class x width = 0.79-0.88 GN (Fenner AS1333 masses)",
              0.79e9, 1.29e9),
         T_t=D(292e3, "f2 = 300 kN at the drive exit (Table 1) less the 51 m of return down to "
                      "the take-up (xi = 0.02, f = 0.019); 300 kN with the take-up at the drive",
               280e3, 300e3),
         n=P(2, "Fig. 5: one belt loop around the take-up pulley; type (gravity or winch) not "
                "stated, gravity assumed"),
         V=P(4.75),
         F_U=P(1.117e6, "f_ef = f1 - f2 (Table 1: 1.417 and 0.300 MN)"),
         f=D(0.019, "fitted to f1 - f2 with the Fig. 5 profile and the published masses "
                    "(belt weight cancels)"),
         carry_profile=D(((0.0, 0.0), (770.0, 770.0 * float(np.sin(np.radians(1.0)))),
                          (2561.0, 770.0 * float(np.sin(np.radians(1.0)))
                           + 1791.0 * float(np.sin(np.radians(9.0))))),
                         "Fig. 5: 770 m at 1 deg from the tail, then 9 deg to the head (1800 m "
                         "drawn; 1791 m closes L). The thesis swapped the two slopes"),
         l_o=A(1.2, "not published", 1.0, 1.5), l_u=A(3.0, "not published", 2.5, 3.5),
         takeup_basis="not stated (1988 Harrison design); T_t matches about 2 % carry sag at "
                      "the tail, at the foot of the 9 deg incline (phase 5.4)",
         basis_short="not stated (about 2\\,\\% sag)",
         notes="copper mine, northern Chile (data from a 1988 Harrison report). The 45.5 t of "
               "Table 1 (\"other masses\") is the driven (tail) pulley m3 = 45.0 t of Table 2, "
               "not a counterweight: at the tail it would put ~500 kN on the slack side "
               "against the published 300 kN. Take-up mass not published (rule 4 with n = 2 "
               "from Fig. 5: 59.6 t). xi from 0.005 to 0.05 (loop not to scale; T_1 changes "
               "1 %). No usable start time: their eq. 30 adds arctangents of accelerations "
               "(unit-dependent). Running T1 / T2 = 4.72 needs e^(mu theta) >= 4.72 (drive "
               "pulleys not described)"),
    Case("Si", "Sinaga 2008 (KPC)", "sinaga2008kpc", L=P(13100.0), sigma_t=[0.001],
         m_b=P(29.7, "ST2100, 1100 mm, 5 + 5 mm covers"),
         m_ic=C(round(cema_reduced(880.0, 7, 3.0), 2),
                "178 mm rolls, 3-roll 35 deg every 3 m; CEMA E7 WK2 interpolated to 43 in", 
                round(0.85 * cema_reduced(880.0, 7, 3.0), 2),
                round(1.15 * cema_reduced(880.0, 7, 3.0), 2)),
         m_ir=C(round(cema_reduced(907.0, 7, 6.0), 2),
                "2-roll 10 deg V return every 6 m; CEMA E7 single return roll, 43 in",
                round(0.85 * cema_reduced(907.0, 7, 6.0), 2),
                round(1.15 * cema_reduced(907.0, 7, 6.0), 2)),
         m_l=D(4200 / 3.6 / 8.5, "4200 t/h at 8.5 m/s (design 4500 t/h)"),
         alpha=A(1.0, "coal", 0.8, 1.0),
         EA=C(72 * 2100e3 * 1.1, "ST2100 x 1.1 m, modulus 72 x class"),
         M_w=P(46.9e3), T_t=P(115e3), V=P(8.5),
         t_a=P(780.0, "after commissioning; design 560 s, 720 s at dry commissioning", 560.0, 780.0),
         profile="S-curve (PLC on scoop-controlled fluid couplings)",
         f=P(0.013, "DIN fictive friction from operating power data"),
         carry_profile=A(((0.0, 0.0), (13100.0, 9.0)),
                         "net lift 9 m only; hilly, 11 downhills (Fig. 11), not digitized"),
         l_o=P(3.0), l_u=P(6.0),
         F_U=D(round(din_peripheral_force(13100.0, 0.013, *_SI_IDLERS, 29.7, 4200 / 3.6 / 8.5, 9.0)),
               "not published: DIN eq. 14 with the published f = 0.013, C = 1, net lift 9 m "
               "(3.2 MW; >3 MW demand needed the fourth 1 MW drive)",
               round(din_peripheral_force(13100.0, 0.013, *_SI_IDLERS, 29.7, 4200 / 3.6 / 8.5, 9.0)),
               round(1.05 * din_peripheral_force(13100.0, 0.013, *_SI_IDLERS, 29.7,
                                                 4200 / 3.6 / 8.5, 9.0))),
         takeup_basis="low tension in full-load braking: take-up raised from 39.1 t to 46.9 t "
                      "(115 kN), the tower maximum",
         basis_short="full-load braking",
         notes="take-up tower at the head end, by the drives; four drives on two head pulleys "
               "(wrap not given); flywheels"),
    Case("WR", "Wheatley and Rubel 2021", "wheatley2021",
         L=D(274.6, "from 274 m and 18 m lift"), sigma_t=[0.05, 0.5, 0.999],
         m_b=C(25.83, "PN1250/4 (polyester-nylon, = EP) 10 + 4 mm: Fenner Dunlop 9.1 kg/m2 "
                      "carcass + 1.4 kg/m2/mm; Continental EP1250/4 gives 21.9",
               (8.0 + 14 * 1.17) * 0.9, 25.83),
         m_ic=C(round(cema_reduced(312.0, 6, 1.2), 2), "CEMA C6, 36 in, every 1.2 m (WK2 312)"),
         m_ir=C(round(cema_reduced(313.0, 6, 3.0), 2), "CEMA C6 flat return every 3 m (WK2 313)"),
         m_l=D(1800 / 3.6 / 2.2, "1800 t/h at 2.2 m/s"), alpha=A(1.0, "iron ore fines", 0.8, 1.0),
         EA=C(13750e3 * 0.9, "EP1250/4: 13 750 N/mm (Continental) x 0.9 m; Zarzycki 2023: "
                             "13-30 % lower at low load; thesis 12 000 N/mm (Fenner 7-1, not seen)",
              0.70 * 13750e3 * 0.9, 13750e3 * 0.9),
         M_w=P(7550.0), V=P(2.2),
         F_U=D(164e3 / 2.2, "164 kW calculated demand / 2.2 m/s (motor side: upper bound)"),
         f=P(0.0233, "DIN, Belt Analyst"),
         carry_profile=A(((0.0, 0.0), (274.6, 18.0)), "straight incline; profile not published"),
         l_o=P(1.2), l_u=P(3.0),
         takeup_basis="not stated; running tensions imply e^(mu theta) = 3.0 and about 1 % sag "
                      "at the tail (phase 5.4)",
         basis_short="not stated (grip at $\\mathrm{e}^{\\mu\\theta} = 3.0$)",
         notes="take-up position and rigging unknown (dotted line on the maps); no start time; "
               "drive 150 kW nameplate, below the 164 kW demand"),
    Case("Su", "Surtees 1995 (SASOL)", "surtees1995runback", L=P(805.0),
         sigma_d=D(float(np.hypot(152.0, 45.0)) / 805.0,
                   "drives 152 m (horizontal) from the head and 45 m below it", 152.0 / 805.0,
                   float(np.hypot(152.0, 45.0)) / 805.0),
         sigma_t=[float(np.hypot(152.0, 45.0)) / 805.0 + 0.01],
         m_b=P(35.6, "both design sheets"),
         m_ic=reduced_idlers(12.0, "18 kg per set every 1.5 m"),
         m_ir=reduced_idlers(16.0 / 3.0, "16 kg every 3 m"),
         m_l=D(3500 / 3.6 / 4.4, "3500 t/h at 4.4 m/s"), alpha=A(1.0, "coal (Secunda)", 0.8, 1.0),
         EA=C(72 * 1250e3 * 1.5, "ST1250 x 1.5 m, modulus 72 x class (design 2)"),
         M_w=P(20387.0, "design 2 (ST1250, T2 = 100 kN), the belt of Fig. 10; design 1: "
                        "4795 kg (T2 = 23.6 kN)", 4795.0, 20387.0),
         n=P(2, "vertical gravity type, M = 2 T2 / g"), i=A(1.0), V=P(4.4),
         t_a=P(25.0, "start-up time required; 25.1 s estimated from breakaway"),
         profile="Voith TSS fluid couplings, 130 % start factor",
         E=D(float(np.exp(0.35 * np.radians(400.0))), "mu = 0.35, 2 x 200 deg wrap"),
         F_U=P(153065.0, "effective tension Te"), f=P(0.020, "C = 1.41"),
         carry_profile=D(((0.0, 0.0), (613.0, 0.0), (805.0, 45.0)),
                         "613 m flat before the rise (horizontal, taken along the belt), 45 m lift"),
         l_o=P(1.5), l_u=P(3.0),
         p_A=P(1.3, "Voith TSS fluid couplings (design sheet)"),
         takeup_basis="holdback on the head pulley: design 1 (23.6 kN = max of grip from the "
                      "installed power and 2 % sag at the tail) could transmit 20.1 of the "
                      "35.3 kNm runback torque; design 2 raised T2 to 100 kN (83.2 kNm)",
         basis_short="holdback",
         notes="design data; take-up right after the secondary drive pulley (Fig. 10, schematic); "
               "intermediate-drive panel only"),
    Case("NC", "Nordell and Ciozda 1984, case 1", "nordell1984", L=P(8150 * FT),
         sigma_d=P(5150 / 8150, "primary and secondary drives together"), sigma_t=[5450 / 8150],
         c_r_given=P(1450.0, "4760 ft/s, empty return (Harrison's formula; BELTFLEX agrees)"),
         gamma_given=D(1450.0 / 590.0, "1450 / 590 m/s (590: fully loaded carry)"),
         V=P(930 * FT / 60, "930 ft/min (the copy's '41 m/s' is a transcription error)"),
         mu_r_given=A(1.0, "placeholder: not given (only frequencies are used)"),
         takeup_basis="unknown: no take-up data",
         basis_short="unknown",
         notes="no take-up mass nor EA: frequencies in the limit beta -> 0 only; retarder "
               "(2720 kgf). Drive and take-up positions read from Fig. 8; event is a stop"),
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
             sigma_t=_su.sigma_t[0], note="design data; take-up after the drives (Fig. 10)")


def points():
    """(tag, beta, gamma, xi, unknown_position) for every case and take-up position."""
    out = []
    for t in MAP_TAGS:
        c = BY_TAG[t]
        for xi in c.xis:
            out.append((t + ("t" if t == "S" and xi > 0.5 else ""), c.beta, c.gamma(), xi,
                        t == UNKNOWN_POSITION))
    return out


def points_alpha():
    """As `points`, plus the coupling range of gamma: (tag, beta, gamma, gamma_lo, gamma_hi, xi,
    unknown_position). gamma_lo = gamma_hi = gamma when gamma is given directly (H)."""
    out = []
    for tag, beta, gam, xi, unknown in points():
        c = BY_TAG[tag.rstrip("t") if tag == "St" else tag]
        rg = c.gamma_range()
        lo, hi = (gam, gam) if rg is None else rg
        out.append((tag, beta, gam, lo, hi, xi, unknown))
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
