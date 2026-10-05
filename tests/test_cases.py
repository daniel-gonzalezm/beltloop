"""Phase 5.2: case data structure (maps/cases.py) and its conventions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "maps"))
import cases as cs  # noqa: E402
from beltloop import GravityTakeUp, Loop, natural_frequencies  # noqa: E402

# (beta, gamma) on the maps at the close of phase 4 (maps/cases.py of step 4.8).
PHASE4 = {"H": (0.012410, 0.97), "S": (0.016727, 1.66587), "G": (0.0055417, 2.20122),
          "LL": (0.084179, 1.97064), "SM": (0.127578, 1.94736), "Lo": (0.204904, 2.75555),
          "Pa": (0.128743, 1.84940), "Si": (0.026559, 2.28853), "WR": (0.884353, 2.92613)}


# Cases revised against their sources in step 5.3(a): (beta, gamma) after the revision.
# SM, Si, Su: idler masses converted with the CEMA ratio; Si: CEMA E7 idlers; Su: design 2.
PHASE53A = {"SM": (0.130963, 1.94987), "Si": (0.025363, 2.24438)}
# Step 5.3(b): Pa with the take-up in the head loop (T_t from f2, n = 2 from Fig. 5).
PHASE53B = {"Pa": (0.168497, 1.84940)}
# Step 5.3(c): S on the published take-up tension (173 kN) instead of the 4500 kg mass.
PHASE53C = {"S": (0.131150, 1.66627)}
REVISED = {**PHASE53A, **PHASE53B, **PHASE53C}


@pytest.mark.parametrize("tag", [t for t in PHASE4 if t not in REVISED])
def test_regression_phase4(tag):
    """The rewrite reproduces the phase-4 map positions (differences come only from the
    rounding of mu_r, mu_c and of M g / T_t in the old table)."""
    c = cs.BY_TAG[tag]
    b0, g0 = PHASE4[tag]
    assert c.beta == pytest.approx(b0, rel=1e-3)
    assert c.gamma() == pytest.approx(g0, rel=1e-3)


def test_legacy_api():
    pts = cs.points()
    assert [p[0] for p in pts] == ["H", "S", "St", "G", "LL", "SM", "Lo", "Pa", "Si", "WR", "WR", "WR"]
    for tag, name, L, mr, mc, M, n, xis, note in cs.CASES:
        c = cs.BY_TAG[tag]
        assert 4 * M / (n * n * mr * L) == pytest.approx(c.beta, rel=1e-12)
        assert np.sqrt(mc / mr) == pytest.approx(c.gamma(), rel=1e-12)
    assert cs.SASOL["gamma"] == pytest.approx(2.58384, rel=1e-5)
    assert cs.SASOL["sigma_d"] == pytest.approx(np.hypot(152, 45) / 805)
    assert cs.NORDELL_CIOZDA["gamma"] == pytest.approx(1450 / 590)
    assert cs.NORDELL_CIOZDA["sigma_t"] == pytest.approx(5450 / 8150)


@pytest.mark.parametrize("n,i,Mc", [(2, 1.0, 0.0), (4, 1.0, 0.0), (2, 0.5, 0.0), (4, 0.25, 0.0),
                                    (6, 2.0, 0.0)])
def test_rigging_free_beta(n, i, Mc):
    """With T_t and M_w known, beta = 4 T_t^2 / (M_w g^2 mu_r L) for any ideal rigging."""
    Mw, mu, L = 30e3, 50.0, 4000.0
    tu = GravityTakeUp(M_w=Mw, M_c=Mc, i=i, strands=n)
    beta_exact = tu.M_belt / (mu * L)
    beta_free = 4 * tu.T_t ** 2 / (Mw * cs.G_STD ** 2 * mu * L)
    assert beta_free == pytest.approx(beta_exact, rel=1e-12)
    c = cs.Case("X", "x", "x", L=cs.P(L), sigma_t=[0.1], mu_r_given=cs.P(mu),
                gamma_given=cs.P(2.0), M_w=cs.P(Mw), T_t=cs.P(tu.T_t))
    assert c.takeup_rule()[0] == "T_t and M_w"
    assert c.beta == pytest.approx(beta_exact, rel=1e-12)
    assert c.takeup().M_belt / (mu * L) == pytest.approx(beta_exact, rel=1e-12)
    assert c.takeup().T_t == pytest.approx(tu.T_t, rel=1e-12)


def test_assumed_rigging_is_upper_bound():
    """Rules 3 and 4 (n = 2 direct) give the largest beta among direct counterweights."""
    mu, L, Mw, Tt = 50.0, 4000.0, 30e3, 150e3
    for n in (4, 6, 8):
        assert 4 * Mw / (n * n * mu * L) < 4 * Mw / (4 * mu * L)          # M_w only
        Mn = n * Tt / cs.G_STD                                            # T_t only
        assert 4 * Mn / (n * n * mu * L) < 4 * (2 * Tt / cs.G_STD) / (4 * mu * L)
    c = cs.Case("X", "x", "x", L=cs.P(L), sigma_t=[0.1], mu_r_given=cs.P(mu),
                gamma_given=cs.P(2.0), T_t=cs.P(Tt))
    assert c.beta == pytest.approx(2 * Tt / (cs.G_STD * mu * L))
    assert "n" in c.assumed_inputs()


def test_coupling_limits():
    mu_u, mu_l = 80.0, 280.0
    assert cs.coupled_carry_density(mu_u, mu_l, 1.0) == pytest.approx(mu_l)
    assert cs.coupled_carry_density(mu_u, mu_l, 0.0) == pytest.approx(mu_u)
    a = np.linspace(0, 1, 11)
    m = [cs.coupled_carry_density(mu_u, mu_l, x) for x in a]
    assert np.all(np.diff(m) > 0)
    # wave-speed interpolation: c(alpha) = c_U - alpha (c_U - c_L), whatever EA
    for EA in (1e7, 2e8):
        cU, cL = np.sqrt(EA / mu_u), np.sqrt(EA / mu_l)
        c = np.sqrt(EA / cs.coupled_carry_density(mu_u, mu_l, 0.3))
        assert c == pytest.approx(cU - 0.3 * (cU - cL), rel=1e-12)
    with pytest.raises(ValueError):
        cs.coupled_carry_density(mu_u, mu_l, 1.2)


def test_provenance():
    for c in cs.CASES_FULL:
        for k, v in c.__dict__.items():
            if isinstance(v, cs.Datum):
                assert v.kind in cs.KINDS
                lo, hi = v.range
                assert lo <= v.value <= hi
        if c.gamma_given is None:
            assert c.alpha.lo is not None, f"{c.tag}: coupling range missing"
    with pytest.raises(ValueError):
        cs.Datum(1.0, "guessed")
    with pytest.raises(ValueError):
        cs.Datum(3.0, "assumed", lo=0.0, hi=1.0)


@pytest.mark.parametrize("tag", ["S", "G", "LL", "SM", "Lo", "Pa", "Si", "WR", "Su"])
def test_conveyor_matches_case(tag):
    """The dimensional Conveyor built from a case has the case's beta, gamma, xi."""
    c = cs.BY_TAG[tag]
    cv = c.conveyor()
    assert cv.beta == pytest.approx(c.beta, rel=1e-12)
    assert cv.gamma == pytest.approx(c.gamma(), rel=1e-12)
    assert cv.xi == pytest.approx(c.xis[0], rel=1e-12)
    assert cv.takeup.T_t > 0


def test_conveyor_needs_EA():
    with pytest.raises(ValueError):
        cs.BY_TAG["H"].conveyor()
    cv = cs.BY_TAG["SM"].conveyor(EA=3e8)
    assert cv.takeup.T_t == pytest.approx(140e3)


def test_intermediate_drive_case_frequencies():
    """NC on the intermediate-drive panel: T1 = 27.5 s with beta -> 0 (section 4.19)."""
    nc = cs.BY_TAG["NC"]
    lp = Loop.from_positions(nc.sigma_d.value, nc.sigma_t[0], nc.gamma())
    Om = natural_frequencies(lp, 0.0, 1)[0]
    assert 2 * np.pi / Om * nc.L.value / nc.c_r == pytest.approx(27.5, abs=0.05)


# ----------------------------------------------------------------- step 5.3(a)
@pytest.mark.parametrize("tag", list(REVISED))
def test_revised_53(tag):
    c = cs.BY_TAG[tag]
    b0, g0 = REVISED[tag]
    assert c.beta == pytest.approx(b0, rel=1e-4)
    assert c.gamma() == pytest.approx(g0, rel=1e-5)


def test_cema_reduced_reproduces_thesis_values():
    """WK2 = 312 and 313 lb in2 (CEMA C6, 36 in) give the thesis' 13.10 and 5.26 kg/m."""
    assert cs.cema_reduced(312.0, 6, 1.2) == pytest.approx(13.10, abs=5e-3)
    assert cs.cema_reduced(313.0, 6, 3.0) == pytest.approx(5.26, abs=5e-3)


def test_reduced_idlers_range():
    d = cs.reduced_idlers(10.0, "x")
    assert d.kind == "derived" and d.value == pytest.approx(8.0)
    assert d.range == pytest.approx((6.0, 9.2))


def test_sm_qnk_consistency():
    """QNK-TT variant 1: belt and material gravity forces reproduce m_b and m_l, the route
    table ends at L and the tension balance across the drive section closes on S(3)."""
    c = cs.BY_TAG["SM"]
    g = 9.81
    assert c.m_b.value * g * 13.96 == pytest.approx(8874, rel=1e-3)
    assert c.m_l.value * g * 13.96 == pytest.approx(25360, rel=1e-3)
    prof = c.carry_profile.value
    assert prof[-1][0] == pytest.approx(c.L.value)
    assert prof[-1][1] == pytest.approx(10.47 + 6 * 13.96 + 36.56, abs=0.02)
    assert 671932 + 5863 - 23241 - c.F_U.value == pytest.approx(c.T_t.value, abs=1)
    assert c.conveyor().takeup.T_t == pytest.approx(140e3)


def test_su_design2():
    """Design 2 (ST1250, Fig. 10): M = 2 T2 / g with T2 = 100 kN, two strands."""
    c = cs.BY_TAG["Su"]
    assert c.M_w.value * cs.G_STD / 2 == pytest.approx(100e3, rel=1e-3)
    assert c.takeup_rule()[0] == "rigging"
    assert c.takeup().T_t == pytest.approx(100e3, rel=1e-3)
    assert c.carry_profile.value[-1] == (805.0, 45.0)


def test_operation_fields_53a():
    for tag in ("SM", "Si", "WR", "Su"):
        c = cs.BY_TAG[tag]
        assert c.EA is not None and c.V is not None and c.f is not None
        assert c.carry_profile is not None
    assert cs.BY_TAG["Si"].t_a.range == (560.0, 780.0)
    assert cs.BY_TAG["Su"].t_a.value == 25.0


# ----------------------------------------------------------------- step 5.3(b): Pascual
def _pa_resistances(c, f):
    g = cs.G_STD
    return dict(r_c=f * g * (c.m_b.value + c.m_ic.value + c.m_l.value),
                r_r=f * g * (c.m_b.value + c.m_ir.value))


def test_pa_published_masses():
    """Table 2 and eq. 19: mu_ef = (m3 + m4 + m5 + m6) / l = 313.8 kg/m with the driven
    (tail) pulley m3 = 45.0 t; the thesis' 305 kg/m leaves m3 out. k = mu_ef v0^2 = 1.74 GN."""
    c = cs.BY_TAG["Pa"]
    L, l = c.L.value, 2 * c.L.value
    m4 = L * (c.m_ic.value + c.m_ir.value)
    m5 = c.m_b.value * l
    m6 = c.m_l.value * L
    assert (m4, m5, m6) == pytest.approx((222.8e3, 604.4e3, 735.0e3), rel=2e-4)
    assert (45.0e3 + m4 + m5 + m6) / l == pytest.approx(313.8, abs=0.05)
    assert (m4 + m5 + m6) / l == pytest.approx(305.0, abs=0.05)
    assert 313.8 * 2355.0 ** 2 == pytest.approx(1.74e9, rel=1e-3)
    assert 4920e3 / 3600 / c.V.value == pytest.approx(c.m_l.value, abs=1.0)


def test_pa_EA_from_mean_wave_speed():
    """Their v0 = (v_r + v_c) / 2 (eqs. 6-8, q = q0) gives EA = 0.98 GN with alpha = 0.3 and
    1.29 GN with alpha = 1: the ends of the range; 1.74 GN is outside it."""
    c = cs.BY_TAG["Pa"]
    mb, mic, mir, ml = c.m_b.value, c.m_ic.value, c.m_ir.value, c.m_l.value

    def v0(EA, a):
        vr, vu, vl = (np.sqrt(EA / m) for m in (mb + mir, mb + mic, mb + mic + ml))
        return 0.5 * (vr + vu - a * (vu - vl))
    assert v0(c.EA.value, 0.3) == pytest.approx(2355.0, rel=2e-3)
    assert v0(c.EA.hi, 1.0) == pytest.approx(2355.0, rel=2e-3)
    assert not c.EA.lo <= 1.74e9 <= c.EA.hi


def test_pa_static_tensions():
    """With the take-up in the head loop and f = 0.019 the running tensions at the drive are
    the published f2 = 300 kN and f1 = 1417 kN; 45.5 t on two strands at the tail would give
    about 500 kN on the slack side."""
    c = cs.BY_TAG["Pa"]
    L = c.L.value
    res = _pa_resistances(c, c.f.value)
    cv = c.conveyor(**res)
    T = cv.running_tension(np.array([0.0, 2 * L - 1e-6]))
    assert T[0] == pytest.approx(300e3, rel=0.01)
    assert T[1] == pytest.approx(1417e3, rel=0.01)
    assert T[1] - T[0] == pytest.approx(c.F_U.value, rel=0.01)
    assert c.carry_profile.value[-1][1] == pytest.approx(293.6, abs=0.1)
    tail = cs.Case("X", "x", "x", L=c.L, sigma_t=[0.999], m_b=c.m_b, m_ic=c.m_ic, m_ir=c.m_ir,
                   m_l=c.m_l, M_w=cs.P(45.5e3), n=cs.P(2), EA=c.EA,
                   carry_profile=c.carry_profile)
    T_tail = tail.conveyor(**res).running_tension(np.array([0.0]))[0]
    assert T_tail > 480e3


def test_pa_takeup_rule():
    """T_t and n known, no mass: direct counterweight on the n strands (M_w = n T_t / g)."""
    c = cs.BY_TAG["Pa"]
    rule, Mw, i, n = c.takeup_rule()
    assert rule == "T_t and n, direct counterweight assumed"
    assert (n, i) == (2, 1.0)
    assert Mw == pytest.approx(59.6e3, rel=2e-3)
    assert c.takeup().T_t == pytest.approx(c.T_t.value, rel=1e-12)
    assert c.beta == pytest.approx(4 * c.T_t.value / (2 * cs.G_STD * c.mu_r * c.L.value))
    assert "i" in c.assumed_inputs() and "n" not in c.assumed_inputs()


@pytest.mark.parametrize("alpha,shift", [(1.0, 0.0119), (0.8, 0.0139), (0.3, 0.0178)])
def test_pa_takeup_mass_shift(alpha, shift):
    """Lengthening of the fundamental by the take-up mass, exact roots (step 5.3(b))."""
    c = cs.BY_TAG["Pa"]
    lp = Loop.from_positions(0.0, c.sigma_t[0], c.gamma(alpha))
    w0 = natural_frequencies(lp, 0.0, 1)[0]
    w = natural_frequencies(lp, c.beta, 1)[0]
    assert w0 / w - 1 == pytest.approx(shift, abs=1e-4)


# ----------------------------------------------------------------- step 5.3(c): literature
def test_lodewijks_ch8_data():
    """Lodewijks (1996, section 8.2): speed from capacity, DIN drive force, Euler factor,
    idlers at 90 % of the roll mass, tensioning force 2 T_t on a two-strand loop."""
    c = cs.BY_TAG["Lo"]
    assert c.V.value == pytest.approx(694.44 / 133.54, abs=0.01)
    assert c.F_U.value == pytest.approx(31.32e3 + 4.23e3)
    assert c.E.value == pytest.approx(3.00, abs=0.01)
    assert c.m_ic.value == pytest.approx(0.9 * 3 * 7.43 / 1.5, abs=0.01)
    assert c.m_ir.value == pytest.approx(0.9 * 19.3 / 2.5, abs=0.01)
    assert c.takeup().T_t * 2 == pytest.approx(42.66e3)
    assert c.takeup_rule()[0] == "T_t and n, direct counterweight assumed"
    assert c.beta == pytest.approx(PHASE4["Lo"][0], rel=1e-3)


def test_li2009_wave_speed():
    """Li and Li (2009): their c = sqrt(E / m) = 837 m/s fixes the carry idler mass."""
    c = cs.BY_TAG["LL"]
    m = c.m_b.value + c.m_ic.value + c.m_l.value
    assert np.sqrt(c.EA.value / m) == pytest.approx(837.0, abs=1.0)
    assert c.m_l.value == pytest.approx(2500 / 3.6 / 4.0, abs=0.1)


def test_song_tension_inconsistency():
    """Song et al. (2012): 173 kN at the tail take-up against 4500 kg on two strands. beta
    follows the tension (rule 0, phase 5.3(c)); the published mass gives 0.017."""
    c = cs.BY_TAG["S"]
    assert c.takeup_rule()[0] == "T_t and n, direct counterweight assumed"
    assert c.takeup().T_t == pytest.approx(173e3)
    assert c.beta == pytest.approx(0.1311, abs=1e-3)
    alt = 4 * c.M_w.value / (4 * c.mu_r * c.L.value)
    assert alt == pytest.approx(PHASE4["S"][0], rel=1e-3)
    assert c.m_l.value == pytest.approx(600 / 3.6 / 2.5, abs=0.05)


def test_tension_takes_precedence():
    """Rule 0: with T_t and n given, beta follows the tension even if a mass is published;
    without n, T_t and M_w together still use the rigging-free rule 2 (Si)."""
    mu, L = 50.0, 4000.0
    c = cs.Case("X", "x", "x", L=cs.P(L), sigma_t=[0.1], mu_r_given=cs.P(mu),
                gamma_given=cs.P(2.0), M_w=cs.P(1000.0), T_t=cs.P(100e3), n=cs.P(2))
    assert c.beta == pytest.approx(2 * 100e3 / (cs.G_STD * mu * L))
    assert cs.BY_TAG["Si"].takeup_rule()[0] == "T_t and M_w"


def test_operation_fields_53c():
    for tag in ("S", "G", "LL", "Lo", "NC", "H"):
        assert cs.BY_TAG[tag].V is not None
    assert cs.BY_TAG["NC"].V.value == pytest.approx(4.72, abs=0.01)
    assert cs.BY_TAG["G"].alpha.range == (0.8, 1.0)
    assert cs.BY_TAG["H"].V.kind == "measured"
