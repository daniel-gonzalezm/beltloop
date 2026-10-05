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


@pytest.mark.parametrize("tag", list(PHASE4))
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
    assert cs.SASOL["gamma"] == pytest.approx(2.56153, rel=1e-5)
    assert cs.SASOL["sigma_d"] == pytest.approx(152 / 805)
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


@pytest.mark.parametrize("tag", ["S", "G", "LL", "Lo", "Pa", "WR"])
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
        cs.BY_TAG["SM"].conveyor()
    cv = cs.BY_TAG["SM"].conveyor(EA=3e8)
    assert cv.takeup.T_t == pytest.approx(140e3)


def test_intermediate_drive_case_frequencies():
    """NC on the intermediate-drive panel: T1 = 27.5 s with beta -> 0 (section 4.19)."""
    nc = cs.BY_TAG["NC"]
    lp = Loop.from_positions(nc.sigma_d.value, nc.sigma_t[0], nc.gamma())
    Om = natural_frequencies(lp, 0.0, 1)[0]
    assert 2 * np.pi / Om * nc.L.value / nc.c_r == pytest.approx(27.5, abs=0.05)
