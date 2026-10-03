"""Natural frequencies: closed forms, limits, independent models, root isolation.
Ports checks 1-6, 9 and 10 of verify_phase2.py and adds checks of the root isolation."""
import numpy as np
import pytest

from beltloop.eigen import (fixed_free_roots, natural_frequencies, natural_frequencies_torque,
                            poles, rayleigh_takeup_bound)
from beltloop.loop import Loop, Segment
from beltloop.transfer import AB, char_fun, char_fun_torque
from reference import dense_scan_roots, fe_frequencies, fe_torque_frequencies

CONFIGS = [(0.0, 0.1, 1.0, 0.1), (0.0, 0.24, 2.0, 0.13), (0.7, 0.75, 2.5, 0.2),
           (0.7, 0.3, 1.8, 3.0), (1.0, 0.4, 2.0, 0.5), (1.4, 0.5, 2.2, 0.05)]


# --------------------------------------------------------------------- geometry
def test_loop_builder_table1():
    """Segment chains reproduce Table 1 of the Model section."""
    g = 2.0
    lp = Loop.from_positions(0.0, 0.3, g)               # head drive
    assert [(round(s.length, 12), s.g) for s in lp.upstream] == [(0.3, 1.0)]
    assert [(round(s.length, 12), s.g) for s in lp.downstream] == [(0.7, 1.0), (1.0, g)]
    lp = Loop.from_positions(0.2, 0.6, g)               # return drive, take-up downstream
    assert [(round(s.length, 12), s.g) for s in lp.upstream] == [(0.4, 1.0)]
    assert [(round(s.length, 12), s.g) for s in lp.downstream] == [(0.4, 1.0), (1.0, g), (0.2, 1.0)]
    lp = Loop.from_positions(0.6, 0.2, g)               # return drive, take-up upstream
    assert [(round(s.length, 12), s.g) for s in lp.upstream] == [(0.4, 1.0), (1.0, g), (0.2, 1.0)]
    assert [(round(s.length, 12), s.g) for s in lp.downstream] == [(0.4, 1.0)]
    lp = Loop.from_positions(1.0, 0.3, g)               # tail drive
    assert [(round(s.length, 12), s.g) for s in lp.upstream] == [(1.0, g), (0.3, 1.0)]
    assert [(round(s.length, 12), s.g) for s in lp.downstream] == [(0.7, 1.0)]
    assert abs(lp.length - 2) < 1e-14 and abs(lp.belt_mass - (1 + g * g)) < 1e-14


def test_char_fun_entire_at_zero():
    lp = Loop.from_positions(0.0, 0.3, 2.0)
    assert char_fun(lp, 0.0, 0.7) == pytest.approx(-4.0)


# --------------------------------------------------------------------- closed forms
@pytest.mark.parametrize("xi,beta", [(0.24, 0.13), (0.5, 1.0), (0.05, 0.1)])
def test_uniform_closed_form(xi, beta):
    """gamma = 1: tan(Om xi) + tan(Om (2 - xi)) = 4/(beta Om)."""
    lp = Loop.from_positions(0.0, xi, 1.0)
    r = natural_frequencies(lp, beta, 12)
    ref = dense_scan_roots(lambda o: beta * o * np.sin(2 * o)
                           - 4 * np.cos(o * xi) * np.cos(o * (2 - xi)), r[-1] + 1e-3)
    np.testing.assert_allclose(r, ref[:12], atol=1e-10)


@pytest.mark.parametrize("xi,g,beta", [(0.1, 2.0, 0.1), (0.6, 1.5, 2.0), (0.9, 3.0, 0.05)])
def test_head_drive_explicit(xi, g, beta):
    """Eq. (chareq_head)."""
    def Dh(o):
        a, b = xi, 1 - xi
        return (beta * o * (g * np.sin(o) * np.cos(g * o) + np.cos(o) * np.sin(g * o))
                - 4 * np.cos(o * a) * (g * np.cos(o * b) * np.cos(g * o) - np.sin(o * b) * np.sin(g * o)))
    lp = Loop.from_positions(0.0, xi, g)
    r = natural_frequencies(lp, beta, 10)
    np.testing.assert_allclose(r, dense_scan_roots(Dh, r[-1] + 1e-3)[:10], atol=1e-10)


# --------------------------------------------------------------------- independent models
@pytest.mark.parametrize("sd,st,g,beta", CONFIGS)
def test_fe_agreement(sd, st, g, beta):
    lp = Loop.from_positions(sd, st, g)
    r = natural_frequencies(lp, beta, 5)
    fe = fe_frequencies(lp, beta)[:5]
    assert np.max(np.abs(fe / r - 1)) < 1e-4


@pytest.mark.parametrize("sd,st,g,beta", CONFIGS)
def test_reversal_invariance(sd, st, g, beta):
    lp = Loop.from_positions(sd, st, g)
    np.testing.assert_allclose(natural_frequencies(lp, beta, 10),
                               natural_frequencies(lp.reversed(), beta, 10), atol=1e-11)


# --------------------------------------------------------------------- root isolation
@pytest.mark.parametrize("sd,st,g,beta", CONFIGS)
def test_root_isolation_vs_dense_scan(sd, st, g, beta):
    """Interlacing method finds the same roots as a fine scan, and D vanishes at them."""
    lp = Loop.from_positions(sd, st, g)
    r = natural_frequencies(lp, beta, 25)
    ref = dense_scan_roots(lambda o: char_fun(lp, o, beta), r[-1] + 1e-4, n=200000)
    np.testing.assert_allclose(r, ref[:25], atol=1e-10)


@pytest.mark.parametrize("sd,st,g", [(0.0, 0.3, 2.0), (0.7, 0.75, 2.5), (1.4, 0.5, 2.2)])
def test_fixed_free_roots(sd, st, g):
    """Pruefer bracketing gives the zeros of A22 and B11, all of them, in order."""
    lp = Loop.from_positions(sd, st, g)
    pa = fixed_free_roots(lp.upstream, 15)
    pb = fixed_free_roots(lp.downstream[::-1], 15)
    assert np.all(np.diff(pa) > 0) and np.all(np.diff(pb) > 0)
    np.testing.assert_allclose([AB(lp, o)[0][1, 1] for o in pa], 0, atol=1e-9)
    np.testing.assert_allclose([AB(lp, o)[1][0, 0] for o in pb], 0, atol=1e-9)
    refa = dense_scan_roots(lambda o: AB(lp, o)[0][1, 1], pa[-1] + 1e-4, n=100000)
    refb = dense_scan_roots(lambda o: AB(lp, o)[1][0, 0], pb[-1] + 1e-4, n=100000)
    np.testing.assert_allclose(pa, refa, atol=1e-10)
    np.testing.assert_allclose(pb, refb, atol=1e-10)


def test_close_roots_small_beta():
    """gamma = 1, xi = 0.5: A22 and B11 share the zero Om = pi, which is a root; with
    beta = 1e-7 a second root lies within O(beta) of it. A coarse scan misses the pair;
    the interlacing method does not."""
    lp = Loop.from_positions(0.0, 0.5, 1.0)
    beta = 1e-7
    r = natural_frequencies(lp, beta, 6)
    p = poles(lp, 6)
    assert np.any(np.abs(r - np.pi) < 1e-12)              # the exact coincident root
    assert np.sum(np.abs(r - np.pi) < 1e-5) == 2          # and its close neighbour
    assert np.all(r[1:] >= p[:-1] - 1e-12) and np.all(r <= p + 1e-12)   # interlacing
    coarse = dense_scan_roots(lambda o: char_fun(lp, o, beta), r[-1] + 1e-3, n=2000)
    assert len(coarse) < 6


def test_general_chain_constructor():
    """Arbitrary chains (e.g. a loading point) are accepted and agree with FE."""
    up = (Segment(0.2, 1.0), Segment(0.3, 1.6, strand="carry"))
    dn = (Segment(0.5, 2.1, strand="carry"), Segment(1.0, 1.0))
    lp = Loop(up, dn)
    r = natural_frequencies(lp, 0.3, 5)
    assert np.max(np.abs(fe_frequencies(lp, 0.3)[:5] / r - 1)) < 1e-4


# --------------------------------------------------------------------- limits in beta
def test_beta_limits():
    lp = Loop.from_positions(0.0, 0.3, 2.0)
    r0 = natural_frequencies(lp, 1e-9, 8)
    np.testing.assert_allclose(r0, poles(lp, 8), atol=1e-6)          # A22 B11 = 0
    np.testing.assert_allclose(natural_frequencies(lp, 0.0, 8), poles(lp, 8), atol=0)
    big = 1e8
    rinf = natural_frequencies(lp, big, 8)
    assert abs(rinf[0] / np.sqrt(2 / big) - 1) < 1e-3                 # take-up mode
    ref = dense_scan_roots(lambda o: (AB(lp, o)[1] @ AB(lp, o)[0])[0, 1], rinf[-1] + 1e-3)
    np.testing.assert_allclose(rinf[1:], ref[:7], atol=1e-6)          # (BA)12 = 0


def test_light_takeup_uniform_quarter_wave():
    lp = Loop.from_positions(0.0, 0.3, 1.0)
    assert abs(natural_frequencies(lp, 1e-10, 1)[0] - np.pi / (2 * 1.7)) < 1e-7


def test_rayleigh_bound():
    lp = Loop.from_positions(0.0, 0.3, 2.0)
    beta = 10.0
    r1 = natural_frequencies(lp, beta, 1)[0]
    rb = rayleigh_takeup_bound(lp, beta)
    assert rb >= r1 and rb / r1 - 1 < 0.02


def test_li_pang_tail_factorisation():
    """Take-up at the tail, gamma = 1: {Om tan Om = 2/beta} U {cos Om = 0} (Li & Pang 2018)."""
    beta = 0.3
    lp = Loop((Segment(1.0),), (Segment(1.0),))
    r = natural_frequencies(lp, beta, 6)
    fam1 = dense_scan_roots(lambda o: o * np.sin(o) - (2 / beta) * np.cos(o), 10)
    fam2 = (2 * np.arange(1, 5) - 1) * np.pi / 2
    np.testing.assert_allclose(r, np.sort(np.r_[fam1, fam2])[:6], atol=1e-10)


# --------------------------------------------------------------------- torque-controlled drive
def test_torque_variant():
    lp = Loop.from_positions(0.0, 0.1, 2.0)
    r = natural_frequencies_torque(lp, 0.2, 1e8, 6)
    np.testing.assert_allclose(r[:4], natural_frequencies(lp, 0.2, 4), atol=1e-5)
    r = natural_frequencies_torque(lp, 0.2, 0.3, 6)[:4]
    fe = fe_torque_frequencies(lp, 0.2, 0.3)[1:5]
    assert np.max(np.abs(fe / r - 1)) < 1e-4


# --------------------------------------------------------------------- Harrison regression
def test_harrison_regression():
    """Slow-mode periods of the state document, section 4.7 (single wave speed)."""
    c, L, M = 1450.0, 5100.0, 20000.0
    table = {39.0: [28.2, 27.8, 27.1, 25.7, 14.8], 79.0: [28.0, 27.6, 26.9, 25.5, 14.4]}
    for rho, ref in table.items():
        beta = M / (rho * L)
        per = [2 * np.pi * L / (c * natural_frequencies(Loop.from_positions(0.0, xi, 1.0), beta, 1)[0])
               for xi in (0.02, 0.05, 0.10, 0.20, 1 - 1e-9)]
        np.testing.assert_allclose(per, ref, atol=0.051)
    lp = Loop.from_positions(0.0, 0.03, 1.0)
    for rho, ref in {39.0: [14.8, 19.7], 79.0: [14.4, 19.4]}.items():
        pt = [2 * np.pi * L / (c * natural_frequencies_torque(lp, M / (rho * L), md, 3)[0])
              for md in (1e-6, 1.0)]
        np.testing.assert_allclose(pt, ref, atol=0.051)


def test_nordell_ciozda_arrival_times():
    """Kinematic check (state document, section 4.7): drive 5150 ft from the head on the
    return strand, take-up 300 ft downstream, 8150 ft conveyor, 1450 and 590 m/s."""
    ft = 0.3048
    Lc, xd = 8150 * ft, 5150 * ft
    tB = xd / 1450 + xd / 590
    tC = xd / 1450 + Lc / 590 + 2700 * ft / 1450
    assert abs(tB - 3.72) < 0.05 and abs(tC - 5.86) < 0.05
