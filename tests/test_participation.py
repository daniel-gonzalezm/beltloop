"""Phase 4.4: modal participation strand by strand (beta -> 0) and its take-up mass
correction; structure of the fundamental for a head drive."""
import numpy as np
import pytest
from scipy.optimize import brentq

from beltloop import (Loop, modal_basis, natural_frequencies, strand_modes, strand_participation,
                      takeup_mass_approx, takeup_mass_modes)


def _strands(xi, gamma):
    lp = Loop.from_positions(0.0, xi, gamma)
    return lp, lp.upstream, lp.downstream[::-1]


# ---------------------------------------------------------------- beta -> 0 (strands)
@pytest.mark.parametrize("xi", [0.2, 0.5, 0.8])
def test_uniform_strand_effective_mass(xi):
    """Uniform fixed-free strand: m_eff_j = 8 l / ((2j - 1)^2 pi^2)."""
    lp, up, dn = _strands(xi, 1.0)
    j = np.arange(1, 5)
    for chain, l in ((up, xi), (dn, 2 - xi)):
        _, me = strand_participation(chain, 4)
        assert np.allclose(me, 8 * l / ((2 * j - 1) ** 2 * np.pi ** 2), rtol=1e-12)


@pytest.mark.parametrize("gamma,xi", [(2.0, 0.3), (1.5, 0.8), (2.9, 0.05), (1.2, 0.6)])
def test_strands_are_loop_modes_as_beta_vanishes(gamma, xi):
    """With beta -> 0 the loop modes and their effective masses are those of the strands
    (cases without coincident poles)."""
    lp, up, dn = _strands(xi, gamma)
    pa, ma = strand_participation(up, 6)
    pb, mb = strand_participation(dn, 6)
    P, M = np.r_[pa, pb], np.r_[ma, mb]
    o = np.argsort(P)[:6]
    mb_ = modal_basis(lp, 1e-8, 6)
    assert np.allclose(mb_.Om, P[o], rtol=1e-6)
    assert np.allclose(mb_.Gamma ** 2, M[o], atol=1e-6)


@pytest.mark.parametrize("gamma,xi", [(1.0, 0.4), (2.0, 0.3), (2.93, 0.9)])
def test_strand_sum_rule(gamma, xi):
    """The strand effective masses add up to the belt mass 1 + gamma^2 (tail ~ 1/n)."""
    lp, up, dn = _strands(xi, gamma)
    n = 1000
    s = strand_participation(up, n)[1].sum() + strand_participation(dn, n)[1].sum()
    deficit = 1 - s / lp.belt_mass
    assert 0 < deficit < 1e-3


def test_participation_consistent_with_end_masses():
    """Same frequencies as strand_modes (shared root finder)."""
    lp, up, dn = _strands(0.37, 1.8)
    for chain in (up, dn):
        assert np.array_equal(strand_participation(chain, 4)[0], strand_modes(chain, 4)[0])


# --------------------------------------------------- fundamental, head drive, gamma >= 1
GRID = [(g, x) for g in (1.0, 1.3, 1.67, 2.0, 2.5, 2.76, 3.0) for x in (1e-4, 0.05, 0.3, 0.6, 0.9, 0.999)]


def test_fundamental_is_downstream_and_bounded():
    """For gamma >= 1 and a head drive the fundamental (beta -> 0) is the downstream strand's
    (take-up -> tail -> carry -> drive): Om_B1 < pi/(2 gamma) <= pi/2 < pi/(2 xi) = Om_A1.
    Its period lies in [0.837, 1] x 4 t_B, with t_B = (1 - xi) + gamma the downstream transit
    time (units of L/c_r); 4 t_B is exact for gamma = 1. It solves tan(Om(1-xi)) tan(gamma Om)
    = gamma."""
    for gamma, xi in GRID:
        _check_fundamental(gamma, xi)


def _check_fundamental(gamma, xi):
    lp, up, dn = _strands(xi, gamma)
    a1 = strand_participation(up, 1)[0][0]
    b1 = strand_participation(dn, 1)[0][0]
    assert a1 == pytest.approx(np.pi / (2 * xi), rel=1e-12)
    assert b1 < np.pi / (2 * gamma) < a1 or (gamma == 1.0 and b1 < a1)
    assert natural_frequencies(lp, 0.0, 1)[0] == pytest.approx(b1, rel=1e-13)
    T1, tB = 2 * np.pi / b1, (1 - xi) + gamma
    assert 0.837 * 4 * tB <= T1 <= 4 * tB * (1 + 1e-12)
    if gamma == 1.0:
        assert T1 == pytest.approx(4 * tB, rel=1e-12)
    assert np.tan(b1 * (1 - xi)) * np.tan(gamma * b1) == pytest.approx(gamma, rel=1e-9)


def test_quarter_wave_shortfall_minimum():
    """Largest shortfall of the quarter-wave transit estimate: take-up at the drive,
    gamma about 2.7, T1 = 0.838 x 4 t_B (closed form: tan(Om) tan(gamma Om) = gamma)."""
    def ratio(g):
        om = brentq(lambda o: np.tan(o) * np.tan(g * o) - g, 1e-9, np.pi / (2 * g) - 1e-12)
        return 2 * np.pi / om / (4 * (1 + g))
    gs = np.linspace(1, 6, 501)
    r = np.array([ratio(g) for g in gs])
    assert r.min() == pytest.approx(0.8379, abs=2e-4)
    assert 2.6 < gs[r.argmin()] < 2.8
    assert ratio(1.0) == pytest.approx(1.0, rel=1e-9)


def test_fundamental_participation_formula():
    """m_eff,1 / (1 + gamma^2) ~ (8/pi^2) (1 - xi + gamma^2)/(1 + gamma^2): the fundamental
    carries about 8/pi^2 of the downstream strand's mass; the formula overestimates by at
    most 3.1 % (exact for gamma = 1)."""
    for gamma, xi in GRID:
        lp, up, dn = _strands(xi, gamma)
        me = strand_participation(dn, 1)[1][0]
        ap = 8 / np.pi ** 2 * (1 - xi + gamma ** 2)
        assert -1e-12 <= ap / me - 1 < 0.031
        if gamma == 1.0:
            assert ap == pytest.approx(me, rel=1e-12)


def test_second_mode_switches_strand():
    """Loop mode 2 (beta -> 0) is B2 below xi_s(gamma) and A1 above; gamma = 1: xi_s = 1/2
    (pi/(2 xi) = 3 pi/(2 (2 - xi))). The participation of loop-ordered mode 2 jumps there,
    which is why the maps follow each mode by its strand."""
    lo = _strands(0.49, 1.0); hi = _strands(0.51, 1.0)
    f = lambda s: modal_basis(s[0], 1e-9, 3).effective_mass_fraction[1]
    assert natural_frequencies(lo[0], 0.0, 2)[1] == pytest.approx(3 * np.pi / (2 * 1.51), rel=1e-12)
    assert natural_frequencies(hi[0], 0.0, 2)[1] == pytest.approx(np.pi / (2 * 0.51), rel=1e-12)
    assert f(hi) - f(lo) > 0.1          # 8 xi/(2 pi^2) ~ 0.21 versus 8 (2 - xi)/(18 pi^2) ~ 0.07


# ------------------------------------------------------------- take-up mass correction
def test_two_pole_frequencies_match():
    """takeup_mass_modes (2 x 2 eigenproblem) and takeup_mass_approx (quadratic) agree,
    including coincident poles (gamma = 1, xi = 0.5 and xi -> 1)."""
    for gamma, xi in [(1.0, 0.37), (2.0, 0.3), (2.93, 0.05), (1.5, 0.95), (1.0, 0.98),
                      (1.0, 0.5), (1.0, 1 - 1e-12)]:
        lp = Loop.from_positions(0.0, xi, gamma)
        for beta in (0.0, 0.01, 0.1, 1.0):
            assert np.allclose(takeup_mass_modes(lp, beta, 4)[0], takeup_mass_approx(lp, beta, 4),
                               rtol=1e-10)


def test_single_pole_participation_scaling():
    """Isolated fundamental (gamma = 1, take-up near the drive): Om^2 and m_eff both scale
    by 1/(1 + beta/(4 m_tilde)), m_tilde = (2 - xi)/2. For Om^2 this is exact to first order;
    for m_eff it misses the first-order coupling with the other modes of the same strand,
    about 0.02-0.03 beta (a tenth of the change itself, about 0.27-0.31 beta)."""
    for xi in (0.05, 0.2):
        lp = Loop.from_positions(0.0, xi, 1.0)
        f0 = 8 * (2 - xi) / np.pi ** 2
        for beta in (1e-3, 1e-2, 0.1):
            ex = modal_basis(lp, beta, 4).Gamma[0] ** 2
            ap = f0 / (1 + beta / (2 * (2 - xi)))
            assert abs(ap / ex - 1) < 0.035 * beta
            assert f0 / ex - 1 > 0.25 * beta


def test_two_pole_participation_accuracy():
    """Mode-1 effective-mass fraction within 3.5e-3 for beta <= 0.1, 1 <= gamma <= 3, any
    take-up position (corner gamma = 1, xi -> 1 included)."""
    worst = 0.0
    for g in (1.0, 1.5, 2.0, 2.5, 3.0):
        for xi in np.linspace(0.02, 0.98, 13):
            lp = Loop.from_positions(0.0, xi, g)
            for beta in (0.01, 0.03, 0.1):
                ex = modal_basis(lp, beta, 3).effective_mass_fraction[0]
                ap = takeup_mass_modes(lp, beta, 1)[1][0] / lp.belt_mass
                worst = max(worst, abs(ap - ex))
    assert worst < 3.5e-3


@pytest.mark.parametrize("beta", [1e-3, 1e-2, 0.1])
def test_tail_uniform_excited_mode_keeps_take_up_still(beta):
    """Uniform loop, take-up at the tail: the drive acceleration excites only the mode at
    pi/2, which keeps the take-up still and does not depend on beta; Li and Pang's root
    (Om tan Om = 2/beta) moves the take-up and has zero participation."""
    lp = Loop.from_positions(0.0, 1 - 1e-12, 1.0)
    ex = modal_basis(lp, beta, 2)
    lipang = brentq(lambda o: o * np.tan(o) - 2 / beta, 1e-6, np.pi / 2 - 1e-12)
    assert ex.Om[0] == pytest.approx(lipang, rel=1e-9)
    assert ex.Om[1] == pytest.approx(np.pi / 2, rel=1e-9)
    assert ex.effective_mass_fraction[0] < 1e-12
    assert ex.effective_mass_fraction[1] == pytest.approx(8 / np.pi ** 2, rel=1e-9)
    assert abs(ex.Y[1]) < 1e-9
    om, me = takeup_mass_modes(lp, beta, 2)
    assert me[0] < 1e-12 and me[1] / lp.belt_mass == pytest.approx(8 / np.pi ** 2, rel=1e-9)


def test_pair_sum_conserved_near_coincidence():
    """gamma = 1, xi = 0.98 (poles 4 % apart): with beta = 0.1 the take-up moves most of the
    participation of B1 to the upper mode of the pair, but the pair keeps the strand sum."""
    lp, up, dn = _strands(0.98, 1.0)
    s0 = (strand_participation(up, 1)[1][0] + strand_participation(dn, 1)[1][0]) / lp.belt_mass
    ex = modal_basis(lp, 0.1, 3).effective_mass_fraction
    assert ex[0] < 0.1 and ex[1] > 0.7
    assert ex[0] + ex[1] == pytest.approx(s0, abs=0.01)
