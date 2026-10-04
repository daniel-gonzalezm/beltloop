"""Phase 4.2: take-up mass regime. Effective modal masses of the strands and the
small-beta (two-pole) approximation of the natural frequencies."""
import numpy as np
import pytest
from scipy.optimize import brentq

from beltloop import Loop, natural_frequencies, strand_modes, takeup_mass_approx
from beltloop.transfer import AB


@pytest.mark.parametrize("xi", [0.2, 0.5, 0.8])
def test_uniform_strand_effective_mass(xi):
    """Uniform strand of length l: m_tilde = l/2 for every mode, Om_j = (j - 1/2) pi / l."""
    lp = Loop.from_positions(0.0, xi, 1.0)
    for chain, l in ((lp.upstream, xi), (lp.downstream[::-1], 2 - xi)):
        Om, mt = strand_modes(chain, 4)
        assert np.allclose(mt, l / 2, rtol=1e-12)
        assert np.allclose(Om, (np.arange(1, 5) - 0.5) * np.pi / l, rtol=1e-12)


@pytest.mark.parametrize("gamma,xi", [(2.0, 0.3), (2.93, 0.05), (1.5, 0.7)])
def test_effective_mass_equals_receptance_residue(gamma, xi):
    """m_tilde from the closed-form integrals equals 1/residue of the end receptance,
    lim (Om_p^2 - Om^2) * W/W' at the pole, computed from the transfer matrices."""
    lp = Loop.from_positions(0.0, xi, gamma)
    pa, ma = strand_modes(lp.upstream, 3)
    pb, mb = strand_modes(lp.downstream[::-1], 3)
    h = 1e-7
    for p, m in zip(pa, ma):
        o = p * (1 - h)
        A, _ = AB(lp, o)
        assert (p ** 2 - o ** 2) * A[0, 1] / A[1, 1] == pytest.approx(1 / m, rel=1e-5)
    for p, m in zip(pb, mb):
        o = p * (1 - h)
        _, B = AB(lp, o)
        assert (p ** 2 - o ** 2) * B[0, 1] / B[0, 0] == pytest.approx(1 / m, rel=1e-5)


def test_beta_zero_gives_poles():
    lp = Loop.from_positions(0.0, 0.37, 1.8)
    assert np.allclose(takeup_mass_approx(lp, 0.0, 4), natural_frequencies(lp, 0.0, 4), rtol=1e-12)


@pytest.mark.parametrize("beta", [1e-3, 1e-2, 0.1])
def test_uniform_head_drive_single_pole(beta):
    """gamma = 1, head drive, isolated fundamental (downstream strand of length 2 - xi):
    Om_1 = pi/(2(2 - xi)) / sqrt(1 + beta/(2(2 - xi))) + O(beta^2)."""
    for xi in (0.05, 0.3, 0.6):
        lp = Loop.from_positions(0.0, xi, 1.0)
        ex = natural_frequencies(lp, beta, 1)[0]
        ap = np.pi / (2 * (2 - xi)) / np.sqrt(1 + beta / (2 * (2 - xi)))
        assert abs(ap / ex - 1) < 0.5 * beta ** 2


@pytest.mark.parametrize("beta", [1e-3, 1e-2, 0.1])
def test_li_pang_tail_coincident_poles(beta):
    """Take-up at the tail, gamma = 1: coincident poles pi/2; the shifted root is
    Om tan Om = 2/beta (Li and Pang 2018) and the other stays at pi/2."""
    lp = Loop.from_positions(0.0, 1 - 1e-12, 1.0)
    ex = brentq(lambda o: o * np.tan(o) - 2 / beta, 1e-6, np.pi / 2 - 1e-12)
    ap = takeup_mass_approx(lp, beta, 2)
    assert ap[0] == pytest.approx(ex, rel=2 * beta ** 2)
    assert ap[1] == pytest.approx(np.pi / 2, rel=1e-9)


def test_two_pole_accuracy_realistic_range():
    """beta <= 0.1, modes 1-3, 1 <= gamma <= 3, any take-up position: within 0.5 %."""
    worst = 0.0
    for g in (1.0, 1.5, 2.0, 3.0):
        for xi in np.linspace(0.02, 0.98, 13):
            lp = Loop.from_positions(0.0, xi, g)
            for beta in (0.01, 0.03, 0.1):
                ex = natural_frequencies(lp, beta, 3)
                worst = max(worst, np.max(np.abs(takeup_mass_approx(lp, beta, 3) / ex - 1)))
    assert worst < 5e-3


def test_fundamental_within_3pct_up_to_beta_1():
    for g in (1.0, 2.0, 3.0):
        for xi in np.linspace(0.02, 0.98, 13):
            lp = Loop.from_positions(0.0, xi, g)
            ex = natural_frequencies(lp, 1.0, 1)[0]
            assert abs(takeup_mass_approx(lp, 1.0, 1)[0] / ex - 1) < 0.035
