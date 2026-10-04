"""Phase 3.4 checks: limits used in the Harrison (1983) analysis (validation/harrison_case.py)."""
import numpy as np
import pytest

from beltloop import (J, Loop, damped_drive_root, natural_frequencies, takeup_transmission)
from beltloop.eigen import fixed_free_roots


@pytest.mark.parametrize("gamma", [1.0, 1.3, 2.0])
@pytest.mark.parametrize("xi", [0.005, 0.2])
def test_two_speed_constant_force_closed_form(gamma, xi):
    """beta -> 0, head drive: the long part (take-up -> return -> tail -> carry -> drive
    entry) is free at the take-up and fixed at the drive: tan(Om (1 - xi)) tan(gamma Om) = gamma,
    written pole-free as cos(Om lr) cos(gamma Om) - sin(Om lr) sin(gamma Om)/gamma = 0."""
    lp = Loop.from_positions(0.0, xi, gamma)
    Om = fixed_free_roots(lp.downstream[::-1], 4)
    lr = 1 - xi
    res = np.cos(Om * lr) * np.cos(gamma * Om) - np.sin(Om * lr) * np.sin(gamma * Om) / gamma
    np.testing.assert_allclose(res, 0.0, atol=1e-9)
    if gamma == 1.0:
        np.testing.assert_allclose(Om, (2 * np.arange(1, 5) - 1) * np.pi / (2 * (2 - xi)), rtol=1e-12)


@pytest.mark.parametrize("beta,Om", [(0.05, np.pi), (0.1, 0.8), (2.0, 5.0)])
def test_takeup_transmission_from_J(beta, Om):
    """Scattering of a wave on the take-up, solved directly with the matrix J:
    left W = e^{-i Om x} + r e^{i Om x}, right W = t e^{-i Om x}, (W, W')_right = J (W, W')_left."""
    Jm = J(beta, Om)
    # unknowns (r, t): J @ [1 + r, -i Om (1 - r)] = [t, -i Om t]
    M = np.array([[Jm[0, 0] + 1j * Om * Jm[0, 1], -1.0],
                  [Jm[1, 0] + 1j * Om * Jm[1, 1], 1j * Om]])
    rhs = -np.array([Jm[0, 0] - 1j * Om * Jm[0, 1], Jm[1, 0] - 1j * Om * Jm[1, 1]])
    r, t = np.linalg.solve(M, rhs)
    assert t == pytest.approx(takeup_transmission(beta, Om), rel=1e-12)
    assert abs(r) ** 2 + abs(t) ** 2 == pytest.approx(1.0, rel=1e-12)      # energy
    assert abs(takeup_transmission(1e-8, Om)) < 1e-7                         # free end
    assert abs(takeup_transmission(1e8, Om)) == pytest.approx(1.0, abs=1e-7)  # transparent


def test_damped_drive_limits():
    """Drive with slip dashpot cd. (i) cd -> infinity: prescribed-velocity root.
    (ii) beta -> 0, md = 0, head drive, gamma = 1: the long part is a rod free at the
    take-up with a dashpot at the drive; reflection coefficient rho = (cd - 1)/(cd + 1)
    real, so Re Om = pi/(2 l) and Im Om = -ln(rho)/(2 l), l = 2 - xi (exact as xi -> 0;
    the short span between drive and take-up adds an O(xi) correction)."""
    xi = 0.005
    lp = Loop.from_positions(0.0, xi, 1.0)
    Om0 = natural_frequencies(lp, 0.05, 2)
    for k in range(2):
        Om = damped_drive_root(lp, 0.05, 0.3, 1e7, Om0[k] + 1e-3j)
        assert Om.real == pytest.approx(Om0[k], rel=1e-6) and abs(Om.imag) < 1e-5
    xi = 1e-5
    lp = Loop.from_positions(0.0, xi, 1.0)
    l = 2 - xi
    for cd in (2.0, 5.0, 20.0):
        Om = damped_drive_root(lp, 1e-9, 0.0, cd, np.pi / (2 * l) + 0.01j)
        assert Om.real == pytest.approx(np.pi / (2 * l), rel=1e-4)
        assert Om.imag == pytest.approx(-np.log((cd - 1) / (cd + 1)) / (2 * l), rel=1e-4)


def test_harrison_phase34_regression():
    """Key numbers of phase 3.4 (state document): slow period with gamma at fixed loop
    transit time 2L/1450 m/s, xi = 0.005, mean density 79 kg/m."""
    L, M, rho = 5100.0, 20000.0, 79.0
    ttr = 2 * L / 1450.0
    ref = {1.0: 28.24, 1.2: 26.73, 1.5: 25.26}
    for g, T in ref.items():
        mur = 2 * rho / (1 + g * g)
        cr = L * (1 + g) / ttr
        Om = natural_frequencies(Loop.from_positions(0.0, 0.005, g), M / (mur * L), 1)[0]
        assert 2 * np.pi * L / (cr * Om) == pytest.approx(T, abs=0.01)


def test_harrison_phase37_double_loop_regression():
    """Phase 3.7: double-loop take-up (n = 4, Harrison 1985 Fig. 2a) and the measured
    gamma = 0.97 (Fig. 4a). Belt-side beta = 4M/(n^2 mu_r L) = beta/4; the slow period
    hardly changes (it is the quarter-wave mode of the long part, nearly free at the
    take-up)."""
    L, M, rho, n = 5100.0, 20000.0, 79.0, 4
    ttr = 2 * L / 1450.0
    ref = {1.0: 28.11, 0.97: 28.38}
    for g, T in ref.items():
        mur = 2 * rho / (1 + g * g)
        cr = L * (1 + g) / ttr
        Om = natural_frequencies(Loop.from_positions(0.0, 0.005, g), 4 * M / (n ** 2 * mur * L), 1)[0]
        assert 2 * np.pi * L / (cr * Om) == pytest.approx(T, abs=0.01)
