"""Phase 4.5: universal fixed-free strand curves and start-up metrics of the loop."""
import numpy as np
import pytest

from beltloop import (PEAK_FACTOR, Loop, StartProfile, fast_start_limit, modal_basis,
                      natural_frequencies, startup_metrics, startup_response, strand_curves,
                      strand_travel_estimate)


def _metrics(xi, gamma, r, beta=1e-3, zeta_hat=0.0, kind="sine"):
    lp = Loop.from_positions(0.0, xi, gamma)
    T1 = 2 * np.pi / natural_frequencies(lp, beta, 1)[0]
    return lp, startup_metrics(lp, beta, StartProfile(kind, r * T1, "none"), zeta_hat)


@pytest.mark.parametrize("kind", ["sine", "triangular", "parabolic"])
def test_fast_start_is_the_wave_law(kind):
    """While the start is shorter than the round trip (tau_a <= T_s / 2) the fixed end carries
    exactly T = Z V_inf: D = 4 r / PEAK_FACTOR."""
    r = np.array([0.1, 0.25, 0.5])
    De, _ = strand_curves(r, kind=kind, n_modes=80)
    assert np.allclose(De, fast_start_limit(r, kind), rtol=2e-3)
    assert PEAK_FACTOR[kind] > 0


def test_slow_start_is_quasi_static():
    De, Df = strand_curves([50.0])
    assert abs(De[0] - 1) < 0.01 and abs(Df[0] - 1) < 0.015


def test_damping_lowers_the_overshoot_only():
    D0, _ = strand_curves([1.0, 50.0])
    D5, _ = strand_curves([1.0, 50.0], zeta1=0.05)
    assert D5[0] < D0[0] - 0.05
    assert abs(D5[1] - D0[1]) < 0.01


@pytest.mark.parametrize("xi,gamma", [(0.05, 2.0), (0.2, 3.0), (0.5, 1.5), (0.8, 2.0)])
def test_exit_follows_the_universal_curve(xi, gamma):
    """Strand A (drive exit -> take-up) is a uniform return segment: the minimum at the drive
    exit follows the strand curve in r_A = tau_a / T_A1 (T_A1 = 4 xi)."""
    lp = Loop.from_positions(0.0, xi, gamma)
    b = modal_basis(lp, 1e-3, 60)
    T1 = 2 * np.pi / b.Om[0]
    for rA in (0.5, 1.0, 2.0, 5.0):
        ta = rA * 4 * xi
        tau = np.linspace(0.0, ta + 3 * T1, int((ta + 3 * T1) / (min(4 * xi, ta) / 200)) + 1)
        res = startup_response(b, StartProfile("sine", ta, "none"), 0.0, tau)
        D_exit = res.tension([0.0])[0].min() / (-xi)
        assert D_exit == pytest.approx(strand_curves([rA])[0][0], rel=0.015)


def test_entry_collapses_on_the_universal_curve():
    """Strand B is composite (return + carry), but for tau_a / T_1 >= 0.8 the drive-entry
    amplification depends on tau_a / T_1 only, within 2.5 %; exactly for gamma = 1."""
    for r in (0.85, 1.6, 4.0):
        ref = strand_curves([r])[0][0]
        for xi in (0.01, 0.3, 0.7, 0.95):
            for g in (1.0, 2.0, 3.0):
                _, m = _metrics(xi, g, r)
                tol = 3e-3 if g == 1.0 else 0.025
                assert m.D_entry == pytest.approx(ref, rel=tol)


def test_loop_fast_start_impedances():
    """Very fast start: entry tension = carry impedance x speed (gamma V), exit = - return
    impedance x speed, before any reflection returns (dimensionless Z_c = gamma, Z_r = 1)."""
    xi, g, ta = 0.3, 2.0, 0.1
    lp = Loop.from_positions(0.0, xi, g)
    b = modal_basis(lp, 1e-3, 400)
    tau = np.arange(0.0, 0.95 * 2 * xi, 0.002)
    res = startup_response(b, StartProfile("sine", ta, "none"), 0.0, tau)
    V = 2 * ta / np.pi
    assert res.tension([2.0])[0].max() == pytest.approx(g * V, rel=0.02)
    assert res.tension([0.0])[0].min() == pytest.approx(-V, rel=0.01)


def test_travel_from_the_strands():
    """Peak take-up travel ~ (D_free(tau_a/T_1) S_B - S_A)/2 within 6 % (gamma >= 2,
    tau_a / T_1 >= 1)."""
    for r in (1.0, 2.0, 5.0):
        Df = strand_curves([r])[1][0]
        for xi in (0.01, 0.3, 0.7, 0.99):
            for g in (2.0, 3.0):
                lp, m = _metrics(xi, g, r)
                assert strand_travel_estimate(lp, Df) == pytest.approx(m.travel_max, rel=0.06)


def test_takeup_mass_barely_matters():
    """beta = 0.1 changes the entry amplification by less than 1.5 % away from the corner
    gamma ~ 1 with the take-up at the tail."""
    for r in (1.0, 3.0):
        for xi, g in [(0.05, 2.0), (0.5, 1.5), (0.05, 2.93)]:
            _, m0 = _metrics(xi, g, r, beta=1e-3)
            _, m1 = _metrics(xi, g, r, beta=0.1)
            assert m1.D_entry == pytest.approx(m0.D_entry, rel=0.015)


# --------------------------------------------------------------- crawl start (phase 4.5b)
from beltloop import Segment, crawl_onset, crawl_start, startup_with_onset  # noqa: E402


def _resistance_only_peak(tau_r, rho=1.0):
    lp = Loop((Segment(1e-3, r=rho),), (Segment(1.0, r=rho),))
    b = modal_basis(lp, 1e-8, 80)
    prof = crawl_start(1.0, 0.0, 12.0, 0.0)                     # main phase only
    tau = np.linspace(0.0, 30.0, 6001)
    x = [lp.length]
    T, _ = startup_with_onset(b, prof, crawl_onset(tau_r), 0.0, tau, x)
    inert = startup_response(b, prof, 0.0, tau).tension(x)
    return (T - inert)[0].max() / rho


def test_abrupt_onset_doubles_the_resistance_tension():
    assert 1.9 < _resistance_only_peak(1e-3) < 2.05


def test_onset_ramp_of_one_period_leaves_no_overshoot():
    """A ramp lasting T_s is a whole number of periods of every strand mode (T_s/(2k-1)), so
    no mode keeps a residual vibration: the resistance part reaches its static value without
    overshoot."""
    assert _resistance_only_peak(4.0) == pytest.approx(1.0, abs=0.01)
    assert _resistance_only_peak(2.0) > 1.3


def test_crawl_start_kinematics():
    prof = crawl_start(2.0, 3.0, 6.0, 0.1)
    tau = np.array([2.0, 4.0, 11.0, 20.0])
    v = prof.velocity(tau)
    assert v[0] == pytest.approx(0.1) and v[1] == pytest.approx(0.1)
    assert v[2] == pytest.approx(1.0) and v[3] == pytest.approx(1.0)
    assert prof.a(np.array([8.0]))[0] == pytest.approx(1.0)     # peak of the main phase
