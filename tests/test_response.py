"""Forced start-up response: exact modal integration, mode acceleration, limits."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from beltloop.eigen import modal_basis
from beltloop.forcing import PROFILES, StartProfile
from beltloop.loop import Loop
from beltloop.response import integrate_modes, residual_amplitude_sine, startup_response


@pytest.fixture(scope="module")
def basis():
    lp = Loop.from_positions(0.0, 0.24, 2.0, r_return=0.4, r_carry=1.1)
    return modal_basis(lp, 0.13, 40)


# ------------------------------------------------------------------ profiles
@pytest.mark.parametrize("kind", PROFILES)
@pytest.mark.parametrize("onset", ["velocity", "step", "none"])
def test_profile_exosystem_matches_closed_form(kind, onset):
    prof = StartProfile(kind, 7.0, onset)
    for ch in prof.chunks():
        end = ch.end if np.isfinite(ch.end) else ch.start + 5.0
        for t in np.linspace(ch.start, end, 7)[:-1] + 1e-9:
            z = expm(ch.E * (t - ch.start)) @ ch.z0
            assert ch.ca @ z == pytest.approx(prof.a(t), abs=1e-12)
            assert ch.cphi @ z == pytest.approx(prof.phi(t), abs=1e-12)


@pytest.mark.parametrize("kind", PROFILES)
def test_profile_normalisation(kind):
    prof = StartProfile(kind, 6.0, "velocity")
    t = np.linspace(0, 6, 60001)
    a = prof.a(t)
    assert a.max() == pytest.approx(1.0, abs=1e-6)                       # unit peak
    assert np.trapezoid(a, t) == pytest.approx(prof.a_integral, rel=1e-8)
    v = prof.velocity(t)
    assert v[-1] == pytest.approx(1.0) and np.all(np.diff(v) >= -1e-15)
    np.testing.assert_allclose(np.gradient(v, t)[1:-1], (a / prof.a_integral)[1:-1], atol=1e-4)
    smooth = np.abs(t - 0.5 * prof.tau_a) > 1e-3          # triangular corner at mid-start
    smooth[:2] = smooth[-2:] = False
    np.testing.assert_allclose(np.gradient(a, t)[smooth], prof.da(t)[smooth], atol=2e-3)


# ------------------------------------------------------------------ modal integration
@pytest.mark.parametrize("kind", PROFILES)
@pytest.mark.parametrize("onset", ["velocity", "step"])
def test_exact_integration_vs_ode(basis, kind, onset):
    prof = StartProfile(kind, 4.0, onset)
    zh = 0.03
    tau = np.linspace(0, 12, 241)
    P, dP = integrate_modes(basis, prof, zh, tau)
    for k in (0, 3, 9):
        Om, G, R = basis.Om[k], basis.Gamma[k], basis.R[k]
        rhs = lambda t, z: [z[1], -2 * zh * Om ** 2 * z[1] - Om ** 2 * z[0] - G * prof.a(t) - R * prof.phi(t)]
        sol = solve_ivp(rhs, [0, 12], [0, 0], t_eval=tau, rtol=1e-11, atol=1e-13, max_step=0.01)
        scale = np.max(np.abs(sol.y[0])) + 1e-30
        assert np.max(np.abs(P[k] - sol.y[0])) / scale < 1e-6
        assert np.max(np.abs(dP[k] - sol.y[1])) / (np.max(np.abs(sol.y[1])) + 1e-30) < 1e-6


def test_residual_amplitude_including_exact_resonance(basis):
    """Undamped sine start, inertial load only: free amplitude after tau_a equals Eq. (residual);
    the matrix exponential also handles the resonant case varpi = Om_1 exactly."""
    Om = basis.Om
    for tau_a in (3.3, 6.0, np.pi / Om[0]):
        prof = StartProfile("sine", tau_a, "none")
        P, dP = integrate_modes(basis, prof, 0.0, np.array([tau_a, tau_a + 3.0]))
        amp = np.hypot(P[:, -1], dP[:, -1] / Om)
        np.testing.assert_allclose(amp, residual_amplitude_sine(basis, tau_a), rtol=1e-8, atol=1e-13)


def test_damping_ratio(basis):
    """Free decay after the start: log decrement of mode k gives zeta_k = zeta_hat Om_k."""
    zh = 0.01
    prof = StartProfile("sine", 3.0, "none")
    k = 2
    Om = basis.Om[k]
    tau = 3.0 + np.array([0.0, 2 * np.pi / (Om * np.sqrt(1 - (zh * Om) ** 2))])
    P, dP = integrate_modes(basis, prof, zh, tau)
    # amplitude of a damped oscillator after one damped period
    ratio = np.hypot(P[k, 1], (dP[k, 1] + zh * Om ** 2 * P[k, 1]) / Om) / \
        np.hypot(P[k, 0], (dP[k, 0] + zh * Om ** 2 * P[k, 0]) / Om)
    zeta = zh * Om
    assert ratio == pytest.approx(np.exp(-2 * np.pi * zeta / np.sqrt(1 - zeta ** 2)), rel=1e-6)


# ------------------------------------------------------------------ tension field
def test_mode_acceleration_convergence():
    lp = Loop.from_positions(0.7, 0.75, 2.3, r_return=0.3, r_carry=0.8)
    prof = StartProfile("sine", 8.0, "velocity")
    tau = np.linspace(0, 40, 801)
    x = np.linspace(0, 2, 41)
    ref = startup_response(modal_basis(lp, 0.15, 300), prof, 0.02, tau).tension(x)
    r40 = startup_response(modal_basis(lp, 0.15, 40), prof, 0.02, tau)
    scale = np.max(np.abs(ref))
    err_ma = np.max(np.abs(r40.tension(x) - ref)) / scale
    err_plain = np.max(np.abs(r40.tension(x, "plain") - ref)) / scale
    assert err_ma < 1e-4 and err_plain > 10 * err_ma


def test_quasi_static_limit_slow_start():
    """tau_a >> fundamental period: dynamic tension -> a(tau) int_xi^x mu_hat (beta-independent)."""
    lp = Loop.from_positions(0.0, 0.1, 2.0)
    x = np.array([0.0, 0.05, 0.5, 1.2, 2.0])
    for beta in (0.05, 0.5):
        mb = modal_basis(lp, beta, 40)
        tau_a = 400 * 2 * np.pi / mb.Om[0]
        tau = np.linspace(0, tau_a, 2001)
        r = startup_response(mb, StartProfile("sine", tau_a, "none"), 0.0, tau)
        T, Tqs = r.tension(x), r.quasi_static_tension(x)
        assert np.max(np.abs(T - Tqs)) / np.max(np.abs(Tqs)) < 0.01


def test_steady_state_with_damping():
    lp = Loop.from_positions(0.7, 0.75, 2.3, r_return=0.3, r_carry=0.8)
    mb = modal_basis(lp, 0.15, 40)
    x = np.linspace(0, 2, 21)
    for kind in PROFILES:
        r = startup_response(mb, StartProfile(kind, 8.0, "velocity"), 0.5, np.linspace(0, 800, 401))
        np.testing.assert_allclose(r.tension(x)[:, -1], lp.quasi_static("r", x), atol=1e-10)
        assert r.takeup_displacement()[-1] == pytest.approx(0.5 * lp.quasi_static_integral("r"), abs=1e-10)


def test_takeup_acceleration_consistency(basis):
    """beta y'' = -2 T(xi) agrees with the second derivative of y(tau)."""
    r = startup_response(basis, StartProfile("parabolic", 5.0, "velocity"), 0.01, np.linspace(0, 20, 20001))
    y = r.takeup_displacement()
    ydd_fd = np.gradient(np.gradient(y, r.tau), r.tau)
    ydd = r.takeup_acceleration()
    inner = slice(50, -50)
    assert np.max(np.abs(ydd_fd[inner] - ydd[inner])) / np.max(np.abs(ydd)) < 5e-3


def test_takeup_tension_vanishes_quasistatically(basis):
    """The take-up anchors the quasi-static dynamic tension at zero."""
    lp = basis.loop
    assert lp.quasi_static("mu", [lp.xi])[0] == pytest.approx(0, abs=1e-14)
    assert lp.quasi_static("r", [lp.xi])[0] == pytest.approx(0, abs=1e-14)


def test_step_overshoot_uniform_light_takeup():
    """Classical factor 2 for a sudden uniform resistance: gamma = 1, beta -> 0."""
    lp = Loop.from_positions(0.0, 0.05, 1.0, r_return=1.0, r_carry=1.0)
    mb = modal_basis(lp, 1e-6, 300)
    tau = np.linspace(0, 60, 6001)
    T = startup_response(mb, StartProfile("sine", 5.0, "step"), 0.0, tau).tension([2.0])
    Ti = startup_response(mb, StartProfile("sine", 5.0, "none"), 0.0, tau).tension([2.0])
    ratio = np.max(np.abs(T - Ti)) / abs(lp.quasi_static("r", [2.0])[0])
    assert 1.99 < ratio < 2.005


def test_step_overshoot_can_exceed_two():
    """With two wave speeds the step overshoot at the drive entry exceeds 2 (2.18 here;
    confirmed with an independent FE model). The Model section must not claim 'up to twice'."""
    lp = Loop.from_positions(0.0, 0.05, 2.0, r_return=1.0, r_carry=1.0)
    mb = modal_basis(lp, 0.1, 300)
    tau = np.linspace(0, 60, 6001)
    T = startup_response(mb, StartProfile("sine", 5.0, "step"), 0.0, tau).tension([2.0])
    Ti = startup_response(mb, StartProfile("sine", 5.0, "none"), 0.0, tau).tension([2.0])
    ratio = np.max(np.abs(T - Ti)) / abs(lp.quasi_static("r", [2.0])[0])
    assert ratio == pytest.approx(2.182, abs=0.005)
