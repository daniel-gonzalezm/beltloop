"""Phase 4.6: validity zones (slack, grip, take-up kinematics)."""
import numpy as np
import pytest

from beltloop import (Conveyor, GravityTakeUp, Loop, StartProfile, modal_basis, natural_frequencies,
                      startup_response, strand_extremes, strand_rebound, takeup_kinematics,
                      tension_requirement)
from beltloop.metrics import _time_grid


def _free_end_velocity(profile, tau, length):
    """d'Alembert: absolute velocity of the free end of a uniform fixed-free strand whose fixed
    end moves with V(t): v = 2 sum_k (-1)^k V(t - (2k + 1) l)."""
    V = lambda t: np.where(t > 0, profile.velocity(np.maximum(t, 0)) * profile.a_integral, 0.0)
    out = np.zeros_like(tau)
    k = 0
    while (2 * k + 1) * length <= tau[-1]:
        out += 2 * (-1) ** k * V(tau - (2 * k + 1) * length)
        k += 1
    return out


@pytest.mark.parametrize("xi", [0.05, 0.3, 0.8])
@pytest.mark.parametrize("ta", [1.0, 6.0])
def test_takeup_velocity_matches_dalembert(xi, ta):
    """gamma = 1, beta -> 0: y' = (v_A - v_B) / 2 from the free ends of both strands."""
    lp = Loop.from_positions(0.0, xi, 1.0)
    pr = StartProfile("sine", ta, "none")
    tau = np.linspace(0, ta + 16, 3001)
    res = startup_response(modal_basis(lp, 1e-6, 150), pr, 0.0, tau)
    ref = 0.5 * (_free_end_velocity(pr, tau, xi) - _free_end_velocity(pr, tau, 2 - xi))
    assert np.abs(res.takeup_velocity() - ref).max() < 1e-4 * pr.a_integral


@pytest.mark.parametrize("kind", ["sine", "triangular", "parabolic"])
def test_takeup_kinematic_bounds_gamma_one(kind):
    """Exact bounds for gamma = 1, beta -> 0: |y'| <= V_inf, |y''| <= 2 a_m."""
    for xi in (0.1, 0.5, 0.9):
        lp = Loop.from_positions(0.0, xi, 1.0)
        T1 = 2 * np.pi / natural_frequencies(lp, 1e-3, 1)[0]
        for r in (0.25, 0.5, 1.0):
            k = takeup_kinematics(lp, StartProfile(kind, r * T1, "none"), points=600)
            assert max(abs(k.v_max), abs(k.v_min)) <= 1.0 + 1e-3
            assert max(abs(k.a_max), abs(k.a_min)) <= 2.0 + 2e-2


def test_takeup_kinematics_small_for_slow_starts():
    """tau_a >= 2 T_1: velocity below 6 % of V_inf, acceleration below 0.6 a_m."""
    for g in (1.0, 2.0, 3.0):
        for xi in (0.05, 0.5, 0.9):
            lp = Loop.from_positions(0.0, xi, g)
            T1 = 2 * np.pi / natural_frequencies(lp, 1e-3, 1)[0]
            k = takeup_kinematics(lp, StartProfile("sine", 2 * T1, "none"))
            assert max(abs(k.v_max), abs(k.v_min)) < 0.06
            assert max(abs(k.a_max), abs(k.a_min)) < 0.6


@pytest.mark.parametrize("rho", [0.0, 1.0, 2.0, 5.0])
def test_exit_quasi_static_limit(rho):
    """Slow sine start with phi = V/V_inf: max_t [a - rho (1 - phi)] = sqrt(1 + rho^2/4) - rho/2."""
    cmax, _ = strand_extremes([20.0], rho)
    assert cmax[0] - rho == pytest.approx(np.sqrt(1 + rho ** 2 / 4) - rho / 2, abs=0.03)


def test_rebound_curve():
    """Fast starts: the rebound equals the wave-law peak; it vanishes where the residual of the
    first mode vanishes (t_a = 1.5 T_s, 2.5 T_s) and decays as ~0.8 / r."""
    Dm, Rp = strand_rebound([0.5, 1.5, 2.5, 4.0, 10.0])
    assert Dm[0] == pytest.approx(8 * 0.5 / np.pi, rel=1e-3)
    assert Dm[1] < 1e-3 and Dm[2] < 1e-3
    assert Dm[3] == pytest.approx(0.8 / 4, rel=0.05)
    assert Dm[4] == pytest.approx(0.8 / 10, rel=0.05)
    assert np.all(Rp <= 1 + 1e-9) and Rp[-1] > 0.99


CASES = [(2.02, 0.93, 0.5, 0.0), (2.90, 0.32, 5.0, 1.0), (1.66, 0.78, 0.5, 0.5),
         (1.91, 0.15, 1.0, 1.0), (2.92, 0.72, 1.0, 1.0), (1.55, 0.17, 1.0, 5.0)]


def _exact_requirement(g, xi, r, rho, euler=None):
    lp = Loop.from_positions(0.0, xi, g, rho, rho * g * g)
    b = modal_basis(lp, 1e-6, 100)
    T1 = 2 * np.pi / b.Om[0]
    tau = _time_grid(r * T1 + 3 * T1, min(T1, r * T1) / 200)
    res = startup_response(b, StartProfile("sine", r * T1, "velocity"), 0.0, tau)
    T = res.tension(np.linspace(0, 2, 401))
    slack = -T.min() - rho * xi
    grip = None
    if euler is not None:
        grip = ((T[-1] - euler * T[0]) / (euler - 1)).max() - rho * xi
    return lp, r * T1, slack, grip


@pytest.mark.parametrize("g,xi,r,rho", CASES)
def test_requirement_against_full_loop(g, xi, r, rho):
    """Strand estimate of the minimum T_2 against the full loop: exact when strand A governs,
    within 3 % of the belt inertia force when the rebound in strand B governs."""
    lp, ta, slack, _ = _exact_requirement(g, xi, r, rho)
    req = tension_requirement(lp, ta, rho)
    est = max(req.exit, req.rebound)
    tol = 1e-3 if req.governing == "exit" else 0.03
    assert est == pytest.approx(slack, abs=tol * (1 + g * g))


@pytest.mark.parametrize("g,xi,r,rho,E", [(2, 0.05, 1, 1, 3.0), (2, 0.95, 1, 1, 16.0),
                                           (2, 0.5, 3, 2, 3.0), (1.5, 0.3, 5, 5, 8.0)])
def test_grip_requirement(g, xi, r, rho, E):
    """Grip estimate within 3 % of the full loop, and it always covers the exit slack
    (T_entry <= E T_exit with T_entry > 0 implies T_exit > 0)."""
    lp, ta, _, grip = _exact_requirement(g, xi, r, rho, E)
    req = tension_requirement(lp, ta, rho, euler=E)
    assert req.grip == pytest.approx(grip, rel=0.03)
    assert req.grip >= req.exit
    assert req.grip >= rho * (1 + g * g) / (E - 1) - 1e-9      # steady running


def test_dimensional_check_follows_requirement():
    """SI conveyor (direct counterweight, so beta = 2 T_t / (g mu_r L) ~ 0.4 here): with T_2
    10 % above the strand estimate the total tension stays positive; 10 % below, the check
    flags it. Fast start with low resistances, so the rebound in strand B governs."""
    L, EA, mu_r, gam, xi, rho, V = 3000.0, 150e6, 60.0, 2.0, 0.1, 0.3, 4.0
    c_r = np.sqrt(EA / mu_r)
    lp = Loop.from_positions(0.0, xi, gam)
    T1 = 2 * np.pi / natural_frequencies(lp, 1e-6, 1)[0]          # units of L / c_r
    tau_a = 0.8 * T1
    t_a = tau_a * L / c_r
    a_m = np.pi / 2 * V / t_a
    req = tension_requirement(lp, tau_a, rho)
    assert req.governing == "rebound"
    flags = {}
    for f in (0.9, 1.1):
        T_t = (f * req.T2_min + rho * xi) * mu_r * L * a_m
        cv = Conveyor(L=L, EA=EA, mu_r=mu_r, mu_c=mu_r * gam ** 2, drive_position=0.0,
                      takeup_position=xi * L, takeup=GravityTakeUp(M_w=2 * T_t / 9.80665),
                      r_r=rho * mu_r * a_m, r_c=rho * mu_r * gam ** 2 * a_m)
        assert cv.beta < 0.5
        flags[f] = cv.start(V, t_a, n_modes=80, n_t=3001).checks()["tension_positive"]
    assert flags[1.1] and not flags[0.9]


@pytest.mark.parametrize("n,M_c,i,kappa", [(2, 0.0, 1.0, 1.0), (4, 0.0, 1.0, 1.0),
                                           (2, 8000.0, 0.5, 0.0), (4, 3000.0, 1.0, 0.3)])
def test_beta_from_takeup_tension(n, M_c, i, kappa):
    """beta = (4/n) lam T_t / (g mu_r L), lam = g / max_acceleration (1 for a direct
    counterweight): the identity behind TensionRequirement.beta_min."""
    tu = GravityTakeUp(M_w=20000.0, M_c=M_c, i=i, kappa=kappa, strands=n)
    cv = Conveyor(L=4000.0, EA=150e6, mu_r=60.0, mu_c=240.0, drive_position=0.0,
                  takeup_position=200.0, takeup=tu)
    lam = tu.g / tu.max_acceleration
    assert cv.beta == pytest.approx(4 / n * lam * tu.T_t / (tu.g * 60.0 * 4000.0), rel=1e-12)
    if M_c == 0.0 and i == 1.0:
        assert lam == pytest.approx(1.0)


def test_beta_min_running_grip():
    """Slow start (tau_a = 20 T_1): beta_min -> (4/n) f [m_belt/(E - 1) + m_A] up to the
    small inertial part, with rho = f g / a_m."""
    g_, xi, E, f, a_over_g = 2.0, 0.1, 3.0, 0.02, 0.01
    rho = f / a_over_g
    lp = Loop.from_positions(0.0, xi, g_)
    T1 = 2 * np.pi / natural_frequencies(lp, 1e-6, 1)[0]
    req = tension_requirement(lp, 20 * T1, rho, euler=E)
    assert req.governing == "grip"
    run = 4 / 2 * f * ((1 + g_ ** 2) / (E - 1) + xi)
    inert = 4 / 2 * a_over_g * (1 + g_ ** 2) * 1.1 / (E - 1)     # D <= 1.1 for slow starts
    assert run < req.beta_min(a_over_g) < run + inert


def test_takeup_acceleration_envelope_fast_start():
    """Very fast start (tau_a = 0.01 T_1), gamma = 3: |y''| reaches but does not exceed
    2.5 a_m; |y'| stays below 1.8 V_inf."""
    for xi in (0.1, 0.5):
        lp = Loop.from_positions(0.0, xi, 3.0)
        T1 = 2 * np.pi / natural_frequencies(lp, 1e-3, 1)[0]
        k = takeup_kinematics(lp, StartProfile("sine", 0.01 * T1, "none"), n_modes=600,
                              points=150, periods_after=2.0)
        assert max(abs(k.a_max), abs(k.a_min)) <= 2.5 + 0.02
        assert max(abs(k.v_max), abs(k.v_min)) <= 1.8


@pytest.mark.parametrize("g,xi,r,rho", [(3.0, 0.05, 0.5, 0.15), (3.0, 0.05, 1.0, 0.3),
                                        (2.0, 0.5, 2.0, 0.6), (3.0, 0.95, 1.0, 3.0)])
def test_requirement_accuracy_band(g, xi, r, rho):
    """Worst cases of the accuracy sweep, tau_a >= T_1/2: the slack estimate is at most
    0.04 m_belt below the full loop."""
    lp, ta, slack, _ = _exact_requirement(g, xi, r, rho)
    req = tension_requirement(lp, ta, rho)
    assert max(req.exit, req.rebound) - slack > -0.04 * (1 + g * g)
