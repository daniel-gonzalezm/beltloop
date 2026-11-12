"""Dimensional layer: take-up reeving, static tension, scaling and validity checks."""
import numpy as np
import pytest

from beltloop.conveyor import Conveyor, GravityTakeUp
from beltloop.eigen import modal_basis
from beltloop.forcing import StartProfile
from beltloop.response import startup_response

G = 9.80665


@pytest.mark.parametrize("Mw,Mc,i,kap", [(5000, 0, 1, 1), (3000, 800, 2, 1), (2000, 600, 3, 0.2), (1500, 400, 4, 0)])
def test_reeving_identity(Mw, Mc, i, kap):
    tu = GravityTakeUp(Mw, Mc, i, kap)
    assert tu.M == pytest.approx(2 * i * tu.T_t / G + (1 - i * kap) * Mc)
    assert tu.force == pytest.approx(G * (i * Mw + kap * Mc))


def test_direct_counterweight():
    tu = GravityTakeUp(20000.0)
    assert tu.max_acceleration == pytest.approx(G)
    cv = Conveyor(L=5100, EA=39 * 1450 ** 2, mu_r=39, mu_c=39, drive_position=0,
                  takeup_position=100, takeup=tu)
    assert cv.beta == pytest.approx(2 * tu.T_t / (G * cv.mu_r * cv.L))
    eps_t = tu.T_t / cv.EA
    assert cv.beta == pytest.approx(2 * cv.c_r ** 2 / (G * cv.L) * eps_t)
    assert cv.beta == pytest.approx(0.1006, abs=5e-4)                  # Harrison (1983) data


def make_inclined(H=50.0, sigma_t=40.0, r=(20.0, 60.0)):
    return Conveyor(L=1000.0, EA=8e7, mu_r=30.0, mu_c=90.0, m_r=20.0, m_c=70.0,
                    drive_position=0.0, takeup_position=sigma_t, takeup=GravityTakeUp(4000.0),
                    r_r=r[0], r_c=r[1], carry_profile=((0, 0), (1000.0, H)))


def test_static_tension_inclined():
    cv = make_inclined()
    H, L, tu = 50.0, 1000.0, cv.takeup
    h_t = H * (L - 40.0) / L
    T0 = cv.static_tension(np.array([0.0, 40.0, 2 * L]))
    assert T0[1] == pytest.approx(tu.T_t)
    assert T0[0] == pytest.approx(tu.T_t + cv.m_r * G * (H - h_t))
    assert T0[2] == pytest.approx(tu.T_t - cv.m_r * G * h_t + cv.m_c * G * H)


def test_static_tension_horizontal_uniform():
    cv = make_inclined(H=0.0)
    s = np.linspace(0, 2000, 101)
    np.testing.assert_allclose(cv.static_tension(s), cv.takeup.T_t, rtol=1e-12)


def test_running_tension_effective_pull():
    cv = make_inclined(H=0.0)
    T = cv.running_tension(np.array([0.0, 2000.0]))
    assert T[1] - T[0] == pytest.approx((cv.r_r + cv.r_c) * cv.L)


def test_dimensional_start_matches_dimensionless():
    cv = make_inclined()
    cv.t_v = 0.2
    V, ta = 4.0, 20.0
    st = cv.start(V, ta, "sine", "velocity", n_modes=30, t_end=40.0, n_t=401)
    a_m = np.pi * V / (2 * ta)
    lp = cv.loop(a_m)
    mb = modal_basis(lp, cv.beta, 30)
    r = startup_response(mb, StartProfile("sine", cv.c_r * ta / cv.L, "velocity"), cv.zeta_hat, st.response.tau)
    s = np.array([0.0, 500.0, 1999.0])
    np.testing.assert_allclose(st.dynamic_tension(s), cv.mu_r * cv.L * a_m * r.tension(s / cv.L), rtol=1e-12)
    # steady running: drive faces differ by the total resistance plus the lift, with damping decayed
    ck = st.checks()
    assert set(ck) >= {"tension_positive", "takeup_follows", "min_total_tension_N"}
    assert ck["takeup_accel_limit_ms2"] == pytest.approx(G)


def test_validity_flags_violation():
    """A light take-up with a hard start must be flagged (slack at the drive exit)."""
    cv = Conveyor(L=2000.0, EA=6e7, mu_r=25.0, mu_c=60.0, drive_position=0.0, takeup_position=50.0,
                  takeup=GravityTakeUp(300.0), r_r=15.0, r_c=40.0)
    ck = cv.start(5.0, 2.0, "triangular", "step", n_modes=40).checks()
    assert not ck["tension_positive"]


def test_takeup_on_tight_side_is_flagged():
    """Take-up 900 m upstream of an intermediate drive (on its tight side): the take-up anchors
    the entry tension and the carry-strand resistance is taken from the slack side."""
    cv = Conveyor(L=3000.0, EA=1.2e8, mu_r=40.0, mu_c=100.0, drive_position=1800.0,
                  takeup_position=900.0, takeup=GravityTakeUp(2 * 70e3 / G), r_r=15.0, r_c=45.0, t_v=0.3)
    ck = cv.start(5.0, 60.0, "parabolic", "velocity", n_modes=40).checks()
    assert not ck["tension_positive"] and ck["min_tension_s_m"] < 1.0      # at the drive exit
