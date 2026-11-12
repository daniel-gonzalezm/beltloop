"""Piecewise-polynomial start-up profiles (Lodewijks 1996, Eqs. 8.35-8.36): closed forms,
exosystem chunks, and the modal response against the independent lumped model."""
import numpy as np
import pytest
from scipy.linalg import expm

from beltloop import Conveyor, GravityTakeUp, PiecewiseProfile
from beltloop.lumped import LumpedModel

PROFILES = [
    PiecewiseProfile.linear(30.0),
    PiecewiseProfile.linear(30.0, v0=0.1, tau_j=0.5),
    PiecewiseProfile.delayed(30.0, 5.0, 5.0),
    PiecewiseProfile((0.0, 4.0, 10.0), ((0.0, 0.5), (2.0, 0.0, -0.05)), onset="step"),
]


@pytest.mark.parametrize("prof", PROFILES)
def test_chunks_reproduce_closed_forms(prof):
    for ch in prof.chunks()[:-1]:
        for u in np.linspace(0.0, ch.end - ch.start, 9)[:-1]:
            z = expm(ch.E * u) @ ch.z0
            assert ch.ca @ z == pytest.approx(float(prof.a(ch.start + u)), abs=1e-12)
            assert ch.cphi @ z == pytest.approx(float(prof.phi(ch.start + u)), abs=1e-12)


@pytest.mark.parametrize("prof", PROFILES)
def test_kinematics_consistent(prof):
    t = np.linspace(0.0, prof.tau_end + 5.0, 40001)
    v = prof.velocity(t) * prof.a_integral
    d = prof.displacement(t)
    inner = np.ones_like(t, bool)
    for b in prof.breaks:
        inner &= np.abs(t - b) > 2e-3
    assert np.max(np.abs(np.gradient(d, t) - v)[inner][1:-1]) < 1e-6
    assert np.max(np.abs(np.gradient(v, t) - prof.a(t))[inner][1:-1]) < 1e-6
    assert prof.velocity(prof.tau_end + 1.0) == pytest.approx(1.0)


def test_linear_offset_matches_eq_8_35_after_jump():
    ta, v0, tj = 30.0, 0.09, 0.4
    p = PiecewiseProfile.linear(ta, v0, tj)
    t = np.linspace(tj, ta, 50)
    np.testing.assert_allclose(p.velocity(t), v0 + (1 - v0) * t / ta, rtol=1e-12)
    assert p.a_integral == pytest.approx(ta)


def test_delayed_has_rest_period():
    p = PiecewiseProfile.delayed(30.0, 5.0, 5.0)
    v = p.velocity(np.array([5.0, 7.5, 10.0, 35.0]))
    np.testing.assert_allclose(v, [5 / 30, 5 / 30, 5 / 30, 1.0], rtol=1e-12)
    assert p.tau_end == 35.0


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        PiecewiseProfile.linear(30.0, v0=0.1)            # offset needs a jump ramp
    with pytest.raises(ValueError):
        PiecewiseProfile((0.0, 1.0), ((1.0,), (1.0,)))   # wrong number of intervals


def _conveyor():
    return Conveyor(L=1000.0, EA=4.2e6, mu_r=21.0, mu_c=160.0, drive_position=0.0,
                    takeup_position=20.0, takeup=GravityTakeUp(M_w=4350.0),
                    r_r=4.0, r_c=31.0, t_v=0.2)


def test_offset_start_against_lumped_model():
    """Linear start with a speed offset (jump ramp 1 s), damped: modal solution vs the
    independent lumped model, tension at both drive faces and take-up travel. The
    acceleration jumps make the viscous tension jump at the wave fronts, so both models
    converge slowly there (~2e-4 with 400 modes); 240 modes and 800 elements give ~4e-4."""
    cv = _conveyor()
    V, ta, v0, tj = 5.0, 30.0, 0.1, 1.0
    a_m = V / ta
    tsc = cv.c_r / cv.L
    prof = PiecewiseProfile.linear(tsc * ta, v0, tsc * tj)
    t_end = 45.0
    ds = cv.start_profile(prof, a_m, t_end, n_modes=240, n_t=451)
    lm = LumpedModel(cv, 800)
    res = lm.start(V, ta, dt=0.005, t_end=t_end, store_every=20,
                   kinematics=cv.profile_kinematics(prof, a_m))
    Tm = ds.total_tension([lm.x_mid[0], lm.x_mid[-1]])
    Tl = res.T_total[[0, -1]]
    Tm = np.array([np.interp(res.t, ds.t, row) for row in Tm])
    scale = np.max(np.abs(Tl - Tl[:, :1]))
    eT = np.max(np.abs(Tm - Tl)) / scale
    ym = np.interp(res.t, ds.t, ds.takeup_displacement())
    ey = np.max(np.abs(ym - res.y)) / np.max(np.abs(res.y))
    print(f"relative difference: tension {eT:.2e}, take-up {ey:.2e}")
    assert eT < 1e-3 and ey < 1e-4
