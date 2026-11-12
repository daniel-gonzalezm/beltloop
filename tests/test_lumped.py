"""Independent lumped-mass model (absolute displacements, gravity, Newmark) vs the modal
solution. Checks the decomposition u = u0 + U + w, the claim that gravity only sets the
static state and the 2:1 take-up kinematics, besides the numbers."""
import numpy as np
import pytest

from beltloop import Conveyor, GravityTakeUp, natural_frequencies
from beltloop.lumped import LumpedModel, drive_kinematics

G = 9.80665


def make(sd=0.0, st=60.0, profile=((0, 0), (1500, 10), (3000, 40)), t_v=0.3):
    return Conveyor(L=3000.0, EA=1.2e8, mu_r=40.0, mu_c=100.0, m_r=30.0, m_c=80.0,
                    drive_position=sd, takeup_position=st, takeup=GravityTakeUp(M_w=2 * 70e3 / G),
                    r_r=15.0, r_c=45.0, t_v=t_v, carry_profile=profile)


@pytest.mark.parametrize("kind", ["sine", "triangular", "parabolic"])
def test_drive_kinematics(kind):
    t = np.linspace(0, 80, 80001)
    a, v, d = drive_kinematics(kind, 5.0, 60.0, t)
    assert np.max(np.abs(np.gradient(d, t) - v)) < 1e-6
    assert np.max(np.abs(np.gradient(v, t) - a)[1:-1]) < 1e-4
    assert drive_kinematics(kind, 5.0, 60.0, 60.0)[2] == pytest.approx(5.0 * 60.0 / 2)


def test_static_state_matches_closed_form():
    """Lumped static equilibrium with gravity and counterweight = T0 of the conveyor
    (inclined, piecewise-linear profile); the take-up holds T_t."""
    cv = make()
    lm = LumpedModel(cv, 600)
    u = lm.static_state()
    T = lm.element_tension(u[:, None], np.zeros((len(u), 1)))[:, 0]
    np.testing.assert_allclose(T, cv.static_tension(lm.x_mid), rtol=1e-9)


def test_rigid_motion_is_strain_free():
    """A uniform belt translation produces no force and no take-up load (K e = 0)."""
    lm = LumpedModel(make(), 300)
    e = np.ones(lm.N + 2); e[-1] = 0.0
    assert np.max(np.abs(lm.K @ e)) < 1e-6 * lm.cv.EA / lm.h.min()


@pytest.mark.parametrize("sd,st", [(0.0, 60.0), (1800.0, 900.0), (900.0, 1500.0), (3000.0, 2400.0)])
def test_eigenfrequencies_converge(sd, st):
    cv = make(sd, st)
    ex = natural_frequencies(cv.loop(), cv.beta, 5) * cv.c_r / cv.L
    e = [np.max(np.abs(LumpedModel(cv, N).eigenfrequencies(5) / ex - 1)) for N in (250, 500)]
    assert e[1] < 2e-4 and 3.5 < e[0] / e[1] < 4.5          # second order


@pytest.mark.parametrize("sd,st,kind", [(0.0, 60.0, "sine"), (1800.0, 900.0, "parabolic")])
def test_startup_agreement(sd, st, kind):
    cv = make(sd, st)
    ref = cv.start(5.0, 60.0, kind, "velocity", n_modes=80, t_end=120.0, n_t=1201)
    errs = []
    for N in (250, 500):
        k = N // 125
        res = LumpedModel(cv, N).start(5.0, 60.0, kind, "velocity", dt=0.1 / k, t_end=120.0, store_every=k)
        Tm = ref.dynamic_tension(res.x_mid)
        errs.append(np.max(np.abs(res.T_dynamic - Tm)) / np.max(np.abs(Tm)))
        ey = np.max(np.abs(res.y - ref.takeup_displacement())) / np.max(np.abs(res.y))
        assert ey < 2e-5
    assert errs[1] < 5e-6 and errs[0] / errs[1] > 3.0       # converging at second order


def test_step_onset_undamped():
    """Sharp wave fronts: both discretisations disperse, agreement at the 0.5 % level."""
    cv = make(t_v=0.0)
    ref = cv.start(5.0, 20.0, "triangular", "step", n_modes=200, t_end=60.0, n_t=601)
    res = LumpedModel(cv, 1000).start(5.0, 20.0, "triangular", "step", dt=0.0125, t_end=60.0, store_every=8)
    Tm = ref.dynamic_tension(res.x_mid)
    assert np.max(np.abs(res.T_dynamic - Tm)) / np.max(np.abs(Tm)) < 5e-3
    assert res.T_dynamic[-1].max() == pytest.approx(Tm[-1].max(), rel=3e-3)
