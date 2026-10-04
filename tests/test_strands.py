"""Take-up carriage carried by n belt strands (n:1 kinematics; Harrison 1985, Fig. 2a has
n = 4). The modal solution uses the exact map onto the single-loop model (belt-side mass
4 M / n^2, carriage travel 2/n of the belt-side travel); the lumped model builds the
n:1 kinematics in directly, so their agreement tests the map."""
import numpy as np
import pytest

from beltloop import Conveyor, GravityTakeUp, natural_frequencies
from beltloop.lumped import LumpedModel

G = 9.80665


def make(n=4, Mw=4 * 70e3 / G, sd=0.0, st=60.0, t_v=0.3):
    return Conveyor(L=3000.0, EA=1.2e8, mu_r=40.0, mu_c=100.0, m_r=30.0, m_c=80.0,
                    drive_position=sd, takeup_position=st,
                    takeup=GravityTakeUp(M_w=Mw, strands=n), r_r=15.0, r_c=45.0, t_v=t_v,
                    carry_profile=((0, 0), (1500, 10), (3000, 40)))


@pytest.mark.parametrize("n", [2, 4, 6])
@pytest.mark.parametrize("Mw,Mc,i,kap", [(5000, 0, 1, 1), (3000, 800, 2, 1), (1500, 400, 4, 0)])
def test_reeving_identity_n(n, Mw, Mc, i, kap):
    """n T_t = g (i M_w + kappa M_c) and M = n i T_t / g + (1 - i kappa) M_c."""
    tu = GravityTakeUp(Mw, Mc, i, kap, strands=n)
    assert n * tu.T_t == pytest.approx(G * (i * Mw + kap * Mc))
    assert tu.M == pytest.approx(n * i * tu.T_t / G + (1 - i * kap) * Mc)
    assert tu.M_belt == pytest.approx(4 * tu.M / n ** 2)
    assert tu.max_acceleration == pytest.approx(n * tu.T_t / tu.M)


@pytest.mark.parametrize("n", [2, 4])
def test_direct_counterweight_beta(n):
    """Directly hung counterweight: beta = 4 T_t / (n g mu_r L). For the same take-up
    tension, a double loop halves beta."""
    cv = make(n=n, Mw=20000.0)
    assert cv.beta == pytest.approx(4 * cv.takeup.T_t / (n * G * cv.mu_r * cv.L))
    assert cv.takeup.max_acceleration == pytest.approx(G)


def test_invalid_strands():
    for n in (1, 3, 2.5, 0):
        with pytest.raises(ValueError):
            GravityTakeUp(1000.0, strands=n)


def test_map_n_onto_single_loop():
    """n = 4 with carriage mass M is the single-loop model with mass M/4: same spectrum and
    dynamic tension; carriage travel halved; same belt length stored in the loop."""
    c4 = make(n=4, Mw=8000.0)
    c2 = make(n=2, Mw=2000.0)
    assert c4.beta == pytest.approx(c2.beta, rel=1e-14)
    np.testing.assert_allclose(natural_frequencies(c4.loop(), c4.beta, 6),
                               natural_frequencies(c2.loop(), c2.beta, 6), rtol=1e-12)
    r4 = c4.start(5.0, 60.0, n_modes=40, t_end=120.0, n_t=241)
    r2 = c2.start(5.0, 60.0, n_modes=40, t_end=120.0, n_t=241)
    s = np.linspace(0, 2 * c4.L, 37)
    np.testing.assert_allclose(r4.dynamic_tension(s), r2.dynamic_tension(s), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(r4.takeup_displacement(), 0.5 * r2.takeup_displacement(), rtol=1e-12)
    np.testing.assert_allclose(r4.loop_storage(), r2.loop_storage(), rtol=1e-12)


def test_lumped_rigid_motion_and_statics_n4():
    """n:1 kinematics in the lumped model: a uniform translation is strain-free and loads
    nothing; the static equilibrium holds the take-up tension n T_t / n = T_t."""
    cv = make(n=4)
    lm = LumpedModel(cv, 300)
    e = np.ones(lm.N + 2); e[-1] = 0.0
    assert np.max(np.abs(lm.K @ e)) < 1e-6 * cv.EA / lm.h.min()
    u = lm.static_state()
    T = lm.element_tension(u[:, None], np.zeros((len(u), 1)))[:, 0]
    np.testing.assert_allclose(T, cv.static_tension(lm.x_mid), rtol=1e-9)


@pytest.mark.parametrize("sd,st", [(0.0, 60.0), (1800.0, 900.0)])
def test_lumped_eigenfrequencies_n4(sd, st):
    cv = make(n=4, sd=sd, st=st)
    ex = natural_frequencies(cv.loop(), cv.beta, 5) * cv.c_r / cv.L
    e = [np.max(np.abs(LumpedModel(cv, N).eigenfrequencies(5) / ex - 1)) for N in (250, 500)]
    assert e[1] < 2e-4 and 3.5 < e[0] / e[1] < 4.5


def test_lumped_startup_n4():
    """Start-up: dynamic tension and carriage travel of the native n:1 lumped model against
    the modal solution through the map (second-order convergence)."""
    cv = make(n=4)
    ref = cv.start(5.0, 60.0, "sine", "velocity", n_modes=80, t_end=120.0, n_t=1201)
    errs = []
    for N in (250, 500):
        k = N // 125
        res = LumpedModel(cv, N).start(5.0, 60.0, "sine", "velocity", dt=0.1 / k, t_end=120.0,
                                       store_every=k)
        Tm = ref.dynamic_tension(res.x_mid)
        errs.append(np.max(np.abs(res.T_dynamic - Tm)) / np.max(np.abs(Tm)))
        ey = np.max(np.abs(res.y - ref.takeup_displacement())) / np.max(np.abs(res.y))
        assert ey < 2e-5
    assert errs[1] < 5e-6 and errs[0] / errs[1] > 3.0
