"""Phase 5.4: take-up tension requirements (maps/beta_min.py) against the sources."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "maps"))
import beta_min as bm  # noqa: E402
import cases as cs  # noqa: E402

G = cs.G_STD


@pytest.fixture(scope="module")
def ds():
    return bm.designs()


def test_lodewijks_design_calculation(ds):
    """Lodewijks (1996, eqs. 8.7-8.9): F_a = 1.2 F_U, e^(mu theta) = 3 gives 21.33 kN at
    start and 17.78 kN in running; his take-up force 42.66 kN = 2 x 21.33 sits on the
    start-grip floor."""
    d = ds["Lo"]
    assert d.T_grip_start == pytest.approx(21.33e3, rel=3e-3)
    assert d.T_grip_run == pytest.approx(17.78e3, rel=3e-3)
    assert d.margin("grip_start") == pytest.approx(1.0, abs=5e-3)
    assert d.E_start == pytest.approx(d.E, rel=5e-3)


def test_surtees_design_sheet(ds):
    """Design sheet (p. 42): 2 % sag at the tail needs 23 595 N; T1 steady = 253 065 N with
    T2 = 100 kN; design 1 hangs 4795 kg (23.5 kN) on two strands."""
    c = cs.BY_TAG["Su"]
    T_sag = G * (c.m_b.value + c.m_l.value) * c.l_o.value / (8 * 0.02)
    assert T_sag == pytest.approx(23595, rel=1e-3)
    st = bm.state(c)
    assert st.T_run[-1] == pytest.approx(253065, rel=5e-3)
    assert ds["Su1"].T_t == pytest.approx(4795 * G / 2, rel=1e-9)
    assert ds["Su1"].T_t == pytest.approx(23620, rel=5e-3)        # sheet's selected design-1 T2
    # design 1 sits on its floor (grip with the 1.3 start factor and 2 % sag), design 2 ~5x
    assert 1.0 < ds["Su1"].T_t / ds["Su1"].floor < 1.3
    assert ds["Su"].T_t / ds["Su"].floor > 4.5


def test_song_running_tensions(ds):
    """Their text gives 337 / 113 kN at the drive with 173 kN at the tail take-up: the running
    state reproduces both, and the ratio needs e^(mu theta) = 3 (mu = 0.3: ~210 deg)."""
    st = bm.state(cs.BY_TAG["S"], sigma_t=0.999)
    assert st.T_run[-1] == pytest.approx(337e3, rel=1e-2)
    assert st.T_run[0] == pytest.approx(113e3, rel=2e-2)
    E = ds["S"].E_run
    assert E == pytest.approx(337 / 113, rel=2e-2)
    assert np.degrees(np.log(E) / 0.3) == pytest.approx(210, abs=5)


def test_qnk_tensions():
    """SM: drive entry 671 932 N (QNK point 2) with the common f; with the return coefficient
    at 0.55 of the carry one (QNK's split) the tail tension is QNK's 109 kN."""
    c = cs.BY_TAG["SM"]
    st = bm.state(c)
    assert st.T_run[-1] == pytest.approx(671932, rel=1e-3)
    st2 = bm.state(c, q_r=0.55)
    tail = st2.T_run[np.argmin(abs(st2.s - c.L.value))]
    assert tail == pytest.approx(109.3e3, rel=0.05)
    assert st2.T_run[-1] == pytest.approx(671932, rel=5e-3)


def test_pascual_drive_tensions_and_lift_floor():
    """Pa: f1 / f2 = 1.417 / 0.300 MN need e^(mu theta) >= 4.72; T_t holds the tail at
    ~2 % sag (1.2 m carry spacing) at the foot of the 9 deg incline: the floor is lift."""
    d = bm.design(cs.BY_TAG["Pa"])
    assert d.E_run == pytest.approx(1417 / 300, rel=2e-3)
    assert d.margin("sag2") == pytest.approx(1.0, abs=0.03)
    lo = bm.design(cs.BY_TAG["Pa"], q_r=0.55).margin("sag2")
    assert 0.88 < lo < 0.95                                         # QNK-like split


def test_li_start_grip(ds):
    """LL: the published start (850 kN in, ~200 kN out) leaves 17-25 % over the start grip
    floor with e^(mu alpha) = 4.81."""
    assert 1.15 < ds["LL"].margin("grip_start") < 1.25
    assert ds["LL"].E_start < 4.81


def test_beta_identity(ds):
    """beta = K T_t / W_r with K = (4 / n) lam: 2 for direct counterweights on two strands,
    1 on four (H) and for Si's rigging-free rule (4 T_t / (M_w g) = 1.0)."""
    for k, d in ds.items():
        assert d.beta == pytest.approx(d.K * d.T_t / d.W_r, rel=1e-12)
        assert d.beta == pytest.approx(d.K * d.psi * d.tau, rel=1e-12)
    for k in ("Lo", "S", "LL", "Pa", "SM", "WR", "Su", "Su1"):
        assert ds[k].K == pytest.approx(2.0, rel=1e-9)
    assert ds["H"].K == pytest.approx(1.0, rel=1e-9)
    c = cs.BY_TAG["Si"]
    assert ds["Si"].K == pytest.approx(4 * c.T_t.value / (c.M_w.value * G), rel=1e-12)


def test_grip_requirement_closes_at_E():
    """With T_t set to the grip requirement, T_entry / T_exit = E exactly (running and start),
    and the start state carries p F_U across the drive."""
    c = cs.BY_TAG["SM"]
    E = c.E.value
    for start in (False, True):
        st = bm.state(c, p=1.4)
        Treq = bm.grip_requirement(st, E, start)
        st2 = bm.state(c, p=1.4, M_w=2 * Treq / G)
        T = st2.T_start if start else st2.T_run
        assert T[-1] / T[0] == pytest.approx(E, rel=1e-9)
    assert (st2.T_start[-1] - st2.T_start[0]) == pytest.approx(1.4 * c.F_U.value, rel=1e-9)
    assert (st2.T_run[-1] - st2.T_run[0]) == pytest.approx(c.F_U.value, rel=1e-9)


@pytest.mark.parametrize("tag", ["Lo", "SM", "Su"])
def test_grip_floor_independent_of_length(tag):
    """Same belt, load, slope and C f on a longer route: F_U and the grip floor scale with L,
    so beta at the grip floor does not change; a fixed tension gives beta ~ 1 / L."""
    c = cs.BY_TAG[tag]
    E = c.E.value
    out = []
    for k in (1.0, 2.0, 5.0):
        ck = bm.scaled_case(c, k)
        st = bm.state(ck, p=1.3)
        T = bm.grip_requirement(st, E, start=True)
        W = G * ck.mu_r * ck.L.value
        out.append((2 * T / W, ck.F_U.value / W))
    for b, psi in out[1:]:
        assert b == pytest.approx(out[0][0], rel=2e-3)
        assert psi == pytest.approx(out[0][1], rel=1e-9)
    T_fix = 30e3
    b = [2 * T_fix / (G * c.mu_r * k * c.L.value) for k in (1.0, 2.0)]
    assert b[1] == pytest.approx(b[0] / 2)


def test_scaled_case_reproduces_original():
    c = cs.BY_TAG["SM"]
    assert bm.scaled_case(c, 1.0).F_U.value == pytest.approx(c.F_U.value, rel=1e-9)


def test_start_factor_from_time():
    """Quasi-static start factors of the published starts: Song's 300 s cycloid gives 1.06
    against their peak 236 / 225 = 1.05; Lodewijks' 30 s profiles give 2.4 (his design used
    1.2: E = 3 would not hold them; checked in 5.5)."""
    assert bm.start_factor_from_time(cs.BY_TAG["S"]) == pytest.approx(236 / 225, abs=0.02)
    assert bm.start_factor_from_time(cs.BY_TAG["Lo"]) == pytest.approx(2.40, abs=0.02)


def test_regime_and_wr_siblings():
    """beta / beta_5(B1): every published conveyor of 1 km or more below 0.3; Wheatley and
    Rubel's B and C (247 and 91 m) have heavier take-ups on shorter belts and C crosses the
    5 % threshold (n = 2 assumed)."""
    rows = {r[0]: r for r in bm.regime()}
    for k in ("Lo", "S", "LL", "Pa", "SM", "Si", "H"):
        assert rows[k][4] < 0.3
    assert rows["WR-B"][2] > rows["WR"][2]
    assert rows["WR-C"][4] > 1.0
    Ts = [d[3] * G / 2 for d in bm.wr_siblings()]
    assert Ts[0] > Ts[1]                       # the 91 m belt is not the lightest take-up


def test_case_fields_54():
    for tag in ("S", "LL", "SM", "Lo", "Pa", "Si", "WR", "Su"):
        c = cs.BY_TAG[tag]
        assert c.l_o is not None and c.l_u is not None and c.F_U is not None
        assert c.takeup_basis
    assert cs.BY_TAG["LL"].F_U.kind == "assumed"
    c = cs.BY_TAG["Si"]
    F = cs.din_peripheral_force(13100, 0.013, c.m_ic.value, c.m_ir.value, 29.7,
                                4200 / 3.6 / 8.5, 9.0)
    assert c.F_U.value == pytest.approx(F, abs=1.0)
