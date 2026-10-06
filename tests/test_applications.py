"""Phase 5.5: application start-ups (maps/applications.py)."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "maps"))

import applications as ap  # noqa: E402
import beta_min as bm  # noqa: E402
import cases as cs  # noqa: E402

G = cs.G_STD
LO, SM, SU = (cs.BY_TAG[t] for t in ("Lo", "SM", "Su"))


def test_slow_start_tends_to_running_grip():
    """Quasi-static limit: with a very slow start the grip requirement tends to the running
    one of phase 5.4, T_2 >= F_U / (E - 1) carried to the take-up."""
    for c, s in ((LO, 0.001), (LO, 0.5), (SU, SU.sigma_t[0])):
        r = ap.run(c, s, 3000.0)
        req_run = bm.grip_requirement(bm.state(c, 1.0, s), c.E.value)
        assert r.req_grip == pytest.approx(req_run, rel=0.02)
        assert r.req_grip >= req_run * 0.999          # the start can only add


def test_running_grip_lodewijks_closed_form():
    """Lo, horizontal, take-up at the drive: running grip needs T_t = F_U / (E - 1) plus the
    return resistance between the drive and the take-up."""
    c = LO
    r_r, _, _ = bm.strand_resistances(c)
    for s in (0.001, 0.4):
        req = bm.grip_requirement(bm.state(c, 1.0, s), c.E.value)
        assert req == pytest.approx(c.F_U.value / (c.E.value - 1) + r_r * s * c.L.value, rel=1e-3)


def test_slack_at_rest_holds_the_return_on_the_slope():
    """SM: with the take-up near the high end, the tension at rest at the tail is T_t minus the
    weight of the return strand between them; the slack requirement cannot be lower than that,
    and it is that value (the rest state governs) for a gentle start."""
    s = SM.sigma_t[0]
    cv = ap.conveyor(SM, s)
    L = cv.L
    sig = np.linspace(0.0, 2 * L, 4001)
    rest = -(cv.static_tension(sig) - cv.takeup.T_t).min()
    h_t = cv.elevation(s * L)
    assert rest == pytest.approx(SM.m_b.value * G * (h_t - cv.elevation(L)), rel=1e-3)
    r = ap.run(SM, s, 120.0)
    assert r.req_slack >= rest * 0.999
    assert r.req_slack == pytest.approx(rest, rel=0.01)


def test_requirement_is_exact_by_superposition():
    """Total tension = T_t + offset, the offset independent of T_t at fixed mass: putting the
    required tension on the take-up (same counterweight mass, so same dynamics) closes the grip
    ratio at E. Checked by shifting the static level, not by recomputing beta."""
    c, s = LO, 0.001
    cv = ap.conveyor(c, s)
    ds = cv.start(c.V.value, 30.0, ap.KIND, n_modes=80, n_t=2001)
    T = ds.total_tension(np.array([0.0, 2 * cv.L]))
    r = ap.run(c, s, 30.0)
    shift = r.req_grip - cv.takeup.T_t
    assert np.max((T[1] + shift) / (T[0] + shift)) == pytest.approx(c.E.value, rel=1e-6)


def test_lodewijks_30s_starts_exceed_the_grip():
    """Section 4.26 finding, full model: every 30 s start of Table 8.7 needs a tension ratio of
    4.2-6.1 against e^(0.35 pi) = 3.0; the sine (Harrison) start needs T_t ~ 2.4 times his."""
    rows = ap.lodewijks_grip()
    ratios = [r[2] for r in rows]
    assert min(ratios) > 4.1 and max(ratios) < 6.2
    sine = [r for r in rows if r[0].startswith("Harrison") and r[1] == 0.0][0]
    assert sine[3] / 21.33e3 == pytest.approx(2.40, abs=0.03)


def test_lodewijks_shortest_start_and_infeasible_tail():
    """With his T_t, sine starts are admissible from about 101 s (3.6 T_1); beyond the
    position where the running grip alone fails, no start is admissible."""
    ta = ap.shortest_start(LO, 0.001)
    assert ta == pytest.approx(101.3, rel=0.02)
    r_r, _, _ = bm.strand_resistances(LO)
    xi_lim = (LO.takeup().T_t - LO.F_U.value / (LO.E.value - 1)) / (r_r * LO.L.value)
    assert 0.81 < xi_lim < 0.84          # 0.822 with the DIN split of the resistances
    assert np.isinf(ap.shortest_start(LO, xi_lim + 0.02))


def test_tight_side_is_impractical():
    """Su: a take-up between the head and the intermediate drive needs an order of magnitude
    more tension than one after the drive (Section 4.18: 8 to 10 times in the generic case)."""
    slack = ap.run(SU, SU.sigma_t[0], SU.t_a.value)
    tight = ap.run(SU, ap.alternative(SU), SU.t_a.value)
    assert ap.alternative(SU) < SU.sigma_d.value
    assert tight.req / slack.req > 8.0
    assert tight.exit_min < 0.0 < slack.exit_min


def test_fundamental_bounds_head_drive():
    """Head drive, beta small: 4 gamma L / c_r < T_1 <= 4 t_B L / c_r (Section 4.15), up to the
    small lengthening by the take-up mass."""
    for c in (LO, SM):
        cv = ap.conveyor(c, 0.5)
        t = cv.L / cv.c_r
        T1 = ap.fundamental(cv)
        assert 4 * cv.gamma * t < T1 <= 4 * (0.5 + cv.gamma) * t * 1.03


def test_carriage_never_governs():
    """Condition 2 is inactive in every application run (margin > 10, Section 4.18)."""
    d = ap.data()
    for tag in ap.TAGS:
        assert d[f"{tag}_accel_margin"].min() > 10.0


def test_quoted_values():
    """Numbers quoted in the caption of fig_applications (phase 5.5)."""
    d = ap.data()
    s, k = d["Lo_sigma"], int(np.argmin(np.abs(d["Lo_sigma"] - 0.001)))
    assert d["Lo_req_grip"][k] / 1e3 == pytest.approx(51.5, abs=0.3)
    kt = int(np.argmax(s))
    assert d["Lo_T1"][kt] / d["Lo_T1"][k] == pytest.approx(0.876, abs=0.005)       # -12 %
    ks = int(np.argmin(np.abs(d["SM_sigma"] - 0.1)))
    assert d["SM_req_slack"][ks] / 1e3 == pytest.approx(60.0, abs=0.5)
    assert d["SM_T1"][int(np.argmax(d["SM_sigma"]))] / d["SM_T1"][ks] == pytest.approx(0.80, abs=0.01)
    assert float(d["Lo_dur_min_ta"]) == pytest.approx(101.3, rel=0.02)
    sig = d["Su_sigma"]
    tight = sig < SU.sigma_d.value
    req = np.maximum.reduce([d["Su_req_slack"], d["Su_req_grip"], d["Su_req_sag"]])
    assert req[tight].min() / req[~tight].max() > 8.0


def test_same_counterweight_convention_is_immaterial_for_requirements():
    """Resizing the counterweight to keep the running T_2 changes neither the required T_t nor
    T_1 by more than 0.3 % (Lo, take-up at the tail)."""
    c, s = LO, 0.995
    st = bm.state(c, 1.0, s)
    T2r = bm.state(c, 1.0, c.sigma_t[0]).T_run[0]
    Tt = T2r - (st.T_run[0] - st.T_t)
    cr = replace(c, M_w=cs.P(2 * Tt / G), T_t=None, n=cs.P(2))
    a, b = ap.run(c, s, 30.0), ap.run(cr, s, 30.0)
    assert b.req == pytest.approx(a.req, rel=0.003)
    assert b.T1 == pytest.approx(a.T1, rel=0.003)
    assert np.isinf(ap.shortest_start(c, s)) and ap.shortest_start(cr, s) == pytest.approx(101, rel=0.02)
