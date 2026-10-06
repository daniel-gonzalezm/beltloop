"""Phase 4.8: take-up mass threshold per strand mode and the numbers quoted in the captions of the
paper figures (maps/paper_figures.py). The figures read cached data; these tests recompute the
quoted values from the package, so text, figures and code cannot drift apart."""
import numpy as np
import pytest
from scipy.optimize import brentq

from beltloop import (Loop, natural_frequencies, poles, strand_curves, strand_modes,
                      strand_participation, takeup_mass_threshold)

TOL = 0.05


def threshold_by_roots(lp, strand, j, tol=TOL):
    """Reference: bisection on the exact root born at the strand pole."""
    chain = lp.upstream if strand == "A" else lp.downstream[::-1]
    Om_p = strand_modes(chain, j)[0][j - 1]
    P = poles(lp, 3 * j + 8)
    k = int(np.argmin(abs(P - Om_p)))
    f = lambda lb: Om_p / natural_frequencies(lp, 10.0 ** lb, k + 1)[k] - 1 - tol
    return 10.0 ** brentq(f, -6, 3, xtol=1e-12)


@pytest.mark.parametrize("xi,gamma,strand,j", [(0.3, 1.0, "B", 1), (0.3, 2.0, "B", 1),
                                               (0.1, 1.95, "B", 2), (0.05, 2.93, "B", 2),
                                               (0.4, 1.0, "B", 2), (0.7, 1.5, "A", 1),
                                               (0.9, 2.0, "A", 1)])
def test_threshold_matches_root_search(xi, gamma, strand, j):
    lp = Loop.from_positions(0.0, xi, gamma)
    assert takeup_mass_threshold(lp, strand, j) == pytest.approx(
        threshold_by_roots(lp, strand, j), rel=1e-8)


@pytest.mark.parametrize("xi,gamma", [(0.2, 1.0), (0.5, 2.0), (0.9, 3.0)])
def test_threshold_puts_the_root_at_the_target(xi, gamma):
    """With beta = beta_5 the root born at B1 sits exactly at Om_p / 1.05."""
    lp = Loop.from_positions(0.0, xi, gamma)
    b5 = takeup_mass_threshold(lp, "B", 1)
    Om_p = natural_frequencies(lp, 0.0, 1)[0]
    assert natural_frequencies(lp, b5, 1)[0] == pytest.approx(Om_p / 1.05, rel=1e-11)


@pytest.mark.parametrize("xi", [0.1, 0.4, 0.8])
def test_uniform_head_drive_closed_form(xi):
    """gamma = 1: both strands uniform, W/W' = tan(Om l)/Om, so
    beta_5 = 4 / (Om_t (tan(Om_t xi) + tan(Om_t (2 - xi)))), Om_t = pi / (2 (2 - xi) 1.05)."""
    lp = Loop.from_positions(0.0, xi, 1.0)
    o = np.pi / (2 * (2 - xi)) / 1.05
    ref = 4 / (o * (np.tan(o * xi) + np.tan(o * (2 - xi))))
    assert takeup_mass_threshold(lp, "B", 1) == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("xi,gamma,strand", [(0.3, 2.0, "B"), (0.8, 1.5, "A"), (0.05, 2.5, "B")])
def test_single_pole_limit(xi, gamma, strand):
    """As tol -> 0 the threshold tends to the single-pole value 4 m_tilde ((1 + tol)^2 - 1)."""
    lp = Loop.from_positions(0.0, xi, gamma)
    chain = lp.upstream if strand == "A" else lp.downstream[::-1]
    mt = strand_modes(chain, 1)[1][0]
    for tol, err in ((1e-3, 2e-2), (1e-4, 2e-3)):
        ex = takeup_mass_threshold(lp, strand, 1, tol)
        assert ex / (4 * mt * ((1 + tol) ** 2 - 1)) == pytest.approx(1.0, abs=err)


def test_coincident_poles_li_pang():
    """Take-up at the tail, gamma = 1: A1 = B1 = pi/2. The lower root carries the shift of both
    strands, Om tan Om = 2/beta (Li and Pang 2018); the other is inside the veering band."""
    lp = Loop.from_positions(0.0, 1 - 1e-9, 1.0)
    o = np.pi / 2 / 1.05
    assert takeup_mass_threshold(lp, "B", 1) == pytest.approx(2 / (o * np.tan(o)), rel=1e-6)
    assert np.isnan(takeup_mass_threshold(lp, "A", 1))


def test_not_reached_and_veering_zones_of_B2():
    """Right of the B2-A1 crossing the root born at B2 cannot drop 5 % even for beta -> infinity
    (inf); at the crossing the shift is not a property of one strand (nan)."""
    assert np.isinf(takeup_mass_threshold(Loop.from_positions(0.0, 0.8, 2.0), "B", 2))
    g = 1.0                                 # xi_s = 0.5: A1 = pi/(2 xi) meets B2 = 3 pi/(2 (2 - xi))
    lp = Loop.from_positions(0.0, 0.501, g)
    assert np.isnan(takeup_mass_threshold(lp, "B", 2))


def test_B1_is_loop_mode_1():
    """For gamma >= 1 the fundamental is B1, so its threshold is the loop's mode-1 threshold of
    phase 4.2 (bisection on the first root)."""
    for xi, g in ((0.05, 1.2), (0.5, 2.5), (0.95, 1.8)):
        lp = Loop.from_positions(0.0, xi, g)
        f = lambda lb: natural_frequencies(lp, 0.0, 1)[0] / natural_frequencies(lp, 10 ** lb, 1)[0] - 1.05
        assert takeup_mass_threshold(lp, "B", 1) == pytest.approx(10 ** brentq(f, -4, 2, xtol=1e-12), rel=1e-8)


def test_A1_backbone():
    """Strand A is uniform: away from veering, beta_5 of A1 is close to 0.205 xi (the quoted
    range of the exact value is -25 % to +0 % plus spikes next to the veering bands)."""
    lp = Loop.from_positions(0.0, 0.999, 1.85)            # Pascual et al. (2005) geometry
    b5 = takeup_mass_threshold(lp, "A", 1)
    assert 0.75 * 0.205 * 0.999 < b5 < 1.0 * 0.205 * 0.999


# ------------------------------------------------------------------ numbers quoted in captions
def test_universal_curve_quoted_values():
    r = np.linspace(0.8, 0.95, 31)
    D, _ = strand_curves(r)
    assert D.max() == pytest.approx(1.590, abs=1e-3)
    assert 0.85 <= r[np.argmax(D)] <= 0.88
    D, Dfree = strand_curves(np.array([1.0, 2.0, 5.0, 10.0]))
    assert D == pytest.approx([1.574, 1.206, 1.077, 1.038], abs=1.5e-3)


def test_intermediate_drive_quoted_values():
    """Take-up right after the drive (xi = 0.01): T_1 / (4 t_B) at l1 = 0.5 and Nordell and
    Ciozda's geometry (27.5 s against 19.8 s with a head drive)."""
    def ratio(sd, st, g):
        B = Loop.from_positions(sd, st, g).downstream[::-1]
        Om = strand_participation(B, 1)[0][0]
        return 2 * np.pi / Om / (4 * sum(s.g * s.length for s in B)), 2 * np.pi / Om
    assert ratio(0.5, 0.51, 2.0)[0] == pytest.approx(1.068, abs=2e-3)
    assert ratio(0.5, 0.51, 3.0)[0] == pytest.approx(1.143, abs=2e-3)
    ft = 0.3048
    s = 8150 * ft / 1450.0
    g = 1450.0 / 590.0
    T_int = ratio(5150 / 8150, 5450 / 8150, g)[1] * s
    T_head = ratio(0.0, 300 / 8150, g)[1] * s
    assert T_int == pytest.approx(27.5, abs=0.05)
    assert T_head == pytest.approx(19.8, abs=0.05)


def test_b2_region_edges_uniform():
    """gamma = 1, closed forms: xi_s from pi/(2 xi) = 3 pi/(2 (2 - xi)) -> 1/2; band edge from
    3 xi = 1.05 (2 - xi) -> 2.1/4.05; F(Om_t) = (tan(Om_t xi) + tan(Om_t (2 - xi)))/Om_t = 0
    first at 2 Om_t = 2 pi -> 2 - xi = 3/2.1, xi = 4/7."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "maps"))
    from paper_figures import b2_region_edges
    x1, x2, x3 = b2_region_edges(1.0)
    assert (x1, x2, x3) == pytest.approx((0.5, 2.1 / 4.05, 4 / 7), abs=1e-9)
    # every point strictly inside each region is classified as such by the threshold function
    for xi, kind in (((x1 + x2) / 2, "nan"), ((x2 + x3) / 2, "inf")):
        b = takeup_mass_threshold(Loop.from_positions(0.0, xi, 1.0), "B", 2)
        assert (np.isnan(b) if kind == "nan" else np.isinf(b))
    x1, x2, x3 = b2_region_edges(2.0)
    for xi, kind in (((x1 + x2) / 2, "nan"), ((x2 + x3) / 2, "inf"), (x3 + 0.01, "finite")):
        b = takeup_mass_threshold(Loop.from_positions(0.0, xi, 2.0), "B", 2)
        assert {"nan": np.isnan, "inf": np.isinf, "finite": np.isfinite}[kind](b)


# ------------------------------------------------------------------ phase 5.3(d): case markers
def _pf():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "maps"))
    import paper_figures
    return paper_figures


def test_gamma_range_bars():
    """Coupling bar of every map case: full coupling on top, lowest coupling of the class at the
    bottom; no bar when gamma is given directly (H)."""
    import cases as cs
    for tag, beta, g, lo, hi, xi, unk in cs.points_alpha():
        c = cs.BY_TAG["S" if tag == "St" else tag]
        if c.gamma_given is not None:
            assert lo == hi == g
            continue
        assert hi == pytest.approx(g, rel=1e-12)
        assert lo == pytest.approx(c.gamma(c.alpha.lo), rel=1e-12) and lo < g
    assert cs.BY_TAG["SM"].gamma_range()[0] == pytest.approx(1.284, abs=2e-3)   # alpha = 0.3
    assert cs.BY_TAG["Pa"].gamma_range()[0] == pytest.approx(1.304, abs=2e-3)
    assert cs.BY_TAG["WR"].gamma_range()[0] == pytest.approx(2.212, abs=2e-3)   # alpha = 0.8


def test_case_lanes_do_not_overlap():
    """Display offsets: bars sharing a lane are separated; offsets stay small; leader labels
    keep their minimum gap."""
    pf = _pf()
    marks = [m for m in pf.case_marks() if not m["unknown"]]
    for m in marks:
        if m["xi"] <= pf.CLUSTER_XI:
            assert abs(m["x"] - m["xi"]) <= 0.06
    lanes = {}
    for m in marks:
        if m["xi"] <= pf.CLUSTER_XI:
            lanes.setdefault(round(m["x"], 6), []).append(m)
    for lane in lanes.values():
        for a in lane:
            for b in lane:
                if a is not b:
                    assert a["gamma"] + pf.LANE_PAD < b["g_lo"] or a["g_lo"] - pf.LANE_PAD > b["gamma"]
    ys = pf.leader_positions([m["gamma"] for m in marks if m["xi"] <= pf.LEADER_XI])
    ys = np.sort(ys)
    assert np.all(np.diff(ys) >= pf.LEADER_GAP - 1e-12) and ys[0] >= 1.0 and ys[-1] <= 3.0


def test_practice_ratios_quoted():
    """tau_a / T_1 of the cases with a published start time (fig_startup, dashed lines)."""
    pf = _pf()
    got = dict(pf.practice())
    want = {"Lo": 1.05, "LL": 1.49, "G": 2.43, "Su": 4.11, "S": 7.43, "Si": 11.81}
    assert set(got) == set(want)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, abs=0.006)


def test_case_margins_quoted():
    """fig_beta (a) caption: conveyors of 1 km or longer at least 3.7 (2.8) times below their
    B1 threshold, fundamental +1.2 % (+1.8 %) at most; WR at 0.6 of it, 0.95-1.02 at alpha 0.8."""
    pf = _pf()
    rows = pf.case_margins()
    long_ = [r for r in rows if r[1] >= 1000.0]
    assert max(r[2] for r in long_) <= 0.21
    assert 1 / max(r[3] for r in long_) == pytest.approx(3.7, abs=0.05)
    assert 1 / max(r[4] for r in long_) == pytest.approx(2.8, abs=0.05)
    assert max(r[5] for r in long_) == pytest.approx(1.23, abs=0.01)
    assert max(r[6] for r in long_) == pytest.approx(1.78, abs=0.01)
    wr = [r for r in rows if r[0] == "WR"]
    assert all(0.55 <= r[3] <= 0.59 for r in wr)
    assert min(r[4] for r in wr) == pytest.approx(0.95, abs=0.01)
    assert max(r[4] for r in wr) == pytest.approx(1.02, abs=0.01)
    assert min(r[6] for r in wr) == pytest.approx(4.74, abs=0.02)
    assert max(r[6] for r in wr) == pytest.approx(5.12, abs=0.02)


def test_sasol_panel_quoted():
    """fig_modal (d): Su at 0.952 with beta -> 0 (0.939 at alpha = 0.8); its beta adds 2.1 %."""
    import cases as cs
    c = cs.BY_TAG["Su"]
    out = []
    for g in (c.gamma(), c.gamma_range()[0]):
        B = Loop.from_positions(c.sigma_d.value, c.sigma_t[0], g).downstream[::-1]
        Om = strand_participation(B, 1)[0][0]
        out.append(2 * np.pi / Om / (4 * sum(s.g * s.length for s in B)))
    assert out == pytest.approx([0.952, 0.939], abs=1e-3)
    lp = Loop.from_positions(c.sigma_d.value, c.sigma_t[0], c.gamma())
    shift = natural_frequencies(lp, 0.0, 1)[0] / natural_frequencies(lp, c.beta, 1)[0] - 1
    assert shift == pytest.approx(0.021, abs=1e-3)


def test_su_margin_for_text():
    """Su, the second short conveyor (text, phase 5.3(d)): beta / beta_5(B1) = 0.42 (0.66 at
    alpha = 0.8); fundamental +2.1 % (+3.3 %)."""
    import cases as cs
    c = cs.BY_TAG["Su"]
    for g, ratio, shift in ((c.gamma(), 0.42, 2.06), (c.gamma_range()[0], 0.66, 3.28)):
        lp = Loop.from_positions(c.sigma_d.value, c.sigma_t[0], g)
        assert c.beta / takeup_mass_threshold(lp, "B", 1) == pytest.approx(ratio, abs=0.005)
        s = natural_frequencies(lp, 0.0, 1)[0] / natural_frequencies(lp, c.beta, 1)[0] - 1
        assert 100 * s == pytest.approx(shift, abs=0.01)


def test_unknown_position_off_the_maps():
    pf = _pf()
    assert not pf.SHOW_UNKNOWN_POSITION
    assert all(not m["unknown"] for m in pf.case_marks())
    assert any(r[0] == "WR" for r in pf.case_margins())       # still quoted in the text


def test_a1_branches_reach_their_asymptotes():
    """fig_beta (c): every finite branch followed by a 'not reached' interval rises past the top
    of the panel (beta_5 = 1), and the class edges are located to within 1e-9 in xi."""
    pf = _pf()
    d = pf.data_a1(2.0, n=600)
    iv = d["iv"]
    branches = np.split(d["y"], np.where(np.isnan(d["y"]))[0])
    branches = [b[np.isfinite(b)] for b in branches if np.isfinite(b).any()]
    fin = [k for k in range(len(iv)) if iv[k, 0] == 0]
    assert len(branches) == len(fin)
    for b, k in zip(branches, fin):
        if k + 1 < len(iv) and iv[k + 1, 0] == 1:
            assert b[-1] > 1.0
    for k in range(len(iv) - 1):
        assert 0 < iv[k + 1, 1] - iv[k, 2] < 1e-9


def test_a1_undefined_strip_and_locked_loop():
    """fig_beta (c): the merged strip starts at each asymptote and ends where the next branch
    starts; in the 'not reached' gaps the root born at A1 cannot drop below 0.952 of its pole
    even for a huge take-up mass (locked take-up), and drops below it in a branch."""
    pf = _pf()
    from beltloop import fixed_free_roots
    d = pf.data_a1(2.0, n=600)
    iv = d["iv"]
    strip = pf.undefined_intervals(iv)
    finite = [(x0, x1) for c, x0, x1 in iv if c == 0]
    for (a, w), (f0, f1) in zip(strip[1:], finite[:-1]):
        assert a == pytest.approx(f1, abs=1e-9)
    assert all(w > 0 for a, w in strip)

    def lowest_ratio(x):
        lp = Loop.from_positions(0.0, x, 2.0)
        Op = fixed_free_roots(lp.upstream, 1)[-1]
        pB = fixed_free_roots(lp.downstream[::-1], 60)
        W = natural_frequencies(lp, 1e9, 80)
        r = W[(W > pB[pB < Op].max()) & (W < Op)]
        return r.min() / Op
    for c, x0, x1 in iv:
        if 0.2 < x0 and c in (0, 1):
            ratio = lowest_ratio(0.5 * (x0 + x1))
            assert (ratio > 1 / 1.05) if c == 1 else (ratio < 1 / 1.05)


def test_a1_band_right_edges():
    """fig_beta (c): each veering gap ends where the pole of the B mode passes the pole of A1;
    the next branch starts there at a finite value below the backbone (no asymptote)."""
    pf = _pf()
    from beltloop import fixed_free_roots, takeup_mass_threshold
    d = pf.data_a1(2.0, n=600)
    for c, x0, x1 in d["iv"]:
        if c != 2 or x0 < 0.2:
            continue
        lp = Loop.from_positions(0.0, x1, 2.0)
        Op = fixed_free_roots(lp.upstream, 1)[-1]
        pB = fixed_free_roots(lp.downstream[::-1], 60)
        assert np.min(abs(pB / Op - 1)) < 1e-7
        b = takeup_mass_threshold(Loop.from_positions(0.0, x1 + 1e-8, 2.0), "A", 1)
        assert 0.73 * 0.205 * x1 < b < 0.205 * x1          # up to 26 % below


def test_a1_reach_quoted():
    """Notes of captions.md: reach of fig_beta (c) (definition, damping, excitation)."""
    pf = _pf()
    import cases as cs
    iv = pf.data_a1(2.0, n=3000)["iv"]
    assert min(x0 for c, x0, x1 in iv if c == 0) == pytest.approx(0.065, abs=5e-4)
    for lim, frac in ((0.1, 0.04), (0.2, 0.29)):
        dfn = sum(max(0, min(x1, lim) - x0) for c, x0, x1 in iv if c == 0 and x0 < lim)
        assert dfn / lim == pytest.approx(frac, abs=0.006)
    # zeta_A1 / zeta_1 = T_1 / T_A1 (stiffness-proportional damping: zeta_k = zeta_hat Om_k)
    lp = Loop.from_positions(0.0, 0.3, 2.0)
    O1 = natural_frequencies(lp, 0.0, 1)[0]
    T1 = 2 * np.pi / O1
    assert (np.pi / (2 * 0.3)) / O1 == pytest.approx(T1 / (4 * 0.3), rel=1e-12)
    want = {"Lo": 1.11, "LL": 1.27, "G": 2.19, "Su": 4.76, "S": 5.83, "Si": 10.8}
    for t, x in want.items():
        c = cs.BY_TAG[t]
        assert c.t_a.value * c.c_r / (12 * c.L.value) == pytest.approx(x, abs=0.01 * max(1, x))
