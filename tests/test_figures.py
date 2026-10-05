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
