"""Phase 4.7: drive anywhere on the return strand (intermediate drive) and take-up anywhere.

With the drive on the return strand (head and tail included) and the take-up on the return
strand, one strand (C) contains the whole carry strand, with return segments l1 (next to the
drive) and l2 (next to the take-up); the other (R) is a uniform return segment of length
1 - l1 - l2. For a slack-side take-up with an intermediate drive, l1 = sigma_d (drive offset
from the head) and l2 = 1 - sigma_t. The fixed-free fundamental of C solves
    gamma tan(a) tan(c) + tan(a) tan(b) + tan(b) tan(c) / gamma = 1,
    a = Om l1, c = gamma Om, b = Om l2,
which reduces to tan(Om (1 - xi)) tan(gamma Om) = gamma for a head drive (l1 = 0) and to
a + b + c = pi/2 (four transits) for gamma = 1.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from beltloop import (Loop, Segment, StartProfile, fixed_free_roots, natural_frequencies,
                      startup_metrics, strand_curves, strand_participation, takeup_mass_approx)

GAMMAS = (1.0, 1.4, 2.0, 2.76, 3.0)


def _three_segment_root(l1, l2, gamma):
    """Independent reference: first root of the closed-form equation (bracketed on a grid
    below pi/(2 gamma), where the fundamental must lie)."""
    f = lambda o: (gamma * np.tan(o * l1) * np.tan(gamma * o) + np.tan(o * l1) * np.tan(o * l2)
                   + np.tan(o * l2) * np.tan(gamma * o) / gamma - 1.0)
    os = np.linspace(1e-9, np.pi / (2 * gamma) * (1 - 1e-12), 40001)
    v = f(os)
    i = np.flatnonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return brentq(f, os[i], os[i + 1], xtol=1e-15)


def _carry_chain(l1, l2, gamma):
    segs = [Segment(l1, 1.0)] if l1 > 0 else []
    segs.append(Segment(1.0, gamma, strand="carry"))
    if l2 > 0:
        segs.append(Segment(l2, 1.0))
    return tuple(segs)


def _split(sd, st, gamma):
    """Carry-containing strand C (fixed at the drive -> free at the take-up), the other
    strand R, and (l1, l2)."""
    lp = Loop.from_positions(sd, st, gamma)
    up, dn = lp.upstream, lp.downstream[::-1]
    has_carry = lambda ch: any(s.strand == "carry" for s in ch)
    C, R = (up, dn) if has_carry(up) else (dn, up)
    assert has_carry(C) and not has_carry(R)
    l1 = C[0].length if C[0].strand == "return" else 0.0
    l2 = C[-1].length if C[-1].strand == "return" else 0.0
    return lp, C, R, l1, l2


POSITIONS = [(sd, st) for sd in (0.0, 0.05, 0.2, 0.5, 0.9, 1.0) for st in (0.02, 0.3, 0.6, 0.97)
             if abs(sd - st) > 1e-9]


@pytest.mark.parametrize("gamma", GAMMAS)
def test_closed_form_equation(gamma):
    """Strand C's fundamental equals the root of the three-segment equation (both sides of
    the drive), and the equation has the head-drive and gamma = 1 limits."""
    for sd, st in POSITIONS:
        lp, C, R, l1, l2 = _split(sd, st, gamma)
        o = fixed_free_roots(C, 1)[0]
        assert o == pytest.approx(_three_segment_root(l1, l2, gamma), rel=1e-10)
    if gamma == 1.0:
        assert _three_segment_root(0.2, 0.3, 1.0) == pytest.approx(np.pi / (2 * 1.5), rel=1e-12)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_fundamental_belongs_to_carry_strand(gamma):
    """For gamma >= 1, drive and take-up anywhere on the return strand (slack or tight side,
    head or tail drive), the loop fundamental (beta -> 0) is the fixed-free fundamental of the
    strand that contains the carry strand: Om_C1 < pi/(2 gamma) <= pi/2 < pi/(2 l_R) = Om_R1
    (Rayleigh: prepending a fixed segment or appending a free one lowers the frequency)."""
    for sd, st in POSITIONS:
        lp, C, R, l1, l2 = _split(sd, st, gamma)
        oc, orr = fixed_free_roots(C, 1)[0], fixed_free_roots(R, 1)[0]
        lr = sum(s.length for s in R)
        assert orr == pytest.approx(np.pi / (2 * lr), rel=1e-12)
        assert oc <= np.pi / (2 * gamma) * (1 + 1e-12) < orr
        assert natural_frequencies(lp, 0.0, 1)[0] == pytest.approx(oc, rel=1e-13)


def test_period_bounds():
    """4 gamma <= T_1 <= 4 t + 4 (gamma - 1) l1, with t = gamma + l1 + l2 (heavier segment l1
    gives the head-drive bound); numerically 0.838 <= T_1 / (4 t) <= 1.39 for 1 <= gamma <= 3,
    and <= 1.13 with the drive within 0.2 L of the head (l1 <= 0.2)."""
    lo, hi, hi02 = np.inf, 0.0, 0.0
    for g in np.linspace(1, 3, 17):
        for l1 in np.linspace(0, 0.99, 34):
            for l2 in np.linspace(0, 0.99 - l1, 12):
                T = 2 * np.pi / fixed_free_roots(_carry_chain(l1, l2, g), 1)[0]
                t = g + l1 + l2
                assert 4 * g * (1 - 1e-12) <= T <= (4 * t + 4 * (g - 1) * l1) * (1 + 1e-12)
                lo, hi = min(lo, T / (4 * t)), max(hi, T / (4 * t))
                if l1 <= 0.2:
                    hi02 = max(hi02, T / (4 * t))
    assert 0.837 < lo < 0.842
    assert 1.30 < hi < 1.39
    assert 1.10 < hi02 < 1.13


def test_participation_bounds():
    """Effective mass of the fundamental: m_eff <= m_C (strand mass; the strand's modes add up
    to it) and m_eff / ((8/pi^2) m_C) in [0.97, 1.17], <= 1.11 for l1 <= 0.2. The return
    segment l1 at the fixed end acts as a soft spring that concentrates the strand's mass in
    the fundamental."""
    lo, hi, hi02 = np.inf, 0.0, 0.0
    for g in np.linspace(1, 3, 9):
        for l1 in np.linspace(0, 0.98, 15):
            for l2 in np.linspace(0, 0.98 - l1, 8):
                ch = _carry_chain(l1, l2, g)
                me = strand_participation(ch, 1)[1][0]
                m = g * g + l1 + l2
                assert me <= m
                r = me / (8 / np.pi ** 2 * m)
                lo, hi = min(lo, r), max(hi, r)
                if l1 <= 0.2:
                    hi02 = max(hi02, r)
    assert 0.969 < lo < 0.975
    assert hi < 1.17
    assert hi02 < 1.11


def test_reversal_maps_slack_onto_tight_side():
    """Reversing belt travel turns a slack-side take-up into a tight-side one with the same
    spectrum (the modal results do not depend on which side of the drive the take-up is)."""
    lp = Loop.from_positions(0.15, 0.4, 2.2)
    for beta in (0.0, 0.1):
        assert np.allclose(natural_frequencies(lp, beta, 5),
                           natural_frequencies(lp.reversed(), beta, 5), rtol=1e-12)


@pytest.mark.parametrize("gamma,sd", [(1.0, 0.5), (2.0, 0.1), (3.0, 0.3), (2.0, 0.7)])
def test_two_pole_correction_intermediate(gamma, sd):
    """The take-up mass correction keeps its head-drive accuracy (<= 0.4 % at beta = 0.1)."""
    for st in np.linspace(0.04, 0.96, 12):
        if abs(st - sd) < 0.02:
            continue
        lp = Loop.from_positions(sd, st, gamma)
        ex = natural_frequencies(lp, 0.1, 3)
        assert np.max(np.abs(takeup_mass_approx(lp, 0.1, 3) / ex - 1)) < 4.5e-3


@pytest.fixture(scope="module")
def universal():
    r = np.array([1.0, 2.0])
    De, Df = strand_curves(r)
    return r, De, Df


@pytest.mark.parametrize("gamma", [2.0, 3.0])
@pytest.mark.parametrize("sd", [0.1, 0.2])
def test_startup_collapse_intermediate(universal, gamma, sd):
    """Slack-side take-up right after an intermediate drive: the drive-exit minimum is exact
    (strand A is a uniform return segment) and the entry peak follows the universal curve with
    T_1 (strand B) and the mass of strand B, slightly above it: <= +4.5 % for tau_a = T_1 and
    <= +2.5 % for tau_a = 2 T_1 with the drive within 0.2 L of the head."""
    r, De, _ = universal
    xA = 0.05
    lp = Loop.from_positions(sd, sd + xA, gamma)
    mB = sum(s.mu * s.length for s in lp.downstream)
    TB = 2 * np.pi / fixed_free_roots(lp.downstream[::-1], 1)[0]
    for ri, Di, tol in zip(r, De, (0.045, 0.025)):
        m = startup_metrics(lp, 1e-8, StartProfile("sine", ri * TB, "none"))
        assert m.T1 == pytest.approx(TB, rel=1e-6)
        dev = m.entry_peak / (Di * mB) - 1
        assert -0.005 < dev < tol
        assert m.exit_qs == pytest.approx(-xA, rel=1e-9)


def test_takeup_at_pulleys_and_on_carry_strand():
    """from_positions accepts the take-up at the tail (sigma_t = 1) or head pulley and on the
    carry strand; the pulley positions are the limits of the return-strand ones, and a carry
    take-up equals the explicit chain."""
    for st, near in ((1.0, 1.0 - 1e-10), (0.0, 1e-10)):
        a = natural_frequencies(Loop.from_positions(0.3, st, 2.0), 0.05, 4)
        b = natural_frequencies(Loop.from_positions(0.3, near, 2.0), 0.05, 4)
        assert np.allclose(a, b, rtol=1e-8)
    lp = Loop.from_positions(1.0, 1.3, 1.8)          # tail drive, take-up on the carry strand
    ref = Loop((Segment(0.3, 1.8, strand="carry"),),
               (Segment(0.7, 1.8, strand="carry"), Segment(1.0, 1.0)))
    assert np.allclose(natural_frequencies(lp, 0.1, 4), natural_frequencies(ref, 0.1, 4),
                       rtol=1e-12)
    with pytest.raises(ValueError):
        Loop.from_positions(0.3, 2.0, 2.0)
    with pytest.raises(ValueError):
        Loop.from_positions(0.3, 0.3, 2.0)
