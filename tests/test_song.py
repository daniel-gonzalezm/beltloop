"""Phase 3.6 checks: Song, Wang & Zuo (2012, J. China Coal Soc. 37(S1), 217-223).

Song et al. neglect the take-up inertia and move the take-up to the tail, so the loop
splits into two independent fixed-free strands (their Eqs. 1-8). That is the limit
beta -> 0, xi -> 1 of the closed-form equation. Their conveyor, however, has the take-up
next to the head drive (their Fig. 1), and there the spectrum is different."""
import numpy as np
import pytest

from beltloop import Loop, natural_frequencies


@pytest.mark.parametrize("gamma", [1.3, 1.666, 2.5])
def test_tail_takeup_massless_splits_into_two_quarter_wave_families(gamma):
    """beta -> 0 with the take-up at the tail: return strand cos(Om) = 0 and carrying
    strand cos(gamma Om) = 0, each fixed at the drive and free at the take-up."""
    lp = Loop.from_positions(0.0, 1 - 1e-7, gamma)
    Om = natural_frequencies(lp, 1e-7, 6)
    k = np.arange(1, 7)
    expected = np.sort(np.concatenate([(2 * k - 1) * np.pi / 2,
                                       (2 * k - 1) * np.pi / (2 * gamma)]))[:6]
    np.testing.assert_allclose(Om, expected, rtol=1e-5)


def test_song_relocation_is_not_equivalent():
    """Song et al.'s example (7117 m, 104 MN, belt 27 kg/m, coal 66.7 kg/m, idlers 11.25 and
    10.8 kg/m, 4500 kg take-up): fundamental period with the take-up at the tail
    (the model they solve) versus next to the head drive (their Fig. 1)."""
    EA, L = 104e6, 7117.0
    m_carry, m_ret = 27 + 600 / 3.6 / 2.5 + 11.25, 27 + 10.8
    c_r, c_c = np.sqrt(EA / m_ret), np.sqrt(EA / m_carry)
    gamma, beta = c_r / c_c, 4500 / (m_ret * L)
    T = lambda xi: 2 * np.pi * L / (c_r * natural_frequencies(Loop.from_positions(0.0, xi, gamma), beta, 1)[0])
    assert T(1 - 1e-3) == pytest.approx(4 * L / c_c, rel=5e-3)   # carrying-strand quarter wave
    assert T(1e-3) == pytest.approx(40.0, abs=0.1)               # take-up at the head
    assert T(1e-3) / T(1 - 1e-3) > 1.35


def test_song_case_summary():
    """Step 6.4 cross-comparison (validation/song_case.py): wave speeds and running drive force
    from their data, take-up tension their counterweight can hold, and the take-up travel that
    their own running tensions imply (half the change of loop elongation)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / "validation"))
    import song_case as sc
    s = sc.summary()
    assert s["c_c"] == pytest.approx(995.6, abs=0.1)
    assert s["c_r"] == pytest.approx(1658.7, abs=0.1)
    assert s["F_run"] / s["pub"]["F_run"] == pytest.approx(1.0, abs=0.015)
    assert s["takeup_mass_tension"] < 0.15 * s["pub"]["T_t"]
    # their tensions and the resistances give the same travel (within 2 %), far from 2.81 m
    assert s["travel_from_their_tensions"] == pytest.approx(s["travel_model"], rel=0.02)
    assert s["travel_from_their_tensions"] < 0.7 * s["pub"]["travel"]
