"""Phase 5.6: LaTeX tables of the paper (maps/paper_tables.py).

Checks: provenance marks, the identity beta = (4/n) lam T_t / W_r, the fundamental periods
against independent routes (validation of Harrison, the dimensional start-ups of 5.5, the
intermediate-drive figure), beta / beta_5 against step 5.4, the take-up mass limit, and that
the committed .tex files are the generator's output.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "maps"))
sys.path.insert(0, str(ROOT / "validation"))
import cases as cs  # noqa: E402
import paper_tables as pt  # noqa: E402

ROWS = pt.case_rows()
BY = {r["tag"]: r for r in ROWS}
PAPER = ROOT / "maps" / "figures" / "paper"


def test_rows_complete_and_ordered():
    assert {r["tag"] for r in ROWS} == {c.tag for c in cs.CASES_FULL}
    Ls = [r["L"] for r in ROWS]
    assert Ls == sorted(Ls, reverse=True)
    assert all(r["basis"] for r in ROWS)


def test_provenance_rules():
    assert pt.weakest([cs.P(1.0), cs.C(1.0), cs.D(1.0)]) == "catalogue"
    assert pt.weakest([cs.P(1.0), cs.A(1.0), cs.C(1.0)]) == "assumed"
    assert pt.weakest([None, cs.Ms(1.0)]) == "measured"
    assert pt.mark([cs.Ms(1.0)]) == "" and "m" in pt.mark([cs.Ms(1.0)], direct=True)
    m = {r["tag"]: r["marks"] for r in ROWS}
    # measured gamma and c_r of Harrison; published T_t with assumed n (SM); assumed n (WR, LL)
    assert "m" in m["H"]["gamma"] and "m" in m["H"]["c_r"] and m["H"]["T1"] == ""
    assert m["SM"]["T_t"] == "" and "a" in m["SM"]["beta"] and "a" in m["SM"]["n"]
    assert "a" in m["WR"]["T_t"] and "a" in m["LL"]["T_t"]
    # catalogue idlers and belt (Si), catalogue EA (SM, Su); published case (Lo) unmarked
    assert "c" in m["Si"]["mu_r"] and "c" in m["Si"]["c_r"] and "c" in m["SM"]["c_r"]
    assert all(v == "" for k, v in m["Lo"].items())
    assert m["Pa"]["beta"] == ""                      # T_t derived, n published (Fig. 5)


def test_beta_identity():
    """beta = (4/n) lam T_t / W_r; lam = 1 for a direct counterweight (rules 0, 3, 4 and the
    rigging cases with i = 1), K = 1 for Sinaga (T_t and M_w, rule 2)."""
    for r in ROWS:
        if r["Tt_Wr"] is None:
            continue
        c = cs.BY_TAG[r["tag"]]
        tu = c.takeup()
        lam = cs.G_STD * tu.M_belt * tu.strands ** 2 / 4 / (tu.strands * tu.T_t)
        assert r["beta"] == pytest.approx(4 / tu.strands * lam * r["Tt_Wr"], rel=1e-9)
    assert BY["Si"]["beta"] == pytest.approx(BY["Si"]["Tt_Wr"], rel=2e-4)   # 4 T_t / (M_w g) = 1.0001
    assert BY["Lo"]["beta"] == pytest.approx(2 * BY["Lo"]["Tt_Wr"], rel=1e-9)
    assert BY["H"]["beta"] == pytest.approx(BY["H"]["Tt_Wr"], rel=1e-9)        # n = 4


def test_beta_ratio_matches_step_54():
    import beta_min as bm
    reg = {k: ratio for k, L, b, b5, ratio in bm.regime()}
    for t in ("Lo", "LL", "SM", "Pa", "Si", "Su", "H"):
        assert BY[t]["beta_ratio"][0] == pytest.approx(reg[t], rel=1e-9)
    assert BY["S"]["beta_ratio"][1] == pytest.approx(reg["S"], rel=1e-9)      # tail
    assert BY["WR"]["beta_ratio"][0] == pytest.approx(reg["WR"], rel=1e-9)    # xi = 0.05
    # the conveyors of 1 km or longer stay below 0.3 of their threshold (text of 5.4)
    long_ = [max(r["beta_ratio"]) for r in ROWS if r["L"] >= 1000 and r["beta_ratio"]]
    assert max(long_) < 0.3


def test_T1_harrison_equals_validation():
    """The case table gives the period of the validation section (transit basis)."""
    import harrison_case as hc
    T = hc.periods(79.0, gamma=hc.GAMMA_MEAS)[0]
    assert BY["H"]["T1"][0] == pytest.approx(T, rel=1e-9)
    assert BY["H"]["T1"][0] == pytest.approx(28.38, abs=0.01)
    lp, beta, cr, mur = hc.setup(79.0, gamma=hc.GAMMA_MEAS)
    assert cs.BY_TAG["H"].beta == pytest.approx(beta, rel=1e-9)
    assert cs.BY_TAG["H"].c_r == pytest.approx(cr, rel=1e-9)


def test_T1_matches_dimensional_startups():
    """Eigenvalue route (table of cases) against the Conveyor route of step 5.5."""
    app = {(r["tag"], r["label"]): r for r in pt.app_rows()}
    for t in ("Lo", "SM", "Su"):
        assert BY[t]["T1"][0] == pytest.approx(app[(t, "real")]["T1"], rel=2e-3)


def test_T1_nordell_ciozda_and_song():
    assert BY["NC"]["T1"][0] == pytest.approx(27.5, abs=0.05)           # fig_modal (d) caption
    head, tail = BY["S"]["T1"]
    assert head > tail                                                  # head > tail (4.12)
    assert head == pytest.approx(40.4, abs=0.05) and tail == pytest.approx(29.0, abs=0.05)


def test_takeup_mass_limit():
    """beta -> 0 gives the free-end period; the case's mass lengthens it by < 5 % for every
    case with beta / beta_5 < 1 (all of them)."""
    for c in cs.CASES_FULL:
        if c.beta is None:
            continue
        for s in pt.positions(c)[:1]:
            T0 = pt.T1_of(c, s, beta=1e-12)
            T = pt.T1_of(c, s)
            assert T >= T0
            assert T / T0 - 1 < 0.05
    c = cs.BY_TAG["Lo"]
    assert pt.T1_of(c, 0.001, beta=1e-9) == pytest.approx(pt.T1_of(c, 0.001, beta=0.0), rel=1e-6)


def test_note_figures():
    nt = pt.case_notes(ROWS)
    assert 0.02 < nt["SM_dT1"] < 0.03 and 0.01 < nt["Pa_dT1"] < 0.015
    assert nt["Su1"][0] == pytest.approx(23.5e3, abs=60)
    assert nt["Su1"][1] == pytest.approx(0.149, abs=5e-4)               # 5.4 table, design 1


def test_tex_files_are_generated():
    assert (PAPER / "table_cases.tex").read_text() == pt.table_cases_tex(ROWS)
    assert (PAPER / "table_applications.tex").read_text() == pt.table_applications_tex()


@pytest.mark.parametrize("name,cols", [("table_cases.tex", 13), ("table_applications.tex", 13)])
def test_tex_structure(name, cols):
    s = (PAPER / name).read_text()
    assert s.count("{") == s.count("}")
    body = s.split("\\midrule", 1)[1].split("\\bottomrule")[0]
    for line in body.splitlines():
        if line.endswith("\\\\") and "\\multicolumn" not in line:
            assert line.count("&") == cols - 1, line
    keys = set(re.findall(r"\\cite[tp](?:\[[^]]*\]\[[^]]*\])?\{([^}]*)\}", s))
    keys = {k for g in keys for k in g.split(",")}
    known = {c.bibkey for c in cs.CASES_FULL} | {"harrison1985b", "lodewijks2002",
                                                 "continental2021", "fennerdunlop2009", "cema"}
    assert keys and keys <= known


def test_application_cache_matches_recomputation():
    """The cached rows equal a fresh run of applications.real_alt_rows (~45 s)."""
    import applications as ap
    fresh = ap.real_alt_rows()
    cached = pt.app_rows()
    assert len(fresh) == len(cached)
    for a, b in zip(fresh, cached):
        assert (a["tag"], a["label"], a["gov"]) == (b["tag"], b["label"], b["gov"])
        for k in ("T1", "t_a", "entry", "exit_min", "travel", "req", "T2_req"):
            assert float(a[k]) == pytest.approx(b[k], rel=1e-9, abs=1e-6)
        assert (np.isfinite(a["t_a_req"]) == np.isfinite(b["t_a_req"]))
