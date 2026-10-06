"""Phase 5.6: LaTeX tables of the paper, generated from the single sources of the figures.

    python maps/paper_tables.py              # writes figures/paper/table_cases.tex and
                                             # figures/paper/table_applications.tex
    python maps/paper_tables.py --recompute  # reruns the application start-ups (~45 s)

Table of cases (`table_cases.tex`), from `cases.py` and `beta_min.py`: geometry, densities,
wave speeds, fundamental period, take-up tension and mass, and what set the take-up tension
according to each source (phase 5.4). Each value carries the weakest provenance among its
inputs (`Datum.kind`): c catalogue, a assumed, m measured (only for a value that is itself a
measurement); published values and values derived from published data are unmarked. The
counterweight is taken as hung directly on the n strands unless the rigging is published (i
is not marked; stated once in the notes).

Table of applications (`table_applications.tex`), from `applications.real_alt_rows` (phase
5.5): the configurations of the last row of `fig_applications`, plus Lo with the take-up at
the tail. The rows are cached in `data/table_applications.json`.

The tables use array, booktabs, threeparttable and natbib (\\citet); keys as in the project .bib files.
`figures/paper/tables_preview.tex` compiles both on their own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import cases as cs

HERE = Path(__file__).parent
OUT = HERE / "figures" / "paper"
DATA = HERE / "data"

RANK = {"published": 0, "derived": 0, "measured": 0, "catalogue": 1, "assumed": 2}
MARK = {"catalogue": "c", "assumed": "a", "measured": "m"}

# Case-specific remarks for the table notes (numbers that are computed are filled in by
# `case_notes`; the rest come from the source notes in cases.py).
WR_XI = (0.05, 0.999)          # range of take-up positions shown for WR (position unknown)


# ------------------------------------------------------------------------------ provenance
def weakest(data) -> str:
    """Weakest provenance among the given Datum objects (None entries are skipped)."""
    kinds = [d.kind for d in data if d is not None]
    if not kinds:
        return "published"
    return max(kinds, key=lambda k: RANK[k])


def mark(data, direct: bool = False) -> str:
    """LaTeX superscript for the weakest provenance; 'm' only for a direct measurement."""
    k = weakest(data)
    if k == "measured" and not direct:
        return ""
    return "" if k not in MARK else f"$^{{\\mathrm{{{MARK[k]}}}}}$"


def _assumed(note="") -> cs.Datum:
    return cs.A(0.0, note)


def mu_r_inputs(c: cs.Case):
    return [c.mu_r_given] if c.mu_r_given is not None else [c.m_b, c.m_ir]


def gamma_inputs(c: cs.Case):
    return [c.gamma_given] if c.gamma_given is not None else [c.m_b, c.m_ic, c.m_l, c.m_ir]


def c_r_inputs(c: cs.Case):
    return [c.c_r_given] if c.c_r_given is not None else [c.EA] + mu_r_inputs(c)


def n_inputs(c: cs.Case):
    """The number of strands used for beta: published, or assumed by the take-up rule."""
    rule = c.takeup_rule()[0]
    if "n = 2 assumed" in rule:
        return [_assumed("n = 2")]
    return [c.n]


def takeup_inputs(c: cs.Case):
    """Inputs of the take-up tension and mass, following the rule of `Case.takeup_rule`."""
    rule = c.takeup_rule()[0]
    if rule.startswith("T_t and n"):
        return [c.T_t, c.n]
    if rule == "rigging":
        return [c.M_w, c.n]
    if rule == "T_t and M_w":
        return [c.T_t, c.M_w]
    if rule.startswith("M_w only"):
        return [c.M_w] + n_inputs(c)
    if rule.startswith("T_t only"):
        return [c.T_t] + n_inputs(c)
    return []


def T_t_inputs(c: cs.Case):
    return [c.T_t] if c.T_t is not None else takeup_inputs(c)


# ---------------------------------------------------------------------------- case table
def _loop(c: cs.Case, sigma_t: float):
    from beltloop import Loop
    return Loop.from_positions(c.sigma_d.value, sigma_t, c.gamma())


def T1_of(c: cs.Case, sigma_t: float, beta: float | None = None) -> float:
    """Fundamental period (s), exact, with the case's beta (beta -> 0 if unknown)."""
    from beltloop import natural_frequencies
    b = (c.beta or 0.0) if beta is None else beta
    return float(2 * np.pi / natural_frequencies(_loop(c, sigma_t), b, 1)[0] * c.L.value / c.c_r)


def beta5_of(c: cs.Case, sigma_t: float) -> float:
    from beltloop import takeup_mass_threshold
    return float(takeup_mass_threshold(_loop(c, sigma_t), "B", 1))


def positions(c: cs.Case):
    """Take-up positions shown in the table (geometric, units of L)."""
    if c.tag == cs.UNKNOWN_POSITION:
        return list(np.linspace(*WR_XI, 41))
    return list(c.sigma_t)


def case_row(c: cs.Case) -> dict:
    """Numbers and provenance marks of one row of the case table."""
    sts = positions(c)
    xis = [(s - c.sigma_d.value) % 2.0 for s in sts]
    T1 = [T1_of(c, s) for s in sts]
    tu = c.takeup()
    known_Tt = c.tag != "G" and tu is not None          # G: number of counterweights unknown
    T_t = tu.T_t if known_Tt else None
    W_r = cs.G_STD * c.mu_r * c.L.value if c.mu_r_given is None or c.mu_r_given.kind != "assumed" \
        else None
    b = c.beta
    ratio = [b / beta5_of(c, s) for s in sts] if b is not None else None
    rg = c.gamma_range()
    n = None if tu is None else tu.strands
    r = dict(tag=c.tag, bibkey=c.bibkey, L=c.L.value, sigma_d=c.sigma_d.value, xi=xis,
             mu_r=None if W_r is None else c.mu_r, gamma=c.gamma(),
             gamma_lo=None if rg is None else rg[0], c_r=c.c_r, T1=T1,
             T_t=T_t, n=n if known_Tt else None,
             Tt_Wr=None if (T_t is None or W_r is None) else T_t / W_r,
             beta=b, beta_ratio=ratio, basis=c.basis_short)
    tk = takeup_inputs(c)
    r["marks"] = dict(
        L=mark([c.L]), mu_r=mark(mu_r_inputs(c)), gamma=mark(gamma_inputs(c), direct=True),
        c_r=mark(c_r_inputs(c), direct=True),
        T1=mark(c_r_inputs(c) + gamma_inputs(c) + tk),
        T_t=mark(T_t_inputs(c)), n=mark(n_inputs(c)),
        Tt_Wr=mark(T_t_inputs(c) + mu_r_inputs(c) + [c.L]),
        beta=mark(tk + mu_r_inputs(c) + [c.L]),
        beta_ratio=mark(tk + mu_r_inputs(c) + [c.L] + gamma_inputs(c)))
    return r


def case_rows():
    """All cases, by decreasing length."""
    rows = [case_row(c) for c in cs.CASES_FULL]
    return sorted(rows, key=lambda r: -r["L"])


def _fmt_list(vals, fmt, sep=" / ", span=False):
    if span:
        lo, hi = min(vals), max(vals)
        return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else \
            f"{fmt.format(lo)}--{fmt.format(hi)}"
    return sep.join(fmt.format(v) for v in vals)


def case_notes(rows) -> dict:
    """Computed figures quoted in the notes of the case table."""
    by = {r["tag"]: r for r in rows}
    sm, pa = cs.BY_TAG["SM"], cs.BY_TAG["Pa"]
    h = cs.BY_TAG["H"]
    su = cs.BY_TAG["Su"]
    su1_Tt = su.M_w.lo * cs.G_STD / 2
    su1_beta = 4 * su.M_w.lo / (4 * su.mu_r * su.L.value)
    return dict(
        SM_dT1=abs(T1_of(sm, 0.0005) / T1_of(sm, 0.1) - 1),
        Pa_dT1=abs(T1_of(pa, 0.005) / T1_of(pa, 0.05) - 1),
        H_mu=(h.mu_r_given.lo, h.mu_r_given.hi),
        H_beta=(h.beta * h.mu_r_given.hi / h.mu_r_given.lo, h.beta),
        Su1=(su1_Tt, su1_beta),
        G_beta=by["G"]["beta"])


def table_cases_tex(rows=None) -> str:
    rows = case_rows() if rows is None else rows
    nt = case_notes(rows)
    lines = [
        "% Generated by maps/paper_tables.py (phase 5.6) from maps/cases.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\begin{threeparttable}",
        "\\caption{Conveyors from the literature, by decreasing length. $\\xi$: take-up position "
        "from the drive exit along the return strand, in units of $L$; $\\mu_r$: inertial line "
        "density of the return strand (belt and reduced idler mass); $\\gamma=c_r/c_c$ at full "
        "material coupling ($\\alpha=1$) and, in brackets, at the lowest coupling of the material "
        "class \\citep{lodewijks2002}; $c_r$: wave speed of the return strand; $T_1$: fundamental "
        "period with the take-up mass ($\\alpha=1$); $T_t$: static belt tension at the take-up; "
        "$n$: belt strands that carry the take-up pulley; $W_r=g\\mu_rL$; "
        "$\\beta=4M/(n^2\\mu_rL)=(4/n)\\lambda T_t/W_r$; $\\beta_5$: take-up mass that lengthens "
        "the fundamental by 5\\,\\% at the case's $\\xi$ and $\\gamma$. Last column: what set "
        "$T_t$ according to the source.}",
        "\\label{tab:cases}",
        "\\begin{tabular}{@{}lrcrlrrrclll>{\\raggedright\\arraybackslash}p{24mm}@{}}",
        "\\toprule",
        "Case & $L$ & $\\xi$ & $\\mu_r$ & $\\gamma$ & $c_r$ & $T_1$ & $T_t$ & $n$ "
        "& $T_t/W_r$ & $\\beta$ & $\\beta/\\beta_5$ & $T_t$ set by \\\\",
        " & (km) & & (kg/m) & & (m/s) & (s) & (kN) & & & & & \\\\",
        "\\midrule",
    ]
    for r in rows:
        m = r["marks"]
        tag = r["tag"]
        if tag == cs.UNKNOWN_POSITION:
            xi = "--"
        elif tag == "S":
            xi = _fmt_list(r["xi"], "{:.3f}")
        else:
            xi = f"{r['xi'][0]:.3f}"
        gam = f"{r['gamma']:.2f}{m['gamma']}"
        if r["gamma_lo"] is not None:
            gam += f" ({r['gamma_lo']:.2f})"
        T1 = _fmt_list(r["T1"], "{:.1f}", span=(tag == cs.UNKNOWN_POSITION)) + m["T1"]
        dash = "--"
        Tt = dash if r["T_t"] is None else f"{r['T_t'] / 1e3:.1f}{m['T_t']}"
        n = dash if r["n"] is None else f"{r['n']}{m['n']}"
        tw = dash if r["Tt_Wr"] is None else f"{r['Tt_Wr']:.3f}{m['Tt_Wr']}"
        b = dash if r["beta"] is None else f"{r['beta']:.3f}{m['beta']}"
        br = dash if r["beta_ratio"] is None else \
            _fmt_list(r["beta_ratio"], "{:.2f}", span=(tag == cs.UNKNOWN_POSITION)) + m["beta_ratio"]
        mu = dash if r["mu_r"] is None else f"{r['mu_r']:.1f}{m['mu_r']}"
        cr = f"{r['c_r']:.0f}{m['c_r']}"
        lines.append(f"{tag} & {r['L'] / 1e3:.2f}{m['L']} & {xi} & {mu} & {gam} & {cr} & "
                     f"{T1} & {Tt} & {n} & {tw} & {b} & {br} & {r['basis']} \\\\")
    H_lo, H_hi = nt["H_mu"]
    cite = {r["tag"]: ("\\citet{harrison1983,harrison1985b}" if r["tag"] == "H"
                       else f"\\citet{{{r['bibkey']}}}") for r in rows}
    extra = {
        "Si": "KPC.",
        "LL": "$n$ read from the tensions of their Fig.~4.",
        "S": "take-up at the head in their Fig.~1 and at the tail in their model, where $T_t$ is "
             "published; $T_1$ and $\\beta/\\beta_5$ for both positions.",
        "H": f"consistency case, stepped-torque drive; $\\mu_r={H_lo:.0f}$--${H_hi:.0f}$~kg/m "
             f"($\\beta={nt['H_beta'][1]:.3f}$--${nt['H_beta'][0]:.3f}$); $\\xi=0.002$--$0.02$.",
        "G": "$\\beta$ of one 1000~kg counterweight on two strands; the number of counterweights "
             "is not given and $\\beta$ scales with it.",
        "SM": f"KGHM, variant~1; take-up between $\\xi=0$ and $0.1$ ($T_1$ changes "
              f"{100 * nt['SM_dT1']:.1f}\\,\\%).",
        "Pa": f"$\\xi=0.005$--$0.05$ ($T_1$ changes {100 * nt['Pa_dT1']:.1f}\\,\\%).",
        "NC": f"case~1; intermediate drive at {cs.BY_TAG['NC'].sigma_d.value:.3f}$L$ from the "
              f"head; no take-up data, $T_1$ with $\\beta\\to0$.",
        "Lo": "chapter~8.",
        "Su": f"SASOL; intermediate drive at {cs.BY_TAG['Su'].sigma_d.value:.3f}$L$ from the head, "
              f"take-up after the secondary drive pulley; design~2 (design~1: "
              f"$T_t={nt['Su1'][0] / 1e3:.1f}$~kN, $\\beta={nt['Su1'][1]:.3f}$).",
        "WR": f"take-up position and rigging unknown; ranges over "
              f"$\\xi={WR_XI[0]:g}$--${WR_XI[1]:g}$.",
    }
    note_by_tag = {t: f"{t}, {cite[t]}: {extra.get(t, '')}".rstrip(": ") for t in cite}
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}[flushleft]",
        "\\footnotesize",
        "\\item Superscripts: weakest provenance among the inputs of each value; "
        "c, belt or idler catalogue \\citep{continental2021,fennerdunlop2009,cema}; a, assumed; "
        "m, measured. Unmarked values are published or derived from published data. The "
        "counterweight is taken as hung directly on the $n$ strands ($\\lambda=1$) unless the "
        "rigging is published; for a roped rig that reduces the travel of the weight this is an "
        "upper bound of $\\beta$. $E$: drive factor $e^{\\mu\\theta}$ implied by the published "
        "running tensions. Full provenance, notes and ranges of every input: \\texttt{cases.py} "
        "in the code archive.",
        "\\item " + " ".join(note_by_tag[r["tag"]] for r in rows),
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- applications table
APP_JSON = DATA / "table_applications.json"
APP_LABELS = {("Lo", "real"): "real, 30~s", ("Lo", "real, shortest start"): "real, shortest start",
              ("Lo", "tail"): "tail, 30~s", ("SM", "real"): "real", ("SM", "tail"): "tail",
              ("Su", "real"): "real (after the drive)", ("Su", "tight side"): "tight side"}
GOVERNS = {"grip": "grip", "slack": "slack", "sag": "sag"}


def app_rows(recompute: bool = False):
    """Rows of `applications.real_alt_rows`, cached as JSON (floats; inf for 'none')."""
    if APP_JSON.exists() and not recompute:
        return json.loads(APP_JSON.read_text(), parse_constant=lambda s: float(s))
    import applications as ap
    rows = ap.real_alt_rows()
    clean = [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
             for r in rows]
    APP_JSON.write_text(json.dumps(clean, indent=1))
    return clean


def table_applications_tex(rows=None) -> str:
    rows = app_rows() if rows is None else rows
    head = {
        "Lo": "Lo: 1~km, one drive pulley ($E=3.0$), $T_t=@T1@$~kN; published start 30~s",
        "SM": "SM: 3~km incline, tandem drive ($E=16.4$), $T_t=@T0@$~kN; start $3T_1$ (assumed)",
        "Su": "Su: 805~m, drive at $\\sigma_d=@SD@$ ($E=11.5$), $T_t=@T0@$~kN; published start 25~s",
    }
    lines = [
        "% Generated by maps/paper_tables.py (phase 5.6) from maps/applications.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{threeparttable}",
        "\\caption{Start-ups of three conveyors in physical units \\citep[Lo, SM, Su:][]{lodewijks1996,suchorab2025longdistance,surtees1995runback}, real take-up position and an "
        "alternative (sine profile, $\\hat\\zeta=0.01$, DIN~22101 strand resistances fitted to "
        "$F_U$ and starting with the belt speed, gravity from the carry profile, the case's "
        "counterweight in every position). $\\sigma_t$: take-up position from the head pulley "
        "along the return strand, in units of $L$. Peak drive-entry tension and its ratio to the "
        "running value; lowest drive-exit tension during and after the start; carriage travel; "
        "take-up tension required by the three conditions (positive tension along the loop, "
        "grip $T_\\mathrm{entry}\\le E\\,T_\\mathrm{exit}$, running sag below 2\\,\\%), the one "
        "that governs and the running exit tension it implies; shortest sine start that the "
        "case's $T_t$ admits.}",
        "\\label{tab:applications}",
        "\\begin{tabular}{@{}lrrrrrrrrrlrr@{}}",
        "\\toprule",
        " & & & & & \\multicolumn{2}{c}{Entry peak} & Exit & & \\multicolumn{3}{c}{Required} "
        "& Shortest \\\\",
        "\\cmidrule(lr){6-7}\\cmidrule(lr){10-12}",
        "Take-up & $\\sigma_t$ & $T_1$ & $t_a$ & $t_a/T_1$ & (kN) & /running & min. & Travel "
        "& $T_t$ & governs & $T_2$ & start \\\\",
        " & & (s) & (s) & & & & (kN) & (m) & (kN) & & (kN) & (s) \\\\",
    ]
    last = None
    for r in rows:
        tag = r["tag"]
        if tag != last:
            c = cs.BY_TAG[tag]
            lines.append("\\midrule")
            txt = (head[tag].replace("@T1@", f"{r['T_t'] / 1e3:.1f}")
                   .replace("@T0@", f"{r['T_t'] / 1e3:.0f}").replace("@SD@", f"{c.sigma_d.value:.3f}"))
            lines.append(f"\\multicolumn{{13}}{{@{{}}l}}{{{txt}}} \\\\")
            last = tag
        sh = "--" if not np.isfinite(r["t_a_req"]) else f"{r['t_a_req']:.0f}"
        exit_min = r["exit_min"] / 1e3
        lines.append(
            f"{APP_LABELS[(tag, r['label'])]} & {r['sigma']:.3f} & {r['T1']:.1f} & {r['t_a']:.0f} & "
            f"{r['t_a'] / r['T1']:.2f} & {r['entry'] / 1e3:.0f} & {r['entry_ratio']:.2f} & "
            f"${exit_min:.0f}$ & {r['travel']:.2f} & {r['req'] / 1e3:.1f} & {GOVERNS[r['gov']]} & "
            f"{r['T2_req'] / 1e3:.1f} & {sh} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}[flushleft]",
        "\\footnotesize",
        "\\item Slack: positive tension along the loop; in SM it is the weight of the return "
        "strand held from the take-up near the high end, at rest. Shortest start: --, none (the "
        "running state already fails with the case's $T_t$). The shortest start does not account "
        "for belt strength or motor torque. Su, tight side: the exit tension is negative already "
        "in running.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


PREVIEW = r"""% Standalone preview of the generated tables (phase 5.6). Compile with
%   pdflatex tables_preview && bibtex tables_preview && pdflatex tables_preview (x2)
% after copying the project .bib files next to it (Referencias.bib and refs_*.bib).
\documentclass[a4paper,10pt]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{amsmath,array,booktabs,threeparttable}
\usepackage[authoryear,round]{natbib}
\begin{document}
\input{table_cases}
\input{table_applications}
\bibliographystyle{plainnat}
\bibliography{Referencias,refs_phase2,refs_phase4,refs_phase45,refs_phase56}
\end{document}
"""


def main(argv):
    recompute = "--recompute" in argv
    OUT.mkdir(parents=True, exist_ok=True)
    rows = case_rows()
    (OUT / "table_cases.tex").write_text(table_cases_tex(rows))
    (OUT / "table_applications.tex").write_text(table_applications_tex(app_rows(recompute)))
    (OUT / "tables_preview.tex").write_text(PREVIEW)
    for r in rows:
        print(f"  {r['tag']:3s} L {r['L'] / 1e3:6.2f} km  T1 {'/'.join(f'{t:.1f}' for t in r['T1'][:3])}"
              f"{' ...' if len(r['T1']) > 3 else ''} s  beta {r['beta'] if r['beta'] is None else round(r['beta'], 4)}"
              f"  T_t/W_r {r['Tt_Wr'] if r['Tt_Wr'] is None else round(r['Tt_Wr'], 4)}")
    nt = case_notes(rows)
    print(f"  notes: SM dT1 {100 * nt['SM_dT1']:.2f} %, Pa dT1 {100 * nt['Pa_dT1']:.2f} %, "
          f"Su design 1 T_t {nt['Su1'][0] / 1e3:.1f} kN beta {nt['Su1'][1]:.3f}")
    print(f"written: {OUT / 'table_cases.tex'}, {OUT / 'table_applications.tex'}")


if __name__ == "__main__":
    main(sys.argv[1:])
