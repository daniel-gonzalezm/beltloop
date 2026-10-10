"""Phase 4.8: final figures of the paper (results of phase 4), one reproducible script.

    python maps/paper_figures.py                 # all figures, cached data where available
    python maps/paper_figures.py modal beta      # some figures
    python maps/paper_figures.py --recompute     # recompute the data (about 4 min in total)

Data go to maps/data/<figure>.npz (small; committed, so the figures can be restyled without
recomputing), figures to maps/figures/paper/<figure>.pdf and .png, and the numbers quoted in
the captions are printed. Case markers come from maps/cases.py. The working figures of each
step (maps/*.py) stay as the record of that step.

Figures (numbering of the manuscript decided in phase 6):
  modal     frequencies and participation in (xi, gamma), beta -> 0, plus the intermediate drive
  beta      take-up mass threshold beta_5, each mode followed by its strand (B1, B2, A1)
  startup   universal start-up curve of the fixed-free strand and the loop collapse
  crawl     initial crawl: duration of the ramp versus duration of the hold
  validity  validity conditions: slack at the exit, rebound, minimum T_2, tight side
  applications  start-ups of Lo, SM and Su in physical units against the take-up position
                (phase 5.5; also writes table_applications.txt, about 30 s)
The loop schematic (Figure 1) is drawn by maps/fig_loop.py and the lumped-mass validation figure
by validation/fig_lumped_validation.py, both with style.py; the numbers that the captions and
the text quote are written by maps/paper_numbers.py (step 6.2).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import style  # noqa: E402
from cases import NORDELL_CIOZDA, SASOL, points  # noqa: E402

DATA = HERE / "data"
TOL5 = 0.05
LOG2 = None


# ---------------------------------------------------------------------------- data cache
def cached(name, compute, recompute):
    path = DATA / f"{name}.npz"
    if path.exists() and not recompute:
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    DATA.mkdir(exist_ok=True)
    d = compute()
    np.savez_compressed(path, **d)
    return d


# Cases near the drive (xi <= CLUSTER_XI) pile up on the left edge of the maps. They are spread
# into lanes LANE_DX apart (display only: the caption gives the true xi) so that their coupling
# bars do not overlap, and the labels of the cases with xi <= LEADER_XI are set in a column at
# LEADER_X with leader lines; a case at the tail is drawn at EDGE_XI (phase 5.3(d)).
CLUSTER_XI, LANE_X0, LANE_DX, LANE_PAD = 0.03, 0.004, 0.016, 0.08
LEADER_XI, LEADER_X, LEADER_GAP = 0.15, 0.16, 0.12
EDGE_XI = 0.985


# A case whose take-up position is unknown (WR) is left off the maps (phase 5.3(d)): its line
# across xi carried only gamma. It stays in the case table and in the text (case_margins).
SHOW_UNKNOWN_POSITION = False


def case_marks(include_unknown=SHOW_UNKNOWN_POSITION):
    """Plot positions of the case markers: dicts with tag, beta, gamma (alpha = 1), g_lo
    (lowest coupling of the material class), true xi, plotted x, unknown-position flag.

    Lanes are assigned greedily, cases sorted by decreasing gamma, each to the first lane where
    its bar [g_lo, gamma] clears the bars already there by LANE_PAD."""
    from cases import points_alpha
    marks = [dict(tag=t, beta=b, gamma=g, g_lo=lo, xi=x, x=x, unknown=u)
             for t, b, g, lo, hi, x, u in points_alpha() if include_unknown or not u]
    lanes = []
    for m in sorted((m for m in marks if m["xi"] <= CLUSTER_XI and not m["unknown"]),
                    key=lambda m: -m["gamma"]):
        for k, lane in enumerate(lanes):
            if all(m["gamma"] + LANE_PAD < o["g_lo"] or m["g_lo"] - LANE_PAD > o["gamma"] for o in lane):
                lane.append(m)
                break
        else:
            k = len(lanes)
            lanes.append([m])
        m["x"] = LANE_X0 + LANE_DX * k
    for m in marks:                            # at the tail the bar would hide under the spine
        if m["xi"] > EDGE_XI and not m["unknown"]:
            m["x"] = EDGE_XI
    return marks


def leader_positions(ys, gap=LEADER_GAP, lo=1.04, hi=2.96):
    """Label heights: the target heights ys pushed apart by at least gap, within [lo, hi]."""
    order = np.argsort(ys)
    y = np.clip(np.asarray(ys, float)[order], lo, hi)
    for k in range(1, len(y)):
        y[k] = max(y[k], y[k - 1] + gap)
    over = y[-1] - hi
    if over > 0:                               # shift down from the top if the column overflows
        y[-1] = hi
        for k in range(len(y) - 2, -1, -1):
            y[k] = min(y[k], y[k + 1] - gap)
    out = np.empty_like(y)
    out[order] = y
    return out


def mark_cases(ax, labels=True, gmin=1.0, gmax=3.0, fs=6, color="k", bars=True):
    """Case markers at gamma(alpha = 1), with the coupling bar down to gamma(alpha_lo)."""
    marks = case_marks()
    clip = lambda g: min(max(g, gmin), gmax)
    for m in marks:
        if m["unknown"]:
            continue
        x, g = m["x"], clip(m["gamma"])
        if bars and m["g_lo"] < m["gamma"]:
            ax.plot([x, x], [clip(m["g_lo"]), g], "-", color=color, lw=0.6, zorder=5,
                    clip_on=False, solid_capstyle="butt")
            ax.plot(x, clip(m["g_lo"]), "_", ms=3.0, mew=0.6, color=color, zorder=5, clip_on=False)
        ax.plot(x, g, "o", ms=3.2, mfc="w", mec=color, mew=0.7, zorder=6, clip_on=False)
    box = dict(fc="white", ec="none", alpha=0.75, pad=0.2)
    if labels:
        near = [m for m in marks if m["xi"] <= LEADER_XI and not m["unknown"]]
        ys = leader_positions([clip(m["gamma"]) for m in near])
        for m, y in zip(near, ys):
            ax.annotate(m["tag"], (m["x"], clip(m["gamma"])), xytext=(LEADER_X, y),
                        textcoords="data", fontsize=fs, ha="left", va="center", zorder=7, bbox=box,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="0.25", shrinkA=0,
                                        shrinkB=2.0))
        for m in marks:
            if m["xi"] > LEADER_XI and not m["unknown"]:
                ax.annotate(m["tag"], (m["x"], clip(m["gamma"])), xytext=(-3, 2),
                            textcoords="offset points", fontsize=fs, ha="right", va="bottom",
                            zorder=7, bbox=box)
    # take-up position unknown: dotted line at gamma(alpha = 1), coupling bar at its tail end
    unk = [m for m in marks if m["unknown"]]
    if unk:
        gc, glo = unk[0]["gamma"], unk[0]["g_lo"]
        x0, x1 = min(m["xi"] for m in unk), max(m["xi"] for m in unk)
        ax.plot([x0, x1], [gc, gc], ":", color=color, lw=0.8, zorder=6)
        if bars and glo < gc:
            ax.plot([EDGE_XI, EDGE_XI], [glo, gc], "-", color=color, lw=0.6, zorder=5)
            ax.plot(EDGE_XI, glo, "_", ms=3.0, mew=0.6, color=color, zorder=5)
        if labels:
            ax.annotate(unk[0]["tag"], (0.62, gc), xytext=(0, 2), textcoords="offset points",
                        fontsize=fs, ha="center", va="bottom", zorder=7, bbox=box)


# ============================================================================ modal
def data_modal():
    from beltloop import Loop, strand_participation
    import modal_maps
    import intermediate_drive as idr
    xis = np.r_[0.002, np.linspace(0.01, 0.99, 50), 0.998]
    gams = np.linspace(1.0, 3.0, 41)
    T1, ratio, F1, dF1 = modal_maps.fields(xis, gams)
    ratio_l1, _ = idr.offset_curves()
    nc = idr.nordell_ciozda()
    # Surtees (SASOL): drives ~158 m from the head, take-up right after them (Fig. 10)
    # (beta -> 0, as the curves), at full coupling and at the lowest of its class (bar)
    from cases import BY_TAG
    su = BY_TAG[SASOL["tag"]]

    def ratio_su(gamma):
        lp = Loop.from_positions(SASOL["sigma_d"], SASOL["sigma_t"], gamma)
        B = lp.downstream[::-1]
        Om, _ = strand_participation(B, 1)
        return 2 * np.pi / Om[0] / (4 * sum(s.g * s.length for s in B))

    sasol = np.array([SASOL["sigma_d"], ratio_su(SASOL["gamma"]), ratio_su(su.gamma_range()[0])])
    return dict(xis=xis, gams=gams, T1=T1, ratio=ratio, F1=F1, dF1=dF1, l1=idr.L1S,
                g_l1=np.array(idr.GAMMAS), ratio_l1=np.array([ratio_l1[g] for g in idr.GAMMAS]),
                nc=np.array([NORDELL_CIOZDA["sigma_d"], nc[0][3], nc[0][1], nc[1][1]]), sasol=sasol)


# Colour maps of fig_modal: (name, lo, hi), dark end cut off (phase 5.3(d)).
CMAPS = dict(a=("viridis", 0.30, 1.0), b=("magma", 0.50, 1.0), c=("cividis", 0.40, 1.0))


def fig_modal(d):
    import matplotlib.pyplot as plt
    xis, gams = d["xis"], d["gams"]
    fig, axs = plt.subplots(2, 2, figsize=(style.DOUBLE, 128 * style.MM), constrained_layout=True)
    specs = [(axs[0, 0], d["T1"], np.arange(4, 14.01, 1.0), CMAPS["a"], "%.0f",
              r"$\tau_1 = t_1 c_r / L$", "(a)"),
             (axs[0, 1], d["ratio"], np.arange(0.83, 1.0001, 0.01), CMAPS["b"], "%.2f",
              r"$\tau_1 / (4 \tau_B)$", "(b)"),
             (axs[1, 0], d["F1"], np.arange(0.40, 0.8201, 0.02), CMAPS["c"], "%.1f",
              r"$\Gamma_1^2 m_1 / (1+\gamma^2)$", "(c)")]
    for ax, Z, lev, cmap, fmt, clab, lab in specs:
        cmap = style.light_cmap(*cmap)
        cs = ax.contourf(xis, gams, Z, levels=lev, cmap=cmap, extend="both")
        step = 2 if lab != "(c)" else 5
        lv = lev[::step]
        # contour lines and labels in white only where the fill is dark
        dark = np.array([style.text_colour(cs.cmap(cs.norm(v))) == "w" for v in lv])
        for sel, col in ((dark, "w"), (~dark, "k")):
            if sel.any():
                cl = ax.contour(xis, gams, Z, levels=lv[sel], colors=col, linewidths=0.4)
                ax.clabel(cl, fmt=fmt if lab != "(b)" else "%.2f", fontsize=6, inline_spacing=2)
        cb = fig.colorbar(cs, ax=ax, pad=0.02)
        cb.set_label(clab)
        cb.ax.tick_params(labelsize=6)
        ax.set(xlim=(0, 1), ylim=(1, 3), xlabel=r"take-up position $\xi$",
               ylabel=r"wave-speed ratio $\gamma = c_r/c_c$")
        mark_cases(ax, labels=(lab == "(a)"))
        style.panel_label(ax, lab)
    ax = axs[1, 0]
    ax.contourf(xis, gams, np.abs(d["dF1"]), levels=[0.05, 10], colors="none", hatches=["////"])
    ax.contour(xis, gams, np.abs(d["dF1"]), levels=[0.05], colors="w", linewidths=0.7)
    ax = axs[1, 1]
    for g, r, c, mk in zip(d["g_l1"], d["ratio_l1"], style.OKABE_ITO, style.MARKERS):
        ax.plot(d["l1"], r, color=c, marker=mk, markevery=(0.06, 0.18), ms=2.8, mfc="w", mew=0.7,
                label=rf"$\gamma = {g:g}$")
        ax.annotate(rf"$\gamma = {g:g}$", (d["l1"][-1], r[-1]), xytext=(3, 0),   # direct label
                    textcoords="offset points", fontsize=6, va="center", color="0.15")
    ax.plot(*d["nc"][:2], "k*", ms=7, zorder=6)
    ax.annotate(NORDELL_CIOZDA["tag"], d["nc"][:2], xytext=(-4, 3), textcoords="offset points", fontsize=6, ha="right")
    xs, ys = d["sasol"][:2]
    if len(d["sasol"]) > 2:                    # coupling bar (phase 5.3(d))
        ax.plot([xs, xs], [d["sasol"][2], ys], "k-", lw=0.6, zorder=5)
        ax.plot(xs, d["sasol"][2], "k_", ms=3.0, mew=0.6, zorder=5)
    ax.plot(xs, ys, "kD", ms=3.5, mfc="w", zorder=6)
    ax.annotate(SASOL["tag"], (xs, ys), xytext=(4, -6), textcoords="offset points", fontsize=6)
    ax.set(xlim=(0, 1.12), xlabel=r"drive offset from the head $\ell_1 = \sigma_d$",
           ylabel=r"$\tau_1 / (4 \tau_B)$")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.grid(alpha=0.25, lw=0.4)                 # curves labelled directly (no legend)
    style.panel_label(ax, "(d)")
    style.save(fig, "fig_modal")
    print(f"  T1 c_r/L {d['T1'].min():.2f}-{d['T1'].max():.2f}; T1/(4 t_B) {d['ratio'].min():.3f}-"
          f"{d['ratio'].max():.3f}; F1 {d['F1'].min():.3f}-{d['F1'].max():.3f}; hatched (|dF1| > 5 % at "
          f"beta = 0.1) {np.mean(np.abs(d['dF1']) > 0.05):.1%} of the plane")
    i = np.argmin(abs(d["l1"] - 0.98))
    print(f"  intermediate drive: T1/(4 t_B) at l1 = {d['l1'][i]:.2f}: "
          + ", ".join(f"gamma {g:g}: {r[i]:.3f}" for g, r in zip(d["g_l1"], d["ratio_l1"])))
    print(f"  Nordell & Ciozda: l1 = {d['nc'][0]:.2f}, T1/(4 t_B) = {d['nc'][1]:.3f}, "
          f"T1 = {d['nc'][2]:.1f} s (head drive, same xi: {d['nc'][3]:.1f} s, "
          f"+{d['nc'][2] / d['nc'][3] - 1:.0%}); SASOL: l1 = {d['sasol'][0]:.2f}, "
          f"T1/(4 t_B) = {d['sasol'][1]:.3f} ({d['sasol'][2]:.3f} at the lowest coupling)")


# ============================================================================ beta
def data_beta():
    from beltloop import Loop, takeup_mass_threshold
    import modal_maps
    xis = np.linspace(0.005, 0.995, 100)
    gams = np.linspace(1.0, 3.0, 81)
    Z = {"B1": np.array([[takeup_mass_threshold(Loop.from_positions(0.0, x, g), "B", 1)
                          for x in xis] for g in gams])}
    xis2 = np.linspace(0.005, 0.995, 300)               # B2: finer, its field has steep edges
    gams2 = np.linspace(1.0, 3.0, 201)
    Z["B2"] = np.array([[takeup_mass_threshold(Loop.from_positions(0.0, x, g), "B", 2)
                         for x in xis2] for g in gams2])
    gb = np.linspace(1.0, 3.0, 401)
    edges = np.array([b2_region_edges(g) for g in gb])
    xl = np.linspace(0.005, 0.995, 400)
    g_line = np.array([1.0, 2.0, 3.0])
    A1 = np.array([[takeup_mass_threshold(Loop.from_positions(0.0, x, g), "A", 1) for x in xl]
                   for g in g_line])
    gl = np.linspace(1.0, 3.0, 41)
    xs = np.array([modal_maps.switch_position(g) or np.nan for g in gl])
    return dict(xis=xis, gams=gams, B1=Z["B1"], xis2=xis2, gams2=gams2, B2=Z["B2"], xl=xl,
                g_line=g_line, A1=A1, gl=gl, xs=xs, gb=gb, edges=edges)


def b2_region_edges(gamma, tol=TOL5):
    """Exact edges, along xi at fixed gamma, of the two grey regions of the B2 map.

    With r(xi) = Om_B2 / Om_A1 increasing in xi (A1 falls, B2 rises as the take-up moves to the
    tail): veering band r in (1, 1 + tol], from xi_1 (r = 1, the switch xi_s) to xi_2
    (r = 1 + tol); beyond it the 5 % shift is out of reach while F(Om_t) <= 0, up to xi_3 where
    F(Om_t) = 0. Edges past the right border are returned as 1; nan if the band does not exist."""
    from scipy.optimize import brentq
    from beltloop import Loop, strand_modes
    from beltloop.transfer import AB

    def parts(xi):
        lp = Loop.from_positions(0.0, xi, gamma)
        a1 = strand_modes(lp.upstream, 1)[0][0]
        b2 = strand_modes(lp.downstream[::-1], 2)[0][1]
        return a1, b2, lp

    def ratio(xi):
        a1, b2, _ = parts(xi)
        return b2 / a1

    def F(xi):
        a1, b2, lp = parts(xi)
        A, B = AB(lp, b2 / (1 + tol))
        return A[0, 1] / A[1, 1] + B[0, 1] / B[0, 0]

    lo, hi = 0.3, 0.9995
    if ratio(hi) <= 1:
        return np.nan, np.nan, np.nan
    x1 = brentq(lambda x: ratio(x) - 1, lo, hi, xtol=1e-12)
    if ratio(hi) <= 1 + tol:
        return x1, 1.0, 1.0
    x2 = brentq(lambda x: ratio(x) - 1 - tol, lo, hi, xtol=1e-12)
    xs = np.linspace(x2 + 1e-7, hi, 150)
    f = np.array([F(x) for x in xs])
    k = np.where(np.diff(np.sign(f)) != 0)[0]
    x3 = brentq(F, xs[k[0]], xs[k[0] + 1], xtol=1e-12) if len(k) else 1.0
    return x1, x2, x3


# Step 6.8 (decision of the author): the A1 panel (c) leaves the paper; its scope stays in the
# text of Section 5.2 and its numbers in paper_numbers.py. True restores the three-panel figure.
BETA_A1_PANEL = False


def fig_beta(d):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Patch
    xis, gams = d["xis"], d["gams"]
    npan = 3 if BETA_A1_PANEL else 2
    fig = plt.figure(figsize=(style.DOUBLE, 64 * style.MM), constrained_layout=True)
    gs = fig.add_gridspec(1, npan, width_ratios=[1, 1, 1.05][:npan])
    axs = [fig.add_subplot(gs[0, k]) for k in range(npan)]
    levels = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    lab_levels = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5]
    for ax, key, lab in ((axs[0], "B1", "(a) B1 (fundamental)"), (axs[1], "B2", "(b) B2")):
        Z = d[key]
        xg, gg = (xis, gams) if key == "B1" else (d["xis2"], d["gams2"])
        # grey regions are drawn from their exact edges; underneath, fill the field so that
        # the colour reaches those edges (non-finite -> above the top level)
        Zf = np.where(np.isinf(Z), 1e3, Z)
        for row in Zf:                       # veering band: carry the value from its left edge,
            for k in range(1, len(row)):     # so no colour step appears at xi_s
                if np.isnan(row[k]):
                    row[k] = row[k - 1]
        Zf = np.where(np.isnan(Zf), 1e3, Zf)
        cs = ax.contourf(xg, gg, Zf, levels=levels, norm=LogNorm(0.05, 2.0), cmap="viridis",
                         extend="both")
        Zl = np.where(np.isfinite(Z), Z, np.nan)
        cl = ax.contour(xg, gg, Zl, levels=lab_levels, colors="k", linewidths=0.4)
        ax.clabel(cl, fmt="%g", fontsize=6, inline_spacing=2)
        if key == "B2":
            gb, (x1, x2, x3) = d["gb"], d["edges"].T
            ok = np.isfinite(x1)
            ax.fill_betweenx(gb[ok], x2[ok], x3[ok], color="0.85", lw=0, zorder=3)
            ax.fill_betweenx(gb[ok], x1[ok], x2[ok], color="0.45", lw=0, zorder=3)
        ax.set(xlim=(0, 1), ylim=(1, 3), xlabel=r"take-up position $\xi$")
        mark_cases(ax, labels=(key == "B1"))
        style.panel_label(ax, lab)
    axs[0].set_ylabel(r"wave-speed ratio $\gamma = c_r/c_c$")
    axs[1].plot(d["edges"][:, 0], d["gb"], "w--", lw=0.9, zorder=4)
    axs[1].legend(handles=[Patch(fc="0.45", ec="none", label="B2 and A1 veer"),
                           Patch(fc="0.85", ec="none", label="5 % not reached")],
                  loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=6, handlelength=1.4)
    cb = fig.colorbar(cs, ax=axs[:2], pad=0.01, aspect=30)
    cb.set_label(r"$\beta_5$ (5 % longer period)")
    cb.ax.tick_params(labelsize=6)
    if BETA_A1_PANEL:
        fig_beta_a1(axs[2], d)
    style.save(fig, "fig_beta")
    for key in ("B1", "B2"):
        Z = d[key]; f = Z[np.isfinite(Z)]
        print(f"  {key}: beta_5 {f.min():.3f}-{f.max():.2f}, median {np.median(f):.2f}; veering "
              f"{np.isnan(Z).mean():.1%}, not reached {np.isinf(Z).mean():.1%} of the plane")
    for row in case_margins():
        print("  {:3s} L = {:5.0f} m, beta = {:.3f}: beta/beta_5(B1) {:.2f} ({:.2f} at the lowest "
              "coupling); fundamental +{:.2f} % (+{:.2f} %)".format(*row))
    for g, y in zip(d["g_line"], d["A1"]):
        ok = np.isfinite(y)
        dev = y[ok] / (0.205 * d["xl"][ok]) - 1
        print(f"  A1, gamma {g:g}: exact / (0.205 xi) - 1 from {dev.min():+.1%} to {dev.max():+.1%}; "
              f"defined at {ok.mean():.0%} of xi")


def fig_beta_a1(ax, d):
    """Panel (c) of fig_beta up to step 6.8: threshold of A1 for gamma = 2 and the tail case."""
    a1 = cached("beta_a1", data_a1, "--recompute" in sys.argv)
    extra = draw_a1(ax, a1, A1_MODE)
    ax.plot(d["xl"], 4 * d["xl"] / 2 * (1.05 ** 2 - 1), "k:", lw=0.9,
            label=r"single pole, $0.41\,\tilde m = 0.205\,\xi$")
    from beltloop import Loop, natural_frequencies, poles, takeup_mass_threshold
    for tag, beta, gam, xi, unknown in points():
        if xi > 0.5 and not unknown:            # take-up at the tail: A1 is the second mode
            lp = Loop.from_positions(0.0, xi, gam)
            b5 = takeup_mass_threshold(lp, "A", 1)
            pA = np.pi / (2 * xi)
            P = poles(lp, 8)
            k = int(np.argmin(abs(P - pA)))
            shift = pA / natural_frequencies(lp, beta, k + 1)[k] - 1
            ax.plot([xi, xi], [beta, b5], ":", color="0.3", lw=0.7, clip_on=False)
            ax.plot(xi, b5, "_", ms=5, color="0.3", mew=0.8, clip_on=False)
            ax.plot(xi, beta, "o", ms=3.2, mfc="w", mec="k", mew=0.7, zorder=6, clip_on=False)
            ax.annotate(f"{tag}: {100 * shift:+.1f} %", (xi, beta), xytext=(-5, -5),
                        textcoords="offset points", fontsize=6, ha="right", va="top")
    ax.set(yscale="log", xlim=(0, 1), ylim=(1e-3, 1.0), xlabel=r"take-up position $\xi$",
           ylabel=r"$\beta_5$ of A1;  $\beta$ of the cases")
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + extra, loc="lower right", fontsize=6)
    ax.grid(alpha=0.25, lw=0.4, which="both")
    style.panel_label(ax, "(c) A1")


def case_margins():
    """Per head-drive case and take-up position: (tag, L, beta, beta / beta_5 of B1 and exact
    lengthening of the fundamental in %, at full coupling and at the lowest of the class)."""
    from cases import BY_TAG
    from beltloop import Loop, natural_frequencies, takeup_mass_threshold
    rows = []
    for m in case_marks(include_unknown=True):
        c = BY_TAG["S" if m["tag"] == "St" else m["tag"]]
        out = []
        for g in (m["gamma"], m["g_lo"]):
            lp = Loop.from_positions(0.0, m["xi"], g)
            shift = natural_frequencies(lp, 0.0, 1)[0] / natural_frequencies(lp, m["beta"], 1)[0] - 1
            out.append((m["beta"] / takeup_mass_threshold(lp, "B", 1), 100 * shift))
        rows.append((m["tag"], c.L.value, m["beta"], out[0][0], out[1][0], out[0][1], out[1][1]))
    return rows


# How fig_beta (c) shows where the threshold of A1 is not defined (phase 5.3(d)):
#   "shade"      grey bands as in (b): veering, not reached;
#   "asymptotes" thin vertical lines at the asymptotes, gaps left blank;
#   "strip"      asymptotes, plus the veering intervals as a strip along the xi axis;
#   "merged"     asymptotes, plus one strip wherever beta_5 of A1 is undefined (not reached
#                or veering: contiguous, the first is too narrow to show on its own);
#   "band"       the merged intervals as light grey bands over the full height; each band
#                starts at an asymptote, so no asymptote lines are drawn.
A1_MODE = "band"


def undefined_intervals(iv):
    """Contiguous intervals where beta_5 of A1 is not finite (classes 'i' and 'n' merged)."""
    out = []
    for c, x0, x1 in iv:
        if c == 0:
            continue
        if out and x0 - out[-1][1] < 1e-8:
            out[-1][1] = x1
        else:
            out.append([x0, x1])
    return [(a, b - a) for a, b in out]


def draw_a1(ax, a1, mode=A1_MODE):
    """Branches of beta_5 of A1 and the marks of the gaps; returns extra legend handles."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    iv = a1["iv"]
    asym = [iv[k, 2] for k in range(len(iv) - 1) if iv[k, 0] == 0 and iv[k + 1, 0] == 1]
    veer = [(x0, x1 - x0) for c, x0, x1 in iv if c == 2]
    extra = []
    if mode == "band":
        for a, w in undefined_intervals(iv):
            ax.axvspan(a, a + w, fc="0.88", ec="none", lw=0, zorder=0)
        extra = [Patch(fc="0.88", ec="none", label="undefined for A1 alone")]
    elif mode == "shade":
        for c, x0, x1 in iv:
            if c != 0:
                ax.axvspan(x0, x1, fc="0.45" if c == 2 else "0.85", alpha=0.5 if c == 2 else 1.0,
                           ec="none", lw=0, zorder=0)
        extra = [Patch(fc="0.45", alpha=0.5, ec="none", label="A1 and a B mode veer"),
                 Patch(fc="0.85", ec="none", label="5 % not reached")]
    else:
        for x in asym:
            ax.axvline(x, color="0.55", lw=0.4, ls=(0, (3, 2)), zorder=1)
        extra = [Line2D([], [], color="0.55", lw=0.4, ls=(0, (3, 2)),
                        label=r"asymptote, $F(\Omega_t) = 0$")]
        if mode in ("strip", "merged"):
            bars = veer if mode == "strip" else undefined_intervals(iv)
            ax.broken_barh(bars, (0.0, 0.03), transform=ax.get_xaxis_transform(), fc="0.35",
                           ec="none", zorder=3)
            extra.append(Patch(fc="0.35", ec="none", label="A1 and a B mode veer"
                               if mode == "strip" else "undefined for A1 alone"))
    ax.plot(a1["x"], a1["y"], color=style.OKABE_ITO[0], lw=0.9, zorder=4,
            label=rf"exact, $\gamma = {float(a1['gamma']):g}$")
    return extra


def a1_class(xi, gamma):
    """beta_5 of A1 at (xi, gamma) and its class: 'f' finite, 'i' not reached (F(Om_t) <= 0),
    'n' veering with a mode of strand B."""
    from beltloop import Loop, takeup_mass_threshold
    y = takeup_mass_threshold(Loop.from_positions(0.0, xi, gamma), "A", 1)
    return y, ("n" if np.isnan(y) else "i" if np.isinf(y) else "f")


def data_a1(gamma=2.0, n=3000, tol=1e-11):
    """beta_5 of A1 along xi, branch by branch (phase 5.3(d)). The edges between classes are
    located by bisection to tol in xi; each finite branch is sampled densely towards its right
    edge, where it ends either at a veering edge (finite value) or at a zero of F(Om_t), where
    beta_5 -> infinity. Branches are returned joined by NaN; intervals as (class, x0, x1)."""
    xg = np.linspace(0.005, 0.995, n)
    cg = [a1_class(x, gamma)[1] for x in xg]
    edges = []
    for k in range(n - 1):
        if cg[k] != cg[k + 1]:
            a, b = xg[k], xg[k + 1]
            while b - a > tol:
                m = 0.5 * (a + b)
                if a1_class(m, gamma)[1] == cg[k]:
                    a = m
                else:
                    b = m
            edges.append((a, b))
    starts = [xg[0]] + [b for a, b in edges]
    ends = [a for a, b in edges] + [xg[-1]]
    classes = [cg[0]] + [cg[k + 1] for k in range(n - 1) if cg[k] != cg[k + 1]]
    X, Y, iv = [], [], []
    for c, x0, x1, nxt in zip(classes, starts, ends, classes[1:] + ["end"]):
        iv.append(("fin".index(c), x0, x1))
        if c != "f":
            continue
        w = x1 - x0
        xs = x0 + w * np.linspace(0.0, 1.0, max(int(w * 2000), 4))
        if nxt == "i":                          # asymptote: geometric towards the edge
            xs = np.union1d(xs, x1 - w * np.geomspace(0.5, 1e-9, 120))
        X += list(xs) + [np.nan]
        Y += [a1_class(x, gamma)[0] for x in xs] + [np.nan]
    return dict(gamma=gamma, x=np.array(X), y=np.array(Y), iv=np.array(iv, float))


# ============================================================================ startup
def data_startup():
    from beltloop import fast_start_limit, strand_curves
    import startup_curves as sc
    R = np.unique(np.r_[np.logspace(-1, np.log10(20), 120), np.linspace(0.7, 1.1, 41)])
    out = {}
    for k in sc.KINDS:
        D, Dfree = strand_curves(R, kind=k)
        out[f"D_{k}"], out[f"Dfree_{k}"] = D, Dfree
        out[f"wave_{k}"] = fast_start_limit(R, k)
    out["D_res"] = strand_curves(R, load="resistance")[0]
    for z in sc.ZETAS:
        out[f"D_z{z:g}"] = strand_curves(R, zeta1=z)[0]
    out.update(collapse_envelope())
    return dict(R=R, **out)


COLLAPSE_XI = (0.01, 0.2, 0.5, 0.8, 0.95)
COLLAPSE_XI_EXIT = (0.2, 0.5, 0.8, 0.95)     # xi = 0.01: T_A1 = 0.04 L/c_r, start times too short
COLLAPSE_G = (1.0, 1.5, 2.0, 3.0)


def collapse_envelope(rc=None):
    """Largest |loop / uniform strand - 1| over the take-up positions, at each start time.

    Entry: tau_a = r T_1, one envelope per gamma (max over xi). Exit: tau_a = r T_A1, one
    envelope over all xi >= 0.2 and gamma. The reference is the uniform-strand curve at the
    same r (sine, beta -> 0, undamped)."""
    from beltloop import Loop, StartProfile, natural_frequencies, startup_metrics, strand_curves
    rc = np.logspace(np.log10(0.2), np.log10(20), 25) if rc is None else np.asarray(rc)
    ref = strand_curves(rc)[0]
    dev_in = np.zeros((len(COLLAPSE_G), len(rc)))
    dev_out = np.zeros(len(rc))
    for ig, g in enumerate(COLLAPSE_G):
        for xi in COLLAPSE_XI:
            lp = Loop.from_positions(0.0, xi, g)
            T1 = 2 * np.pi / natural_frequencies(lp, 1e-3, 1)[0]
            TA1 = 4 * xi                                   # strand A: uniform return
            for k, r in enumerate(rc):
                m = startup_metrics(lp, 1e-3, StartProfile("sine", r * T1, "none"))
                dev_in[ig, k] = max(dev_in[ig, k], abs(m.D_entry / ref[k] - 1))
                if xi in COLLAPSE_XI_EXIT:
                    m = startup_metrics(lp, 1e-3, StartProfile("sine", r * TA1, "none"))
                    dev_out[k] = max(dev_out[k], abs(m.D_exit / ref[k] - 1))
    return dict(rc=rc, dev_in=dev_in, dev_out=dev_out, g_c=np.array(COLLAPSE_G))


def practice():
    """(tag, tau_a / T_1) for every case with a published start time: exact T_1 of the loop with
    the case's beta, gamma (alpha = 1) and the take-up at its first listed position (S: at the
    head, as in their Fig. 1). Phase 5.3(d): from cases.py instead of hard-coded ratios."""
    from cases import BY_TAG, CASES_FULL
    from beltloop import Loop, natural_frequencies
    out = []
    for c in CASES_FULL:
        if c.t_a is None or c.beta is None or c.c_r is None:
            continue
        lp = Loop.from_positions(c.sigma_d.value, c.sigma_t[0], c.gamma())
        T1 = 2 * np.pi / natural_frequencies(lp, c.beta, 1)[0] * c.L.value / c.c_r
        out.append((c.tag, c.t_a.value / T1))
    return sorted(out, key=lambda p: p[1])


PRACTICE = practice()


def fig_startup(d):
    import matplotlib.pyplot as plt
    from beltloop import PEAK_FACTOR
    R = d["R"]
    kinds = (("sine", "sine"), ("triangular", "triangular"), ("parabolic", "parabolic"))
    lss = ("-", "--", "-.")                    # profiles differ by line style as well as colour
    fig, axs = plt.subplots(2, 2, figsize=(style.DOUBLE, 112 * style.MM), constrained_layout=True)
    ax = axs[0, 0]
    for (k, lab), c, ls in zip(kinds, style.OKABE_ITO, lss):
        ax.plot(R, d[f"D_{k}"], color=c, ls=ls, label=lab)
        w = d[f"wave_{k}"]
        ax.plot(R[w < 2.4], w[w < 2.4], ":", color=c, lw=0.8)
    ax.plot([], [], ":", color="0.3", lw=0.8, label="wave law")
    ax.axvspan(2, 3, color="0.9", zorder=0)
    for tag, r in PRACTICE:
        ax.annotate(tag, (r, 0.06), xytext=(2, 0), textcoords="offset points", ha="left",
                    fontsize=6, color="0.3")
    ax.set(ylabel=r"$D = T_{\max} / (m a_m)$")
    ax.legend(loc="upper right")
    ax = axs[0, 1]
    for (k, lab), c, ls in zip(kinds, style.OKABE_ITO, lss):
        ax.plot(R, d[f"D_{k}"] * PEAK_FACTOR[k], color=c, ls=ls, label=lab)
    ax.legend(loc="lower right")
    ax.set(ylabel=r"$T_{\max} / (m V_\infty / t_a)$", ylim=(0, 3.0))
    ax = axs[1, 0]
    rc = d["rc"]
    floor = 0.01                                           # %, below this: plotted at the floor
    for g, dv, mk, c in zip(d["g_c"], d["dev_in"], ("o", "s", "^", "D"), style.OKABE_ITO):
        ax.plot(rc, np.maximum(100 * dv, floor), "-", marker=mk, ms=2.8, mfc="none", mew=0.7,
                color=c, lw=0.9, label=rf"entry, $\gamma = {g:g}$")
    ax.plot(rc, np.maximum(100 * d["dev_out"], floor), "k--", marker="x", ms=3, mew=0.7, lw=0.9,
            label=r"exit, all $\gamma$")
    ax.axhline(2.0, color="0.4", lw=0.6, ls=":")
    ax.set(yscale="log", ylim=(floor, 100), ylabel=r"largest $|$loop / strand $- 1|$ (%)")
    ax.legend(loc="upper right", fontsize=6, ncol=1)
    ax = axs[1, 1]
    ax.plot(R, d["Dfree_sine"], color=style.OKABE_ITO[0], label="take-up travel (free end)")
    ax.plot(R, d["D_res"], "--", color=style.OKABE_ITO[1], label=r"resistances, $\varphi = V/V_\infty$")
    ax.set(ylabel="peak / quasi-static")
    ax.legend(loc="upper right")
    for ax, lab in zip(axs.ravel(), ("(a)", "(b)", "(c)", "(d)")):
        ax.set(xscale="log", xlim=(0.1, 20))
        if lab in ("(a)", "(d)"):
            ax.set_ylim(0, 2.1)
        if lab != "(c)":
            ax.axhline(1.0, color="k", lw=0.5)
        if lab != "(c)":
            for tag, r in PRACTICE:
                ax.axvline(r, color="0.55", lw=0.5, ls=(0, (1, 2)), zorder=1)
        ax.grid(alpha=0.25, lw=0.4, which="both")
        ax.set_xlabel(r"$\tau_a / \tau_s$" if lab != "(c)" else
                      r"$\tau_a / \tau_s$ ($\tau_1$ at the entry, $\tau_{A1}$ at the exit)")
        style.panel_label(ax, lab)
    style.save(fig, "fig_startup")
    print("  practice, tau_a / T_1: " + ", ".join(f"{t} {r:.2f}" for t, r in PRACTICE))
    D = d["D_sine"]; i = np.argmax(D)
    print(f"  sine: max D = {D[i]:.3f} at tau_a/T = {R[i]:.2f}; D(1) = {np.interp(1, R, D):.3f}, "
          f"D(2) = {np.interp(2, R, D):.3f}, D(5) = {np.interp(5, R, D):.3f}, D(10) = {np.interp(10, R, D):.3f}")
    for k, _ in kinds:
        m = d[f"D_{k}"] * PEAK_FACTOR[k]
        print(f"  {k:10s}: max D {d[f'D_{k}'].max():.3f}; per mean acceleration at r = 1, 2, 5: "
              + ", ".join(f"{np.interp(r, R, m):.2f}" for r in (1, 2, 5)))
    print("  damping: max D " + ", ".join(f"zeta1 = {z}: {d[f'D_z{z:g}'].max():.3f}" for z in (0.0, 0.02, 0.05, 0.1)))
    print(f"  take-up travel max {d['Dfree_sine'].max():.2f}; resistances D_r(1, 2, 5) = "
          + ", ".join(f"{np.interp(r, R, d['D_res']):.3f}" for r in (1, 2, 5)))
    rc = d["rc"]
    for r0 in (0.5, 0.8):
        ok = rc >= r0 - 1e-9
        print(f"  collapse, tau_a/T_s >= {r0}: entry within {d['dev_in'][:, ok].max():.1%} "
              f"(per gamma: " + ", ".join(f"{v:.1%}" for v in d["dev_in"][:, ok].max(1))
              + f"), exit within {d['dev_out'][ok].max():.1%}")
    print(f"  collapse, fastest start (tau_a/T_s = {rc[0]:.2f}): entry " +
          ", ".join(f"{v:.0%}" for v in d["dev_in"][:, 0]) + f"; exit {d['dev_out'][0]:.1%}")


# ============================================================================ crawl
def data_crawl():
    import crawl_start as cs
    trs = np.r_[0.0, np.linspace(0.05, 2.0, 40)]
    tps = np.linspace(0.0, 3.0, 61)
    out = dict(trs=trs, tps=tps)
    for rho in (1.0, 5.0):
        lp, b = cs.strand(rho)
        for z in (0.01, 0.1):
            out[f"ramp_{rho:g}_{z:g}"] = np.array([cs.peaks(lp, b, rho, tr, 0.0, z) for tr in trs])
            out[f"hold_{rho:g}_{z:g}"] = np.array([cs.peaks(lp, b, rho, 0.05, tp, z) for tp in tps])
    return out


def fig_crawl(d):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(style.DOUBLE, 58 * style.MM), sharey=True,
                            constrained_layout=True)
    # rho by colour and marker, zeta_1 by line style (readable without colour)
    sty = {(1.0, 0.01): (style.OKABE_ITO[0], "-", "o"), (1.0, 0.1): (style.OKABE_ITO[0], "--", "o"),
           (5.0, 0.01): (style.OKABE_ITO[1], "-", "s"), (5.0, 0.1): (style.OKABE_ITO[1], "--", "s")}
    for (rho, z), (c, ls, mk) in sty.items():
        lab = rf"$\rho = {rho:g}$, $\zeta_1 = {z:g}$"
        kw = dict(color=c, ls=ls, marker=mk, ms=2.8, mfc="w", mew=0.7, label=lab)
        axs[0].plot(d["trs"], d[f"ramp_{rho:g}_{z:g}"][:, 0], markevery=(2, 5), **kw)
        axs[1].plot(d["tps"], d[f"hold_{rho:g}_{z:g}"][:, 0], markevery=(3, 6), **kw)
    axs[0].set(xlabel=r"crawl ramp $\tau_r / \tau_1$ (no hold)", ylabel=r"entry peak / $(R + m a_m)$",
               xlim=(0, 2), ylim=(0.95, 1.75))
    axs[1].set(xlabel=r"hold $\tau_p / \tau_1$ (ramp $\tau_r = 0.05\,\tau_1$)", xlim=(0, 3))
    axs[0].legend(loc="upper right")
    for ax, lab in zip(axs, ("(a)", "(b)")):
        ax.axhline(1.0, color="k", lw=0.5)
        ax.grid(alpha=0.25, lw=0.4)
        style.panel_label(ax, lab)
    style.save(fig, "fig_crawl")
    for rho in (1.0, 5.0):
        for z in (0.01, 0.1):
            r = d[f"ramp_{rho:g}_{z:g}"]; h = d[f"hold_{rho:g}_{z:g}"]
            print(f"  rho {rho:g}, zeta1 {z:g}: abrupt {r[0, 0]:.3f}, ramp 0.5 T1 "
                  f"{np.interp(0.5, d['trs'], r[:, 0]):.3f}, ramp 1 T1 {np.interp(1.0, d['trs'], r[:, 0]):.3f}; "
                  f"hold range {h[:, 0].min():.3f}-{h[:, 0].max():.3f}; travel abrupt {r[0, 1]:.2f}, "
                  f"ramp 1 T1 {np.interp(1.0, d['trs'], r[:, 1]):.3f}")


# ============================================================================ validity
def data_validity():
    from beltloop import Loop, natural_frequencies
    import validity as va
    ext, Dm1 = va.curves()
    out = dict(R=va.R, Dm1=Dm1, rhos=np.array(va.RHOS))
    for rho in va.RHOS:
        out[f"cmax_{rho:g}"], out[f"cmin_{rho:g}"] = ext[rho]
    gam, rho = 2.0, 1.0
    xis = np.linspace(0.02, 0.98, 49)
    for r in (1.0, 3.0):
        rows = []
        for xi in xis:
            lp = Loop.from_positions(0.0, xi, gam)
            T1 = 2 * np.pi / natural_frequencies(lp, 1e-3, 1)[0]
            e, b, gr, _ = va.requirement(lp, r * T1, rho, ext[rho])
            rows.append((e, b, *gr))
        out[f"req_{r:g}"] = np.array(rows) / (1 + gam ** 2)
    out["xis"] = xis
    sig_d, tau_a = 0.5, 20.0
    sts = np.r_[np.linspace(0.03, 0.47, 12), np.linspace(0.53, 0.97, 12)]
    rows = []
    for st in sts:
        lp = Loop.from_positions(sig_d, st, gam, rho, rho * gam ** 2)
        T2, mA = va.exact_requirement(lp, tau_a, rho, va.EULERS[1])
        rows.append((T2 + rho * mA, rho * mA))
    out["sts"], out["tight"] = sts, np.array(rows) / (1 + gam ** 2)
    return out


def fig_validity(d):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    R = d["R"]
    fig, axs = plt.subplots(2, 2, figsize=(style.DOUBLE, 112 * style.MM), constrained_layout=True)
    ax = axs[0, 0]
    me = np.unique(np.linspace(4, len(R) - 3, 9).astype(int))   # R is log-spaced: even in log
    for rho, c, mk in zip(d["rhos"], style.OKABE_ITO, style.MARKERS):
        ax.plot(R, d[f"cmax_{rho:g}"] - rho, color=c, marker=mk, markevery=list(me), ms=2.8,
                mfc="w", mew=0.7, label=rf"$\rho = {rho:g}$")
        ax.axhline(np.sqrt(1 + rho ** 2 / 4) - rho / 2, color=c, ls=":", lw=0.8)
    ax.set(xscale="log", xlim=(R[0], R[-1]), ylim=(0, 4.6), xlabel=r"$\tau_a / \tau_{A1}$",
           ylabel=r"required $T_2 / (m_A a_m)$, exit")
    ax.legend(loc="upper right")
    ax = axs[0, 1]
    for rho, c, mk in zip(d["rhos"], style.OKABE_ITO, style.MARKERS):
        ax.plot(R, np.maximum(-d[f"cmin_{rho:g}"], 0.0), color=c, marker=mk, markevery=list(me),
                ms=2.8, mfc="w", mew=0.7, label=rf"$\rho = {rho:g}$")
    ax.plot(R, d["Dm1"], "k--", lw=0.8, label=r"$\rho = 0$, $\zeta_1 = 0.01$")
    ax.plot(R, 0.8 / R, "k:", lw=0.8, label=r"$0.8\,\tau_1/\tau_a$")
    ax.set(xscale="log", xlim=(R[0], R[-1]), ylim=(0, 1.6), xlabel=r"$\tau_a / \tau_1$",
           ylabel=r"rebound below running $/ (m_B a_m)$")
    ax.legend(loc="upper right", fontsize=6)
    ax = axs[1, 0]
    names = ("exit (A)", "rebound (B)", r"grip, $\mathrm{e}^{\mu\theta} = 3$",
             r"grip, $\mathrm{e}^{\mu\theta} = 16$")
    for k, (nm, c, mk) in enumerate(zip(names, style.OKABE_ITO, style.MARKERS)):
        for r, ls in ((1.0, "-"), (3.0, "--")):
            v = np.maximum(d[f"req_{r:g}"][:, k], 0.0)
            ax.plot(d["xis"], v, color=c, ls=ls)
            sel = np.zeros(len(v), bool)
            sel[2 + 2 * k::8] = True
            sel &= v > 2e-3                      # no markers where the requirement vanishes
            ax.plot(d["xis"][sel], v[sel], ls="none", color=c, marker=mk, ms=2.8, mfc="w", mew=0.7)
    h1 = [Line2D([], [], color=c, marker=mk, ms=2.8, mfc="w", mew=0.7, label=nm)
          for nm, c, mk in zip(names, style.OKABE_ITO, style.MARKERS)]
    h2 = [Line2D([], [], color="k", ls=ls, label=lab) for ls, lab in
          (("-", r"$\tau_a = \tau_1$"), ("--", r"$\tau_a = 3\tau_1$"))]
    blank = [Line2D([], [], ls="none", label=" ")] * (len(h1) - len(h2))
    ax.legend(handles=h1 + h2 + blank, loc="upper left", bbox_to_anchor=(0.02, 0.62), fontsize=6,
              ncol=2, columnspacing=1.2)          # column 1: conditions; column 2: start times
    ax.set(xlim=(0, 1), ylim=(0, 1.4), xlabel=r"take-up position $\xi$ (head drive)",
           ylabel=r"minimum $T_2 / (m_\mathrm{belt} a_m)$")
    ax = axs[1, 1]
    sts, T = d["sts"], d["tight"]
    left = sts < 0.5
    ax.axvspan(0, 0.5, color=style.OKABE_ITO[1], alpha=0.08, lw=0)
    ax.plot(sts[left], T[left, 0], "o-", ms=2.5, color=style.OKABE_ITO[1], label=r"required $T_t$ ($\mathrm{e}^{\mu\theta} = 16$, $\tau_a = 20$)")
    ax.plot(sts[~left], T[~left, 0], "s-", ms=2.5, color=style.OKABE_ITO[0])
    ax.plot(sts[left], T[left, 1], "k:", lw=0.9, label="resistance, drive exit to take-up")
    ax.plot(sts[~left], T[~left, 1], "k:", lw=0.9)
    ax.axvline(0.5, color="k", lw=0.6)
    ax.text(0.25, 2.25, "tight side", ha="center", fontsize=7)
    ax.text(0.75, 2.25, "slack side", ha="center", fontsize=7)
    ax.set(xlim=(0, 1), ylim=(0, 2.45), xlabel=r"take-up position from the head $\sigma_t$ (drive at $\sigma_d = 0.5$)",
           ylabel=r"take-up tension $T_t / (m_\mathrm{belt} a_m)$")
    ax.legend(loc="center right", fontsize=6)
    for ax, lab in zip(axs.ravel(), ("(a)", "(b)", "(c)", "(d)")):
        ax.grid(alpha=0.25, lw=0.4, which="both")
        style.panel_label(ax, lab)
    style.save(fig, "fig_validity")
    for r in (1.0, 3.0):
        q = d[f"req_{r:g}"]
        print(f"  tau_a = {r:g} T1, gamma 2, rho 1: T2_min/(m_belt a_m) exit {q[:, 0].min():.2f}-{q[:, 0].max():.2f}, "
              f"rebound {q[:, 1].min():.2f}-{q[:, 1].max():.2f}, grip E=3 {q[:, 2].min():.2f}-{q[:, 2].max():.2f}, "
              f"E=16 {q[:, 3].min():.2f}-{q[:, 3].max():.2f}")
    print(f"  tight side T_t {T[left, 0].min():.2f}-{T[left, 0].max():.2f}; slack side "
          f"{T[~left, 0].min():.2f}-{T[~left, 0].max():.2f} (m_belt a_m)")


def data_applications():
    import applications as ap
    return ap.compute()


def fig_applications(d):
    import applications as ap
    ap.figure(d, style.OUT / "fig_applications.pdf")
    print("figure:", style.OUT / "fig_applications.pdf")
    print(ap.write_table(ap.real_alt_rows(), style.OUT / "table_applications.txt"))


FIGS = dict(modal=(data_modal, fig_modal), beta=(data_beta, fig_beta), startup=(data_startup, fig_startup),
            crawl=(data_crawl, fig_crawl), validity=(data_validity, fig_validity),
            applications=(data_applications, fig_applications))


def main(argv):
    recompute = "--recompute" in argv
    names = [a for a in argv if not a.startswith("--")] or list(FIGS)
    style.apply()
    for name in names:
        compute, plot = FIGS[name]
        print(f"[{name}]")
        plot(cached(name, compute, recompute))


if __name__ == "__main__":
    main(sys.argv[1:])
