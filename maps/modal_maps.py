"""Phase 4.4: natural frequencies and modal participation in the (xi, gamma) plane.

Head drive, take-up at xi on the return strand, limit beta -> 0: the take-up is a free end
and the loop modes are the fixed-free modes of the upstream strand A (drive exit -> take-up,
uniform, length xi) and of the downstream strand B (take-up -> tail -> carry -> drive
entry). Each mode is followed by its strand (loop ordering puts kinks and jumps in the maps
where modes of the two strands cross). Strand A is in closed form,
    Om_Aj = (2j - 1) pi / (2 xi),   m_eff,Aj = 8 xi / ((2j - 1)^2 pi^2);
strand B solves tan(Om (1 - xi)) tan(gamma Om) = gamma. For gamma >= 1 the fundamental is
always B1. The take-up mass enters as the two-pole correction (beta_regime.py, eigen.takeup_
mass_modes); the hatched corner (gamma near 1, take-up towards the tail, where B1 and A1 are
close) is where beta = 0.1 changes the fundamental's participation by more than 5 %: there the
take-up exchanges participation between B1 and A1.

Panels: (a) fundamental period T1 c_r/L; (b) T1 / (4 t_B), t_B = (1 - xi) + gamma the
downstream transit time (the quarter-wave transit estimate is exact for gamma = 1 and at most
16 % long); (c) effective-mass fraction of the fundamental, Gamma_1^2 / (1 + gamma^2);
(d) effective-mass fractions of B1, B2 and A1 along xi for gamma = 1, 2, 3, with the position
xi_s where A1 overtakes B2 as the second loop mode.

Run:  python maps/modal_maps.py   (prints the case table, writes maps/figures/modal_maps.pdf)
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from beltloop import Loop, modal_basis, strand_participation, takeup_mass_modes

from beta_regime import CASES

HERE = Path(__file__).parent
OUT = HERE / "figures"
BETA_REF = 0.1          # reference take-up mass ratio for the hatched corner
DF_TOL = 0.05           # relative change of the fundamental's participation flagged at BETA_REF


def strands(xi, gamma, n=2):
    lp = Loop.from_positions(0.0, xi, gamma)
    a = strand_participation(lp.upstream, n)
    b = strand_participation(lp.downstream[::-1], n)
    return lp, a, b


def switch_position(gamma):
    """xi_s where A1 = pi/(2 xi) equals B2 (loop mode 2 changes strand); None if xi_s >= 1."""
    f = lambda x: np.pi / (2 * x) - strands(x, gamma)[2][0][1]
    lo, hi = 0.05, 1 - 1e-9
    if f(hi) >= 0:
        return None
    return brentq(f, lo, hi, xtol=1e-10)


def fields(xis, gams):
    T1 = np.empty((len(gams), len(xis))); ratio = T1.copy(); F1 = T1.copy(); dF1 = T1.copy()
    for i, g in enumerate(gams):
        for j, x in enumerate(xis):
            lp, (pa, ma), (pb, mb) = strands(x, g)
            T1[i, j] = 2 * np.pi / pb[0]
            ratio[i, j] = T1[i, j] / (4 * ((1 - x) + g))
            F1[i, j] = mb[0] / lp.belt_mass
            dF1[i, j] = modal_basis(lp, BETA_REF, 2).effective_mass_fraction[0] / F1[i, j] - 1
    return T1, ratio, F1, dF1


def case_table():
    print(f"{'case':34s}{'beta':>7}{'gamma':>7}{'xi':>7}{'T1':>8}{'T1/4tB':>8}"
          f"{'F1(0)':>7}{'F1(b)':>7}{'F1 2p':>7}{'F_A1':>7}  mode 2")
    rows = []
    for name, L, mr, mc, M, n, xis, note in CASES:
        beta = 4 * M / (n * n * mr * L)
        gam = np.sqrt(mc / mr)
        for xi in xis:
            lp, (pa, ma), (pb, mb) = strands(xi, gam)
            bm = lp.belt_mass
            T1 = 2 * np.pi / min(pa[0], pb[0])
            ex = modal_basis(lp, beta, 2).effective_mass_fraction[0]
            tp = takeup_mass_modes(lp, beta, 1)[1][0] / bm
            second = "A1" if pa[0] < pb[1] and pb[0] < pa[0] else ("B2" if pb[0] < pa[0] else "B1")
            print(f"{name:34s}{beta:7.3f}{gam:7.2f}{xi:7.3f}{T1:8.3f}{T1 / (4 * (1 - xi + gam)):8.3f}"
                  f"{mb[0] / bm:7.3f}{ex:7.3f}{tp:7.3f}{ma[0] / bm:7.3f}  {second}")
            rows.append((name, beta, gam, xi))
    print("T1 in units of L/c_r; F = effective-mass fraction of the belt mass; F1(0): beta -> 0;"
          " F1(b): exact with the case's beta; F1 2p: two-pole approximation.")
    return rows


def main():
    rows = case_table()
    xis = np.r_[0.002, np.linspace(0.01, 0.99, 50), 0.998]
    gams = np.linspace(1.0, 3.0, 41)
    T1, ratio, F1, dF1 = fields(xis, gams)
    print(f"T1 c_r/L: {T1.min():.2f} to {T1.max():.2f}; T1/(4 t_B): {ratio.min():.3f} to {ratio.max():.3f}; "
          f"F1: {F1.min():.3f} to {F1.max():.3f}")
    print(f"beta = {BETA_REF}: relative change of F1 above {DF_TOL:.0%} at {np.mean(np.abs(dF1) > DF_TOL):.1%} "
          f"of the grid (max {np.abs(dF1).max():.0%}); median {np.median(np.abs(dF1)):.1%}")
    gl = np.linspace(1.0, 3.0, 21)
    xs = [switch_position(g) for g in gl]
    for g, x in zip(gl[::4], xs[::4]):
        print(f"gamma = {g:.1f}: mode 2 switches from B2 to A1 at xi_s = " + (f"{x:.3f}" if x else "> 1"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    OUT.mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.6), constrained_layout=True)

    def mark_cases(ax):
        for name, beta, gam, xi in rows:
            if 0.95 <= gam <= 3.05 and "Belt C" not in name:
                ax.plot(xi, min(max(gam, 1.0), 3.0), "o", ms=4, mfc="w", mec="k", zorder=5)
        ax.plot([0.05, 0.999], [2.93, 2.93], "k:", lw=1)            # belt C, position unknown

    def xs_line(ax):
        ok = [(x, g) for x, g in zip(xs, gl) if x is not None]
        ax.plot([p[0] for p in ok], [p[1] for p in ok], "w--", lw=1.2)

    panels = [(axs[0, 0], T1, np.arange(4, 14.5, 1.0), "viridis", r"(a) fundamental period $T_1 c_r/L$"),
              (axs[0, 1], ratio, np.arange(0.83, 1.001, 0.02), "magma", r"(b) $T_1/(4t_B)$, $t_B = 1-\xi+\gamma$"),
              (axs[1, 0], F1, np.arange(0.40, 0.82, 0.05), "cividis", r"(c) effective-mass fraction of mode 1")]
    for ax, Z, lev, cmap, title in panels:
        cs = ax.contourf(xis, gams, Z, levels=lev, cmap=cmap, extend="both")
        cl = ax.contour(xis, gams, Z, levels=lev[::2], colors="k", linewidths=0.5)
        ax.clabel(cl, fmt="%.2f" if Z is not T1 else "%.0f", fontsize=7)
        fig.colorbar(cs, ax=ax)
        mark_cases(ax)
        xs_line(ax)
        ax.set(xlabel=r"take-up position $\xi$", ylabel=r"$\gamma = c_r/c_c$", title=title)
    axs[1, 0].contourf(xis, gams, np.abs(dF1), levels=[DF_TOL, 10], colors="none", hatches=["////"])
    axs[1, 0].contour(xis, gams, np.abs(dF1), levels=[DF_TOL], colors="w", linewidths=0.8)

    ax = axs[1, 1]
    xl = np.linspace(0.005, 0.995, 120)
    for g, c in ((1.0, "C0"), (2.0, "C1"), (3.0, "C2")):
        fb = np.array([strands(x, g)[2][1] for x in xl]) / (1 + g * g)
        ax.plot(xl, fb[:, 0], "-", color=c, label=rf"B1, $\gamma$ = {g:g}")
        ax.plot(xl, fb[:, 1], ":", color=c, label="B2")
        ax.plot(xl, 8 * xl / (np.pi ** 2 * (1 + g * g)), "--", color=c, label="A1")
        x = switch_position(g)
        if x is not None:
            ax.axvline(x, color=c, lw=0.6, alpha=0.6)
    ax.set(xlabel=r"take-up position $\xi$", ylabel="effective-mass fraction", ylim=(0, 0.85),
           title=r"(d) strand modes ($\beta\to 0$); vertical: $\xi_s$")
    ax.legend(fontsize=7, ncol=3, loc="center left", bbox_to_anchor=(0.0, 0.40))
    fig.savefig(OUT / "modal_maps.pdf")
    fig.savefig(OUT / "modal_maps.png", dpi=150)
    print("figure:", OUT / "modal_maps.pdf")


if __name__ == "__main__":
    main()
