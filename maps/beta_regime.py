"""Phase 4.2: where does the take-up mass matter?

(1) Map of beta_5(xi, gamma): the belt-side mass ratio at which the k-th natural period
    departs 5 % from its beta -> 0 value (two decoupled fixed-free strands), k = 1, 2, 3,
    head drive, exact roots. (2) Table of beta and of the period shifts for the conveyors
    in the literature, exact and with the two-pole approximation.

Run:  python maps/beta_regime.py   (prints the table, writes maps/figures/beta_regime.pdf)
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from beltloop import Loop, natural_frequencies, takeup_mass_approx

G_STD = 9.80665
HERE = Path(__file__).parent
OUT = HERE / "figures"
TOL = 0.05

# name, L [m], mu_r, mu_c [kg/m], take-up mass M [kg], strands n, xi values, note
#
# Li and Li (2009): 7600 m, ST-2000 1400 mm, belt 54 kg/m, "elasticity 1300 kN/cm" read as per cm
# of width (EA = 182 MN), 2500 t/h at 4 m/s (173.6 kg/m), idlers 159 mm at 1.2 m (carry) and 3 m
# (return), 42 800 kg take-up next to the second (tandem) head drive pulley. Their theoretical
# wave speed c = sqrt(E/m) = 837 m/s implies m = 259.8 kg/m on the carrying strand, i.e. 32.2 kg/m
# of carrying idlers; the return idlers (same roll length per set, 3 m pitch) are scaled to 12.9
# kg/m, so mu_r = 66.9 kg/m. n = 2 is supported by their Fig. 4 (about 200 kN on the slack side
# versus 42.8 t * g / 2 = 210 kN; the "20 kN" of their text is read as a typo). Varying the idler
# estimate by +-50 % moves gamma within 1.94-2.01 and beta within 0.077-0.093.
CASES = [
    ("Harrison 1983/85 (79 kg/m)", 5100, 79.0, 79.0 * 0.97 ** 2, 20e3, 4, [0.005], "gamma = 0.97 measured"),
    ("Harrison 1983/85 (39 kg/m)", 5100, 39.0, 39.0 * 0.97 ** 2, 20e3, 4, [0.005], "density of the text"),
    ("Song et al. 2012", 7117, 37.8, 104.9, 4500, 2, [0.001, 0.999], "head (Fig. 1) / tail (model)"),
    ("Gao et al. 2026", 4500, 40.1, 194.3, 1000, 2, [0.001], "no idler mass given"),
    ("Li and Li 2009 (AMESim)", 7600, 66.9, 259.8, 42800, 2, [0.005], "idlers from their c = 837 m/s"),
    ("Lodewijks 1996, ch. 8", 1000, 21.23, 161.2, 42.66e3 / G_STD, 2, [0.001], "M from the take-up force"),
    ("Belt A (Pascual et al. 2005)", 2561, 138.0, 472.0, 45.5e3, 2, [0.999], "n not stated"),
    ("Belt C (Wheatley and Rubel 2021)", 274.6, 31.09, 266.2, 7550, 2, [0.05, 0.5, 0.999], "take-up position unknown"),
]


def period_shift(loop, beta, k):
    """Relative increase of the k-th period (k from 0) with respect to beta -> 0."""
    return natural_frequencies(loop, 0.0, k + 1)[k] / natural_frequencies(loop, beta, k + 1)[k] - 1


def beta_threshold(xi, gamma, k, tol=TOL):
    lp = Loop.from_positions(0.0, xi, gamma)
    f = lambda lb: period_shift(lp, 10.0 ** lb, k) - tol
    lo, hi = -4.0, 2.0
    if f(hi) < 0:
        return np.nan            # never reaches tol (e.g. a mode pinned at a coincident pole)
    return 10.0 ** brentq(f, lo, hi, xtol=2e-3)


def case_table():
    print(f"{'case':34s}{'n':>3}{'beta':>8}{'gamma':>7}{'xi':>7}"
          f"{'dT1':>8}{'dT1 ap':>8}{'dT2':>8}{'dT2 ap':>8}  note")
    rows = []
    for name, L, mr, mc, M, n, xis, note in CASES:
        beta = 4 * M / (n * n * mr * L)
        gam = np.sqrt(mc / mr)
        for xi in xis:
            lp = Loop.from_positions(0.0, xi, gam)
            w0 = natural_frequencies(lp, 0.0, 2)
            w = natural_frequencies(lp, beta, 2)
            wa = takeup_mass_approx(lp, beta, 2)
            d, da = w0 / w - 1, w0 / wa - 1
            print(f"{name:34s}{n:3d}{beta:8.3f}{gam:7.2f}{xi:7.3f}"
                  f"{d[0]:8.2%}{da[0]:8.2%}{d[1]:8.2%}{da[1]:8.2%}  {note}")
            rows.append((name, beta, gam, xi))
    return rows


def main():
    rows = case_table()
    xis = np.linspace(0.01, 0.99, 41)
    gams = np.linspace(1.0, 3.0, 17)
    maps = np.array([[[beta_threshold(x, g, k) for x in xis] for g in gams] for k in range(3)])
    for k in range(3):
        m = maps[k]
        print(f"mode {k + 1}: beta_5 from {np.nanmin(m):.3f} to {np.nanmax(m):.3f}; "
              f"median {np.nanmedian(m):.3f}; never reached at {np.isnan(m).sum()} of {m.size} points")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        return
    OUT.mkdir(exist_ok=True)
    levels = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, constrained_layout=True)
    for k, ax in enumerate(axs):
        cs = ax.contourf(xis, gams, maps[k], levels=levels, norm=LogNorm(), cmap="viridis", extend="both")
        cl = ax.contour(xis, gams, maps[k], levels=levels, colors="k", linewidths=0.5)
        ax.clabel(cl, fmt="%g", fontsize=7)
        for name, beta, gam, xi in rows:
            if 1.0 <= gam <= 3.0:
                ax.plot(xi, gam, "o", ms=4, mfc="w", mec="k")
                ax.annotate(f"{beta:.2g}", (xi, gam), xytext=(3, 3), textcoords="offset points", fontsize=6)
        ax.set(xlabel=r"take-up position $\xi$", title=f"mode {k + 1}")
    axs[0].set_ylabel(r"wave-speed ratio $\gamma = c_r/c_c$")
    fig.colorbar(cs, ax=axs, label=r"$\beta$ for a 5 % period shift")
    fig.savefig(OUT / "beta_regime.pdf")
    fig.savefig(OUT / "beta_regime.png", dpi=150)
    print("figure:", OUT / "beta_regime.pdf")


if __name__ == "__main__":
    main()
