"""Phase 4.5: start-up tension and take-up travel through the universal strand curves.

With beta -> 0 each strand (fixed at the drive, free at the take-up) responds on its own, so
the drive-entry peak follows the curve of strand B in r = tau_a / T_1, the drive-exit minimum
follows the same curve in r_A = tau_a / T_A1 (T_A1 = 4 xi L / c_r), and the take-up travel is
governed by the free-end displacement of strand B. In SI (inertial part, peak acceleration a_m):
    T_entry,peak ~ D(tau_a/T_1) * a_m * (mass of strand B)
    T_exit,min   ~ -D(tau_a/T_A1) * a_m * mu_r * xi * L
    y_max        ~ (D_free(tau_a/T_1) S_B - S_A) / 2 * a_m L^2 / c_r^2
Panels: (a) D for the three profiles, scaled by the peak acceleration, with the two limits
(wave law T = Z V and quasi-static T = m a); (b) the same per unit mean acceleration V/t_a, the
fair comparison at equal start time; (c) damping; (d) collapse of the loop on the curve, entry
and exit, over (xi, gamma); (e) free-end displacement (take-up travel) and resistance onset.

Run:  python maps/startup_curves.py   (prints tables, writes maps/figures/startup_curves.pdf)
"""
from pathlib import Path

import numpy as np

from beltloop import (PEAK_FACTOR, Loop, StartProfile, fast_start_limit, modal_basis,
                      natural_frequencies, startup_metrics, startup_response, strand_curves)

HERE = Path(__file__).parent
OUT = HERE / "figures"
R = np.logspace(np.log10(0.1), np.log10(20), 70)
KINDS = ("sine", "triangular", "parabolic")
ZETAS = (0.0, 0.02, 0.05, 0.1)

# Start time over fundamental period for cases with published start time (illustration only):
# Lodewijks 1996 ch. 8 (30 s, T1 = 28.5 s from this model, linear profile in the source);
# Song et al. 2012 (300 s, T1 = 40.0 s with the take-up at the head);
# KPC, Sinaga 2008 (780 s, T1 ~ 66-73 s estimated in the phase 4.4 review).
CASES = [("Lodewijks 1996, ch. 8", 30.0 / 28.5), ("Song et al. 2012 (head)", 300.0 / 40.0),
         ("KPC (Sinaga 2008)", 780.0 / 70.0)]


def curves():
    out = {k: strand_curves(R, kind=k) for k in KINDS}
    damp = {z: strand_curves(R, zeta1=z)[0] for z in ZETAS}
    res = strand_curves(R, load="resistance")[0]
    return out, damp, res


def collapse(rs=(0.3, 0.5, 0.8, 1.2, 2.0, 4.0, 8.0)):
    pts_in, pts_out = [], []
    for r in rs:
        for xi in (0.01, 0.2, 0.5, 0.8, 0.95):
            for g in (1.0, 1.5, 2.0, 3.0):
                lp = Loop.from_positions(0.0, xi, g)
                T1 = 2 * np.pi / natural_frequencies(lp, 1e-3, 1)[0]
                m = startup_metrics(lp, 1e-3, StartProfile("sine", r * T1, "none"))
                pts_in.append((r, m.D_entry, g))
                pts_out.append((r * T1 / m.T_A1, m.D_exit, g))
    return np.array(pts_in), np.array(pts_out)


def main():
    out, damp, res = curves()
    De = out["sine"][0]
    i = np.argmax(De)
    print(f"sine: peak amplification {De[i]:.3f} at tau_a/T_s = {R[i]:.2f}; "
          f"D(1) = {np.interp(1, R, De):.3f}, D(2) = {np.interp(2, R, De):.3f}, "
          f"D(5) = {np.interp(5, R, De):.3f}, D(10) = {np.interp(10, R, De):.3f}")
    for k in KINDS:
        d = out[k][0] * PEAK_FACTOR[k]
        print(f"{k:10s}: max D = {out[k][0].max():.3f}; per unit V/t_a at r = 1, 2, 5: "
              f"{np.interp(1, R, d):.2f}, {np.interp(2, R, d):.2f}, {np.interp(5, R, d):.2f}")
    for z in ZETAS:
        print(f"zeta1 = {z:.2f}: max D = {damp[z].max():.3f}")
    print(f"resistance onset phi = V/V_inf: D_r(1) = {np.interp(1, R, res):.3f}, "
          f"D_r(2) = {np.interp(2, R, res):.3f}, D_r(5) = {np.interp(5, R, res):.3f}")
    for name, r in CASES:
        print(f"{name:28s} tau_a/T1 = {r:5.2f}  D(sine) = {np.interp(r, R, De):.3f}")
    pin, pout = collapse()
    for r in np.unique(pin[:, 0]):
        sel = pin[:, 0] == r
        ref = np.interp(r, R, De)
        print(f"collapse entry tau_a/T1 = {r:4.2f}: D in [{pin[sel, 1].min():.3f}, "
              f"{pin[sel, 1].max():.3f}], curve {ref:.3f}")
    ok = pout[:, 0] <= 20
    err = np.abs(pout[ok, 1] / np.interp(pout[ok, 0], R, De) - 1)
    print(f"collapse exit: max deviation {err.max():.2%} over {ok.sum()} points")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    OUT.mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.6), constrained_layout=True)
    ax = axs[0, 0]
    for k, c in zip(KINDS, ("C0", "C1", "C2")):
        ax.plot(R, out[k][0], color=c, label=k)
        ax.plot(R, fast_start_limit(R, k), ":", color=c, lw=0.8)
    ax.axhline(1, color="k", lw=0.6)
    ax.set(xscale="log", ylim=(0, 2.2), xlabel=r"$\tau_a/T_s$", ylabel="D (peak / quasi-static)",
           title="(a) drive-face tension, per peak acceleration")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    for k, c in zip(KINDS, ("C0", "C1", "C2")):
        ax.plot(R, out[k][0] * PEAK_FACTOR[k], color=c, label=k)
    ax.axhline(1, color="k", lw=0.6)
    ax.set(xscale="log", ylim=(0, 4), xlabel=r"$\tau_a/T_s$", ylabel=r"peak / ($m\,V_\infty/t_a$)",
           title="(b) same, per unit mean acceleration")
    ax.legend(fontsize=8)
    ax = axs[0, 2]
    for z in ZETAS:
        ax.plot(R, damp[z], label=rf"$\zeta_1$ = {z:g}")
    ax.axhline(1, color="k", lw=0.6)
    ax.set(xscale="log", ylim=(0, 2.2), xlabel=r"$\tau_a/T_s$", title="(c) damping (sine)")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    ax.plot(R, De, "k", lw=1, label="uniform strand")
    for g, mk in zip((1.0, 1.5, 2.0, 3.0), ("o", "s", "^", "D")):
        s = pin[:, 2] == g
        ax.plot(pin[s, 0], pin[s, 1], mk, ms=4, mfc="none", label=rf"entry, $\gamma$ = {g:g}")
    ax.set(xscale="log", ylim=(0, 2.2), xlabel=r"$\tau_a/T_1$", ylabel="D at the drive entry",
           title=r"(d) loop, drive entry (all $\xi$)")
    ax.legend(fontsize=7)
    ax = axs[1, 1]
    ax.plot(R, De, "k", lw=1, label="uniform strand")
    sel = pout[:, 0] <= 20
    ax.plot(pout[sel, 0], pout[sel, 1], "o", ms=3, mfc="none", color="C3", label="exit, all cases")
    ax.set(xscale="log", ylim=(0, 2.2), xlabel=r"$\tau_a/T_{A1}$, $T_{A1} = 4\xi L/c_r$",
           ylabel="D at the drive exit", title="(e) loop, drive exit")
    ax.legend(fontsize=8)
    ax = axs[1, 2]
    ax.plot(R, out["sine"][1], label="free-end displacement (take-up)")
    ax.plot(R, res, label=r"resistances, $\varphi = V/V_\infty$")
    ax.axhline(1, color="k", lw=0.6)
    for name, r in CASES:
        ax.axvline(r, color="0.6", lw=0.6, ls="--")
    ax.set(xscale="log", ylim=(0, 2.2), xlabel=r"$\tau_a/T_s$", title="(f) travel and resistances (sine)")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "startup_curves.pdf")
    fig.savefig(OUT / "startup_curves.png", dpi=150)
    print("figure:", OUT / "startup_curves.pdf")


if __name__ == "__main__":
    main()
