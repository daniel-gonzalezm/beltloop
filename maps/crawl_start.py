"""Phase 4.5b: does an initial crawl (plateau) help? Linear model, strand picture.

Start: constant acceleration to the crawl speed v_p V_inf over tau_r, hold tau_p, parabolic main
acceleration (peak a_m) over tau_a (beltloop.crawl_start). Resistances, Coulomb-like, appear as
the belt starts to move: phi = min(1, tau/tau_r) (crawl_onset); tau_r -> 0 is a step onset (the
linear model's upper bound: in reality the onset follows the stretch front). Practice: Harrison
(1985b) steps to ~70 % speed, then ~50 s (~2 T1); ZISCO ramps to 4 % in 20 s and holds 60 s;
Henderson PC2 holds 60 s at 10 %.

Load ratio rho = R / (m a_m): running resistance over inertial force of the strand (Lodewijks
1996 ch. 8 ~ 0.7; long, slow starts such as KPC ~ 5-10). Metrics: peak fixed-end (drive entry)
tension over R + m a_m, and peak free-end displacement (take-up travel) over its quasi-static
running value plus inertial peak. Uniform strand (exact for strand A, within a few % for strand
B, phase 4.5), plus a loop check.

Run:  python maps/crawl_start.py   (prints tables, writes maps/figures/crawl_start.pdf)
"""
from pathlib import Path

import numpy as np

from beltloop import (Loop, Segment, crawl_onset, crawl_start, modal_basis, startup_with_onset)

HERE = Path(__file__).parent
OUT = HERE / "figures"
TS = 4.0                      # fundamental period of the unit strand
TA = 3.0                      # main acceleration, in units of T_s
VP = 0.05                     # crawl speed / final speed


def strand(rho):
    lp = Loop((Segment(1e-3, r=rho),), (Segment(1.0, r=rho),))
    return lp, modal_basis(lp, 1e-8, 60)


def peaks(lp, b, rho_tot, tr, tp, zeta1, ta=TA, vp=VP, x=None, after=3.0):
    """Normalised peaks (entry tension, take-up travel) for a crawl start; tr, tp, ta in T_1."""
    T1 = 2 * np.pi / b.Om[0]
    tr_ = max(tr, 1e-3) * T1
    prof = crawl_start(tr_, tp * T1, ta * T1, vp if tr > 0 else 0.0)
    tend = prof.tau_end + after * T1
    tau = np.linspace(0.0, tend, int(tend / (T1 / 300)) + 1)
    x = np.array([lp.length]) if x is None else x
    T, y = startup_with_onset(b, prof, crawl_onset(tr_), zeta1 / b.Om[0], tau, x)
    y_ref = 0.5 * (lp.quasi_static_integral("mu") + lp.quasi_static_integral("r"))
    q_ref = lp.quasi_static("mu", x)[0] + lp.quasi_static("r", x)[0]
    return T[0].max() / q_ref, y.max() / y_ref


def main():
    rows = {}
    trs = np.r_[0.0, np.linspace(0.1, 2.0, 20)]
    tps = np.linspace(0.0, 3.0, 13)
    for rho in (1.0, 5.0):
        lp, b = strand(rho)
        for z in (0.01, 0.1):
            ramp = np.array([peaks(lp, b, rho, tr, 0.0, z) for tr in trs])
            hold = {tr: np.array([peaks(lp, b, rho, tr, tp, z) for tp in tps]) for tr in (0.05, 0.5, 1.0)}
            rows[(rho, z)] = (ramp, hold)
            print(f"rho = {rho}, zeta1 = {z}: entry peak, abrupt onset {ramp[0, 0]:.3f}; "
                  f"crawl ramp 0.5 T1 {np.interp(0.5, trs, ramp[:, 0]):.3f}; 1 T1 "
                  f"{np.interp(1.0, trs, ramp[:, 0]):.3f}; 2 T1 {ramp[-1, 0]:.3f}  | travel "
                  f"{ramp[0, 1]:.3f} / {np.interp(1.0, trs, ramp[:, 1]):.3f}")
            for tr, h in hold.items():
                print(f"     ramp {tr:4.2f} T1, hold 0 / 1 / 2 / 3 T1: entry "
                      f"{h[0, 0]:.3f} {h[4, 0]:.3f} {h[8, 0]:.3f} {h[12, 0]:.3f}")
    # loop check: gamma = 2, take-up near the head drive, resistances 1 : 3 (return : carry)
    lp = Loop.from_positions(0.0, 0.05, 2.0, r_return=0.4, r_carry=1.2)
    b = modal_basis(lp, 1e-3, 60)
    for tr, tp in [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.05, 2.0)]:
        e, t = peaks(lp, b, None, tr, tp, 0.01, x=np.array([2.0]))
        print(f"loop gamma = 2, xi = 0.05: ramp {tr} T1, hold {tp} T1 -> entry {e:.3f}, travel {t:.3f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    OUT.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    styles = {(1.0, 0.01): ("C0", "-"), (1.0, 0.1): ("C0", "--"), (5.0, 0.01): ("C3", "-"), (5.0, 0.1): ("C3", "--")}
    for key, (ramp, hold) in rows.items():
        c, ls = styles[key]
        lab = rf"$\rho$ = {key[0]:g}, $\zeta_1$ = {key[1]:g}"
        axs[0].plot(trs, ramp[:, 0], color=c, ls=ls, label=lab)
        axs[1].plot(tps, hold[0.05][:, 0], color=c, ls=ls, label=lab)
        axs[2].plot(trs, ramp[:, 1], color=c, ls=ls, label=lab)
    axs[0].set(xlabel=r"crawl ramp $\tau_r/T_1$ (no hold)", ylabel=r"entry peak / $(R + m a_m)$",
               title="(a) duration of the onset", ylim=(0.9, 1.8))
    axs[1].set(xlabel=r"hold $\tau_p/T_1$ (abrupt crawl, $\tau_r = 0.05\,T_1$)",
               title="(b) duration of the hold", ylim=(0.9, 1.8))
    axs[2].set(xlabel=r"crawl ramp $\tau_r/T_1$ (no hold)", ylabel="take-up travel / quasi-static",
               title="(c) take-up travel", ylim=(0.9, 1.9))
    for ax in axs:
        ax.axhline(1, color="k", lw=0.6)
        ax.legend(fontsize=7)
    fig.savefig(OUT / "crawl_start.pdf")
    fig.savefig(OUT / "crawl_start.png", dpi=150)
    print("figure:", OUT / "crawl_start.pdf")


if __name__ == "__main__":
    main()
