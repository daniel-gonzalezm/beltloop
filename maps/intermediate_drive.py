"""Phase 4.7: intermediate drive (drive anywhere on the return strand) in the limit beta -> 0.

With drive and take-up on the return strand, one strand (C) holds the whole carry strand, with
a return segment l1 between the drive and the carry strand (at C's fixed end) and a return
segment l2 between the carry strand and the take-up (at C's free end); the other strand is a
uniform return segment. For a slack-side take-up with the drive l1 = sigma_d from the head,
C is the downstream strand B and l2 = 1 - sigma_t. Results:

  * the fundamental is always C's (gamma >= 1), whichever side of the drive the take-up is on;
  * it solves gamma tan(a) tan(c) + tan(a) tan(b) + tan(b) tan(c)/gamma = 1, with a = Om l1,
    c = gamma Om, b = Om l2, and 4 gamma < T_1 <= 4 t + 4 (gamma - 1) l1, t = gamma + l1 + l2;
  * the return run l1 acts as a spring at the fixed end: for the same strand lengths, moving
    return belt from the free end to the fixed end lengthens T_1 beyond four transits (up to
    1.38 x 4t for gamma = 3) and concentrates the strand's mass in the fundamental (effective
    mass up to the whole strand mass, against 8/pi^2 of it for a uniform strand);
  * the start-up formulas of phase 4.5 still hold with T_1 and m_B of strand B: exit exact
    (strand A uniform), take-up travel within ~1.5 %, entry slightly above the universal curve.

Panels: (a) T_1 / (4 t_B) against the drive offset l1, take-up right after the drive
(xi = 0.01); its value at l1 = 0 is the head drive, so the curves also read as T_1 relative to
a head drive with the same take-up placement and nearly the same 4 t_B; (b) effective mass of
the fundamental over (8/pi^2) m_B; (c) peak dynamic tension at the drive entry over m_B a_m
against tau_a / T_1, for several drive offsets, with the universal curve of a uniform strand.
Nordell and Ciozda's (1984) conveyor (drive 5150 ft from the head of an 8150 ft conveyor,
take-up 300 ft downstream, 1450 and 590 m/s) is marked: l1 = 0.63, gamma = 2.46.

Run:  python maps/intermediate_drive.py  (prints tables, writes maps/figures/intermediate_drive.pdf)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beltloop import Loop, StartProfile, startup_metrics, strand_curves, strand_participation

HERE = Path(__file__).parent
OUT = HERE / "figures"
XI = 0.01                           # take-up right after the drive (loop coordinate)
GAMMAS = (1.0, 1.5, 2.0, 2.5, 3.0)
L1S = np.linspace(0.0, 0.98, 50)
FT = 0.3048
NC = dict(L=8150 * FT, c_r=1450.0, gamma=1450.0 / 590.0, sd=5150 / 8150, st=5450 / 8150)
R = np.r_[np.linspace(0.3, 1.0, 15), np.linspace(1.1, 5.0, 25)]


def strand_B(sd, st, gamma, n=1):
    lp = Loop.from_positions(sd, st, gamma)
    B = lp.downstream[::-1]
    Om, me = strand_participation(B, n)
    tB = sum(s.g * s.length for s in B)
    mB = sum(s.mu * s.length for s in B)
    return lp, Om, me, tB, mB


def offset_curves():
    ratio, part = {}, {}
    for g in GAMMAS:
        rr, pp = [], []
        for l1 in L1S:
            _, Om, me, tB, mB = strand_B(l1, l1 + XI, g)
            rr.append(2 * np.pi / Om[0] / (4 * tB))
            pp.append(me[0] / (8 / np.pi ** 2 * mB))
        ratio[g], part[g] = np.array(rr), np.array(pp)
    return ratio, part


def nordell_ciozda():
    """Fundamental of Nordell and Ciozda's geometry and of the same take-up placement with a
    head drive (beta -> 0)."""
    rows = []
    for label, sd, st in (("intermediate drive", NC["sd"], NC["st"]),
                          ("head drive, same xi", 0.0, NC["st"] - NC["sd"])):
        _, Om, me, tB, mB = strand_B(sd, st, NC["gamma"], 2)
        s = NC["L"] / NC["c_r"]
        rows.append((label, 2 * np.pi / Om[0] * s, 4 * tB * s, 2 * np.pi / Om[0] / (4 * tB),
                     me[0] / (1 + NC["gamma"] ** 2), 2 * np.pi / Om[1] * s))
    return rows


def entry_collapse(cases=((2.0, 0.0), (2.0, 0.2), (2.0, 0.5), (NC["gamma"], NC["sd"]))):
    out = {}
    for g, l1 in cases:
        lp, Om, _, _, mB = strand_B(l1, l1 + XI if l1 != NC["sd"] else NC["st"], g)
        T1 = 2 * np.pi / Om[0]
        out[(g, l1)] = np.array([startup_metrics(lp, 1e-8, StartProfile("sine", r * T1, "none")).entry_peak
                                 / mB for r in R])
    return out


def main():
    OUT.mkdir(exist_ok=True)
    ratio, part = offset_curves()
    print("T_1 / (4 t_B) against the drive offset l1 (take-up right after the drive):")
    print("  l1    " + "  ".join(f"g={g:<4}" for g in GAMMAS))
    for l1 in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        i = np.argmin(abs(L1S - l1))
        print(f"  {L1S[i]:.2f}  " + "  ".join(f"{ratio[g][i]:.3f} " for g in GAMMAS))
    print("Effective mass of the fundamental / ((8/pi^2) m_B):")
    for l1 in (0.0, 0.1, 0.2, 0.5, 0.9):
        i = np.argmin(abs(L1S - l1))
        print(f"  {L1S[i]:.2f}  " + "  ".join(f"{part[g][i]:.3f} " for g in GAMMAS))
    print("\nNordell and Ciozda (1984) geometry, beta -> 0:")
    for label, T1, T4, rt, F1, T2 in nordell_ciozda():
        print(f"  {label:22s} T1 = {T1:5.2f} s, 4 t_B = {T4:5.2f} s (T1/4t_B = {rt:.3f}), "
              f"F1 = {F1:.3f}, T2 = {T2:.2f} s")

    D, _ = strand_curves(R)
    coll = entry_collapse()
    print("\nDrive-entry peak / (D(tau_a/T_1) m_B a_m) - 1, in %:")
    for (g, l1), v in coll.items():
        dev = 100 * (v / D - 1)
        sel = [np.argmin(abs(R - r)) for r in (0.5, 0.8, 1.0, 2.0, 3.0)]
        print(f"  gamma = {g:.2f}, l1 = {l1:.2f}: " + "  ".join(f"r={R[i]:.1f}: {dev[i]:+.1f}" for i in sel))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    for g in GAMMAS:
        ax[0].plot(L1S, ratio[g], label=fr"$\gamma$ = {g}")
        ax[1].plot(L1S, part[g], label=fr"$\gamma$ = {g}")
    nc = nordell_ciozda()[0]
    ax[0].plot(NC["sd"], nc[3], "k*", ms=10, label="Nordell & Ciozda (1984)")
    _, _, me, _, mB = strand_B(NC["sd"], NC["st"], NC["gamma"])
    ax[1].plot(NC["sd"], me[0] / (8 / np.pi ** 2 * mB), "k*", ms=10)
    ax[0].axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax[1].axhline(np.pi ** 2 / 8, color="0.5", lw=0.8, ls=":")
    ax[1].text(0.02, np.pi ** 2 / 8 + 0.005, r"$m_{eff} = m_B$", fontsize=8, color="0.4")
    ax[0].set(xlabel=r"drive offset from the head, $\ell_1 = \sigma_d$ (units of $L$)",
              ylabel=r"$T_1 / (4 t_B)$", title="(a) fundamental period over four transits")
    ax[1].set(xlabel=r"drive offset from the head, $\ell_1$",
              ylabel=r"$m_{eff,1} / ((8/\pi^2) m_B)$", title="(b) participation of the fundamental")
    ax[0].legend(fontsize=8)
    ax[2].plot(R, D, "k-", lw=2, label="uniform strand (universal curve)")
    for (g, l1), v in coll.items():
        lab = (fr"$\gamma$ = {g:.2f}, $\ell_1$ = {l1:.2f}" + (" (N&C)" if l1 == NC["sd"] else ""))
        ax[2].plot(R, v, "--", label=lab)
    ax[2].set(xlabel=r"$\tau_a / T_1$", ylabel=r"$T_{entry,max} / (m_B a_m)$",
              title="(c) drive-entry peak, intermediate drive")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.savefig(OUT / "intermediate_drive.pdf")
    fig.savefig(OUT / "intermediate_drive.png", dpi=150)


if __name__ == "__main__":
    main()
