"""Validation figure: modal solution vs independent lumped-mass model (phase 3.3).

Two illustrative conveyors (invented parameters, same belt):
  A  head drive, take-up 60 m downstream of the drive, inclined carry strand
     (lift 40 m with a break of grade), sine start;
  B  drive on the return strand 1200 m from the head, take-up 300 m downstream of it
     (xi = 0.1), horizontal, parabolic start.
(The take-up-upstream configuration, which puts the take-up on the tight side, is covered
by tests/test_lumped.py; it drives the slack side negative and is not a validation case.)
Both: L = 3000 m, EA = 120 MN, mu_r = 40, mu_c = 100 kg/m, directly hung counterweight
with T_t = 70 kN, r_r = 15, r_c = 45 N/m, t_v = 0.3 s, V = 5 m/s, t_a = 60 s,
resistance onset phi = V/V_inf.

Run:  python validation/fig_lumped_validation.py   ->  validation/figures/lumped_validation.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beltloop import Conveyor, GravityTakeUp, natural_frequencies
from beltloop.lumped import LumpedModel

G = 9.80665
V, TA, TEND = 5.0, 60.0, 150.0
OUT = Path(__file__).parent / "figures"


def conveyor(sd, st, profile):
    return Conveyor(L=3000.0, EA=1.2e8, mu_r=40.0, mu_c=100.0, m_r=30.0, m_c=80.0,
                    drive_position=sd, takeup_position=st, takeup=GravityTakeUp(M_w=2 * 70e3 / G),
                    r_r=15.0, r_c=45.0, t_v=0.3, carry_profile=profile)


CASES = {
    "A": (conveyor(0.0, 60.0, ((0, 0), (1500, 10), (3000, 40))), "sine"),
    "B": (conveyor(1200.0, 1500.0, ((0, 0), (3000, 0))), "parabolic"),
}


def run_case(cv, kind, N=1000):
    k = N // 125
    ref = cv.start(V, TA, kind, "velocity", n_modes=120, t_end=TEND, n_t=1501)
    res = LumpedModel(cv, N).start(V, TA, kind, "velocity", dt=0.1 / k, t_end=TEND, store_every=k)
    return ref, res


def convergence(cv, kind, Ns=(125, 250, 500, 1000)):
    ref = cv.start(V, TA, kind, "velocity", n_modes=120, t_end=TEND, n_t=1501)
    ex = natural_frequencies(cv.loop(), cv.beta, 5) * cv.c_r / cv.L
    ef, et = [], []
    for N in Ns:
        lm = LumpedModel(cv, N)
        ef.append(np.max(np.abs(lm.eigenfrequencies(5) / ex - 1)))
        k = N // 125
        res = lm.start(V, TA, kind, "velocity", dt=0.1 / k, t_end=TEND, store_every=k)
        Tm = ref.dynamic_tension(res.x_mid)
        et.append(np.max(np.abs(res.T_dynamic - Tm)) / np.max(np.abs(Tm)))
    return np.array(Ns), np.array(ef), np.array(et)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "lines.linewidth": 1.0})
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))
    colors = {"entry": "C0", "exit": "C3"}
    summary = []
    for panel, (name, (cv, kind)) in zip(ax[0], CASES.items()):
        ref, res = run_case(cv, kind)
        x = res.x_mid[[-1, 0]]
        Tm = (cv.static_tension(x)[:, None] + ref.dynamic_tension(x)) / 1e3
        Tl = res.T_total[[-1, 0]] / 1e3
        sub = slice(0, None, 30)
        for i, lab in enumerate(("entry", "exit")):
            panel.plot(ref.t, Tm[i], color=colors[lab], label=f"drive {lab}, modal")
            panel.plot(res.t[sub], Tl[i][sub], "o", ms=2.5, mfc="none", color=colors[lab],
                       label=f"drive {lab}, lumped")
        panel.axvline(TA, color="0.6", lw=0.6, ls="--")
        panel.set_xlabel("$t$ (s)"); panel.set_ylabel("belt tension (kN)")
        panel.set_title(f"({'a' if name == 'A' else 'b'}) conveyor {name}: "
                        f"$\\xi$ = {cv.xi:.2f}, $\\beta$ = {cv.beta:.3f}, $\\gamma$ = {cv.gamma:.2f}",
                        fontsize=8, loc="left")
        Td = ref.dynamic_tension(res.x_mid)
        summary.append((name, np.max(np.abs(res.T_dynamic - Td)) / np.max(np.abs(Td)),
                        np.max(np.abs(res.y - ref.takeup_displacement())) / np.max(np.abs(res.y))))
        ax[1, 0].plot(ref.t, ref.takeup_displacement(), color="C0" if name == "A" else "C2",
                      label=f"{name}, modal")
        ax[1, 0].plot(res.t[sub], res.y[sub], "o", ms=2.5, mfc="none",
                      color="C0" if name == "A" else "C2", label=f"{name}, lumped")
    ax[0, 0].legend(loc="center right", frameon=False)
    ax[1, 0].set_xlabel("$t$ (s)"); ax[1, 0].set_ylabel("take-up travel $y$ (m)")
    ax[1, 0].set_title("(c) take-up motion", fontsize=8, loc="left")
    ax[1, 0].legend(frameon=False, ncol=2)

    a = ax[1, 1]
    for name, (cv, kind) in CASES.items():
        Ns, ef, et = convergence(cv, kind)
        mk = "s" if name == "A" else "^"
        a.loglog(Ns, ef, mk + "-", ms=3.5, color="C1", label=f"{name}: first five frequencies")
        a.loglog(Ns, et, mk + "--", ms=3.5, color="C4", label=f"{name}: tension field, all $t$")
        summary.append((name + " conv", ef, et))
    Ns = np.array([125, 1000])
    a.loglog(Ns, 3e-3 * (Ns / 125.0) ** -2, ":", color="0.4", label="slope $-2$")
    a.set_xticks([125, 250, 500, 1000]); a.set_xticklabels(["125", "250", "500", "1000"])
    a.minorticks_off()
    a.set_xlabel("number of elements $N$"); a.set_ylabel("max. relative difference")
    a.set_title("(d) lumped model vs closed form", fontsize=8, loc="left")
    a.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "lumped_validation.pdf"); fig.savefig(OUT / "lumped_validation.png", dpi=200)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
