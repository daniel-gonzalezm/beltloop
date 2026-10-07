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


SUMMARY = Path(__file__).parent / "data" / "lumped_validation.json"


def main():
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "maps"))
    import style                                   # common style of the paper figures
    style.apply()
    C = style.OKABE_ITO
    fig, ax = plt.subplots(2, 2, figsize=(style.DOUBLE, 112 * style.MM), constrained_layout=True)
    # entry and exit differ by colour and marker; conveyors A and B in (c) by colour, marker
    # and line style (readable without colour, step 6.2)
    sty = {"entry": (C[0], "o"), "exit": (C[1], "s")}
    summary = {}
    for panel, (name, (cv, kind)), lab_p in zip(ax[0], CASES.items(), ("(a)", "(b)")):
        ref, res = run_case(cv, kind)
        x = res.x_mid[[-1, 0]]
        Tm = (cv.static_tension(x)[:, None] + ref.dynamic_tension(x)) / 1e3
        Tl = res.T_total[[-1, 0]] / 1e3
        sub = slice(0, None, 30)
        for i, lab in enumerate(("entry", "exit")):
            c, mk = sty[lab]
            panel.plot(ref.t, Tm[i], color=c, label=f"drive {lab}, modal")
            panel.plot(res.t[sub], Tl[i][sub], mk, ms=2.6, mfc="none", mew=0.6, color=c,
                       label=f"drive {lab}, lumped")
        panel.axvline(TA, color="0.6", lw=0.6, ls="--")
        panel.set(xlabel=r"$t$ (s)", ylabel="belt tension (kN)")
        style.panel_label(panel, f"{lab_p} conveyor {name}: $\\xi$ = {cv.xi:.2f}, "
                                 f"$\\beta$ = {cv.beta:.3f}, $\\gamma$ = {cv.gamma:.2f}")
        Td = ref.dynamic_tension(res.x_mid)
        summary[name] = dict(tension_err=float(np.max(np.abs(res.T_dynamic - Td)) / np.max(np.abs(Td))),
                             travel_err=float(np.max(np.abs(res.y - ref.takeup_displacement()))
                                              / np.max(np.abs(res.y))),
                             xi=cv.xi, beta=cv.beta, gamma=cv.gamma)
        c, mk, ls = (C[0], "o", "-") if name == "A" else (C[2], "s", "--")
        ax[1, 0].plot(ref.t, ref.takeup_displacement(), color=c, ls=ls, label=f"{name}, modal")
        ax[1, 0].plot(res.t[sub], res.y[sub], mk, ms=2.6, mfc="none", mew=0.6, color=c,
                      label=f"{name}, lumped")
    ax[0, 0].legend(loc="center right")
    ax[1, 0].set(xlabel=r"$t$ (s)", ylabel=r"take-up travel $y$ (m)")
    style.panel_label(ax[1, 0], "(c) take-up motion")
    ax[1, 0].legend(ncol=2, loc="lower right")

    a = ax[1, 1]
    for name, (cv, kind) in CASES.items():
        Ns, ef, et = convergence(cv, kind)
        mk = "s" if name == "A" else "^"
        a.loglog(Ns, ef, mk + "-", ms=3.2, mfc="w", color=C[4], label=f"{name}: first five frequencies")
        a.loglog(Ns, et, mk + "--", ms=3.2, mfc="w", color=C[3], label=f"{name}: tension field, all $t$")
        summary[name].update(N=[int(n) for n in Ns], freq_err=[float(e) for e in ef],
                             field_err=[float(e) for e in et],
                             freq_order=float(-np.polyfit(np.log(Ns), np.log(ef), 1)[0]),
                             field_order=float(-np.polyfit(np.log(Ns[:3]), np.log(et[:3]), 1)[0]))
    Ns = np.array([125, 1000])
    a.loglog(Ns, 3e-3 * (Ns / 125.0) ** -2, ":", color="0.4", label="slope $-2$")
    a.set_xticks([125, 250, 500, 1000]); a.set_xticklabels(["125", "250", "500", "1000"])
    a.minorticks_off()
    a.set(xlabel=r"number of elements $N$", ylabel="max. relative difference")
    style.panel_label(a, "(d) lumped model against closed form")
    a.legend(fontsize=6, loc="lower left")
    for b in ax.flat:
        b.grid(alpha=0.25, lw=0.4, which="both")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "lumped_validation.pdf"); fig.savefig(OUT / "lumped_validation.png", dpi=200)
    summary["note"] = ("written by validation/fig_lumped_validation.py; field_order from N = "
                       "125-500 (above N = 1000 a round-off floor of about 1e-6 appears)")
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
