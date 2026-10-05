"""Phase 4.6: validity zones of the prescribed-velocity linear model.

Three conditions, all in units of the belt inertia force (horizontal conveyor, resistances
proportional to the inertial line density, r = rho mu a_m, onset phi = V/V_inf):

  * slack in strand A during the start (drive exit): T_2 >= m_A [C_max(r_A, rho) - rho];
  * slack in strand B after the start (rebound): T_2 >= -m_B C_min(r, rho) - rho m_A;
  * grip of the drive (no slip, E = exp(mu theta)):
      T_2 >= [m_B C_max(r, rho) + E m_A C_max(r_A, rho)] / (E - 1) - rho m_A,
    which always covers the exit condition.
The take-up follows the belt (belt taut at the take-up, counterweight rope taut) as long as
its acceleration stays below n T_t / M (= g for a direct counterweight); the belt-side
acceleration never exceeds about 2 a_m, so this condition is inactive for speed-controlled
starts.

Panels: (a) exit requirement per unit mass of strand A; (b) rebound per unit mass of strand B;
(c) take-up velocity and acceleration envelopes over (xi, gamma); (d) minimum running
slack-side tension along the take-up position, drive at the head; (e) required take-up
tension with an intermediate drive (the tight-side region; full loop, not the strand
estimate, which misses up to ~5 % there); (f) the requirement as a lower bound on beta for a
direct counterweight, beta_min = (4/n)(a_m/g) T_t_min.

Run:  python maps/validity.py   (prints tables, writes maps/figures/validity.pdf)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beltloop import (Loop, StartProfile, modal_basis, natural_frequencies, startup_response,
                      strand_extremes, strand_participation, strand_rebound, takeup_kinematics,
                      tension_requirement)
from beltloop.metrics import _time_grid

HERE = Path(__file__).parent
OUT = HERE / "figures"
R = np.logspace(np.log10(0.2), np.log10(20), 45)
RHOS = (0.0, 1.0, 2.0, 5.0)
EULERS = (3.0, 16.0)        # 180 deg wrap with mu = 0.35; tandem drive with ~460 deg wrap


def curves():
    ext = {rho: strand_extremes(R, rho) for rho in RHOS}
    Dm1, _ = strand_rebound(R, zeta1=0.01)
    return ext, Dm1


def kinematics(rs=(0.01, 0.03, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0, 5.0)):
    env_v, env_a = [], []
    for r in rs:
        v = a = 0.0
        for g in (1.0, 1.5, 2.0, 3.0):
            for xi in (0.1, 0.3, 0.5, 0.7, 0.9):
                lp = Loop.from_positions(0.0, xi, g)
                T1 = 2 * np.pi / natural_frequencies(lp, 1e-3, 1)[0]
                nm = int(min(600, max(120, 15 / r)))
                k = takeup_kinematics(lp, StartProfile("sine", r * T1, "none"), n_modes=nm,
                                      points=150 if r < 0.2 else 400, periods_after=2.0)
                v = max(v, abs(k.v_max), abs(k.v_min))
                a = max(a, abs(k.a_max), abs(k.a_min))
        env_v.append(v)
        env_a.append(a)
    return np.array(rs), np.array(env_v), np.array(env_a)


def requirement(lp, tau_a, rho, ext_cache):
    """(exit, rebound, grip for each E) for one loop, interpolating the strand extremes."""
    TA = 2 * np.pi / strand_participation(lp.upstream, 1)[0][0]
    TB = 2 * np.pi / strand_participation(lp.downstream[::-1], 1)[0][0]
    mA = sum(s.mu * s.length for s in lp.upstream)
    mB = lp.belt_mass - mA
    cmax, cmin = ext_cache
    f = lambda c, r: np.interp(np.log(np.clip(r, R[0], R[-1])), np.log(R), c)
    cA, cBmax, cBmin = f(cmax, tau_a / TA), f(cmax, tau_a / TB), f(cmin, tau_a / TB)
    exit_ = mA * (cA - rho)
    reb = -mB * cBmin - rho * mA
    grip = [(mB * cBmax + E * mA * cA) / (E - 1) - rho * mA for E in EULERS]
    return exit_, reb, grip, mA


def exact_requirement(lp, tau_a, rho, euler, n_modes=120):
    """Full-loop requirement (beta -> 0): T_2 for positive tension and for grip."""
    b = modal_basis(lp, 1e-6, n_modes)
    T1 = 2 * np.pi / b.Om[0]
    tau = _time_grid(tau_a + 3 * T1, min(T1, tau_a) / 200)
    T = startup_response(b, StartProfile("sine", tau_a, "velocity"), 0.0, tau).tension(
        np.linspace(0, 2, 801))
    mA = sum(s.mu * s.length for s in lp.upstream)
    slack = -T.min() - rho * mA
    grip = ((T[-1] - euler * T[0]) / (euler - 1)).max() - rho * mA
    return max(slack, grip, 0.0), mA


def main():
    OUT.mkdir(exist_ok=True)
    ext, Dm1 = curves()
    print("Strand curves (sine, undamped): exit requirement C_max - rho, rebound -C_min")
    print("   r   " + "  ".join(f"rho={rho:<4}" for rho in RHOS))
    for j in range(0, len(R), 6):
        print(f"{R[j]:5.2f} " + "  ".join(f"{ext[rho][0][j] - rho:5.3f}/{-ext[rho][1][j]:5.3f}" for rho in RHOS))
    rs, ev, ea = kinematics()
    print("\nTake-up envelopes over xi in [0.05, 0.95], gamma in [1, 3], beta -> 0:")
    for r, v, a in zip(rs, ev, ea):
        print(f"  tau_a/T1 = {r:4.2f}: |y'|/V_inf <= {v:5.3f}, |y''|/a_m <= {a:5.3f}")

    # (d) drive at the head, gamma = 2, rho = 1
    gam, rho = 2.0, 1.0
    xis = np.linspace(0.02, 0.98, 33)
    lp_ref = Loop.from_positions(0.0, 0.5, gam)
    panel_d = {}
    for r in (1.0, 3.0):
        rows = []
        for xi in xis:
            lp = Loop.from_positions(0.0, xi, gam)
            T1 = 2 * np.pi / natural_frequencies(lp, 1e-3, 1)[0]
            e, b, gr, _ = requirement(lp, r * T1, rho, ext[rho])
            rows.append((e, b, *gr))
        panel_d[r] = np.array(rows) / (1 + gam ** 2)
    print("\nMinimum T_2 / (m_belt a_m), drive at head, gamma = 2, rho = 1 (exit, rebound, grip E=3, E=16):")
    for r, rows in panel_d.items():
        for xi, row in list(zip(xis, rows))[::8]:
            print(f"  tau_a/T1 = {r}: xi = {xi:4.2f}: " + " ".join(f"{v:6.3f}" for v in row))

    # (e) intermediate drive at sigma_d = 0.5, take-up anywhere on the return strand
    sig_d, tau_a = 0.5, 20.0
    sts = np.r_[np.linspace(0.04, 0.46, 8), np.linspace(0.54, 0.96, 8)]
    rows_e = []
    for st in sts:                                          # exact, full loop (grip with E = 16)
        lp = Loop.from_positions(sig_d, st, gam, rho, rho * gam ** 2)
        T2, mA = exact_requirement(lp, tau_a, rho, EULERS[1])
        rows_e.append((T2 + rho * mA, rho * mA))            # take-up tension and its resistance part
    rows_e = np.array(rows_e) / (1 + gam ** 2)
    print("\nIntermediate drive (sigma_d = 0.5), gamma = 2, rho = 1, tau_a = 20 L/c_r, E = 16, "
          "full loop: T_t / (m_belt a_m)")
    for st, row in zip(sts, rows_e):
        print(f"  sigma_t = {st:4.2f}: T_t = {row[0]:6.3f} (resistance part {row[1]:5.3f})")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = ax.ravel()
    for rho in RHOS:
        l, = ax[0].semilogx(R, ext[rho][0] - rho, label=rf"$\rho$ = {rho:g}")
        ax[0].axhline(np.sqrt(1 + rho ** 2 / 4) - rho / 2, color=l.get_color(), ls=":", lw=0.8)
        ax[1].semilogx(R, -ext[rho][1], color=l.get_color(), label=rf"$\rho$ = {rho:g}")
    ax[1].semilogx(R, Dm1, "k--", lw=0.8, label=r"$\rho$ = 0, $\zeta_1$ = 0.01")
    ax[1].semilogx(R, 0.8 / R, "k:", lw=0.8, label=r"$0.8\,T_1/\tau_a$")
    ax[0].set(xlabel=r"$\tau_a / T_{A1}$", ylabel=r"required $T_2 / (m_A a_m)$",
              title="(a) slack at the drive exit, strand A (dotted: slow start)")
    ax[1].set(xlabel=r"$\tau_a / T_1$", ylabel=r"rebound below running / $(m_B a_m)$",
              title="(b) rebound in strand B after the start", ylim=(0, 1.6))
    ax[2].semilogx(rs, ev, "o-", label=r"$|\dot y| / V_\infty$")
    ax[2].semilogx(rs, ea, "s-", label=r"$|\ddot y| / a_m$")
    ax[2].axhline(1, color="C0", ls=":")
    ax[2].axhline(2, color="C1", ls=":")
    ax[2].set(xlabel=r"$\tau_a / T_1$", title="(c) take-up kinematics, envelope (belt side)")
    for r, ls in zip(panel_d, ("-", "--")):
        rows = panel_d[r]
        for k, (lab, c) in enumerate(zip(("exit", "rebound", "grip $E$ = 3", "grip $E$ = 16"), ("C0", "C1", "C2", "C3"))):
            ax[3].plot(xis, rows[:, k], color=c, ls=ls, label=rf"{lab}, $\tau_a = {r:g}\,T_1$")
    ax[3].axhline(0, color="k", lw=0.5)
    ax[3].set(xlabel=r"take-up position $\xi$ (head drive)", ylabel=r"minimum $T_2 / (m_{belt} a_m)$",
              title=r"(d) minimum $T_2$, head drive, $\gamma$ = 2, $\rho$ = 1")
    ax[4].plot(sts[sts < sig_d], rows_e[sts < sig_d, 0], "C3")
    ax[4].plot(sts[sts > sig_d], rows_e[sts > sig_d, 0], "C0")
    ax[4].plot(sts, rows_e[:, 1], "k:", label="resistance between drive exit and take-up")
    ax[4].axvspan(0, sig_d, color="C3", alpha=0.08, label="take-up on the tight side")
    ax[4].axvline(sig_d, color="k", lw=0.8)
    ax[4].set(xlabel=r"take-up position from the head $\sigma_t$", ylabel=r"required $T_t / (m_{belt} a_m)$",
              title=r"(e) drive at $\sigma_d$ = 0.5 (full loop, $E$ = 16)")
    for a in ax[:5]:
        a.legend(fontsize=7)
        a.grid(alpha=0.3, which="both")
    # (f) lower bound on beta of a direct counterweight (n = 2), f = 0.02, a_m = 0.01 g
    f_res, a_g = 0.02, 0.01
    rho_f = f_res / a_g
    xis_f = np.linspace(0.02, 0.98, 17)
    print(f"\nbeta_min (direct counterweight, n = 2), gamma = 2, f = {f_res}, a_m/g = {a_g}, tau_a = 3 T_1:")
    for E, c in zip(EULERS, ("C2", "C3")):
        bm = []
        for xi in xis_f:
            lp = Loop.from_positions(0.0, xi, gam)
            T1 = 2 * np.pi / natural_frequencies(lp, 1e-3, 1)[0]
            bm.append(tension_requirement(lp, 3 * T1, rho_f, euler=E).beta_min(a_g))
        run = 2 * f_res * ((1 + gam ** 2) / (E - 1) + xis_f)
        ax[5].plot(xis_f, bm, color=c, label=rf"$E$ = {E:g} (start, $\tau_a = 3\,T_1$)")
        ax[5].plot(xis_f, run, color=c, ls=":", label=rf"$E$ = {E:g}, running grip only")
        print(f"  E = {E:g}: " + " ".join(f"{x:.2f}:{v:.3f}" for x, v in list(zip(xis_f, bm))[::4]))
    ax[5].set(xlabel=r"take-up position $\xi$ (head drive)", ylabel=r"$\beta_{\min}$",
              title=r"(f) $\beta_{\min}$, direct counterweight, $\gamma$ = 2", ylim=(0, None))
    ax[5].legend(fontsize=7)
    ax[5].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "validity.pdf")
    fig.savefig(OUT / "validity.png", dpi=110)


if __name__ == "__main__":
    main()
