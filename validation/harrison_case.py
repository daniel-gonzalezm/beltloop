"""Harrison (1983) case in detail (phase 3.4): sensitivity of the slow take-up period to
the take-up position xi, the line density rho (beta), the wave-speed ratio gamma and the
drive model (mass md, slip dashpot cd), and the features the linear model does not capture.

Data (Harrison 1983, Beltcon 2, section 6 and Fig. 5; empty belt):
  L = 5.1 km, M = 20 t, wave period t1 = 7 s read as the loop transit 2L/c (c = 1450 m/s;
  Eq. 2 gives 1464 m/s with 9 kg/m of steel cord and 79 kg/m of belt + rotating parts),
  slow take-up period 25 s. The 79 kg/m figure follows from Eq. 2; the paper's
  m = 398 100 kg corresponds to 39 kg/m of loop. Both densities are used.
  Take-up a few tens of metres downstream of the head drive (Fig. 5a inset): xi = 0.005
  (25 m) as base value, 0.002-0.02 as range.
Values read from Fig. 5 (scale bar 10 s = 30 px, +-1 px): wave period 7.3 +- 0.4 s (Fig. 5c,
mean of four peak and four trough spacings); slow period 24.9 s (Fig. 5e, measured take-up
displacement after removing the 7 s ripple with a one-period moving average: four peak
spacings 23.0-26.7 s). The ratio of the two does not depend on the time scale.

Run:  python validation/harrison_case.py  ->  validation/figures/harrison_case.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from beltloop import (Loop, StartProfile, damped_drive_root, modal_basis, natural_frequencies,
                      natural_frequencies_torque, startup_response, takeup_transmission)

OUT = Path(__file__).parent / "figures"

L, M, C = 5100.0, 20000.0, 1450.0
T_TR = 2 * L / C                       # loop transit time (7.03 s)
RHOS = (39.0, 79.0)
XI0 = 0.005
MEAS_SLOW, MEAS_SLOW_BAND = 24.9, (23.0, 26.7)
MEAS_WAVE, MEAS_WAVE_BAND = 7.3, (6.9, 7.7)
HARRISON_SLOW, HARRISON_WAVE = 25.0, 7.0


def setup(rho, gamma=1.0, xi=XI0, basis="transit"):
    """Loop, beta and c_r for a mean loop density rho (kg/m) split so that
    mu_c/mu_r = gamma^2. basis='transit': loop transit time fixed at 2L/1450 m/s
    (the 7 s period); basis='return': c_r = 1450 m/s (two sensors on the return strand)."""
    mur = 2 * rho / (1 + gamma ** 2)
    cr = L * (1 + gamma) / T_TR if basis == "transit" else C
    return Loop.from_positions(0.0, xi, gamma), M / (mur * L), cr, mur


def periods(rho, gamma=1.0, xi=XI0, basis="transit", n=2):
    lp, beta, cr, _ = setup(rho, gamma, xi, basis)
    return 2 * np.pi * L / (cr * natural_frequencies(lp, beta, n))


def periods_torque(rho, md, gamma=1.0, xi=XI0, n=2):
    """Torque-controlled drive (cd = 0); md in units of mu_r L."""
    lp, beta, cr, _ = setup(rho, gamma, xi)
    Om = natural_frequencies_torque(lp, beta, md, 6.0, 20000)
    return 2 * np.pi * L / (cr * Om[:n])


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Loop transit 2L/c = {T_TR:.2f} s; measured ratio slow/wave = "
          f"{MEAS_SLOW / MEAS_WAVE:.2f} (Fig. 5 read-out), {HARRISON_SLOW / HARRISON_WAVE:.2f} (stated)")

    # ---------------------------------------------------------------- xi and rho
    print("\nPrescribed drive, gamma = 1: slow / second period [s]")
    for rho in RHOS:
        for xi in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            T = periods(rho, xi=xi)
            print(f"  rho={rho:4.0f} xi={xi:5.3f}: {T[0]:6.2f} / {T[1]:5.2f}  ratio {T[0] / T_TR:.3f}")
    print("  wave-speed uncertainty: c = 1464 m/s (Eq. 2) shortens all periods by 1.0 %")

    # ---------------------------------------------------------------- gamma
    print("\nPrescribed drive, xi = 0.005: slow / second period [s] versus gamma")
    for basis in ("transit", "return"):
        for g in (1.0, 1.1, 1.2, 1.3, 1.5):
            T = [periods(rho, g, basis=basis) for rho in RHOS]
            mur = 2 * 79 / (1 + g * g)
            print(f"  {basis:7s} gamma={g:.1f} (mu_r, mu_c = {mur:5.1f}, {g * g * mur:5.1f} kg/m at 79): "
                  + "  ".join(f"rho={r:.0f}: {t[0]:5.2f}/{t[1]:4.2f}" for r, t in zip(RHOS, T)))
    gneed = brentq(lambda g: periods(79.0, g)[0] - MEAS_SLOW, 1.0, 3.0)
    print(f"  gamma for {MEAS_SLOW} s (transit basis, rho = 79): {gneed:.2f}")

    # ---------------------------------------------------------------- drive mass
    print("\nTorque-controlled drive (cd = 0), xi = 0.005, gamma = 1: slow / second period [s]")
    for md in (1e-4, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        T = [periods_torque(rho, md) for rho in RHOS]
        print(f"  md={md:7.4f}: " + "  ".join(f"rho={r:.0f}: {t[0]:5.2f}/{t[1]:4.2f}" for r, t in zip(RHOS, T)))
    for rho in RHOS:
        lm = brentq(lambda lm: periods_torque(rho, 10 ** lm)[0] - MEAS_SLOW, -1, 2)
        md = 10 ** lm
        Md = md * rho * L
        J = Md * (3.7 / (1490 * np.pi / 30)) ** 2      # reduced to a 1490 rpm motor shaft
        print(f"  rho={rho:.0f}: md = {md:.2f} for {MEAS_SLOW} s -> M_d = {Md / 1e3:.0f} t, "
              f"J = {J:.0f} kg m^2 at 1490 rpm (assumed 4-pole, 50 Hz); second period "
              f"{periods_torque(rho, md)[1]:.2f} s")

    # ---------------------------------------------------------------- drive slip dashpot
    print("\nDrive with slip dashpot cd = c_d/(mu_r c_r), md = 0.3, xi = 0.005, gamma = 1, rho = 79")
    Z = 79.0 * C
    for P, s in ((0.6e6, 0.03), (1.2e6, 0.01)):     # assumed rated power and slip (not in the paper)
        print(f"  induction motor on its running characteristic, P = {P / 1e6:.1f} MW, slip {s:.0%}: "
              f"c_d = P/(s V^2) = {P / (s * 3.7 ** 2) / 1e6:.1f} MN s/m, cd = {P / (s * 3.7 ** 2) / Z:.0f}")
    lp, beta, cr, _ = setup(79.0)
    Om0 = natural_frequencies(lp, beta, 1)[0]
    guess = Om0 + 1e-4j
    for cd in (1e3, 100, 30, 10, 5, 3, 2):
        Om = damped_drive_root(lp, beta, 0.3, cd, guess)
        guess = Om
        print(f"  cd={cd:6.0f}: period {2 * np.pi * L / (cr * Om.real):5.2f} s, "
              f"damping ratio {Om.imag / abs(Om):.3f}")

    # ---------------------------------------------------------------- belt damping
    zeta_w = np.log(1 / 0.8) / (2 * np.pi)            # ~0.8 amplitude ratio per 7 s cycle, Fig. 5c
    tv = 2 * zeta_w / (2 * np.pi / MEAS_WAVE)
    zs = tv * (2 * np.pi / periods(79.0)[0]) / 2
    print(f"\nKelvin-Voigt bound: attributing the 7 s decay (zeta ~ {zeta_w:.3f}) to the belt gives "
          f"t_v ~ {tv:.2f} s, zeta_slow ~ {zs:.4f}, period change {1 / np.sqrt(1 - zs ** 2) - 1:.1e}")

    # ---------------------------------------------------------------- take-up transparency
    print("\nWave transmission through the take-up, |t| = beta Om/sqrt(beta^2 Om^2 + 4)")
    for rho in RHOS:
        _, beta, cr, _ = setup(rho)
        for P in (25.0, 7.0, 1.0):
            Om = 2 * np.pi * L / (cr * P)
            print(f"  rho={rho:.0f} period {P:4.1f} s: |t| = {abs(takeup_transmission(beta, Om)):.3f}")

    # ---------------------------------------------------------------- hard start, time domain
    print("\nHard first step (sine acceleration, peak 2.2 m/s^2, to 2.6 m/s), undamped, rho = 79")
    lp, beta, cr, mur = setup(79.0)
    b = modal_basis(lp, beta, 300)
    V, am = 2.6, 2.2
    ts = L / cr
    prof = StartProfile("sine", np.pi * V / (2 * am) / ts, "none")
    t = np.linspace(0, 30, 601)
    r = startup_response(b, prof, 0.0, t / ts)
    y = r.takeup_displacement() * am * L ** 2 / cr ** 2
    W = b.W(np.array([XI0 + 1e-7]))[:, 0]
    vS1 = (W @ r.dp) * am * L / cr + V * prof.velocity(t / ts)      # absolute belt speed
    i7 = np.argmin(abs(t - 7.0))
    print(f"  take-up travel at 7 s: {y[i7]:.1f} m (record: about 2 m); belt speed just downstream "
          f"of the take-up stays {abs(vS1[:i7 - 5]).max():.3f} m/s until the wave from the drive entry "
          f"arrives ({(2 - XI0) * ts:.1f} s) (record: immediate response, 2.22 m/s^2 peak)")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.6))

    def band(a, lo, hi, val, txt):
        a.axhspan(lo, hi, color="0.88", zorder=0)
        a.axhline(val, color="0.5", lw=0.8, ls=":")
        a.text(0.02, val, txt, transform=a.get_yaxis_transform(), va="bottom", fontsize=7, color="0.35")

    xis = np.geomspace(0.002, 0.3, 40)
    a = ax[0, 0]
    band(a, *MEAS_SLOW_BAND, MEAS_SLOW, "measured (Fig. 5e)")
    for rho, ls in zip(RHOS, ("-", "--")):
        a.plot(xis, [periods(rho, xi=x)[0] for x in xis], "k" + ls, label=rf"$\rho$ = {rho:.0f} kg/m")
    a.axvspan(0.002, 0.02, color="tab:blue", alpha=0.08)
    a.set(xscale="log", xlabel=r"take-up position $\xi$", ylabel="slow period (s)", ylim=(20, 30))
    a.legend(fontsize=8, loc="lower left")
    a.set_title(r"(a) position and density, $\gamma$ = 1", fontsize=9, loc="left")

    gs = np.linspace(1.0, 2.0, 31)
    a = ax[0, 1]
    band(a, *MEAS_SLOW_BAND, MEAS_SLOW, "measured")
    for basis, ls in (("transit", "-"), ("return", "--")):
        a.plot(gs, [periods(79.0, g, basis=basis)[0] for g in gs], "k" + ls,
               label="loop transit fixed (7.03 s)" if basis == "transit" else r"$c_r$ fixed (1450 m/s)")
    a.set(xlabel=r"wave-speed ratio $\gamma = c_r/c_c$", ylabel="slow period (s)", ylim=(20, 36))
    a.legend(fontsize=8, loc="upper left")
    a.set_title(r"(b) two wave speeds, $\xi$ = 0.005, $\rho$ = 79 kg/m", fontsize=9, loc="left")

    mds = np.geomspace(0.01, 100, 40)
    a = ax[1, 0]
    band(a, *MEAS_SLOW_BAND, MEAS_SLOW, "slow, measured")
    band(a, *MEAS_WAVE_BAND, MEAS_WAVE, "wave, measured")
    Tt = np.array([periods_torque(79.0, m) for m in mds])
    a.plot(mds, Tt[:, 0], "k-", label="first")
    a.plot(mds, Tt[:, 1], "k--", label="second")
    Tp = periods(79.0)
    a.plot([mds[-1]] * 2, Tp, "ko", ms=4, label="prescribed velocity")
    a.set(xscale="log", xlabel=r"reduced drive mass $m_d = M_d/(\mu_r L)$", ylabel="period (s)", ylim=(0, 32))
    a.legend(fontsize=8, loc="center right")
    a.set_title("(c) drive without speed control", fontsize=9, loc="left")

    Ps = np.geomspace(0.3, 60, 200)
    a = ax[1, 1]
    for rho, ls in zip(RHOS, ("-", "--")):
        _, beta, cr, _ = setup(rho)
        a.plot(Ps, abs(takeup_transmission(beta, 2 * np.pi * L / (cr * Ps))), "k" + ls,
               label=rf"$\rho$ = {rho:.0f} kg/m ($\beta$ = {beta:.3f})")
    for P in (MEAS_WAVE, MEAS_SLOW):
        a.axvline(P, color="0.5", lw=0.8, ls=":")
    a.set(xscale="log", xlabel="wave period (s)", ylabel="transmission |t|", ylim=(0, 1))
    a.legend(fontsize=8)
    a.set_title("(d) waves crossing the take-up", fontsize=9, loc="left")

    for a in ax.flat:
        a.tick_params(labelsize=8)
        a.xaxis.label.set_size(9)
        a.yaxis.label.set_size(9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"harrison_case.{ext}", dpi=200)
    print(f"\nFigure written to {OUT}/harrison_case.(pdf|png)")


if __name__ == "__main__":
    main()
