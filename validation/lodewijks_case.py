"""Lodewijks (1996), Ch. 8: velocity-controlled starts of a loaded 1 km conveyor (phase 3.5).

Source: G. Lodewijks, Dynamics of Belt Systems, PhD thesis, TU Delft, 1996 (scanned PDF;
printed page = PDF page - 20). Data used (printed pages):
  * belt EP 500/5, B = 1.2 m, E = 340.917 MPa, A = 0.01236 m^2, 14.28 kg/m (p. 153, 163);
  * bulk 133.54 kg/m on the carry strand (Eq. 8.2); reduced idler masses 13.38 (carry) and
    6.95 kg/m (return) (Eq. 8.5); L = 1000 m, horizontal (Table 8.1, Fig. 8.1);
  * running resistances, loaded: carry 31.32 kN, return 4.23 kN (Eq. 8.6, p. 154);
  * tensioning force 42.66 kN = 2 T_t (p. 155); head drive, take-up on the return just after
    the drive (Fig. 8.1); in the FE model the two pulleys are merged (Fig. 7.2, Eq. 7.55);
  * profiles: linear offset (Eq. 8.35), linear, linear with a rest of 5 s after 5 s (Eq. 8.36),
    Harrison (Eq. 6.1, our 'sine'), Nordell (Eq. 6.2, our 'triangular'); 30 s (Sec. 8.6.2);
  * targets: Table 8.7 (p. 186), Figs. 8.41-8.42 (p. 181, digitised, see data/),
    Figs. 8.44, 8.46, 8.47 (p. 183, read by eye at t_a = 10, 20, 30, 40, 50 s).
Derived here: the a_max column of Table 8.7 implies a final speed of ~5.11 m/s (motor slip;
linear 0.170 x 30, Harrison 0.268 x 60/pi, Nordell 0.341 x 15) and an initial offset
V_0 = 5.11 - 0.155 x 30 = 0.46 m/s. The speed jump of the offset start is a ramp of
T_J = 0.5 s (Fig. 8.38 shows an almost instantaneous jump). No longitudinal damping is given
in Ch. 8; zeta_1 = 0 and 0.05 bracket it (the fitted decay of Fig. 8.42 gives ~0.05).

Configurations compared:
  A  our model with the take-up pulley 1 m after the drive (xi = 0.001): the take-up
     anchors the slack side (physical lay-out of Fig. 8.1);
  B  sliding drive: drive pulley on the take-up carriage, ends at w(0) = -y, w(2L) = +y,
     M y'' = -(T(0) + T(2L)) (literal reading of Fig. 8.3 and Eq. 7.55). Characteristic
     equation 2 + tr P - beta Om^2 P_12 = 0 (P: whole loop); diagnostic only.

Run:  python validation/lodewijks_case.py  ->  validation/figures/lodewijks_case.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G_STD
from scipy.optimize import brentq, curve_fit

from beltloop import (Conveyor, GravityTakeUp, Loop, PiecewiseProfile, StartProfile,
                      natural_frequencies, natural_frequencies_torque)
from beltloop.transfer import chain

HERE = Path(__file__).parent
OUT = HERE / "figures"

# ------------------------------------------------------------------------------ data
L = 1000.0
EA = 340.917e6 * 0.01236
MU_R = 14.28 + 6.95
MU_C = 14.28 + 13.38 + 133.54
MU_C_EMPTY = 14.28 + 13.38
R_R, R_C = 4.23, 31.32
F_T = 42.66e3
M_W = F_T / G_STD
V_INF, V_0, T_J, T_A = 5.11, 0.46, 0.5, 30.0
XI = 0.001

TABLE_8_7 = {           # y_max [m], eps_1,max [-], a_max [m/s^2]
    "linear offset": (6.101, 0.0238, 0.155),
    "linear": (6.292, 0.0246, 0.170),
    "linear delayed": (5.148, 0.0236, 0.170),
    "Harrison (6.1)": (9.345, 0.0303, 0.268),
    "Nordell (6.2)": (10.748, 0.0382, 0.341),
}
SWEEP = {               # t_a: (eps*_max, y_max, eps_1,min), Figs. 8.44, 8.46, 8.47
    10: (0.0345, 10.3, -0.0063), 20: (0.0285, 7.7, 0.0029), 30: (0.0238, 6.1, 0.0049),
    40: (0.0193, 3.0, 0.0050), 50: (0.0170, 2.0, 0.0050),
}


def conveyor(zeta1=0.0, mu_c=MU_C, M_extra=0.0, xi=XI):
    cv = Conveyor(L=L, EA=EA, mu_r=MU_R, mu_c=mu_c, drive_position=0.0,
                  takeup_position=xi * L, takeup=GravityTakeUp(M_w=M_W, M_c=M_extra, kappa=0.0),
                  r_r=R_R, r_c=R_C)
    if zeta1:
        Om1 = natural_frequencies(cv.loop(), cv.beta, 1)[0]
        cv.t_v = 2 * zeta1 / (Om1 * cv.c_r / L)       # zeta_1 = t_v omega_1 / 2
    return cv


def profile(name, cv, t_a=T_A, onset="velocity", t_j=T_J):
    """Dimensionless profile and acceleration scale a_m (SI)."""
    s = cv.c_r / L
    if name == "linear offset":
        return PiecewiseProfile.linear(s * t_a, V_0 / V_INF, s * t_j, onset), V_INF / t_a
    if name == "linear":
        return PiecewiseProfile.linear(s * t_a, onset=onset), V_INF / t_a
    if name == "linear delayed":
        return PiecewiseProfile.delayed(s * t_a, s * 5.0, s * 5.0, onset), V_INF / t_a
    kind = "sine" if name.startswith("Harrison") else "triangular"
    pf = {"sine": np.pi / 2, "triangular": 2.0}[kind]
    return StartProfile(kind, s * t_a, onset), pf * V_INF / t_a


def run(name, zeta1=0.0, t_a=T_A, onset="velocity", t_j=T_J, mu_c=MU_C, M_extra=0.0,
        t_end=None, n_modes=120, n_t=3001):
    cv = conveyor(zeta1, mu_c, M_extra)
    prof, a_m = profile(name, cv, t_a, onset, t_j)
    t_end = t_end or t_a + 40.0
    ds = cv.start_profile(prof, a_m, t_end, n_modes=n_modes, n_t=n_t)
    s = np.unique(np.r_[np.linspace(0, 2 * L, 401), XI * L])
    T = ds.total_tension(s)
    return dict(t=ds.t, eps_max=T.max() / EA, eps_min=T.min() / EA,
                eps_entry=T[-1] / EA, eps_exit=T[0] / EA, y=ds.takeup_displacement(),
                a_max=a_m * np.max(prof.a(ds.response.tau)), cv=cv)


# ------------------------------------------------------------------------------ diagnostics
def sliding_drive_char(Om, beta, loop):
    P = chain(loop.segments, Om)
    return 2.0 + np.trace(P) - beta * Om ** 2 * P[0, 1]


def sliding_drive_periods(mu_c=MU_C, M=M_W, n=3, Om_max=8.0):
    cr = np.sqrt(EA / MU_R)
    lp = Loop.from_positions(0.0, 0.5, np.sqrt(mu_c / MU_R))      # take-up split irrelevant
    beta = M / (MU_R * L)
    g = np.linspace(1e-3, Om_max, 80001)
    v = np.array([sliding_drive_char(o, beta, lp) for o in g])
    idx = np.nonzero(v[:-1] * v[1:] < 0)[0][:n]
    Om = np.array([brentq(sliding_drive_char, g[i], g[i + 1], args=(beta, lp)) for i in idx])
    return 2 * np.pi * L / (cr * Om)


def check_sliding_drive_uniform():
    """gamma = 1: 2 + 2 cos 2Om - beta Om sin 2Om = 2 cos Om (2 cos Om - beta Om sin Om)."""
    lp = Loop.from_positions(0.0, 0.5, 1.0)
    for beta in (0.05, 0.5, 3.0):
        for Om in (0.3, 1.1, 2.7):
            lhs = sliding_drive_char(Om, beta, lp)
            rhs = 2 * np.cos(Om) * (2 * np.cos(Om) - beta * Om * np.sin(Om))
            assert abs(lhs - rhs) < 1e-12


def fit_period(t, v, t0, t1, T0):
    m = (t >= t0) & (t <= t1)
    f = lambda t, c, A, T, ph, s: c + A * np.cos(2 * np.pi * (t - t0) / T + ph) * np.exp(-s * (t - t0))
    p, _ = curve_fit(f, t[m], v[m], p0=(np.mean(v[m]), np.ptp(v[m]) / 2, T0, 0.0, 0.0), maxfev=20000)
    return p     # mean, amplitude, period, phase, decay rate


def load_digitised():
    rows = [ln.split(",") for ln in (HERE / "data" / "lodewijks1996_fig8_41_42.csv").read_text().splitlines()
            if ln and not ln.startswith("#") and not ln.startswith("series")]
    out = {}
    for k in ("eps_drive", "eps_takeup", "y_m"):
        sel = np.array([[float(r[1]), float(r[2])] for r in rows if r[0] == k])
        out[k] = (sel[:, 0], sel[:, 1])
    return out


# ------------------------------------------------------------------------------ main
def main():
    OUT.mkdir(exist_ok=True)
    cv0 = conveyor()
    print(f"c_r = {cv0.c_r:.1f} m/s (C_v {cv0.c_r / 543.21:.3f}; Lodewijks 0.81), "
          f"c_c = {cv0.c_c:.1f} m/s (C_v {cv0.c_c / 543.21:.3f}; 0.30), gamma = {cv0.gamma:.3f}, "
          f"beta = {cv0.beta:.3f}, T_t = {cv0.takeup.T_t / 1e3:.2f} kN")
    print(f"running tension at drive entry {cv0.running_tension(2 * L - 1e-9) / 1e3:.2f} kN "
          f"(Lodewijks F_1 = 56.88 kN)")

    # -------------------------------------------------------------- Table 8.7
    print("\nTable 8.7 (t_a = 30 s): eps_max / y_max [m]; model A, onset phi = V/V_inf")
    print(f"{'profile':16s} {'Lodewijks':>16s} {'zeta1=0':>16s} {'zeta1=0.05':>16s}   a_max L / model")
    table = {}
    for name, (y_l, e_l, a_l) in TABLE_8_7.items():
        r0, r5 = run(name, 0.0), run(name, 0.05)
        table[name] = (r0, r5)
        print(f"{name:16s} {e_l:7.4f} / {y_l:6.2f} {r0['eps_max']:7.4f} / {r0['y'].max():6.2f} "
              f"{r5['eps_max']:7.4f} / {r5['y'].max():6.2f}   {a_l:.3f} / "
              f"{(r0['a_max'] if name != 'linear offset' else (V_INF - V_0) / T_A):.3f}")

    print("\nSensitivity (linear offset, zeta1 = 0.05): eps_max / y_max")
    for label, kw in [("base", {}), ("jump ramp 0.25 s", dict(t_j=0.25)), ("jump ramp 1 s", dict(t_j=1.0)),
                      ("step onset of resistances", dict(onset="step")),
                      ("take-up mass + 2 t", dict(M_extra=2000.0))]:
        r = run("linear offset", 0.05, **kw)
        print(f"  {label:28s} {r['eps_max']:.4f} / {r['y'].max():.2f}")

    # -------------------------------------------------------------- start-up time sweep
    print("\nStart-up time sweep, linear offset (Figs. 8.44, 8.46, 8.47): eps_max, y_max, eps_min")
    sweep = {}
    for ta, (e_l, y_l, m_l) in SWEEP.items():
        rs = [run("linear offset", z, t_a=ta) for z in (0.0, 0.05)]
        sweep[ta] = rs
        print(f"  t_a={ta:2d}: L {e_l:.4f} {y_l:5.1f} {m_l:+.4f} | z=0 {rs[0]['eps_max']:.4f} "
              f"{rs[0]['y'].max():5.2f} {rs[0]['eps_min']:+.4f} | z=0.05 {rs[1]['eps_max']:.4f} "
              f"{rs[1]['y'].max():5.2f} {rs[1]['eps_min']:+.4f}")

    # -------------------------------------------------------------- time histories, periods
    dig = load_digitised()
    te, ee = dig["eps_drive"]
    ty, yy = dig["y_m"]
    pe, py = fit_period(te, ee, 30, 60, 18), fit_period(ty, -yy, 30, 60, 18)
    r = run("linear offset", 0.0, t_end=90.0)
    pm = fit_period(r["t"], r["y"], 30, 90, 26)
    print("\nOscillation after the start (fit c + A cos(2 pi t/T + ph) exp(-s t), 30-60 s):")
    print(f"  Lodewijks: T = {py[2]:.1f} s (y), {pe[2]:.1f} s (strain); zeta ~ "
          f"{py[4] * py[2] / (2 * np.pi):.3f}; mean y {py[0]:.2f} m, mean strain {pe[0]:.4f}")
    print(f"  model A:   T = {pm[2]:.1f} s; mean y {pm[0]:.2f} m; running strain at entry "
          f"{cv0.running_tension(2 * L - 1e-9) / EA:.4f}")
    _, es = dig["eps_takeup"]
    print(f"  slack side, Lodewijks: strain {es.mean():.4f} +- {es.std():.4f} "
          f"(T_t/EA = {cv0.takeup.T_t / EA:.4f})")

    print("\nFundamental period [s] of candidate models (loaded unless stated):")
    for xi in (0.001, 0.25, 0.5, 0.75, 0.999):
        T = [2 * np.pi * L / (cv0.c_r * natural_frequencies(Loop.from_positions(0, xi, cv0.gamma), b, 1)[0])
             for b in (0.0, cv0.beta, 3 * cv0.beta)]
        print(f"  A, xi = {xi:5.3f}: beta = 0 / {cv0.beta:.2f} / {3 * cv0.beta:.2f}: "
              + " / ".join(f"{x:.1f}" for x in T))
    print(f"  carry strand alone, fixed at the drive and free at the tail: {4 * L / cv0.c_c:.1f}")
    check_sliding_drive_uniform()
    print("  B sliding drive (gamma = 1 factorisation checked): "
          + ", ".join(f"{x:.1f}" for x in sliding_drive_periods()))
    lp = cv0.loop()
    for md in (0.95, 3.0):
        Om = natural_frequencies_torque(lp, cv0.beta, md, 3.0, 20000)
        print(f"  A with torque drive, m_d = {md}: {2 * np.pi * L / (cv0.c_r * Om[0]):.1f}"
              + ("  (m_d = 0.95: motor + gearbox + pulley, Eqs. 8.14-8.16)" if md == 0.95 else ""))
    cv_e = conveyor(mu_c=MU_C_EMPTY)
    print(f"  A, empty carry strand: "
          f"{2 * np.pi * L / (cv_e.c_r * natural_frequencies(cv_e.loop(), cv_e.beta, 1)[0]):.1f}")
    rb = sliding_drive_slack()
    print(f"  B, linear offset start (lumped, undamped): slack-side tension {rb[0] / 1e3:.1f} to "
          f"{rb[1] / 1e3:.1f} kN; Lodewijks: constant at about T_t")

    # -------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    r5 = run("linear offset", 0.05, t_end=60.0)
    a = ax[0, 0]
    a.plot(te, ee, "k.", ms=1.5, label="Lodewijks, drive pulley")
    a.plot(*dig["eps_takeup"], ".", color="0.5", ms=1.5, label="Lodewijks, take-up")
    for rr, ls, lab in ((table["linear offset"][0], "-", r"model A, $\zeta_1=0$"),
                        (r5, "--", r"model A, $\zeta_1=0.05$")):
        a.plot(rr["t"], rr["eps_entry"], "C0" + ls, lw=1, label=lab)
        a.plot(rr["t"], rr["eps_exit"], "C1" + ls, lw=1)
    a.set(xlim=(0, 60), xlabel="t [s]", ylabel="strain [-]", title="(a) linear offset start, 30 s")
    a.legend(fontsize=7)
    a = ax[0, 1]
    a.plot(ty, -yy, "k.", ms=1.5, label="Lodewijks")
    a.plot(table["linear offset"][0]["t"], table["linear offset"][0]["y"], "C0-", lw=1, label=r"$\zeta_1=0$")
    a.plot(r5["t"], r5["y"], "C0--", lw=1, label=r"$\zeta_1=0.05$")
    a.set(xlim=(0, 60), xlabel="t [s]", ylabel="take-up travel [m]", title="(b) take-up pulley")
    a.legend(fontsize=7)
    a = ax[1, 0]
    names = list(TABLE_8_7)
    x = np.arange(len(names))
    a.bar(x - 0.27, [TABLE_8_7[n][1] for n in names], 0.27, color="k", label="Lodewijks")
    a.bar(x, [table[n][0]["eps_max"] for n in names], 0.27, color="C0", label=r"$\zeta_1=0$")
    a.bar(x + 0.27, [table[n][1]["eps_max"] for n in names], 0.27, color="C0", alpha=0.5,
          label=r"$\zeta_1=0.05$")
    a.set_xticks(x, ["offset", "linear", "delayed", "Harrison", "Nordell"], fontsize=8)
    a.set(ylabel=r"$\varepsilon_{max}$ [-]", title="(c) Table 8.7")
    a.legend(fontsize=7)
    a = ax[1, 1]
    tas = list(SWEEP)
    a.plot(tas, [SWEEP[t][1] for t in tas], "ks-", label=r"Lodewijks $y_{max}$")
    a.plot(tas, [sweep[t][0]["y"].max() for t in tas], "C0o-", label=r"model, $\zeta_1=0$")
    a.plot(tas, [sweep[t][1]["y"].max() for t in tas], "C0o--", label=r"model, $\zeta_1=0.05$")
    a.set(xlabel="start-up time [s]", ylabel=r"$y_{max}$ [m]", title="(d) linear offset, start-up time")
    a2 = a.twinx()
    a2.plot(tas, [SWEEP[t][0] for t in tas], "k^:", ms=4)
    a2.plot(tas, [sweep[t][1]["eps_max"] for t in tas], "C1^:", ms=4)
    a2.set_ylabel(r"$\varepsilon_{max}$ (triangles; black Lodewijks, orange $\zeta_1=0.05$)", fontsize=7)
    a.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"lodewijks_case.{ext}", dpi=150)


def sliding_drive_slack(N=400, dt=0.01, t_end=60.0):
    """Configuration B with an independent lumped model (undamped, linear offset start):
    range of the slack-side tension. Quasi-statically T(0) + T(2L) = F_T, so the slack
    side must unload by half of the drive force."""
    h = 2 * L / N
    x = np.arange(N + 1) * h
    mu = np.where(0.5 * (x[1:] + x[:-1]) < L, MU_R, MU_C)
    r = np.where(0.5 * (x[1:] + x[:-1]) < L, R_R, R_C)
    m = np.zeros(N + 1); fr = np.zeros(N + 1)
    for e in range(N):
        m[e:e + 2] += mu[e] * h / 2
        fr[e:e + 2] += r[e] * h / 2
    k = EA / h
    G = np.zeros((N + 1, N))                      # w = G q, q = (w_1..w_{N-1}, y)
    G[1:N, :N - 1] = np.eye(N - 1)
    G[0, -1], G[N, -1] = -1.0, 1.0
    D = np.diff(np.eye(N + 1), axis=0)            # element elongation
    K = k * G.T @ D.T @ D @ G
    M = G.T @ np.diag(m) @ G
    M[-1, -1] += M_W
    fa, fres = -G.T @ m, -G.T @ fr

    def kin(t):
        if t >= T_A:
            return 0.0, V_INF
        a = (V_INF - V_0) / T_A + (V_0 / T_J if t < T_J else 0.0)
        v = V_0 * min(t / T_J, 1.0) + (V_INF - V_0) * t / T_A
        return a, v

    q = np.zeros(N); v = np.zeros(N)
    a0, v0 = kin(0.0)
    acc = np.linalg.solve(M, fa * a0 + fres * v0 / V_INF)
    Kinv = np.linalg.inv(K + 4 / dt ** 2 * M)
    Tslack = [F_T / 2]
    for n in range(1, int(round(t_end / dt)) + 1):
        ak, vk = kin(n * dt)
        F = fa * ak + fres * vk / V_INF
        qn = Kinv @ (F + M @ (4 / dt ** 2 * q + 4 / dt * v + acc))
        vn = 2 / dt * (qn - q) - v
        acc = 4 / dt ** 2 * (qn - q) - 4 / dt * v - acc
        q, v = qn, vn
        Tslack.append(F_T / 2 + k * (D @ G @ q)[0])
    return min(Tslack), max(Tslack)


if __name__ == "__main__":
    main()
