"""Illustrative start-up of a horizontal head-drive conveyor (parameters are invented for
illustration, not taken from any case in the literature).

Run:  python examples/startup_demo.py   (writes startup_demo.png if matplotlib is present)
"""
import numpy as np

from beltloop import Conveyor, GravityTakeUp

g = 9.80665
cv = Conveyor(
    L=3000.0, EA=1.2e8, mu_r=40.0, mu_c=100.0, m_r=30.0, m_c=80.0,
    drive_position=0.0, takeup_position=60.0,
    takeup=GravityTakeUp(M_w=2 * 70e3 / g),          # T_t = 70 kN, directly hung
    r_r=15.0, r_c=45.0, t_v=0.3,
)
print(f"c_r = {cv.c_r:.0f} m/s, c_c = {cv.c_c:.0f} m/s, gamma = {cv.gamma:.3f}")
print(f"beta = {cv.beta:.3f}, xi = {cv.xi:.3f}, zeta_hat = {cv.zeta_hat:.4f}")

run = cv.start(V=5.0, t_a=60.0, kind="sine", onset="velocity", n_modes=60, t_end=150.0, n_t=3001)
b = run.response.basis
T1 = 2 * np.pi * cv.L / (cv.c_r * b.Om[:4])
print("first periods [s]:", np.round(T1, 2))
print("effective mass fractions:", np.round(b.effective_mass_fraction[:4], 3))
for k, v in run.checks().items():
    print(f"  {k}: {v}")

s = np.array([0.0, 2 * cv.L])
Tt = run.total_tension(s)
print(f"drive exit: min {Tt[0].min()/1e3:.1f} kN; drive entry: max {Tt[1].max()/1e3:.1f} kN")
print(f"running state: exit {cv.running_tension([0.0])[0]/1e3:.1f} kN, "
      f"entry {cv.running_tension([2*cv.L])[0]/1e3:.1f} kN")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(run.t, Tt[1] / 1e3, label="drive entry (tight side)")
    ax[0].plot(run.t, Tt[0] / 1e3, label="drive exit (slack side)")
    ax[0].axvline(60.0, color="0.6", lw=0.8, ls="--")
    ax[0].set_ylabel("total tension [kN]"); ax[0].legend()
    ax[1].plot(run.t, run.takeup_displacement())
    ax[1].set_ylabel("take-up travel y [m]"); ax[1].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig("startup_demo.png", dpi=130)
    print("figure: startup_demo.png")
except ImportError:
    pass
