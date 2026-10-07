"""Phase 5.5: application start-ups on conveyors from the literature.

For each case the take-up is moved along the return strand with the case's counterweight
(same mass, hence the same T_t and beta) and a speed-controlled start is simulated with the
full loop model (`Conveyor.start`, sine profile, resistances with onset phi = V / V_inf,
Kelvin-Voigt damping zeta_hat = 0.01). Strand resistances follow DIN 22101 eq. 14 with a
common C f fitted to the case's F_U (`beta_min.strand_resistances`); gravity enters the
static state through the case's carry profile.

Total tension = T_t + offset(s, t), and the offset does not depend on T_t once the
counterweight mass is fixed, so every requirement is a closed bound on T_t from one run:

  slack      T(s, t) > 0 on the whole loop, during the start and after it
             (validity of the linear model, Section 4.18, condition 1);
  grip       T_entry(t) <= E T_exit(t), E = exp(mu theta) of the source (condition 3);
  sag        running carry and return tensions above g m' l / (8 h_rel), h_rel = 2 %
             (DIN 22101 eqs. 51-52 at the practice of the sources; phase 5.4).

The required take-up tension T_t,req is the largest of the three. The shortest admissible
start t_a,req is the shortest sine start for which the case's own T_t meets every condition
from that duration on (inf when the running state alone fails). The carriage check (condition
2) is reported but never governs (margin g / |y''| > 10 in every run).

Cases (phase 5.5, decided with Daniel): Lo (Lodewijks 1996, 1 km, one drive pulley, E = 3,
published 30 s starts), SM (Suchorab-Matuszewska et al. 2025, 3 km incline, tandem drive,
E = 16.4, no published start: reference 3 T_1 at the real position, the rule of Section 4.16)
and Su (Surtees 1995, 805 m, intermediate drive, E = 11.5, published 25 s). Si, LL and Pa were
set aside (Si: E and profile unknown, start at 12 T_1; LL: motor steps, F_U assumed, decline;
Pa: no start time, no E).

Sweep convention: same counterweight at every position. With the counterweight resized to keep
the running T_2, the requirements and T_1 do not change (< 0.3 %); only the shortest start
does, and for Lo it becomes ~101 s at every position (grip depends on T_2 alone).

Time histories (last row of the figure): SM real against the tail (its best position), Su real
against the tight side (its worst); Lo at its real position, which is also its best, with the
published 30 s start against the shortest admissible one (~101 s). Each pair is simulated over a
common window.

    python maps/applications.py            # tables and working figure (cached data)
    python maps/applications.py --recompute    # about 6 min
    python maps/paper_figures.py applications  # figure and table of the paper
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import beta_min as bm  # noqa: E402
import cases as cs  # noqa: E402
from beltloop import natural_frequencies  # noqa: E402

G = cs.G_STD
ZETA_HAT = 0.01            # base damping of the project (Section 4.4)
H_REL_SAG = 0.02           # sag criterion in running (practice of the sources, phase 5.4)
KIND = "sine"
TAGS = ("Lo", "SM", "Su")
REF_MULT = 3.0             # reference start for cases without a published duration: 3 T_1
TA_GRID = np.geomspace(4.0, 1200.0, 46)


# ----------------------------------------------------------------------------- one run
def conveyor(c: cs.Case, sigma_t: float):
    """Dimensional conveyor of the case with the take-up at sigma_t (units of L), the case's
    counterweight, DIN strand resistances and the base damping."""
    cv = bm.running_conveyor(c, sigma_t=sigma_t)
    cv.t_v = 2.0 * ZETA_HAT * cv.L / cv.c_r
    return cv


def fundamental(cv) -> float:
    """Fundamental period of the loop, s (exact, with the case's beta)."""
    return 2 * np.pi * cv.L / (cv.c_r * natural_frequencies(cv.loop(), cv.beta, 1)[0])


def sag_requirement(c: cs.Case, sigma_t: float, h_rel: float = H_REL_SAG) -> float:
    """T_t for running sag below h_rel on both strands (DIN eqs. 51-52)."""
    return bm.sag_requirement(c, bm.state(c, 1.0, sigma_t), h_rel)[0]


@dataclass
class Run:
    t_a: float
    T1: float
    T_t: float               # the case's take-up tension, N
    req_slack: float         # T_t needed for positive tension everywhere, N
    req_grip: float          # T_t needed for T_entry <= E T_exit at all times, N
    req_sag: float           # T_t needed for running sag <= H_REL_SAG, N
    entry_peak: float        # peak total tension at the drive entry, N (with the case's T_t)
    exit_min: float          # minimum total tension at the drive exit, N
    travel: float            # peak carriage travel from rest, m
    accel_margin: float      # (n T_t / M) / max carriage acceleration
    t: np.ndarray | None = None
    T_entry: np.ndarray | None = None
    T_exit: np.ndarray | None = None
    y: np.ndarray | None = None

    @property
    def req(self) -> float:
        return max(self.req_slack, self.req_grip, self.req_sag)

    @property
    def governing(self) -> str:
        v = {"slack": self.req_slack, "grip": self.req_grip, "sag": self.req_sag}
        return max(v, key=v.get)

    @property
    def ok(self) -> bool:
        return self.T_t >= self.req


def run(c: cs.Case, sigma_t: float, t_a: float, keep: bool = False, req_sag: float | None = None,
        n_modes: int = 80, n_t: int = 2001, n_s: int = 601, t_end: float | None = None) -> Run:
    cv = conveyor(c, sigma_t)
    ds = cv.start(c.V.value, t_a, KIND, n_modes=n_modes, n_t=n_t, t_end=t_end)
    L, Tt = cv.L, cv.takeup.T_t
    s = np.unique(np.r_[np.linspace(0.0, 2 * L, n_s), cv.xi * L])
    T = ds.total_tension(s)
    off = T - Tt
    E = c.E.value
    a, b = off[0], off[-1]                       # drive exit (s = 0), drive entry (s = 2L)
    ydd = ds.takeup_acceleration()
    y = ds.takeup_displacement()
    if req_sag is None:
        req_sag = sag_requirement(c, sigma_t)
    out = Run(t_a, fundamental(cv), Tt, float(-off.min()), float(np.max((b - E * a) / (E - 1.0))),
              float(req_sag), float(T[-1].max()), float(T[0].min()), float(y.max()),
              float(cv.takeup.max_acceleration / max(np.abs(ydd).max(), 1e-12)))
    if keep:
        out.t, out.T_entry, out.T_exit, out.y = ds.t, T[-1], T[0], y
    return out


def shortest_start(c: cs.Case, sigma_t: float, grid=TA_GRID, tol: float = 0.01) -> float:
    """Shortest sine start for which the case's own T_t meets slack, grip and sag from that
    duration on (requirements are not monotonic in t_a: residual oscillations vanish at
    some durations). inf if the longest start of the grid fails (running state)."""
    req_sag = sag_requirement(c, sigma_t)
    ok = np.array([run(c, sigma_t, ta, req_sag=req_sag).ok for ta in grid])
    if not ok[-1]:
        return np.inf
    bad = np.nonzero(~ok)[0]
    if bad.size == 0:
        return float(grid[0])
    lo, hi = grid[bad[-1]], grid[bad[-1] + 1]
    while hi / lo - 1 > tol:
        mid = np.sqrt(lo * hi)
        if run(c, sigma_t, mid, req_sag=req_sag).ok:
            hi = mid
        else:
            lo = mid
    return float(hi)


# ----------------------------------------------------------------------------- per case
def reference_start(c: cs.Case) -> tuple[float, str]:
    """(t_a, provenance): published duration, else REF_MULT T_1 at the real position."""
    if c.t_a is not None:
        return c.t_a.value, "published"
    return REF_MULT * fundamental(conveyor(c, c.sigma_t[0])), f"{REF_MULT:g} T_1 (assumed)"


def positions(c: cs.Case, n: int = 25) -> np.ndarray:
    """Take-up positions along the return strand (units of L), the real one included; with an
    intermediate drive both faces (tight side sigma_t < sigma_d), the drive itself excluded."""
    sd = c.sigma_d.value
    s = np.linspace(0.005, 0.995, n)
    s = s[np.abs(s - sd) > 0.012]
    return np.unique(np.r_[s, c.sigma_t[0], sd + 0.002 if sd > 0 else 0.001])


def alternative(c: cs.Case) -> float:
    """Alternative take-up position for the time histories: the tail end of the return with a
    head drive; with an intermediate drive, the tight side, half way between the head and the
    drive (the zone that Section 4.18 calls impractical)."""
    sd = c.sigma_d.value
    return 0.995 if sd == 0.0 else 0.5 * sd


def sweep(c: cs.Case, n: int = 25, t_a_req: bool = True) -> dict:
    ta, prov = reference_start(c)
    sig = positions(c, n)
    rows = [run(c, s, ta) for s in sig]
    out = dict(sigma=sig, t_a=ta, T1=np.array([r.T1 for r in rows]),
               req_slack=np.array([r.req_slack for r in rows]),
               req_grip=np.array([r.req_grip for r in rows]),
               req_sag=np.array([r.req_sag for r in rows]),
               entry=np.array([r.entry_peak for r in rows]),
               exit_min=np.array([r.exit_min for r in rows]),
               travel=np.array([r.travel for r in rows]),
               accel_margin=np.array([r.accel_margin for r in rows]),
               T_t=rows[0].T_t)
    if t_a_req:
        out["t_a_req"] = np.array([shortest_start(c, s) for s in sig])
    return out


def histories(c: cs.Case, t_a: float) -> dict:
    """Drive-entry and drive-exit tensions and carriage travel at the real and the alternative
    take-up positions, reference start, case's counterweight."""
    d = {}
    pos = (("real", c.sigma_t[0]), ("alt", alternative(c)))
    t_end = t_a + 3.0 * max(fundamental(conveyor(c, s)) for _, s in pos)   # common window
    for lab, s in pos:
        r = run(c, s, t_a, keep=True, n_t=3001, t_end=t_end)
        d[f"{lab}_sigma"] = s
        d[f"{lab}_t"], d[f"{lab}_entry"], d[f"{lab}_exit"], d[f"{lab}_y"] = r.t, r.T_entry, r.T_exit, r.y
    return d


def duration_histories(c: cs.Case, t_a: float) -> dict:
    """Last row for Lo (decision of phase 5.5): same (real) position, the reference start and the
    shortest admissible start with the case's T_t, both simulated over the same window."""
    d = {}
    ta_min = shortest_start(c, c.sigma_t[0])
    T1 = fundamental(conveyor(c, c.sigma_t[0]))
    t_end = max(t_a, ta_min) + 3.0 * T1          # common window for both starts
    for lab, ta in (("ref", t_a), ("min", ta_min)):
        r = run(c, c.sigma_t[0], ta, keep=True, n_t=4001, t_end=t_end)
        d[f"dur_{lab}_ta"] = ta
        d[f"dur_{lab}_t"], d[f"dur_{lab}_entry"], d[f"dur_{lab}_exit"] = r.t, r.T_entry, r.T_exit
    return d


def compute(tags=TAGS) -> dict:
    d = {}
    for tag in tags:
        c = cs.BY_TAG[tag]
        sw = sweep(c)
        sw.update(histories(c, sw["t_a"]))
        if tag == "Lo":
            sw.update(duration_histories(c, sw["t_a"]))
        for k, v in sw.items():
            d[f"{tag}_{k}"] = np.asarray(v)
    return d


def data(recompute: bool = False) -> dict:
    path = HERE / "data" / "applications.npz"
    if path.exists() and not recompute:
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    d = compute()
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **d)
    return d


# ----------------------------------------------------------------------------- Lodewijks
LO_PROFILES = ("linear offset", "linear", "linear delayed", "Harrison (6.1)", "Nordell (6.2)")


def lodewijks_grip(zetas=(0.0, 0.05)):
    """Grip of the 30 s starts of Lodewijks (1996, Table 8.7) with his take-up (21.33 kN) and
    one drive pulley (E = e^(0.35 pi) = 3.00), with the profiles and the 5.11 m/s effective
    speed of validation/lodewijks_case.py. Rows: (profile, zeta1, max T_entry / T_exit,
    T_t needed for grip, N)."""
    sys.path.insert(0, str(HERE.parent / "validation"))
    import lodewijks_case as lc
    E = cs.BY_TAG["Lo"].E.value
    rows = []
    for name in LO_PROFILES:
        for z in zetas:
            cv = lc.conveyor(z)
            prof, a_m = lc.profile(name, cv)
            ds = cv.start_profile(prof, a_m, lc.T_A + 40.0, n_modes=120, n_t=3001)
            T = ds.total_tension(np.array([0.0, 2 * lc.L]))
            off = T - cv.takeup.T_t
            req = float(np.max((off[1] - E * off[0]) / (E - 1.0)))
            rows.append((name, z, float(np.max(T[1] / T[0])), req))
    return rows


# ----------------------------------------------------------------------------- output
def table(d: dict):
    for tag in TAGS:
        c = cs.BY_TAG[tag]
        sig = d[f"{tag}_sigma"]
        ta = float(d[f"{tag}_t_a"])
        Tt = float(d[f"{tag}_T_t"])
        print(f"\n{tag}: {c.name}; t_a = {ta:.1f} s ({reference_start(c)[1]}); T_t = {Tt / 1e3:.1f} kN; "
              f"E = {c.E.value:.2f}; sigma_d = {c.sigma_d.value:.3f}; real sigma_t = {c.sigma_t[0]:.3f}")
        print(f"{'sig_t':>6} {'T1 s':>6} {'ta/T1':>6} {'slack':>7} {'grip':>7} {'sag2%':>7} {'req':>7} "
              f"{'gov':>5} {'entry':>7} {'exit':>7} {'travel':>6} {'ta_req':>7} {'acc':>5}")
        for k, s in enumerate(sig):
            r = [d[f"{tag}_{q}"][k] for q in ("req_slack", "req_grip", "req_sag")]
            gov = ("slack", "grip", "sag")[int(np.argmax(r))]
            mark = " *" if abs(s - c.sigma_t[0]) < 1e-9 else ""
            print(f"{s:6.3f} {d[f'{tag}_T1'][k]:6.2f} {ta / d[f'{tag}_T1'][k]:6.2f} "
                  + " ".join(f"{v / 1e3:7.1f}" for v in r) + f" {max(r) / 1e3:7.1f} {gov:>5} "
                  f"{d[f'{tag}_entry'][k] / 1e3:7.0f} {d[f'{tag}_exit_min'][k] / 1e3:7.1f} "
                  f"{d[f'{tag}_travel'][k]:6.2f} {d[f'{tag}_t_a_req'][k]:7.1f} "
                  f"{d[f'{tag}_accel_margin'][k]:5.0f}{mark}")


# ----------------------------------------------------------------------------- paper table
def real_alt_rows():
    """Rows of the paper table: the configurations of the last row of the figure, plus Lo with
    its take-up at the tail. Each row: dict with the case, label, sigma_t, xi, T1, t_a, peak entry
    tension and its ratio to the running one, minimum exit tension, carriage travel, required
    T_t and governing criterion, required running T_2, the case's T_t and the shortest start."""
    rows = []
    for tag in TAGS:
        c = cs.BY_TAG[tag]
        ta, prov = reference_start(c)
        confs = [("real", c.sigma_t[0], ta)]
        if tag == "Lo":
            confs += [("real, shortest start", c.sigma_t[0], shortest_start(c, c.sigma_t[0])),
                      ("tail", alternative(c), ta)]
        else:
            confs += [("tail" if c.sigma_d.value == 0 else "tight side", alternative(c), ta)]
        for lab, s, t in confs:
            r = run(c, s, t)
            st = bm.state(c, 1.0, s)
            rows.append(dict(tag=tag, label=lab, sigma=s, xi=(s - c.sigma_d.value) % 2.0, T1=r.T1,
                             t_a=t, t_a_prov=prov if t == ta else "shortest admissible",
                             entry=r.entry_peak, entry_ratio=r.entry_peak / st.T_run[-1],
                             exit_min=r.exit_min, travel=r.travel, req=r.req, gov=r.governing,
                             T2_req=r.req - (st.T_t - st.T_run[0]), T_t=r.T_t,
                             t_a_req=shortest_start(c, s)))
    return rows


def write_table(rows, path):
    head = (f"{'case':4} {'take-up':22} {'sig_t':>6} {'xi':>6} {'T1 s':>6} {'t_a s':>6} {'t_a/T1':>6} "
            f"{'entry kN':>8} {'/run':>5} {'exit min':>8} {'travel m':>8} {'Tt req':>7} {'governs':>8} "
            f"{'T2 req':>7} {'Tt':>6} {'t_a,min s':>9}")
    lines = ["Application start-ups (phase 5.5): sine profile, zeta_hat = 0.01, DIN resistances with "
             "onset phi = V/V_inf, case's counterweight. Tensions in kN.", head]
    for r in rows:
        tm = "-" if not np.isfinite(r["t_a_req"]) else f"{r['t_a_req']:.0f}"
        lines.append(f"{r['tag']:4} {r['label']:22} {r['sigma']:6.3f} {r['xi']:6.3f} {r['T1']:6.2f} "
                     f"{r['t_a']:6.1f} {r['t_a'] / r['T1']:6.2f} {r['entry'] / 1e3:8.0f} "
                     f"{r['entry_ratio']:5.2f} {r['exit_min'] / 1e3:8.1f} {r['travel']:8.2f} "
                     f"{r['req'] / 1e3:7.1f} {r['gov']:>8} {r['T2_req'] / 1e3:7.1f} {r['T_t'] / 1e3:6.1f} {tm:>9}")
    lines += ["", "t_a,min: shortest sine start that the case's T_t admits (slack, grip, sag 2 %); "
              "'-': none (the running state fails).", "SM reference start: 3 T_1 at the real position "
              "(assumed; no published start)."]
    Path(path).write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figure
LABELS = {"Lo": "Lo: 1 km, one drive pulley", "SM": "SM: 3 km incline, tandem drive",
          "Su": "Su: 805 m, intermediate drive"}


def figure(d: dict, path=None, lo_last_row: str = "duration"):
    """Working figure (3 x 3). Columns: cases. Rows: (1) take-up tension required at the
    reference start against the take-up position, by criterion, with the case's T_t;
    (2) shortest admissible start with the case's T_t, the 3 T_1 rule and the reference start;
    (3) drive-entry and drive-exit tensions, real (solid) and alternative (dashed) position."""
    import matplotlib.pyplot as plt
    import style
    style.apply()
    fig, axs = plt.subplots(3, 3, figsize=(style.DOUBLE, 0.78 * style.DOUBLE),
                            constrained_layout=True)
    cols = style.OKABE_ITO
    for j, tag in enumerate(TAGS):
        c = cs.BY_TAG[tag]
        sig = d[f"{tag}_sigma"]
        sd, sr, sa = c.sigma_d.value, float(d[f"{tag}_real_sigma"]), float(d[f"{tag}_alt_sigma"])
        Tt, ta = float(d[f"{tag}_T_t"]) / 1e3, float(d[f"{tag}_t_a"])
        # split at the drive so that lines do not cross it
        parts = [sig < sd, sig > sd] if sd > 0 else [np.ones_like(sig, bool)]
        ax = axs[0, j]
        for m in parts:
            ax.plot(sig[m], d[f"{tag}_req_grip"][m] / 1e3, "-", color=cols[0])
            ax.plot(sig[m], d[f"{tag}_req_slack"][m] / 1e3, ":", color=cols[1])
            ax.plot(sig[m], d[f"{tag}_req_sag"][m] / 1e3, "--", color=cols[2])
        ax.axhline(Tt, color="k", lw=0.8)
        ax.text(0.02 if sd == 0 else 0.98, Tt, r"$T_t$ of the case", ha="left" if sd == 0 else "right",
                va="bottom", fontsize=7, transform=ax.get_yaxis_transform())
        if sd > 0:
            ax.set_yscale("log")
            ax.axvline(sd, color="0.6", lw=0.6)
            ax.text(sd, 0.02, " drive", transform=ax.get_xaxis_transform(), fontsize=6.5,
                    color="0.4")
        if sd == 0:
            ax.set_ylim(0, 1.12 * max(Tt, np.nanmax(d[f"{tag}_req_grip"]) / 1e3,
                                      np.nanmax(d[f"{tag}_req_slack"]) / 1e3))
        ax.set_ylabel(r"required $T_t$ (kN)" if j == 0 else "")
        style.panel_label(ax, f"({'abc'[j]}) {LABELS[tag]}")
        if j == 0:
            ax.plot([], [], "-", color=cols[0], label="grip")
            ax.plot([], [], ":", color=cols[1], label="slack")
            ax.plot([], [], "--", color=cols[2], label="sag 2 %")
            ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
        # row 2: start duration
        bx = axs[1, j]
        tr = d[f"{tag}_t_a_req"]
        for m in parts:
            fin = m & np.isfinite(tr)
            bx.plot(sig[fin], tr[fin], "-", color=cols[0])
            bx.plot(sig[m], 3 * d[f"{tag}_T1"][m], "--", color=cols[3])
            inf = m & ~np.isfinite(tr)
            if inf.any():
                bx.axvspan(sig[inf].min(), sig[inf].max(), color="0.88", lw=0)
        bx.axhline(ta, color="k", lw=0.8, ls=":")
        bx.set_yscale("log")
        from matplotlib.ticker import FuncFormatter, LogLocator
        bx.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        bx.yaxis.set_minor_locator(LogLocator(subs=np.arange(1, 10)))
        bx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        bx.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        fin_all = tr[np.isfinite(tr)]
        bx.set_ylim(0.7 * min(ta, fin_all.min(), 3 * d[f"{tag}_T1"].min()),
                    1.5 * max(ta, fin_all.max(), 3 * d[f"{tag}_T1"].max()))
        bx.set_ylabel(r"start duration (s)" if j == 0 else "")
        style.panel_label(bx, f"({'def'[j]})")
        if j == 0:
            bx.plot([], [], "-", color=cols[0], label=r"shortest admissible, $T_t$ of the case")
            bx.plot([], [], "--", color=cols[3], label=r"$3t_1$")
            bx.plot([], [], ":", color="k", label="reference start")
            bx.legend(loc="upper left")
        lo_by_duration = tag == "Lo" and lo_last_row == "duration"
        for a in (ax, bx):
            a.set_xlim(0, 1)
            a.plot([sr], [0.97], "v", color="k", transform=a.get_xaxis_transform(), clip_on=False)
            if not lo_by_duration:
                a.plot([sa], [0.97], "v", mfc="w", mec="k", transform=a.get_xaxis_transform(),
                       clip_on=False)
        if lo_by_duration:
            tmin = float(d[f"{tag}_dur_min_ta"])
            bx.plot([sr], [tmin], "o", color=cols[0], ms=4, clip_on=False)
            bx.annotate(f"{tmin:.0f} s", (sr, tmin), xytext=(6, 4), textcoords="offset points",
                        fontsize=6.5, color=cols[0])
        bx.set_xlabel(r"take-up position $\sigma_t$ (from the head, units of $L$)")
        # row 3: histories
        cx = axs[2, j]
        by_duration = tag == "Lo" and lo_last_row == "duration"
        for lab, ls in ((("ref", "-"), ("min", "--")) if by_duration else (("real", "-"), ("alt", "--"))):
            key = f"{tag}_dur_{lab}" if by_duration else f"{tag}_{lab}"
            t = d[f"{key}_t"]
            # entry plain, exit with open squares, entry / e^{mu theta} thin: readable without
            # colour (step 6.2); the curves are also labelled directly in panel (g)
            cx.plot(t, d[f"{key}_entry"] / 1e3, ls, color=cols[0])
            cx.plot(t, d[f"{key}_exit"] / 1e3, ls, color=cols[1], marker="s", ms=2.4, mfc="w",
                    mew=0.6, markevery=(0.04 if ls == "-" else 0.09, 0.1))
            cx.plot(t, d[f"{key}_entry"] / 1e3 / c.E.value, ls, color=cols[0], lw=0.5)
            if by_duration:
                tak = float(d[f"{key}_ta"])
                cx.axvline(tak, color="0.6", lw=0.6, ls=ls)
                right = lab == "ref"              # the longer start is labelled on its left,
                cx.text(tak, 0.97, f" {tak:.0f} s" if right else f"{tak:.0f} s ",   # clear of the legend
                        transform=cx.get_xaxis_transform(), fontsize=6.5, color="0.4", va="top",
                        ha="left" if right else "right")
        if not by_duration:
            cx.axvline(ta, color="0.6", lw=0.6)
        cx.axhline(0, color="0.6", lw=0.6)
        cx.set_xlabel("time (s)")
        cx.set_ylabel("tension (kN)" if j == 0 else "")
        style.panel_label(cx, f"({'ghi'[j]})" + (" same position, two start durations"
                                                   if by_duration else ""))
        if j == 0:
            cx.plot([], [], "-", color=cols[0], label="drive entry")
            cx.plot([], [], "-", color=cols[0], lw=0.5, label=r"entry / $\mathrm{e}^{\mu\theta}$")
            cx.plot([], [], "-", color=cols[1], marker="s", ms=2.4, mfc="w", mew=0.6,
                    label="drive exit")
            cx.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="none",
                      facecolor="w", borderpad=0.3)
        if j == 1:
            cx.plot([], [], "-", color="0.3", label="real position")
            cx.plot([], [], "--", color="0.3", label="alternative")
            cx.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=True,
                      framealpha=0.85, edgecolor="none", facecolor="w", borderpad=0.3)
    path = HERE / "figures" / "applications.pdf" if path is None else path
    fig.savefig(path)
    fig.savefig(Path(path).with_suffix(".png"), dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    d = data("--recompute" in sys.argv)
    table(d)
    print("\nLodewijks (1996) 30 s starts with his take-up, E = 3.00 (Table 8.7 profiles):")
    for name, z, ratio, req in lodewijks_grip():
        print(f"  {name:16s} zeta1 = {z:4.2f}: max T_entry / T_exit = {ratio:4.2f}; "
              f"grip needs T_t = {req / 1e3:5.1f} kN (has 21.33)")
    print(figure(d))
