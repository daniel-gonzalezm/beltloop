# beltloop

Longitudinal dynamics of a belt-conveyor loop with a gravity take-up at an arbitrary
position and the drive anywhere on the loop. Continuous 1-D model with two wave speeds
(return and carry strands), take-up modelled as a moving pulley with mass (2:1 kinematics
for a single loop; a carriage carried by n belt strands maps onto it exactly with the
belt-side mass 4M/n^2), prescribed drive velocity and stiffness-proportional Kelvin-Voigt
damping. The derivation is in the "Model" section of the manuscript (`model.tex`).

Code for the paper *Longitudinal modal analysis of long belt conveyors with a gravity take-up
at an arbitrary position* (in preparation; title provisional).

## Main results reproduced here

* **Closed-form characteristic equation** `beta Om^2 (BA)_12 - 4 A_22 B_11 = 0` for the whole
  loop, with transfer matrices `A` (drive exit to take-up) and `B` (take-up to drive entry).
* **The take-up as a free end.** As `beta -> 0` the loop splits into two strands, fixed at the
  drive and free at the take-up; every loop mode lies between consecutive modes of the two
  strands. The take-up mass enters as a tip-mass correction given in closed form
  (`takeup_mass_approx`, exact threshold `takeup_mass_threshold`).
* **The take-up mass follows from the take-up tension.** With a gravity take-up,
  `beta = (4/n) lam T_t / W_r`, `W_r = g mu_r L` the weight of the return strand (`lam = 1` for a
  directly hung counterweight). Designs set `T_t` from length-proportional requirements (drive
  grip, holding the return strand on a slope), so `beta` does not grow with length; in the
  literature `T_t / W_r` is 0.01-0.10 for conveyors of 1 km or more (`maps/beta_min.py`).
* **Fundamental mode** belongs to the strand that contains the carry strand (for `gamma >= 1`),
  with closed-form bounds on its period and participation (`strand_participation`).
* **Start-up metrics** follow a universal curve of the fixed-free strand: peak tension at the
  drive entry, minimum at the drive exit and take-up travel (`metrics.py`).
* **Validity checks** with closed-form estimates: belt slack (drive exit during the start and
  post-start rebound), take-up following the belt, and no slip on the drive pulley.

## Layout

| Module | Content |
|---|---|
| `loop.py` | Loop geometry from drive and take-up positions (take-up anywhere on the loop, including head and tail pulleys); segment chains upstream/downstream of the take-up; quasi-static fields `Q(x) = int_xi^x q` |
| `transfer.py` | Transfer matrices `S`, `J`; characteristic functions (prescribed velocity; drive without speed control, with mass and optional slip dashpot); wave transmission through the take-up; Pruefer angle |
| `eigen.py` | Natural frequencies by root interlacing; mass-normalised modes with exact integrals; participation factors; fixed-free strand modes; take-up mass correction (one- and two-pole) and exact `beta` threshold per strand |
| `forcing.py` | Start-up profiles (sine, triangular, parabolic; `PiecewiseProfile` for piecewise-polynomial accelerations) and resistance onset (`velocity`, `step`, `none`) as piecewise exosystems |
| `response.py` | Exact modal integration (matrix exponential); tension with the lagged quasi-static split; take-up motion and velocity |
| `metrics.py` | Universal fixed-free strand curves, start-up metrics, start with an initial crawl plateau, take-up kinematics, tension requirements (slack, rebound, drive grip) |
| `conveyor.py` | SI layer: gravity take-up with reeving and n belt strands, static tension with an elevation profile, running tension, start-up in SI units, validity checks |
| `lumped.py` | Independent lumped-mass model for validation: absolute displacements with the drive moving, gravity and counterweight as loads, Newmark average acceleration |

Outside the package:

| Folder | Content |
|---|---|
| `validation/` | Lumped-mass validation figure; Harrison (1983, 1985) and Lodewijks (1996, ch. 8) case studies, with digitised data in `validation/data/` |
| `maps/` | Parametric study. `cases.py` holds the conveyors from the literature (single source, with the provenance of every input and the conventions used to derive the line densities and the take-up mass ratio). `paper_figures.py` regenerates the figures of the paper from stored data in `maps/data/` (`--recompute` to recalculate, ~5 min); the other scripts are the working figures of each step |
| `examples/` | Minimal start-up example |

## Numerical method (summary)

* **Roots.** The characteristic equation is equivalent to `A12/A22 + B12/B11 = 4/(beta Om^2)`.
  Both receptances increase with `Om` between poles, so the natural frequencies strictly
  interlace with the merged fixed-free frequencies of the two strands (zeros of `A22` and
  `B11`), which are bracketed exactly with the Pruefer angle. Every root is found, including
  pairs closer than any scanning grid (small `beta`, coincident fixed-free modes).
* **Modes.** Propagated analytically segment by segment; modal mass, participation factors
  and the Rayleigh quotient use closed-form segment integrals (no quadrature).
* **Time integration.** Each modal equation is augmented with the exosystem that generates
  the load on each time interval and integrated with the matrix exponential: exact for any
  damping, overdamped high modes and exact resonance.
* **Tension.** Quasi-static split: without inertia a Kelvin-Voigt belt carries exactly the
  static tension `a Q_mu + phi Q_r` for any damping, while the strain lags through a
  first-order filter with time constant `2 zeta_hat`, the same for every mode. The modal
  remainder is taken about that lagged state and converges fast with or without damping
  (20 modes: ~1e-6 relative in the cases tested).

## Validation

`validation/fig_lumped_validation.py` compares the modal solution with the lumped model
(`validation/figures/lumped_validation.pdf`): second-order convergence in the number of
elements, ~1e-6 relative difference in the whole tension field at N = 1000.

`validation/harrison_case.py` and `validation/lodewijks_case.py` compare with the
measurements of Harrison (1983, 1985) and with the speed-controlled starts of Lodewijks
(1996, ch. 8). Both are consistency checks and cross-checks, not full validations; what the
model does and does not reproduce is printed by the scripts.

## Scaling

`x = s/L` (loop coordinate from the drive exit), `tau = c_r t/L`, `Om = omega L/c_r`,
`gamma = c_r/c_c`, `beta = M_belt/(mu_r L)` with `M_belt = 4M/n^2`, `xi = s_t/L`,
`zeta_hat = c_r t_v/(2L)`; tensions in units of `mu_r L a_m`, displacements in
`a_m L^2/c_r^2`, resistances in `mu_r a_m`.

## Use

```bash
pip install -e ".[dev,plots]"
pytest -q                          # ~3.5 min (349 tests)
python examples/startup_demo.py
python maps/paper_figures.py       # figures of the paper from stored data
python maps/cases.py               # table of the literature cases
python maps/beta_min.py            # take-up tension against its DIN requirements
```

```python
import numpy as np
from beltloop import Loop, modal_basis, StartProfile, startup_response

loop = Loop.from_positions(sigma_d=0.0, sigma_t=0.05, gamma=1.6, r_return=0.3, r_carry=0.9)
basis = modal_basis(loop, beta=0.12, n=60)
resp = startup_response(basis, StartProfile("sine", tau_a=30.0, onset="velocity"),
                        zeta_hat=0.05, tau=np.linspace(0, 120, 2401))
T = resp.tension([0.0, 2.0])           # drive exit and entry
```

## Scope

Out of scope by design: take-up pulley inertia and friction, retarders and capstans,
drive slip, non-homogeneous strands, sag nonlinearity at low tension. The validity checks
(positive total tension, take-up acceleration below `n T_t / M`, no slip on the drive pulley)
flag when the linear model stops applying.

## License

Not yet licensed; a licence and an archived version with a DOI will be added on publication.
