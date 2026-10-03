# beltloop

Longitudinal dynamics of a belt-conveyor loop with a gravity take-up at an arbitrary
position on the return strand and the drive anywhere on the loop. Continuous 1-D model
with two wave speeds (return and carry strands), take-up modelled as a moving pulley with
mass (2:1 kinematics), prescribed drive velocity and stiffness-proportional Kelvin-Voigt
damping. The derivation is in the "Model" section of the manuscript (`model.tex`).

## Layout

| Module | Content |
|---|---|
| `loop.py` | Loop geometry: segment chains upstream/downstream of the take-up; quasi-static fields `Q(x) = int_xi^x q` |
| `transfer.py` | Transfer matrices `S`, `J`; characteristic functions (prescribed velocity, torque-controlled drive); Pruefer angle |
| `eigen.py` | Natural frequencies by root isolation; mass-normalised modes with exact integrals; participation factors |
| `forcing.py` | Start-up profiles (sine, triangular, parabolic) and resistance onset (`velocity`, `step`, `none`) as piecewise exosystems |
| `response.py` | Exact modal integration (matrix exponential); tension with the mode-acceleration correction; take-up motion |
| `conveyor.py` | SI layer: take-up reeving, static tension at rest, running tension, start-up in SI units, validity checks |
| `lumped.py` | Independent lumped-mass model for validation: absolute displacements with the drive moving, gravity and counterweight as loads, static equilibrium by linear solve, Newmark average acceleration |

## Numerical method (summary)

* **Roots.** The characteristic equation is equivalent to `A12/A22 + B12/B11 = 4/(beta Om^2)`.
  Both receptances increase with `Om` between poles, so the natural frequencies strictly
  interlace with the merged fixed-free frequencies of the two parts (zeros of `A22` and
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
elements, ~1e-6 relative difference in the whole tension field at N = 1000. Beyond that a
round-off floor of ~1e-6 appears (the lumped model integrates absolute displacements of
hundreds of metres); it has no practical relevance.

## Scaling

`x = s/L`, `tau = c_r t/L`, `Om = omega L/c_r`, `gamma = c_r/c_c`, `beta = M/(mu_r L)`,
`zeta_hat = c_r t_v/(2L)`; tensions in units of `mu_r L a_m`, displacements in
`a_m L^2/c_r^2`, resistances in `mu_r a_m`.

## Use

```bash
pip install -e ".[dev,plots]"
pytest -q                      # ~1.5 min
python examples/startup_demo.py
python validation/fig_lumped_validation.py   # ~1 min
```

```python
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
(positive total tension, take-up acceleration below `2 T_t / M`) flag when the linear
model stops applying.
