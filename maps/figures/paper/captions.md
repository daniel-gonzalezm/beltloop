# Draft captions of the paper figures (phase 4.8; numbering fixed in phase 6)

Figures are produced by `python maps/paper_figures.py`; the numbers below are the ones it prints
and are pinned by `tests/test_figures.py` (and the phase 4 tests) where marked †.

Case markers (all map panels), tagged by first author: H Harrison (1983, 1985b; gamma = 0.97
drawn at gamma = 1); S / St Song et al. (2012), take-up at the head / at the tail; G Gao et al.
(2026); LL Li and Li (2009); SM Suchorab-Matuszewska et al. (2025), KGHM variant 1; Lo Lodewijks
(1996, ch. 8); Pa Pascual et al. (2005); Si Sinaga (2008), KPC, provisional; WR Wheatley and
Rubel (2021), take-up position unknown, dotted line. Parameters in Table X (phase 5).

## fig_modal

**Natural frequencies and modal participation, head drive, beta -> 0 (the take-up as a free end).**
(a) Fundamental period T_1 c_r / L, from 4.01 to 13.42. (b) T_1 over four transits of the
downstream strand, t_B = 1 - xi + gamma (units of L/c_r): between 0.838 and 1 †; the
quarter-wave rule overestimates the period by up to 19 %, and is exact for gamma = 1 or a
take-up at the tail. (c) Effective-mass fraction of the fundamental, 0.41 to 0.81; hatched: a
take-up mass beta = 0.1 changes it by more than 5 % (7 % of the plane, where B1 and A1 are close
and exchange participation). (d) Intermediate drive at l_1 = sigma_d from the head, take-up right
after it: the return run between the head and the drive sits at the fixed end of the heavy
strand and acts as a spring, so T_1 exceeds four transits by up to 38 % (gamma = 3) †.
NC: Nordell and Ciozda (1984), T_1 = 27.5 s against 19.8 s with a head drive (+39 %) †;
Su: Surtees (1995), SASOL feed conveyor, design data, take-up assumed next to the drives.

## fig_beta

**Take-up mass at which a natural period becomes 5 % longer than with beta -> 0, each mode
followed by its strand.** Exact, from beta_5 = 4 / (Om_t^2 F(Om_t)), Om_t = Om_p / 1.05 †.
(a) B1, the fundamental: beta_5 from 0.11 (take-up at the tail, gamma = 1, where B1 and A1
coincide) to 1.66; every conveyor longer than 1 km lies far below its threshold.
(b) B2: median 0.39; dark grey, B2 and A1 within 5 % of each other (veering: the shift is not
a property of one strand); light grey, the 5 % shift is not reached even for beta -> infinity;
dashed, xi_s where A1 overtakes B2. Region edges computed exactly (Om_B2/Om_A1 = 1 and 1.05;
F(Om_t) = 0). (c) A1 for gamma = 2: the single-pole estimate beta_5 = 0.41 m_tilde =
0.205 xi (uniform strand) is the backbone; gaps and spikes mark veering with modes of strand B.
Circle: beta of the only case with the take-up at the tail (St, Song et al. as modelled),
where A1 is the second mode; the tick marks its own threshold and the label the exact
lengthening of A1 (+3.2 %, with beta from the published take-up tension). (Pa moved to the
head loop in phase 5.3(b); S takes beta from the tension since 5.3(c).)

## fig_startup

**Universal start-up curve of a fixed-free strand (beta -> 0, undamped).** tau_a: start time,
T_s: fundamental period of the strand. (a) Peak tension at the fixed end over m a_m for the sine,
triangular and parabolic profiles; dotted, the wave law T = Z V, exact for tau_a <= T_s / 2.
Sine: maximum 1.590 at tau_a / T_s = 0.87; D(1) = 1.574, D(2) = 1.206, D(5) = 1.077 †. Grey band:
design rule tau_a = 2-3 T_1. Dashed lines (all panels except c): start times of Lodewijks (1996;
linear profile in the source), Song et al. (2012) and Sinaga (2008). (b) The same per unit mean acceleration V_inf / t_a (same start time and final
speed): the triangular profile is the worst, by about 25 % on slow starts. (c) How far the full loop departs from the curve of (a): largest |loop / strand - 1| over the
take-up positions (xi = 0.01-0.95; 0.2-0.95 at the exit), at each start time, sine profile,
beta -> 0. Entry (one line per gamma) against tau_a / T_1; exit (all gamma) against
tau_a / T_A1, T_A1 = 4 xi L / c_r. For tau_a / T_s >= 0.8 the loop stays within 1.1 % of the
curve at the entry and 0.3 % at the exit; within 5.2 % from 0.5. Faster starts with gamma != 1
depart by up to 57 % at the entry (impedance jump at the tail, outside the strand picture); with
gamma = 1 the entry strand is uniform and the curve holds at all start times. Dotted: 2 %.
(d) Free-end displacement (take-up travel, maximum 1.80) and the tension due to resistances
starting with the belt speed (phi = V / V_inf). Damping lowers the maximum of (a) to 1.549,
1.496 and 1.425 for zeta_1 = 0.02, 0.05 and 0.1 (text).

## fig_crawl

**Initial crawl at 5 % of the final speed before a parabolic acceleration of 3 T_1 (uniform
strand, resistances starting in full when the belt moves).** rho: running resistance over the
inertial force; zeta_1: damping of the fundamental. (a) Duration of the ramp to crawl speed, no
hold: a ramp of one period T_1 removes the overshoot of the resistance part (peak 1.01-1.07,
against 1.17-1.66 with an abrupt onset). (b) Duration of the hold after an abrupt ramp: the hold
alone barely helps; with light damping it only changes the phase of the residual oscillation.
The take-up travel follows the same pattern (from 1.2-1.8 times quasi-static to about 1.03; text).

## fig_validity

**Validity of the linear prescribed-velocity model (beta -> 0, sine profile, undamped,
resistances proportional to the inertial density, r = rho mu a_m).** (a) Running tension at the
drive exit needed to keep strand A taut during the start, per unit mass of strand A; dotted,
slow-start limit sqrt(1 + rho^2/4) - rho/2. (b) Rebound of strand B below its running tension
after the start: up to 1.43 m_B a_m without resistances, zero at tau_a = (j + 1/2) T_1, decaying
as about 0.8 T_1 / tau_a; dashed, zeta_1 = 0.01. (c) Minimum running tension T_2 along the take-up
position, head drive, gamma = 2, rho = 1: grip governs with a single drive pulley (E = 3); with a
tandem drive (E = 16) slack, rebound and grip are comparable. (d) Required take-up tension with a
drive at mid-length of the return strand (full loop, E = 16, tau_a = 20 L/c_r): 1.8-2.1 m_belt a_m
on the tight side against 0.14-0.27 on the slack side, an impractical but valid region.
