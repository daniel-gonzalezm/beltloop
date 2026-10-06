# Draft captions of the paper figures (phase 4.8, revised in phase 5.3(d); numbering fixed in phase 6)

Figures are produced by `python maps/paper_figures.py`; the numbers below are the ones it prints
and are pinned by `tests/test_figures.py` (and the phase 4-5 tests) where marked †.

Case markers (all map panels), tagged by first author: H Harrison (1983, 1985b); S / St Song et
al. (2012), take-up at the head (their Fig. 1) / at the tail (their model); G Gao et al. (2026);
LL Li and Li (2009); SM Suchorab-Matuszewska et al. (2025), KGHM variant 1; Lo Lodewijks (1996,
ch. 8); Pa Pascual et al. (2005); Si Sinaga (2008), KPC. Circles at full material coupling
(alpha = 1); bars down to the lowest coupling of the material class (Lodewijks 2002):
alpha = 0.8 for fine or wet material and coal (S, G, LL, Lo, Si), alpha = 0.3 for coarse dry
rock (SM, Pa). H has no bar: gamma = 0.97 (0.90-1.06) was measured on the empty belt and
is drawn at gamma = 1. Cases with the take-up within xi <= 0.02 of the drive are spread over
xi <= 0.05 so that their bars do not overlap (display only), and St is drawn at xi = 0.985
instead of 0.999. Parameters in Table X (phase 5).

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
Su: Surtees (1995), SASOL plant feed conveyor, design data, take-up after the secondary drive
(their Fig. 10): T_1 / (4 t_B) = 0.952 with beta -> 0 (0.939 at alpha = 0.8, bar); its take-up
mass (beta = 0.64) lengthens T_1 by a further 2.1 %.

## fig_beta

**Take-up mass at which a natural period becomes 5 % longer than with beta -> 0, each mode
followed by its strand.** Exact, from beta_5 = 4 / (Om_t^2 F(Om_t)), Om_t = Om_p / 1.05 †.
(a) B1, the fundamental: beta_5 from 0.11 (take-up at the tail, gamma = 1, where B1 and A1
coincide) to 1.66. The conveyors of 1 km or longer have beta <= 0.21 and stay at least 3.7
times below their own threshold (2.8 times at the lowest coupling of their material); the take-up
mass lengthens their fundamental by at most 1.2 % (1.8 %). (b) B2: median 0.39; dark grey, B2 and A1 within 5 % of each other (veering: the shift
is not a property of one strand); light grey, the 5 % shift is not reached even for
beta -> infinity; dashed, xi_s where A1 overtakes B2. Region edges computed exactly
(Om_B2/Om_A1 = 1 and 1.05; F(Om_t) = 0). (c) A1 for gamma = 2: the single-pole estimate
beta_5 = 0.41 m_tilde = 0.205 xi (uniform strand) is the backbone. As the take-up nears the
drive, the pole of A1, pi/(2 xi), crosses the modes of strand B one after another (B2 at
xi = 0.69-0.72, B3 at 0.48-0.50, ..., B7 at 0.21). Before each crossing the exact threshold
leaves the backbone and rises to infinity (F(Om_t) = 0). Beyond it beta_5 of A1 is undefined
(grey bands, each starting at an asymptote): first the 5 % shift is not reached even for beta -> infinity (the
take-up stands still and the loop behaves as one string fixed at both sides of the drive; its
nearest mode lies 4.2-4.6 % below the pole of A1), then A1 veers with the mode of B and the shift
is no longer a property of one strand. The band ends where the pole of the B mode passes that
of A1; the new branch starts there at a finite value, up to 26 % below the backbone (no
asymptote on this side). Circle: beta of the only case with the take-up at the tail (St, Song et
al. as modelled), where A1 is the second mode; the tick marks its own threshold and the label the
exact lengthening of A1 (+3.2 %).

## fig_startup

**Universal start-up curve of a fixed-free strand (beta -> 0, undamped).** tau_a: start time,
T_s: fundamental period of the strand. (a) Peak tension at the fixed end over m a_m for the sine,
triangular and parabolic profiles; dotted, the wave law T = Z V, exact for tau_a <= T_s / 2.
Sine: maximum 1.590 at tau_a / T_s = 0.87; D(1) = 1.574, D(2) = 1.206, D(5) = 1.077 †. Grey band:
design rule tau_a = 2-3 T_1. Dashed lines (all panels except c): start times of the cases over
their own T_1 (exact, with their beta): Lo 1.05 (speed-controlled, linear profile in the
source), LL 1.49 (motors switched in 4 s steps), G 2.43 (simulated), Su 4.11 (fluid couplings,
design value), S 7.43 (take-up at the head), Si 11.8 (fluid couplings with fill control).
(b) The same per unit mean acceleration V_inf / t_a (same start time and final speed): the
triangular profile is the worst, by about 25 % on slow starts. (c) How far the full loop departs
from the curve of (a): largest |loop / strand - 1| over the take-up positions (xi = 0.01-0.95;
0.2-0.95 at the exit), at each start time, sine profile, beta -> 0. Entry (one line per gamma)
against tau_a / T_1; exit (all gamma) against tau_a / T_A1, T_A1 = 4 xi L / c_r. For
tau_a / T_s >= 0.8 the loop stays within 1.1 % of the curve at the entry and 0.3 % at the exit;
within 5.2 % from 0.5. Faster starts with gamma != 1 depart by up to 57 % at the entry (impedance
jump at the tail, outside the strand picture); with gamma = 1 the entry strand is uniform and the
curve holds within 0.6 % at all start times. Dotted: 2 %. (d) Free-end displacement (take-up
travel, maximum 1.80) and the tension due to resistances starting with the belt speed
(phi = V / V_inf). Damping lowers the maximum of (a) to 1.549, 1.496 and 1.425 for
zeta_1 = 0.02, 0.05 and 0.1 (text).

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

## Notes for phase 6 (not part of the captions)

- Phase 5.3(d) changes: coupling bars on all map panels and on Su in fig_modal (d); clustered
  cases spread into lanes, labels with leader lines; practice lines of fig_startup computed from
  cases.py (Lo, S, Si before; LL, G, Su added; Si moved from 11.1 to 11.8 with T_1 = 66 s);
  Su revised (design 2, take-up per Fig. 10, l_1 = 0.197); Pa at the head loop (5.3(b)); S with
  beta from the take-up tension (5.3(c)); Si no longer provisional.
- WR (Wheatley and Rubel 2021) left off the maps in 5.3(d): its take-up position is unknown, so
  its line across xi carried only gamma. It stays in the case table and goes to the text, with Su,
  as the short conveyors where beta starts to matter. Sentence for the text (numbers printed by
  `paper_figures.py beta`, pinned in tests): "The short conveyor of Wheatley and Rubel (275 m,
  beta = 0.88) sits at about 0.6 of its threshold for any take-up position, and at 0.95-1.02 of it
  with alpha = 0.8 (fundamental +4.7 to +5.1 %)." Su (805 m, beta = 0.64, intermediate drive):
  beta / beta_5(B1) = 0.42 (0.66 at alpha = 0.8), fundamental +2.1 % (+3.3 %).
- fig_modal colour maps cut at their dark end (viridis 0.30-1, magma 0.50-1, cividis 0.40-1);
  contour lines switch to white only over dark fill. fig_beta keeps the full viridis.
- fig_validity (c): legend with the four conditions in one column and the two start times in
  the other.
- fig_beta (c): branches now end on their exact asymptotes (before, the 400-point grid cut each
  spike at an arbitrary height, which looked like a numerical error). Gaps: one light grey band
  per gap, "not reached" and veering merged (A1_MODE = "band"). Tried and set aside: two greys as
  in (b) (the "not reached" part is 0.0005-0.006 wide, indistinguishable), asymptote lines with
  a strip along the xi axis ("strip", "merged"). Linear xi kept, as in (a) and (b); log xi was
  tried (it spreads the crossings and makes the backbone straight) and set aside.
- Reach of fig_beta (c), for the text (phase 5.3(d); pinned in tests/test_figures.py):
  (1) definition: for gamma = 2 the threshold of A1 is undefined everywhere below xi = 0.065
  (defined on 4 % of [0, 0.1], 29 % of [0, 0.2]); (2) damping: with Kelvin-Voigt,
  zeta_A1 = zeta_1 T_1 / T_A1 = zeta_1 (T_1 c_r / L) / (4 xi), so A1 is overdamped for
  xi <~ zeta_1 (T_1 c_r / L) / 4, between zeta_1 and 3.4 zeta_1 over the map (T_1 c_r / L =
  4.01-13.42); (3) excitation: tau_a / T_A1 = tau_a c_r / (4 xi L) reaches the design value 3
  only at xi = 1.11 (Lo), 1.27 (LL), 2.19 (G), 4.76 (Su), 5.83 (S), 10.8 (Si): no published start
  excites A1 beyond quasi-static, even with the take-up at the tail. A1 matters only with the
  take-up near the tail and a fast start (St).
- fig_beta (c) has a single case left (St): candidate for removal (section 4.20 of the notes).
- Practice lines mix drive types (speed control, fluid couplings, stepped motors): they show
  where real start times sit relative to T_1, not that those drives follow the sine profile.
- Accessibility (colour-only distinctions) still to be discussed before phase 6.
