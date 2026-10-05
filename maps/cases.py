"""Conveyors from the literature placed on the design maps (single source for all figures).

Tags follow the first author of the source (phase 4.8: the thesis names belt A and belt C are
dropped; they are Pascual et al. 2005 and Wheatley and Rubel 2021). Parameters as derived in phases 4.2-4.4 (sources and derivations in the comments of
beta_regime.py and in the project notes); they are revised in phase 5, so the figures read
them from here. Each entry: tag (marker label), name, L [m], mu_r, mu_c [kg/m], take-up mass
M [kg], number of strands n carrying the carriage, take-up positions xi (head drive, loop
coordinate), note. beta = 4 M / (n^2 mu_r L) (exact mapping of an n-strand take-up),
gamma = sqrt(mu_c / mu_r).

Sinaga 2008 (KPC), added in phase 4.8 as provisional: belt 29.7 kg/m, 4200 t/h at 8.5 m/s
(137.3 kg/m), catalogue idler estimates of 4 kg/m (return) and 9.5 kg/m (carry), 46.9 t
take-up with 115 kN on the belt (M g / T_t = 4.0, read as n = 4 direct; beta does not depend
on the rigging for an ideal rigging), next to the secondary head drive (xi ~ 0).
Surtees 1995 (SASOL) is not on the head-drive maps: its drives sit 152 m from the head
(l1 = 0.19), so it appears only on the intermediate-drive panel, with the take-up assumed
next to the drives.
"""
import numpy as np

G_STD = 9.80665

CASES = [
    # tag, name, L, mu_r, mu_c, M, n, xis, note
    ("H", "Harrison 1983/85", 5100, 79.0, 79.0 * 0.97 ** 2, 20e3, 4, [0.005],
     "gamma = 0.97 measured (plotted at gamma = 1); 79 kg/m"),
    ("S", "Song et al. 2012", 7117, 37.8, 104.9, 4500, 2, [0.001, 0.999],
     "head (their Fig. 1) / tail (their model)"),
    ("G", "Gao et al. 2026", 4500, 40.1, 194.3, 1000, 2, [0.001], "no idler mass given"),
    ("LL", "Li and Li 2009", 7600, 66.9, 259.8, 42800, 2, [0.005], "idlers from their c = 837 m/s"),
    ("SM", "Suchorab-Matuszewska et al. 2025 (KGHM, variant 1)", 3000, 74.6, 282.9,
     2 * 140e3 / G_STD, 2, [0.1], "QNK-TT report; n assumed"),
    ("Lo", "Lodewijks 1996, ch. 8", 1000, 21.23, 161.2, 42.66e3 / G_STD, 2, [0.001],
     "M from the take-up force"),
    ("Pa", "Pascual et al. 2005", 2561, 138.0, 472.0, 45.5e3, 2, [0.999], "n not stated"),
    ("Si", "Sinaga 2008 (KPC)", 13100, 33.7, 176.5, 46.9e3, 4, [0.001],
     "provisional; idler masses from catalogue"),
    ("WR", "Wheatley and Rubel 2021", 274.6, 31.09, 266.2, 7550, 2, [0.05, 0.5, 0.999],
     "take-up position unknown (drawn as a dotted line)"),
]

# Case whose take-up position is unknown (drawn as a line across xi).
UNKNOWN_POSITION = "WR"

# Intermediate drives (panel of T_1 / (4 t_B) against the drive offset l1 = sigma_d).
FT = 0.3048
NORDELL_CIOZDA = dict(tag="NC", L=8150 * FT, c_r=1450.0, gamma=1450.0 / 590.0,
                      sigma_d=5150 / 8150, sigma_t=5450 / 8150)
SASOL = dict(tag="Su", L=805.0, gamma=np.sqrt((35.6 + 3500 / 3.6 / 4.4 + 12.0) / (35.6 + 5.33)),
             sigma_d=152.0 / 805.0, sigma_t=152.0 / 805.0 + 0.01,
             note="design data; take-up position assumed next to the drives")


def points():
    """(tag, beta, gamma, xi, unknown_position) for every case and take-up position."""
    out = []
    for tag, name, L, mr, mc, M, n, xis, note in CASES:
        beta = 4 * M / (n * n * mr * L)
        gam = float(np.sqrt(mc / mr))
        for xi in xis:
            out.append((tag + ("t" if tag == "S" and xi > 0.5 else ""), beta, gam, xi, tag == UNKNOWN_POSITION))
    return out
