"""Song, Wang and Zuo (2012), J. China Coal Soc. 37(S1), 217-223: cross-comparison numbers
for the validation section (step 6.4).

Their model is the limit beta -> 0 of ours with the take-up moved to the tail: two
independent fixed-free strands (tests/test_song.py). Their example (section 3.1):
L = 7117 m, horizontal, 2.5 m/s, 600 t/h; ST belt 1 m wide, 27 kg/m, 104 000 N/mm;
resistance f = a + b v on both strands with a = 0.016 and b = 0.0026 (per m/s), friction
0.3; tensioning weight 4500 kg; equivalent idler masses 11.25 (carry) and 10.8 kg/m
(return). Published results used here: wave speeds 995.6 (carry) and 1668.673 m/s
(return) (section 3.3(2)); running drive force from the resistances 2.2682e5 N (end of
section 3.1); running tensions at the drive 3.37e5 N (entry) and 1.13e5 N (exit) and
steady take-up travel 2.81 m (section 3.1, Figs. 4-6); tension at the tail take-up
1.73e5 N (section 3.3(1)).

Run:  python validation/song_case.py   (prints the summary)
"""
import numpy as np

G = 9.81
L, V, EA = 7117.0, 2.5, 104e6
M_BELT, M_LOAD = 27.0, 600 / 3.6 / 2.5
M_IC, M_IR = 11.25, 10.8
A_RES, B_RES = 0.016, 0.0026
M_TAKEUP = 4500.0
PUB = dict(c_c=995.6, c_r=1668.673, F_run=2.2682e5, T_entry=3.37e5, T_exit=1.13e5,
           travel=2.81, T_t=1.73e5)


def summary():
    mu_c, mu_r = M_BELT + M_LOAD + M_IC, M_BELT + M_IR
    f = (A_RES + B_RES * V) * G                       # resistance per unit weight, N/(kg/m) per m
    f_c, f_r = f * mu_c, f * mu_r                     # N/m on each strand
    F_run = (f_c + f_r) * L
    # Take-up (pulley) travel from rest to running with the take-up at the tail and constant
    # tension there: half the change of loop elongation, from their own running tensions
    # (linear tension along each strand between the tail and the drive faces).
    T_t = PUB["T_t"]
    dl_carry = 0.5 * (PUB["T_entry"] - T_t) * L / EA
    dl_return = 0.5 * (PUB["T_exit"] - T_t) * L / EA
    travel_pub_tensions = 0.5 * (dl_carry + dl_return)
    travel_model = 0.25 * (f_c - f_r) * L ** 2 / EA
    return dict(c_c=np.sqrt(EA / mu_c), c_r=np.sqrt(EA / mu_r), F_run=F_run,
                travel_from_their_tensions=travel_pub_tensions, travel_model=travel_model,
                takeup_mass_tension=M_TAKEUP * G / 2, pub=PUB)


if __name__ == "__main__":
    for k, v in summary().items():
        print(k, v)
