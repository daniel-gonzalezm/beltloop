"""Transfer matrices and characteristic functions.

State vector z = (W, W'), where W' is the dimensionless dynamic tension of a mode.
A homogeneous segment of length l and wavenumber kappa maps z by S(l, kappa); the
take-up pulley maps it by J (tension continuous, displacement jump -4 W'/(beta Om^2)).

The dimensionless model is written for a single loop (2:1 kinematics). A take-up whose
carriage is carried by n belt strands maps onto it exactly with the belt-side mass ratio
beta = 4 M / (n^2 mu_r L) and the belt-side travel (n/2) y (see conveyor.GravityTakeUp).
"""
from __future__ import annotations

import numpy as np

from .loop import Loop, Segment


def S(l: float, kappa) -> np.ndarray:
    """Segment transfer matrix; entire in kappa (finite at kappa = 0); kappa may be complex."""
    c, s = np.cos(kappa * l), np.sin(kappa * l)
    return np.array([[c, l * np.sinc(kappa * l / np.pi)], [-kappa * s, c]])


def J(beta: float, Om) -> np.ndarray:
    """Take-up transfer matrix (Om != 0, beta > 0)."""
    return np.array([[1.0, -4.0 / (beta * Om ** 2)], [0.0, 1.0]])


def chain(segments, Om: float) -> np.ndarray:
    """Product of segment matrices, first segment acting first."""
    P = np.eye(2)
    for seg in segments:
        P = S(seg.length, seg.g * Om) @ P
    return P


def AB(loop: Loop, Om: float) -> tuple[np.ndarray, np.ndarray]:
    """Upstream (drive exit -> take-up) and downstream (take-up -> drive entry) products."""
    return chain(loop.upstream, Om), chain(loop.downstream, Om)


def char_fun(loop: Loop, Om: float, beta: float) -> float:
    """Pole-free characteristic function D = beta Om^2 (BA)_12 - 4 A_22 B_11
    (prescribed drive velocity). D(0) = -4."""
    A, B = AB(loop, Om)
    return beta * Om ** 2 * (B @ A)[0, 1] - 4.0 * A[1, 1] * B[0, 0]


def char_fun_torque(loop: Loop, Om, beta: float, md: float, cd: float = 0.0):
    """Drive that is not velocity-controlled (limit of validity): drive mass
    md = M_d/(mu_r L) and, optionally, a dashpot cd = c_d/(mu_r c_r) to the reference
    speed (linearised motor slip characteristic). Characteristic function
        beta * Om^2 * [2 - tr P + (md Om^2 - i cd Om) P_12],  P = B J A,
    written pole-free. Time dependence exp(i Om tau): damped roots have Im(Om) > 0.
    cd = 0: torque-controlled drive, real roots, rigid-body root Om = 0 (double).
    md -> infinity or cd -> infinity: prescribed velocity. Accepts complex Om."""
    A, B = AB(loop, Om)
    BA = B @ A
    trP_part = beta * Om ** 2 * (2.0 - np.trace(BA)) + 4.0 * (A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0])
    P12_part = beta * Om ** 2 * BA[0, 1] - 4.0 * A[1, 1] * B[0, 0]
    dyn = md * Om ** 2 - 1j * cd * Om if cd else md * Om ** 2
    return trP_part + dyn * P12_part


def prufer_angle(segments, Om: float) -> float:
    """Scaled Pruefer angle phi = atan2(kappa W, W'), continuous, at the end of a chain
    started from z = (0, 1) (phi = 0). Inside a segment phi grows by kappa*l; at a density
    change phi is mapped within its quadrant, so the count of crossings of
    phi = (n - 1/2) pi (W' = 0) is the classical Sturm count (monotone in Om)."""
    phi, kprev = 0.0, None
    for seg in segments:
        k = seg.g * Om
        if kprev is not None and k != kprev:
            sn, cs = np.sin(phi), np.cos(phi)
            d = np.arctan2(k / kprev * sn, cs) - np.arctan2(sn, cs)
            phi += (d + np.pi) % (2 * np.pi) - np.pi
        phi += k * seg.length
        kprev = k
    return phi


def takeup_transmission(beta: float, Om) -> complex:
    """Complex transmission coefficient of a harmonic wave crossing the take-up pulley in
    an unbounded uniform belt (return-strand wavenumber Om), from the matrix J:
        t = 1 / (1 - 2i/(beta Om)),   |t|^2 + |1 - t|^2 = 1.
    |t| -> 0 for beta Om -> 0 (free end: the take-up reflects), |t| -> 1 for
    beta Om -> infinity (transparent pulley). In SI, beta Om = M omega/(mu_r c_r) = M omega/Z
    for a single loop, 4 M omega/(n^2 Z) with n strands."""
    return 1.0 / (1.0 - 2j / (beta * Om))
