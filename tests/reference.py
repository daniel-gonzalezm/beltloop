"""Independent reference implementations used only by the tests.

* dense_scan_roots: root finding by sign changes on a fine grid (the phase-2 method);
* fe_frequencies: linear finite elements with consistent mass, take-up handled by the
  constraint w(xi+) = w(xi-) - 2y and an extra dof y with mass beta (from verify_phase2.py);
* fe_torque_frequencies: same with a torque-controlled drive of mass md.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq


def dense_scan_roots(f, Omax, n=40000, Omin=1e-6):
    g = np.linspace(Omin, Omax, n)
    v = np.array([f(o) for o in g])
    idx = np.nonzero(v[:-1] * v[1:] < 0)[0]
    return np.array([brentq(f, g[i], g[i + 1], xtol=1e-13) for i in idx])


def _assemble(loop, beta, ne_per_unit):
    xs, elems, x, first_after = [0.0], [], 0.0, False
    for j, seg in enumerate(loop.segments):
        if j == len(loop.upstream):
            first_after = True
        ne = max(2, int(round(ne_per_unit * seg.length))); h = seg.length / ne
        for i in range(ne):
            a = len(xs) - 1; xs.append(x + (i + 1) * h)
            elems.append((a, len(xs) - 1, h, seg.mu, first_after and i == 0))
        first_after = False; x += seg.length
    nn = len(xs); nd = nn + 1
    K = np.zeros((nd, nd)); M = np.zeros((nd, nd))
    for a, b, h, mu, plus in elems:
        T = np.zeros((2, nd)); T[0, a] = 1; T[1, b] = 1
        if plus:
            T[0, -1] = -2.0
        K += T.T @ (np.array([[1, -1], [-1, 1]]) / h) @ T
        M += T.T @ (mu * h / 6 * np.array([[2, 1], [1, 2]])) @ T
    M[-1, -1] += beta
    return K, M, nn


def fe_frequencies(loop, beta, ne_per_unit=400):
    K, M, nn = _assemble(loop, beta, ne_per_unit)
    free = [i for i in range(nn + 1) if i not in (0, nn - 1)]
    return np.sqrt(np.abs(eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True)))


def fe_torque_frequencies(loop, beta, md, ne_per_unit=300):
    K, M, nn = _assemble(loop, beta, ne_per_unit)
    nd = nn + 1
    Tc = np.eye(nd)[:, [i for i in range(nd) if i != nn - 1]]
    Tc[nn - 1, 0] = 1.0                       # w(2) = w(0): the drive closes the loop
    K2, M2 = Tc.T @ K @ Tc, Tc.T @ M @ Tc
    M2[0, 0] += md
    return np.sqrt(np.abs(eigh(K2, M2, eigvals_only=True)))
