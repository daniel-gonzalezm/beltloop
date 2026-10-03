"""Mode shapes, orthogonality, participation factors and quasi-static sums.
Ports check 7 of verify_phase2.py and adds interface and closed-form-integral checks."""
import numpy as np
import pytest

from beltloop.eigen import modal_basis
from beltloop.loop import Loop


def trap(f, x):
    return np.sum(0.5 * (f[..., 1:] + f[..., :-1]) * np.diff(x), axis=-1)


@pytest.fixture(scope="module")
def case():
    lp = Loop.from_positions(0.7, 0.75, 2.3, r_return=0.3, r_carry=0.8)
    beta = 0.15
    return lp, beta, modal_basis(lp, beta, 200)


def dense_grid(lp, n=20001):
    """Grid on each segment separately (W jumps at the take-up)."""
    xs, mus, rs = [], [], []
    for a, b, seg in zip(lp.x0[:-1], lp.x0[1:], lp.segments):
        k = max(50, int(n * (b - a) / 2))
        xs.append(np.linspace(a, b, k)); mus.append(np.full(k, seg.mu)); rs.append(np.full(k, seg.r))
    return xs, mus, rs


def integrate(basis, fun):
    xs, mus, rs = dense_grid(basis.loop)
    out = 0.0
    for i, (x, mu, r) in enumerate(zip(xs, mus, rs)):
        side = "left" if i == len(basis.loop.upstream) - 1 else "right"
        out = out + fun(basis.W(x, side=side), basis.dW(x), mu, r, x)
    return out


def test_interface_conditions(case):
    lp, beta, mb = case
    xi = lp.xi
    np.testing.assert_allclose(mb.W([0.0])[:, 0], 0, atol=1e-12)
    np.testing.assert_allclose(mb.W([2.0])[:, 0], 0, atol=1e-9)
    jump = mb.W([xi], side="right")[:, 0] - mb.W([xi], side="left")[:, 0]
    np.testing.assert_allclose(jump, -2 * mb.Y, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(beta * mb.Om ** 2 * mb.Y, 2 * mb.dW([xi])[:, 0], rtol=1e-9, atol=1e-12)
    for xb in lp.x0[1:-1]:                                   # head and tail: W, W' continuous
        if abs(xb - xi) > 1e-12:
            e = 1e-12          # dW changes by ~kappa^2 W e across the offset: compare 50 modes
            np.testing.assert_allclose(mb.W([xb - e])[:50], mb.W([xb + e])[:50], atol=1e-9)
            np.testing.assert_allclose(mb.dW([xb - e])[:50], mb.dW([xb + e])[:50], atol=1e-7)


def test_closed_form_integrals(case):
    lp, beta, mb = case
    sub = slice(0, 8)
    m = integrate(mb, lambda W, dW, mu, r, x: trap(mu * W[sub] ** 2, x)) + beta * mb.Y[sub] ** 2
    G = integrate(mb, lambda W, dW, mu, r, x: trap(mu * W[sub], x))
    R = integrate(mb, lambda W, dW, mu, r, x: trap(r * W[sub], x))
    K = integrate(mb, lambda W, dW, mu, r, x: trap(dW[sub] ** 2, x))
    np.testing.assert_allclose(m, 1.0, rtol=1e-6)
    np.testing.assert_allclose(G, mb.Gamma[sub], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(R, mb.R[sub], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(K, mb.Om[sub] ** 2, rtol=1e-6)
    np.testing.assert_allclose(mb.stiffness_check, 1.0, atol=1e-10)


def test_weighted_orthogonality(case):
    lp, beta, mb = case
    sub = slice(0, 6)
    Gw = integrate(mb, lambda W, dW, mu, r, x: trap(mu * W[sub, None, :] * W[None, sub, :], x)) \
        + beta * np.outer(mb.Y[sub], mb.Y[sub])
    Gu = integrate(mb, lambda W, dW, mu, r, x: trap(W[sub, None, :] * W[None, sub, :], x))
    off = lambda A: np.max(np.abs(A - np.diag(np.diag(A))))
    assert off(Gw) < 1e-6
    assert off(Gu) > 0.05                       # the thesis' unweighted product fails


def test_parseval(case):
    lp, beta, mb = case
    frac = np.sum(mb.effective_mass_fraction)
    assert 0.99 < frac <= 1.0 + 1e-9


def test_quasi_static_sums(case):
    lp, beta, mb = case
    x = np.linspace(0, 2, 801)
    inner = (x > 0.1) & (x < 1.9)
    qa = -(mb.Gamma / mb.Om ** 2) @ mb.dW(x)
    qr = -(mb.R / mb.Om ** 2) @ mb.dW(x)
    assert np.max(np.abs(qa - lp.quasi_static("mu", x))[inner]) < 2e-3
    assert np.max(np.abs(qr - lp.quasi_static("r", x))[inner]) < 2e-3
    # static take-up displacement: y = (1/2) int_0^2 Q dx
    assert abs(-(mb.Gamma / mb.Om ** 2) @ mb.Y / (0.5 * lp.quasi_static_integral("mu")) - 1) < 1e-6
    assert abs(-(mb.R / mb.Om ** 2) @ mb.Y / (0.5 * lp.quasi_static_integral("r")) - 1) < 1e-6


def test_quasi_static_closed_form_uniform():
    """gamma = 1, head drive: Q(x) = x - xi; int_0^2 Q = 2 - 2 xi."""
    lp = Loop.from_positions(0.0, 0.3, 1.0)
    x = np.linspace(0, 2, 11)
    np.testing.assert_allclose(lp.quasi_static("mu", x), x - 0.3, atol=1e-14)
    assert lp.quasi_static_integral("mu") == pytest.approx(2 - 0.6)
