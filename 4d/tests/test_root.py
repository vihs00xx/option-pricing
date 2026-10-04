import unittest

import mpmath
import numpy as np
from numpy.testing import assert_allclose

from src.pricing import Basket, factorization, gauss_hermite_values
from src.pricing.conditional import _root

CORRELATION_STAR = np.array([[1, -0.5, 0.3, 0.6], [-0.5, 1, -0.4, -0.3],
                             [0.3, -0.4, 1, 0.2], [0.6, -0.3, 0.2, 1]])


def c_star(**kwargs):
    values = dict(spots=(100, 95, 105, 100), strike=100, tau=0.001, rate=0.03,
                  volatilities=(0.1, 0.6, 0.3, 0.45), dividends=(0, 0.02, 0.04, 0.01),
                  weights=(0.4, 0.3, 0.2, 0.1), correlation=CORRELATION_STAR)
    values.update(kwargs)
    return Basket(**values)


def near_singular(matrix, gap=1e-10):
    eigenvalues, vectors = np.linalg.eigh(matrix)
    v = vectors[:, 0]
    moved = matrix - (eigenvalues[0] - gap) * np.outer(v, v)
    scale = 1 / np.sqrt(np.diag(moved))
    moved = moved * np.outer(scale, scale)
    moved = (moved + moved.T) / 2
    np.fill_diagonal(moved, 1.0)  # rescaling leaves the diagonal a few ulps off
    return moved


class RootTests(unittest.TestCase):
    def test_short_maturity_converges_to_rounding_level(self):
        c = c_star()
        n = 15  # price, Deltas and Gammas of four stocks
        coarse, fine = (gauss_hermite_values(c, nodes)[:n] for nodes in (16, 32))
        assert_allclose(fine, coarse, rtol=1e-12, atol=1e-14)

        # Rebuild the 32-node root problem exactly as the arithmetic estimator does.
        nodes, _ = np.polynomial.hermite_e.hermegauss(32)
        U = np.column_stack([g.ravel() for g in np.meshgrid(*([nodes] * 3), indexing="ij")])
        b, A = factorization(c)
        log_c = np.log(c.weights) + c.log_drift() + U @ A.T + np.log(c.spots)
        y = _root(log_c, b, np.log(c.strike))
        # np.longdouble is float64 on Windows, so the residual is recomputed in mpmath.
        with mpmath.workdps(40):
            log_strike = mpmath.log(mpmath.mpf(c.strike))
            residual = max(abs(mpmath.log(mpmath.fsum(mpmath.exp(mpmath.mpf(lc) + mpmath.mpf(bi) * mpmath.mpf(yr))
                                                      for lc, bi in zip(row, b))) - log_strike)
                           for row, yr in zip(log_c.tolist(), y.tolist()))
        self.assertLessEqual(float(residual), 4 * np.spacing(np.log(100.0)))

    def test_near_singular_correlation_long_maturity(self):
        c = c_star(tau=3, correlation=near_singular(CORRELATION_STAR))
        values = gauss_hermite_values(c, 32, greeks=False)
        self.assertTrue(np.all(np.isfinite(values)))


if __name__ == "__main__":
    unittest.main()
