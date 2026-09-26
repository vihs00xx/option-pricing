import unittest

import numpy as np

from experiments.comparison.summarize_shared import metric, violations


class SharedSummaryTests(unittest.TestCase):
    def test_errors_and_relative_floor(self):
        m=metric(np.array([.1,3.,5.]),np.array([0.,2.,4.]),.01)
        self.assertAlmostEqual(m['rmse'],np.sqrt(2.01/3))
        self.assertEqual(m['relative_error_cases'],2)
        self.assertAlmostEqual(m['relative_rmse'],np.sqrt((.5**2+.25**2)/2))

    def test_call_put_bounds_and_tolerance(self):
        cases=[dict(S=100,K=100,tau=1,r=0,q=0,option=o) for o in ('call','put')]
        values=dict(price=np.array([101.,-2.]),delta=np.array([1.1,-1.2]),gamma=np.array([-1e-6,-.01]))
        result=violations(cases,values,1e-5)
        self.assertEqual(result['price_bounds']['count'],2)
        self.assertEqual(result['price_bounds']['maximum'],2)
        self.assertEqual(result['delta_bounds']['count'],2)
        self.assertEqual(result['negative_gamma']['count'],1)


if __name__=='__main__':unittest.main()
