import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.comparison.shared_evaluation import (
    ROOT, CORE, Predictor, arrays, cn_core, digest, financial_keys, make_cases, verify_pins,
)
from src.crank_nicolson import crank_nicolson_values
from src.neural.deeponet import ParityDeepONet
from src.neural.fixed_pinn import FixedBlackScholesPINN, FixedBlackScholesProblem


class SharedEvaluationTests(unittest.TestCase):
    def test_new_design_and_domain_separation(self):
        cases=make_cases()
        self.assertEqual(cases,make_cases())
        self.assertEqual(len(cases),512)
        self.assertEqual(len(financial_keys(*arrays(cases))),512)
        fixed=[c for c in cases if c['panel']=='fixed']
        varying=[c for c in cases if c['panel']=='varying']
        self.assertEqual(len(fixed),256)
        self.assertTrue(all(c['option']=='call' and c['K']==100 and c['r']==.05 and c['q']==0
                            and c['sigma']==.2 and 0<c['tau']<=1 for c in fixed))
        self.assertEqual(sum(c['option']=='put' for c in varying),128)
        self.assertTrue(all(0<c['tau']<=c['T'] for c in cases))
        for pair in {c['pair_id'] for c in varying}:
            a,b=[c for c in varying if c['pair_id']==pair]
            for k in ('K','T','S','tau','r','q','sigma'):
                self.assertEqual(a[k],b[k])

    def test_changed_case_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cases.json'
            p.write_text('[]')
            manifest=dict(cases_sha256=digest(p),source_sha256={},models={})
            verify_pins(manifest,Path(tmp))
            p.write_text('[1]')
            with self.assertRaisesRegex(RuntimeError,'cases changed'):
                verify_pins(manifest,Path(tmp))

    def test_four_output_cn_equals_existing_full_solver(self):
        settings=json.loads((ROOT/'experiments/config/classical/common_benchmark_settings.json').read_text())
        settings['crank_nicolson']['base_grid_size']=32
        case=make_cases()[0]
        actual=cn_core(case,settings)
        expected=crank_nicolson_values(**{k:case[k] for k in ('S','K','tau','r','sigma','q','option')},
                                      n_space=32,n_time=32,richardson=True,mesh='strike_clustered',concentration=7)
        for key in CORE:
            self.assertAlmostEqual(actual[key],expected[key],places=12)

    def test_selective_deeponet_derivatives_equal_native(self):
        predictor=Predictor('fixture',{'models':{}})
        predictor.model=ParityDeepONet(width=8,depth=1,rank=8)
        cases=make_cases()[:4]
        actual,_=predictor(cases)
        b,t=arrays(cases)
        expected=predictor.model.price_and_greeks(torch.tensor(b),torch.tensor(t))
        for key in CORE:
            np.testing.assert_allclose(actual[key],expected[key].numpy(),rtol=1e-12,atol=1e-12)

    def test_pinn_rejects_unsupported_contract(self):
        predictor=Predictor('fixture',{'models':{}})
        predictor.method='anchored_pinn'
        predictor.model=FixedBlackScholesPINN(FixedBlackScholesProblem(100,1,.05,.2,0,.05,4)).double()
        for change in (dict(option='put'),dict(sigma=.3),dict(K=101),dict(tau=1.1)):
            with self.assertRaisesRegex(ValueError,'fixed-contract scope'):
                predictor([{**make_cases()[0],**change}])


if __name__=='__main__':unittest.main()
