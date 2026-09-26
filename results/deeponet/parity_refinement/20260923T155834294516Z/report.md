# Parity under the established optimizer-refinement procedure

Both architectures trained identically (matched-update-count parents, uniform sampling, frozen dataset) and refined with the established two-arm procedure (small-step Adam vs full-batch L-BFGS, 180 s per arm). Selection used validation prices with the Greek guardrail. Fresh evaluation cases were opened after all selection decisions and are now observed. The final benchmark was evaluated once in an earlier study and is already observed; this work did not access it again.

| Architecture | Seed | Parent price RMSE | Adam price RMSE | L-BFGS price RMSE | Selected |
|---|---|---:|---:|---:|---|
| control | 22092026 | 0.144267 | 0.0931229 | 0.0919143 | lbfgs |
| control | 22092027 | 0.142833 | 0.077935 | 0.0784053 | adam_small_steps |
| control | 22092028 | 0.147265 | 0.10603 | 0.103962 | lbfgs |
| parity | 22092026 | 0.105819 | 0.0664712 | 0.0672145 | adam_small_steps |
| parity | 22092027 | 0.106484 | 0.0658941 | 0.0651153 | lbfgs |
| parity | 22092028 | 0.100346 | 0.067216 | 0.0663438 | lbfgs |

Validation RMSE after refinement (mean ± sample SD over seeds):

| Quantity | Refined control | Refined parity | Change |
|---|---:|---:|---:|
| price | 0.0912706 ± 0.01303 | 0.0659767 ± 0.0007488 | -27.7% |
| delta | 0.0106279 ± 0.001788 | 0.00838085 ± 0.000645 | -21.1% |
| gamma | 0.00272163 ± 0.0003639 | 0.00245613 ± 5.944e-05 | -9.8% |
| vega | 0.999758 ± 0.03471 | 0.825435 ± 0.04381 | -17.4% |
| theta | 1.35106 ± 0.216 | 1.11657 ± 0.07178 | -17.4% |
| rho | 1.14539 ± 0.1443 | 1.03017 ± 0.0173 | -10.1% |

Fresh in-domain RMSE after refinement (fresh cases, mean ± SD):

| Quantity | Refined control | Refined parity | Change |
|---|---:|---:|---:|
| price | 0.092167 ± 0.009487 | 0.0687556 ± 0.003228 | -25.4% |
| delta | 0.0106827 ± 0.001545 | 0.00868515 ± 0.0005277 | -18.7% |
| gamma | 0.0024892 ± 0.0002634 | 0.00220497 ± 9.87e-05 | -11.4% |
| vega | 0.942535 ± 0.06451 | 0.836199 ± 0.008126 | -11.3% |
| theta | 1.49771 ± 0.2743 | 1.2 ± 0.1103 | -19.9% |
| rho | 1.13213 ± 0.1139 | 1.01945 ± 0.04272 | -10.0% |

Reference: the established refinement on the family checkpoints improved validation price RMSE by 29% to 36% across the same seeds (Adam selected on two seeds, L-BFGS one).

## Verdict

- Refined parity improves 6 of 6 validation RMSEs versus refined control at the same procedure and seeds.
- Parity call-put consistency is exact by construction for the parity arms; control arms retain measured inconsistencies.
- Any regression above is listed in the Change columns; no unconditional claim is made.
