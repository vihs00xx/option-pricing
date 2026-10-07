# Phase 0: one-stock Heston reference check

Development contracts only (seed 20261006): 4096 random domain contracts, 512 short-maturity contracts (tau 0.005 to 0.05) and 11 hand-picked hard ones. Everything is per unit strike (K = 1, S = moneyness); Vega and Volga are with respect to v0; Theta is calendar, -dV/dtau per year.

Reference: Lewis single integral, c = e^x - (e^{x/2}/pi) int_0^inf Re[e^{iux} phi(u - i/2)] / (u^2 + 1/4) du, with the little-trap phi written in a form that stays accurate as xi -> 0. The price and all 11 sensitivities come from closed-form integrands: spot and v0 derivatives directly, the kappa, theta, xi and rho derivatives by forward-mode differentiation of C and D, rate rho by the chain rule, Theta from the PDE. Adaptive 16-point Gauss-Legendre panels on [0, U] with a per-contract ladder: levels are tightened until two agree to 1e-9 on every quantity.

## Summary

| Check | Result | Max error |
|---|---|---:|
| Convergence ladder | 4619 contracts, all accepted at level 1; 0 not converged | last change 9.9e-14 |
| 1. Lewis vs Gil-Pelaez (two-probability form, scipy quad) | all 4619 contracts, calls and puts | 1.1e-15 |
| 2. Black-Scholes limit, rho = 0 | error falls like xi^2 (ratios 100.0, 100.0) | 4.6e-10 at xi = 1e-4 |
| 2. Black-Scholes limit, domain rho | error falls like xi (ratios 10.1, 10.0) | 2.4e-06 at xi = 1e-4 |
| 3. COS c2 L12 256 terms: price / Delta / Gamma | price within 1e-9 for 94.3% of contracts; worst feller_strong_skew_put | 3.2e-04 / 2.7e-04 / 4.9e-01 |
| 3. COS c2 L12 512 terms: price / Delta / Gamma | price within 1e-9 for 94.3% of contracts; worst feller_strong_skew_put | 3.3e-04 / 6.3e-04 / 2.6e-02 |
| 3. COS c4 L10 256 terms: price / Delta / Gamma | price within 1e-9 for 98.3% of contracts; worst feller_strong_skew_put | 1.7e-04 / 9.4e-03 / 1.8e+00 |
| 3. COS c4 L10 512 terms: price / Delta / Gamma | price within 1e-9 for 99.7% of contracts; worst feller_strong_skew_put | 9.8e-06 / 4.5e-03 / 5.0e-01 |
| 4. QuantLib 1.43 AnalyticHestonEngine | 200 contracts | 8.5e-15 |
| 5. QE Monte Carlo, 2^20 paths | 30 contracts; 90% inside 95%, 100% inside 99%; RMS z 1.04, mean z 0.11 | max abs z 2.39 |
| 6. Torch twin vs NumPy, all 12 quantities | ladder converged everywhere | 7.8e-14 |
| 6. Parameter derivatives vs Richardson FD | 331 contracts | 7.7e-11 |
| 6. Theta (PDE identity) vs FD in tau | same contracts | 6.1e-10 |
| 7. v_max, P(v_tau > v_max) <= 1e-8 / 1e-10 | worst corner v0 = theta = 0.36, kappa = 0.5, xi = 1, tau = 2 | 12.15 / 15.28 |
| 8. Timing (one thread, per contract) | see below | - |

No contract failed or was left unconverged.

## Findings

- **Reference.** Every contract is accepted at level 1 (level 0 and level 1 agree to 1e-9); the largest last change is 9.9e-14. Two independent formulas (Gil-Pelaez, QuantLib) agree to about 1e-14, so the reference is limited by double-precision rounding, not by quadrature. The integrand tail decides the work: U is 64 at the median but 11585 for the short-maturity contracts with xi = 1 and v0 = 0.01, where |phi(u - i/2)| decays only like exp(-sqrt(1 - rho^2) (v0 + kappa theta tau) u / xi).
- **Parity.** Puts are formed from calls by parity in closed form, so parity holds to rounding by construction. Check 1 prices calls and puts with an independent formula and a different integrator.
- **Black-Scholes limit.** With rho = 0 the gap to Black-Scholes at the average variance w falls by 100 per decade of xi; with rho != 0 by 10, as expected. At xi = 1e-4 the rho = 0 gap is 4.6e-10, so the xi-stable form of C and D has no cancellation problem.
- **COS.** With the plain cumulant rule (L = 12, exact c2) COS matches the reference to 1e-9 for most contracts but has a truncation floor of about 1e-4 in the price (and up to 0.5 in Gamma) for long maturities with large xi and Feller violation: the distribution of log S has a heavy left tail that 12 standard deviations do not cover, and more terms do not help. The Fang-Oosterlee rule with the fourth cumulant (L = 10, half-width sqrt(c2 + sqrt(c4))) covers the tail but then needs more terms. Scored over all 12 quantities on the domain and short contracts, the c2 rule is better at every setting from 32 to 256 terms (score 4.6e-2, 1.3e-2, 3.8e-3, 3.9e-4) and stops improving at 2.9e-4 from 512 terms; the c4 rules only win at 1024 terms. Phase 1 therefore freezes the c2 rule with L = 12 and reports its floor.
- **Monte Carlo.** QE with the martingale correction (psi_c = 1.5, max(32, 256 tau) steps) is unbiased within its error bars on all 30 contracts, including strong Feller violations; no martingale-correction failure occurred.
- **Torch twin.** The batched complex128 twin with autograd agrees with the closed-form NumPy integrands to 7.8e-14 on all 12 quantities, and both agree with Richardson differences to 6.1e-10. The node ladder (u = U s^2, 16-point Gauss-Legendre panels) needs 256 nodes at the domain median, 512 at the 99th percentile and 1024 at most; short maturities need up to 1024, and the hand-picked tau = 0.005, xi = 1 cases up to 4096.
- **v_max.** The worst corner is v0 = theta = 0.36, kappa = 0.5, xi = 1, tau = 2. P(v_tau > v) is 1e-8 at v = 12.1 and 1e-10 at v = 15.3, an order of magnitude above the plan's guess of 1.0 to 1.5. The ADI grid in Phase 1 uses the 1e-10 value with points packed near v = 0; the price error from truncating v is far smaller than the tail probability, but the brief asked for the tail criterion.
- **Feller.** 21.8% of the domain sample violates the Feller condition (2 kappa theta < xi^2); nothing in the reference depends on it.
- **Hardest regions.** Short maturity with large xi (long, slowly decaying integrand: more panels and torch nodes) and long maturity with large xi (heavy left tail: the COS truncation floor). The table below is by tau and xi bins.

## Timing (one thread)

| Method | Seconds per contract |
|---|---:|
| NumPy reference, all 12 quantities, including the two ladder levels | 0.0037 |
| NumPy reference, price only, two levels | 0.0013 |
| Torch twin, batch of 4096, fixed 512 nodes, all 12 by autograd | 0.00050 |
| Torch twin, batch of 4096, full node ladder | 0.00066 |
| COS (c2 rule) 256 terms, price/Delta/Gamma, batch of 4096 | 0.00018 |
| COS (c2 rule) 256 terms, price/Delta/Gamma, one contract | 0.00193 |
| COS (c2 rule) 512 terms, price/Delta/Gamma, batch of 4096 | 0.00037 |
| COS (c2 rule) 512 terms, price/Delta/Gamma, one contract | 0.00199 |

## Regions (tau by xi; domain and short contracts)

| tau | xi | n | median U | median / max torch nodes | COS c2-256 max price error | max Gamma error |
|---|---|---:|---:|---:|---:|---:|
| 0.005-0.05 | 0.1-0.4 | 196 | 167 | 512 / 1024 | 4.4e-16 | 1.6e-12 |
| 0.005-0.05 | 0.4-0.7 | 174 | 181 | 512 / 1024 | 6.7e-12 | 2.4e-08 |
| 0.005-0.05 | 0.7-1.0 | 142 | 256 | 512 / 1024 | 1.4e-10 | 4.6e-07 |
| 0.05-0.25 | 0.1-0.4 | 146 | 76 | 256 / 1024 | 1.2e-10 | 1.6e-07 |
| 0.05-0.25 | 0.4-0.7 | 132 | 140 | 384 / 1024 | 1.1e-08 | 2.1e-06 |
| 0.05-0.25 | 0.7-1.0 | 118 | 181 | 512 / 1024 | 2.9e-08 | 4.0e-06 |
| 0.25-1.0 | 0.1-0.4 | 511 | 45 | 256 / 1024 | 1.1e-09 | 1.6e-07 |
| 0.25-1.0 | 0.4-0.7 | 549 | 64 | 256 / 1024 | 2.8e-06 | 1.3e-05 |
| 0.25-1.0 | 0.7-1.0 | 527 | 108 | 256 / 1024 | 4.0e-06 | 2.3e-04 |
| 1.0-2.0 | 0.1-0.4 | 703 | 27 | 128 / 512 | 1.2e-09 | 1.4e-07 |
| 1.0-2.0 | 0.4-0.7 | 710 | 45 | 256 / 512 | 2.5e-06 | 2.9e-05 |
| 1.0-2.0 | 0.7-1.0 | 700 | 54 | 256 / 1024 | 1.1e-04 | 2.6e-02 |

## Black-Scholes limit

| rho | xi | max abs error | max rel error | ratio to previous |
|---|---:|---:|---:|---:|
| zero | 0.01 | 4.6e-06 | 3.0e-04 | nan |
| zero | 0.001 | 4.6e-08 | 3.0e-06 | 100.0 |
| zero | 0.0001 | 4.6e-10 | 3.0e-08 | 100.0 |
| domain | 0.01 | 2.4e-04 | 1.3e-02 | nan |
| domain | 0.001 | 2.4e-05 | 1.3e-03 | 10.1 |
| domain | 0.0001 | 2.4e-06 | 1.3e-04 | 10.0 |

## Torch twin and finite differences

| Quantity | torch - NumPy | NumPy - Richardson FD | torch - Richardson FD |
|---|---:|---:|---:|
| price | 1.2e-15 | - | - |
| delta | 8.6e-16 | - | - |
| gamma | 7.8e-14 | - | - |
| vega | 5.6e-16 | - | - |
| vanna | 5.8e-15 | - | - |
| volga | 1.4e-14 | - | - |
| theta_cal | 6.7e-16 | 6.1e-10 | 6.1e-10 |
| rho_rate | 1.6e-15 | 1.3e-10 | 1.3e-10 |
| d_kappa | 4.3e-17 | 1.0e-11 | 1.0e-11 |
| d_theta | 1.1e-15 | 7.7e-11 | 7.7e-11 |
| d_xi | 6.8e-17 | 1.0e-11 | 1.0e-11 |
| d_rho | 2.8e-17 | 6.7e-12 | 6.7e-12 |

Node counts needed (16-point panels doubled until two counts agree to 1e-9):

- domain: 128 nodes: 1131, 256 nodes: 2426, 512 nodes: 506, 1024 nodes: 33
- short: 256 nodes: 121, 512 nodes: 350, 1024 nodes: 41
- hard: 256 nodes: 3, 512 nodes: 2, 1024 nodes: 2, 2048 nodes: 2, 4096 nodes: 2

## Monte Carlo rows

| Contract | tau | steps | MC | SE | reference | z |
|---|---:|---:|---:|---:|---:|---:|
| domain0232 | 1.892 | 485 | 0.145478 | 4.2e-04 | 0.145285 | 0.46 |
| domain2829 | 1.531 | 392 | 0.126773 | 3.9e-04 | 0.127279 | -1.29 |
| domain2203 | 0.289 | 74 | 0.027507 | 7.0e-05 | 0.027425 | 1.18 |
| domain3297 | 1.733 | 444 | 0.127280 | 3.2e-04 | 0.127106 | 0.54 |
| domain3343 | 0.323 | 83 | 0.087958 | 1.6e-04 | 0.087964 | -0.04 |
| domain3830 | 0.140 | 36 | 0.010041 | 3.8e-05 | 0.010004 | 0.95 |
| domain1307 | 0.815 | 209 | 0.129471 | 1.8e-04 | 0.129113 | 2.04 |
| domain1103 | 0.569 | 146 | 0.199190 | 2.9e-04 | 0.199475 | -0.99 |
| domain1732 | 1.123 | 288 | 0.140465 | 1.9e-04 | 0.140416 | 0.25 |
| domain1575 | 0.368 | 95 | 0.080144 | 1.4e-04 | 0.079931 | 1.49 |
| domain2746 | 0.904 | 232 | 0.080170 | 1.3e-04 | 0.080258 | -0.70 |
| domain2751 | 0.240 | 62 | 0.159194 | 1.3e-04 | 0.158931 | 1.96 |
| domain1336 | 0.812 | 208 | 0.133156 | 2.3e-04 | 0.133184 | -0.12 |
| domain0515 | 1.344 | 344 | 0.308641 | 1.6e-04 | 0.308695 | -0.35 |
| domain2789 | 1.360 | 349 | 0.095079 | 1.6e-04 | 0.095460 | -2.39 |
| domain0094 | 0.481 | 124 | 0.226210 | 2.7e-04 | 0.226291 | -0.30 |
| domain1889 | 0.277 | 72 | 0.018447 | 4.6e-05 | 0.018436 | 0.23 |
| domain1802 | 0.964 | 247 | 0.258792 | 2.7e-04 | 0.258638 | 0.56 |
| domain2637 | 0.816 | 209 | 0.297572 | 1.9e-04 | 0.297477 | 0.50 |
| domain1833 | 1.938 | 497 | 0.152849 | 2.2e-04 | 0.152522 | 1.51 |
| domain2363 | 0.680 | 174 | 0.111642 | 1.6e-04 | 0.111671 | -0.19 |
| domain2961 | 0.754 | 194 | 0.098278 | 1.6e-04 | 0.098281 | -0.02 |
| domain2654 | 0.516 | 133 | 0.220173 | 3.4e-04 | 0.219823 | 1.02 |
| domain1997 | 0.861 | 221 | 0.297473 | 2.3e-04 | 0.297537 | -0.27 |
| feller_strong_atm | 2.000 | 512 | 0.071055 | 1.4e-04 | 0.071066 | -0.08 |
| feller_strong_skew_put | 1.000 | 256 | 0.128842 | 6.0e-05 | 0.128982 | -2.32 |
| rho_m09_long_itm_call | 2.000 | 512 | 0.535268 | 5.5e-04 | 0.535458 | -0.34 |
| v0_high_theta_low_fast | 0.500 | 128 | 0.109790 | 1.6e-04 | 0.109732 | 0.36 |
| v0_high_theta_low_slow_otm | 2.000 | 512 | 0.019900 | 5.4e-05 | 0.019881 | 0.35 |
| short_atm_max_gamma | 0.005 | 32 | 0.002769 | 3.3e-06 | 0.002771 | -0.62 |

## Files

- `checks.json`: every number above.
- `summary_by_contract.csv`: inputs, the 12 reference quantities, ladder level, U, torch nodes and the check-1 and COS errors per contract.
- `run.log`: the run log.

Regenerate: `python -m experiments.reference.check_reference` then `python -m experiments.reference.make_report` (from heston/).
