# Neural pricers for one-stock Heston options: PINN and DeepONet

Two neural pricers for European calls and puts under the Heston model, with the four Heston parameters as inputs so that one network covers the whole parameter range. Each is built as a ladder in which every level adds one technique and is scored on the same validation panels. The pre-declared rule selected **`P10_anchored`** for the PINN and **`D9_ensemble`** for the DeepONet; both were frozen by hash (2026-10-07 11:06 UTC) before being scored once on the sealed test set `heston_test_v1`, then compared with COS, ADI, QE Monte Carlo and QE-RQMC.

## Results at a glance

Test RMSEs on all 252 cases, price and 11 sensitivities per unit strike; microseconds per contract for the price and all 11 sensitivities, one thread (networks in a batch of 4,096; classical methods per contract, quiet machine). Training seconds include the warm-start chain and the label building, scaled to an idle machine (see Break-even).

| Pricer | Selection score | Test score | Price RMSE | Delta RMSE | Gamma RMSE | Vega RMSE | Theta RMSE | us per contract | Training + labels s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PINN `P10_anchored` | 0.122 | 0.144 | 0.00142 | 0.0191 | 0.19 | 0.0112 | 0.00271 | 12.7 | 3,427 |
| DeepONet `D9_ensemble` | 0.0602 | 0.0733 | 0.000646 | 0.0107 | 0.127 | 0.00355 | 0.00147 | 245 | 21,281 |
| Black-Scholes base (no training) |  | 0.344 | 0.0091 | 0.043 | 0.304 | 0.0382 | 0.0107 | 1.77 | 0 |
| `cos_32` |  | 0.081 | 0.000412 | 0.0127 | 0.154 | 0.0106 | 0.000428 | 3,043 (87.4 batched) | 0 |
| `cos_64` |  | 0.0399 | 7.95e-05 | 0.00353 | 0.178 | 0.00209 | 6.84e-05 | 3,106 (157 batched) | 0 |
| `cos_128` |  | 0.0123 | 2.18e-05 | 0.000836 | 0.0409 | 0.000937 | 2.93e-05 | 3,461 (417 batched) | 0 |
| `cos_256` |  | 0.0019 | 7.9e-06 | 9.27e-05 | 0.00949 | 0.000401 | 3.64e-05 | 3,708 (970 batched) | 0 |
| `adi_80x40x40` |  | 0.0226 | 0.000111 | 0.00207 | 0.0606 | 0.00181 | 0.000358 | 368,336 | 0 |
| `adi_160x80x80` |  | 0.00731 | 2.77e-05 | 0.000607 | 0.0182 | 0.000564 | 0.000107 | 1,831,776 | 0 |
| `mc_100000_dt32` |  | 0.185 | 0.000992 | 0.00194 | 0.0318 | 0.00232 | 0.00151 | 5,300,746 | 0 |
| `mc_1000000_dt32` |  | 0.11 | 0.000286 | 0.00063 | 0.0229 | 0.00127 | 0.000506 | 57,961,262 | 0 |
| `rqmc_2^12x16_dt32` |  | 0.337 | 0.00019 | 0.00068 | 0.0328 | 0.00399 | 0.000348 | 2,952,713 | 0 |
| `rqmc_2^16x16_dt32` |  | 0.0736 | 3.77e-05 | 0.000188 | 0.022 | 0.000573 | 0.000131 | 63,722,278 | 0 |

## Findings

- **COS beats both networks, by a wide margin.** On the 252 test cases the selected DeepONet
  (`D9_ensemble`) scores 0.073 and the selected PINN (`P10_anchored`) 0.144, against 0.0019 for
  COS with 256 terms (price RMSE 7.9e-6 against 6.5e-4 and 1.4e-3). COS needs 3.7 ms per contract
  one at a time and 0.97 ms in a batch, with all 12 quantities. The PINN is faster in a batch
  (13 microseconds per contract), but against batched COS its training pays back only after about
  24 to 46 million contracts. The DeepONet (245 microseconds) is slower than batched COS with
  32 terms, which already matches its price error. Against COS with 128 terms, which matches it
  on all 12 quantities, it pays back only after about 120 million contracts. One at a time, COS
  is faster than both networks. For one stock under Heston, COS is the pricer to use.
- **The networks do beat the simulation methods on the Greeks.** No Monte Carlo or RQMC setting run
  (up to 10^6 paths, or 16 x 2^16 Sobol points) matches either network on all 12 quantities at once:
  Volga, a bumped second difference, stays too noisy.
  On the price alone the DeepONet's error matches QE Monte Carlo with 10^6 paths (dt 1/8, 13 s per
  contract) and RQMC with 16 x 2^10 points (1.1 s). Its training pays back after 1,700 and 20,000
  contracts respectively. Against ADI it pays back after 58,000 contracts when matched on all 12
  quantities (ADI 80 x 40 x 40, 0.37 s per contract).
- **The DeepONet beats the PINN at every stage**, as in 2D: test 0.073 against 0.144, and 0.088
  against 0.175 for the best levels without anchors or ensembling (`D8`, `P9`). The anchored PINN
  `P10` is the hybrid that uses 4,096 reference prices; `P9` uses no labels at all.
- **The structure of the output decided the result, not the optimizer.** The first-generation
  outputs (Black-Scholes at the expected average variance plus tau N) plateaued near validation
  0.11 and could not beat the Black-Scholes base on short maturities. An implied-volatility output,
  c = BS(x, sqrt(w tau) e^N), with kappa and theta entering only through the variance drift, cut
  the DeepONet's selection score from 0.20 (`D2`, the best earlier level) to 0.095 for one network
  and 0.060 for the ensemble. The PINN went from 0.18 (the anchored level with the old output) to
  0.12. The diagnosis and the variants are in the next section. Among the optimizer and size
  levels, L-BFGS improved validation by 21% (PINN) and 33% (DeepONet, though the DeepONet's
  short-maturity score got worse). Newton-CG added 7%. Doubling the width helped the PINN (0.199
  against 0.245 for its base, although its Adam stage hit the wall-clock cap at step 19,200 of
  25,000). For the DeepONet it improved validation but not short maturities.
- **Extending the variance domain to v = 3 for the v = 0 boundary (`P4`) cost accuracy.** It
  raised the validation score from 0.188 to 0.241, and packing 85% of the collocation at
  v <= 0.6 did not recover it. Every later PINN level inherits that cost.
- **Short maturities and extrapolation hold up after the change.** By split, the DeepONet test
  scores are 0.063 interpolation, 0.087 short maturity (tau 0.005 to 0.05, outside the training
  domain), 0.073 extrapolation and 0.144 anchors. The anchors include the hardest cases: strong
  Feller violation, rho = -0.9 with xi = 1, and v0 far from theta.
- **Calibration: faster, but not accurate enough to identify the parameters.** Fitting
  (v0, kappa, theta, xi, rho) to 100 reference prices from 8 random starts, COS (128 terms)
  recovers every parameter to 3e-6 relative in 1.7 s per calibration. The DeepONet takes 0.32 s
  and the PINN 0.25 s, but neither gets within 10% from any start. Median worst relative parameter
  error: 28% for the DeepONet (for example kappa 1.92 against 1.5), about 200% for the PINN. Their
  price errors (1e-4 to 1e-3) are larger than the price differences that separate the parameter
  sets along the surface's flat directions, notably kappa against theta. So there is no
  calibration break-even at this accuracy. The networks would need roughly COS's price accuracy
  first.
- **No-arbitrage holds** for both selected networks on all 252 test cases and both 4,096-contract
  validation panels: no price below intrinsic, no Delta out of bounds, no negative Gamma or Vega.
  The implied-volatility output is the reason. Every level built on it (`P9`, `P10`, `D8`, `D9`) has
  no violation, because a Black-Scholes price with a positive volatility cannot leave these bounds.
  Every level before it has some: `P0` has 26 prices below intrinsic, and `D2` to `D7` have 6 to 12.
- **Against Monte Carlo intervals** (QE, dt 1/32), the reference lies inside the 95% interval for
  90 to 96% of cases per quantity at 10^5 paths. The DeepONet lies inside for 49 to 86% and the
  PINN for 40 to 80%. At 10^6 paths: DeepONet 29 to 72%, PINN 17 to 57%. Monte Carlo at those
  path counts can detect the networks' Greek errors, which the exact-reference RMSEs quantify.
- **Training cost** on an idle machine: about 1 hour for the PINN (P9 then P10) and 5.9 hours for
  the DeepONet ensemble (four D8 trainings; one member alone is about 1.5 hours). These are
  wall-clock seconds measured with 3 to 4 levels sharing the CPU, multiplied by a load factor of
  0.34 (PINN) and 0.35 (DeepONet). The factor comes from re-running a pilot of one level alone.
  Labelling 32,768 contracts with the torch twin takes about 15 s.

## Problem and model

European calls and puts on one stock under

    dS = (r - q) S dt + sqrt(v) S dW1,   dv = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1, W2> = rho dt,

with constant r and q. A contract has nine inputs (S/K, tau, r, q, v0, kappa, theta, xi, rho) and a call/put flag; the pricers return the price and 11 sensitivities per unit strike: Delta, Gamma, Vega and Volga with respect to v0, Vanna (d2V/dS dv0), Theta (calendar, -dV/dtau per year), rate rho, and the derivatives with respect to kappa, theta, xi and rho.

| Input | Training domain |
|---|---|
| moneyness | 0.7 to 1.3 |
| tau | 0.05 to 2 |
| r | -0.02 to 0.1 |
| q | 0 to 0.06 |
| v0 | 0.01 to 0.36 |
| theta | 0.01 to 0.36 |
| kappa | 0.5 to 5 |
| xi | 0.1 to 1 |
| rho | -0.9 to 0.3 |

The Feller condition 2 kappa theta >= xi^2 is not imposed: it fails for 21.8% of a uniform sample of the domain, and for 91 of the 252 test cases. The variance process has a heavy upper tail at the domain's worst corner (v0 = theta = 0.36, kappa = 0.5, xi = 1, tau = 2): P(v_tau > v) is 1e-8 at v = 12.1 and 1e-10 at v = 15.3, ten times the plan's first guess of 1 to 1.5. The ADI grid runs to v_max = 15.28; the networks stop at v_hi = 3 (next section).

**Exact reduction.** With F = S e^{(r-q) tau} and x = log(F/K), V = K e^{-r tau} c(x, v, tau; kappa, theta, xi, rho), and the normalized call solves the driftless PDE

    c_tau = 1/2 v (c_xx - c_x) + rho xi v c_xv + 1/2 xi^2 v c_vv + kappa (theta - v) c_v,   c(tau = 0) = (e^x - 1)^+,

which at v = 0 degenerates to c_tau = kappa theta c_v. Puts follow from parity. Every sensitivity of a network comes from automatic differentiation of the price through this map (`price_and_greeks` in `src/neural/reference.py`).

**Base.** Black-Scholes at the expected average variance w = theta + (v - theta)(1 - e^{-kappa tau})/(kappa tau), exact as xi -> 0, is the base the networks correct (output `bs`: c = BS + tau N; `bs_envelope`: c = BS + tau phi(d2) N).

**Computational box.** x in [-3, 2.5] (the domain maps into [-0.52, 0.46]); tau from 0; v in [0, 1] for the levels before the variance boundary and [0, 3] from `P4`/`D4` on. The far-v condition c_v = 0 at v_hi = 3 was chosen on development contracts with the ADI solver at a fixed fine grid: against the full domain it moves prices at v0 <= 0.49 by at most 4e-07 (Volga 1e-04); at v_hi = 2 both c_v = 0 and the Black-Scholes base value move prices by about 4e-05, and at v_hi = 1 by about 4e-3 (`results/neural/variance_cap_study`). Collocation is packed near v = 0 (v = 3 u^2) and the networks see sqrt(v + 0.01): plain sqrt(v) would make c_v infinite at v = 0, where the degenerate equation needs it finite.

## Data, seeds and selection

| Set | Contracts | Seed | Use |
|---|---:|---:|---|
| DeepONet labels | 32,768 (131,072 for the scale-up; 25% short maturity from D5) | 8001 | training |
| PINN anchors | 4,096 | 8002 | `P10_anchored` only |
| Monitor | 512 | 8201 | checkpoint choice within a run |
| Validation | 4,096 | 8301 | selection |
| Short-maturity validation | 4,096 (tau 0.005-0.05) | 8302 | selection |
| Test set `heston_test_v1` | 252 | 20261007 | final assessment, once |

Labels come from the batched complex128 torch twin of the Lewis reference (Phase 0: it agrees with the NumPy reference to 8e-14 on all 12 quantities), which gives the call and all its derivatives by autograd.

Within each family the level with the lowest selection score is selected. Selection score = mean of the validation score and the short-maturity validation score, each the mean over price and the 11 sensitivities of RMSE / RMS(reference) on 4,096 contracts (training domain; tau 0.005 to 0.05). A level within 1% of the best yields to the earlier level.

## How the ladders changed during the run

The ladders were changed mid-run after a diagnosis on the validation panels only; the test set stayed sealed throughout (its seal `956e8e0725874dff...` was checked again immediately before it was opened). The first PINN and DeepONet levels reached validation scores near 0.11 and short-maturity scores of 0.25 to 0.36, worse than the Black-Scholes base alone on the short panel (table below). The per-quantity breakdown showed why: price, Delta and Theta were fine, but Volga, Vanna and the xi and rho sensitivities stayed close to the base's own errors, and on the short panel the networks did not move them at all (Vanna 0.187 and Volga 0.311 for five different models). The learning curves were flat at the end, so the wall-clock cap was not the limit.

The cause is structural. The first outputs add a correction tau N (or tau phi(d_2) N) to Black-Scholes at the expected average variance, so the correction is O(tau) near expiry. Under Heston the departure from that base at short maturity is an implied-volatility skew and smile of order one, which is O(sqrt(tau) phi(d_2)) in price, and the same smile terms carry Volga, Vanna and the xi and rho sensitivities at every maturity. Two changes followed, each piloted on the validation panels: (1) an implied-volatility output c = BS(x, sqrt(w tau) e^N), whose first-order correction sqrt(w tau) phi(d_2) N fades like the time value; (2) kappa and theta entering only through the variance drift over the option's life, as inputs kappa tau and kappa theta tau for the PINN and as a tau-weighted second branch for the DeepONet, so that their sensitivities vanish at the true O(tau^1.5) rate instead of leaking at O(sqrt(tau)). They became the envelope levels `P9` and `D8` (and so `P10` and `D9`). The runs they replaced are kept as variants below. The variance-boundary level `P4` was also re-examined with 85% of the collocation at v <= 0.6: better than the original P4 but still worse than P3, so P4 was left as configured. The wall-clock caps were raised from 2 h / 4 h to 4 h / 6 h for the levels that ran after the change.

Per-quantity RMSE / RMS(reference) on the validation panels for the Black-Scholes base and the levels that led to the change:

| Pricer | Panel | price | delta | gamma | vega | vanna | volga | theta_cal | rho_rate | d_kappa | d_theta | d_xi | d_rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Black-Scholes base | validation | 0.0356 | 0.0465 | 0.156 | 0.089 | 0.26 | 0.619 | 0.076 | 0.0727 | 0.338 | 0.0502 | 1 | 1 |
| Black-Scholes base | short | 0.00227 | 0.00732 | 0.0665 | 0.0423 | 0.188 | 0.311 | 0.0442 | 0.00906 | 0.0636 | 0.0436 | 1 | 1 |
| P2_hard_payoff | validation | 0.00845 | 0.0255 | 0.125 | 0.0712 | 0.257 | 0.611 | 0.0329 | 0.0275 | 0.138 | 0.0475 | 0.378 | 0.392 |
| P2_hard_payoff | short | 0.00204 | 0.00734 | 0.0665 | 0.0413 | 0.187 | 0.311 | 0.037 | 0.00901 | 0.548 | 0.129 | 0.959 | 0.925 |
| P9 tau-envelope (variant) | validation | 0.00939 | 0.0202 | 0.102 | 0.0451 | 0.22 | 0.506 | 0.0315 | 0.0216 | 0.161 | 0.0385 | 0.455 | 0.429 |
| P9 tau-envelope (variant) | short | 0.00187 | 0.0063 | 0.0633 | 0.0386 | 0.191 | 0.328 | 0.0355 | 0.00765 | 0.697 | 0.124 | 1.04 | 0.889 |
| P10_anchored | validation | 0.00313 | 0.0107 | 0.0675 | 0.0324 | 0.176 | 0.426 | 0.0138 | 0.0123 | 0.0625 | 0.0325 | 0.163 | 0.211 |
| P10_anchored | short | 0.000841 | 0.00334 | 0.0508 | 0.032 | 0.175 | 0.302 | 0.0137 | 0.00377 | 0.0587 | 0.0441 | 0.601 | 0.441 |
| D3_differential | validation | 0.00368 | 0.0191 | 0.137 | 0.0389 | 0.243 | 0.506 | 0.0235 | 0.0207 | 0.0634 | 0.035 | 0.101 | 0.123 |
| D3_differential | short | 0.00262 | 0.00732 | 0.0664 | 0.0456 | 0.187 | 0.311 | 0.0413 | 0.0089 | 1.18 | 0.394 | 1.09 | 1.04 |
| D8_envelope | validation | 0.00474 | 0.00719 | 0.0564 | 0.0203 | 0.117 | 0.239 | 0.022 | 0.00765 | 0.0424 | 0.027 | 0.0612 | 0.0819 |
| D8_envelope | short | 0.000884 | 0.00299 | 0.0431 | 0.0298 | 0.152 | 0.225 | 0.0173 | 0.0034 | 0.177 | 0.222 | 0.403 | 0.318 |

Pilots behind the change (5,000 Adam steps each; `variants/output_and_vmix_pilots.txt`):

```
P3_time_balanced | bs |  | 1601s | valid 0.2650 (volga 0.615 vanna 0.260 dxi 0.644 drho 0.946 dkap 0.235) | short 0.3494 (volga 0.311 vanna 0.188 dxi 1.252 drho 1.151 dkap 0.946)
P3_time_balanced | bs_vol |  | 1673s | valid 0.1721 (volga 0.505 vanna 0.213 dxi 0.389 drho 0.483 dkap 0.185) | short 0.2789 (volga 0.295 vanna 0.180 dxi 1.002 drho 0.720 dkap 0.753)
P4_variance_boundary | bs |  | 1874s | valid 0.3353 (volga 0.651 vanna 0.259 dxi 1.009 drho 1.008 dkap 0.379) | short 0.5730 (volga 0.311 vanna 0.188 dxi 2.172 drho 1.096 dkap 1.781)
P4_variance_boundary | bs | vmix | 1879s | valid 0.3175 (volga 0.613 vanna 0.260 dxi 0.938 drho 1.121 dkap 0.223) | short 0.5056 (volga 0.318 vanna 0.188 dxi 1.632 drho 1.255 dkap 1.295)
D2_hard_payoff | bs | 5000 steps | valid 0.1621 (volga 0.614 vanna 0.259 dxi 0.307 drho 0.276 dkap 0.129) | short 0.2663 (volga 0.311 vanna 0.188 dxi 1.095 drho 0.915 dkap 0.356)
D2_hard_payoff | bs_vol | 5000 steps | valid 0.1095 (volga 0.460 vanna 0.200 dxi 0.155 drho 0.206 dkap 0.092) | short 0.4090 (volga 0.327 vanna 0.179 dxi 0.934 drho 0.676 dkap 1.994)
P3_time_balanced | bs_vol | drift | 1680s | valid 0.1862 (volga 0.535 vanna 0.225 dxi 0.412 drho 0.534 dkap 0.204 dthe 0.058) | short 0.2245 (volga 0.310 vanna 0.185 dxi 1.108 drho 0.836 dkap 0.068 dthe 0.044)
D5_short_maturity | bs_vol |  | 4260s | valid 0.0786 (volga 0.288 vanna 0.141 dxi 0.112 drho 0.140 dkap 0.069 dthe 0.037) | short 0.3568 (volga 0.247 vanna 0.163 dxi 0.707 drho 0.512 dkap 1.407 dthe 1.110)
D5_short_maturity | bs_vol | split | 6397s | valid 0.0912 (volga 0.334 vanna 0.156 dxi 0.118 drho 0.167 dkap 0.072 dthe 0.045) | short 0.1667 (volga 0.266 vanna 0.166 dxi 0.669 drho 0.466 dkap 0.145 dthe 0.157)
```

## PINN ladder

| Level | Technique added | Parameters | Validation | Short-maturity | Selection | Price RMSE | Gamma RMSE | Vega RMSE | Theta RMSE | Training s (wall) | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P0_vanilla` | Textbook PINN: MLP on the 9 raw contract inputs plus a put flag; Heston PDE residual in (S, v, tau) on the training box; soft payoff loss; Adam | 13,249 | 0.501 | 4.27 | 2.39 | 0.0336 | 0.545 | 0.0877 | 0.0425 | 2,878 |  |
| `P1_reduced` | Exact reduction to (x, v, tau) and (kappa, theta, xi, rho); driftless reduced PDE; put-call parity; far x boundary at the Black-Scholes base | 13,057 | 0.604 | 8.37 | 4.49 | 0.017 | 0.403 | 0.0643 | 0.0234 | 3,046 |  |
| `P2_hard_payoff` | Hard terminal condition: c = BS(x, w tau) + tau N(x, v, sqrt(tau), kappa, theta, xi, rho) with w the expected average variance, so the payoff holds exactly and the network learns the correction | 13,057 | 0.176 | 0.269 | 0.222 | 0.00169 | 0.145 | 0.0117 | 0.00288 | 3,971 |  |
| `P3_time_balanced` | Loss balancing: residual weighted by tau / tau_max against the 1/sqrt(tau) blow-up at the strike, with collocation tau = tau_max u^2 | 13,057 | 0.188 | 0.286 | 0.237 | 0.00214 | 0.154 | 0.012 | 0.00329 | 3,719 |  |
| `P4_variance_boundary` | Variance boundary (new): the degenerate PDE c_tau = kappa theta c_v as a residual on v = 0; v extended to v_hi = 3 with c_v = 0 there (moves interior prices by at most 4e-7, results/neural/variance_cap_study); collocation packed near v = 0 (v = v_hi u^2) and sqrt(v + 0.01) as the network input | 13,057 | 0.241 | 0.302 | 0.272 | 0.00376 | 0.181 | 0.0112 | 0.00471 | 4,624 |  |
| `P5_adaptive` | Residual-based adaptive sampling (RAD): every 1000 steps keep 32,768 of 131,072 candidates drawn with probability proportional to |residual| / mean + 1 | 13,057 | 0.245 | 0.295 | 0.27 | 0.00456 | 0.181 | 0.0109 | 0.005 | 4,654 |  |
| `P6_lbfgs` | Quasi-Newton polish: full-batch L-BFGS (strong Wolfe line search) warm-started from P5 on 16,384 RAD-selected collocation points | 13,057 | 0.193 | 0.295 | 0.244 | 0.00213 | 0.147 | 0.0104 | 0.00313 | 3,450 |  |
| `P7_nncg` | Second-order refinement: Nystrom-preconditioned Newton-CG (Rathore et al., 2024) warm-started from P6 | 13,057 | 0.179 | 0.299 | 0.239 | 0.00172 | 0.144 | 0.00991 | 0.00293 | 907 |  |
| `P8_scale_up` | Scale up: width 128 and 25,000 Adam steps from a fresh start, then 1,000 L-BFGS iterations and 20 Newton-CG steps | 50,689 | 0.199 | 0.295 | 0.247 | 0.00244 | 0.164 | 0.0103 | 0.00358 | 13,282 | adam stopped at step 19200 of 25000: wall-clock cap |
| `P9_envelope` | Enveloped correction as an implied-volatility output: c = BS(x, sqrt(w tau) e^N), whose first-order correction sqrt(w tau) phi(d_2) N fades like the time value near expiry and away from the strike, with kappa and theta fed as kappa tau and kappa theta tau so that their effect vanishes as tau -> 0 (two earlier variants are reported: the additive tau phi(d_2) N envelope, and the implied-volatility output with plain kappa, theta inputs); trained from a fresh start through the Adam, L-BFGS and Newton-CG stages | 13,057 | 0.146 | 0.227 | 0.187 | 0.00162 | 0.0966 | 0.00655 | 0.00223 | 7,431 |  |
| `P10_anchored` | Anchored (hybrid) PINN: fine-tune P9 with the PDE residual plus reference prices at 4,096 anchor contracts of the training domain (their own seed), then L-BFGS | 13,057 | 0.101 | 0.144 | 0.122 | 0.000627 | 0.0786 | 0.00534 | 0.00121 | 2,510 |  |
| Black-Scholes base | no network | 0 | 0.312 | 0.231 | 0.272 |  |  |  |  | 0 | baseline |
| `P10_anchored_tau_envelope` (variant) | Anchored (hybrid) PINN: fine-tune P9 with the PDE residual plus reference prices at 4,096 anchor contracts of the training domain (their own seed), then L-BFGS | 13,057 | 0.114 | 0.246 | 0.18 | 0.000768 | 0.0984 | 0.00683 | 0.00152 | 3,223 | replaced after the diagnosis; not eligible for selection |
| `P9_envelope_bsvol_plain_drift` (variant) | Enveloped correction as an implied-volatility output: c = BS(x, sqrt(w tau) e^N), whose first-order correction sqrt(w tau) phi(d_2) N fades like the time value near expiry and away from the strike (the additive tau phi(d_2) N envelope, run first, is reported as a variant); trained from a fresh start through the Adam, L-BFGS and Newton-CG stages | 13,057 | 0.135 | 0.419 | 0.277 | 0.00151 | 0.0851 | 0.00588 | 0.00209 | 5,566 | replaced after the diagnosis; not eligible for selection |
| `P9_envelope_tau_envelope` (variant) | Enveloped correction: c = BS + tau phi(d_2) N, which fades like the time value away from the strike; trained from a fresh start through the Adam, L-BFGS and Newton-CG stages | 13,057 | 0.17 | 0.285 | 0.228 | 0.00188 | 0.119 | 0.00742 | 0.00276 | 7,216 | replaced after the diagnosis; not eligible for selection |

## DeepONet ladder

| Level | Technique added | Parameters | Validation | Short-maturity | Selection | Price RMSE | Gamma RMSE | Vega RMSE | Theta RMSE | Training s (wall) | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `D0_vanilla` | Textbook DeepONet: branch on (r, q, kappa, theta, xi, rho) plus a put flag, trunk on (S/K, v0, tau); price labels from the reference; Adam | 34,049 | 0.267 | 6.08 | 3.17 | 0.00212 | 0.413 | 0.0228 | 0.0152 | 416 |  |
| `D1_reduced` | Exact reduction: branch on (kappa, theta, xi, rho), trunk on (x, v, tau); put-call parity | 33,857 | 0.278 | 5.07 | 2.67 | 0.00212 | 0.44 | 0.026 | 0.0155 | 416 |  |
| `D2_hard_payoff` | Black-Scholes at the expected average variance as the base: c = BS + tau <branch, trunk(x, v, sqrt(tau))> | 33,857 | 0.127 | 0.272 | 0.199 | 0.000714 | 0.135 | 0.00935 | 0.00211 | 437 |  |
| `D3_differential` | Differential (Sobolev) training: add the 10 derivative labels c_x, c_v, c_tau, c_kappa, c_theta, c_xi, c_rho, c_xx, c_xv, c_vv | 33,857 | 0.11 | 0.364 | 0.237 | 0.000737 | 0.16 | 0.00641 | 0.00205 | 3,870 |  |
| `D4_physics` | Physics-informed DeepONet: PDE residual on unlabelled collocation over the computational box (tau from 0, v to v_hi = 3 packed near 0, sqrt(v + 0.01) in the trunk), the degenerate equation on v = 0, c_v = 0 at v_hi and the far x boundary | 33,857 | 0.11 | 0.374 | 0.242 | 0.000899 | 0.132 | 0.00767 | 0.00268 | 8,549 |  |
| `D5_short_maturity` | Short maturities: 25% of the labelled contracts and of the collocation drawn with tau in 0.005 to 0.05 | 33,857 | 0.116 | 0.305 | 0.21 | 0.00119 | 0.137 | 0.00874 | 0.00278 | 8,561 |  |
| `D6_lbfgs` | Quasi-Newton polish: full-batch L-BFGS warm-started from D5 on 4,096 labelled contracts and 4,096 collocation points | 33,857 | 0.077 | 0.391 | 0.234 | 0.000537 | 0.1 | 0.00453 | 0.00195 | 1,035 |  |
| `D7_scale_up` | Scale up: width and rank 128 and 131,072 labelled contracts from a fresh start, then the L-BFGS stage of D6 | 133,249 | 0.0897 | 0.362 | 0.226 | 0.000621 | 0.113 | 0.00587 | 0.00167 | 13,710 |  |
| `D8_envelope` | Enveloped correction as an implied-volatility output: c = BS(x, sqrt(w tau) e^N), whose first-order correction fades like the time value near expiry and away from the strike, with a drift split N = <branch(xi, rho), trunk_1> + (tau / tau_max) <branch(kappa, theta, xi, rho), trunk_2> so that kappa and theta drop out as tau -> 0 (the implied-volatility output without the split is reported as a variant); trained from a fresh start through the Adam and L-BFGS stages | 54,849 | 0.0573 | 0.133 | 0.095 | 0.000949 | 0.0657 | 0.00334 | 0.00192 | 16,436 |  |
| `D9_ensemble` | Ensemble: average of four independently seeded D8 trainings (seeds 0-3); Greeks are member averages | 219,396 | 0.0452 | 0.0752 | 0.0602 | 0.00058 | 0.0496 | 0.00213 | 0.000968 | 60,484 |  |
| Black-Scholes base | no network | 0 | 0.312 | 0.231 | 0.272 |  |  |  |  | 0 | baseline |
| `D8_envelope_no_drift_split` (variant) | Enveloped correction as an implied-volatility output: c = BS(x, sqrt(w tau) e^N) with N = <branch, trunk>, whose first-order correction fades like the time value near expiry and away from the strike; trained from a fresh start through the Adam and L-BFGS stages | 33,857 | 0.0419 | 0.193 | 0.117 | 0.000392 | 0.0379 | 0.00196 | 0.000851 | 9,057 | replaced after the diagnosis; not eligible for selection |

## Frozen test set

Selected pricers and the Black-Scholes base on every split (score = mean over the 12 quantities of RMSE / RMS(reference) within the split).

| Pricer | Split | Cases | Score | price RMSE | delta RMSE | gamma RMSE | vega RMSE | volga RMSE | theta_cal RMSE | d_xi RMSE | max price error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P10_anchored | interpolation | 128 | 0.171 | 0.000965 | 0.0115 | 0.158 | 0.00719 | 0.321 | 0.00129 | 0.00266 | 0.00571 |
| P10_anchored | short_maturity | 32 | 0.137 | 0.000112 | 0.00192 | 0.0492 | 0.00134 | 0.0563 | 0.00491 | 0.000248 | 0.000288 |
| P10_anchored | extrapolation | 72 | 0.147 | 0.000914 | 0.0103 | 0.269 | 0.00908 | 0.545 | 0.00334 | 0.00302 | 0.00345 |
| P10_anchored | anchor | 20 | 0.228 | 0.00407 | 0.0579 | 0.171 | 0.0307 | 2.43 | 0.00179 | 0.00423 | 0.0109 |
| P10_anchored | all | 252 | 0.144 | 0.00142 | 0.0191 | 0.19 | 0.0112 | 0.778 | 0.00271 | 0.00276 | 0.0109 |
| D9_ensemble | interpolation | 128 | 0.0626 | 0.000642 | 0.00354 | 0.107 | 0.00291 | 0.0942 | 0.000785 | 0.000585 | 0.0019 |
| D9_ensemble | short_maturity | 32 | 0.0867 | 7.72e-05 | 0.000911 | 0.0357 | 0.000597 | 0.0147 | 0.00302 | 9.18e-05 | 0.000251 |
| D9_ensemble | extrapolation | 72 | 0.0727 | 0.000616 | 0.00407 | 0.114 | 0.00417 | 0.304 | 0.00138 | 0.00144 | 0.00139 |
| D9_ensemble | anchor | 20 | 0.144 | 0.00111 | 0.0362 | 0.282 | 0.00647 | 0.815 | 0.00142 | 0.00252 | 0.00294 |
| D9_ensemble | all | 252 | 0.0733 | 0.000646 | 0.0107 | 0.127 | 0.00355 | 0.289 | 0.00147 | 0.00113 | 0.00294 |
| Black-Scholes at the expected average variance | interpolation | 128 | 0.418 | 0.00899 | 0.0334 | 0.279 | 0.0231 | 0.492 | 0.00928 | 0.0138 | 0.0434 |
| Black-Scholes at the expected average variance | short_maturity | 32 | 0.263 | 0.000323 | 0.00471 | 0.128 | 0.00274 | 0.074 | 0.0154 | 0.000517 | 0.000825 |
| Black-Scholes at the expected average variance | extrapolation | 72 | 0.34 | 0.00898 | 0.0372 | 0.353 | 0.0175 | 0.586 | 0.0111 | 0.0154 | 0.0313 |
| Black-Scholes at the expected average variance | anchor | 20 | 0.412 | 0.0154 | 0.106 | 0.435 | 0.118 | 3.73 | 0.00762 | 0.0205 | 0.0562 |
| Black-Scholes at the expected average variance | all | 252 | 0.344 | 0.0091 | 0.043 | 0.304 | 0.0382 | 1.15 | 0.0107 | 0.014 | 0.0562 |

Every ladder level on all 252 cases (context only; the selection was frozen before):

| Level | Test score | Price RMSE | Gamma RMSE | Vega RMSE |
|---|---:|---:|---:|---:|
| Black-Scholes at the expected average variance | 0.344 | 0.0091 | 0.304 | 0.0382 |
| P10_anchored | 0.144 | 0.00142 | 0.19 | 0.0112 |
| D9_ensemble | 0.0733 | 0.000646 | 0.127 | 0.00355 |
| P0_vanilla | 0.574 | 0.0276 | 1.11 | 0.11 |
| P1_reduced | 0.738 | 0.0259 | 1.15 | 0.107 |
| P2_hard_payoff | 0.21 | 0.00196 | 0.267 | 0.0326 |
| P3_time_balanced | 0.218 | 0.0024 | 0.276 | 0.033 |
| P4_variance_boundary | 0.276 | 0.00528 | 0.304 | 0.0288 |
| P5_adaptive | 0.273 | 0.00575 | 0.304 | 0.0287 |
| P6_lbfgs | 0.236 | 0.00324 | 0.268 | 0.0259 |
| P7_nncg | 0.21 | 0.00219 | 0.264 | 0.0235 |
| P8_scale_up | 0.234 | 0.00362 | 0.288 | 0.0263 |
| P9_envelope | 0.175 | 0.0023 | 0.21 | 0.0114 |
| D0_vanilla | 0.447 | 0.017 | 1.17 | 0.0833 |
| D1_reduced | 0.398 | 0.00525 | 1.05 | 0.0636 |
| D2_hard_payoff | 0.16 | 0.001 | 0.253 | 0.0277 |
| D3_differential | 0.133 | 0.00115 | 0.262 | 0.0122 |
| D4_physics | 0.147 | 0.00193 | 0.258 | 0.0162 |
| D5_short_maturity | 0.148 | 0.00162 | 0.273 | 0.0198 |
| D6_lbfgs | 0.106 | 0.000999 | 0.292 | 0.0111 |
| D7_scale_up | 0.104 | 0.000892 | 0.268 | 0.00903 |
| D8_envelope | 0.0881 | 0.00104 | 0.149 | 0.00514 |

## Benchmark, cost at equal accuracy and break-even

Classical results on the same 252 cases are in `results/classical_v1/report.md`. For each classical method and quantity, the cheapest setting whose test RMSE is at most the network's:

**PINN `P10_anchored`**

| Quantity | Network RMSE | COS | ADI | QE Monte Carlo, dt 1/8 | QE Monte Carlo, dt 1/32 | QE-RQMC, dt 1/32 |
|---|---:|---:|---:|---:|---:|---:|
| price | 0.00142 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_100000_dt8` (1.2 s) | `mc_100000_dt32` (5.3 s) | `rqmc_2^10x16_dt32` (1.07 s) |
| delta | 0.0191 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_10000_dt8` (0.0974 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| gamma | 0.19 | `cos_32` (0.00304 s) | `adi_80x40x40` (0.368 s) | `mc_10000_dt8` (0.0974 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| vega | 0.0112 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_10000_dt8` (0.0974 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| vanna | 0.238 | `cos_64` (0.00311 s) | `adi_40x20x20` (0.123 s) | `mc_100000_dt8` (1.2 s) | `mc_100000_dt32` (5.3 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| volga | 0.778 | `cos_32` (0.00304 s) | `adi_80x40x40` (0.368 s) | none | none | none |
| theta_cal | 0.00271 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_100000_dt8` (1.2 s) | `mc_100000_dt32` (5.3 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| rho_rate | 0.027 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_10000_dt8` (0.0974 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| d_kappa | 0.00183 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_10000_dt8` (0.0974 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| d_theta | 0.0207 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_100000_dt8` (1.2 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| d_xi | 0.00276 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_10000_dt8` (0.0974 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^10x16_dt32` (1.07 s) |
| d_rho | 0.00342 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_100000_dt8` (1.2 s) | `mc_100000_dt32` (5.3 s) | `rqmc_2^10x16_dt32` (1.07 s) |

**DeepONet `D9_ensemble`**

| Quantity | Network RMSE | COS | ADI | QE Monte Carlo, dt 1/8 | QE Monte Carlo, dt 1/32 | QE-RQMC, dt 1/32 |
|---|---:|---:|---:|---:|---:|---:|
| price | 0.000646 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_1000000_dt8` (12.9 s) | `mc_1000000_dt32` (58 s) | `rqmc_2^10x16_dt32` (1.07 s) |
| delta | 0.0107 | `cos_64` (0.00311 s) | `adi_40x20x20` (0.123 s) | `mc_10000_dt8` (0.0974 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| gamma | 0.127 | `cos_128` (0.00346 s) | `adi_80x40x40` (0.368 s) | `mc_10000_dt8` (0.0974 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| vega | 0.00355 | `cos_64` (0.00311 s) | `adi_80x40x40` (0.368 s) | `mc_100000_dt8` (1.2 s) | `mc_100000_dt32` (5.3 s) | `rqmc_2^14x16_dt32` (10.9 s) |
| vanna | 0.16 | `cos_64` (0.00311 s) | `adi_80x40x40` (0.368 s) | `mc_100000_dt8` (1.2 s) | `mc_100000_dt32` (5.3 s) | `rqmc_2^12x16_dt32` (2.95 s) |
| volga | 0.289 | `cos_32` (0.00304 s) | `adi_80x40x40` (0.368 s) | none | none | none |
| theta_cal | 0.00147 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_100000_dt8` (1.2 s) | `mc_1000000_dt32` (58 s) | `rqmc_2^10x16_dt32` (1.07 s) |
| rho_rate | 0.016 | `cos_64` (0.00311 s) | `adi_40x20x20` (0.123 s) | `mc_10000_dt8` (0.0974 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| d_kappa | 0.00121 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_10000_dt8` (0.0974 s) | `mc_10000_dt32` (0.388 s) | `rqmc_2^8x16_dt32` (0.636 s) |
| d_theta | 0.0111 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_100000_dt8` (1.2 s) | `mc_100000_dt32` (5.3 s) | `rqmc_2^12x16_dt32` (2.95 s) |
| d_xi | 0.00113 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_100000_dt8` (1.2 s) | `mc_100000_dt32` (5.3 s) | `rqmc_2^12x16_dt32` (2.95 s) |
| d_rho | 0.00148 | `cos_32` (0.00304 s) | `adi_40x20x20` (0.123 s) | `mc_1000000_dt8` (12.9 s) | `mc_1000000_dt32` (58 s) | `rqmc_2^10x16_dt32` (1.07 s) |

**Break-even.** N* = (training seconds + label seconds) / (classical seconds - network seconds) per contract, against the cheapest classical setting that matches the network on the price, and on all 12 quantities at once. Training seconds are wall-clock: the ladders ran 3 levels at a time with 4 threads each next to the classical benchmark, so each family's wall-clock is multiplied by a load factor measured by re-running a pilot of one level alone (PINN 0.34, DeepONet 0.35; `load_factor.json`). Process CPU seconds (all threads) are in `training_cost.json` for the levels that recorded them; they are not used because a 4-thread process's CPU seconds overstate its wall-clock on an idle machine.

| Network | Training chain | Wall s (under load) | Scaled s | Label s | Total s |
|---|---:|---:|---:|---:|---:|
| PINN `P10_anchored` | P9_envelope -> P10_anchored | 9,940 | 3,425 | 1.8 | 3,427 |
| DeepONet `D9_ensemble` | D9_ensemble | 60,484 | 21,267 | 14.7 | 21,281 |

| Network | Method | Matched on | Classical setting | Classical s per contract | N* (network batched) | N* (network one at a time) |
|---|---:|---:|---:|---:|---:|---:|
| pinn | COS | price | `cos_32` | 0.00304 (8.74e-05 batched) | 45,849,783 | never (classical is faster) |
| pinn | COS | all 12 | `cos_64` | 0.00311 (0.000157 batched) | 23,765,900 | never (classical is faster) |
| pinn | ADI | price | `adi_40x20x20` | 0.123 | 27,750 | 28,934 |
| pinn | ADI | all 12 | `adi_80x40x40` | 0.368 | 9,303 | 9,433 |
| pinn | QE Monte Carlo, dt 1/8 | price | `mc_100000_dt8` | 1.2 | 2,865 | 2,878 |
| pinn | QE Monte Carlo, dt 1/8 | all 12 | none | - | classical never matches | classical never matches |
| pinn | QE Monte Carlo, dt 1/32 | price | `mc_100000_dt32` | 5.3 | 646 | 647 |
| pinn | QE Monte Carlo, dt 1/32 | all 12 | none | - | classical never matches | classical never matches |
| pinn | QE-RQMC, dt 1/32 | price | `rqmc_2^10x16_dt32` | 1.07 | 3,195 | 3,210 |
| pinn | QE-RQMC, dt 1/32 | all 12 | none | - | classical never matches | classical never matches |
| deeponet | COS | price | `cos_32` | 0.00304 (8.74e-05 batched) | never (classical is faster) | never (classical is faster) |
| deeponet | COS | all 12 | `cos_128` | 0.00346 (0.000417 batched) | 123,437,189 | never (classical is faster) |
| deeponet | ADI | price | `adi_40x20x20` | 0.123 | 172,669 | 225,755 |
| deeponet | ADI | all 12 | `adi_80x40x40` | 0.368 | 57,815 | 62,757 |
| deeponet | QE Monte Carlo, dt 1/8 | price | `mc_1000000_dt8` | 12.9 | 1,654 | 1,658 |
| deeponet | QE Monte Carlo, dt 1/8 | all 12 | none | - | classical never matches | classical never matches |
| deeponet | QE Monte Carlo, dt 1/32 | price | `mc_1000000_dt32` | 58 | 367 | 367 |
| deeponet | QE Monte Carlo, dt 1/32 | all 12 | none | - | classical never matches | classical never matches |
| deeponet | QE-RQMC, dt 1/32 | price | `rqmc_2^10x16_dt32` | 1.07 | 19,844 | 20,395 |
| deeponet | QE-RQMC, dt 1/32 | all 12 | none | - | classical never matches | classical never matches |

Batched networks are compared with COS priced as a batch and with ADI, Monte Carlo and RQMC per contract; one-at-a-time networks with every method one contract at a time. 'Classical never matches': no setting run reaches the network's RMSE on every quantity; for Monte Carlo and RQMC the quantity they never reach is Volga (a bumped second difference).

## Calibration test

Fit (v0, kappa, theta, xi, rho) to 100 out-of-the-money options (S/K 0.8 to 1.2, tau 0.1 to 2) priced by the reference at {'v0': 0.04, 'kappa': 1.5, 'theta': 0.06, 'xi': 0.6, 'rho': -0.65}, r = 0.02, q = 0.01; 8 random starts in the training domain; box-constrained Levenberg-Marquardt on price residuals; one thread. COS Jacobians come from its own sensitivities, the networks' exactly from autograd.

| Pricer | Seconds per calibration | Iterations | Median max relative parameter error | Within 1% | Within 10% | Median fit RMSE |
|---|---:|---:|---:|---:|---:|---:|
| cos_128 | 1.71 | 12 | 3.17e-06 | 8/8 | 8/8 | 1.79e-08 |
| cos_256 | 3.81 | 11.9 | 3.17e-06 | 8/8 | 8/8 | 1.76e-08 |
| pinn_P10_anchored | 0.254 | 86.5 | 1.96 | 0/8 | 0/8 | 0.000482 |
| deeponet_D9_ensemble | 0.324 | 10.5 | 0.282 | 0/8 | 0/8 | 0.000137 |

- PINN: no calibration break-even. It is 6.7 times faster per calibration than COS-128, but it recovers the parameters from none of the starts, so it does not do the job COS does. If speed alone counted, training would pay back after 2,360 calibrations.
- DeepONet: no calibration break-even. It is 5.3 times faster per calibration than COS-128, but it recovers the parameters from none of the starts, so it does not do the job COS does. If speed alone counted, training would pay back after 15,392 calibrations.

## No-arbitrage and Greek validation

Counts of violations (price below discounted intrinsic, Delta outside [0, e^{-q tau}] for calls or [-e^{-q tau}, 0] for puts, negative Gamma, negative Vega):

| Pricer / panel | Cases | Price < intrinsic | Delta out of bounds | Gamma < 0 | Vega < 0 |
|---|---:|---:|---:|---:|---:|
| bs_base | 252 | 0 | 0 | 0 | 0 |
| pinn_selected | 252 | 0 | 0 | 0 | 0 |
| deeponet_selected | 252 | 0 | 0 | 0 | 0 |
| P0_vanilla | 252 | 26 | 14 | 18 | 11 |
| P1_reduced | 252 | 16 | 0 | 0 | 6 |
| P2_hard_payoff | 252 | 6 | 3 | 1 | 3 |
| P3_time_balanced | 252 | 5 | 2 | 1 | 5 |
| P4_variance_boundary | 252 | 0 | 2 | 1 | 1 |
| P5_adaptive | 252 | 1 | 2 | 1 | 1 |
| P6_lbfgs | 252 | 6 | 3 | 1 | 3 |
| P7_nncg | 252 | 5 | 3 | 1 | 4 |
| P8_scale_up | 252 | 6 | 2 | 1 | 3 |
| P9_envelope | 252 | 0 | 0 | 0 | 0 |
| D0_vanilla | 252 | 19 | 1 | 26 | 10 |
| D1_reduced | 252 | 19 | 0 | 18 | 3 |
| D2_hard_payoff | 252 | 6 | 4 | 1 | 5 |
| D3_differential | 252 | 7 | 4 | 1 | 4 |
| D4_physics | 252 | 12 | 4 | 1 | 1 |
| D5_short_maturity | 252 | 7 | 3 | 1 | 3 |
| D6_lbfgs | 252 | 7 | 4 | 2 | 4 |
| D7_scale_up | 252 | 6 | 5 | 3 | 4 |
| D8_envelope | 252 | 0 | 0 | 0 | 0 |
| pinn_selected_validation | 4096 | 0 | 0 | 0 | 0 |
| deeponet_selected_validation | 4096 | 0 | 0 | 0 | 0 |
| pinn_selected_short_validation | 4096 | 0 | 0 | 0 | 0 |
| deeponet_selected_short_validation | 4096 | 0 | 0 | 0 | 0 |

Share of test cases inside the Monte Carlo 95% interval (QE, dt 1/32), per quantity:

| Monte Carlo | Values | price | delta | gamma | vega | vanna | volga | theta_cal | rho_rate | d_kappa | d_theta | d_xi | d_rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mc_100000_dt32 | reference | 0.956 | 0.948 | 0.94 | 0.948 | 0.937 | 0.929 | 0.948 | 0.937 | 0.901 | 0.94 | 0.948 | 0.94 |
| mc_100000_dt32 | pinn_selected | 0.73 | 0.631 | 0.77 | 0.619 | 0.766 | 0.802 | 0.655 | 0.536 | 0.397 | 0.607 | 0.413 | 0.663 |
| mc_100000_dt32 | deeponet_selected | 0.778 | 0.75 | 0.778 | 0.802 | 0.853 | 0.861 | 0.726 | 0.671 | 0.488 | 0.639 | 0.746 | 0.821 |
| mc_1000000_dt32 | reference | 0.968 | 0.925 | 0.913 | 0.948 | 0.937 | 0.948 | 0.948 | 0.929 | 0.833 | 0.913 | 0.929 | 0.921 |
| mc_1000000_dt32 | pinn_selected | 0.552 | 0.337 | 0.544 | 0.294 | 0.571 | 0.54 | 0.381 | 0.238 | 0.171 | 0.294 | 0.19 | 0.381 |
| mc_1000000_dt32 | deeponet_selected | 0.484 | 0.48 | 0.571 | 0.492 | 0.702 | 0.663 | 0.476 | 0.361 | 0.294 | 0.365 | 0.433 | 0.718 |

## Files and reproduction

- `ladder/`: every level's checkpoint, log and record (settings, source hashes, scores, history).
- `ladder.csv`, `selection.json` (frozen selection with hashes), `test_metrics.csv`, `test_predictions.csv`, `test_checks.json`, `neural_timing.json`, `equal_accuracy.csv`, `break_even.csv`, `training_cost.json`, `load_factor.json`, `calibration.csv`, `calibration.json`.
- Reproduce (from heston/): `python -m experiments.neural.run_ladders`, then `select_models`, `evaluate_test`, `load_factor`, `compare`, `calibration`, `make_report`; verify with `python -m experiments.neural.verify_results`.
