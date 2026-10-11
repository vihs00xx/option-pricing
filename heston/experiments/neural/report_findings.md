- **COS beats both networks, by a wide margin.** On the 252 test cases the selected DeepONet
  (`D9_ensemble`) scores 0.073 and the selected PINN (`P10_anchored`) 0.144, against 0.0019 for
  COS with 256 terms (price RMSE 7.9e-6 against 6.5e-4 and 1.4e-3). COS needs 3.7 ms per contract
  one at a time and 0.97 ms in a batch, with all 12 quantities. The PINN is faster in a batch
  (13 microseconds per contract), but against batched COS its training pays back only after about
  24 to 46 million contracts. The DeepONet (245 microseconds) is slower than batched COS with
  32 terms, which already matches its price error. Against COS with 128 terms, which matches it
  on all 12 quantities, it pays back only after about 120 million contracts. One at a time, COS
  is faster than both networks. For one stock under Heston, COS is the pricer to use. All
  timings here, network and classical, are for the price with all 12 quantities, including
  comparisons matched on the price alone; a price-only workload was not timed. The break-even
  counts use the estimated training costs in the last item.
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
- **Short maturities and the extrapolation split hold up after the change.** By split, the
  DeepONet test scores are 0.063 interpolation, 0.087 short maturity, 0.073 extrapolation and
  0.144 anchors. The short-maturity cases (tau 0.005 to 0.05) are independently drawn contracts,
  not extrapolation: from `D5` on, 25% of the DeepONet's labelled contracts and collocation come
  from that range, and the selection score of both families averages in a short-maturity
  validation panel. The anchors include the hardest cases: strong
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
- **No arbitrage violations were found** for both selected networks on all 252 test cases and
  both 4,096-contract validation panels: no price below intrinsic, no Delta out of bounds, no
  negative Gamma or Vega. The implied-volatility output guarantees only the price bounds: a
  Black-Scholes price with a positive volatility cannot leave them. Delta, Gamma and Vega also
  pick up derivatives of the learned volatility (for example Delta = Delta_BS + Vega_BS
  d sigma / dS), so their bounds are not guaranteed; they held on these panels. Every level built
  on that output (`P9`, `P10`, `D8`, `D9`) has no violation. Every level before it has some: `P0` has 26 prices below intrinsic, and `D2` to `D7` have 6 to 12.
- **Against Monte Carlo intervals** (QE, dt 1/32), the reference lies inside the 95% interval for
  90 to 96% of cases per quantity at 10^5 paths. The DeepONet lies inside for 49 to 86% and the
  PINN for 40 to 80%. At 10^6 paths: DeepONet 29 to 72%, PINN 17 to 57%. Monte Carlo at those
  path counts can detect the networks' Greek errors, which the exact-reference RMSEs quantify.
- **Training cost, estimated** for an idle machine: about 1 hour for the PINN (P9 then P10) and
  5.9 hours for the DeepONet ensemble (four D8 trainings; one member alone is about 1.5 hours).
  These are not measured complete trainings. They are wall-clock seconds measured with 3 to 4
  levels sharing the CPU, multiplied by a load factor of 0.34 (PINN) and 0.35 (DeepONet). The
  factor comes from 300 Adam steps of `P4` and `D4` run alone, and is applied to whole trainings
  of other levels, including their L-BFGS and Newton-CG stages, where it was not measured.
  Labelling 32,768 contracts with the torch twin takes about 15 s.
