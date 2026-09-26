Corrected report (generated 2026-09-23 17:02 UTC). The original report.md in this folder is preserved unchanged; corrections cover terminology (matched update counts), gate-driven conclusions, and explicit exceptions.

# Parity under the established optimizer-refinement procedure (corrected)

Aggregates below are recomputed from the saved per-run metrics and selection records; identities are verified with the corrected scope-aware verifier. The original report.md in this folder is preserved unchanged. The final benchmark was evaluated once in an earlier study (`results/deeponet/primary/final_test_metrics.json`) and is already observed; this work did not access it again. Reused evaluation sets are observed benchmarks, not untouched tests.

## Identity verification (corrected verifier)

| Run | Role | Checkpoint | Dataset hashes | Scope | Status |
|---|---|---|---|---|---|
| control_22092026 | parent | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| control_22092026 | adam_small_steps | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| control_22092026 | lbfgs | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| control_22092027 | parent | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| control_22092027 | adam_small_steps | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| control_22092027 | lbfgs | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| control_22092028 | parent | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| control_22092028 | adam_small_steps | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| control_22092028 | lbfgs | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| parity_22092026 | parent | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| parity_22092026 | adam_small_steps | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| parity_22092026 | lbfgs | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| parity_22092027 | parent | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| parity_22092027 | adam_small_steps | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| parity_22092027 | lbfgs | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| parity_22092028 | parent | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| parity_22092028 | adam_small_steps | ok | content_sha256, trusted_file_sha256 | train_ | verified |
| parity_22092028 | lbfgs | ok | content_sha256, trusted_file_sha256 | train_ | verified |

All recorded checkpoint and dataset identities verify under the corrected verifier (legacy training-scope content hashes verified for training arrays; validation coverage established by the independently recorded frozen-manifest full-file hash). No provenance was repaired or regenerated.

## Selections and optimizer counts (generated from selection records)

| Architecture | Seed | Recorded | Recomputed | Eligible | Confirmed |
|---|---|---|---|---|---|
| control | 22092026 | lbfgs | lbfgs | baseline, adam_small_steps, lbfgs | True |
| control | 22092027 | adam_small_steps | adam_small_steps | baseline, adam_small_steps, lbfgs | True |
| control | 22092028 | lbfgs | lbfgs | baseline, adam_small_steps, lbfgs | True |
| parity | 22092026 | adam_small_steps | adam_small_steps | baseline, adam_small_steps, lbfgs | True |
| parity | 22092027 | lbfgs | lbfgs | baseline, adam_small_steps, lbfgs | True |
| parity | 22092028 | lbfgs | lbfgs | baseline, adam_small_steps, lbfgs | True |

Optimizer selection across the six architecture-seed combinations: Adam 2, L-BFGS 4.

- control: Adam 1, L-BFGS 2.
- parity: Adam 1, L-BFGS 2.

These counts are separate from the earlier three-seed refinement study on family checkpoints, where Adam won 2 and L-BFGS 0.

Validation RMSE (mean ± sample SD over seeds):

| Quantity | Refined control | Refined parity | Change |
|---|---:|---:|---:|
| price | 0.0912706 ± 0.01303 | 0.0659767 ± 0.0007488 | -27.7% |
| delta | 0.0106279 ± 0.001788 | 0.00838085 ± 0.000645 | -21.1% |
| gamma | 0.00272163 ± 0.0003639 | 0.00245613 ± 5.944e-05 | -9.8% |
| vega | 0.999758 ± 0.03471 | 0.825435 ± 0.04381 | -17.4% |
| theta | 1.35106 ± 0.216 | 1.11657 ± 0.07178 | -17.4% |
| rho | 1.14539 ± 0.1443 | 1.03017 ± 0.0173 | -10.1% |

Fresh in-domain RMSE (observed benchmark, mean ± SD):

| Quantity | Refined control | Refined parity | Change |
|---|---:|---:|---:|
| price | 0.092167 ± 0.009487 | 0.0687556 ± 0.003228 | -25.4% |
| delta | 0.0106827 ± 0.001545 | 0.00868515 ± 0.0005277 | -18.7% |
| gamma | 0.0024892 ± 0.0002634 | 0.00220497 ± 9.87e-05 | -11.4% |
| vega | 0.942535 ± 0.06451 | 0.836199 ± 0.008126 | -11.3% |
| theta | 1.49771 ± 0.2743 | 1.2 ± 0.1103 | -19.9% |
| rho | 1.13213 ± 0.1139 | 1.01945 ± 0.04272 | -10.0% |

Fresh stress RMSE (observed benchmark, mean ± sample SD):

| Quantity | Refined control | Refined parity | Change |
|---|---:|---:|---:|
| price | 0.237184 ± 0.0148 | 0.207886 ± 0.02278 | -12.4% |
| delta | 0.0190428 ± 0.002507 | 0.0176644 ± 0.00198 | -7.2% |
| gamma | 0.00362411 ± 0.0002543 | 0.00335537 ± 0.0002395 | -7.4% |
| vega | 1.94712 ± 0.02781 | 1.89595 ± 0.07608 | -2.6% |
| theta | 1.98533 ± 0.2714 | 1.75464 ± 0.2019 | -11.6% |
| rho | 3.07657 ± 0.5105 | 2.79431 ± 0.266 | -9.2% |

The stress evaluation set samples a broader domain than training (wider K, T, r, q, sigma and moneyness ranges with a lower time-to-expiry floor). It mixes coordinates inside and outside the training domain, so it is not a pure extrapolation sample.

## Financial constraints (frequency and maximum severity reported separately)

Severity aggregates are labeled precisely: "mean of per-seed maxima" averages the per-seed maximum violation; "worst maximum" is the largest maximum across all three seeds.

| validation | Constraint | Metric | Refined control | Refined parity | Change |
|---|---|---|---:|---:|---:|
| validation | price_bounds | frequency | 0.0668945 | 0.0750326 | +12.2% |
| validation | price_bounds | mean of per-seed maxima | 0.303048 | 0.304187 | +0.4% |
| validation | price_bounds | worst maximum | 0.369003 | 0.420602 | +14.0% |
| validation | delta_bounds | frequency | 0.0735677 | 0.0694987 | -5.5% |
| validation | delta_bounds | mean of per-seed maxima | 0.0811204 | 0.0708566 | -12.7% |
| validation | delta_bounds | worst maximum | 0.105628 | 0.0825393 | -21.9% |
| validation | negative_gamma | frequency | 0.0449219 | 0.0488281 | +8.7% |
| validation | negative_gamma | mean of per-seed maxima | 0.0187981 | 0.0214256 | +14.0% |
| validation | negative_gamma | worst maximum | 0.0248642 | 0.0265765 | +6.9% |
| in_domain | Constraint | Metric | Refined control | Refined parity | Change |
|---|---|---|---:|---:|---:|
| in_domain | price_bounds | frequency | 0.0696615 | 0.0745443 | +7.0% |
| in_domain | price_bounds | mean of per-seed maxima | 0.550426 | 0.322464 | -41.4% |
| in_domain | price_bounds | worst maximum | 0.754473 | 0.336177 | -55.4% |
| in_domain | delta_bounds | frequency | 0.0592448 | 0.0622559 | +5.1% |
| in_domain | delta_bounds | mean of per-seed maxima | 0.0486713 | 0.0513791 | +5.6% |
| in_domain | delta_bounds | worst maximum | 0.0521186 | 0.0622512 | +19.4% |
| in_domain | negative_gamma | frequency | 0.0472005 | 0.0450033 | -4.7% |
| in_domain | negative_gamma | mean of per-seed maxima | 0.0146892 | 0.0148844 | +1.3% |
| in_domain | negative_gamma | worst maximum | 0.0161798 | 0.0185301 | +14.5% |
| stress | Constraint | Metric | Refined control | Refined parity | Change |
|---|---|---|---:|---:|---:|
| stress | price_bounds | frequency | 0.120768 | 0.134115 | +11.1% |
| stress | price_bounds | mean of per-seed maxima | 1.37065 | 1.36503 | -0.4% |
| stress | price_bounds | worst maximum | 1.83944 | 1.55571 | -15.4% |
| stress | delta_bounds | frequency | 0.0701497 | 0.0579427 | -17.4% |
| stress | delta_bounds | mean of per-seed maxima | 0.107357 | 0.103435 | -3.7% |
| stress | delta_bounds | worst maximum | 0.135435 | 0.131135 | -3.2% |
| stress | negative_gamma | frequency | 0.138835 | 0.151855 | +9.4% |
| stress | negative_gamma | mean of per-seed maxima | 0.0216983 | 0.0222106 | +2.4% |
| stress | negative_gamma | worst maximum | 0.0280275 | 0.0308178 | +10.0% |

Exact call-put parity does not guarantee price bounds, a valid Delta, or a nonnegative Gamma; those constraints are measured separately below.

## Seed stability (qualified)

- Measured validation price RMSE sample standard deviation: control 0.0130257, parity 0.000748788.
- Parity showed approximately 17x lower sample standard deviation across these three seeds.
- This ratio measures variability of aggregate validation RMSE across these three seeds under this procedure. It does not establish general seed robustness or statistical significance, it is not uncertainty in individual predictions, and the price-stability result does not automatically extend to the Greeks.
- Greek variabilities differ from the price result and are not covered by this claim: delta SD control 0.00178767 vs parity 0.000645001, gamma SD control 0.000363854 vs parity 5.94407e-05, vega SD control 0.0347108 vs parity 0.0438123, theta SD control 0.215982 vs parity 0.0717799, rho SD control 0.144302 vs parity 0.0173.

## Verdict (computed)

- Refined parity improves 6 of 6 validation RMSEs versus refined control at matched parents, procedure and seeds, and enforces exact call-put consistency by construction.
- Financial constraint violations and limited seed coverage remain material limitations (see the constraint tables and the stability qualification above).

## Measured regressions

- price_bounds frequency on validation: regresses (0.0668945 -> 0.0750326, +12.2%).
- price_bounds worst maximum on validation: regresses (0.369003 -> 0.420602, +14.0%).
- negative_gamma frequency on validation: regresses (0.0449219 -> 0.0488281, +8.7%).
- negative_gamma mean of per-seed maxima on validation: regresses (0.0187981 -> 0.0214256, +14.0%).
- negative_gamma worst maximum on validation: regresses (0.0248642 -> 0.0265765, +6.9%).
- price_bounds frequency on in_domain: regresses (0.0696615 -> 0.0745443, +7.0%).
- delta_bounds frequency on in_domain: regresses (0.0592448 -> 0.0622559, +5.1%).
- delta_bounds mean of per-seed maxima on in_domain: regresses (0.0486713 -> 0.0513791, +5.6%).
- delta_bounds worst maximum on in_domain: regresses (0.0521186 -> 0.0622512, +19.4%).
- negative_gamma mean of per-seed maxima on in_domain: regresses (0.0146892 -> 0.0148844, +1.3%).
- negative_gamma worst maximum on in_domain: regresses (0.0161798 -> 0.0185301, +14.5%).
- price_bounds frequency on stress: regresses (0.120768 -> 0.134115, +11.1%).
- negative_gamma frequency on stress: regresses (0.138835 -> 0.151855, +9.4%).
- negative_gamma mean of per-seed maxima on stress: regresses (0.0216983 -> 0.0222106, +2.4%).
- negative_gamma worst maximum on stress: regresses (0.0280275 -> 0.0308178, +10.0%).
