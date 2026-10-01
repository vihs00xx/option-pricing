# DeepONet optimizer refinement

The selected weight-0.01 model was refined without changing architecture, training data or Greek-loss weight. Two arms started from the same checkpoint: Adam with learning rate 0.00002 and full-batch L-BFGS with a strong-Wolfe line search. Each received a nominal 180-second budget; L-BFGS completed its final step at 181.63 seconds. L-BFGS accumulated exact full-data gradients over chunks of 1,024 examples. Multiplying its objective by 10,000 improved numerical scaling without changing the minimizer.

Candidate checkpoints were selected on validation price MSE normalized by strike. The arm-selection rule required each Greek RMSE to remain within 10% of the original model, then minimized price RMSE. L-BFGS won narrowly on price; Adam had lower RMSE for every Greek than L-BFGS.

| Validation RMSE | Original | Smaller-step Adam | L-BFGS |
|---|---:|---:|---:|
| Price | 0.08741 | 0.06286 | 0.06180 |
| Delta | 0.00858 | 0.00703 | 0.00728 |
| Gamma | 0.002225 | 0.002091 | 0.002139 |
| Vega | 0.8643 | 0.8395 | 0.8573 |
| Theta | 0.9979 | 0.9500 | 0.9845 |
| Rho | 0.9474 | 0.8645 | 0.8898 |

## Fresh audit

The old 68-case final benchmark was not reused. Before refinement, a new analytical audit dataset was generated and hashed: 4,096 coordinates on 256 contracts inside the training domain and 2,048 coordinates on 128 contracts in a broader stress domain. Its contracts were checked against training and validation contracts. Both the domain and random seeds were fixed before training. Audit predictions were evaluated only after the validation decision was recorded.

| Price RMSE | Original | L-BFGS | Reduction |
|---|---:|---:|---:|
| In training range | 0.09683 | 0.06294 | 35.0% |
| Broader stress range | 0.18826 | 0.17180 | 8.7% |

All Greek RMSEs improved inside the training range. Under stress, Theta RMSE rose from 1.6651 to 1.6686 and Rho from 2.3550 to 2.3667. Other stress Greek RMSEs improved modestly; this is primarily an improvement in price accuracy, not evidence that extrapolation or Greek estimation is solved. The stress distribution includes both in-domain and out-of-domain coordinates and must not be described as a pure extrapolation sample.

## Financial consistency

Violations were measured with an absolute tolerance of 1e-8. On validation points, price-bound violations increased from 5.86% to 6.84%, while maximum severity decreased from 0.4542 to 0.4139. Delta-bound violations decreased from 7.08% to 6.25%. Negative-Gamma points increased from 4.64% to 4.79%. On the broader stress sample the refined model had price-bound violations at 12.16% of points and negative Gamma at 13.82% of points. These diagnostics limit claims of financial reliability despite the lower RMSE.

This is a single-parent-checkpoint refinement study. It does not establish gains across random seeds. Both refined checkpoints are retained; the model registry lists the L-BFGS checkpoint as an audited refinement candidate. The prior primary model and its final-test results remain available for comparison. The new audit has now been opened and must not be used as an untouched test for future tuning.
