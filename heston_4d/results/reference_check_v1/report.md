# Phase 0: Heston basket reference check

Development contracts only.

## Time step (price, 16 x 16,384 Sobol paths, same normals)

| Contract | 1/8 - 1/128 | 1/16 - 1/128 | 1/32 - 1/128 | 1/64 - 1/128 | price at 1/128 | SE |
|---|---:|---:|---:|---:|---:|---:|
| random00 | -1.2e-02 | -4.9e-03 | -2.9e-03 | -1.5e-03 | 7.3431 | 6.4e-04 |
| random01 | -8.3e-03 | -3.1e-03 | -2.2e-03 | -2.0e-03 | 4.4418 | 1.3e-03 |
| random02 | -1.1e-03 | -2.9e-04 | -3.0e-04 | +5.8e-05 | 0.3703 | 1.7e-04 |
| random03 | +1.5e-03 | +1.5e-03 | +4.9e-04 | +3.1e-04 | 12.4468 | 7.3e-04 |
| random04 | -1.1e-02 | -8.0e-04 | +5.8e-04 | -1.5e-04 | 11.7901 | 2.3e-03 |
| random05 | -9.2e-03 | -2.8e-03 | -8.1e-04 | -7.6e-05 | 11.6275 | 6.5e-05 |
| random06 | -1.2e-03 | +1.4e-03 | +1.5e-03 | +2.6e-03 | 6.9829 | 1.2e-03 |
| random07 | -1.4e-02 | -4.1e-03 | -4.7e-04 | -1.3e-03 | 24.1642 | 4.8e-04 |

## Points ladder (every quantity, protocol dt)

| Points | Contracts | Median price SE | Max price SE | Share <= target | Median s | Max s |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 8 | 1.8e-03 | 6.0e-03 | 12% | 161 | 271 |

## Training labels and evaluation references against the truth

Label price relative error (max(|truth|, 0.01)): RMS 7.9e-03, max 1.8e-02; 0.148 s per contract.

Evaluation rule: median relative standard error per quantity, largest six: vol_correlation_14 8.2e-01, vol_correlation_23 7.1e-01, vol_correlation_34 4.0e-01, vol_correlation_13 2.9e-01, vol_correlation_12 2.8e-01, vol_correlation_24 2.4e-01. Seconds per contract: median 15.6.
