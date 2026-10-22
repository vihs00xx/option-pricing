# Phase 0: Heston basket reference check

Development contracts only.

## Time step (price, 16 x 16,384 Sobol paths, same normals)

| Contract | 1/8 - 1/128 | 1/16 - 1/128 | 1/32 - 1/128 | 1/64 - 1/128 | price at 1/128 | SE |
|---|---:|---:|---:|---:|---:|---:|
| random00 | +5.1e-04 | -1.5e-03 | -1.0e-03 | -3.8e-04 | 7.9002 | 1.8e-04 |
| random01 | -2.2e-02 | -9.7e-03 | -4.4e-03 | -4.0e-04 | 4.5280 | 8.3e-04 |
| random02 | -2.6e-03 | -1.5e-03 | -4.0e-04 | -1.9e-04 | 10.6344 | 7.0e-05 |
| random03 | -2.3e-04 | +1.3e-03 | -8.7e-04 | -3.0e-04 | 16.5369 | 7.6e-04 |
| random04 | -8.2e-03 | -7.7e-04 | -5.2e-04 | +1.7e-03 | 15.1536 | 5.4e-04 |
| random05 | +1.9e-03 | +5.2e-04 | +6.8e-05 | +6.5e-05 | 7.8854 | 3.6e-05 |
| random06 | -2.2e-02 | -6.4e-03 | -5.4e-04 | -2.9e-03 | 22.3053 | 2.6e-03 |
| random07 | -1.9e-02 | -6.2e-03 | -1.7e-03 | -3.7e-04 | 4.1889 | 2.7e-05 |
| random08 | -4.2e-02 | -1.3e-02 | -5.2e-03 | -2.1e-03 | 20.2540 | 1.1e-03 |
| random09 | +1.5e-02 | +3.8e-03 | +5.2e-04 | -2.9e-04 | 4.3296 | 1.4e-04 |

## Points ladder (every quantity, protocol dt)

| Points | Contracts | Median price SE | Max price SE | Share <= target | Median s | Max s |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 10 | 1.2e-03 | 4.9e-03 | 50% | 48 | 86 |
| 16384 | 10 | 5.1e-04 | 2.6e-03 | 50% | 184 | 393 |

## Training labels and evaluation references against the truth

Label price relative error (max(|truth|, 0.01)): RMS 1.4e-03, max 2.5e-03; 0.033 s per contract.

Evaluation rule: median relative standard error per quantity, largest six: d_leverage_1 1.0e-01, vol_correlation_12 9.4e-02, d_xi_1 3.1e-02, d_leverage_2 2.5e-02, d_xi_2 1.7e-02, d_kappa_1 1.1e-02. Seconds per contract: median 6.1.
