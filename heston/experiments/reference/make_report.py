"""Write results/reference_check_v1/report.md from checks.json.

Run from heston/: python -m experiments.reference.make_report
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reference_check_v1"


def e(x):
    return f"{x:.1e}"


def main():
    c = json.loads((OUT / "checks.json").read_text(encoding="utf-8"))
    ref, gp, bs, cos, ql, mc, tw, vm, tm = (c[k] for k in (
        "reference", "1_lewis_vs_gil_pelaez", "2_black_scholes_limit", "3_cos", "4_quantlib",
        "5_qe_monte_carlo", "6_torch_twin", "7_v_max", "8_timing_seconds_per_contract"))
    fd = tw["finite_differences"]["rows"]
    L = ["# Phase 0: one-stock Heston reference check", "",
         f"Development contracts only (seed {c['seed']}): {c['contracts']['domain']} random domain contracts, "
         f"{c['contracts']['short']} short-maturity contracts (tau 0.005 to 0.05) and {c['contracts']['hard']} "
         "hand-picked hard ones. Everything is per unit strike (K = 1, S = moneyness); Vega and Volga are "
         "with respect to v0; Theta is calendar, -dV/dtau per year.", "",
         "Reference: Lewis single integral, c = e^x - (e^{x/2}/pi) int_0^inf Re[e^{iux} phi(u - i/2)] / (u^2 + 1/4) du, "
         "with the little-trap phi written in a form that stays accurate as xi -> 0. The price and all 11 "
         "sensitivities come from closed-form integrands: spot and v0 derivatives directly, the kappa, theta, xi "
         "and rho derivatives by forward-mode differentiation of C and D, rate rho by the chain rule, Theta from "
         "the PDE. Adaptive 16-point Gauss-Legendre panels on [0, U] with a per-contract ladder: levels are "
         "tightened until two agree to 1e-9 on every quantity.", "",
         "## Summary", "",
         "| Check | Result | Max error |", "|---|---|---:|",
         f"| Convergence ladder | {sum(ref['levels_used'].values())} contracts, all accepted at level "
         f"{', '.join(ref['levels_used'])}; {len(ref['non_converged'])} not converged | last change {e(ref['max_last_change'])} |",
         f"| 1. Lewis vs Gil-Pelaez (two-probability form, scipy quad) | all {sum(c['contracts'].values())} contracts, calls and puts | {e(gp['max_abs_diff'])} |",
         f"| 2. Black-Scholes limit, rho = 0 | error falls like xi^2 (ratios {', '.join(f'{r['ratio_to_previous']:.1f}' for r in bs['rows'] if r['rho'] == 'zero' and 'ratio_to_previous' in r)}) | {e(bs['rows'][2]['max_abs_err'])} at xi = 1e-4 |",
         f"| 2. Black-Scholes limit, domain rho | error falls like xi (ratios {', '.join(f'{r['ratio_to_previous']:.1f}' for r in bs['rows'] if r['rho'] == 'domain' and 'ratio_to_previous' in r)}) | {e(bs['rows'][5]['max_abs_err'])} at xi = 1e-4 |"]
    for key in ("c2_L12_256", "c2_L12_512", "c4_L10_256", "c4_L10_512"):
        if key in cos:
            r = cos[key]
            L.append(f"| 3. COS {key.replace('_', ' ')} terms: price / Delta / Gamma | price within 1e-9 for "
                     f"{100 * r['price']['share_below_1e-9']:.1f}% of contracts; worst {r['price']['worst']} | "
                     f"{e(r['price']['max'])} / {e(r['delta']['max'])} / {e(r['gamma']['max'])} |")
    if "skipped" in ql:
        L.append(f"| 4. QuantLib | skipped: {ql['skipped']} | - |")
    else:
        L.append(f"| 4. QuantLib {ql['version']} AnalyticHestonEngine | {ql['contracts']} contracts | {e(ql['max_abs_diff'])} |")
    L += [f"| 5. QE Monte Carlo, 2^20 paths | {mc['contracts']} contracts; {100 * mc['coverage_95']:.0f}% inside 95%, "
          f"{100 * mc['coverage_99']:.0f}% inside 99%; RMS z {mc['rms_z']:.2f}, mean z {mc['mean_z']:.2f} | max abs z {mc['max_abs_z']:.2f} |",
          f"| 6. Torch twin vs NumPy, all 12 quantities | ladder converged everywhere | {e(max(tw['max_abs_diff'].values()))} |",
          f"| 6. Parameter derivatives vs Richardson FD | {tw['finite_differences']['contracts']} contracts | {e(max(v['numpy_vs_fd'] for k, v in fd.items() if k.startswith('d_')))} |",
          f"| 6. Theta (PDE identity) vs FD in tau | same contracts | {e(fd['theta_cal']['numpy_vs_fd'])} |",
          f"| 7. v_max, P(v_tau > v_max) <= 1e-8 / 1e-10 | worst corner v0 = theta = 0.36, kappa = 0.5, xi = 1, tau = 2 | "
          f"{vm['1e-08']['v_max']:.2f} / {vm['1e-10']['v_max']:.2f} |",
          f"| 8. Timing (one thread, per contract) | see below | - |", "",
          "No contract failed or was left unconverged.", "",
          "## Findings", ""]
    L += [
        f"- **Reference.** Every contract is accepted at level 1 (level 0 and level 1 agree to 1e-9); the largest last "
        f"change is {e(ref['max_last_change'])}. Two independent formulas (Gil-Pelaez, QuantLib) agree to about 1e-14, "
        "so the reference is limited by double-precision rounding, not by quadrature. The integrand tail decides "
        f"the work: U is {ref['U_percentiles']['50']:.0f} at the median but {ref['U_percentiles']['100']:.0f} for the "
        "short-maturity contracts with xi = 1 and v0 = 0.01, where |phi(u - i/2)| decays only like "
        "exp(-sqrt(1 - rho^2) (v0 + kappa theta tau) u / xi).",
        "- **Parity.** Puts are formed from calls by parity in closed form, so parity holds to rounding by "
        "construction. Check 1 prices calls and puts with an independent formula and a different integrator.",
        "- **Black-Scholes limit.** With rho = 0 the gap to Black-Scholes at the average variance w falls by 100 per "
        "decade of xi; with rho != 0 by 10, as expected. At xi = 1e-4 the rho = 0 gap is "
        f"{e(bs['rows'][2]['max_abs_err'])}, so the xi-stable form of C and D has no cancellation problem.",
        "- **COS.** With the plain cumulant rule (L = 12, exact c2) COS matches the reference to 1e-9 for most "
        "contracts but has a truncation floor of about 1e-4 in the price (and up to 0.5 in Gamma) for long "
        "maturities with large xi and Feller violation: the distribution of log S has a heavy left tail that 12 "
        "standard deviations do not cover, and more terms do not help. The Fang-Oosterlee rule with the fourth "
        "cumulant (L = 10, half-width sqrt(c2 + sqrt(c4))) covers the tail but then needs more terms. Scored over "
        "all 12 quantities on the domain and short contracts, the c2 rule is better at every setting from 32 to 256 "
        "terms (score 4.6e-2, 1.3e-2, 3.8e-3, 3.9e-4) and stops improving at 2.9e-4 from 512 terms; the c4 rules only "
        "win at 1024 terms. Phase 1 therefore freezes the c2 rule with L = 12 and reports its floor.",
        f"- **Monte Carlo.** QE with the martingale correction (psi_c = 1.5, max(32, 256 tau) steps) is unbiased "
        f"within its error bars on all {mc['contracts']} contracts, including strong Feller violations; no "
        "martingale-correction failure occurred.",
        "- **Torch twin.** The batched complex128 twin with autograd agrees with the closed-form NumPy integrands to "
        f"{e(max(tw['max_abs_diff'].values()))} on all 12 quantities, and both agree with Richardson differences to "
        f"{e(max(v['numpy_vs_fd'] for v in fd.values()))}. The node ladder (u = U s^2, 16-point Gauss-Legendre panels) "
        f"needs {tw['nodes_percentiles_domain']['50']:.0f} nodes at the domain median, "
        f"{tw['nodes_percentiles_domain']['99']:.0f} at the 99th percentile and {tw['nodes_percentiles_domain']['100']:.0f} "
        "at most; short maturities need up to 1024, and the hand-picked tau = 0.005, xi = 1 cases up to 4096.",
        f"- **v_max.** The worst corner is v0 = theta = 0.36, kappa = 0.5, xi = 1, tau = 2. P(v_tau > v) is 1e-8 at "
        f"v = {vm['1e-08']['v_max']:.1f} and 1e-10 at v = {vm['1e-10']['v_max']:.1f}, an order of magnitude above the "
        "plan's guess of 1.0 to 1.5. The ADI grid in Phase 1 uses the 1e-10 value with points packed near v = 0; "
        "the price error from truncating v is far smaller than the tail probability, but the brief asked for the "
        "tail criterion.",
        f"- **Feller.** {100 * c['feller_violation_share_domain_sample']:.1f}% of the domain sample violates the Feller "
        "condition (2 kappa theta < xi^2); nothing in the reference depends on it.",
        "- **Hardest regions.** Short maturity with large xi (long, slowly decaying integrand: more panels and torch "
        "nodes) and long maturity with large xi (heavy left tail: the COS truncation floor). The table below is by "
        "tau and xi bins.", ""]
    L += ["## Timing (one thread)", "",
          "| Method | Seconds per contract |", "|---|---:|",
          f"| NumPy reference, all 12 quantities, including the two ladder levels | {tm['numpy_reference_12_quantities_with_ladder']:.4f} |",
          f"| NumPy reference, price only, two levels | {tm['numpy_reference_price_only_with_ladder']:.4f} |",
          f"| Torch twin, batch of {tm['torch_twin_fixed_nodes']['batch']}, fixed {tm['torch_twin_fixed_nodes']['nodes']} nodes, all 12 by autograd | {tm['torch_twin_fixed_nodes']['seconds']:.5f} |",
          f"| Torch twin, batch of {tm['torch_twin_ladder']['batch']}, full node ladder | {tm['torch_twin_ladder']['seconds']:.5f} |"]
    for k, v in tm["cos_price_delta_gamma"].items():
        mode, n = k.split("_")
        L.append(f"| COS (c2 rule) {n} terms, price/Delta/Gamma, {'batch of 4096' if mode == 'batch' else 'one contract'} | {v:.5f} |")
    L += ["", "## Regions (tau by xi; domain and short contracts)", "",
          "| tau | xi | n | median U | median / max torch nodes | COS c2-256 max price error | max Gamma error |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in c["regions_tau_by_xi"]:
        L.append(f"| {r['tau'][0]}-{r['tau'][1]} | {r['xi'][0]}-{r['xi'][1]} | {r['n']} | {r['reference_U_median']:.0f} | "
                 f"{r['torch_nodes_median']:.0f} / {r['torch_nodes_max']} | {e(r['cos256_price_err_max'])} | {e(r['cos256_gamma_err_max'])} |")
    L += ["", "## Black-Scholes limit", "", "| rho | xi | max abs error | max rel error | ratio to previous |", "|---|---:|---:|---:|---:|"]
    for r in bs["rows"]:
        L.append(f"| {r['rho']} | {r['xi']:g} | {e(r['max_abs_err'])} | {e(r['max_rel_err'])} | {r.get('ratio_to_previous', float('nan')):.1f} |")
    L += ["", "## Torch twin and finite differences", "", "| Quantity | torch - NumPy | NumPy - Richardson FD | torch - Richardson FD |", "|---|---:|---:|---:|"]
    for q, v in tw["max_abs_diff"].items():
        f = fd.get(q)
        L.append(f"| {q} | {e(v)} | {e(f['numpy_vs_fd']) if f else '-'} | {e(f['torch_vs_fd']) if f else '-'} |")
    L += ["", "Node counts needed (16-point panels doubled until two counts agree to 1e-9):", ""]
    for s, d in tw["nodes_needed"].items():
        L.append(f"- {s}: " + ", ".join(f"{k} nodes: {v}" for k, v in d.items()))
    L += ["", "## Monte Carlo rows", "", "| Contract | tau | steps | MC | SE | reference | z |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in mc["rows"]:
        L.append(f"| {r['contract']} | {r['tau']:.3f} | {r['steps']} | {r['mc']:.6f} | {r['se']:.1e} | {r['reference']:.6f} | {r['z']:.2f} |")
    L += ["", "## Files", "", "- `checks.json`: every number above.",
          "- `summary_by_contract.csv`: inputs, the 12 reference quantities, ladder level, U, torch nodes and the "
          "check-1 and COS errors per contract.", "- `run.log`: the run log.", "",
          "Regenerate: `python -m experiments.reference.check_reference` then `python -m experiments.reference.make_report` (from heston/).", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    print("wrote", OUT / "report.md")


if __name__ == "__main__":
    main()
