"""Matched-budget training variants for the fixed Black--Scholes PINN.

All variants share the network architecture and output construction of
``FixedBlackScholesPINN`` so that comparisons isolate the training
method. Three variants are provided:

``baseline_matched``
    The original physics-only recipe (PDE residual plus terminal
    payoff) under the shared reduced budget.

``constrained``
    Adds one-sided financial constraints (price bounds, Delta bounds,
    and the log-moneyness convexity condition u_yy - u_y >= 0) as
    penalties. Category weights are rebalanced by gradient-norm
    annealing, individual points are weighted by their violation size,
    and some constraint batches are drawn from regions where violations
    were recently largest.

``causal``
    Physics-only training with causal bin weights over time-to-maturity,
    in the natural forward-tau order starting from the terminal payoff.

Penalties are evaluated at sampled points; they encourage but do not
mathematically guarantee the constraints between sample points.
"""

from time import perf_counter

import torch

from src.neural.fixed_pinn import sample_collocation, sample_terminal
from src.neural.memory import MemoryTracker


CONSTRAINT_CATEGORIES = ("price_bounds", "delta_bounds", "gamma")


def compute_constraint_violations(model, y, tau, create_graph=True):
    """Differentiable positive-part violations at (y, tau) points.

    Coordinates follow the model: y = log(S/K) and u = C/K. The
    constraints in these coordinates are

        relu(x e^{-q tau} - e^{-r tau}, 0) <= u <= x e^{-q tau},
        0 <= u_y e^{-y} <= e^{-q tau},
        u_yy - u_y >= 0,

    where x = e^y. The last condition is the convexity of C in S
    transformed to log-moneyness coordinates; it is not plain convexity
    of u in y. Set ``create_graph=False`` for detached evaluation, for
    example when scoring candidate points for guided sampling.
    """
    problem = model.problem
    value, value_y, value_yy, _ = model.derivatives(
        y, tau, create_graph=create_graph
    )
    x = torch.exp(y)
    discount_q = torch.exp(-problem.q * tau)
    discount_r = torch.exp(-problem.r * tau)
    lower = torch.relu(x * discount_q - discount_r)
    upper = x * discount_q
    delta = value_y / x
    convexity_quantity = value_yy - value_y
    return {
        "price_below_intrinsic": torch.relu(lower - value),
        "price_above_discounted_spot": torch.relu(value - upper),
        "delta_below_zero": torch.relu(-delta),
        "delta_above_discount_factor": torch.relu(delta - discount_q),
        "gamma_negative": torch.relu(-convexity_quantity),
    }


def causal_bin_weights(bin_losses, epsilon):
    """Causal training weights over ordered time bins.

    Bins are ordered forward in time-to-maturity (bin 0 at the terminal
    payoff). The weight of bin i is ``exp(-epsilon * c_i)`` where ``c_i``
    is the accumulated loss of all earlier bins divided by the *total*
    bin loss, so ``c_i`` lies in [0, 1) and ``epsilon`` is scale-free:
    the same epsilon gives the same concentration regardless of the raw
    residual magnitude, which drifts over training. Weights are
    normalised to mean one. Early bins are emphasised until the
    accumulated residual there becomes small.
    """
    detached = bin_losses.detach()
    cumulative = torch.cumsum(detached, dim=0) - detached
    scaled = cumulative / (torch.sum(detached) + 1e-12)
    weights = torch.exp(-epsilon * scaled)
    weights = weights / torch.mean(weights)
    return weights, cumulative


def effective_bin_count(weights):
    """Effective number of bins trained, as the participation ratio
    ``(sum w)^2 / sum(w^2)``. Uniform weights over B bins give B; a single
    dominating bin gives 1. (The earlier ``1 / sum(w^2)`` diagnostic gave
    ``1/B`` for uniform weights, i.e. it was inverted.)"""
    return float(torch.sum(weights)**2 / torch.sum(weights**2))


def category_gradient_norm(loss, params):
    """Mean absolute gradient entry of one loss category (stop-gradient)."""
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True
    )
    entries = [g.abs().mean() for g in grads if g is not None]
    if not entries:
        return torch.tensor(0.0)
    return torch.stack(entries).mean()


def update_category_weights(weights, losses, ema, minimum, maximum):
    """Balance the constraint categories against one another.

    The reference is the *mean of the constraint-category losses*, not the
    PDE loss, so the balancing stays inside the constraint group and each
    category is pulled toward an equal share of the constraint
    contribution: a category whose raw loss is large is scaled down and a
    small one scaled up.

    Weights are clipped to ``[minimum, maximum]``. The floor keeps Delta
    and Gamma from collapsing toward zero; the ceiling stops the price
    bounds -- whose violation loss is typically tiny next to the Delta
    violation loss -- from being amplified without limit and dominating
    the constraint budget (the previous ceiling of 100 pinned the price
    weight at ~98).

    Returns the updated weights and the detached loss magnitudes used for
    logging.
    """
    magnitudes = {name: float(value.detach()) for name, value in losses.items()}
    reference = sum(magnitudes[name] for name in weights) / max(len(weights), 1)
    updated = dict(weights)
    for name in weights:
        if magnitudes[name] <= 0.0 or reference <= 0.0:
            continue
        target = min(max(reference / magnitudes[name], minimum), maximum)
        updated[name] = ema * weights[name] + (1.0 - ema) * target
    return updated, magnitudes


def violation_point_weights(violation_total, eta):
    """Per-point weights proportional to recent violation size, mean one."""
    detached = violation_total.detach()
    mean = torch.mean(detached)
    weights = 1.0 + eta * detached / (mean + 1e-12)
    return weights / torch.mean(weights)


def violation_total_magnitude(violations):
    """Sum of every positive-part violation, detached, for sampling."""
    total = None
    for magnitudes in violations.values():
        term = magnitudes.detach()
        total = term if total is None else total + term
    return total


class VariantTrainer:
    """Shared Adam training loop for all study variants.

    The loop reproduces the fixed-PINN recipe (fresh Sobol collocation
    and terminal batches per epoch, cosine learning-rate annealing) so
    that the only differences between variants are the loss terms.
    An optional wall-clock cap keeps the matched budget honest when a
    variant is slower per epoch.
    """

    def __init__(self, model, budget, variant, variant_settings, seed,
                 causal_settings=None, dtype=torch.float64):
        self.model = model
        self.problem = model.problem
        self.budget = budget
        self.variant = variant
        self.variant_settings = variant_settings
        self.seed = seed
        self.causal_settings = causal_settings or {}
        self.dtype = dtype
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            self.params, lr=budget["adam_learning_rate"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=budget["adam_epochs"],
            eta_min=budget["adam_learning_rate"] * 0.01,
        )
        self.category_weights = {name: 1.0 for name in CONSTRAINT_CATEGORIES}
        self.category_norms = {}

    def _sample_batches(self, epoch):
        budget = self.budget
        x, tau = sample_collocation(
            self.problem, budget["adam_batch_size"], self.seed + epoch,
            budget["near_strike_fraction"], budget["near_strike_half_width"],
            budget["minimum_tau"], self.dtype,
        )
        terminal_x, terminal_tau, payoff = sample_terminal(
            self.problem, budget["terminal_batch_size"], self.seed + 50000 + epoch,
            budget["near_strike_fraction"], budget["near_strike_half_width"],
            self.dtype,
        )
        return x, tau, terminal_x, terminal_tau, payoff

    def _causal_pde_loss(self, residual_sq, epoch):
        settings = self.causal_settings
        bins = settings["causal_bins"]
        epochs = self.budget["adam_epochs"]
        anneal_fraction = settings.get("epsilon_anneal_fraction", 0.75)
        progress = min(1.0, epoch / max(1.0, anneal_fraction * epochs))
        epsilon = settings["epsilon_start"] + (
            settings["epsilon_final"] - settings["epsilon_start"]
        ) * progress
        tau_order = torch.argsort(self._current_tau.reshape(-1))
        chunks = torch.chunk(tau_order, bins)
        bin_losses = torch.stack([
            torch.mean(residual_sq.reshape(-1)[chunk]) for chunk in chunks
        ])
        weights, _ = causal_bin_weights(bin_losses, epsilon)
        effective = effective_bin_count(weights)
        return torch.sum(weights * bin_losses) / torch.sum(weights), epsilon, effective

    def _constraint_losses(self, epoch):
        settings = self.variant_settings
        if (
            settings["violation_top_k"] > 0
            and epoch % settings["resample_every"] == 0
        ):
            candidates_y, candidates_tau = sample_collocation(
                self.problem, settings["violation_candidates"],
                self.seed + 70000 + epoch,
                self.budget["near_strike_fraction"],
                self.budget["near_strike_half_width"],
                self.budget["minimum_tau"], self.dtype,
            )
            candidates = compute_constraint_violations(
                self.model, candidates_y, candidates_tau, create_graph=False
            )
            totals = violation_total_magnitude(candidates).reshape(-1)
            top = torch.topk(
                totals, min(settings["violation_top_k"], totals.numel())
            ).indices
            y = candidates_y[top].detach().requires_grad_(True)
            tau = candidates_tau[top].detach().requires_grad_(True)
            source = "violation_guided"
        else:
            y, tau = sample_collocation(
                self.problem, settings["constraint_points"],
                self.seed + 90000 + epoch,
                self.budget["near_strike_fraction"],
                self.budget["near_strike_half_width"],
                self.budget["minimum_tau"], self.dtype,
            )
            source = "sobol"
        violations = compute_constraint_violations(self.model, y, tau)
        point_weights = violation_point_weights(
            violation_total_magnitude(violations),
            settings["point_weight_eta"],
        )
        weighted = {
            name: point_weights * magnitudes.reshape(-1)**2
            for name, magnitudes in violations.items()
        }
        losses = {
            "price_bounds": torch.mean(
                weighted["price_below_intrinsic"]
                + weighted["price_above_discounted_spot"]
            ),
            "delta_bounds": torch.mean(
                weighted["delta_below_zero"] + weighted["delta_above_discount_factor"]
            ),
            "gamma": torch.mean(weighted["gamma_negative"]),
        }
        return losses, source

    def train(self, wall_clock_cap=None, log_every=20, log=print):
        """Run the Adam loop; returns (history rows, metadata)."""
        budget = self.budget
        history = []
        start = perf_counter()
        hit_wall_cap = False
        completed = 0
        memory = MemoryTracker(interval=0.1).start()
        for epoch in range(1, budget["adam_epochs"] + 1):
            x, tau, terminal_x, terminal_tau, payoff = self._sample_batches(epoch)
            self._current_tau = tau
            self.optimizer.zero_grad(set_to_none=True)
            residual_sq = self.model.pde_residual(x, tau)**2
            if self.variant == "causal":
                pde_loss, epsilon, effective_bins = self._causal_pde_loss(
                    residual_sq, epoch
                )
            else:
                pde_loss = torch.mean(residual_sq)
                epsilon = None
                effective_bins = None
            terminal_loss = torch.mean(
                (self.model(terminal_x, terminal_tau) - payoff)**2
            )
            row = {
                "phase": "adam", "step": epoch,
                "pde_loss": float(pde_loss.detach()),
                "terminal_loss": float(terminal_loss.detach()),
                "category_weight_price_bounds": self.category_weights["price_bounds"],
                "category_weight_delta_bounds": self.category_weights["delta_bounds"],
                "category_weight_gamma": self.category_weights["gamma"],
                "causal_epsilon": epsilon,
                "causal_effective_bins": effective_bins,
            }
            if self.variant == "constrained":
                constraint_losses, constraint_source = self._constraint_losses(epoch)
                if (
                    epoch % self.variant_settings["balance_every"] == 0
                    and epoch > 1
                ):
                    self.category_weights, self.category_norms = update_category_weights(
                        self.category_weights, constraint_losses,
                        self.variant_settings["balance_ema"],
                        self.variant_settings["category_weight_min"],
                        self.variant_settings["category_weight_max"],
                    )
                constraint_total = sum(
                    self.variant_settings["constraint_weight"]
                    * self.category_weights[name] * loss
                    for name, loss in constraint_losses.items()
                )
                loss = (
                    pde_loss
                    + budget["terminal_loss_weight"] * terminal_loss
                    + constraint_total
                )
                row["constraint_source"] = constraint_source
                for name, value in constraint_losses.items():
                    row[f"constraint_loss_{name}"] = float(value.detach())
                row["constraint_total"] = float(constraint_total.detach())
            else:
                loss = pde_loss + budget["terminal_loss_weight"] * terminal_loss
            row["loss"] = float(loss.detach())
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            completed = epoch
            row["elapsed_seconds"] = perf_counter() - start
            if epoch == 1 or epoch % log_every == 0:
                history.append(row)
                log(
                    f"[{self.variant}] epoch {epoch}/{budget['adam_epochs']} "
                    f"loss={row['loss']:.3e} pde={row['pde_loss']:.3e} "
                    f"terminal={row['terminal_loss']:.3e} "
                    f"elapsed={row['elapsed_seconds']:.1f}s"
                )
            if wall_clock_cap is not None and row["elapsed_seconds"] >= wall_clock_cap:
                hit_wall_cap = True
                log(f"[{self.variant}] wall-clock cap reached at epoch {epoch}")
                break
        memory.stop()
        metadata = {
            "variant": self.variant,
            "seed": self.seed,
            "adam_epochs_configured": budget["adam_epochs"],
            "adam_epochs_completed": completed,
            "hit_wall_cap": hit_wall_cap,
            "wall_seconds": perf_counter() - start,
            "parameter_count": sum(p.numel() for p in self.params),
            "final_training_loss": history[-1]["loss"] if history else None,
            "baseline_rss_mb": memory.baseline_mb,
            "peak_rss_mb": memory.peak_mb,
            "incremental_peak_mb": memory.incremental_peak_mb,
        }
        return history, metadata

    def _refinement_loss(self, pde_y, pde_tau, terminal_x, terminal_tau,
                         payoff, constraint_y, constraint_tau):
        """Full-batch objective matching the variant's own training loss.

        PDE and terminal points are fixed; causal variants use causal bin
        weights at the final epsilon, and constrained variants add the
        constraint penalties with the category weights frozen at their
        current (post-Adam) values.
        """
        budget = self.budget
        residual_sq = self.model.pde_residual(pde_y, pde_tau)**2
        if self.variant == "causal":
            pde_loss, _, _ = self._causal_pde_loss(residual_sq, budget["adam_epochs"])
        else:
            pde_loss = torch.mean(residual_sq)
        terminal_loss = torch.mean(
            (self.model(terminal_x, terminal_tau) - payoff)**2
        )
        loss = pde_loss + budget["terminal_loss_weight"] * terminal_loss
        components = {"pde": float(pde_loss.detach()),
                      "terminal": float(terminal_loss.detach())}
        if self.variant == "constrained":
            violations = compute_constraint_violations(
                self.model, constraint_y, constraint_tau
            )
            point_weights = violation_point_weights(
                violation_total_magnitude(violations),
                self.variant_settings["point_weight_eta"],
            )
            weighted = {
                name: point_weights * magnitudes.reshape(-1)**2
                for name, magnitudes in violations.items()
            }
            constraint_losses = {
                "price_bounds": torch.mean(
                    weighted["price_below_intrinsic"]
                    + weighted["price_above_discounted_spot"]
                ),
                "delta_bounds": torch.mean(
                    weighted["delta_below_zero"]
                    + weighted["delta_above_discount_factor"]
                ),
                "gamma": torch.mean(weighted["gamma_negative"]),
            }
            loss = loss + sum(
                self.variant_settings["constraint_weight"]
                * self.category_weights[name] * value
                for name, value in constraint_losses.items()
            )
            for name, value in constraint_losses.items():
                components[name] = float(value.detach())
        return loss, components

    def refine_lbfgs(self, refinement, log=print):
        """Full-batch L-BFGS refinement over the variant's own objective.

        ``refinement`` carries ``pde_points``, ``terminal_points`` and an
        ``lbfgs`` config (``max_iter``, ``history_size``,
        ``tolerance_grad``, ``tolerance_change``). A wall-clock cap stops
        a stalled optimisation without biasing the comparison. Returns
        ``(history rows, metadata)``.
        """
        budget = self.budget
        seed = self.seed + 1000000
        dtype = self.dtype
        cap = refinement.get("wall_clock_seconds")
        pde_y, pde_tau = sample_collocation(
            self.problem, refinement["pde_points"], seed,
            budget["near_strike_fraction"], budget["near_strike_half_width"],
            budget["minimum_tau"], dtype,
        )
        terminal_x, terminal_tau, payoff = sample_terminal(
            self.problem, refinement["terminal_points"], seed + 1,
            budget["near_strike_fraction"], budget["near_strike_half_width"], dtype,
        )
        self._current_tau = pde_tau
        constraint_y = constraint_tau = None
        if self.variant == "constrained":
            constraint_y, constraint_tau = sample_collocation(
                self.problem, self.variant_settings["constraint_points"], seed + 2,
                budget["near_strike_fraction"], budget["near_strike_half_width"],
                budget["minimum_tau"], dtype,
            )
        config = refinement["lbfgs"]
        optimizer = torch.optim.LBFGS(
            self.params, max_iter=config["max_iter"],
            history_size=config["history_size"],
            line_search_fn="strong_wolfe",
            tolerance_grad=config["tolerance_grad"],
            tolerance_change=config["tolerance_change"],
        )

        class WallClockExceeded(Exception):
            pass

        start = perf_counter()
        memory = MemoryTracker(interval=0.1).start()
        calls = 0
        history = []

        def closure():
            nonlocal calls
            if cap is not None and perf_counter() - start >= cap:
                raise WallClockExceeded
            optimizer.zero_grad(set_to_none=True)
            loss, _ = self._refinement_loss(
                pde_y, pde_tau, terminal_x, terminal_tau, payoff,
                constraint_y, constraint_tau,
            )
            loss.backward()
            calls += 1
            return loss

        try:
            optimizer.step(closure)
        except WallClockExceeded:
            pass
        memory.stop()
        final_loss, components = self._refinement_loss(
            pde_y, pde_tau, terminal_x, terminal_tau, payoff,
            constraint_y, constraint_tau,
        )
        history.append({"phase": "lbfgs", "step": calls, "loss": float(final_loss.detach()),
                        **components})
        metadata = {
            "optimizer": "lbfgs",
            "closure_calls": calls,
            "wall_seconds": perf_counter() - start,
            "final_loss": float(final_loss.detach()),
            "final_components": components,
            "objective_points": {"pde": refinement["pde_points"],
                                 "terminal": refinement["terminal_points"]},
            "baseline_rss_mb": memory.baseline_mb,
            "peak_rss_mb": memory.peak_mb,
            "incremental_peak_mb": memory.incremental_peak_mb,
        }
        return history, metadata
