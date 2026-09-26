"""Supervised price and Greek losses and staged training for the anchored
PINN ablation.

The variants share one architecture, one dataset and one optimizer
schedule, and differ only in which loss terms they may receive:

``data_only``     price and Theta supervision (plus the shared hard
                  spatial boundaries the architecture always imposes);
``physics_only``  PDE residual and terminal payoff only (no labels);
``hybrid``        price, Theta, PDE and terminal;
``v2``            hybrid plus sparse Delta and Gamma supervision.

Weight handling: exactly zero disables a term, positive weights are
clipped into ``[weight_min, weight_max]``, and negative or non-finite
weights raise a configuration error. A stage's active terms are the
intersection of the variant's permitted terms and its positive stage
weights; a stage with no active terms is skipped rather than silently
optimising an all-zero objective, and disabled terms incur no derivative
work.

Checkpoint selection uses an immutable, per-variant validation objective
declared in the settings, evaluated with the same normalisation scales for
every candidate. Stage training losses are logged separately because they
are not comparable across stages.
"""

import math
from time import perf_counter

import numpy as np
import torch

from src.neural.memory import MemoryTracker


MODES = {
    "data_only": ("price", "theta"),
    "physics_only": ("pde", "terminal"),
    "hybrid": ("price", "theta", "pde", "terminal"),
    "v2": ("price", "theta", "delta", "gamma", "pde", "terminal"),
    # Controls that isolate the PDE term: identical supervised labels to
    # their counterpart, but no PDE residual.
    "supervised_hybrid_control": ("price", "theta", "terminal"),
    "supervised_v2_control": ("price", "theta", "delta", "gamma", "terminal"),
}


def bound_weight(value, settings):
    """Zero disables a term; positive weights are clipped into the band."""
    if not math.isfinite(value):
        raise ValueError(f"loss weight must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"loss weight must be non-negative, got {value!r}")
    if value == 0:
        return 0.0
    return float(min(max(value, settings["weight_min"]),
                     settings["weight_max"]))


def stage_active_terms(mode, stage_weights, settings):
    """Terms enabled by both the variant and the stage's positive weights."""
    return tuple(
        name for name in MODES[mode]
        if bound_weight(stage_weights.get(name, 0.0), settings) > 0.0
    )


def tensor(array, dtype, requires_grad=False):
    value = torch.tensor(np.asarray(array), dtype=dtype).reshape(-1, 1)
    return value.requires_grad_(requires_grad)


def quantities(model, y, tau, need):
    """Model price and only the Greeks the active terms require."""
    value = model(y, tau)
    result = {"price": value}
    if not (need & {"delta", "gamma", "theta"}):
        return result
    ones = torch.ones_like(value)
    value_y = torch.autograd.grad(
        value, y, ones, create_graph=True, retain_graph=True)[0]
    x = torch.exp(y)
    result["delta"] = value_y / x
    if "theta" in need:
        value_tau = torch.autograd.grad(
            value, tau, ones, create_graph=True, retain_graph=True)[0]
        result["theta"] = -value_tau
    if "gamma" in need:
        value_yy = torch.autograd.grad(
            value_y, y, torch.ones_like(value_y),
            create_graph=True, retain_graph=True)[0]
        result["gamma"] = (value_yy - value_y) / (model.problem.K * x**2)
    return result


def training_batches(dataset, dtype):
    return {
        "anchor": (tensor(dataset["anchor_y"], dtype, True),
                   tensor(dataset["anchor_tau"], dtype, True),
                   tensor(dataset["anchor_price"], dtype)),
        "greek": (tensor(dataset["greek_y"], dtype, True),
                  tensor(dataset["greek_tau"], dtype, True),
                  tensor(dataset["greek_delta"], dtype),
                  tensor(dataset["greek_gamma"], dtype),
                  tensor(dataset["greek_theta"], dtype)),
        "pde": (tensor(dataset["pde_y"], dtype, True),
                tensor(dataset["pde_tau"], dtype, True)),
        "terminal": (tensor(dataset["terminal_y"], dtype, True),
                     tensor(dataset["terminal_tau"], dtype, True),
                     tensor(dataset["terminal_payoff"], dtype)),
    }


def validation_batches(dataset, dtype):
    """Validation batches, including *independent* PDE and terminal samples."""
    return {
        "anchor": (tensor(dataset["validation_y"], dtype, True),
                   tensor(dataset["validation_tau"], dtype, True),
                   tensor(dataset["validation_price"], dtype)),
        "greek": (tensor(dataset["validation_y"], dtype, True),
                  tensor(dataset["validation_tau"], dtype, True),
                  tensor(dataset["validation_delta"], dtype),
                  tensor(dataset["validation_gamma"], dtype),
                  tensor(dataset["validation_theta"], dtype)),
        "pde": (tensor(dataset["validation_pde_y"], dtype, True),
                tensor(dataset["validation_pde_tau"], dtype, True)),
        "terminal": (tensor(dataset["validation_terminal_y"], dtype, True),
                     tensor(dataset["validation_terminal_tau"], dtype, True),
                     tensor(dataset["validation_terminal_payoff"], dtype)),
    }


def term_losses(model, batches, denominators, active):
    """Per-term losses for exactly the ``active`` terms."""
    losses = {}
    greek_need = set(active) & {"theta", "delta", "gamma"}
    if greek_need:
        gy, gtau, gdelta, ggamma, gtheta = batches["greek"]
        greek = quantities(model, gy, gtau, greek_need)
        if "theta" in greek_need:
            losses["theta"] = torch.mean(
                (greek["theta"] - gtheta) ** 2) / denominators["theta"]
        if "delta" in greek_need:
            losses["delta"] = torch.mean(
                (greek["delta"] - gdelta) ** 2) / denominators["delta"]
        if "gamma" in greek_need:
            losses["gamma"] = torch.mean(
                (greek["gamma"] - ggamma) ** 2) / denominators["gamma"]
    if "price" in active:
        ay, atau, aprice = batches["anchor"]
        losses["price"] = torch.mean(
            (model(ay, atau) - aprice) ** 2) / denominators["price"]
    if "pde" in active:
        py, ptau = batches["pde"]
        losses["pde"] = torch.mean(model.pde_residual(py, ptau) ** 2)
    if "terminal" in active:
        ty, ttau, tpayoff = batches["terminal"]
        losses["terminal"] = torch.mean((model(ty, ttau) - tpayoff) ** 2)
    return losses


def weighted_total(losses, weights):
    """Weighted sum over exactly the terms present in ``losses``."""
    total = None
    for name, value in losses.items():
        term = weights[name] * value
        total = term if total is None else total + term
    return total


class AnchoredTrainer:
    """Staged Adam + L-BFGS training with a fixed selection objective."""

    def __init__(self, model, dataset, scales, settings, mode, seed,
                 dtype=torch.float64):
        self.model = model
        self.settings = settings
        self.mode = mode
        self.seed = seed
        self.permitted = MODES[mode]
        self.batches = training_batches(dataset, dtype)
        self.valid = validation_batches(dataset, dtype)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.denominators = {
            name: scales[name] ** 2 + settings["scale_epsilon"]
            for name in scales
        }
        self.selection_weights = {}
        for name, weight in settings["selection"][mode].items():
            bounded = bound_weight(weight, settings)
            if bounded > 0.0:
                self.selection_weights[name] = bounded
        self.selection_terms = tuple(
            name for name in self.permitted if name in self.selection_weights)
        if not self.selection_terms:
            raise ValueError(
                f"selection objective for '{mode}' shares no term with its "
                f"permitted terms {self.permitted}")

    def _snapshot(self):
        return {key: value.detach().clone()
                for key, value in self.model.state_dict().items()}

    def selection_score(self):
        """Fixed validation objective, independent of stage weights."""
        losses = term_losses(self.model, self.valid, self.denominators,
                             self.selection_terms)
        total = weighted_total(losses, self.selection_weights)
        return (float(total.detach()),
                {name: float(value.detach()) for name, value in losses.items()})

    def _row(self, stage, phase, step, start, **extra):
        score, terms = self.selection_score()
        row = {"stage": stage, "phase": phase, "step": step,
               "selection_score": score,
               "elapsed_seconds": perf_counter() - start, **extra}
        for name, value in terms.items():
            row[f"validation_{name}"] = value
        return row, score

    def _score_of(self, state):
        """Selection score of an arbitrary state (restores afterwards)."""
        current = self._snapshot()
        self.model.load_state_dict(state)
        score, _ = self.selection_score()
        self.model.load_state_dict(current)
        return score

    def train(self, stages, wall_clock_cap=None, log=print):
        """Staged Adam + L-BFGS. Returns (history, metadata, states)."""
        history = []
        start = perf_counter()
        memory = MemoryTracker(interval=0.1).start()
        optimizer = torch.optim.Adam(
            self.params, lr=self.settings["adam_learning_rate"])
        optimizer_steps = 0
        hit_cap = False
        lbfgs_interrupted = False
        stage_active = {}

        row, score = self._row("initial", "selection", 0, start)
        history.append(row)
        best = {"score": score, "state": self._snapshot(), "label": "initial"}

        for stage in stages:
            if hit_cap:
                break
            active = stage_active_terms(self.mode, stage["weights"], self.settings)
            stage_active[stage["name"]] = list(active)
            if not active:
                log(f"[{self.mode}] skipping stage '{stage['name']}': "
                    "no active terms for this variant")
                continue
            weights = {name: bound_weight(stage["weights"].get(name, 0.0),
                                          self.settings) for name in active}
            for epoch in range(1, stage["adam_epochs"] + 1):
                optimizer.zero_grad(set_to_none=True)
                losses = term_losses(self.model, self.batches,
                                     self.denominators, active)
                total = weighted_total(losses, weights)
                total.backward()
                optimizer.step()
                optimizer_steps += 1
                if epoch % self.settings["validation_every"] == 0:
                    row, score = self._row(
                        stage["name"], "adam", epoch, start,
                        stage_loss=float(total.detach()))
                    for name, value in losses.items():
                        row[f"train_{name}"] = float(value.detach())
                    history.append(row)
                    if score < best["score"]:
                        best = {"score": score, "state": self._snapshot(),
                                "label": f"{stage['name']}_adam"}
                if (wall_clock_cap is not None
                        and perf_counter() - start >= wall_clock_cap):
                    hit_cap = True
                    break
            max_steps = int(stage.get("lbfgs_iterations", 0))
            if max_steps > 0 and not hit_cap:
                lbfgs = torch.optim.LBFGS(
                    self.params, max_iter=1,
                    history_size=self.settings["lbfgs_history_size"],
                    line_search_fn="strong_wolfe",
                    tolerance_grad=self.settings["lbfgs_tolerance_grad"],
                    tolerance_change=self.settings["lbfgs_tolerance_change"])

                counters = {"closures": 0}

                def closure():
                    # A line-search trial point, never a checkpoint candidate.
                    counters["closures"] += 1
                    lbfgs.zero_grad(set_to_none=True)
                    losses = term_losses(self.model, self.batches,
                                         self.denominators, active)
                    value = weighted_total(losses, weights)
                    value.backward()
                    return value

                def training_objective():
                    """Stage training objective, for logging only."""
                    with torch.enable_grad():
                        losses = term_losses(self.model, self.batches,
                                             self.denominators, active)
                        value = weighted_total(losses, weights)
                        return float(value.detach())

                steps = 0
                # Validate periodically *inside* the loop, so an
                # intermediate improvement is not missed when the loop
                # runs long or is interrupted at the cap.
                validate_every = int(self.settings.get("lbfgs_validation_every", 10))
                while steps < max_steps:
                    # Checked between completed steps, so an interrupted
                    # budget always leaves an accepted parameter state.
                    if (wall_clock_cap is not None
                            and perf_counter() - start >= wall_clock_cap):
                        hit_cap = True
                        lbfgs_interrupted = True
                        break
                    lbfgs.step(closure)
                    steps += 1
                    if steps % validate_every == 0 or steps == max_steps:
                        row, score = self._row(
                            stage["name"], "lbfgs", steps, start,
                            stage_loss=training_objective(),
                            closure_calls=counters["closures"],
                            accepted_steps=steps)
                        history.append(row)
                        if score < best["score"]:
                            best = {"score": score, "state": self._snapshot(),
                                    "label": f"{stage['name']}_lbfgs_step{steps}"}
                optimizer_steps += steps
                row, score = self._row(
                    stage["name"], "lbfgs", steps, start,
                    stage_loss=training_objective(),
                    closure_calls=counters["closures"], accepted_steps=steps)
                history.append(row)
                if score < best["score"]:
                    best = {"score": score, "state": self._snapshot(),
                            "label": f"{stage['name']}_lbfgs"}
            row, score = self._row(stage["name"], "stage_end", None, start)
            history.append(row)
            if score < best["score"]:
                best = {"score": score, "state": self._snapshot(),
                        "label": f"{stage['name']}_end"}

        last_state = self._snapshot()
        last_score = self._score_of(last_state)
        self.model.load_state_dict(best["state"])
        final_score, final_terms = self.selection_score()
        history.append({"stage": "final", "phase": "selected", "step": None,
                        "selection_score": final_score,
                        "elapsed_seconds": perf_counter() - start,
                        **{f"validation_{k}": v for k, v in final_terms.items()}})
        memory.stop()
        metadata = {
            "model": self.mode,
            "seed": self.seed,
            "permitted_terms": list(self.permitted),
            "stage_active_terms": stage_active,
            "selection_terms": list(self.selection_terms),
            "selection_weights": self.selection_weights,
            "selected_label": best["label"],
            "selected_selection_score": best["score"],
            "last_selection_score": last_score,
            "final_selection_score": final_score,
            "optimizer_steps": optimizer_steps,
            "wall_seconds": perf_counter() - start,
            "hit_wall_cap": hit_cap,
            "lbfgs_interrupted": lbfgs_interrupted,
            "wall_clock_cap_seconds": wall_clock_cap,
            "parameter_count": sum(p.numel() for p in self.params),
            "baseline_rss_mb": memory.baseline_mb,
            "peak_rss_mb": memory.peak_mb,
            "incremental_peak_mb": memory.incremental_peak_mb,
        }
        return (history, metadata,
                {"selected": self._snapshot(), "last": last_state})


# ---PART-4---