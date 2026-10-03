"""Labels, losses, optimisers and metrics for the neural basket pricers.

Training data live in scale-free coordinates r = (f_1, f_2, a_1, a_2, rho). Labels
are the reference normalized call c, its gradient in r and the forward-moneyness
Hessian block (c_11, c_22, c_12), all from the batched quadrature by autograd.
Loss terms are mean squared errors divided by the training-set mean square of
their labels, so the weights in ``LossWeights`` are dimensionless.

The PDE term is the scale-free two-asset Black-Scholes equation
    a_1 c_a1 + a_2 c_a2 = sum_ij rho_ij a_i a_j f_i f_j c_ij,
which follows from u_tau = 1/2 sum rho_ij sigma_i sigma_j f_i f_j u_ij for the
driftless forwards after the substitution a_i = sigma_i sqrt(tau).
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .domain import reduced_coordinates
from .reference import QUANTITIES, normalized_call, price_and_greeks, trapezoid_rule

DERIVATIVE_NAMES = ("f1", "f2", "a1", "a2", "rho")
SECOND_NAMES = ("f1f1", "f2f2", "f1f2")


def reduced_tensor(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return torch.stack(reduced_coordinates(x)[:5], dim=-1)


def reference_labels(r, order=2, chunk=4096, nodes=1001, limit=10.0):
    """Reference c, dc/dr and the (f_1, f_2) Hessian block for reduced rows r."""
    rule = trapezoid_rule(nodes, limit)
    values, firsts, seconds = [], [], []
    for start in range(0, len(r), chunk):
        rs = r[start:start+chunk].clone().requires_grad_(order > 0)
        with torch.set_grad_enabled(order > 0):
            c = normalized_call(*rs.unbind(-1), rule=rule)
            values.append(c.detach())
            if order == 0:
                continue
            gradient, = torch.autograd.grad(c.sum(), rs, create_graph=order > 1)
            firsts.append(gradient.detach())
            if order > 1:
                row_1, = torch.autograd.grad(gradient[:, 0].sum(), rs, retain_graph=True)
                row_2, = torch.autograd.grad(gradient[:, 1].sum(), rs)
                seconds.append(torch.stack((row_1[:, 0], row_2[:, 1], row_1[:, 1]), dim=-1))
    labels = {"c": torch.cat(values)}
    if firsts:
        labels["dc"] = torch.cat(firsts)
    if seconds:
        labels["d2c"] = torch.cat(seconds)
    return labels


@dataclass
class LossWeights:
    value: float = 1.0
    first: float = 0.0
    second: float = 0.0
    pde: float = 0.0


def label_scales(labels):
    scales = {"c": labels["c"].square().mean()}
    for key in ("dc", "d2c"):
        if key in labels:
            scales[key] = labels[key].square().mean(dim=0)
    return scales


def model_derivatives(model, r, order):
    r = r.clone().requires_grad_(order > 0)
    c = model.call_from_reduced(*r.unbind(-1))
    out = {"c": c}
    if order > 0:
        gradient, = torch.autograd.grad(c.sum(), r, create_graph=True)
        out["dc"] = gradient
    if order > 1:
        row_1, = torch.autograd.grad(gradient[:, 0].sum(), r, create_graph=True)
        row_2, = torch.autograd.grad(gradient[:, 1].sum(), r, create_graph=True)
        out["d2c"] = torch.stack((row_1[:, 0], row_2[:, 1], row_1[:, 1]), dim=-1)
    return out, r


def pde_residual(r, dc, d2c):
    f1, f2, a1, a2, rho = r.unbind(-1)
    left = a1*dc[:, 2]+a2*dc[:, 3]
    right = (a1*a1*f1*f1*d2c[:, 0]+a2*a2*f2*f2*d2c[:, 1]+2*rho*a1*a2*f1*f2*d2c[:, 2])
    return left-right


def loss_terms(model, batch, weights, scales):
    """Weighted loss and its named parts for a batch dict with r (or x) and labels."""
    if "x" in batch:
        c = model.normalized_call(batch["x"])
        part = {"value": (c-batch["c"]).square().mean()/scales["c"]}
        return weights.value*part["value"], part
    order = 2 if (weights.second or weights.pde) else 1 if weights.first else 0
    out, r = model_derivatives(model, batch["r"], order)
    part = {"value": (out["c"]-batch["c"]).square().mean()/scales["c"]}
    total = weights.value*part["value"]
    if weights.first:
        part["first"] = ((out["dc"]-batch["dc"]).square().mean(dim=0)/scales["dc"]).mean()
        total = total+weights.first*part["first"]
    if weights.second:
        part["second"] = ((out["d2c"]-batch["d2c"]).square().mean(dim=0)/scales["d2c"]).mean()
        total = total+weights.second*part["second"]
    if weights.pde:
        part["pde"] = pde_residual(r, out["dc"], out["d2c"]).square().mean()/scales["c"]
        total = total+weights.pde*part["pde"]
    return total, part


def subset(data, index):
    return {key: value[index] for key, value in data.items()}


@dataclass
class Schedule:
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-6
    lbfgs_iterations: int = 0
    lbfgs_chunk: int = 16384
    lbfgs_history: int = 50
    evaluate_every: int = 10
    extra: dict = field(default_factory=dict)


def _validation_loss(model, validation, weights, scales, chunk=8192):
    total = 0.0
    for start in range(0, len(validation["c"]), chunk):
        batch = subset(validation, slice(start, start+chunk))
        loss, _ = loss_terms(model, batch, weights, scales)
        total += float(loss.detach())*len(batch["c"])
    return total/len(validation["c"])


def train(model, data, validation, weights, schedule, seed, log=print, resume_path=None, save_every=5):
    """Adam with cosine decay, optional full-batch L-BFGS, best-validation snapshot.

    With ``resume_path`` the Adam phase saves its full state every ``save_every``
    epochs and continues from that file after an interruption.
    """
    torch.manual_seed(seed)
    scales = label_scales(data)
    count = len(data["c"])
    if not schedule.epochs and not schedule.lbfgs_iterations:
        return [], 0.0
    steps = max(schedule.epochs*math.ceil(count/schedule.batch_size), 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=schedule.learning_rate)
    ratio = schedule.final_learning_rate/schedule.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: ratio+(1-ratio)*0.5*(1+math.cos(math.pi*min(step, steps)/steps)))
    history, start, offset, first_epoch = [], perf_counter(), 0.0, 1
    best = (math.inf, None)
    if resume_path is not None and Path(resume_path).exists():
        state = torch.load(resume_path, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        history, best, offset = state["history"], state["best"], state["seconds"]
        first_epoch = state["epoch"]+1
        log(f"resumed after epoch {state['epoch']}")

    def checkpoint(stage, step, train_loss):
        nonlocal best
        score = _validation_loss(model, validation, weights, scales)
        history.append(dict(stage=stage, step=step, seconds=offset+perf_counter()-start,
                            train_loss=train_loss, validation_loss=score))
        if score < best[0]:
            best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()})
        log(f"{stage} {step}: train {train_loss:.3e} validation {score:.3e} "
            f"({offset+perf_counter()-start:.0f}s)")

    if not schedule.epochs:
        checkpoint("initial", 0, math.nan)
    for epoch in range(first_epoch, schedule.epochs+1):
        running, seen = 0.0, 0
        for index in torch.randperm(count).split(schedule.batch_size):
            loss, _ = loss_terms(model, subset(data, index), weights, scales)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())*len(index)
            seen += len(index)
        if epoch % schedule.evaluate_every == 0 or epoch == schedule.epochs:
            checkpoint("adam", epoch, running/seen)
        if resume_path is not None and (epoch % save_every == 0 or epoch == schedule.epochs):
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), rng=torch.get_rng_state(), history=history,
                            best=best, seconds=offset+perf_counter()-start, epoch=epoch),
                       f"{resume_path}.tmp")
            Path(f"{resume_path}.tmp").replace(resume_path)

    if schedule.lbfgs_iterations:
        model.load_state_dict(best[1])
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=1,
                                      history_size=schedule.lbfgs_history, tolerance_grad=0.0,
                                      tolerance_change=0.0, line_search_fn="strong_wolfe")
        chunks = [slice(i, i+schedule.lbfgs_chunk) for i in range(0, count, schedule.lbfgs_chunk)]

        def closure():
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for part in chunks:
                batch = subset(data, part)
                loss, _ = loss_terms(model, batch, weights, scales)
                (loss*len(batch["c"])/count).backward()
                total += float(loss.detach())*len(batch["c"])/count
            return torch.tensor(total, dtype=torch.float64)

        for iteration in range(1, schedule.lbfgs_iterations+1):
            value = float(optimizer.step(closure))
            if not math.isfinite(value):
                log("L-BFGS stopped on a non-finite loss")
                break
            if iteration % schedule.evaluate_every == 0 or iteration == schedule.lbfgs_iterations:
                checkpoint("lbfgs", iteration, value)
    model.load_state_dict(best[1])
    return history, offset+perf_counter()-start


def quantity_metrics(predicted, reference):
    """RMSE, maximum absolute error and normalised RMSE for each quantity."""
    rows = {}
    for name in QUANTITIES:
        error = predicted[name]-reference[name]
        scale = math.sqrt(float(np.mean(reference[name]**2)))
        rmse = math.sqrt(float(np.mean(error**2)))
        rows[name] = dict(rmse=rmse, max_abs=float(np.max(np.abs(error))),
                          normalized_rmse=rmse/scale if scale > 0 else math.nan)
    return rows


def selection_score(metrics):
    """Pre-declared rule: mean normalised RMSE over price and the ten Greeks."""
    return float(np.mean([metrics[name]["normalized_rmse"] for name in QUANTITIES]))


def evaluate_pricer(model, x, put, reference, chunk=4096):
    model.eval()
    start = perf_counter()
    predicted = price_and_greeks(model, x, put, chunk=chunk)
    seconds = perf_counter()-start
    metrics = quantity_metrics(predicted, reference)
    return predicted, metrics, seconds
