"""Losses, samplers, optimizers and scoring for the one-stock Heston PINN and DeepONet ladders.

Loss terms (each a mean square; weight zero disables a term):

``pde``            Heston PDE residual at collocation points (reduced or raw form)
``terminal``       payoff mismatch at tau = 0 (soft terminal condition)
``boundary``       c - BS base at the far x edges of the computational box
``zero_variance``  residual of the degenerate equation c_tau = kappa theta c_v on v = 0
``variance_cap``   far-v condition at v = v_hi: c_v = 0 ("zero_slope") or c = BS base ("base_value")
``price``          supervised price labels (DeepONet data, PINN anchors)
``derivatives``    supervised derivative labels (Sobolev / differential training)

Collocation is redrawn every step, or, with residual-based adaptive sampling (RAD),
selected from a uniform pool in proportion to |residual|^k / mean + c every few hundred
steps. Adam runs first; optional L-BFGS and Nystrom Newton-CG stages polish on a fixed
batch. The checkpoint kept is the one with the best monitor score.
Ported from the two-stock study (2d/src/neural/training.py).
"""

import copy
import math
from time import perf_counter

import numpy as np
import torch

from .coordinates import COLUMN, FIELDS, base_call, reduced_coordinates, sample_domain
from .nys_newton_cg import NysNewtonCG
from .pde import raw_residual, reduced_residual, variance_slope, zero_variance_residual
from .reference import QUANTITIES, derivative_labels, price_and_greeks, reference_labels


# ---------------------------------------------------------------- domains
def reduced_box(ranges, settings):
    """Computational box of the reduced coordinates (x and v from the settings, tau from 0)."""
    return dict(x=tuple(settings["x"]), v=tuple(settings["v"]), tau=(0.0, ranges["tau"][1]),
                kappa=tuple(ranges["kappa"]), theta=tuple(ranges["theta"]), xi=tuple(ranges["xi"]),
                rho=tuple(ranges["rho"]))


def raw_box(ranges):
    box = {n: tuple(ranges["moneyness" if n == "S" else n]) for n in FIELDS}
    box["tau"] = (0.0, ranges["tau"][1])
    return box


def training_region(ranges):
    """Range of x = log forward moneyness over the training domain."""
    tau = ranges["tau"][1]
    lo = math.log(ranges["moneyness"][0]) + min((ranges["r"][0] - ranges["q"][1]) * tau, 0)
    hi = math.log(ranges["moneyness"][1]) + max((ranges["r"][1] - ranges["q"][0]) * tau, 0)
    return lo, hi


# ---------------------------------------------------------------- samplers
def _uniform(generator, count, low, high):
    return low + (high - low) * torch.rand(count, generator=generator, dtype=torch.float64)


def sample_reduced(generator, count, box, sampling, tau_min=1e-6):
    """Collocation features (count, 7).

    A ``focus`` share of x lies in the training region; tau = tau_max u^tau_power; v =
    v_lo + (v_hi - v_lo) u^v_power (a power above 1 packs points near v = 0), or with
    ``v_inner = [inner, share]`` that share in [v_lo, inner] and the rest out to v_hi; a ``short``
    share of tau is drawn from the short-maturity range.
    """
    focus = round(count * sampling.get("focus", 0.0))
    x = _uniform(generator, count, *box["x"])
    if focus:
        x[:focus] = _uniform(generator, focus, *sampling["focus_x"])
    u = torch.rand(count, generator=generator, dtype=torch.float64)
    tau = tau_min + (box["tau"][1] - tau_min) * u ** sampling.get("tau_power", 1.0)
    short = round(count * sampling.get("short_share", 0.0))
    if short:
        tau[-short:] = _uniform(generator, short, *sampling["short_tau"])
    w = torch.rand(count, generator=generator, dtype=torch.float64)
    v = box["v"][0] + (box["v"][1] - box["v"][0]) * w ** sampling.get("v_power", 1.0)
    if "v_inner" in sampling:
        # a share of points in [v_lo, inner] (packed near 0 by v_power), the rest uniform out to v_hi
        inner, share = sampling["v_inner"]
        k = round(count * share)
        v[:k] = box["v"][0] + (inner - box["v"][0]) * w[:k] ** sampling.get("v_power", 1.0)
        v[k:] = inner + (box["v"][1] - inner) * w[k:]
    return torch.stack((x, v, tau, _uniform(generator, count, *box["kappa"]),
                        _uniform(generator, count, *box["theta"]), _uniform(generator, count, *box["xi"]),
                        _uniform(generator, count, *box["rho"])), dim=1)


def sample_raw(generator, count, box, tau_power=1.0, tau_min=1e-6):
    x = torch.empty(count, len(FIELDS), dtype=torch.float64)
    for name, (low, high) in box.items():
        if name == "tau":
            u = torch.rand(count, generator=generator, dtype=torch.float64)
            x[:, COLUMN[name]] = tau_min + (high - tau_min) * u ** tau_power
        else:
            x[:, COLUMN[name]] = _uniform(generator, count, low, high)
    return x, torch.rand(count, generator=generator) < 0.5


def residual_weight(tau, kind, tau_max):
    if kind == "none":
        return torch.ones_like(tau)
    if kind == "time":
        # c_tau grows like 1/sqrt(tau) at the strike; weighting by tau / tau_max balances maturities.
        return tau / tau_max
    raise ValueError(f"unknown residual weight {kind!r}")


# ---------------------------------------------------------------- data
def labelled_data(ranges, count, seed, short_share=0.0, short_tau=(0.005, 0.05)):
    """Domain contracts (a share from the short-maturity range) with reference derivative labels."""
    short = round(count * short_share)
    x, put = sample_domain(ranges, count - short, seed)
    if short:
        xs, ps = sample_domain(ranges, short, seed + 1, tau=short_tau)
        x, put = np.concatenate([x, xs]), np.concatenate([put, ps])
    x = torch.as_tensor(x)
    features, scale = reduced_coordinates(x)
    t = perf_counter()
    labels = reference_labels(features)
    seconds = perf_counter() - t
    put = torch.as_tensor(put)
    parity = torch.where(put, torch.exp(features[:, 0]) - 1, torch.zeros_like(scale))
    return dict(x=x, put=put, features=features, labels=labels, normalized_price=scale * (labels[:, 0] - parity),
                label_seconds=seconds)


# ---------------------------------------------------------------- scoring
def evaluate(model, x, put, chunk=2048):
    return price_and_greeks(model.price, x, put, chunk)


def score(values, reference):
    """Mean over price and the 11 sensitivities of RMSE / RMS(reference), plus each ratio."""
    ratios = {q: float(np.sqrt(np.mean((values[q] - reference[q]) ** 2)) / np.sqrt(np.mean(reference[q] ** 2)))
              for q in QUANTITIES}
    return float(np.mean(list(ratios.values()))), ratios


# ---------------------------------------------------------------- training
class Trainer:
    """One ladder level: losses from ``settings['loss']``, sampling, Adam, then L-BFGS / Newton-CG."""

    def __init__(self, model, settings, ranges, monitor, data=None, log=print):
        self.model, self.settings, self.log = model, settings, log
        self.ranges = ranges
        self.weights = {k: float(v) for k, v in settings["loss"].items() if float(v) > 0}
        self.sampling = dict(settings.get("sampling", {}))
        self.sampling.setdefault("focus_x", list(training_region(ranges)))
        self.monitor = monitor
        self.data = data
        self.layout = model.layout
        self.box = (reduced_box(ranges, settings["computational_domain"]) if self.layout == "reduced"
                    else raw_box(ranges))
        self.tau_max = ranges["tau"][1]
        if data is not None:
            self.label_scale = data["labels"].pow(2).mean(0).sqrt()
            self.price_scale = data["normalized_price"].pow(2).mean().sqrt()
            self.data_weight = torch.ones(len(data["x"]), dtype=torch.float64)
        self.pool = None
        self.history = []
        self.best = (math.inf, None, 0)

    # -- collocation
    def _residual(self, points):
        if self.layout == "reduced":
            residual, _ = reduced_residual(self.model.call, points)
            return residual * torch.sqrt(residual_weight(points[:, 2], self.sampling.get("weight", "none"), self.tau_max))
        x, put = points
        residual = raw_residual(self.model.normalized_price, x, put)
        return residual * torch.sqrt(residual_weight(x[:, COLUMN["tau"]], self.sampling.get("weight", "none"),
                                                     self.tau_max))

    def _draw(self, generator, count):
        if self.layout == "reduced":
            return sample_reduced(generator, count, self.box, self.sampling)
        return sample_raw(generator, count, self.box, self.sampling.get("tau_power", 1.0))

    def _rad_refresh(self, generator):
        """Residual-based adaptive distribution (Wu et al., 2023)."""
        rad = self.sampling["rad"]
        size, pool_size = rad["size"], rad["size"] * rad["pool_factor"]
        candidates = self._draw(generator, pool_size)
        magnitudes = []
        for start in range(0, pool_size, 8192):
            part = (candidates[start:start + 8192] if self.layout == "reduced"
                    else (candidates[0][start:start + 8192], candidates[1][start:start + 8192]))
            magnitudes.append(self._residual(part).detach().abs())
        density = torch.cat(magnitudes) ** rad["k"]
        density = density / density.mean() + rad["c"]
        index = torch.multinomial(density, size, replacement=False, generator=generator)
        self.pool = candidates[index] if self.layout == "reduced" else (candidates[0][index], candidates[1][index])

    def _collocation(self, generator, count):
        if self.pool is None:
            return self._draw(generator, count)
        size = len(self.pool) if self.layout == "reduced" else len(self.pool[0])
        index = torch.randint(size, (count,), generator=generator)
        return self.pool[index] if self.layout == "reduced" else (self.pool[0][index], self.pool[1][index])

    # -- losses
    def losses(self, generator, batch):
        terms = {}
        if "pde" in self.weights:
            terms["pde"] = self._residual(self._collocation(generator, batch["collocation"])).pow(2).mean()
        if "terminal" in self.weights:
            if self.layout == "reduced":
                points = sample_reduced(generator, batch["terminal"], self.box, dict(self.sampling, tau_power=1.0,
                                                                                    short_share=0.0))
                points[:, 2] = 0.0
                payoff = torch.relu(torch.expm1(points[:, 0]))
                terms["terminal"] = (self.model.call(points) - payoff).pow(2).mean()
            else:
                x, put = sample_raw(generator, batch["terminal"], self.box)
                x[:, COLUMN["tau"]] = 0.0
                S = x[:, COLUMN["S"]]
                payoff = torch.relu(torch.where(put, 1 - S, S - 1))
                terms["terminal"] = (self.model.normalized_price(x, put) - payoff).pow(2).mean()
        if "boundary" in self.weights:
            points = sample_reduced(generator, batch["boundary"], self.box, dict(self.sampling, focus=0.0))
            edge = torch.rand(len(points), generator=generator) < 0.5
            points[:, 0] = torch.where(edge, self.box["x"][0], self.box["x"][1])
            terms["boundary"] = (self.model.call(points) - base_call(points)).pow(2).mean()
        if "zero_variance" in self.weights:
            points = sample_reduced(generator, batch["zero_variance"], self.box, self.sampling)
            points[:, 1] = 0.0
            r = zero_variance_residual(self.model.call, points)
            terms["zero_variance"] = (r * torch.sqrt(residual_weight(points[:, 2], self.sampling.get("weight", "none"),
                                                                     self.tau_max))).pow(2).mean()
        if "variance_cap" in self.weights:
            points = sample_reduced(generator, batch["variance_cap"], self.box, self.sampling)
            points[:, 1] = self.box["v"][1]
            if self.settings.get("variance_cap_condition", "zero_slope") == "zero_slope":
                terms["variance_cap"] = variance_slope(self.model.call, points).pow(2).mean()
            else:   # "base_value": c equals the Black-Scholes base at v_hi
                terms["variance_cap"] = (self.model.call(points) - base_call(points)).pow(2).mean()
        if "price" in self.weights or "derivatives" in self.weights:
            index = torch.randint(len(self.data["x"]), (batch["data"],), generator=generator)
            if self.layout == "raw":
                prediction = self.model.normalized_price(self.data["x"][index], self.data["put"][index])
                terms["price"] = ((prediction - self.data["normalized_price"][index]) / self.price_scale).pow(2).mean()
            elif "derivatives" in self.weights:
                predicted = derivative_labels(self.model.call, self.data["features"][index], create_graph=True)
                error = ((predicted - self.data["labels"][index]) / self.label_scale).pow(2).mean(0)
                terms["price"] = error[0]
                terms["derivatives"] = error[1:].mean()
            else:
                prediction = self.model.call(self.data["features"][index])
                terms["price"] = ((prediction - self.data["labels"][index, 0]) / self.label_scale[0]).pow(2).mean()
        total = sum(self.weights.get(name, 0.0) * value for name, value in terms.items())
        return total, terms

    # -- monitor and checkpoint
    def check(self, step, stage, terms, elapsed):
        values = evaluate(self.model, *self.monitor["inputs"])
        value, ratios = score(values, self.monitor["reference"])
        if not math.isfinite(value):
            value = math.inf
        entry = dict(stage=stage, step=step, seconds=round(elapsed, 1), monitor=value,
                     price_ratio=ratios["price"], **{k: float(v.detach()) for k, v in terms.items()})
        self.history.append(entry)
        if value < self.best[0]:
            self.best = (value, copy.deepcopy(self.model.state_dict()), f"{stage} {step}")
        self.log(f"{stage} {step}: monitor {value:.4g} price {ratios['price']:.3g} "
                 + " ".join(f"{k} {float(v.detach()):.3g}" for k, v in terms.items()) + f" [{elapsed:.0f}s]")

    def batch_sizes(self):
        b = self.settings["batch"]
        return {k: b.get(k, 0) for k in ("collocation", "terminal", "boundary", "zero_variance", "variance_cap", "data")}

    def adam(self, seed, resume_path=None, deadline=None):
        """Adam with an exponentially decaying rate; resumable; stops at the wall-clock deadline."""
        o = self.settings["optimizer"]
        steps = o["adam_steps"]
        if not steps:
            return 0.0
        generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=o["learning_rate"])
        decay = (o["final_learning_rate"] / o["learning_rate"]) ** (1 / max(steps, 1))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay)
        batch = self.batch_sizes()
        rad = self.sampling.get("rad")
        first, previous = 1, 0.0
        if resume_path is not None and resume_path.exists():
            state = torch.load(resume_path, weights_only=False)
            self.model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            generator.set_state(state["generator"])
            self.pool, self.history, self.best = state["pool"], state["history"], state["best"]
            first, previous = state["step"] + 1, state["elapsed"]
            self.log(f"resumed after step {state['step']}")
        start = perf_counter()
        self.stopped_early = None
        for step in range(first, steps + 1):
            if rad and (step == 1 or step % rad["every"] == 0):
                self._rad_refresh(generator)
            optimizer.zero_grad(set_to_none=True)
            total, terms = self.losses(generator, batch)
            total.backward()
            optimizer.step()
            scheduler.step()
            elapsed = previous + perf_counter() - start
            out_of_time = deadline is not None and elapsed > deadline
            if step % o["check_every"] == 0 or step == steps or out_of_time:
                self.check(step, "adam", terms, elapsed)
                if resume_path is not None:
                    torch.save(dict(model=self.model.state_dict(), optimizer=optimizer.state_dict(),
                                    scheduler=scheduler.state_dict(), generator=generator.get_state(),
                                    pool=self.pool, history=self.history, best=self.best, step=step,
                                    elapsed=elapsed), resume_path)
            if out_of_time:
                self.stopped_early = f"adam stopped at step {step} of {steps}: wall-clock cap"
                self.log(self.stopped_early)
                break
        return previous + perf_counter() - start

    def lbfgs(self, seed, deadline=None):
        o = self.settings["optimizer"]
        iterations = o.get("lbfgs_iterations", 0)
        if not iterations:
            return
        generator = torch.Generator().manual_seed(seed + 1)
        if self.sampling.get("rad"):
            self._rad_refresh(generator)
        fixed = dict(o["lbfgs_batch"])
        state = generator.get_state()
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=o.get("lbfgs_chunk", 50),
                                      history_size=o.get("lbfgs_history", 50), tolerance_grad=0,
                                      tolerance_change=0, line_search_fn="strong_wolfe")
        holder = {}

        def closure():
            generator.set_state(state)
            optimizer.zero_grad(set_to_none=True)
            total, terms = self.losses(generator, fixed)
            total.backward()
            holder["terms"] = terms
            return total

        start = perf_counter()
        done = 0
        while done < iterations:
            loss = optimizer.step(closure)
            done += o.get("lbfgs_chunk", 50)
            if not torch.isfinite(loss):
                self.log("lbfgs stopped: non-finite loss")
                break
            self.check(done, "lbfgs", holder["terms"], perf_counter() - start)
            if deadline is not None and perf_counter() - start > deadline:
                self.log(f"lbfgs stopped after {done} iterations: wall-clock cap")
                break

    def nncg(self, seed, deadline=None):
        o = self.settings["optimizer"]
        steps = o.get("nncg_steps", 0)
        if not steps:
            return
        generator = torch.Generator().manual_seed(seed + 2)
        if self.sampling.get("rad"):
            self._rad_refresh(generator)
        fixed = dict(o["nncg_batch"])
        state = generator.get_state()
        holder = {}

        def loss_fn():
            generator.set_state(state)
            total, terms = self.losses(generator, fixed)
            holder["terms"] = terms
            return total

        optimizer = NysNewtonCG(self.model.parameters(), loss_fn, sketch_rank=o.get("nncg_rank", 32),
                                cg_max_steps=o.get("nncg_cg_steps", 30), damping=o.get("nncg_damping", 1e-6),
                                seed=seed)
        start = perf_counter()
        for step in range(1, steps + 1):
            outcome = optimizer.step()
            if not outcome["accepted"]:
                self.log(f"nncg stopped at step {step}: {outcome['reason']}")
                break
            if step % o.get("nncg_check_every", 5) == 0 or step == steps:
                self.check(step, "nncg", holder["terms"], perf_counter() - start)
            if deadline is not None and perf_counter() - start > deadline:
                self.log(f"nncg stopped after {step} steps: wall-clock cap")
                break

    def run(self, seed, resume_path=None, cap_seconds=None):
        """All stages within a total wall-clock cap (seconds)."""
        torch.manual_seed(seed)
        resuming = resume_path is not None and resume_path.exists()
        if self.settings.get("warm_start") and not resuming:
            self.check(0, "start", {}, 0.0)
        o = self.settings["optimizer"]
        has_polish = o.get("lbfgs_iterations", 0) or o.get("nncg_steps", 0)
        adam_cap = None if cap_seconds is None else cap_seconds * (0.75 if has_polish and o["adam_steps"] else 1.0)
        adam_seconds = self.adam(seed, resume_path, adam_cap)
        start = perf_counter()
        remaining = None if cap_seconds is None else max(cap_seconds - adam_seconds, 60.0)
        if self.best[1] is not None and o.get("lbfgs_from_best", True):
            self.model.load_state_dict(self.best[1])
        self.lbfgs(seed, None if remaining is None else remaining * (0.6 if o.get("nncg_steps", 0) else 1.0))
        if self.best[1] is not None:
            self.model.load_state_dict(self.best[1])
        used = perf_counter() - start
        self.nncg(seed, None if remaining is None else max(remaining - used, 60.0))
        if self.best[1] is not None:
            self.model.load_state_dict(self.best[1])
        return dict(adam_seconds=adam_seconds, total_seconds=adam_seconds + perf_counter() - start,
                    best_monitor=self.best[0], best_step=self.best[2], history=self.history,
                    stopped_early=getattr(self, "stopped_early", None))
