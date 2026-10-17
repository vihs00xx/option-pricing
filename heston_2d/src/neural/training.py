"""Losses, samplers, optimizers and scoring for the Heston basket PINN and DeepONet ladders.

Loss terms (each is a mean square; a weight of zero disables the term):

``pde``            residual of the pricing PDE at collocation points (reduced or raw form)
``terminal``       payoff mismatch at tau = 0 (soft terminal condition)
``boundary``       c - Levy at the far z edges of the computational domain
``zero_variance``  the PDE residual with one stock's variance at v = 0, where the equation
                   degenerates (that stock's diffusion terms vanish)
``variance_cap``   c_{v_i} = 0 with one stock's variance at the cap v_hi
``price``          supervised price labels (DeepONet data, PINN anchors)
``derivatives``    supervised derivative labels in the log forwards (Sobolev training)

Collocation is redrawn every step, or, with residual-based adaptive sampling (RAD),
selected from a uniform pool in proportion to |residual|^k / mean + c every few
hundred steps. Adam runs first; optional L-BFGS and Nystrom Newton-CG stages polish
on a fixed batch. The checkpoint kept is the one with the best monitor score.
Follows the four-stock study (4d/src/neural/training.py) and the one-stock Heston
variance boundary (heston/src/neural/training.py).
"""

import copy
import math
from time import perf_counter

import numpy as np
import torch

from .coordinates import box_names, levy_call, permute, reduced_coordinates, sample_domain
from .nys_newton_cg import NysNewtonCG
from .pde import raw_residual, reduced_residual, variance_slope
from .reference import derivative_labels, permute_labels, price_and_greeks, reference_labels

PARAMETERS = ("kappa", "theta", "xi", "leverage")


# ---------------------------------------------------------------- domains
def reduced_box(stocks, domain, settings):
    """Computational box of the reduced coordinates.

    The Heston parameters, correlations and the maturity cap come from the protocol's
    training domain; tau starts at 0 because the PDE starts at the payoff. z and the shares
    span the region the training domain maps to, widened like a grid solver's far boundary;
    the variances run from 0 to the cap v_hi of the settings.
    """
    d = stocks.d
    floor = settings["share_floor"]
    box = dict(z=tuple(settings["z"]), tau=(0.0, domain["tau"][1]))
    box.update({f"s_{i}": (floor, 1-(d-1)*floor) for i in range(1, d)})
    box.update({f"v_{i}": tuple(settings["v"]) for i in range(1, d+1)})
    for p in PARAMETERS:
        box.update({f"{p}_{i}": tuple(domain[f"{p}_{i}"]) for i in range(1, d+1)})
    box.update({name: tuple(domain["correlation"]) for name in stocks.features if name.startswith("rho_")})
    box.update({name: tuple(domain["vol_correlation"]) for name in stocks.features if name.startswith("eta_")})
    box.update(share_floor=floor, min_eigenvalue=domain["min_eigenvalue"],
               vol_correlation=tuple(domain["vol_correlation"]),
               residual_correlation=tuple(domain["residual_correlation"]))
    return box


def raw_box(stocks, domain):
    d = stocks.d
    floor = domain["weight_floor"]
    box = dict(tau=(0.0, domain["tau"][1]), rate=tuple(domain["rate"]))
    for i in range(1, d+1):
        box[f"spot_{i}"] = tuple(domain[f"moneyness_{i}"])
        box[f"dividend_{i}"] = tuple(domain[f"dividend_{i}"])
        for p in ("v0",)+PARAMETERS:
            box[f"{p}_{i}"] = tuple(domain[f"{p}_{i}"])
    box.update({f"weight_{i}": (floor, 1-(d-1)*floor) for i in range(1, d)})
    for i, j in stocks.pairs:
        box[f"correlation_{i+1}{j+1}"] = tuple(domain["correlation"])
        box[f"vol_correlation_{i+1}{j+1}"] = tuple(domain["vol_correlation"])
    box.update(weight_floor=floor, min_eigenvalue=domain["min_eigenvalue"],
               vol_correlation=tuple(domain["vol_correlation"]),
               residual_correlation=tuple(domain["residual_correlation"]))
    return box


# ---------------------------------------------------------------- samplers
def _uniform(generator, count, low, high):
    return low+(high-low)*torch.rand(count, generator=generator, dtype=torch.float64)


def _simplex(generator, count, d, floor):
    """Uniform points of {w : w_i >= floor, sum w = 1}, shape (count, d)."""
    e = -torch.log(torch.rand(count, d, generator=generator, dtype=torch.float64))
    return floor+(1-d*floor)*e/e.sum(1, keepdim=True)


def _entries(stocks, generator, count, low, high, min_eigenvalue):
    """Correlation entries uniform in [low, high] with rejection on the smallest eigenvalue."""
    out = torch.empty(0, len(stocks.pairs), dtype=torch.float64)
    while len(out) < count:
        entries = low+(high-low)*torch.rand(3*(count-len(out))+16, len(stocks.pairs), generator=generator,
                                            dtype=torch.float64)
        smallest = torch.linalg.eigvalsh(stocks.correlation(entries))[:, 0]
        out = torch.cat((out, entries[smallest >= min_eigenvalue]))
    return out[:count]


def _correlations(stocks, generator, count, box, leverage):
    """(R entries, C entries): C and P drawn as in the protocol's domain, R = (rho rho') o C + (rbar rbar') o P."""
    C = _entries(stocks, generator, count, *box["vol_correlation"], box["min_eigenvalue"])
    P = _entries(stocks, generator, count, *box["residual_correlation"], box["min_eigenvalue"])
    rbar = torch.sqrt(1-leverage**2)
    R = (leverage[:, :, None]*leverage[:, None, :]*stocks.correlation(C)
         + rbar[:, :, None]*rbar[:, None, :]*stocks.correlation(P))
    pairs = torch.tensor(stocks.pairs)
    return R[:, pairs[:, 0], pairs[:, 1]], C


def sample_reduced(stocks, generator, count, box, sampling, tau_min=1e-6):
    """Collocation features (count, F).

    A ``focus`` share of z lies in the training region, the rest across the whole box;
    the shares are uniform on the floored simplex; tau = tau_max u^tau_power concentrates
    points at short maturities; each variance is v_hi u^v_power (a power above 1 packs
    points near v = 0); correlations are the domain's.
    """
    d = stocks.d
    focus = round(count*sampling.get("focus", 0.0))
    z = _uniform(generator, count, *box["z"])
    if focus:
        z[:focus] = _uniform(generator, focus, *sampling["focus_z"])
    shares = _simplex(generator, count, d, box["share_floor"])
    v_low, v_high = box["v_1"]
    v = v_low+(v_high-v_low)*torch.rand(count, d, generator=generator, dtype=torch.float64)**sampling.get("v_power", 1.0)
    u = torch.rand(count, generator=generator, dtype=torch.float64)
    tau = tau_min+(box["tau"][1]-tau_min)*u**sampling.get("tau_power", 1.0)
    parameters = [torch.stack([_uniform(generator, count, *box[f"{p}_{i}"]) for i in range(1, d+1)], dim=1)
                  for p in PARAMETERS]
    R, C = _correlations(stocks, generator, count, box, parameters[-1])
    return torch.cat((z[:, None], shares[:, :-1], v, tau[:, None], *parameters, R, C), dim=1)


def sample_boundary(stocks, generator, count, box, sampling):
    features = sample_reduced(stocks, generator, count, box, dict(sampling, focus=0.0))
    edge = torch.rand(count, generator=generator) < 0.5
    features[:, 0] = torch.where(edge, box["z"][0], box["z"][1])
    return features


def sample_variance_face(stocks, generator, count, box, sampling, value):
    """Collocation with one random stock's variance at ``value`` (0 or the cap)."""
    features = sample_reduced(stocks, generator, count, box, sampling)
    stock = torch.randint(stocks.d, (count,), generator=generator)
    features[torch.arange(count), stocks.v.start+stock] = value
    return features, stock


def sample_raw(stocks, generator, count, box, strike, tau_power=1.0, tau_min=0.0):
    """Contract rows uniform in the raw box (spots from moneyness) and random put flags."""
    d, col = stocks.d, stocks.column
    x = torch.empty(count, len(stocks.fields), dtype=torch.float64)
    x[:, col["strike"]] = strike
    u = torch.rand(count, generator=generator, dtype=torch.float64)
    x[:, col["tau"]] = tau_min+(box["tau"][1]-tau_min)*u**tau_power
    x[:, col["rate"]] = _uniform(generator, count, *box["rate"])
    for i in range(1, d+1):
        x[:, col[f"spot_{i}"]] = strike*_uniform(generator, count, *box[f"spot_{i}"])
        for p in ("dividend", "v0")+PARAMETERS:
            x[:, col[f"{p}_{i}"]] = _uniform(generator, count, *box[f"{p}_{i}"])
    weights = _simplex(generator, count, d, box["weight_floor"])
    for i in range(1, d):
        x[:, col[f"weight_{i}"]] = weights[:, i-1]
    R, C = _correlations(stocks, generator, count, box, stocks.take(x, "leverage_"))
    for k, (i, j) in enumerate(stocks.pairs):
        x[:, col[f"correlation_{i+1}{j+1}"]] = R[:, k]
        x[:, col[f"vol_correlation_{i+1}{j+1}"]] = C[:, k]
    return x, torch.rand(count, generator=generator) < 0.5


def random_orders(generator, count, d):
    return torch.argsort(torch.rand(count, d, generator=generator), dim=1)


# ---------------------------------------------------------------- residuals
def residual_weight(tau, kind, tau_max):
    if kind == "none":
        return torch.ones_like(tau)
    if kind == "time":
        # c_tau grows like 1/sqrt(tau) at the strike; weighting by tau / tau_max
        # (sqrt in the residual) keeps every maturity on a comparable scale.
        return tau/tau_max
    raise ValueError(f"unknown residual weight {kind!r}")


# ---------------------------------------------------------------- data
def labelled_data(stocks, domain, strike, count, seed, settings):
    """Contracts from the protocol's training domain with simulated derivative labels.

    Labels are for the normalized call (``reference.labels``); the put flag is kept for
    the raw layout. ``settings`` is the label simulation (points, steps, seed).
    """
    x, put = sample_domain(stocks, domain, strike, count, seed)
    x = torch.as_tensor(x)
    features, scale = reduced_coordinates(stocks, x)
    start = perf_counter()
    labels = reference_labels(stocks, features, settings)
    seconds = perf_counter()-start
    put = torch.as_tensor(put)
    parity = torch.where(put, torch.exp(features[:, 0])-1, torch.zeros_like(scale))
    normalized_price = scale/x[:, stocks.column["strike"]]*(labels[:, 0]-parity)
    return dict(x=x, put=put, features=features, labels=labels, normalized_price=normalized_price,
                label_seconds=seconds)


# ---------------------------------------------------------------- scoring
def evaluate(model, x, put, chunk=2048):
    return price_and_greeks(model.stocks, model.price, x, put, chunk)


def score(values, reference):
    """Mean over price and every Greek of RMSE / RMS(reference), plus each ratio."""
    ratios = {q: float(np.sqrt(np.mean((values[q]-reference[q])**2))/np.sqrt(np.mean(reference[q]**2)))
              for q in reference}
    return float(np.mean(list(ratios.values()))), ratios


def noise_floor(reference, errors):
    """RMS(reference standard error) / RMS(reference) per quantity: the score a perfect pricer would get."""
    return {q: float(np.sqrt(np.mean(errors[q]**2))/np.sqrt(np.mean(reference[q]**2))) for q in reference}


# ---------------------------------------------------------------- training
class Trainer:
    """One ladder level: losses from ``settings['loss']``, sampling, Adam, then L-BFGS."""

    def __init__(self, model, settings, domain, strike, monitor, data=None, log=print):
        self.model, self.settings, self.log = model, settings, log
        self.domain, self.strike = domain, strike
        self.weights = {k: float(v) for k, v in settings["loss"].items() if float(v) > 0}
        self.sampling = settings.get("sampling", {})
        self.monitor = monitor
        self.data = data
        self.layout, self.stocks = model.layout, model.stocks
        self.box = (reduced_box(self.stocks, domain, settings["computational_domain"]) if self.layout == "reduced"
                    else raw_box(self.stocks, domain))
        self.augment = settings.get("augment")
        if self.augment not in (None, "permutation") or (self.augment and self.layout != "reduced"):
            raise ValueError(f"unknown augmentation {self.augment!r} for the {self.layout} layout")
        if self.layout == "raw" and ({"zero_variance", "variance_cap", "boundary"} & set(self.weights)):
            raise ValueError("the variance faces and the far boundary are reduced-layout terms")
        self.tau_max = domain["tau"][1]
        if data is not None:
            labels = data["labels"]
            self.label_scale = labels.pow(2).mean(0).sqrt()
            self.price_scale = data["normalized_price"].pow(2).mean().sqrt()
        self.pool = None
        self.history = []
        self.best = (math.inf, None, 0)

    # -- collocation
    def _residual(self, points):
        weight = self.sampling.get("weight", "none")
        if self.layout == "reduced":
            residual, _ = reduced_residual(self.stocks, self.model.call, points)
            return residual*torch.sqrt(residual_weight(points[:, self.stocks.tau], weight, self.tau_max))
        x, put = points
        residual = raw_residual(self.stocks, self.model.normalized_price, x, put)
        return residual*torch.sqrt(residual_weight(x[:, self.stocks.column["tau"]], weight, self.tau_max))

    def _draw(self, generator, count):
        if self.layout == "reduced":
            return sample_reduced(self.stocks, generator, count, self.box, self.sampling)
        return sample_raw(self.stocks, generator, count, self.box, self.strike, self.sampling.get("tau_power", 1.0))

    def _rad_refresh(self, generator):
        """Residual-based adaptive distribution (Wu et al., 2023)."""
        rad = self.sampling["rad"]
        size, pool_size = rad["size"], rad["size"]*rad["pool_factor"]
        candidates = self._draw(generator, pool_size)
        magnitudes = []
        for start in range(0, pool_size, 8192):
            part = (candidates[start:start+8192] if self.layout == "reduced"
                    else (candidates[0][start:start+8192], candidates[1][start:start+8192]))
            magnitudes.append(self._residual(part).detach().abs())
        density = torch.cat(magnitudes)**rad["k"]
        density = density/density.mean()+rad["c"]
        index = torch.multinomial(density, size, replacement=False, generator=generator)
        self.pool = (candidates[index] if self.layout == "reduced"
                     else (candidates[0][index], candidates[1][index]))

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
                points = sample_reduced(self.stocks, generator, batch["terminal"], self.box,
                                        dict(self.sampling, tau_power=1.0))
                points[:, self.stocks.tau] = 0.0
                payoff = torch.relu(torch.expm1(points[:, 0]))
                terms["terminal"] = (self.model.call(points)-payoff).pow(2).mean()
            else:
                x, put = sample_raw(self.stocks, generator, batch["terminal"], self.box, self.strike)
                x[:, self.stocks.column["tau"]] = 0.0
                basket = (self.stocks.weights(x)*self.stocks.take(x, "spot_")).sum(1)/self.strike
                payoff = torch.relu(torch.where(put, 1-basket, basket-1))
                terms["terminal"] = (self.model.normalized_price(x, put)-payoff).pow(2).mean()
        if "boundary" in self.weights:
            points = sample_boundary(self.stocks, generator, batch["boundary"], self.box, self.sampling)
            terms["boundary"] = (self.model.call(points)-levy_call(self.stocks, points)).pow(2).mean()
        if "zero_variance" in self.weights:
            points, _ = sample_variance_face(self.stocks, generator, batch["zero_variance"], self.box, self.sampling,
                                             self.box["v_1"][0])
            terms["zero_variance"] = self._residual(points).pow(2).mean()
        if "variance_cap" in self.weights:
            points, stock = sample_variance_face(self.stocks, generator, batch["variance_cap"], self.box,
                                                 self.sampling, self.box["v_1"][1])
            slope = variance_slope(self.stocks, self.model.call, points)
            terms["variance_cap"] = slope[torch.arange(len(stock)), stock].pow(2).mean()
        if "price" in self.weights or "derivatives" in self.weights:
            index = torch.randint(len(self.data["x"]), (batch["data"],), generator=generator)
            if self.layout == "raw":
                prediction = self.model.normalized_price(self.data["x"][index], self.data["put"][index])
                terms["price"] = ((prediction-self.data["normalized_price"][index])/self.price_scale).pow(2).mean()
            else:
                features, labels = self.data["features"][index], self.data["labels"][index]
                if self.augment == "permutation":
                    # The call is unchanged by relabelling the stocks: each draw sees its
                    # contract under a random relabelling.
                    order = random_orders(generator, len(index), self.stocks.d)
                    features = permute(self.stocks, features, order)
                    labels = permute_labels(self.stocks, labels, order)
                if "derivatives" in self.weights:
                    predicted = derivative_labels(self.stocks, self.model.call, features, create_graph=True)
                    error = ((predicted-labels)/self.label_scale).pow(2).mean(0)
                    terms["price"] = error[0]
                    terms["derivatives"] = error[1:].mean()
                else:
                    prediction = self.model.call(features)
                    terms["price"] = ((prediction-labels[:, 0])/self.label_scale[0]).pow(2).mean()
        total = sum(self.weights.get(name, 0.0)*value for name, value in terms.items())
        return total, terms

    # -- monitor and checkpoint
    def check(self, step, stage, terms, elapsed):
        values = evaluate(self.model, *self.monitor["inputs"])
        value, ratios = score(values, self.monitor["reference"])
        entry = dict(stage=stage, step=step, seconds=round(elapsed, 1), monitor=value,
                     price_ratio=ratios["price"], **{k: float(v.detach()) for k, v in terms.items()})
        self.history.append(entry)
        if value < self.best[0]:
            self.best = (value, copy.deepcopy(self.model.state_dict()), f"{stage} {step}")
        self.log(f"{stage} {step}: monitor {value:.4g} price {ratios['price']:.3g} "
                 + " ".join(f"{k} {float(v.detach()):.3g}" for k, v in terms.items()) + f" [{elapsed:.0f}s]")

    def batch_sizes(self):
        b = self.settings["batch"]
        return {k: b.get(k, 0) for k in ("collocation", "terminal", "boundary", "zero_variance", "variance_cap",
                                         "data")}

    def adam(self, seed, resume_path=None):
        """Adam with an exponentially decaying rate; resumable from ``resume_path``.

        The resume file holds the model, optimizer, scheduler, sampling generator, RAD
        pool, history and best checkpoint, so an interrupted run continues identically.
        """
        o = self.settings["optimizer"]
        steps = o["adam_steps"]
        if not steps:
            return 0.0
        generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=o["learning_rate"])
        decay = (o["final_learning_rate"]/o["learning_rate"])**(1/max(steps, 1))
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
            first, previous = state["step"]+1, state["elapsed"]
            self.log(f"resumed after step {state['step']}")
        start = perf_counter()
        for step in range(first, steps+1):
            if rad and (step == 1 or step % rad["every"] == 0):
                self._rad_refresh(generator)
            optimizer.zero_grad(set_to_none=True)
            total, terms = self.losses(generator, batch)
            total.backward()
            optimizer.step()
            scheduler.step()
            if step % o["check_every"] == 0 or step == steps:
                elapsed = previous+perf_counter()-start
                self.check(step, "adam", terms, elapsed)
                if resume_path is not None:
                    torch.save(dict(model=self.model.state_dict(), optimizer=optimizer.state_dict(),
                                    scheduler=scheduler.state_dict(), generator=generator.get_state(),
                                    pool=self.pool, history=self.history, best=self.best, step=step,
                                    elapsed=elapsed), resume_path)
        return previous+perf_counter()-start

    def lbfgs(self, seed):
        o = self.settings["optimizer"]
        iterations = o.get("lbfgs_iterations", 0)
        if not iterations:
            return
        generator = torch.Generator().manual_seed(seed+1)
        if self.sampling.get("rad"):
            self._rad_refresh(generator)
        fixed = dict(o["lbfgs_batch"])
        # A fixed batch: freeze one draw of every term by reseeding before each evaluation.
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
            self.check(done, "lbfgs", holder["terms"], perf_counter()-start)

    def nncg(self, seed):
        """Nystrom-preconditioned Newton-CG on a fixed batch (Rathore et al., 2024)."""
        o = self.settings["optimizer"]
        steps = o.get("nncg_steps", 0)
        if not steps:
            return
        generator = torch.Generator().manual_seed(seed+2)
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
        for step in range(1, steps+1):
            outcome = optimizer.step()
            if not outcome["accepted"]:
                self.log(f"nncg stopped at step {step}: {outcome['reason']}")
                break
            if step % o.get("nncg_check_every", 5) == 0 or step == steps:
                self.check(step, "nncg", holder["terms"], perf_counter()-start)

    def run(self, seed, resume_path=None):
        torch.manual_seed(seed)
        resuming = resume_path is not None and resume_path.exists()
        if self.settings.get("warm_start") and not resuming:
            # Refining a warm start: its starting point competes for the best checkpoint.
            self.check(0, "start", {}, 0.0)
        adam_seconds = self.adam(seed, resume_path)
        start = perf_counter()
        if self.best[1] is not None and self.settings["optimizer"].get("lbfgs_from_best", True):
            self.model.load_state_dict(self.best[1])
        self.lbfgs(seed)
        if self.best[1] is not None:
            self.model.load_state_dict(self.best[1])
        self.nncg(seed)
        if self.best[1] is not None:
            self.model.load_state_dict(self.best[1])
        return dict(adam_seconds=adam_seconds, total_seconds=adam_seconds+perf_counter()-start,
                    best_monitor=self.best[0], best_step=self.best[2], history=self.history)
