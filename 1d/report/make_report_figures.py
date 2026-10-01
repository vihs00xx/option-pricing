"""Rebuild the report figures from recorded experiment outputs."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
STUDY = ROOT / "results/deeponet/parity_refinement/20260923T155834294516Z"
SEEDS = (22092026, 22092027, 22092028)
QUANTITIES = ("price", "delta", "gamma", "vega", "theta", "rho")
SPLITS = ("validation", "in_domain", "stress")
NAVY, TEAL, BLUE, VIOLET, RED = "#14243A", "#04A8A5", "#3573C4", "#7259BA", "#D14B62"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#AAB6C5", "axes.labelcolor": NAVY,
    "text.color": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 220,
})


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_values(architecture, seed, split):
    folder = STUDY / f"{architecture}_{seed}"
    choice = json.loads((folder / "selection.json").read_text())["selected"]
    if split == "validation":
        metrics = json.loads((folder / choice / "validation_metrics.json").read_text())
        return {q: metrics[q]["rmse"] for q in QUANTITIES}
    path = folder / "fresh" / split / "validation_regions.csv"
    rows = read_csv(path)
    return {row["quantity"]: float(row["rmse"])
            for row in rows if row["dimension"] == "all" and row["group"] == "all"}


def neural_profile():
    ratios = np.zeros((len(QUANTITIES), len(SPLITS)))
    control = {}
    parity = {}
    for j, split in enumerate(SPLITS):
        control[split] = [selected_values("control", seed, split) for seed in SEEDS]
        parity[split] = [selected_values("parity", seed, split) for seed in SEEDS]
        for i, q in enumerate(QUANTITIES):
            a = np.mean([v[q] for v in control[split]])
            b = np.mean([v[q] for v in parity[split]])
            ratios[i, j] = b / a

    fig, ax = plt.subplots(figsize=(8.3, 4.9), layout="constrained")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ratio", ["#C9F1E8", "#FFFFFF", "#F2C3CC"])
    image = ax.imshow(ratios, cmap=cmap, vmin=0.72, vmax=1.12, aspect="auto")
    ax.set_xticks(range(3), ["Validation", "Fresh in-domain", "Fresh stress"])
    ax.set_yticks(range(6), [q.capitalize() for q in QUANTITIES])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0, pad=9)
    for i in range(6):
        for j in range(3):
            ax.text(j, i, f"{(ratios[i, j]-1)*100:+.1f}%", ha="center", va="center",
                    fontsize=11, weight="bold", color=NAVY)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Effect of parity on error", loc="left", fontsize=16, weight="bold", pad=27)
    cbar = fig.colorbar(image, ax=ax, shrink=.78, pad=.03)
    cbar.set_label("Parity / control RMSE")
    fig.savefig(OUT / "deeponet_error_profile.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.5), sharey=True, layout="constrained")
    for ax, split, name in zip(axes, SPLITS, ("Validation", "Fresh in-domain", "Fresh stress")):
        for k, seed in enumerate(SEEDS):
            a = control[split][k]["price"]
            b = parity[split][k]["price"]
            ax.plot([0, 1], [a, b], "-", color="#B8C6D5", linewidth=1.7, zorder=1)
            ax.scatter(0, a, color=BLUE, s=43, zorder=2)
            ax.scatter(1, b, color=TEAL, s=43, zorder=2)
        ax.set_xticks([0, 1], ["Control", "Parity"])
        ax.set_xlim(-.25, 1.25)
        ax.set_title(name, fontsize=11, weight="bold")
        ax.grid(axis="y", color="#E6EDF3")
    axes[0].set_ylabel("Price RMSE")
    fig.suptitle("Paired seeds under matched refinement", x=.04, ha="left",
                 fontsize=15, weight="bold")
    fig.savefig(OUT / "deeponet_paired_seeds.pdf")
    plt.close(fig)


def pinn_atlas():
    rows = read_csv(ROOT / "results/pinn/primary/anchored/tables/surface_predictions.csv")
    x = np.unique([float(row["x"]) for row in rows])
    tau = np.unique([float(row["tau"]) for row in rows])
    error = np.full((len(tau), len(x)), np.nan)
    gamma = np.full_like(error, np.nan)
    for row in rows:
        i = np.searchsorted(tau, float(row["tau"]))
        j = np.searchsorted(x, float(row["x"]))
        error[i, j] = abs(float(row["price_error"]))
        if row["gamma"]:
            gamma[i, j] = float(row["gamma"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), layout="constrained")
    extent = (x[0], x[-1], tau[0], tau[-1])
    img = axes[0].imshow(np.maximum(error, 1e-5), origin="lower", aspect="auto",
                         extent=extent, cmap="magma", norm=LogNorm(vmin=1e-5, vmax=max(.1, np.nanmax(error))))
    axes[0].set_title("Absolute price error", weight="bold")
    cbar = fig.colorbar(img, ax=axes[0], pad=.02)
    cbar.set_label("Currency units")
    neg = np.where(np.isnan(gamma), np.nan, (gamma < -1e-5).astype(float))
    axes[1].imshow(neg, origin="lower", aspect="auto", extent=extent,
                   cmap=__import__("matplotlib").colors.ListedColormap(["#E8F2F1", RED]), vmin=0, vmax=1)
    axes[1].set_title("Negative Gamma locations", weight="bold")
    for ax in axes:
        ax.axvline(1.0, color=TEAL if ax is axes[0] else NAVY, linestyle="--", linewidth=1)
        ax.set_xlabel("Moneyness  S/K")
        ax.set_ylabel("Time to maturity (years)")
        ax.set_xlim(.05, 2.0)
    fig.suptitle("Anchored PINN: accuracy and financial shape", x=.04, ha="left",
                 fontsize=15, weight="bold")
    fig.savefig(OUT / "pinn_error_atlas.pdf")
    plt.close(fig)


def classical_error_matrix():
    rows = read_csv(ROOT / "results/common_benchmark/tables/classical_metrics.csv")
    matrix = np.zeros((2, 6))
    for i, method in enumerate(("conditional_rqmc", "crank_nicolson")):
        for j, q in enumerate(QUANTITIES):
            row = next(row for row in rows if row["method"] == method and row["split"] == "all"
                       and row["quantity"] == q)
            matrix[i, j] = float(row["rmse"])
    fig, ax = plt.subplots(figsize=(8.5, 2.6), layout="constrained")
    relative = matrix / np.min(matrix, axis=0, keepdims=True)
    img = ax.imshow(np.log10(relative), cmap="Blues", aspect="auto", vmin=0,
                    vmax=max(1.0, float(np.log10(relative).max())))
    ax.set_yticks(range(2), ["Conditional RQMC", "Crank--Nicolson"])
    ax.set_xticks(range(6), [q.capitalize() for q in QUANTITIES])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0, pad=8)
    for i in range(2):
        for j in range(6):
            ax.text(j, i, f"{matrix[i,j]:.1e}", ha="center", va="center", fontsize=9,
                    color="white" if np.log10(relative[i,j]) > 2.5 else NAVY, weight="bold")
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("68-case benchmark: error by method and quantity", loc="left",
                 fontsize=15, weight="bold", pad=25)
    cbar = fig.colorbar(img, ax=ax, pad=.02, shrink=.85)
    cbar.set_label(r"$\log_{10}$(RMSE / best in column)", fontsize=8)
    fig.savefig(OUT / "classical_error_matrix.pdf")
    plt.close(fig)


def classical_convergence():
    """Panel A: Crank--Nicolson grid refinement. Panel B: MC/Sobol/RQMC pooled RMSE."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), layout="constrained")

    ax = axes[0]
    fd = read_csv(ROOT / "results/finite_difference/tables/fd_convergence.csv")
    rich = read_csv(ROOT / "results/finite_difference/tables/fd_richardson_convergence.csv")
    show = ("atm_call", "dividend_put", "short_otm_call", "high_vol_put")
    shades = [BLUE, VIOLET, TEAL, "#C46A1B"]
    for case, colour in zip(show, shades):
        pts = sorted((int(r["n_space"]), float(r["absolute_error"]))
                     for r in fd if r["case"] == case)
        n = np.array([p[0] for p in pts]); e = np.array([p[1] for p in pts])
        ax.plot(n, e, "-o", color=colour, markersize=4, linewidth=1.5,
                label=case.replace("_", " "))
    rp = sorted((int(r["base_grid_size"]), float(r["absolute_error"]))
                for r in rich if r["case"] == "atm_call")
    rn = np.array([p[0] for p in rp]); re = np.array([p[1] for p in rp])
    ax.plot(rn, re, "--s", color=NAVY, markersize=5, linewidth=1.7,
            label="Richardson (atm call)")
    ns = np.array([100.0, 1600.0])
    ax.plot(ns, 4.0e-2 * (ns / 100.0) ** -2.0, ":", color="#9AA7B6", linewidth=1.3)
    ax.text(240, 3.6e-5, r"$\mathcal{O}(N^{-2})$", color="#5A6B7E", fontsize=9, rotation=-27)
    ax.plot(ns, 2.7e-6 * (ns / 100.0) ** -4.0, ":", color="#9AA7B6", linewidth=1.3)
    ax.text(430, 4.0e-9, r"$\mathcal{O}(N^{-4})$", color="#5A6B7E", fontsize=9, rotation=-47)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Space/time intervals $N$ (Richardson: base grid)", fontsize=9)
    ax.set_ylabel("Absolute price error")
    ax.set_title("Crank--Nicolson grid refinement", loc="left", fontsize=12, weight="bold")
    ax.grid(True, which="major", color="#E6EDF3"); ax.grid(True, which="minor", color="#F2F6FA", linewidth=.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left", handlelength=1.7)

    ax = axes[1]
    pooled = read_csv(ROOT / "results/monte_carlo/tables/mc_convergence_pooled.csv")
    series = {"plain": ("Plain MC", BLUE, 0.509),
              "sobol_r8": ("Sobol, 8 scrambles", VIOLET, 0.916),
              "sobol_control_r8": ("Sobol + control variate", TEAL, 1.154)}
    for key, (name, colour, alpha) in series.items():
        pts = sorted((int(r["n_samples"]), float(r["pooled_rmse"]))
                     for r in pooled if r["method"] == key)
        n = np.array([p[0] for p in pts]); e = np.array([p[1] for p in pts])
        ax.plot(n, e, "-o", color=colour, markersize=3.5, linewidth=1.5, label=name)
        ax.annotate(rf"$\alpha={alpha:.2f}$", xy=(n[-1], e[-1]), xytext=(-4, 9),
                    textcoords="offset points", ha="right", fontsize=9,
                    weight="bold", color=colour)
    ns = np.array([1e3, 2e6])
    ax.plot(ns, 28 * ns ** -0.5, ":", color="#9AA7B6", linewidth=1.3)
    ax.text(2.4e4, 0.052, r"$N^{-1/2}$", color="#5A6B7E", fontsize=9, rotation=-13)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Payoff evaluations $N$")
    ax.set_ylabel("Pooled price RMSE")
    ax.set_title("Monte Carlo sampling convergence", loc="left", fontsize=12, weight="bold")
    ax.grid(True, which="major", color="#E6EDF3"); ax.grid(True, which="minor", color="#F2F6FA", linewidth=.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left", handlelength=1.7)

    fig.suptitle("Measured convergence of the classical solvers", x=.04, ha="left",
                 fontsize=15, weight="bold")
    fig.savefig(OUT / "classical_convergence.pdf")
    plt.close(fig)


def pinn_greeks_profiles():
    """Predicted vs analytical price, Delta and Gamma at a short maturity."""
    rows = read_csv(ROOT / "results/pinn/primary/anchored/tables/surface_predictions.csv")
    tau_target = 0.06
    sel = [r for r in rows if abs(float(r["tau"]) - tau_target) < 1e-9
           and 0.4 <= float(r["x"]) <= 1.7]
    sel.sort(key=lambda r: float(r["x"]))
    x = np.array([float(r["x"]) for r in sel])
    panels = [("price", "Price", "Currency units"),
              ("delta", "Delta", r"$\partial C/\partial S$"),
              ("gamma", "Gamma", r"$\partial^2 C/\partial S^2$")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), layout="constrained")
    for ax, (q, title, ylab) in zip(axes, panels):
        ref = np.array([float(r[f"{q}_reference"]) for r in sel])
        pred = np.array([float(r[q]) for r in sel])
        ax.plot(x, ref, "-", color="#B8C6D5", linewidth=2.4, label="Black--Scholes")
        ax.plot(x, pred, "-", color=TEAL, linewidth=1.4, label="Anchored PINN")
        if q == "price":
            err = np.abs(pred - ref) * 100
            ax2 = ax.twinx()
            ax2.plot(x, err, "-", color=RED, linewidth=1.2, alpha=.85)
            ax2.set_ylabel("Absolute price error (cents)", color=RED, fontsize=9)
            ax2.tick_params(axis="y", colors=RED, labelsize=8)
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_color(RED)
        ax.axvline(1.0, color=NAVY, linestyle="--", linewidth=.9, alpha=.7)
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_xlabel(r"Moneyness $S/K$")
        ax.set_ylabel(ylab)
        ax.grid(True, color="#E6EDF3")
        if q == "gamma":
            detail = ax.inset_axes([.52, .43, .45, .43])
            detail.plot(x, ref, color="#B8C6D5", linewidth=1.4)
            detail.plot(x, pred, color=TEAL, linewidth=1.1)
            detail.axhline(0, color=NAVY, linewidth=.6)
            detail.set_xlim(1.25, 1.5)
            detail.set_ylim(-.00065, .00015)
            detail.set_xticks([1.3, 1.4, 1.5])
            detail.tick_params(labelsize=6)
            detail.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            detail.yaxis.get_offset_text().set_fontsize(6)
            detail.set_title("Negative-Gamma detail", fontsize=7)
            detail.grid(True, color="#E6EDF3")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(r"Anchored PINN and Black--Scholes profiles at $\tau=0.06$",
                 x=.04, ha="left", fontsize=15, weight="bold")
    fig.savefig(OUT / "pinn_greeks_profiles.pdf")
    plt.close(fig)


def pinn_surfaces():
    """3D price surfaces: analytical vs anchored PINN, plus the log error map."""
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rows = read_csv(ROOT / "results/pinn/primary/anchored/tables/surface_predictions.csv")
    grid = {}
    for r in rows:
        x, tau = float(r["x"]), float(r["tau"])
        if x <= 2.0 + 1e-9:
            grid[(round(x, 6), round(tau, 6))] = (float(r["price"]),
                                                  float(r["price_reference"]))
    xs = sorted({k[0] for k in grid})
    taus = sorted({k[1] for k in grid})
    X, Y = np.meshgrid(xs, taus)
    Zp = np.array([[grid[(x, t)][0] for x in xs] for t in taus])
    Zr = np.array([[grid[(x, t)][1] for x in xs] for t in taus])
    err = np.log10(np.maximum(np.abs(Zp - Zr), 1e-5))

    surf_cmap = LinearSegmentedColormap.from_list(
        "report", ["#0D1B2E", "#1C3B63", "#3573C4", "#04A8A5", "#7FE3D2"])
    fig = plt.figure(figsize=(10.8, 3.7), layout="constrained")
    panels = ((Zr, surf_cmap, "Black\u2013Scholes", None),
              (Zp, surf_cmap, "Anchored PINN", None),
              (err, "magma", r"$\log_{10}$ |error|", (-5, -1)))
    axes = []
    for i, (Z, cmap, title, norm) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        axes.append(ax)
        n_mesh = 70 if i < 2 else 48
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, rcount=n_mesh, ccount=n_mesh,
                               linewidth=0, antialiased=True,
                               vmin=norm[0] if norm else None,
                               vmax=norm[1] if norm else None)
        ax.set_title(title, fontsize=11.5, weight="bold", color=NAVY, pad=2)
        ax.set_xlabel(r"$S/K$", labelpad=-5, fontsize=9)
        ax.set_ylabel(r"$\tau$", labelpad=-7, fontsize=9)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_yticks([0, 0.5, 1.0])
        if i < 2:
            ax.set_zlabel("Price", labelpad=-8, fontsize=9)
            ax.set_zticks([0, 50, 100])
        else:
            ax.set_zticks([-5, -4, -3, -2])
            cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.02,
                              ticks=[-5, -4, -3, -2])
            cb.ax.tick_params(labelsize=7.5)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=7.5, pad=-2)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("#F5F8FB")
            pane.set_edgecolor("#C9D4DF")
    fig.suptitle("Anchored PINN price surface against the analytical solution",
                 x=.04, ha="left", fontsize=15, weight="bold")
    fig.savefig(OUT / "pinn_surfaces.pdf")
    plt.close(fig)


def pinn_loss_history():
    """Weighted contributions in the recorded final theta-anchor stage."""
    rows = read_csv(ROOT / "results/pinn/primary/anchored/tables/anchored_refinement_history.csv")
    metadata = json.loads((ROOT / "results/pinn/primary/anchored/tables/training_metadata.json").read_text())
    settings = metadata["anchored_refinement"]["settings"]
    stage = settings["stages"][-1]
    weights = {"pde_loss": stage["pde_loss_weight"],
               "price_loss": stage["price_loss_weight"],
               "theta_loss": stage["theta_loss_weight"],
               "terminal_loss": settings["terminal_loss_weight"]}
    for row in rows:
        weighted_total = sum(weights[key] * float(row[key]) for key in weights)
        if not np.isclose(weighted_total, float(row["loss"]), rtol=1e-10, atol=1e-14):
            raise ValueError("Recorded loss does not match the final-stage weights")
    parts = {"loss": (r"Total $\mathcal{L}$", NAVY, 2.2, "-"),
             "pde_loss": (r"$w_R\mathcal{L}_{\mathrm{pde}}$", TEAL, 1.5, "-"),
             "price_loss": (r"$w_C\mathcal{L}_{\mathrm{price}}$", BLUE, 1.5, "-"),
             "theta_loss": (r"$w_\Theta\mathcal{L}_{\Theta}$", VIOLET, 1.5, "-"),
             "terminal_loss": (r"$w_0\mathcal{L}_{0}$", RED, 1.4, "--")}
    fig, ax = plt.subplots(figsize=(8.6, 3.9), layout="constrained")
    offset = 0
    for phase, xoff in (("anchored_adam", 0), ("anchored_lbfgs", 300)):
        rs = sorted((int(r["step"]), r) for r in rows if r["phase"] == phase)
        steps = np.array([p[0] + xoff for p in rs], dtype=float)
        for key, (name, colour, lw, ls) in parts.items():
            vals = np.array([max(weights.get(key, 1.0) * float(p[1][key]), 1e-12) for p in rs])
            label = name if phase == "anchored_adam" else None
            ax.plot(steps, vals, ls, color=colour, linewidth=lw, label=label)
        offset = max(offset, steps[-1])
    # shaded phase bands with labels at the top, clear of the curves
    ax.set_yscale("log")
    ax.axvspan(0, 300, color="#EEF3F8", zorder=0)
    ax.axvspan(300, offset, color="#FFFFFF", zorder=0)
    ax.axvline(300, color="#9AA7B6", linewidth=1.1, linestyle=":")
    ax.set_xlabel("Adam step, then L-BFGS closure evaluation (offset by 300)")
    ax.set_ylabel("Weighted loss contribution")
    ax.set_xlim(0, offset * 1.02)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 10 ** 0.55)
    band_top = ymax * 10 ** 0.24
    ax.text(150, band_top, "Adam phase", ha="center", va="center",
            fontsize=9.5, color="#5A6B7E", weight="bold")
    ax.text(505, band_top, "L-BFGS phase", ha="center", va="center",
            fontsize=9.5, color="#5A6B7E", weight="bold")
    ax.annotate("switch to full-batch\nevaluation at the break",
                xy=(300, band_top), xytext=(312, band_top),
                ha="left", va="center", fontsize=7.8, color="#8A97A8",
                style="italic", linespacing=1.25)
    ax.set_title("Final Theta-anchor stage: weighted loss contributions",
                 loc="left", fontsize=15, weight="bold", pad=26)
    ax.grid(True, which="major", color="#E6EDF3")
    ax.legend(frameon=False, fontsize=8.5, ncol=5, loc="lower left",
              bbox_to_anchor=(0.0, 1.005), handlelength=1.5, columnspacing=1.0)
    fig.savefig(OUT / "pinn_loss_history.pdf")
    plt.close(fig)


def _rounded_box(ax, xy, w, h, face, edge, lw=1.4, radius=.035):
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0.008,rounding_size={radius}",
                         facecolor=face, edgecolor=edge, linewidth=lw,
                         mutation_aspect=1.2)
    ax.add_patch(box)


def _arrow(ax, start, end, colour="#5A6B7E", lw=1.5):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 color=colour, linewidth=lw, shrinkA=2, shrinkB=2))


def deeponet_schematic():
    """Branch/trunk operator schematic with the exact-parity readout."""
    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")

    mono = dict(family="monospace", fontsize=8.6)
    NAVY_L, TEAL_L, BLUE_L, RED_L = "#EDF2F8", "#E4F6F4", "#EAF1FA", "#FBEDEF"

    _rounded_box(ax, (0.15, 2.55), 2.05, 1.65, NAVY_L, NAVY)
    ax.text(1.18, 3.95, "Branch input $a$", ha="center", fontsize=10, weight="bold")
    ax.text(0.28, 3.06, "K  T  r  q\n$\\sigma$  call/put", va="center", **mono)

    _rounded_box(ax, (0.15, 0.35), 2.05, 1.65, TEAL_L, TEAL)
    ax.text(1.18, 1.75, "Trunk input $z$", ha="center", fontsize=10, weight="bold")
    ax.text(0.28, 0.88, "S  $\\tau$\n(spot, time\n to expiry)", va="center", **mono)

    _rounded_box(ax, (3.0, 2.7), 1.9, 1.35, "white", NAVY)
    ax.text(3.95, 3.72, "Branch net", ha="center", fontsize=10, weight="bold")
    ax.text(3.95, 3.28, "3 layers x 64\ntanh", ha="center", fontsize=8.6, color="#3D4E63")

    _rounded_box(ax, (3.0, 0.5), 1.9, 1.35, "white", TEAL)
    ax.text(3.95, 1.52, "Trunk net", ha="center", fontsize=10, weight="bold")
    ax.text(3.95, 1.08, "3 layers x 64\ntanh", ha="center", fontsize=8.6, color="#3D4E63")

    for y in (3.18, 3.38, 3.58):
        ax.plot([5.12, 5.52], [y, y], color=NAVY, linewidth=1.3)
    ax.text(5.32, 3.72, "$b_1\\ldots b_p$", ha="center", fontsize=9)
    for y in (0.98, 1.18, 1.38):
        ax.plot([5.12, 5.52], [y, y], color=TEAL, linewidth=1.3)
    ax.text(5.32, 1.52, "$t_1\\ldots t_p$", ha="center", fontsize=9)

    _arrow(ax, (2.2, 3.37), (2.97, 3.37), NAVY)
    _arrow(ax, (2.2, 1.17), (2.97, 1.17), TEAL)
    _arrow(ax, (4.9, 3.37), (5.10, 3.38), NAVY, lw=1.2)
    _arrow(ax, (4.9, 1.17), (5.10, 1.38), TEAL, lw=1.2)

    from matplotlib.patches import Circle
    circ = Circle((6.05, 2.28), 0.46, facecolor="white", edgecolor=NAVY, linewidth=1.5)
    ax.add_patch(circ)
    ax.text(6.05, 2.42, r"$\langle b,t\rangle/\sqrt{p}$", ha="center", va="center", fontsize=9.5)
    ax.text(6.05, 2.06, r"$+\,b_0$", ha="center", va="center", fontsize=9.5)
    _arrow(ax, (5.55, 3.28), (5.80, 2.60), NAVY)
    _arrow(ax, (5.55, 1.28), (5.80, 1.96), TEAL)

    _arrow(ax, (6.54, 2.28), (6.82, 2.28))
    _rounded_box(ax, (6.85, 1.86), 1.05, 0.84, NAVY_L, NAVY)
    ax.text(7.37, 2.44, "$\\times\\,K$", ha="center", fontsize=10.5, weight="bold")
    ax.text(7.37, 2.08, "$M_\\theta=(C{+}P)/2$", ha="center", fontsize=8.2)

    _arrow(ax, (7.93, 2.5), (8.42, 3.32), TEAL)
    _arrow(ax, (7.93, 2.06), (8.42, 1.28), RED)
    _rounded_box(ax, (8.45, 3.02), 1.35, 0.72, TEAL_L, TEAL)
    ax.text(9.12, 3.5, "Call", ha="center", fontsize=9.5, weight="bold")
    ax.text(9.12, 3.18, r"$C_\theta=M_\theta+\frac{D}{2}$", ha="center", fontsize=9)
    _rounded_box(ax, (8.45, 0.86), 1.35, 0.72, RED_L, RED)
    ax.text(9.12, 1.34, "Put", ha="center", fontsize=9.5, weight="bold")
    ax.text(9.12, 1.02, r"$P_\theta=M_\theta-\frac{D}{2}$", ha="center", fontsize=9)

    ax.text(9.12, 0.55, r"$D=Se^{-q\tau}-Ke^{-r\tau}$", ha="center", fontsize=8.6,
            color="#3D4E63")
    ax.text(9.12, 0.24, r"$C_\theta-P_\theta\equiv D$  exactly", ha="center",
            fontsize=9, weight="bold", color=TEAL)

    ax.text(6.05, 2.85, "", fontsize=8)
    _arrow(ax, (6.05, 1.9), (6.05, 1.02), "#8A97A8", lw=1.2)
    _rounded_box(ax, (4.7, 0.42), 2.7, 0.55, "white", "#8A97A8", lw=1.1)
    ax.text(6.05, 0.69, r"autograd: $\Delta,\ \Gamma,\ \mathcal{V},\ \Theta,\ \rho$"
            "\n" r"$\partial_S,\ \partial_S^2,\ \partial_\sigma,\ \partial_r,\ -\partial_\tau$",
            ha="center", va="center", fontsize=8.0, color="#3D4E63", linespacing=1.25)

    ax.text(0.15, 4.42, "Parity DeepONet architecture with exact-parity readout",
            fontsize=15, weight="bold", color=NAVY)
    fig.savefig(OUT / "deeponet_schematic.pdf")
    plt.close(fig)


def deeponet_regional_errors():
    """Price RMSE by moneyness and remaining-maturity band, control vs parity."""
    STUDY = ROOT / "results/deeponet/parity_refinement/20260923T155834294516Z"
    SEEDS = (22092026, 22092027, 22092028)
    bands = {"moneyness": [("<0.9", "$S/K<0.9$"), ("0.9-1.1", r"$0.9\leq S/K<1.1$"),
                           (">=1.1", r"$S/K\geq1.1$")],
             "remaining_maturity": [("<0.1", r"$\tau<0.1$"), ("0.1-0.5", r"$0.1\leq\tau<0.5$"),
                                    ("0.5-1", r"$0.5\leq\tau<1$"), (">=1", r"$\tau\geq1$")]}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=False, layout="constrained")
    for ax, (dim, groups) in zip(axes, bands.items()):
        ys = np.arange(len(groups))[::-1]
        for y, (grp, label) in zip(ys, groups):
            means = {}
            for k, (arch, colour, name) in enumerate(
                    (("control", BLUE, "Control"), ("parity", TEAL, "Parity"))):
                vals = []
                for seed in SEEDS:
                    folder = STUDY / f"{arch}_{seed}"
                    choice = json.loads((folder / "selection.json").read_text())["selected"]
                    with (folder / choice / "validation_regions.csv").open(
                            newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    row = next(r for r in rows if r["dimension"] == dim
                               and r["group"] == grp and r["quantity"] == "price")
                    vals.append(float(row["rmse"]))
                mean = float(np.mean(vals))
                means[name] = mean
                ax.scatter([mean], [y + (0.16 if arch == "control" else -0.16)],
                           s=58, color=colour, zorder=3,
                           label=name if y == ys[0] else None)
                for v in vals:
                    ax.plot([v, v], [y - 0.28, y + 0.28], color=colour, linewidth=.7,
                            alpha=.4, zorder=2)
            drop = (means["Parity"] / means["Control"] - 1) * 100
            ax.plot([means["Control"], means["Parity"]], [y, y], color="#B8C6D5",
                    linewidth=1.4, zorder=1)
            ax.annotate(f"{drop:+.0f}%", ((means["Control"] + means["Parity"]) / 2, y + 0.33),
                        ha="center", fontsize=8.5, weight="bold",
                        color=TEAL if drop < 0 else RED)
        ax.set_yticks(ys, [label for _, label in groups])
        ax.set_xlabel("Validation price RMSE")
        ax.set_title({"moneyness": "By moneyness band",
                      "remaining_maturity": "By remaining maturity"}[dim],
                     loc="left", fontsize=12, weight="bold")
        ax.grid(axis="x", color="#E6EDF3")
        ax.set_ylim(-0.65, len(groups) - 0.2)
        if dim == "moneyness":
            ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Validation price RMSE by region (mean of three matched seeds)",
                 x=.04, ha="left", fontsize=15, weight="bold")
    fig.savefig(OUT / "deeponet_regional_errors.pdf")
    plt.close(fig)


def pinn_architecture():
    """Transformer-style block diagram of the anchored PINN training loop."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.4, 8.6))
    ax.set_xlim(0, 10.9); ax.set_ylim(14.4, 0); ax.axis("off")

    FILL = {"lav": "#EADCF5", "org": "#FBDCB6", "yel": "#F5F6C6",
            "blu": "#CDE4F6", "pur": "#D8C7EE", "grn": "#E9F4E2"}
    EDGE = {"lav": "#B294CC", "org": "#D99A4E", "yel": "#B9BC57",
            "blu": "#6FA8D8", "pur": "#9A79C6", "grn": "#9DBB92"}
    BOX = dict(boxstyle="round,pad=0.10,rounding_size=0.16", linewidth=1.4)

    def block(cx, cy, w, h, text, kind, fs=9.3):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     facecolor=FILL[kind], edgecolor=EDGE[kind], **BOX))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                linespacing=1.25)

    def container(x, y, w, h, label):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.10,rounding_size=0.22",
                     facecolor=FILL["grn"], edgecolor=EDGE["grn"],
                     linewidth=1.3, zorder=0))
        ax.text(x + 0.16, y + 0.30, label, ha="left", va="center",
                fontsize=9.5, color="#3D4E63", linespacing=1.2)

    def arrow(p, q, rad=0.0, lw=1.4):
        from matplotlib.patches import FancyArrowPatch
        ax.add_patch(FancyArrowPatch(
            p, q, arrowstyle="-|>", mutation_scale=13, color="black",
            linewidth=lw, shrinkA=1, shrinkB=1,
            connectionstyle=f"arc3,rad={rad}", zorder=3))

    def elbow(pts, lw=1.4):
        """Right-angle connector: polyline segments, arrowhead on the last."""
        from matplotlib.patches import FancyArrowPatch
        for a, b in zip(pts[:-2], pts[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=lw,
                    solid_joinstyle="round", solid_capstyle="round", zorder=3)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                     mutation_scale=13, color="black", linewidth=lw,
                     shrinkA=0, shrinkB=1, zorder=3))

    LX, RX = 3.4, 7.7            # column centres
    BW, BH = 3.1, 0.86

    # ---------------- left column: forward model ----------------
    container(1.35, 4.20, 4.1, 6.42, "Differentiable\nprice model")
    block(LX, 3.35, BW, BH, "Training points\nSobol + strike-concentrated", "lav")
    block(LX, 5.45, BW, BH, "Network  $N_\\theta$\n4 layers $\\times$ 64 tanh", "org")
    block(LX, 7.35, BW, BH, "Ansatz\n$\\xi\\,u_R(\\tau)+4\\xi(1-\\xi)\\,N_\\theta$", "yel")
    block(LX, 9.35, BW, BH, "Price surface\n$u_\\theta(y,\\tau)$", "yel")
    block(LX, 11.35, BW, BH, "Strike scaling\n$C_\\theta = K\\,u_\\theta$", "blu")
    ax.text(LX, 12.75, "Price prediction", ha="center", va="top", fontsize=10.5)
    ax.text(LX, 2.62, "collocation and\nanchor queries", ha="center", va="bottom",
            fontsize=9.5, linespacing=1.25)

    arrow((LX, 3.80), (LX, 5.00))
    arrow((LX, 5.88), (LX, 6.92))
    arrow((LX, 7.78), (LX, 8.92))
    arrow((LX, 9.78), (LX, 10.92))
    arrow((LX, 11.78), (LX, 12.46))
    # skip: boundary terms bypass the network into the ansatz
    elbow([(LX - BW / 2, 3.35), (1.02, 3.35), (1.02, 7.35),
           (LX - BW / 2 - 0.04, 7.35)])
    ax.text(0.62, 5.35, "boundary\nterms", ha="center", va="center",
            fontsize=8.2, rotation=90, color="#3D4E63")

    # ---------------- right column: loss stage ----------------
    container(5.75, 4.20, 3.9, 6.42, "Loss stage")
    block(RX, 3.35, BW, BH, "Analytical anchors\n$C_{\\rm BS},\\ \\Theta_{\\rm BS}$", "lav")
    block(RX, 5.45, BW, BH + 0.28, "Automatic differentiation\n$\\partial_y,\\ \\partial_{yy},\\ \\partial_\\tau$", "lav")
    block(RX, 7.45, BW, BH, "PDE residual  $R_\\theta$", "org")
    block(RX, 9.45, BW, BH + 0.24, "Anchored loss\nresidual + payoff + anchors", "yel", fs=8.4)
    block(RX, 11.35, BW, BH, "Refinement\nAdam $\\rightarrow$ L-BFGS", "blu")
    ax.text(RX, 12.60, "Refined parameters $\\theta$\n(pinned checkpoint)",
            ha="center", va="top", fontsize=10.5, linespacing=1.3)
    ax.text(RX, 2.62, "reference labels", ha="center", va="bottom", fontsize=9.5)

    arrow((RX, 3.80), (RX, 4.92))
    arrow((RX, 6.02), (RX, 7.02))
    arrow((RX, 7.88), (RX, 8.94))
    arrow((RX, 10.00), (RX, 10.92))
    arrow((RX, 11.78), (RX, 12.28))
    # skip: gradients from the differentiation block join the loss directly
    elbow([(RX + BW / 2, 5.45), (10.28, 5.45), (10.28, 9.62),
           (RX + BW / 2 + 0.04, 9.62)])

    # cross-connection: surface feeds residual and loss (encoder -> decoder)
    elbow([(LX + BW / 2, 9.35), (5.42, 9.35), (5.42, 7.45),
           (RX - BW / 2 - 0.04, 7.45)])
    arrow((LX + BW / 2 + 0.04, 9.62), (RX - BW / 2 - 0.04, 9.62))

    ax.text(0.026, 0.978, "Anchored PINN: forward model and loss stages",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=14, weight="bold", color=NAVY)
    fig.savefig(OUT / "pinn_architecture.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    classical_error_matrix()
    pinn_atlas()
    neural_profile()
    classical_convergence()
    pinn_greeks_profiles()
    pinn_surfaces()
    pinn_loss_history()
    deeponet_schematic()
    deeponet_regional_errors()
    pinn_architecture()
