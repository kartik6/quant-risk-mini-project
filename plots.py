"""The four figures for the README.

Each plot answers one question that a table answers less well.

    1. What is VaR?              A cutoff on a distribution.
    2. Do the methods agree?     At 95% yes, at 99% no.
    3. What does scenario 3 do?  It changes correlations and nothing else.
    4. How bad is a crisis?      Far worse than any VaR figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FIGSIZE = (10, 6)
DPI = 130
LOSS_COLOUR = "#c0392b"
BASE_COLOUR = "#2c3e50"
ALT_COLOUR = "#2980b9"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_return_distribution(
    portfolio_returns: pd.Series,
    var_by_confidence: dict[float, float],
    portfolio_value: float,
    path: Path,
) -> Path:
    """The return distribution, plus a magnified view of the left tail.

    The left panel shows what VaR is. It is a cutoff on a distribution, and
    the shaded area holds the 5% of days that breach the 95% figure.

    The right panel shows why the normal assumption fails. It magnifies the
    loss tail, where the real bars stand above the fitted bell curve. That
    excess is what "fat tails" means, and it is why the parametric method
    understates the 99% loss.
    """
    values = portfolio_returns.to_numpy()
    sigma = values.std(ddof=1)

    # A few extreme days would stretch the axis and squash the region that
    # matters. Clip the view and say how many days fall outside it.
    edge = max(abs(np.percentile(values, 0.3)), np.percentile(values, 99.7)) * 1.15
    hidden = int((values < -edge).sum() + (values > edge).sum())

    fig, (ax, tail) = plt.subplots(
        1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.75, 1]}
    )
    bins = np.linspace(values.min(), values.max(), 160)
    grid = np.linspace(-edge, edge, 600)
    normal = stats.norm.pdf(grid, 0.0, sigma)
    cutoffs = {c: -v / portfolio_value for c, v in var_by_confidence.items()}

    # --- left panel: the whole distribution ---
    ax.hist(values, bins=bins, density=True, color="#bdc3c7",
            edgecolor="white", linewidth=0.3, label="Actual daily returns")
    ax.plot(grid, normal, color=BASE_COLOUR, linewidth=1.8,
            label=f"Normal fit (sigma = {sigma:.4%}, mean = 0)")
    ax.set_xlim(-edge, edge)
    ax.axvspan(-edge, max(cutoffs.values()), color=LOSS_COLOUR, alpha=0.09,
               label="Worst 5% of days")

    for height, (confidence, cutoff) in zip(
        (0.94, 0.72), sorted(cutoffs.items(), reverse=True)
    ):
        ax.axvline(cutoff, color=LOSS_COLOUR, linestyle="--", linewidth=1.6)
        ax.annotate(
            f"{confidence:.0%} VaR\n${var_by_confidence[confidence]:,.0f}",
            xy=(cutoff, ax.get_ylim()[1] * height),
            xytext=(9, 0), textcoords="offset points",
            ha="left", va="top", fontsize=9.5,
            color=LOSS_COLOUR, fontweight="bold",
        )

    ax.set_title("VaR is a cutoff on this distribution", fontsize=11.5)
    ax.set_xlabel("Daily portfolio return"
                  + (f"   ({hidden} extreme days outside this range)" if hidden else ""))
    ax.set_ylabel("Density")
    ax.set_xticks(np.arange(-0.06, 0.061, 0.02))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    # --- right panel: the loss tail, magnified ---
    tail_from, tail_to = -edge, -1.8 * sigma
    tail.hist(values, bins=bins, density=True, color="#bdc3c7",
              edgecolor="white", linewidth=0.3, label="Actual daily returns")
    tail.fill_between(grid, normal, color=BASE_COLOUR, alpha=0.16,
                      label="What a normal curve allows")
    tail.plot(grid, normal, color=BASE_COLOUR, linewidth=1.8)

    for confidence, cutoff in cutoffs.items():
        if tail_from <= cutoff <= tail_to:
            tail.axvline(cutoff, color=LOSS_COLOUR, linestyle="--", linewidth=1.4)
            tail.annotate(f"{confidence:.0%}", xy=(cutoff, 0),
                          xytext=(3, 4), textcoords="offset points",
                          fontsize=8.5, color=LOSS_COLOUR, fontweight="bold")

    tail.set_xlim(tail_from, tail_to)
    tail.set_ylim(0, float(stats.norm.pdf(tail_to, 0.0, sigma)) * 1.35)
    tail.set_title("The loss tail, magnified\nGrey above blue is the fat tail",
                   fontsize=11.5)
    tail.set_xlabel("Daily portfolio return")
    tail.xaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    tail.legend(frameon=False, loc="upper left", fontsize=8.5)

    for panel in (ax, tail):
        panel.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def plot_method_comparison(results: pd.DataFrame, path: Path) -> Path:
    """Grouped bars of every VaR and ES figure, by method.

    The methods agree closely at 95% and separate at 99%. That gap is the
    normal assumption failing in the tail, which is where a risk number is
    supposed to work hardest.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    methods = list(results.index)
    measures = list(results.columns)
    x = np.arange(len(measures))
    width = 0.8 / len(methods)
    colours = [BASE_COLOUR, ALT_COLOUR, "#7f8c8d", "#95a5a6"]

    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, results.loc[method], width,
                      label=method, color=colours[i % len(colours)])
        ax.bar_label(bars, labels=[f"${v:,.0f}" for v in results.loc[method]],
                     fontsize=7, padding=2)

    ax.set_title("The three methods compared\n"
                 "Close agreement at 95%. They separate at 99%, where the tail is fat.")
    ax.set_ylabel("Loss (USD)")
    ax.set_xticks(x, measures)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def plot_correlation_shift(base: pd.DataFrame, stressed: pd.DataFrame, path: Path) -> Path:
    """The base and stressed correlation matrices, side by side.

    Scenario 3 is about correlation, so the correlation matrix is the thing to
    show. Only the off-diagonal cells change. Every volatility is untouched.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, matrix, title in zip(axes, [base, stressed],
                                 ["Base (measured)", "Stressed (all pairs 0.85)"]):
        im = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="RdYlBu_r")
        ax.set_xticks(range(len(matrix)), matrix.columns)
        ax.set_yticks(range(len(matrix)), matrix.index)
        ax.set_title(title, fontsize=11)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                value = matrix.to_numpy()[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        color="white" if value > 0.7 else "black",
                        fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03, label="Correlation")
    fig.suptitle("Scenario 3: correlation spike. No stock became more volatile.",
                 fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scenarios_vs_var(
    var_by_confidence: dict[float, float],
    scenario_losses: dict[str, float],
    path: Path,
) -> Path:
    """Stress losses beside the VaR figures.

    The 99% VaR is meant to be a once-in-a-hundred-days loss, and a modest
    crisis doubles it. That is the argument for stress testing in one image.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    labels = [f"VaR {c:.0%}" for c in sorted(var_by_confidence)] + list(scenario_losses)
    values = [var_by_confidence[c] for c in sorted(var_by_confidence)] + list(scenario_losses.values())
    colours = [BASE_COLOUR] * len(var_by_confidence) + [LOSS_COLOUR] * len(scenario_losses)

    bars = ax.bar(labels, values, color=colours)
    ax.bar_label(bars, labels=[f"${v:,.0f}" for v in values], padding=3, fontsize=9)

    ax.set_title("Stress losses against VaR\n"
                 "VaR describes a bad day. A scenario describes a crisis.")
    ax.set_ylabel("Loss (USD)")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.margins(y=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    return _save(fig, path)
