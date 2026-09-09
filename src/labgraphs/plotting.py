"""Matplotlib plots without global style changes or implicit GUI actions."""

from typing import TYPE_CHECKING

import numpy as np

from .fitting import FitResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_fit(
    result: FitResult,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (10.5, 4.5),
) -> tuple["Figure", tuple["Axes", "Axes"]]:
    """Return a figure and (fit, residual) axes for customization or saving.

    Residuals are ``y - model(x)`` at observed x, with original measurement
    uncertainties. They are not normalized ODR distances or confidence intervals.
    Call ``plt.show()`` explicitly when an interactive window is wanted.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Plotting requires: pip install "labgraphs[plot]"') from exc

    data = result.data
    x_grid = np.linspace(data.x.min(), data.x.max(), 500)
    # Evaluate before creating the figure, so an invalid domain leaves no open figure.
    y_grid = result.predict(x_grid)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
        main, residual = axes
        fig.suptitle(title if title is not None else f"{result.model.name} fit", fontsize=14)
        for ax in axes:
            ax.set_xlabel(data.x_label)
            ax.grid(alpha=0.18)
            ax.set_axisbelow(True)

        errorbar_style = dict(
            fmt="o", markersize=4, color="#236477", ecolor="#7d929c", capsize=2, elinewidth=1
        )
        main.errorbar(
            data.x, data.y, xerr=data.sx, yerr=data.sy, label="Measurements", **errorbar_style
        )
        main.plot(x_grid, y_grid, color="#bd572c", linewidth=1.8, label=result.model.name)
        main.set_ylabel(data.y_label)
        main.set_title("Measurements and model", fontsize=11)
        main.legend(frameon=False)

        residual.errorbar(data.x, result.residuals, xerr=data.sx, yerr=data.sy, **errorbar_style)
        residual.axhline(0, color="#bd572c", linestyle="--", linewidth=1.2)
        residual.set_ylabel(f"{data.y_label} − fit")
        residual.set_title("Vertical residuals at measured x", fontsize=11)
        residual.text(
            0.98,
            0.97,
            f"{result.method} · reduced χ² = {result.reduced_chi_square:.3g}\n"
            f"p ≈ {result.p_value:.3g} · dof = {result.degrees_of_freedom}",
            transform=residual.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        residual.margins(y=0.3)
    return fig, (main, residual)
