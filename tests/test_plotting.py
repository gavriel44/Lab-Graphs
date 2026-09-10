import builtins

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from labgraphs import Dataset, fit, models, plot_fit


@pytest.mark.parametrize("sx", [None, 0.1])
def test_plot_saves_real_figure_and_preserves_other_figures_and_style(tmp_path, monkeypatch, sx):
    existing = plt.figure()
    original_size = mpl.rcParams["font.size"]

    def no_show():
        pytest.fail("plot_fit must not call show()")

    monkeypatch.setattr(plt, "show", no_show)
    data = Dataset(
        [3, 1, 4, 2, 0],
        [7.1, 3, 8.9, 5.2, 1.1],
        sy=0.1,
        sx=sx,
        x_label="Time (s)",
        y_label="Distance (m)",
    )
    result = fit(data, models.linear)
    fig, (main, residual) = plot_fit(result, title="Example")
    try:
        assert plt.fignum_exists(existing.number)
        assert mpl.rcParams["font.size"] == original_size
        assert main.get_xlabel() == "Time (s)"
        assert residual.get_ylabel() == "Distance (m) − fit"
        curve = main.lines[-1]
        assert np.all(np.diff(curve.get_xdata()) > 0)
        assert_allclose(curve.get_ydata(), result.predict(curve.get_xdata()))
        assert_allclose(residual.lines[0].get_ydata(), result.residuals)
        output = tmp_path / "plot.png"
        fig.savefig(output)
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(fig)
        plt.close(existing)


def test_plot_extra_has_actionable_error(monkeypatch):
    result = fit(Dataset([0, 1, 2], [0.1, 1, 2.1], sy=0.1), models.linear)
    original = builtins.__import__

    def without_plot(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_plot)
    with pytest.raises(ImportError, match=r"labgraphs\[plot\]"):
        plot_fit(result)
