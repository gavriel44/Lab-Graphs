"""Reproducible synthetic Ohm's-law experiment; no private data or GUI required."""

import argparse
from pathlib import Path

import numpy as np

from labgraphs import Dataset, fit, models, plot_fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/ohms_law.png"))
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    current = np.linspace(0.01, 0.1, 18)
    data = Dataset(
        x=current + rng.normal(0, 0.001, current.size),
        y=0.08 + 47.0 * current + rng.normal(0, 0.06, current.size),
        sx=0.001,
        sy=0.06,
        x_label="Current (A)",
        y_label="Voltage (V)",
    )
    result = fit(data, models.linear)
    print(result.summary())
    print(f"\nEstimated resistance: {result.parameter_values['slope']:.2f} Ω (true: 47 Ω)")
    fig, _ = plot_fit(result, title="Ohm's law · synthetic measurements")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
