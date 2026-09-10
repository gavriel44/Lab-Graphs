"""Fit a damped oscillation with a custom equation and informed starting values."""

import numpy as np

from labgraphs import Dataset, Model, fit


def damped_oscillation(t, amplitude, decay_rate, angular_frequency):
    return amplitude * np.exp(-decay_rate * t) * np.cos(angular_frequency * t)


def main() -> None:
    model = Model(
        name="Damped oscillation",
        function=damped_oscillation,
        parameters=("amplitude", "decay_rate", "angular_frequency"),
    )
    rng = np.random.default_rng(7)
    t = np.linspace(0, 8, 100)
    y = damped_oscillation(t, 2.0, 0.3, 2.5) + rng.normal(0, 0.08, t.size)
    result = fit(Dataset(t, y, sy=0.08), model, initial_guess=[1.8, 0.2, 2.4])
    print(result.summary())


if __name__ == "__main__":
    main()
