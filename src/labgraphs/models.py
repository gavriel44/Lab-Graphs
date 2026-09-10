"""Small model catalog and an explicit interface for custom equations."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import _vector


@dataclass(frozen=True)
class Model:
    """A vectorized ``function(x, *parameters)`` with ordered parameter names.

    Functions must return one finite value for each input x. Nonlinear models
    should normally receive experiment-specific starting values through ``fit``.
    """

    name: str
    function: Callable[..., ArrayLike]
    parameters: tuple[str, ...]
    initial_guess: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A model must have a nonempty name.")
        if not callable(self.function):
            raise TypeError("function must be callable.")
        names = tuple(self.parameters)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("parameters must contain nonempty names.")
        if len(set(names)) != len(names):
            raise ValueError("Parameter names must be unique.")
        object.__setattr__(self, "parameters", names)
        if self.initial_guess is not None:
            guess = _vector(self.initial_guess, "initial_guess")
            if len(guess) != len(names):
                raise ValueError("initial_guess must match the number of parameters.")
            object.__setattr__(self, "initial_guess", tuple(guess))

    def __call__(self, x: ArrayLike, *parameters: float) -> NDArray[np.float64]:
        if len(parameters) != len(self.parameters):
            raise ValueError(f"{self.name} expects {len(self.parameters)} parameters.")
        x_array = np.asarray(x, dtype=np.float64)
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                y = np.asarray(self.function(x_array, *parameters), dtype=np.float64)
            except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                raise ValueError(
                    f"{self.name} is undefined at these x/parameter values. "
                    "Check the equation's domain and initial_guess."
                ) from exc
        if y.shape != x_array.shape:
            raise ValueError(f"{self.name} must return the same shape as x.")
        if not np.all(np.isfinite(y)):
            raise ValueError(f"{self.name} returned nonfinite values.")
        return y


def _linear(x, intercept, slope):
    return intercept + slope * x


def _quadratic(x, intercept, linear, quadratic):
    return intercept + linear * x + quadratic * x**2


def _exponential(x, offset, amplitude, rate):
    return offset + amplitude * np.exp(rate * x)


def _sinusoidal(x, offset, amplitude, angular_frequency, phase):
    return offset + amplitude * np.sin(angular_frequency * x + phase)


linear = Model("Linear", _linear, ("intercept", "slope"), (0.0, 1.0))
quadratic = Model("Quadratic", _quadratic, ("intercept", "linear", "quadratic"), (0, 1, 0))
exponential = Model("Exponential", _exponential, ("offset", "amplitude", "rate"))
sinusoidal = Model("Sinusoidal", _sinusoidal, ("offset", "amplitude", "angular_frequency", "phase"))
