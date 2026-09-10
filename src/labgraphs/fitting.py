"""Weighted regression with explicit uncertainty and convergence semantics."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from odrpack import odr_fit
from scipy.stats import chi2 as chi2_distribution

from .data import Dataset, _readonly, _vector
from .models import Model


class FitError(RuntimeError):
    """The solver failed to produce a converged, full-rank, finite result."""


@dataclass(frozen=True, eq=False)
class FitResult:
    """A successful fit. Parameter arrays follow ``model.parameters`` order.

    ``covariance`` and ``standard_errors`` use the same scaling. With the default
    ``absolute_sigma=True``, supplied uncertainties set the absolute scale.
    ``residuals`` are vertical y deviations at observed x; ``x_adjustments`` and
    ``y_adjustments`` instead describe the points projected onto the ODR curve.
    """

    data: Dataset
    model: Model
    parameters: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    covariance: NDArray[np.float64]
    residuals: NDArray[np.float64]
    x_adjustments: NDArray[np.float64]
    y_adjustments: NDArray[np.float64]
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    method: Literal["ODR", "WLS"]
    absolute_sigma: bool
    stop_reason: str

    @property
    def reduced_chi_square(self) -> float:
        return self.chi_square / self.degrees_of_freedom

    @property
    def parameter_values(self) -> dict[str, float]:
        """A name-to-value mapping suitable for downstream calculations."""
        return dict(zip(self.model.parameters, map(float, self.parameters), strict=True))

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the fitted model, without adding a prediction interval."""
        return self.model(x, *self.parameters)

    def summary(self) -> str:
        """Format parameters and diagnostics without printing as a side effect."""
        lines = [f"{self.model.name} fit ({self.method}, {len(self.data)} observations)"]
        for name, value, error in zip(
            self.model.parameters, self.parameters, self.standard_errors, strict=True
        ):
            lines.append(f"  {name}: {value:.6g} ± {error:.3g}")
        lines.extend(
            [
                f"chi-square: {self.chi_square:.6g}",
                f"reduced chi-square: {self.reduced_chi_square:.6g}",
                f"degrees of freedom: {self.degrees_of_freedom}",
                f"p-value (approximate): {self.p_value:.6g}",
                "parameter errors: "
                + ("absolute sigma" if self.absolute_sigma else "scaled sigma"),
                f"solver: {self.stop_reason}",
            ]
        )
        return "\n".join(lines)


def fit(
    data: Dataset,
    model: Model,
    *,
    initial_guess: ArrayLike | None = None,
    absolute_sigma: bool = True,
    max_iterations: int = 200,
) -> FitResult:
    """Fit a model using ODR when sx is given, or weighted least squares otherwise.

    ``absolute_sigma=False`` multiplies parameter covariance by reduced chi-square.
    It does not change the objective or fitted parameters. A failed solver raises
    ``FitError``; invalid data/model configuration raises ``ValueError``.
    """
    if not isinstance(absolute_sigma, bool):
        raise ValueError("absolute_sigma must be a boolean.")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")
    guess = model.initial_guess if initial_guess is None else initial_guess
    if guess is None:
        raise ValueError(f"Provide initial_guess for the {model.name} model: {model.parameters}.")
    beta0 = _vector(guess, "initial_guess")
    if len(beta0) != len(model.parameters):
        raise ValueError("initial_guess must match the number of model parameters.")
    dof = len(data) - len(beta0)
    if dof <= 0:
        raise ValueError("Fitting requires more observations than model parameters.")
    if np.ptp(data.x) == 0:
        raise ValueError("x must contain at least two distinct values.")
    model(data.x, *beta0)

    # ODRPACK accepts inverse variances, not standard deviations.
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        weight_y = np.square(1.0 / data.sy)
        weight_x = None if data.sx is None else np.square(1.0 / data.sx)
    for weights in (weight_x, weight_y):
        if weights is not None and (not np.all(np.isfinite(weights)) or np.any(weights == 0)):
            raise ValueError("Uncertainty magnitudes exceed numerical limits; rescale your units.")

    output = odr_fit(
        lambda x, beta: model(x, *beta),
        data.x.copy(),
        data.y.copy(),
        beta0.copy(),
        weight_x=weight_x,
        weight_y=weight_y,
        task="OLS" if data.sx is None else "explicit-ODR",
        maxit=max_iterations,
        diff_scheme="central",
    )
    if not output.success or output.irank != 0:
        raise FitError(
            f"{model.name} fit failed: {output.stopreason} "
            "Check model identifiability, initial_guess, data range, or max_iterations."
        )
    # Finite differences can make exactly redundant parameters appear full-rank.
    # Reject fits that have lost roughly half of float64's significant digits.
    if not np.isfinite(output.inv_condnum) or output.inv_condnum <= np.sqrt(np.finfo(float).eps):
        raise FitError(
            f"{model.name} fit is ill-conditioned: parameters cannot be reliably separated. "
            "Remove redundant parameters or rescale the data/parameter units."
        )
    chi_square = float(output.sum_square)
    covariance = np.asarray(output.cov_beta) * (1.0 if absolute_sigma else chi_square / dof)
    if (
        not np.isfinite(chi_square)
        or chi_square < 0
        or not np.all(np.isfinite(output.beta))
        or not np.all(np.isfinite(covariance))
        or np.any(np.diag(covariance) < 0)
    ):
        raise FitError("The solver returned invalid parameters, covariance, or chi-square.")
    return FitResult(
        data=data,
        model=model,
        parameters=_readonly(output.beta),
        standard_errors=_readonly(np.sqrt(np.diag(covariance))),
        covariance=_readonly(covariance),
        residuals=_readonly(data.y - model(data.x, *output.beta)),
        x_adjustments=_readonly(output.delta),
        y_adjustments=_readonly(output.eps),
        chi_square=chi_square,
        degrees_of_freedom=dof,
        p_value=float(chi2_distribution.sf(chi_square, dof)),
        method="WLS" if data.sx is None else "ODR",
        absolute_sigma=absolute_sigma,
        stop_reason=output.stopreason,
    )
