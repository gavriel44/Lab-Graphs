import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from labgraphs import Dataset, FitError, Model, fit, models


def test_wls_matches_analytical_coefficients_covariance_and_chi_square():
    x = np.array([0.0, 0.5, 1.5, 2.0, 4.0, 5.0])
    y = np.array([1.1, 2.2, 3.7, 5.3, 8.8, 11.2])
    sy = np.array([0.1, 0.2, 0.1, 0.4, 0.3, 0.2])
    design = np.column_stack([np.ones_like(x), x])
    weights = np.diag(sy**-2)
    expected_covariance = np.linalg.inv(design.T @ weights @ design)
    expected_beta = expected_covariance @ design.T @ weights @ y
    expected_chi_square = np.sum(((y - design @ expected_beta) / sy) ** 2)

    result = fit(Dataset(x, y, sy=sy), models.linear, initial_guess=np.array([0.5, 1.5]))
    assert result.method == "WLS"
    assert_allclose(result.parameters, expected_beta, rtol=1e-5)
    assert_allclose(result.covariance, expected_covariance, rtol=2e-4)
    assert_allclose(result.standard_errors**2, np.diag(result.covariance))
    assert_allclose(result.chi_square, expected_chi_square, rtol=1e-7)
    assert_allclose(result.p_value, chi2.sf(expected_chi_square, 4), rtol=1e-6)
    assert result.degrees_of_freedom == 4
    assert_allclose(result.x_adjustments, 0)
    assert_allclose(result.predict(x), design @ expected_beta, rtol=1e-5)
    assert result.parameter_values["slope"] == result.parameters[1]
    assert "intercept:" in result.summary()
    assert "absolute sigma" in result.summary()


def test_odr_matches_deming_solution_and_both_axis_objective():
    x = np.array([0, 1.5, 1.8, 3.5, 3.7, 5.5, 5.8, 7.5], dtype=float)
    y = np.array([0.4, 2.8, 4.3, 6.7, 8.0, 10.7, 12.1, 14.5])
    sx, sy = 0.45, 0.2
    xc, yc = x - x.mean(), y - y.mean()
    sxx, syy, sxy = xc @ xc, yc @ yc, xc @ yc
    ratio = sy**2 / sx**2
    slope = (syy - ratio * sxx + np.sqrt((syy - ratio * sxx) ** 2 + 4 * ratio * sxy**2)) / (2 * sxy)
    intercept = y.mean() - slope * x.mean()

    result = fit(Dataset(x, y, sx=sx, sy=sy), models.linear)
    wls = fit(Dataset(x, y, sy=sy), models.linear)
    assert result.method == "ODR"
    assert_allclose(result.parameters, [intercept, slope], rtol=2e-5, atol=2e-5)
    assert abs(result.parameters[1] - wls.parameters[1]) > 0.01
    objective = np.sum((result.x_adjustments / sx) ** 2 + (result.y_adjustments / sy) ** 2)
    assert_allclose(result.chi_square, objective, rtol=1e-7)
    assert_allclose(result.predict(x + result.x_adjustments), y + result.y_adjustments)
    assert_allclose(result.residuals, y - result.predict(x))


@pytest.mark.parametrize("sx", [None, 0.1])
def test_covariance_scaling_is_explicit_and_consistent(sx):
    data = Dataset([0, 1, 2, 3, 4], [1.1, 3.5, 4.5, 7.2, 8.9], sy=0.1, sx=sx)
    absolute = fit(data, models.linear)
    scaled = fit(data, models.linear, absolute_sigma=False)
    assert_allclose(scaled.parameters, absolute.parameters)
    assert_allclose(scaled.covariance, absolute.covariance * absolute.reduced_chi_square)
    assert_allclose(scaled.standard_errors**2, np.diag(scaled.covariance))
    assert scaled.p_value == absolute.p_value
    assert "scaled sigma" in scaled.summary()


@pytest.mark.parametrize(
    ("model", "parameters", "guess", "x"),
    [
        (models.quadratic, [1, 2, 0.5], None, np.linspace(-2, 2, 30)),
        (models.exponential, [0.3, 3, -0.7], [0.2, 2.5, -0.6], np.linspace(0, 5, 50)),
        (models.sinusoidal, [0.2, 2, 1.5, 0.4], [0, 1.8, 1.4, 0.3], np.linspace(0, 8, 80)),
    ],
)
def test_models_recover_known_parameters(model, parameters, guess, x):
    rng = np.random.default_rng(81)
    y = model(x, *parameters) + rng.normal(0, 0.005, len(x))
    result = fit(Dataset(x, y, sy=0.005), model, initial_guess=guess)
    assert_allclose(result.parameters, parameters, atol=0.015)


def test_custom_model():
    custom = Model("Square", lambda x, scale: scale * x**2, ("scale",), (1.0,))
    result = fit(Dataset([1, 2, 3, 4], [2.1, 8, 17.9, 32.1], sy=0.1), custom)
    assert_allclose(result.parameters, [2], atol=0.01)
    assert "scale:" in result.summary()
    with pytest.raises(ValueError, match="read-only"):
        result.parameters[0] = 0


def test_exact_line_has_finite_absolute_errors_even_with_zero_intercept():
    result = fit(Dataset([0, 1, 2, 3], [0, 2, 4, 6], sy=0.1), models.linear)
    assert_allclose(result.parameters, [0, 2], atol=1e-8)
    assert np.all(np.isfinite(result.standard_errors))
    assert np.all(result.standard_errors > 0)
    assert "intercept:" in result.summary()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_guess": [1]}, "number of model parameters"),
        ({"initial_guess": [1, np.nan]}, "finite"),
        ({"max_iterations": 0}, "positive integer"),
        ({"max_iterations": True}, "positive integer"),
        ({"max_iterations": 2.5}, "positive integer"),
        ({"absolute_sigma": "yes"}, "boolean"),
    ],
)
def test_invalid_fit_configuration(kwargs, message):
    data = Dataset([0, 1, 2], [0, 1, 2], sy=0.1)
    with pytest.raises(ValueError, match=message):
        fit(data, models.linear, **kwargs)


def test_insufficient_data_and_constant_x():
    with pytest.raises(ValueError, match="more observations"):
        fit(Dataset([0, 1], [1, 2], sy=0.1), models.linear)
    with pytest.raises(ValueError, match="distinct"):
        fit(Dataset([1, 1, 1], [1, 2, 3], sy=0.1), models.linear)


def test_nonlinear_model_requires_starting_values():
    with pytest.raises(ValueError, match="Provide initial_guess"):
        fit(Dataset([1, 2, 3, 4], [1, 2, 4, 8], sy=0.1), models.exponential)


def test_iteration_limit_is_an_error_not_a_successful_result():
    x = np.linspace(0, 5, 30)
    data = Dataset(x, 0.2 + 4 * np.exp(-0.8 * x), sy=0.1)
    with pytest.raises(FitError, match="failed"):
        fit(data, models.exponential, initial_guess=[1, 1, -0.1], max_iterations=1)


def test_unidentifiable_parameters_are_rejected():
    model = Model("Redundant", lambda x, a, b: (a + b) * x, ("a", "b"), (1, 1))
    with pytest.raises(FitError):
        fit(Dataset([0, 1, 2, 3], [0, 2.1, 4.1, 5.9], sy=0.1), model)


@pytest.mark.parametrize("sigma", [1e-200, 1e200])
def test_unrepresentable_weights_are_rejected(sigma):
    with pytest.raises(ValueError, match="rescale your units"):
        fit(Dataset([0, 1, 2], [1, 2, 3], sy=sigma), models.linear)
