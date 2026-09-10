import numpy as np
import pytest
from numpy.testing import assert_allclose

from labgraphs import Model, models


def test_model_parameter_order():
    assert_allclose(models.linear([0, 1, 2], 3, 4), [3, 7, 11])
    assert_allclose(models.quadratic([0, 1, 2], 1, 2, 3), [1, 6, 17])
    assert_allclose(models.exponential([0, 1], 1, 2, -1), [3, 1 + 2 / np.e])
    assert_allclose(models.sinusoidal([0, np.pi / 2], 1, 2, 1, 0), [1, 3])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"name": ""}, "nonempty name"),
        ({"function": None}, "callable"),
        ({"parameters": ()}, "nonempty names"),
        ({"parameters": ("a", "a")}, "unique"),
        ({"parameters": ("a", "")}, "nonempty names"),
        ({"initial_guess": (1, 2)}, "number of parameters"),
    ],
)
def test_invalid_model_definition(kwargs, error):
    config = {"name": "Custom", "function": lambda x, a: a * x, "parameters": ("a",)}
    config.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=error):
        Model(**config)


def test_model_contract_and_domain_failures_are_clear():
    with pytest.raises(ValueError, match="expects 2"):
        models.linear([0, 1], 1)
    scalar = Model("Scalar", lambda x, a: a, ("a",))
    with pytest.raises(ValueError, match="same shape"):
        scalar([0, 1], 1)
    invalid = Model("Invalid", lambda x, a: np.full_like(x, np.nan), ("a",))
    with pytest.raises(ValueError, match="nonfinite"):
        invalid([0, 1], 1)
    with pytest.raises(ValueError, match="undefined"):
        models.exponential([0, 1000], 0, 1, 1)
