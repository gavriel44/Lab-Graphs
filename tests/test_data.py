import numpy as np
import pytest
from numpy.testing import assert_array_equal

from labgraphs import Dataset


def test_inputs_are_copied_and_uncertainties_broadcast():
    x = np.arange(4.0)
    data = Dataset(x, x**2, sy=0.2, sx=[0.1] * 4, x_label="Time (s)")
    x[0] = 99
    assert data.x[0] == 0
    assert len(data) == 4
    assert data.x_label == "Time (s)"
    assert_array_equal(data.sy, [0.2] * 4)
    with pytest.raises(ValueError, match="read-only"):
        data.y[0] = 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"x": []}, "nonempty"),
        ({"x": [[1, 2, 3]]}, "one-dimensional"),
        ({"x": [1, 2]}, "same length"),
        ({"y": [1, np.nan, 3]}, "finite"),
        ({"x": [1, np.inf, 3]}, "finite"),
        ({"y": [1, "bad", 3]}, "numeric"),
        ({"sy": 0}, "strictly positive"),
        ({"sx": -1}, "strictly positive"),
        ({"sx": [0.1, 0, 0.1]}, "strictly positive"),
        ({"sy": [0.1, 0.2]}, "same length"),
        ({"sy": np.inf}, "finite"),
        ({"x_label": None}, "labels"),
    ],
)
def test_invalid_measurements_fail_early(overrides, message):
    args = {"x": [1, 2, 3], "y": [2, 4, 6], "sy": 0.1}
    args.update(overrides)
    with pytest.raises(ValueError, match=message):
        Dataset(**args)
