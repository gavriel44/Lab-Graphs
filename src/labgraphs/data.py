"""Validated measurement data, independent of file formats and plotting."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = _readonly(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values; check for missing data.")
    return array


def _uncertainty(values: ArrayLike, name: str, size: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, array.item())
    array = _vector(array, name)
    if len(array) != size:
        raise ValueError(f"{name} must be a scalar or have the same length as x and y.")
    if np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive standard deviations.")
    return array


@dataclass(frozen=True, eq=False, init=False)
class Dataset:
    """Paired observations with independent, one-sigma measurement uncertainties.

    ``sy`` is required. ``sx=None`` treats x as exact and selects weighted least
    squares. Supplying positive ``sx`` selects orthogonal distance regression.
    Uncertainties may be scalars or arrays. Inputs are copied into read-only arrays.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    sx: NDArray[np.float64] | None
    sy: NDArray[np.float64]
    x_label: str
    y_label: str

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        sy: ArrayLike,
        sx: ArrayLike | None = None,
        x_label: str = "x",
        y_label: str = "y",
    ) -> None:
        x_array = _vector(x, "x")
        y_array = _vector(y, "y")
        if len(x_array) != len(y_array):
            raise ValueError("x and y must have the same length.")
        if not isinstance(x_label, str) or not isinstance(y_label, str):
            raise ValueError("Axis labels must be strings.")
        for name, value in {
            "x": x_array,
            "y": y_array,
            "sx": None if sx is None else _uncertainty(sx, "sx", len(x_array)),
            "sy": _uncertainty(sy, "sy", len(x_array)),
            "x_label": x_label,
            "y_label": y_label,
        }.items():
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return len(self.x)
