import numpy as np
from numcompute.rank import percentile


def _validate_numeric(data):
    arr = np.asarray(data)

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("Only numeric arrays are allowed.")

    if arr.size == 0:
        raise ValueError("Empty array is not allowed.")

    return arr


def _validate_axis(axis, ndim):
    if axis is not None and not isinstance(axis, int):
        raise TypeError("axis must be an integer or None.")

    if axis is not None and not (-ndim <= axis < ndim):
        raise ValueError(f"axis {axis} is out of bounds for array with {ndim} dimensions.")


def mean(data, axis=None, keepdims=False):
    arr = _validate_numeric(data)
    _validate_axis(axis, arr.ndim)

    total = np.sum(arr, axis=axis, keepdims=keepdims)

    if axis is None:
        return total / arr.size

    return total / arr.shape[axis]


def median(data, axis=None):
    arr = _validate_numeric(data)
    _validate_axis(axis, arr.ndim)

    sorted_arr = np.sort(arr, axis=axis, kind="stable")

    if axis is None:
        flat = sorted_arr.ravel()
        n = flat.size
        mid = n // 2

        if n % 2 == 0:
            return (flat[mid - 1] + flat[mid]) / 2

        return flat[mid]

    n = sorted_arr.shape[axis]
    mid = n // 2

    if n % 2 == 0:
        lower = np.take(sorted_arr, mid - 1, axis=axis)
        higher = np.take(sorted_arr, mid, axis=axis)
        return (lower + higher) / 2

    return np.take(sorted_arr, mid, axis=axis)


def std(data, axis=None, ddof=0):
    arr = _validate_numeric(data)
    _validate_axis(axis, arr.ndim)

    if not isinstance(ddof, int):
        raise TypeError("ddof must be an integer.")

    n = arr.size if axis is None else arr.shape[axis]

    if n - ddof <= 0:
        raise ValueError("ddof is too large for the number of values.")

    arr_mean = mean(arr, axis=axis, keepdims=True)
    squared_deviation = (arr - arr_mean) ** 2
    variance = np.sum(squared_deviation, axis=axis) / (n - ddof)

    return np.sqrt(variance)


def min(data, axis=None):
    arr = _validate_numeric(data)
    _validate_axis(axis, arr.ndim)
    return np.min(arr, axis=axis)


def max(data, axis=None):
    arr = _validate_numeric(data)
    _validate_axis(axis, arr.ndim)
    return np.max(arr, axis=axis)


def histogram(data, bins=10):
    arr = _validate_numeric(data).ravel()

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer.")

    min_val = np.min(arr)
    max_val = np.max(arr)

    edges = np.linspace(min_val, max_val, bins + 1)

    if min_val == max_val:
        counts = np.zeros(bins, dtype=int)
        counts[0] = arr.size
        return edges, counts

    bin_indices = np.searchsorted(edges, arr, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    counts = np.bincount(bin_indices, minlength=bins)

    return edges, counts


def quantiles(data, q, interpolation="linear"):
    arr = _validate_numeric(data)
    q_arr = np.asarray(q, dtype=float)

    if np.any(q_arr < 0) or np.any(q_arr > 1):
        raise ValueError("q must be between 0 and 1.")

    clean_arr = arr[~np.isnan(arr)]

    if clean_arr.size == 0:
        raise ValueError("No valid numeric values after removing NaNs.")

    return percentile(clean_arr, q_arr * 100, interpolation=interpolation)