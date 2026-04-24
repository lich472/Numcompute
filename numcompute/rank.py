import numpy as np


def rank(data, method="average"):
    """
    Compute 0-based ranks for 1D data.

    Parameters
    ----------
    data : array-like of shape (n,)
        Input values.
    method : {'average', 'dense', 'ordinal'}, default='average'
        Ranking method.

    Returns
    -------
    np.ndarray
        Rank values.
    """
    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError("rank only supports 1D arrays.")

    if data.size == 0:
        raise ValueError("rank is undefined for empty arrays.")

    method = method.lower()

    if method == "dense":
        unique_values = np.unique(data)
        return np.searchsorted(unique_values, data)

    if method == "ordinal":
        return np.argsort(np.argsort(data, kind="stable"), kind="stable")

    if method == "average":
        ordinal_rank = np.argsort(np.argsort(data, kind="stable"), kind="stable")
        _, inverse, counts = np.unique(data, return_inverse=True, return_counts=True)
        rank_sums = np.bincount(inverse, weights=ordinal_rank)
        avg_ranks = rank_sums / counts
        return avg_ranks[inverse]

    raise ValueError("method must be 'average', 'dense', or 'ordinal'.")


def percentile(data, q, interpolation="linear"):
    """
    Compute percentile values for 1D numeric data.

    Parameters
    ----------
    data : array-like of shape (n,)
        Input numeric data.
    q : float or array-like
        Percentile value(s), between 0 and 100.
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, default='linear'
        Interpolation method.

    Returns
    -------
    float or np.ndarray
        Percentile value(s).
    """
    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError("percentile only supports 1D arrays.")

    if data.size == 0:
        raise ValueError("percentile is undefined for empty arrays.")

    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Only numeric arrays are allowed.")

    q_array = np.asarray(q, dtype=float)

    if np.any(q_array < 0) or np.any(q_array > 100):
        raise ValueError("q must be between 0 and 100.")

    sorted_arr = np.sort(data, kind="stable")
    position = (q_array / 100) * (data.size - 1)

    lower_idx = np.floor(position).astype(int)
    higher_idx = np.ceil(position).astype(int)

    interpolation = interpolation.lower()

    if interpolation == "lower":
        result = sorted_arr[lower_idx]
    elif interpolation == "higher":
        result = sorted_arr[higher_idx]
    elif interpolation == "midpoint":
        result = (sorted_arr[lower_idx] + sorted_arr[higher_idx]) / 2
    elif interpolation == "linear":
        fraction = position - lower_idx
        result = sorted_arr[lower_idx] + fraction * (
            sorted_arr[higher_idx] - sorted_arr[lower_idx]
        )
    else:
        raise ValueError(
            "interpolation must be 'linear', 'lower', 'higher', or 'midpoint'."
        )

    if np.ndim(result) == 0:
        return float(result)

    return result