import numpy as np


def stable_sort(arr, axis=-1):
    """
    Stable sort wrapper using NumPy.

    Parameters
    ----------
    arr : array-like
        Input array.
    axis : int, default=-1
        Axis along which to sort.

    Returns
    -------
    np.ndarray
        Sorted array.
    """
    arr = np.asarray(arr)
    return np.sort(arr, axis=axis, kind="stable")


def multi_key_sort(arr, keys, ascending=True):
    """
    Sort a 2D array by multiple column keys.

    Parameters
    ----------
    arr : array-like of shape (n_samples, n_features)
        Input 2D array.
    keys : list[int]
        Column indices used for sorting. Earlier keys have higher priority.
    ascending : bool, default=True
        If True, sort ascending. If False, sort descending.

    Returns
    -------
    np.ndarray
        Sorted 2D array.
    """
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError("multi_key_sort only supports 2D arrays.")

    if len(keys) == 0:
        raise ValueError("keys must contain at least one column index.")

    n_cols = arr.shape[1]

    for key in keys:
        if key < 0 or key >= n_cols:
            raise ValueError("key index out of range.")

    order = np.lexsort(tuple(arr[:, key] for key in reversed(keys)))
    sorted_arr = arr[order]

    if not ascending:
        sorted_arr = sorted_arr[::-1]

    return sorted_arr


def topk(values, k, largest=True, return_indices=True):
    """
    Return top-k largest or smallest values using np.argpartition.

    Parameters
    ----------
    values : array-like of shape (n,)
        Input values.
    k : int
        Number of elements to return.
    largest : bool, default=True
        If True, return k largest values. If False, return k smallest values.
    return_indices : bool, default=True
        If True, return indices. Otherwise, return values.

    Returns
    -------
    np.ndarray
        Top-k indices or values.

    Raises
    ------
    ValueError
        If input is not 1D or k is outside valid range.
    """
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError("topk only supports 1D arrays.")

    if k < 0 or k > values.size:
        raise ValueError("k must be between 0 and len(values).")

    if k == 0:
        return np.array([], dtype=int if return_indices else values.dtype)

    if largest:
        indices = np.argpartition(values, -k)[-k:]
        sorted_order = np.argsort(values[indices])[::-1]
    else:
        indices = np.argpartition(values, k - 1)[:k]
        sorted_order = np.argsort(values[indices])

    indices = indices[sorted_order]

    if return_indices:
        return indices

    return values[indices]


def _partition(arr, left, right):
    """
    Partition helper for quickselect.
    """
    pivot = arr[right]
    i = left

    for j in range(left, right):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[right] = arr[right], arr[i]
    return i


def quickselect(arr, k):
    """
    Return the kth smallest element using quickselect.

    Parameters
    ----------
    arr : array-like of shape (n,)
        Input values.
    k : int
        0-based rank of element to find.

    Returns
    -------
    scalar
        kth smallest value.

    Raises
    ------
    ValueError
        If input is empty, not 1D, or k is invalid.
    """
    arr = np.asarray(arr).copy()

    if arr.ndim != 1:
        raise ValueError("quickselect only supports 1D arrays.")

    if arr.size == 0:
        raise ValueError("quickselect is undefined for empty arrays.")

    if k < 0 or k >= arr.size:
        raise ValueError("k must be between 0 and len(arr)-1.")

    left = 0
    right = arr.size - 1

    while left <= right:
        pivot_index = _partition(arr, left, right)

        if pivot_index == k:
            return arr[pivot_index]

        if pivot_index < k:
            left = pivot_index + 1
        else:
            right = pivot_index - 1


def binary_search(sorted_array, x):
    """
    Binary search on a sorted 1D array.

    Parameters
    ----------
    sorted_array : array-like of shape (n,)
        Sorted input array.
    x : scalar
        Value to search.

    Returns
    -------
    tuple[int, bool]
        insertion_index, exists
    """
    sorted_array = np.asarray(sorted_array)

    if sorted_array.ndim != 1:
        raise ValueError("binary_search only supports 1D arrays.")

    idx = np.searchsorted(sorted_array, x, side="left")
    exists = idx < sorted_array.size and sorted_array[idx] == x

    return int(idx), bool(exists)