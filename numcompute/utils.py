import numpy as np

# utils.py - shared helpers used across the toolkit
# Activations + logsumexp + distance funcs + topk helper + batch iterator
# Everything is vectorized, nothing here uses python loops on array elements
# (except iter_batches which yields slices, not per-element work)


def sigmoid(x):
    # input: array of any shape (or scalar)
    # output: same shape, all values in (0, 1)
    # complexity: O(n) time, O(n) extra space for the output
    # raises: nothing - non-numeric input will fail at np.array()
    arr = np.array(x, dtype=np.float64)
    # naive 1/(1+exp(-x)) blows up for very negative x since exp(-x) -> huge
    # fix: rewrite as exp(x)/(1+exp(x)) when x<0, keeps exponent <= 0 so safe
    result = np.empty_like(arr)
    pos_mask = arr >= 0
    # positive branch: exp(-x) <= 1 here, no overflow
    result[pos_mask] = 1.0 / (1.0 + np.exp(-arr[pos_mask]))
    # negative branch: exp(x) <= 1 here, no overflow
    exp_x = np.exp(arr[~pos_mask])
    result[~pos_mask] = exp_x / (1.0 + exp_x)
    return result


def relu(x):
    # input: array of any shape, output: same shape with negatives clipped to 0
    # O(n) time and space
    arr = np.array(x)
    # max(0, x) elementwise, just clip the negative side to 0
    return np.maximum(0, arr)


def tanh(x):
    # input: array of any shape, output: same shape, values in (-1, 1)
    # O(n) time and space
    arr = np.array(x)
    # numpy's built-in tanh is already numerically stable
    return np.tanh(arr)


def softmax(x, axis=-1):
    # input: array of any shape, axis: which dim to normalize along
    # output: same shape as input, each slice along axis sums to 1
    # complexity: O(n) where n is total elements
    # numerical note: uses max-shift so exp() never overflows
    arr = np.array(x, dtype=np.float64)
    # max-shift trick: softmax(x) == softmax(x - c) for any constant c
    # because c cancels in the ratio. Subtract max -> largest exponent is 0,
    # rest are negative, so np.exp wont overflow
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def logsumexp(x, axis=None, keepdims=False):
    # input: array, axis to reduce on (None means everything)
    # output: scalar if axis=None, else array with that axis removed
    # (or kept as size 1 if keepdims=True)
    # complexity: O(n) time
    # numerical note: stable form, handles all -inf slices without nans
    arr = np.array(x, dtype=np.float64)
    # naive log(sum(exp(x))) overflows when x has large values
    # identity: log(sum(exp(x))) = m + log(sum(exp(x - m))) where m = max(x)
    # mathematically equal but the exp() inside now stays <= 1
    m = np.max(arr, axis=axis, keepdims=True)
    # if a slice is all -inf, m is -inf and arr-m gives nan -> guard with 0
    safe_m = np.where(np.isfinite(m), m, 0.0)
    summed = np.sum(np.exp(arr - safe_m), axis=axis, keepdims=True)
    result = np.log(summed) + safe_m
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


def euclidean_distance(a, b):
    # inputs: a, b same shape - either two 1D vectors or two 2D arrays
    # output: scalar if 1D inputs, 1D array of length n if (n,d) inputs
    # complexity: O(n*d) time
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    # L2 distance -> sqrt(sum((a-b)^2)), reduce on last axis so it works
    # for both 1D vectors and row-wise on 2D matrices
    diff = arr_a - arr_b
    squared = diff * diff
    return np.sqrt(np.sum(squared, axis=-1))


def manhattan_distance(a, b):
    # same shape rules as euclidean_distance
    # L1 distance instead of L2 - sum of abs differences
    # O(n*d) time
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    # L1 distance -> sum(|a-b|) along last axis
    return np.sum(np.abs(arr_a - arr_b), axis=-1)


def cosine_distance(a, b):
    # inputs: same shape, output: distance value(s) in [0, 2]
    # 0 = identical direction, 1 = orthogonal, 2 = opposite direction
    # O(n*d) time
    # eps guards zero vectors from division by zero (returns ~1)
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    # cosine sim = (a . b) / (||a|| * ||b||), distance = 1 - sim
    norm_a = np.linalg.norm(arr_a, axis=-1)
    norm_b = np.linalg.norm(arr_b, axis=-1)
    dot = np.sum(arr_a * arr_b, axis=-1)
    eps = 1e-12  # tiny guard so zero-norm vectors dont blow up the division
    return 1.0 - dot / (norm_a * norm_b + eps)


def pairwise_distances(X, Y=None, metric='euclidean'):
    # inputs: X is (n, d), Y is (m, d) or None (then compares X with itself)
    # output: (n, m) matrix where result[i,j] = distance(X[i], Y[j])
    # raises: ValueError if metric is unknown
    # complexity: euclidean and cosine are O(n*m*d) but use a matmul
    # so its way faster than a python double loop. manhattan is O(n*m*d) too
    # but materializes the full (n,m,d) diff so it eats more memory
    arr_X = np.array(X, dtype=np.float64)
    if Y is None:
        arr_Y = arr_X
    else:
        arr_Y = np.array(Y, dtype=np.float64)
    # returns a matrix of shape (len(X), len(Y))
    match(metric.lower()):
        case 'euclidean':
            # expand ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
            # avoids a triple loop -> just 2 sums + 1 matmul
            x_sq = np.sum(arr_X * arr_X, axis=1, keepdims=True)
            y_sq = np.sum(arr_Y * arr_Y, axis=1, keepdims=True).T
            cross = arr_X @ arr_Y.T
            sq_dist = x_sq + y_sq - 2.0 * cross
            sq_dist = np.maximum(sq_dist, 0.0)
            return np.sqrt(sq_dist)
        case 'manhattan':
            return np.sum(np.abs(arr_X[:, None, :] - arr_Y[None, :, :]), axis=-1)
        case 'cosine':
            X_norm = arr_X / (np.linalg.norm(arr_X, axis=1, keepdims=True) + 1e-12)
            Y_norm = arr_Y / (np.linalg.norm(arr_Y, axis=1, keepdims=True) + 1e-12)
            return 1.0 - X_norm @ Y_norm.T
        case _:
            raise ValueError(f"unknown metric: {metric}")


def topk_indices(values, k, largest=True):
    # input: values is any array, k is how many to pick
    # output: indices of the k extreme values along the last axis,
    # ordered from most-extreme to least-extreme
    # raises: ValueError if k <= 0 (k > axis-length is silently clamped)
    # complexity: O(n) for the partition + O(k log k) to sort the k chosen
    arr = np.array(values)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    n = arr.shape[-1]
    if k > n:
        k = n
    if largest:
        part_idx = np.argpartition(-arr, k - 1, axis=-1)[..., :k]
    else:
        part_idx = np.argpartition(arr, k - 1, axis=-1)[..., :k]
    part_vals = np.take_along_axis(arr, part_idx, axis=-1)
    if largest:
        order = np.argsort(-part_vals, axis=-1)
    else:
        order = np.argsort(part_vals, axis=-1)
    return np.take_along_axis(part_idx, order, axis=-1)


def iter_batches(n_samples, batch_size, shuffle=False, seed=None):
    # generator - yields 1D index arrays one batch at a time
    # last batch may be smaller if n_samples doesnt divide evenly
    # raises: ValueError on bad inputs (negative n, non-positive batch_size)
    # complexity: O(n) memory for the index array, then constant per yield
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if n_samples < 0:
        raise ValueError("n_samples cannot be negative")
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        yield indices[start:start + batch_size]