import numpy as np


def _validate_same_shape(y_true, y_pred):
    """
    Validate that y_true and y_pred have the same shape.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels or values.
    y_pred : array-like of shape (n_samples,)
        Predicted labels or values.

    Returns
    -------
    y_true_arr : np.ndarray
        Validated 1D NumPy array.
    y_pred_arr : np.ndarray
        Validated 1D NumPy array.

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true_arr.shape}, "
            f"but y_pred has shape {y_pred_arr.shape}."
        )

    return y_true_arr, y_pred_arr


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    float
        Fraction of correctly predicted labels.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes.
    """
    y_true_arr, y_pred_arr = _validate_same_shape(y_true, y_pred)

    if y_true_arr.size == 0:
        raise ValueError("accuracy is undefined for empty arrays.")

    return np.mean(y_true_arr == y_pred_arr)

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute the confusion matrix for classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    labels : array-like of shape (n_classes,), optional
        List of labels to index the matrix. If None, labels are inferred
        from the sorted unique values in y_true and y_pred combined.

    Returns
    -------
    np.ndarray of shape (n_classes, n_classes)
        Confusion matrix where rows represent true labels and columns
        represent predicted labels.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes.
        If input arrays are empty.
    """
    y_true_arr, y_pred_arr = _validate_same_shape(y_true, y_pred)

    if y_true_arr.size == 0:
        raise ValueError("confusion_matrix is undefined for empty arrays.")

    if labels is None:
        labels = np.unique(np.concatenate((y_true_arr, y_pred_arr)))
    else:
        labels = np.asarray(labels)

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((labels.size, labels.size), dtype=int)

    for true_label, pred_label in zip(y_true_arr, y_pred_arr):
        if true_label in label_to_index and pred_label in label_to_index:
            i = label_to_index[true_label]
            j = label_to_index[pred_label]
            matrix[i, j] += 1

    return matrix


def mse(y_true, y_pred):
    """
    Compute mean squared error for regression.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        Mean squared error.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes.
        If input arrays are empty.
    """
    y_true_arr, y_pred_arr = _validate_same_shape(y_true, y_pred)

    if y_true_arr.size == 0:
        raise ValueError("mse is undefined for empty arrays.")

    y_true_arr = y_true_arr.astype(float)
    y_pred_arr = y_pred_arr.astype(float)

    return np.mean((y_true_arr - y_pred_arr) ** 2)

def precision(y_true, y_pred, positive_label=1):
    """
    Compute precision for binary classification.

    Precision = TP / (TP + FP)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    positive_label : label considered as positive class

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If input arrays are empty or shapes mismatch.
    """
    y_true_arr, y_pred_arr = _validate_same_shape(y_true, y_pred)

    if y_true_arr.size == 0:
        raise ValueError("precision is undefined for empty arrays.")

    tp = np.sum((y_true_arr == positive_label) & (y_pred_arr == positive_label))
    fp = np.sum((y_true_arr != positive_label) & (y_pred_arr == positive_label))

    denom = tp + fp
    if denom == 0:
        return 0.0  # avoid division by zero

    return tp / denom


def recall(y_true, y_pred, positive_label=1):
    """
    Compute recall for binary classification.

    Recall = TP / (TP + FN)
    """
    y_true_arr, y_pred_arr = _validate_same_shape(y_true, y_pred)

    if y_true_arr.size == 0:
        raise ValueError("recall is undefined for empty arrays.")

    tp = np.sum((y_true_arr == positive_label) & (y_pred_arr == positive_label))
    fn = np.sum((y_true_arr == positive_label) & (y_pred_arr != positive_label))

    denom = tp + fn
    if denom == 0:
        return 0.0

    return tp / denom


def f1(y_true, y_pred, positive_label=1):
    """
    Compute F1 score for binary classification.

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)

    denom = p + r
    if denom == 0:
        return 0.0

    return 2 * (p * r) / denom