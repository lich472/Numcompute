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

class StreamingMetrics:
    def __init__(self, labels=None, positive_label=1, window_size=None):
        self.labels = labels
        self.positive_label = positive_label
        self.window_size = window_size
        self.y_true_chunks = []
        self.y_pred_chunks = []
        self.y_score_chunks = []

    def update(self, y_true_chunk, y_pred_chunk, y_score_chunk=None):
        y_t, y_p = _validate_same_shape(y_true_chunk, y_pred_chunk)
        
        if y_score_chunk is None:
            y_s = y_p.astype(float)
        else:
            y_s = np.asarray(y_score_chunk)
        
        self.y_true_chunks.append(y_t)
        self.y_pred_chunks.append(y_p)
        self.y_score_chunks.append(y_s)

        # drop old chunks if window_size is set
        if self.window_size is not None and len(self.y_true_chunks) > self.window_size:
            self.y_true_chunks.pop(0)
            self.y_pred_chunks.pop(0)
            self.y_score_chunks.pop(0)

    def reset(self):
        self.y_true_chunks = []
        self.y_pred_chunks = []
        self.y_score_chunks = []

    def result(self):
        if len(self.y_true_chunks) == 0:
            raise ValueError("No data yet, call update() first.")

        y_true = np.concatenate(self.y_true_chunks)
        y_pred = np.concatenate(self.y_pred_chunks)
        y_score = np.concatenate(self.y_score_chunks)

        pos_scores = []
        neg_scores = []
        for i in range(len(y_true)):
            if y_true[i] == self.positive_label:
                pos_scores.append(y_score[i])
            else:
                neg_scores.append(y_score[i])
        
        # If the history only contains 1 class, AUC cannot be computed
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            auc_val = 0
        else:
            matches = 0
            for p in pos_scores:
                for n in neg_scores:
                    if p > n:
                        matches += 1
                    elif p == n:
                        matches += 0.5
            auc_val = matches / (len(pos_scores) * len(neg_scores))

        return {
            'accuracy': accuracy(y_true, y_pred),
            'precision': precision(y_true, y_pred, self.positive_label),
            'recall': recall(y_true, y_pred, self.positive_label),
            'f1': f1(y_true, y_pred, self.positive_label),
            # Added self.labels to keep matrix size fixed
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=self.labels),
            'auc': auc_val
        }
