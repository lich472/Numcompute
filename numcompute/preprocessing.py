import numpy as np

__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder"]


class StandardScaler:
    """
    Standardize features using z-score scaling.

    Formula
    -------
    X_scaled = (X - mean) / std

    Handles NaN values using np.nanmean and np.nanvar.
    Constant columns are scaled using std = 1 to avoid division by zero.
    """

    def __init__(self, copy=True):
        self.copy = copy
        self.fitted = False

    def fit(self, X):
        """
        Compute mean and standard deviation for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("StandardScaler expects 1D or 2D input.")

        if X.shape[0] == 0:
            raise ValueError("StandardScaler: no data to fit on.")

        self.mean = np.nanmean(X, axis=0)
        self.var = np.nanvar(X, axis=0)
        self.scale = np.where(self.var == 0, 1.0, np.sqrt(self.var))
        self.n_features = X.shape[1]
        self.fitted = True
        return self

    def transform(self, X):
        """
        Apply standard scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Scaled data.
        """
        if not self.fitted:
            raise RuntimeError("StandardScaler: call fit() before transform().")

        X = np.array(X, dtype=np.float64) if self.copy else np.asarray(X, dtype=np.float64)

        one_dimensional = X.ndim == 1
        if one_dimensional:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("StandardScaler expects 1D or 2D input.")

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"StandardScaler: expected {self.n_features} features but got {X.shape[1]}."
            )

        X_scaled = (X - self.mean) / self.scale
        return X_scaled.squeeze(axis=1) if one_dimensional else X_scaled

    def fit_transform(self, X):
        """
        Fit the scaler and transform X.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reverse standard scaling.
        """
        if not self.fitted:
            raise RuntimeError("StandardScaler: call fit() before inverse_transform().")

        X = np.array(X, dtype=np.float64)

        one_dimensional = X.ndim == 1
        if one_dimensional:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("StandardScaler expects 1D or 2D input.")

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"StandardScaler: expected {self.n_features} features but got {X.shape[1]}."
            )

        X_original = X * self.scale + self.mean
        return X_original.squeeze(axis=1) if one_dimensional else X_original


class MinMaxScaler:
    """
    Scale features to a fixed range.

    Formula
    -------
    X_scaled = X * scale + min

    Handles NaN values using np.nanmin and np.nanmax.
    Constant columns are protected against division by zero.
    """

    def __init__(self, feature_range=(0, 1), copy=True):
        lower, upper = feature_range

        if lower >= upper:
            raise ValueError(
                f"MinMaxScaler: feature_range needs min < max, got {feature_range}."
            )

        self.feature_range = feature_range
        self.copy = copy
        self.fitted = False

    def fit(self, X):
        """
        Compute min and max values for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects 1D or 2D input.")

        if X.shape[0] == 0:
            raise ValueError("MinMaxScaler: no data to fit on.")

        lower, upper = self.feature_range

        self.data_min = np.nanmin(X, axis=0)
        self.data_max = np.nanmax(X, axis=0)
        self.data_range = self.data_max - self.data_min
        self.scale = (upper - lower) / np.where(self.data_range == 0, 1.0, self.data_range)
        self.min = lower - self.data_min * self.scale
        self.n_features = X.shape[1]
        self.fitted = True
        return self

    def transform(self, X):
        """
        Apply min-max scaling.
        """
        if not self.fitted:
            raise RuntimeError("MinMaxScaler: call fit() before transform().")

        X = np.array(X, dtype=np.float64) if self.copy else np.asarray(X, dtype=np.float64)

        one_dimensional = X.ndim == 1
        if one_dimensional:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects 1D or 2D input.")

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"MinMaxScaler: expected {self.n_features} features but got {X.shape[1]}."
            )

        X_scaled = X * self.scale + self.min
        return X_scaled.squeeze(axis=1) if one_dimensional else X_scaled

    def fit_transform(self, X):
        """
        Fit the scaler and transform X.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reverse min-max scaling.
        """
        if not self.fitted:
            raise RuntimeError("MinMaxScaler: call fit() before inverse_transform().")

        X = np.array(X, dtype=np.float64)

        one_dimensional = X.ndim == 1
        if one_dimensional:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects 1D or 2D input.")

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"MinMaxScaler: expected {self.n_features} features but got {X.shape[1]}."
            )

        X_original = (X - self.min) / self.scale
        return X_original.squeeze(axis=1) if one_dimensional else X_original


class OneHotEncoder:
    """
    One-hot encode categorical features.

    Parameters
    ----------
    handle_unknown : {'ignore', 'error'}, default='ignore'
        Strategy for unknown categories during transform.
    sparse : bool, default=False
        Present for API compatibility. Output is dense NumPy array.
    """

    def __init__(self, handle_unknown="ignore", sparse=False):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError(
                f"handle_unknown must be 'ignore' or 'error', got '{handle_unknown}'."
            )

        self.handle_unknown = handle_unknown
        self.sparse = sparse
        self.fitted = False

    def fit(self, X):
        """
        Learn unique categories for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("OneHotEncoder expects 1D or 2D input.")

        if X.shape[0] == 0:
            raise ValueError("OneHotEncoder: no data to fit on.")

        self.n_features_in = X.shape[1]
        self.categories = [np.unique(X[:, j]) for j in range(self.n_features_in)]
        self.fitted = True
        return self

    def transform(self, X):
        """
        Transform categories into one-hot encoded columns.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        np.ndarray
            One-hot encoded dense matrix.
        """
        if not self.fitted:
            raise RuntimeError("OneHotEncoder: call fit() first.")

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("OneHotEncoder expects 1D or 2D input.")

        if X.shape[0] == 0:
            raise ValueError("OneHotEncoder: no data to transform.")

        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f"OneHotEncoder: expected {self.n_features_in} features, got {X.shape[1]}."
            )

        encoded_parts = []

        for j, categories in enumerate(self.categories):
            column = X[:, j]
            indicator = (column[:, np.newaxis] == categories[np.newaxis, :]).astype(
                np.float64
            )

            unknown_mask = indicator.sum(axis=1) == 0

            if self.handle_unknown == "error" and unknown_mask.any():
                raise ValueError(f"OneHotEncoder: unknown category found in column {j}.")

            encoded_parts.append(indicator)

        return np.hstack(encoded_parts)

    def fit_transform(self, X):
        """
        Fit the encoder and transform X.
        """
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names.

        Parameters
        ----------
        input_features : list[str], optional
            Original input feature names.

        Returns
        -------
        list[str]
            Generated one-hot feature names.
        """
        if not self.fitted:
            raise RuntimeError("OneHotEncoder: call fit() first.")

        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in)]

        if len(input_features) != self.n_features_in:
            raise ValueError(
                f"Expected {self.n_features_in} input feature names, got {len(input_features)}."
            )

        return [
            f"{feature}_{category}"
            for feature, categories in zip(input_features, self.categories)
            for category in categories
        ]