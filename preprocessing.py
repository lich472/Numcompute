import numpy as np

__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder"]


class StandardScaler:

    def __init__(self, copy=True):
        self.copy = copy
        self.fitted = False

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.shape[0] == 0:
            raise ValueError("StandardScaler: no data to fit on.")
        self.mean = np.nanmean(X, axis=0)
        self.var = np.nanvar(X, axis=0)
        self.scale = np.where(self.var == 0, 1.0, np.sqrt(self.var))
        self.n_features = X.shape[1]
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError(
                "StandardScaler: call fit() before transform().")
        X = np.array(X, dtype=np.float64) if self.copy else np.asarray(
            X, dtype=np.float64)
        sq = X.ndim == 1
        if sq:
            X = X[:, np.newaxis]
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"StandardScaler: expected {self.n_features} features but got {X.shape[1]}.")
        X = (X - self.mean) / self.scale
        return X.squeeze(axis=1) if sq else X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if not self.fitted:
            raise RuntimeError(
                "StandardScaler: call fit() before inverse_transform().")
        X = np.array(X, dtype=np.float64)
        sq = X.ndim == 1
        if sq:
            X = X[:, np.newaxis]
        X = X * self.scale + self.mean
        return X.squeeze(axis=1) if sq else X


class MinMaxScaler:

    def __init__(self, feature_range=(0, 1), copy=True):
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(
                f"MinMaxScaler: feature_range needs min < max, got {feature_range}.")
        self.feature_range = feature_range
        self.copy = copy
        self.fitted = False

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.shape[0] == 0:
            raise ValueError("MinMaxScaler: no data to fit on.")
        lo, hi = self.feature_range
        self.data_min = np.nanmin(X, axis=0)
        self.data_max = np.nanmax(X, axis=0)
        self.data_range = self.data_max - self.data_min
        self.scale = (hi - lo) / np.where(self.data_range ==
                                          0, 1.0, self.data_range)
        self.min = lo - self.data_min * self.scale
        self.n_features = X.shape[1]
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("MinMaxScaler: call fit() before transform().")
        X = np.array(X, dtype=np.float64) if self.copy else np.asarray(
            X, dtype=np.float64)
        sq = X.ndim == 1
        if sq:
            X = X[:, np.newaxis]
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"MinMaxScaler: got {X.shape[1]} features, expected {self.n_features}.")
        X = X * self.scale + self.min
        return X.squeeze(axis=1) if sq else X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if not self.fitted:
            raise RuntimeError(
                "MinMaxScaler: call fit() before inverse_transform().")
        X = np.array(X, dtype=np.float64)
        sq = X.ndim == 1
        if sq:
            X = X[:, np.newaxis]
        X = (X - self.min) / self.scale
        return X.squeeze(axis=1) if sq else X


class OneHotEncoder:

    def __init__(self, handle_unknown="ignore", sparse=False):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError(
                f"handle_unknown must be 'ignore' or 'error', got '{handle_unknown}'.")
        self.handle_unknown = handle_unknown
        self.sparse = sparse
        self.fitted = False

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X))
        if X.shape[0] == 0:
            raise ValueError("OneHotEncoder: no data to fit on.")
        if X.shape[0] == 1:
            X = X.T
        self.n_features_in = X.shape[1]
        self.categories = [np.unique(X[:, j])
                           for j in range(self.n_features_in)]
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("OneHotEncoder: call fit() first.")
        X = np.atleast_2d(np.asarray(X))
        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f"OneHotEncoder: expected {self.n_features_in} features, got {X.shape[1]}.")
        parts = []
        for j, cats in enumerate(self.categories):
            col = X[:, j]
            indicator = (col[:, np.newaxis] ==
                         cats[np.newaxis, :]).astype(np.float64)
            if self.handle_unknown == "error" and (indicator.sum(axis=1) == 0).any():
                raise ValueError(
                    f"OneHotEncoder: unknown value in column {j}.")
            parts.append(indicator)
        return np.hstack(parts)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        if not self.fitted:
            raise RuntimeError("OneHotEncoder: call fit() first.")
        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in)]
        return [f"{feat}_{cat}" for feat, cats in zip(input_features, self.categories) for cat in cats]
