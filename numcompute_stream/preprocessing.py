import numpy as np

__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder", "Imputer"]


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
        self.total = 0
        self.running_mean = None
        self.running_M2 = None

    def partial_fit(self, X_chunk):

        """
        Using Welford technique for partial_fit()
        1. validate and reshape
        2. initialise running state unless first calls
        3. else update running mean/M2 using Welford
        4. recompute self.mean, self.var, self.scale
        5. update self.n_features
        6. set self.fitted = True
        7. return self
        """
        X = np.asarray(X_chunk, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("StandardScaler expects 1D or 2D input.")

        if X.shape[0] == 0:
            raise ValueError("StandardScaler: no data to fit on.")
        
        chunk_n = X.shape[0]
        chunk_mean = np.nanmean(X, axis=0)
        chunk_var = np.nanvar(X, axis=0)
        
        if self.total == 0:
            self.total = X_chunk.size
            self.running_mean = self.mean
            self.running_M2 = self.chunk_var * self.chunk_n
        else :
            # Welford update
            new_total += self.total + chunk_n
            delta = chunk_mean - self.running_mean
            self.running_mean += delta * chunk_n / new_total
            delta2 = chunk_mean - self.running_mean
            self.running_M2 += chunk_var * chunk_n + delta * delta2 * self.total * chunk_n / new_total
        
        self.total += chunk_n
        self.mean = self.running_mean
        self.var = self.running_M2 / self.total
        self.scale = np.where(self.var == 0, 1.0, np.sqrt(self.var))
        self.n_features = X.shape[1]
        self.fitted = True
        return self
    
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

        self.total = 0
        self.feature_range = feature_range
        self.copy = copy
        self.fitted = False

    def partial_fit(self, X_chunk):
        """
        TODO
            # 1. validate and reshape
            # 2. initialise data_min, data_max from chunk unless first calls
            # 3. else: -> update data_min, data_max using np.minimum() and np.maximum()
            # 4. recompute scale and min
            # 5. update total, n_features, fitted
            # 6. return self
        """
        X = np.asarray(X_chunk, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects 1D or 2D input.")

        if X.shape[0] == 0:
            raise ValueError("MinMaxScaler: no data to fit on.")
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)

        if self.total == 0:
            self.data_min = chunk_min
            self.data_max = chunk_max
        else:
            self.data_min = np.minimum(self.data_running_min, chunk_min)
            self.data_max = np.minimum(self.data_running_max, chunk_max)

        self.total += X.shape[0]

        lower, upper = self.feature_range
        self.data_range = self.data_max - self.data_min
        self.scale = (upper - lower) / np.where(self.data_range == 0, 1.0, self.data_range)
        self.min = lower - self.data_min * self.scale
        self.n_features = X.shape[1]
        self.fitted = True
        return self

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

        self.total = 0
        self.handle_unknown = handle_unknown
        self.sparse = sparse
        self.fitted = False

    def partial_fit(self, X_chunk):
        """
        TODO
            1. validate and reshape
            2. check n_features consistent if not first call
            3. initialise categories from chunk if first call
            4. else: expand categories using np.union1d() per column 
            5. update total, n_features_in, fitted
            6. return self
        """
        X = np.asarray(X_chunk)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("OneHotEncoder expects 1D or 2D input.")

        if X.shape[0] == 0:
            raise ValueError("OneHotEncoder: no data to fit on.")

        if self.total == 0:
            self.n_features_in = X.shape[1]
            self.categories = [np.unique(X[:, j]) for j in range(self.n_features_in)]
        else:
            if self.n_features_in != X.shape[1]: # raise the error if n_features_in not equal to X.shape[1]
                raise ValueError(f"OneHotEncoder: expected {self.n_features_in} features but {X.shape[1]}")
            self.categories = [np.union1d(self.categories[j], np.unique(X[:, j])) for j in range(self.n_features_in)]
            # np.union1d(existing_categories, new_categories) -> return the union 1d from 2 arrays. ['a','b'],['b','c'] -> ['a', 'b', 'c']

        self.total += X.shape[0]
        self.fitted = True
        return self


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
    
class Imputer:
    def __init__(self, strategy = 'mean', value_for_constant = None):
        if strategy.lower() not in ('mean', 'median', 'constant'):
            raise ValueError("strategy must be 'mean', 'median', or 'constant' ")
        self.strategy = strategy
        self.value_for_constant = value_for_constant
        self.statistic = None 
        self.total = 0
        self.fitted = False
    
    def fit(self, X):
        """
            1. validate and reshape
            2. compute statistics based on strategy:
               if 'mean'  -> np.nanmean(X, axis=0)
               if 'median'  -> np.nanmedian(X, axis=0)
               if 'constant' -> np.full(X.shape[1], self.value_for_constant)
            3. set total, fitted=True
            4. return self
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("Imputer expects 1D or 2D input.")

        if X.shape[0] == 0:
            raise ValueError("Imputer: no data to fit on.")

        if (self.strategy).lower() == 'mean':
            self.statistic = np.nanmean(X)
        elif (self.strategy).lower() == 'median':
            self.statistic = np.nanmedian(X)
        elif self.strategy(self.strategy).lower() == 'constant':
            if self.fill_value is None:
                raise ValueError("value_for_constant must be set when strategy='constant'.")
            self.statistic = np.full(X.shape[1], self.fill_value)

        self.total = X.shape[0]
        self.fitted = True
        return self
    
    def partial_fit(self, X_chunk):
        X = np.asarray(X_chunk, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("Imputer expects 1D or 2D input.")

        if X.shape[0] == 0:
            raise ValueError("Imputer: no data to fit on.")

        chunk_n = X.shape[0]

        if self.strategy == 'mean':
            chunk_mean = np.nanmean(X, axis=0)
            if self.total == 0:
                self.running_mean = chunk_mean
            else:
                new_total = self.total + chunk_n
                delta = chunk_mean - self.running_mean
                self.running_mean += delta * chunk_n / new_total
            self.statistics = self.running_mean

        elif self.strategy == 'median':
            self.statistics = np.nanmedian(X, axis=0)

        elif self.strategy == 'constant':
            if self.fill_value is None:
                raise ValueError("fill_value must be set when strategy='constant'.")
            self.statistics = np.full(X.shape[1], self.fill_value)

        self.total += chunk_n
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Imputer: call fit() or partial_fit() before transform().")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("Imputer expects 1D or 2D input.")

        if X.shape[1] != self.n_features:
            raise ValueError(f"Imputer: expected {self.n_features} features, got {X.shape[1]}.")
        
        updated_arr = np.where(np.isnan(X), self.statistics, X)

        return updated_arr
    
    def fit_transform(self, X):
        """
        Fit the scaler and transform X.
        """
        return self.fit(X).transform(X)