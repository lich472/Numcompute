import unittest
import numpy as np

from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, Imputer


class TestStandardScaler(unittest.TestCase):
    def test_basic(self):
        X = np.array([[1, 2], [3, 4]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.assertEqual(X_scaled.shape, X.shape)

    def test_inverse(self):
        X = np.array([[1, 2], [3, 4]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_back = scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X, X_back)

    def test_not_fitted(self):
        scaler = StandardScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform([1, 2])
    
    def test_partial_fit_single_chunk(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler_fit = StandardScaler().fit(X)
        scaler_partial = StandardScaler()
        scaler_partial.partial_fit(X)
        np.testing.assert_allclose(scaler_fit.mean, scaler_partial.mean)
        np.testing.assert_allclose(scaler_fit.var, scaler_partial.var)
    
    def test_partial_fit_multiple_chunks(self):
        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([[5, 6], [7, 8]])
        X_full = np.vstack([X1, X2])
        scaler = StandardScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)
        # mean should approximate full data mean
        np.testing.assert_allclose(scaler.mean, np.nanmean(X_full, axis=0), rtol=1e-5)

    def test_partial_fit_with_nans(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        scaler = StandardScaler()
        scaler.partial_fit(X)  # should work with crash
        self.assertTrue(scaler.fitted)


class TestMinMaxScaler(unittest.TestCase):
    def test_basic(self):
        X = np.array([[1, 2], [3, 4]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        self.assertTrue((X_scaled >= 0).all())
        self.assertTrue((X_scaled <= 1).all())

    def test_invalid_range(self):
        with self.assertRaises(ValueError):
            MinMaxScaler(feature_range=(1, 1))

    def test_partial_fit_updates_min_max(self):
        X1 = np.array([[2, 3], [4, 5]])
        X2 = np.array([[0, 1], [6, 8]])  
        scaler = MinMaxScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)
        np.testing.assert_allclose(scaler.data_min, np.array([0, 1]))
        np.testing.assert_allclose(scaler.data_max, np.array([6, 8]))

    def test_partial_fit_consistent_with_fit(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler_fit = MinMaxScaler().fit(X)
        scaler_partial = MinMaxScaler()
        scaler_partial.partial_fit(X)
        np.testing.assert_allclose(scaler_fit.data_min, scaler_partial.data_min)
        np.testing.assert_allclose(scaler_fit.data_max, scaler_partial.data_max)


class TestOneHotEncoder(unittest.TestCase):
    def test_basic(self):
        X = np.array(["A", "B", "A"])
        enc = OneHotEncoder()
        out = enc.fit_transform(X)
        self.assertEqual(out.shape[0], 3)

    def test_unknown_ignore(self):
        X = np.array(["A", "B"])
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(X)
        out = enc.transform(["A", "C"])
        self.assertEqual(out.shape[0], 2)

    def test_unknown_error(self):
        X = np.array(["A", "B"])
        enc = OneHotEncoder(handle_unknown="error")
        enc.fit(X)
        with self.assertRaises(ValueError):
            enc.transform(["C"])

    def test_partial_fit_new_categories(self):
        X1 = np.array([["A"], ["B"]])
        X2 = np.array([["C"]])  # new category
        enc = OneHotEncoder()
        enc.partial_fit(X1)
        enc.partial_fit(X2)
        self.assertIn("C", enc.categories[0])

    def test_partial_fit_consistent_features(self):
        X1 = np.array([["A", "B"]])
        X2 = np.array([["L"]])  # wrong number of features
        enc = OneHotEncoder()
        enc.partial_fit(X1)
        with self.assertRaises(ValueError):
            enc.partial_fit(X2)
    
class TestImputer(unittest.TestCase):
    def test_imputer_mean(self):
        X = np.array([[1, np.nan], [3, 4], [5, 6]])
        imputer = Imputer(strategy='mean')
        out = imputer.fit_transform(X)
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_allclose(out[0, 1], np.nanmean(X[:, 1]))

    def test_imputer_median(self):
        X = np.array([[1, np.nan], [3, 4], [5, 6]])
        imputer = Imputer(strategy='median')
        out = imputer.fit_transform(X)
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_allclose(out[0, 1], np.nanmedian(X[:, 1]))

    def test_imputer_constant(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        imputer = Imputer(strategy='constant', fill_value=0)
        out = imputer.fit_transform(X)
        self.assertFalse(np.isnan(out).any())
        self.assertEqual(out[0, 1], 0.0)

    def test_imputer_partial_fit(self):
        X1 = np.array([[1, np.nan], [3, 4]])
        X2 = np.array([[5, 6], [7, np.nan]])
        imputer = Imputer(strategy='mean')
        imputer.partial_fit(X1)
        imputer.partial_fit(X2)
        self.assertTrue(imputer.fitted)
        self.assertIsNotNone(imputer.statistics)

    def test_imputer_not_fitted(self):
        imputer = Imputer(strategy='mean')
        with self.assertRaises(RuntimeError):
            imputer.transform(np.array([[1, 2]]))

    def test_imputer_wrong_features(self):
        X = np.array([[1, 2], [3, 4]])
        imputer = Imputer(strategy='mean')
        imputer.fit(X)
        with self.assertRaises(ValueError):
            imputer.transform(np.array([[1, 2, 3]]))  # 3 features instead of 2

if __name__ == "__main__":
    unittest.main()