import unittest
import numpy as np

from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


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


if __name__ == "__main__":
    unittest.main()