import unittest
import numpy as np
from numcompute_stream.ensemble import RandomForestClassifier


class TestRandomForestClassifier(unittest.TestCase):

    def test_fit_basic(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        rf = RandomForestClassifier(n_estimators=5)
        rf.fit(X, y)
        self.assertTrue(rf.fitted)

    def test_predict_before_fit(self):
        rf = RandomForestClassifier()
        with self.assertRaises(RuntimeError):
            rf.predict(np.array([[1, 2]]))

    def test_n_estimators(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        rf = RandomForestClassifier(n_estimators=7)
        rf.fit(X, y)
        self.assertEqual(len(rf.trees), 7) # should have 7 trees

    def test_bootstrap_shape(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        rf = RandomForestClassifier()
        X_boot, y_boot = rf._bootstrap_sample(X, y)
        self.assertEqual(X_boot.shape, X.shape)
        self.assertEqual(y_boot.shape, y.shape)

    def test_partial_fit_two_chunks(self):
        X1 = np.array([[1, 2], [3, 4]])
        y1 = np.array([0, 0])
        X2 = np.array([[5, 6], [7, 8]])
        y2 = np.array([1, 1])
        rf = RandomForestClassifier(n_estimators=5)
        rf.partial_fit(X1, y1)
        rf.partial_fit(X2, y2)
        self.assertEqual(len(rf.X_all_chunks), 2)

    def test_partial_fit_then_predict(self):
        X1 = np.array([[1, 0], [2, 0]])
        y1 = np.array([0, 0])
        X2 = np.array([[8, 0], [9, 0]])
        y2 = np.array([1, 1])
        rf = RandomForestClassifier(n_estimators=10)
        rf.partial_fit(X1, y1)
        rf.partial_fit(X2, y2)
        preds = rf.predict(np.array([[1, 0], [9, 0]]))
        self.assertEqual(preds.shape, (2,))

    def test_single_class(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])
        rf = RandomForestClassifier(n_estimators=5)
        rf.fit(X, y)
        preds = rf.predict(X)
        np.testing.assert_array_equal(preds, y)


if __name__ == "__main__":
    unittest.main()