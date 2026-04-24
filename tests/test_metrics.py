import unittest
import numpy as np
from numcompute.metrics import accuracy, confusion_matrix, mse, precision, recall, f1

from numcompute.metrics import accuracy


class TestAccuracy(unittest.TestCase):
    def test_accuracy_basic(self):
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])
        result = accuracy(y_true, y_pred)
        self.assertAlmostEqual(result, 0.75)

    def test_accuracy_all_correct(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        result = accuracy(y_true, y_pred)
        self.assertAlmostEqual(result, 1.0)

    def test_accuracy_all_wrong(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        result = accuracy(y_true, y_pred)
        self.assertAlmostEqual(result, 0.0)

    def test_accuracy_shape_mismatch(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0])
        with self.assertRaises(ValueError):
            accuracy(y_true, y_pred)

    def test_accuracy_empty(self):
        y_true = np.array([])
        y_pred = np.array([])
        with self.assertRaises(ValueError):
            accuracy(y_true, y_pred)
class TestConfusionMatrix(unittest.TestCase):
    def test_confusion_matrix_binary(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        result = confusion_matrix(y_true, y_pred)
        expected = np.array([[2, 0],
                             [1, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_confusion_matrix_with_labels(self):
        y_true = np.array([2, 1, 2, 1])
        y_pred = np.array([2, 2, 1, 1])
        result = confusion_matrix(y_true, y_pred, labels=[1, 2])
        expected = np.array([[1, 1],
                             [1, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_confusion_matrix_empty(self):
        y_true = np.array([])
        y_pred = np.array([])
        with self.assertRaises(ValueError):
            confusion_matrix(y_true, y_pred)


class TestMSE(unittest.TestCase):
    def test_mse_basic(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        result = mse(y_true, y_pred)
        self.assertAlmostEqual(result, 1 / 3)

    def test_mse_zero(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        result = mse(y_true, y_pred)
        self.assertAlmostEqual(result, 0.0)

    def test_mse_shape_mismatch(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0])
        with self.assertRaises(ValueError):
            mse(y_true, y_pred)

    def test_mse_empty(self):
        y_true = np.array([])
        y_pred = np.array([])
        with self.assertRaises(ValueError):
            mse(y_true, y_pred)
            
class TestClassificationMetrics(unittest.TestCase):
    def test_precision_basic(self):
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        result = precision(y_true, y_pred)
        self.assertAlmostEqual(result, 2/3)

    def test_recall_basic(self):
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        result = recall(y_true, y_pred)
        self.assertAlmostEqual(result, 2/3)

    def test_f1_basic(self):
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        result = f1(y_true, y_pred)
        self.assertAlmostEqual(result, 2/3)

    def test_precision_zero_division(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        result = precision(y_true, y_pred)
        self.assertEqual(result, 0.0)

    def test_recall_zero_division(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        result = recall(y_true, y_pred)
        self.assertEqual(result, 0.0)

    def test_f1_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        result = f1(y_true, y_pred)
        self.assertEqual(result, 0.0)



if __name__ == "__main__":
    unittest.main()



