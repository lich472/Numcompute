import unittest
import numpy as np

from numcompute.stat import mean, median, std, min, max, histogram, quantiles


class TestMean(unittest.TestCase):
    def test_axis_1(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = mean(arr, axis=1)
        np.testing.assert_array_equal(result, np.array([2.0, 5.0]))

    def test_basic(self):
        arr = np.array([1, 2, 3, 4])
        self.assertEqual(mean(arr), 2.5)

    def test_axis(self):
        arr = np.array([[1, 2], [3, 4]])
        result = mean(arr, axis=0)
        np.testing.assert_array_equal(result, np.array([2.0, 3.0]))

    def test_empty(self):
        with self.assertRaises(ValueError):
            mean([])


class TestMedian(unittest.TestCase):
    def test_axis_1(self):
        arr = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
        result = median(arr, axis=1)
        np.testing.assert_array_equal(result, np.array([2.5, 5.5, 8.5]))

    def test_odd(self):
        arr = np.array([1, 3, 2])
        self.assertEqual(median(arr), 2)

    def test_even(self):
        arr = np.array([1, 2, 3, 4])
        self.assertEqual(median(arr), 2.5)

    def test_axis(self):
        arr = np.array([[1, 2], [3, 4]])
        result = median(arr, axis=0)
        np.testing.assert_array_equal(result, np.array([2.0, 3.0]))


class TestStd(unittest.TestCase):
    def test_basic(self):
        arr = np.array([1, 2, 3, 4])
        result = std(arr)
        self.assertAlmostEqual(result, np.std(arr))

    def test_ddof(self):
        arr = np.array([1, 2, 3, 4])
        result = std(arr, ddof=1)
        self.assertAlmostEqual(result, np.std(arr, ddof=1))

    def test_invalid_ddof(self):
        with self.assertRaises(ValueError):
            std([1, 2], ddof=2)


class TestMinMax(unittest.TestCase):
    def test_min(self):
        arr = np.array([3, 1, 2])
        self.assertEqual(min(arr), 1)

    def test_max(self):
        arr = np.array([3, 1, 2])
        self.assertEqual(max(arr), 3)


class TestHistogram(unittest.TestCase):
    def test_histogram_assignment_sample(self):
        arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
        edges, counts = histogram(arr, bins=5)
        np.testing.assert_allclose(edges, np.array([1., 3.2, 5.4, 7.6, 9.8, 12.]))
        np.testing.assert_array_equal(counts, np.array([3, 2, 2, 2, 3]))
        
    def test_basic(self):
        arr = np.array([1, 2, 3, 4])
        edges, counts = histogram(arr, bins=2)
        self.assertEqual(len(edges), 3)
        self.assertEqual(len(counts), 2)

    def test_all_same(self):
        arr = np.array([5, 5, 5])
        edges, counts = histogram(arr, bins=3)
        self.assertEqual(counts[0], 3)

    def test_invalid_bins(self):
        with self.assertRaises(ValueError):
            histogram([1, 2, 3], bins=0)


class TestQuantiles(unittest.TestCase):
    def test_quantiles_assignment_sample(self):
        data = np.array([10, 20, np.nan, 40, 50])
        self.assertEqual(quantiles(data, 0), 10.0)
        self.assertEqual(quantiles(data, 0.5), 30.0)
        self.assertEqual(quantiles(data, 0.75), 42.5)
        self.assertEqual(quantiles(data, 1), 50.0)
        
    def test_basic(self):
        arr = np.array([10, 20, 30, 40, 50])
        result = quantiles(arr, 0.5)
        self.assertEqual(result, 30)

    def test_multiple_q(self):
        arr = np.array([10, 20, 30, 40, 50])
        result = quantiles(arr, [0.25, 0.75])
        np.testing.assert_array_equal(result, np.array([20, 40]))

    def test_with_nan(self):
        arr = np.array([10, 20, np.nan, 40])
        result = quantiles(arr, 0.5)
        self.assertEqual(result, 20)

    def test_invalid_q(self):
        with self.assertRaises(ValueError):
            quantiles([1, 2, 3], -0.1)


if __name__ == "__main__":
    unittest.main()