import unittest
import numpy as np

from numcompute_stream.stat import chunk_mean, chunk_variance, chunk_quantiles, chunk_histogram, max, histogram, quantiles


class TestMean(unittest.TestCase):

    def test_basic(self):
        X_chunk = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        result = chunk_mean(X_chunk)
        np.testing.assert_array_equal(result, np.array([4.0, 5.0, 6.0]))

    def test_NaN(self):
        X_chunk = np.array([
            [1.0, np.nan, 3.0],
            [4.0, 5.0,   6.0],
            [7.0, 8.0,   np.nan]
        ])
        result = chunk_mean(X_chunk)
        np.testing.assert_array_equal(result, np.array([4.0, 6.5, 4.5]))

    def test_full_collumn_NaN(self):
        X_chunk = np.array([
            [1.0, np.nan],
            [4.0, np.nan],
        ])
        result = chunk_mean(X_chunk)
        np.testing.assert_array_equal(result, np.array([2.5, nan]))

    def test_empty(self):
        with self.assertRaises(ValueError):
            chunk_mean([])


class TestVariance(unittest.TestCase):

    def test_basic(self):
        X_chunk = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        result = chunk_variance(X_chunk)
        np.testing.assert_array_equal(result, np.array([6.0, 6.0, 6.0]))

    def test_NaN(self):
        X_chunk = np.array([
            [1.0, np.nan, 3.0],
            [4.0, 5.0,   6.0],
            [7.0, 8.0,   np.nan]
        ])
        result = chunk_variance(X_chunk)
        np.testing.assert_array_equal(result, np.array([6.0, 2.25, 2.25]))

    def test_full_collumn_NaN(self):
        X_chunk = np.array([
            [1.0, np.nan],
            [4.0, np.nan],
        ])
        result = chunk_variance(X_chunk)
        np.testing.assert_array_equal(result, np.array([2.5, nan]))

    def test_empty(self):
        with self.assertRaises(ValueError):
            chunk_variance([])


class TestQuantiles(unittest.TestCase):
        
    def test_basic(self):
        X_chunk = np.array([[1.0, 2.0], [3.0, 4.0]])
        q = 0.5
        result = chunk_quantiles(X_chunk, q)
        np.testing.assert_array_almost_equal(result, np.array([[2.0, 3.0]]))

    def test_multiple_q(self):
        X_chunk = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ])
        q = [0.25, 0.75]
        result = chunk_quantiles(X_chunk, q)
        np.testing.assert_array_almost_equal(result, np.array([[2.5, 3.5], [5.5, 6.5]]))

    def test_with_nan(self):
        X_chunk = np.array([
            [1.0, np.nan],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        q = [0.5]
        result = chunk_quantiles(X_chunk, q)
        np.testing.assert_array_almost_equal(result, np.array([[3.0, 5.0]])) # ← NaN ignored in column 1

    def test_invalid_q(self):
        with self.assertRaises(ValueError):
            chunk_quantiles([1, 2, 3], -0.1)


class TestHistogram(unittest.TestCase):
    def test_outside_bin_range_left(self):
        X_chunk = np.array([-1.0, 3.0, 5.0])
        edges = np.array([2.0, 4.0, 6.0, 8.0])
        np.testing.assert_allclose(chunk_histogram(X_chunk,edges), np.array([2, 1, 0])) # -1.0 gets clipped into first bin

    def test_outside_bin_range_right(self):
        X_chunk = np.array([3.0, 5.0, 99.0])
        edges = np.array([2.0, 4.0, 6.0, 8.0])
        np.testing.assert_allclose(chunk_histogram(X_chunk,edges), np.array([1, 1, 1])) # 99.0 gets clipped into last bin
        
    def test_basic(self):
        X_chunk = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        edges = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])  # fixed bin edges set upfront
        np.testing.assert_array_equal(chunk_histogram(X_chunk, edges), np.array([1, 1, 1, 1, 1]))


    def test_single_value(self):
        X_chunk = np.array([3.0])
        edges = np.array([0.0, 2.0, 4.0, 6.0])
        np.testing.assert_allclose(chunk_histogram(X_chunk,edges), np.array([0, 1, 0])) 



if __name__ == "__main__":
    unittest.main()