import unittest
import numpy as np

from numcompute.rank import rank, percentile


class TestRank(unittest.TestCase):
    def test_rank_assignment_sample(self):
        data = np.array([0.7, 0.9, 0.8, 0.8, 0.8])
        np.testing.assert_allclose(rank(data, "average"), np.array([0., 4., 2., 2., 2.]))
        np.testing.assert_array_equal(rank(data, "ordinal"), np.array([0, 4, 1, 2, 3]))
        np.testing.assert_array_equal(rank(data, "dense"), np.array([0, 2, 1, 1, 1]))

    def test_dense_rank_with_ties(self):
        data = np.array([10, 30, 20, 20])
        result = rank(data, method="dense")
        expected = np.array([0, 2, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_ordinal_rank_stable(self):
        data = np.array([10, 30, 20, 20])
        result = rank(data, method="ordinal")
        expected = np.array([0, 3, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_average_rank_with_ties(self):
        data = np.array([10, 30, 20, 20])
        result = rank(data, method="average")
        expected = np.array([0.0, 3.0, 1.5, 1.5])
        np.testing.assert_allclose(result, expected)

    def test_rank_default_average(self):
        data = np.array([5, 5, 10])
        result = rank(data)
        expected = np.array([0.5, 0.5, 2.0])
        np.testing.assert_allclose(result, expected)

    def test_rank_empty_array(self):
        with self.assertRaises(ValueError):
            rank(np.array([]))

    def test_rank_invalid_method(self):
        with self.assertRaises(ValueError):
            rank(np.array([1, 2, 3]), method="wrong")

    def test_rank_non_1d_array(self):
        with self.assertRaises(ValueError):
            rank(np.array([[1, 2], [3, 4]]))


class TestPercentile(unittest.TestCase):
    def test_percentile_linear(self):
        data = np.array([10, 20, 30, 40, 50])
        result = percentile(data, 30, interpolation="linear")
        self.assertAlmostEqual(result, 22.0)

    def test_percentile_lower(self):
        data = np.array([10, 20, 30, 40, 50])
        result = percentile(data, 30, interpolation="lower")
        self.assertAlmostEqual(result, 20.0)

    def test_percentile_higher(self):
        data = np.array([10, 20, 30, 40, 50])
        result = percentile(data, 30, interpolation="higher")
        self.assertAlmostEqual(result, 30.0)

    def test_percentile_midpoint(self):
        data = np.array([10, 20, 30, 40, 50])
        result = percentile(data, 30, interpolation="midpoint")
        self.assertAlmostEqual(result, 25.0)

    def test_percentile_multiple_q(self):
        data = np.array([10, 20, 30, 40, 50])
        result = percentile(data, np.array([0, 50, 100]))
        expected = np.array([10.0, 30.0, 50.0])
        np.testing.assert_allclose(result, expected)

    def test_percentile_unsorted_data(self):
        data = np.array([50, 10, 40, 20, 30])
        result = percentile(data, 50)
        self.assertAlmostEqual(result, 30.0)

    def test_percentile_invalid_q(self):
        with self.assertRaises(ValueError):
            percentile(np.array([1, 2, 3]), 120)

    def test_percentile_invalid_interpolation(self):
        with self.assertRaises(ValueError):
            percentile(np.array([1, 2, 3]), 50, interpolation="wrong")

    def test_percentile_empty_array(self):
        with self.assertRaises(ValueError):
            percentile(np.array([]), 50)

    def test_percentile_non_numeric(self):
        with self.assertRaises(ValueError):
            percentile(np.array(["a", "b"]), 50)

    def test_percentile_non_1d_array(self):
        with self.assertRaises(ValueError):
            percentile(np.array([[1, 2], [3, 4]]), 50)


if __name__ == "__main__":
    unittest.main()