import unittest
import numpy as np

from numcompute.utils import (
    sigmoid,
    relu,
    tanh,
    softmax,
    logsumexp,
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
    pairwise_distances,
    topk_indices,
    iter_batches,
)


class TestActivations(unittest.TestCase):
    def test_sigmoid_basic(self):
        result = sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, np.array([0.5]))

    def test_sigmoid_stability(self):
        result = sigmoid(np.array([-1000.0, 1000.0]))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_relu(self):
        result = relu(np.array([-2, 0, 3]))
        expected = np.array([0, 0, 3])
        np.testing.assert_array_equal(result, expected)

    def test_tanh(self):
        result = tanh(np.array([0.0]))
        np.testing.assert_allclose(result, np.array([0.0]))


class TestSoftmaxLogsumexp(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_softmax_axis(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = softmax(x, axis=1)
        np.testing.assert_allclose(np.sum(result, axis=1), np.array([1.0, 1.0]))

    def test_logsumexp_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = logsumexp(x)
        expected = np.log(np.sum(np.exp(x)))
        self.assertAlmostEqual(result, expected)

    def test_logsumexp_stability(self):
        x = np.array([1000.0, 1000.0])
        result = logsumexp(x)
        self.assertFalse(np.isinf(result))


class TestDistances(unittest.TestCase):
    def test_euclidean_distance(self):
        result = euclidean_distance(np.array([0, 0]), np.array([3, 4]))
        self.assertEqual(result, 5.0)

    def test_manhattan_distance(self):
        result = manhattan_distance(np.array([1, 2]), np.array([4, 6]))
        self.assertEqual(result, 7.0)

    def test_cosine_distance_same_direction(self):
        result = cosine_distance(np.array([1, 0]), np.array([1, 0]))
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_pairwise_euclidean(self):
        X = np.array([[0, 0], [3, 4]])
        result = pairwise_distances(X, metric="euclidean")
        expected = np.array([[0.0, 5.0], [5.0, 0.0]])
        np.testing.assert_allclose(result, expected)

    def test_pairwise_manhattan(self):
        X = np.array([[0, 0], [1, 1]])
        Y = np.array([[2, 2]])
        result = pairwise_distances(X, Y, metric="manhattan")
        expected = np.array([[4], [2]])
        np.testing.assert_array_equal(result, expected)

    def test_pairwise_invalid_metric(self):
        with self.assertRaises(ValueError):
            pairwise_distances(np.array([[1, 2]]), metric="wrong")


class TestTopKIndices(unittest.TestCase):
    def test_topk_largest(self):
        arr = np.array([10, 50, 20, 40])
        result = topk_indices(arr, 2, largest=True)
        expected = np.array([1, 3])
        np.testing.assert_array_equal(result, expected)

    def test_topk_smallest(self):
        arr = np.array([10, 50, 20, 40])
        result = topk_indices(arr, 2, largest=False)
        expected = np.array([0, 2])
        np.testing.assert_array_equal(result, expected)

    def test_topk_invalid_k(self):
        with self.assertRaises(ValueError):
            topk_indices(np.array([1, 2, 3]), 0)

    def test_topk_clamps_k(self):
        arr = np.array([3, 1])
        result = topk_indices(arr, 5)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(result, expected)


class TestIterBatches(unittest.TestCase):
    def test_iter_batches_basic(self):
        batches = list(iter_batches(5, 2))
        expected = [
            np.array([0, 1]),
            np.array([2, 3]),
            np.array([4]),
        ]

        self.assertEqual(len(batches), 3)
        for result, exp in zip(batches, expected):
            np.testing.assert_array_equal(result, exp)

    def test_iter_batches_shuffle_reproducible(self):
        b1 = list(iter_batches(5, 2, shuffle=True, seed=42))
        b2 = list(iter_batches(5, 2, shuffle=True, seed=42))

        for x, y in zip(b1, b2):
            np.testing.assert_array_equal(x, y)

    def test_iter_batches_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            list(iter_batches(5, 0))

    def test_iter_batches_negative_samples(self):
        with self.assertRaises(ValueError):
            list(iter_batches(-1, 2))


if __name__ == "__main__":
    unittest.main()