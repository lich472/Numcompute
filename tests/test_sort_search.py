import unittest
import numpy as np

from numcompute.sort_search import (
    stable_sort,
    multi_key_sort,
    topk,
    quickselect,
    binary_search,
)


class TestStableSort(unittest.TestCase):
    def test_stable_sort_strings(self):
        arr = np.array(["Bob", "Alice", "Henry", "Alice", "Luca", "Braydon"])
        expected = np.array(["Alice", "Alice", "Bob", "Braydon", "Henry", "Luca"])
        np.testing.assert_array_equal(stable_sort(arr), expected)

    def test_stable_sort_basic(self):
        arr = np.array([3, 1, 2])
        result = stable_sort(arr)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_stable_sort_2d_axis(self):
        arr = np.array([[3, 1], [4, 2]])
        result = stable_sort(arr, axis=1)
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(result, expected)


class TestMultiKeySort(unittest.TestCase):
    def test_multi_key_sort_ascending(self):
        arr = np.array([
            [2, 3],
            [1, 5],
            [1, 2],
            [2, 1],
        ])
        result = multi_key_sort(arr, keys=[0, 1])
        expected = np.array([
            [1, 2],
            [1, 5],
            [2, 1],
            [2, 3],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_multi_key_sort_descending(self):
        arr = np.array([
            [2, 3],
            [1, 5],
            [1, 2],
            [2, 1],
        ])
        result = multi_key_sort(arr, keys=[0, 1], ascending=False)
        expected = np.array([
            [2, 3],
            [2, 1],
            [1, 5],
            [1, 2],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_multi_key_sort_invalid_dim(self):
        with self.assertRaises(ValueError):
            multi_key_sort(np.array([1, 2, 3]), keys=[0])

    def test_multi_key_sort_empty_keys(self):
        with self.assertRaises(ValueError):
            multi_key_sort(np.array([[1, 2], [3, 4]]), keys=[])

    def test_multi_key_sort_invalid_key(self):
        with self.assertRaises(ValueError):
            multi_key_sort(np.array([[1, 2], [3, 4]]), keys=[2])


class TestTopK(unittest.TestCase):
    def test_topk_largest_indices(self):
        arr = np.array([10, 50, 20, 40])
        result = topk(arr, 2, largest=True, return_indices=True)
        expected = np.array([1, 3])
        np.testing.assert_array_equal(result, expected)

    def test_topk_largest_values(self):
        arr = np.array([10, 50, 20, 40])
        result = topk(arr, 2, largest=True, return_indices=False)
        expected = np.array([50, 40])
        np.testing.assert_array_equal(result, expected)

    def test_topk_smallest_indices(self):
        arr = np.array([10, 50, 20, 40])
        result = topk(arr, 2, largest=False, return_indices=True)
        expected = np.array([0, 2])
        np.testing.assert_array_equal(result, expected)

    def test_topk_smallest_values(self):
        arr = np.array([10, 50, 20, 40])
        result = topk(arr, 2, largest=False, return_indices=False)
        expected = np.array([10, 20])
        np.testing.assert_array_equal(result, expected)

    def test_topk_zero(self):
        arr = np.array([1, 2, 3])
        result = topk(arr, 0)
        expected = np.array([], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_topk_invalid_k(self):
        with self.assertRaises(ValueError):
            topk(np.array([1, 2, 3]), 5)

    def test_topk_non_1d(self):
        with self.assertRaises(ValueError):
            topk(np.array([[1, 2], [3, 4]]), 1)


class TestQuickSelect(unittest.TestCase):
    def test_quickselect_basic(self):
        arr = np.array([7, 2, 5, 1, 9])
        result = quickselect(arr, 2)
        self.assertEqual(result, 5)

    def test_quickselect_min(self):
        arr = np.array([7, 2, 5, 1, 9])
        result = quickselect(arr, 0)
        self.assertEqual(result, 1)

    def test_quickselect_max(self):
        arr = np.array([7, 2, 5, 1, 9])
        result = quickselect(arr, 4)
        self.assertEqual(result, 9)

    def test_quickselect_with_duplicates(self):
        arr = np.array([3, 1, 2, 2, 5])
        result = quickselect(arr, 2)
        self.assertEqual(result, 2)

    def test_quickselect_does_not_modify_original(self):
        arr = np.array([3, 1, 2])
        original = arr.copy()
        quickselect(arr, 1)
        np.testing.assert_array_equal(arr, original)

    def test_quickselect_invalid_k(self):
        with self.assertRaises(ValueError):
            quickselect(np.array([1, 2, 3]), 3)

    def test_quickselect_empty(self):
        with self.assertRaises(ValueError):
            quickselect(np.array([]), 0)

    def test_quickselect_non_1d(self):
        with self.assertRaises(ValueError):
            quickselect(np.array([[1, 2], [3, 4]]), 1)


class TestBinarySearch(unittest.TestCase):
    def test_binary_search_assignment_samples(self):
        arr = np.array([1, 3, 5, 7, 9, 11])
        self.assertEqual(binary_search(arr, -1), (0, False))
        self.assertEqual(binary_search(arr, 8), (4, False))
        self.assertEqual(binary_search(arr, 9), (4, True))
        self.assertEqual(binary_search(arr, 12), (6, False))

    def test_binary_search_exists(self):
        arr = np.array([1, 3, 5, 7])
        idx, exists = binary_search(arr, 5)
        self.assertEqual(idx, 2)
        self.assertTrue(exists)

    def test_binary_search_not_exists_middle(self):
        arr = np.array([1, 3, 5, 7])
        idx, exists = binary_search(arr, 4)
        self.assertEqual(idx, 2)
        self.assertFalse(exists)

    def test_binary_search_insert_start(self):
        arr = np.array([10, 20, 30])
        idx, exists = binary_search(arr, 5)
        self.assertEqual(idx, 0)
        self.assertFalse(exists)

    def test_binary_search_insert_end(self):
        arr = np.array([10, 20, 30])
        idx, exists = binary_search(arr, 40)
        self.assertEqual(idx, 3)
        self.assertFalse(exists)

    def test_binary_search_duplicates(self):
        arr = np.array([1, 2, 2, 2, 3])
        idx, exists = binary_search(arr, 2)
        self.assertEqual(idx, 1)
        self.assertTrue(exists)

    def test_binary_search_empty(self):
        arr = np.array([])
        idx, exists = binary_search(arr, 5)
        self.assertEqual(idx, 0)
        self.assertFalse(exists)

    def test_binary_search_non_1d(self):
        with self.assertRaises(ValueError):
            binary_search(np.array([[1, 2], [3, 4]]), 2)


if __name__ == "__main__":
    unittest.main()