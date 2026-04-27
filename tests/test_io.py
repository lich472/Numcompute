import unittest
import numpy as np
import os

from numcompute.io import read_csv


class TestIO(unittest.TestCase):

    def setUp(self):
        self.file = "test.csv"
        with open(self.file, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")

    def tearDown(self):
        os.remove(self.file)

    def test_basic_read(self):
        data = read_csv(self.file)
        self.assertIn("a", data)
        np.testing.assert_array_equal(data["a"], np.array([1.0, 4.0]))

    def test_missing_values(self):
        with open(self.file, "w") as f:
            f.write("a,b\n1,\n,3\n")

        data = read_csv(self.file)
        self.assertTrue(np.isnan(data["a"][1]))
        self.assertTrue(np.isnan(data["b"][0]))

    def test_no_header(self):
        with open(self.file, "w") as f:
            f.write("1,2\n3,4\n")

        data = read_csv(self.file, has_header=False)
        self.assertIn("col_0", data)

    def test_return_matrix(self):
        data = read_csv(self.file, return_dict=False)
        self.assertEqual(data.shape, (2, 3))

    def test_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            read_csv("does_not_exist.csv")

    def test_empty_file(self):
        with open(self.file, "w") as f:
            f.write("")

        with self.assertRaises(ValueError):
            read_csv(self.file)

    def test_chunking(self):
        chunks = list(read_csv(self.file, chunk_size=1))
        self.assertEqual(len(chunks), 2)

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            read_csv(self.file, chunk_size=0)


if __name__ == "__main__":
    unittest.main()