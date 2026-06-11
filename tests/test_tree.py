import unittest
import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier


class TestDecisionTreeClassifier(unittest.TestCase):

    def test_fit_basic(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        self.assertTrue(tree.fitted)

    def test_predict_correct(self):
        # simple linearly separable data
        X = np.array([[1, 0], [2, 0], [8, 0], [9, 0]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        predict = tree.predict(X)

        np.testing.assert_array_equal(predict, y)

    def test_predict_before_fit(self):
        tree = DecisionTreeClassifier()

        with self.assertRaises(RuntimeError):
            tree.predict(np.array([[1, 2]]))

    def test_max_depth(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)
        self.assertTrue(tree.fitted)

    def test_criterion_gini(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier(criterion='gini')
        tree.fit(X, y)
        self.assertTrue(tree.fitted)

    def test_criterion_entropy(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier(criterion='entropy')
        tree.fit(X, y)
        self.assertTrue(tree.fitted)

    def test_single_class(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        predict = tree.predict(X)
        np.testing.assert_array_equal(predict, y) # all same class -> should just return leaf

    def test_partial_fit_multiple_chunks(self):
        X1 = np.array([[1, 2], [3, 4]])
        y1 = np.array([0, 0])
        X2 = np.array([[5, 6], [7, 8]])
        y2 = np.array([1, 1])
        tree = DecisionTreeClassifier()
        tree.partial_fit(X1, y1)
        tree.partial_fit(X2, y2)
        predict = tree.predict(np.array([[1, 2], [7, 8]]))
        self.assertEqual(predict.shape, (2,))

    def test_impurity_gini(self):
        tree = DecisionTreeClassifier(criterion='gini')
        self.assertAlmostEqual(tree.impurity(np.array([0, 1, 0, 1])), 0.5)
        self.assertAlmostEqual(tree.impurity(np.array([1, 1, 1])), 0.0)


if __name__ == "__main__":
    unittest.main()