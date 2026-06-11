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

    def test_predict_basic(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        predict = tree.predict(X)

        self.assertEqual(predict.shape, y.shape)

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

    def test_min_samples_split(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier(min_samples_split=10)
        tree.fit(X, y)

        predict = tree.predict(X)
        self.assertEqual(predict.shape, y.shape) # tree should just return leaf with majority class

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

    def test_max_features(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier(max_features=2)
        tree.fit(X, y)
        self.assertTrue(tree.fitted)

    def test_single_class(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        predict = tree.predict(X)
        np.testing.assert_array_equal(predict, y) # all same class -> should just return leaf

    def test_partial_fit_first_chunk(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier()
        tree.partial_fit(X, y)
        self.assertTrue(tree.fitted)

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

    def test_partial_fit_predict_correct(self):
        X1 = np.array([[1, 0], [2, 0]])
        y1 = np.array([0, 0])
        X2 = np.array([[8, 0], [9, 0]])
        y2 = np.array([1, 1])
        tree = DecisionTreeClassifier()
        tree.partial_fit(X1, y1)
        tree.partial_fit(X2, y2)
        predict = tree.predict(np.array([[1, 0], [9, 0]]))
        np.testing.assert_array_equal(predict, np.array([0, 1]))

    def test_impurity_gini_pure(self):
        tree = DecisionTreeClassifier(criterion='gini')
        y = np.array([1, 1, 1])
        self.assertAlmostEqual(tree.impurity(y), 0.0)

    def test_impurity_gini_mixed(self):
        tree = DecisionTreeClassifier(criterion='gini')
        y = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(tree.impurity(y), 0.5)

    def test_impurity_entropy_pure(self):
        tree = DecisionTreeClassifier(criterion='entropy')
        y = np.array([1, 1, 1])
        self.assertAlmostEqual(tree.impurity(y), 0.0, places=3)

    def test_impurity_empty(self):
        tree = DecisionTreeClassifier()
        self.assertEqual(tree.impurity(np.array([])), 0.0)

    def test_leaf_value_majority(self):
        tree = DecisionTreeClassifier()
        y = np.array([0, 0, 1])
        self.assertEqual(tree.leaf_value(y), 0)

    def test_leaf_value_empty(self):
        tree = DecisionTreeClassifier()
        self.assertIsNone(tree.leaf_value(np.array([])))


if __name__ == "__main__":
    unittest.main()